"""Freshness monitoring, decay, and state machine policies.

Enforces Section 3.10 & 3.12:
- State transitions: HEALTHY -> LATE -> STALE -> RECOVERING -> HEALTHY
- Terminal states: QUARANTINED, RIGHTS_BLOCKED, DISABLED
- Exponential freshness decay: 2^(-overdue / half_life)
"""

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from src.domain.identity import HorizonId
from src.domain.time import ensure_utc


class FreshnessState(StrEnum):
    """Source / dataset freshness lifecycle state."""

    HEALTHY = "HEALTHY"
    LATE = "LATE"
    STALE = "STALE"
    RECOVERING = "RECOVERING"
    QUARANTINED = "QUARANTINED"
    RIGHTS_BLOCKED = "RIGHTS_BLOCKED"
    DISABLED = "DISABLED"


# Hard-stop thresholds in seconds for venue trade staleness (PRD Section 3.12)
VENUE_TRADE_HARD_STOP_SECONDS: dict[HorizonId, float] = {
    HorizonId.SCALP_15M: 10.0,
    HorizonId.SWING_24H: 60.0,
    HorizonId.POSITION_30D: 300.0,
}


@dataclass(frozen=True, slots=True)
class FreshnessEvaluation:
    """Freshness status and decay factor evaluation."""

    state: FreshnessState
    freshness_factor: float  # [0.0, 1.0]
    is_hard_stop: bool
    age_seconds: float
    overdue_seconds: float


def calculate_freshness_decay(
    last_update_at: datetime,
    as_of_time: datetime,
    expected_cadence_seconds: float,
    half_life_seconds: float = 3600.0,
    hard_stop_seconds: float = 7200.0,
) -> FreshnessEvaluation:
    """Calculate exponential freshness decay: 2^(-overdue / half_life).

    Freshness is 1.0 while age <= expected_cadence.
    Once overdue, it decays exponentially until hard_stop_seconds is reached.
    """
    last_utc = ensure_utc(last_update_at)
    curr_utc = ensure_utc(as_of_time)

    age_seconds = max(0.0, (curr_utc - last_utc).total_seconds())
    overdue_seconds = max(0.0, age_seconds - expected_cadence_seconds)

    if age_seconds >= hard_stop_seconds:
        return FreshnessEvaluation(
            state=FreshnessState.STALE,
            freshness_factor=0.0,
            is_hard_stop=True,
            age_seconds=age_seconds,
            overdue_seconds=overdue_seconds,
        )

    if overdue_seconds == 0.0:
        return FreshnessEvaluation(
            state=FreshnessState.HEALTHY,
            freshness_factor=1.0,
            is_hard_stop=False,
            age_seconds=age_seconds,
            overdue_seconds=0.0,
        )

    # Exponential decay
    decay_factor = math.pow(2.0, -overdue_seconds / half_life_seconds)
    bounded_decay = min(1.0, max(0.0, decay_factor))

    return FreshnessEvaluation(
        state=FreshnessState.LATE,
        freshness_factor=bounded_decay,
        is_hard_stop=False,
        age_seconds=age_seconds,
        overdue_seconds=overdue_seconds,
    )


def is_market_tape_stale(
    horizon: HorizonId,
    last_trade_time: datetime,
    decision_cutoff: datetime,
) -> bool:
    """Check whether market trades violate the horizon hard-stop rule."""
    hard_stop = VENUE_TRADE_HARD_STOP_SECONDS[horizon]
    diff = (ensure_utc(decision_cutoff) - ensure_utc(last_trade_time)).total_seconds()
    return diff > hard_stop
