"""Signal contracts, display labels, and uncertainty definitions.

Enforces Section 1.5, 3.9 & 4.6:
- 5 user-facing display labels: Strong Buy, Buy, Neutral, Sell, Strong Sell
- Explicit confidence definition matching direction class probability
- Block bootstrap uncertainty interval
- Abstain / Unavailable statuses with explicit reason codes
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from src.domain.features import SourceCoverageMode
from src.domain.identity import HorizonId, MarketId
from src.domain.predictions import ProbabilityVector
from src.domain.time import ensure_utc


class SignalLabel(StrEnum):
    """The 5 user-facing signal classifications (PRD Section 1.1 & 3.9)."""

    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    NEUTRAL = "Neutral"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"


class SignalStatus(StrEnum):
    """Signal decision lifecycle status (PRD Section 3.13)."""

    READY = "READY"  # Published usable directional signal
    ABSTAIN = "ABSTAIN"  # Usable data, but conviction below threshold
    UNAVAILABLE = "UNAVAILABLE"  # Data stale, mask unsupported, uncalibrated, etc.
    EXPIRED = "EXPIRED"  # Current time exceeds signal TTL


class SignalReasonCode(StrEnum):
    """Structured reason codes for abstention and explanations."""

    # Abstention / degradation reasons
    LOW_CONVICTION = "LOW_CONVICTION"
    REQUIRED_MARKET_DATA_STALE = "REQUIRED_MARKET_DATA_STALE"
    UNSUPPORTED_SOURCE_MASK = "UNSUPPORTED_SOURCE_MASK"
    UNCALIBRATED_COHORT = "UNCALIBRATED_COHORT"
    EVENT_BLACKOUT = "EVENT_BLACKOUT"
    RIGHTS_BLOCKED = "RIGHTS_BLOCKED"
    MODEL_APPROVAL_EXPIRED = "MODEL_APPROVAL_EXPIRED"

    # Explanation factor codes
    TECH_TREND_SUPPORT = "TECH_TREND_SUPPORT"
    TECH_MOMENTUM_SUPPORT = "TECH_MOMENTUM_SUPPORT"
    ETF_RECENT_FLOW_SUPPORT = "ETF_RECENT_FLOW_SUPPORT"
    ETF_OUTFLOW_PRESSURE = "ETF_OUTFLOW_PRESSURE"
    MACRO_RATE_PRESSURE = "MACRO_RATE_PRESSURE"
    MACRO_INFLATION_EASING = "MACRO_INFLATION_EASING"
    NEWS_SENTIMENT_BULLISH = "NEWS_SENTIMENT_BULLISH"
    NEWS_REGULATORY_HEADWIND = "NEWS_REGULATORY_HEADWIND"


@dataclass(frozen=True, slots=True)
class CohortReliabilityInterval95:
    """95% block-bootstrap accuracy interval for prediction cohort (PRD Section 3.9)."""

    lower: float
    upper: float
    raw_observations: int
    independent_blocks: int
    method: str = "stationary_block_bootstrap"


@dataclass(frozen=True, slots=True)
class SignalExplanationFactor:
    """Attributed supporting or opposing factor."""

    code: SignalReasonCode | str
    direction: str  # "UP", "DOWN", "FLAT"
    attribution_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class Signal:
    """Complete immutable signal entity matching PRD Section 4.6 contract."""

    signal_id: uuid.UUID
    sequence: int
    market_id: MarketId
    horizon: HorizonId
    status: SignalStatus
    mode: SourceCoverageMode
    cutoff_at: datetime
    issued_at: datetime
    expires_at: datetime
    probabilities: ProbabilityVector | None = None
    label: SignalLabel | None = None
    confidence: float | None = None
    confidence_event: str | None = None
    data_quality: float = 0.0
    outcome_hurdle_log_return: float = 0.0
    reason_code: SignalReasonCode | None = None
    reasons: tuple[SignalExplanationFactor, ...] = field(default_factory=tuple)
    cohort_reliability: CohortReliabilityInterval95 | None = None
    model_bundle: str = ""
    policy_version: str = "labels_1.0.0"
    feature_set_version: str = "features_1.0.0"
    snapshot_id: uuid.UUID | None = None
    rights_policy_version: str = "rights_1.0.0"
    is_replay: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoff_at", ensure_utc(self.cutoff_at))
        object.__setattr__(self, "issued_at", ensure_utc(self.issued_at))
        object.__setattr__(self, "expires_at", ensure_utc(self.expires_at))

    def is_expired_at(self, current_time: datetime) -> bool:
        """Check if signal has expired relative to current tradable clock."""
        return ensure_utc(current_time) >= self.expires_at
