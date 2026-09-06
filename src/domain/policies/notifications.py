"""Notification and alert dispatch policies.

Enforces Section 4.6:
- Alerts fire ONLY on:
    1. Label change meeting rule
    2. Same-direction confidence increase >= 0.10
    3. Recovery to eligible Strong label
- Replayed historical signals never trigger live notifications (PRD Section 4.3 & 4.6)
- Cooldowns:
    scalp: 15m (900s)
    swing: 4h (14400s)
    position: 24h (86400s)
- Max 8 directional notifications per account per day
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.domain.identity import HorizonId
from src.domain.signals import Signal, SignalLabel
from src.domain.time import ensure_utc

NOTIFICATION_COOLDOWNS: dict[HorizonId, timedelta] = {
    HorizonId.SCALP_15M: timedelta(minutes=15),
    HorizonId.SWING_24H: timedelta(hours=4),
    HorizonId.POSITION_30D: timedelta(hours=24),
}

MAX_DAILY_NOTIFICATIONS_PER_ACCOUNT = 8


@dataclass(frozen=True, slots=True)
class NotificationEvaluation:
    """Result of evaluating whether an alert notification should be dispatched."""

    should_dispatch: bool
    reason: str


def evaluate_notification_dispatch(
    current_signal: Signal,
    prior_signal: Signal | None,
    last_dispatched_at: datetime | None,
    daily_dispatch_count: int,
    current_time: datetime,
) -> NotificationEvaluation:
    """Evaluate whether an alert should be dispatched under policy rules."""
    curr_t = ensure_utc(current_time)

    # Invariant: Replayed signals never send live alerts
    if current_signal.is_replay:
        return NotificationEvaluation(should_dispatch=False, reason="SUPPRESSED_REPLAY_EVENT")

    # Invariant: Expired signals cannot send alerts
    if current_signal.is_expired_at(curr_t):
        return NotificationEvaluation(should_dispatch=False, reason="SIGNAL_EXPIRED")

    # Signals without a valid label cannot dispatch directional alerts
    if current_signal.label is None or current_signal.confidence is None:
        return NotificationEvaluation(should_dispatch=False, reason="NO_DIRECTIONAL_LABEL")

    # Quota check: max 8 / day
    if daily_dispatch_count >= MAX_DAILY_NOTIFICATIONS_PER_ACCOUNT:
        return NotificationEvaluation(should_dispatch=False, reason="DAILY_QUOTA_EXHAUSTED")

    # Cooldown check
    cooldown = NOTIFICATION_COOLDOWNS[current_signal.horizon]
    if last_dispatched_at is not None:
        elapsed = curr_t - ensure_utc(last_dispatched_at)
        if elapsed < cooldown:
            sec = elapsed.total_seconds()
            cd_sec = cooldown.total_seconds()
            reason = f"COOLDOWN_ACTIVE: {sec:.0f}s < {cd_sec:.0f}s"
            return NotificationEvaluation(
                should_dispatch=False,
                reason=reason,
            )

    # If first signal, dispatch if directional
    if prior_signal is None or prior_signal.label is None:
        return NotificationEvaluation(should_dispatch=True, reason="INITIAL_SIGNAL_TRIGGER")

    # Case 1: Label change
    if current_signal.label != prior_signal.label:
        return NotificationEvaluation(should_dispatch=True, reason="LABEL_CHANGED")

    # Case 2: Same direction confidence increase >= 0.10
    if prior_signal.confidence is not None:
        conf_increase = current_signal.confidence - prior_signal.confidence
        if conf_increase >= 0.10:
            return NotificationEvaluation(
                should_dispatch=True, reason="CONFIDENCE_INCREASE_GE_10_BPS"
            )

    # Case 3: Recovery to Strong label
    if current_signal.label in (
        SignalLabel.STRONG_BUY,
        SignalLabel.STRONG_SELL,
    ) and prior_signal.label not in (
        SignalLabel.STRONG_BUY,
        SignalLabel.STRONG_SELL,
    ):
        return NotificationEvaluation(should_dispatch=True, reason="STRONG_LABEL_RECOVERY")

    return NotificationEvaluation(should_dispatch=False, reason="NO_SIGNIFICANT_STATE_CHANGE")
