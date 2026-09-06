"""Notification dispatch worker enforcing user alert criteria and cooldown policies.

Enforces PRD Section 1.5, 4.2 & 4.5:
- Filters incoming signals against user rules (market, horizon, min_conviction)
- Enforces anti-spam cooldown periods per market/horizon
- Dispatches alert payloads to destination channels
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.domain.identity import HorizonId, MarketId
from src.domain.policies.notifications import evaluate_notification_dispatch
from src.domain.signals import Signal, SignalLabel, SignalStatus


@dataclass
class UserAlertSubscription:
    """User alert subscription state tracking."""

    subscription_id: uuid.UUID
    market_id: MarketId
    horizon_id: HorizonId
    min_confidence: float = 0.60
    cooldown_seconds: float = 900.0
    last_dispatched_at: datetime | None = None
    allowed_labels: tuple[SignalLabel, ...] = (
        SignalLabel.STRONG_BUY,
        SignalLabel.BUY,
        SignalLabel.SELL,
        SignalLabel.STRONG_SELL,
    )


class NotificationWorker:
    """Processes committed signals and triggers outbound notifications when criteria match."""

    def __init__(self) -> None:
        self._subscriptions: dict[uuid.UUID, UserAlertSubscription] = {}
        self._dispatched_alerts: list[dict[str, Any]] = []
        self._prior_signals: dict[tuple[MarketId, HorizonId], Signal] = {}
        self._daily_counts: dict[uuid.UUID, int] = {}

    def register_subscription(self, sub: UserAlertSubscription) -> None:
        """Register a user subscription."""
        self._subscriptions[sub.subscription_id] = sub

    @property
    def dispatched_alerts(self) -> list[dict[str, Any]]:
        return list(self._dispatched_alerts)

    async def handle_committed_signal(
        self,
        signal: Signal,
        current_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate incoming signal against all active subscriptions."""
        dispatched: list[dict[str, Any]] = []
        now = current_time or datetime.now(tz=UTC)

        # Ignore signals that are expired or not ready or replayed (DEFECT-06)
        if signal.status != SignalStatus.READY or signal.label is None or signal.is_replay:
            return []

        prior_sig = self._prior_signals.get((signal.market_id, signal.horizon))

        for sub_id, sub in self._subscriptions.items():
            # Check market and horizon match
            if sub.market_id != signal.market_id or sub.horizon_id != signal.horizon:
                continue

            # Check label match
            if signal.label not in sub.allowed_labels:
                continue

            # Check confidence threshold
            if signal.confidence is None or signal.confidence < sub.min_confidence:
                continue

            daily_count = self._daily_counts.get(sub_id, 0)
            eval_res = evaluate_notification_dispatch(
                current_signal=signal,
                prior_signal=prior_sig,
                last_dispatched_at=sub.last_dispatched_at,
                daily_dispatch_count=daily_count,
                current_time=now,
            )

            if not eval_res.should_dispatch:
                continue

            # Construct notification message
            alert_payload = {
                "subscription_id": str(sub_id),
                "signal_id": str(signal.signal_id),
                "market_id": signal.market_id.value,
                "horizon": signal.horizon.value,
                "label": signal.label.value,
                "confidence": signal.confidence,
                "data_quality": signal.data_quality,
                "dispatched_at": now.isoformat(),
                "reason": eval_res.reason,
            }

            sub.last_dispatched_at = now
            self._daily_counts[sub_id] = daily_count + 1
            self._dispatched_alerts.append(alert_payload)
            dispatched.append(alert_payload)

        # Track prior signal for state change tracking
        self._prior_signals[(signal.market_id, signal.horizon)] = signal
        return dispatched
