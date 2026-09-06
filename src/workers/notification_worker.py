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
from src.domain.signals import Signal, SignalLabel, SignalStatus


@dataclass
class UserAlertSubscription:
    """User alert subscription state tracking."""

    subscription_id: uuid.UUID
    market_id: MarketId
    horizon_id: HorizonId
    min_confidence: float = 0.60
    cooldown_seconds: float = 300.0
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

    def register_subscription(self, sub: UserAlertSubscription) -> None:
        """Register a user subscription."""
        self._subscriptions[sub.subscription_id] = sub

    @property
    def dispatched_alerts(self) -> list[dict[str, Any]]:
        return list(self._dispatched_alerts)

    async def handle_committed_signal(self, signal: Signal) -> list[dict[str, Any]]:
        """Evaluate incoming signal against all active subscriptions."""
        dispatched: list[dict[str, Any]] = []
        now = datetime.now(tz=UTC)

        # Ignore signals that are expired or not ready
        if signal.status != SignalStatus.READY or signal.label is None:
            return []

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

            # Check cooldown
            if sub.last_dispatched_at is not None:
                elapsed = (now - sub.last_dispatched_at).total_seconds()
                if elapsed < sub.cooldown_seconds:
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
            }

            sub.last_dispatched_at = now
            self._dispatched_alerts.append(alert_payload)
            dispatched.append(alert_payload)

        return dispatched
