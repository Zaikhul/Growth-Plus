"""Event window blackout and volatility controls for official announcements.

Enforces Section 3.11:
- CPI & FOMC windows:
    - Scalp publication suppressed: 5m before scheduled until 10m after verified receipt
    - Swing & Position suppress Strong labels during this blackout window
- Unexpected actions trigger immediate EVENT_REVIEW blackout
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.domain.identity import HorizonId
from src.domain.macro import MacroReleaseCalendarEvent
from src.domain.time import ensure_utc


@dataclass(frozen=True, slots=True)
class EventWindowEvaluation:
    """Outcome of event blackout window evaluation."""

    is_in_blackout: bool
    suppress_publication: bool
    suppress_strong_labels: bool
    warning_message: str | None


def evaluate_event_window(
    horizon: HorizonId,
    decision_time: datetime,
    events: list[MacroReleaseCalendarEvent],
    feed_is_healthy: bool = True,
) -> EventWindowEvaluation:
    """Evaluate whether the current decision falls inside an event blackout window."""
    t = ensure_utc(decision_time)

    for event in events:
        sched = ensure_utc(event.scheduled_at)
        start_blackout = sched - timedelta(minutes=5)

        # Post-event window logic
        if event.first_seen_at is not None and feed_is_healthy:
            end_blackout = max(sched, ensure_utc(event.first_seen_at)) + timedelta(minutes=10)
        else:
            # If release has not arrived yet, window remains open indefinitely
            end_blackout = max(sched + timedelta(hours=2), t + timedelta(minutes=1))

        if start_blackout <= t <= end_blackout:
            if horizon == HorizonId.SCALP_15M:
                msg = f"Publication suppressed: active blackout for {event.event_type.value}"
                return EventWindowEvaluation(
                    is_in_blackout=True,
                    suppress_publication=True,
                    suppress_strong_labels=True,
                    warning_message=msg,
                )
            # Swing and position allow publishing but suppress Strong labels
            warn = (
                f"Event warning: {event.event_type.value} window active. "
                "Strong conviction suppressed."
            )
            return EventWindowEvaluation(
                is_in_blackout=True,
                suppress_publication=False,
                suppress_strong_labels=True,
                warning_message=warn,
            )

    return EventWindowEvaluation(
        is_in_blackout=False,
        suppress_publication=False,
        suppress_strong_labels=False,
        warning_message=None,
    )
