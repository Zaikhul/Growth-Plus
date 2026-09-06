"""Compatibility export; canonical event contract belongs to ports."""

from src.ports.event_envelope import (
    MAX_PAYLOAD_BYTES,
    TARGET_PAYLOAD_BYTES,
    EventEnvelope,
    EventSubject,
)

__all__ = [
    "EventEnvelope",
    "EventSubject",
    "MAX_PAYLOAD_BYTES",
    "TARGET_PAYLOAD_BYTES",
]
