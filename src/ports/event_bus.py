"""Abstract port for event messaging bus."""

from collections.abc import AsyncIterator
from typing import Protocol

from src.ports.event_envelope import EventEnvelope


class EventBus(Protocol):
    """Port for publishing and consuming domain event envelopes."""

    async def publish(self, event: EventEnvelope) -> None:
        """Publish an event to the bus."""
        ...

    def subscribe(
        self,
        subject: str,
        consumer_name: str,
    ) -> AsyncIterator[EventEnvelope]:
        """Subscribe to an event subject using a named consumer."""
        ...
