"""In-memory event bus implementation for testing and local decoupled workflows."""

import asyncio
from collections.abc import AsyncIterator

from src.adapters.messaging.envelope import EventEnvelope


class InMemoryEventBus:
    """Thread-safe in-memory event bus."""

    def __init__(self) -> None:
        self._published: list[EventEnvelope] = []
        self._queues: dict[str, list[asyncio.Queue[EventEnvelope]]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, event: EventEnvelope) -> None:
        """Publish event to internal history and dispatch to active subscriber queues."""
        async with self._lock:
            self._published.append(event)
            subject_queues = self._queues.get(str(event.subject), [])
            for q in subject_queues:
                await q.put(event)

    async def subscribe(
        self,
        subject: str,
        consumer_name: str,
    ) -> AsyncIterator[EventEnvelope]:
        """Subscribe to a subject and yield events as they arrive."""
        q: asyncio.Queue[EventEnvelope] = asyncio.Queue()
        async with self._lock:
            if subject not in self._queues:
                self._queues[subject] = []
            self._queues[subject].append(q)

        try:
            while True:
                event = await q.get()
                yield event
                q.task_done()
        finally:
            async with self._lock:
                if subject in self._queues and q in self._queues[subject]:
                    self._queues[subject].remove(q)

    def get_published(self, subject: str | None = None) -> list[EventEnvelope]:
        """Inspect published events for testing assertions."""
        if subject is None:
            return list(self._published)
        return [e for e in self._published if str(e.subject) == subject]
