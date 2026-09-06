"""Transactional outbox relay worker draining pending outbox records to NATS JetStream.

Enforces PRD Section 4.2 & 4.3:
- Guarantees at-least-once durable event publishing from database outbox
- Prevents dual-write inconsistencies between database commits and broker publishing
"""

import uuid

from src.adapters.messaging.envelope import EventEnvelope
from src.ports.event_bus import EventBus
from src.ports.repositories import OutboxRepository


class OutboxRelayWorker:
    """Relays committed outbox entries to durable message bus."""

    def __init__(
        self,
        outbox_repo: OutboxRepository,
        event_bus: EventBus,
    ) -> None:
        self._outbox_repo = outbox_repo
        self._event_bus = event_bus

    async def drain_pending_outbox(self, batch_size: int = 100) -> int:
        """Fetch pending outbox records, publish each to event bus, and mark as published."""
        pending = await self._outbox_repo.get_pending(limit=batch_size)
        if not pending:
            return 0

        dispatched_ids: list[uuid.UUID] = []

        for record in pending:
            raw_id = record["id"]
            event_id = raw_id if isinstance(raw_id, uuid.UUID) else uuid.UUID(str(raw_id))
            envelope = EventEnvelope(
                event_id=event_id,
                subject=str(record["event_type"]),
                source="outbox-relay",
                occurred_at=record["created_at"],
                dedupe_key=str(record["dedupe_key"]),
                payload=record["payload"],
            )
            # Publish to JetStream / event bus
            await self._event_bus.publish(envelope)
            await self._outbox_repo.mark_dispatched([event_id])
            dispatched_ids.append(event_id)

        return len(dispatched_ids)
