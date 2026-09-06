"""NATS JetStream event bus adapter for Growth+.

Enforces Section 4.3:
- Durable pull consumers
- Explicit manual acknowledgment AFTER database transaction commit
- Message deduplication using dedupe_key via Nats-Msg-Id header
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

import nats
from nats.js import JetStreamContext
from nats.js.api import ConsumerConfig, DeliverPolicy, StreamConfig

from src.adapters.messaging.envelope import EventEnvelope

logger = logging.getLogger(__name__)


class JetStreamEventBus:
    """Production NATS JetStream event transport adapter."""

    def __init__(self, servers: list[str] | None = None) -> None:
        self._servers = servers or ["nats://localhost:4222"]
        self._nc: Any = None
        self._js: JetStreamContext | None = None

    async def connect(self) -> None:
        """Establish connection to NATS cluster and bind JetStream context."""
        if self._nc is None or not self._nc.is_connected:
            self._nc = await nats.connect(servers=self._servers)
            self._js = self._nc.jetstream()

            # Ensure canonical stream exists
            stream_cfg = StreamConfig(
                name="GROWTH_EVENTS",
                subjects=["growth.>"],
                max_msg_size=262144,  # 256 KiB
            )
            await self._js.add_stream(stream_cfg)

    async def close(self) -> None:
        """Close connection cleanly."""
        if self._nc and self._nc.is_connected:
            await self._nc.drain()
            await self._nc.close()
            self._nc = None
            self._js = None

    async def publish(self, event: EventEnvelope) -> None:
        """Publish event envelope with deduplication header."""
        if self._js is None:
            await self.connect()
        assert self._js is not None

        headers = {
            "Nats-Msg-Id": event.dedupe_key,
            "X-Event-Id": str(event.event_id),
            "X-Is-Replay": str(event.is_replay).lower(),
        }

        data_bytes = event.to_json().encode("utf-8")
        ack = await self._js.publish(
            subject=str(event.subject),
            payload=data_bytes,
            headers=headers,
        )
        logger.debug("Published event %s to %s (seq=%s)", event.event_id, event.subject, ack.seq)

    async def subscribe(
        self,
        subject: str,
        consumer_name: str,
    ) -> AsyncIterator[EventEnvelope]:
        """Subscribe via durable pull consumer with manual ACK."""
        if self._js is None:
            await self.connect()
        assert self._js is not None

        consumer_cfg = ConsumerConfig(
            durable_name=consumer_name,
            deliver_policy=DeliverPolicy.ALL,
            ack_wait=30,  # 30 seconds
            max_deliver=5,
        )

        sub = await self._js.pull_subscribe(
            subject=subject,
            durable=consumer_name,
            config=consumer_cfg,
        )

        while True:
            try:
                msgs = await sub.fetch(batch=1, timeout=5)
                for msg in msgs:
                    envelope = EventEnvelope.from_json(msg.data.decode("utf-8"))
                    yield envelope
                    await msg.ack()
            except TimeoutError:
                continue
