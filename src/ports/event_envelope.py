"""Event envelope and messaging contracts for NATS JetStream.

Enforces Section 4.3:
- Message payload bounds (target <= 64 KiB, hard limit 256 KiB)
- Strict subject namespaces
- Deduplication keys and replay tracking
"""

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from src.domain.errors import InvariantViolationError
from src.domain.time import ensure_utc

MAX_PAYLOAD_BYTES = 262144  # 256 KiB hard limit
TARGET_PAYLOAD_BYTES = 65536  # 64 KiB target


class EventSubject(StrEnum):
    """Canonical event subjects (PRD Section 4.3)."""

    RAW_MARKET = "growth.raw.market.v1"
    RAW_SCHEDULED = "growth.raw.scheduled.v1"
    OBSERVATION_VALIDATED = "growth.observation.validated.v1"
    FEATURE_READY = "growth.feature.ready.v1"
    SIGNAL_COMMITTED = "growth.signal.committed.v1"
    SOURCE_STATUS = "growth.source.status.v1"
    MODEL_PROMOTED = "growth.model.promoted.v1"
    DEADLETTER = "growth.deadletter.v1"


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """Standardized event envelope transported over JetStream."""

    event_id: uuid.UUID
    subject: EventSubject | str
    occurred_at: datetime
    dedupe_key: str
    source: str
    payload: Mapping[str, Any]
    is_replay: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "occurred_at", ensure_utc(self.occurred_at))

        # Validate serialized payload size
        serialized = json.dumps(self.payload)
        payload_size = len(serialized.encode("utf-8"))
        if payload_size > MAX_PAYLOAD_BYTES:
            msg = f"Payload size ({payload_size} bytes) exceeds limit ({MAX_PAYLOAD_BYTES})"
            raise InvariantViolationError(
                msg,
                details={"payload_size": payload_size, "limit": MAX_PAYLOAD_BYTES},
            )

    @property
    def payload_bytes(self) -> int:
        """Return UTF-8 byte length of the serialized payload."""
        return len(json.dumps(self.payload).encode("utf-8"))

    def to_json(self) -> str:
        """Serialize envelope to JSON string."""
        return json.dumps(
            {
                "event_id": str(self.event_id),
                "subject": str(self.subject),
                "occurred_at": self.occurred_at.isoformat(),
                "dedupe_key": self.dedupe_key,
                "source": self.source,
                "payload": self.payload,
                "is_replay": self.is_replay,
                "metadata": dict(self.metadata),
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> "EventEnvelope":
        """Deserialize envelope from JSON string."""
        data = json.loads(json_str)
        return cls(
            event_id=uuid.UUID(data["event_id"]),
            subject=data["subject"],
            occurred_at=datetime.fromisoformat(data["occurred_at"]).astimezone(UTC),
            dedupe_key=data["dedupe_key"],
            source=data["source"],
            payload=data["payload"],
            is_replay=bool(data.get("is_replay", False)),
            metadata=data.get("metadata", {}),
        )
