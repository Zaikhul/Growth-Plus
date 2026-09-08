"""In-memory Unit of Work implementation with state rollback support."""

import uuid
from types import TracebackType
from typing import Any

from src.adapters.persistence.in_memory.repositories import (
    InMemoryInboxRepository,
    InMemoryOutboxRepository,
    InMemorySignalRepository,
    InMemorySnapshotRepository,
)
from src.domain.features import FeatureSnapshot
from src.domain.signals import Signal
from src.ports.unit_of_work import UnitOfWork


class InMemoryUnitOfWork(UnitOfWork):
    """In-memory Unit of Work simulating atomic transaction and rollback semantics."""

    signals: InMemorySignalRepository
    snapshots: InMemorySnapshotRepository
    outbox: InMemoryOutboxRepository
    inbox: InMemoryInboxRepository | None

    def __init__(
        self,
        signals: InMemorySignalRepository | None = None,
        snapshots: InMemorySnapshotRepository | None = None,
        outbox: InMemoryOutboxRepository | None = None,
        inbox: InMemoryInboxRepository | None = None,
    ) -> None:
        self.signals = signals or InMemorySignalRepository()
        self.snapshots = snapshots or InMemorySnapshotRepository()
        self.outbox = outbox or InMemoryOutboxRepository()
        self.inbox = inbox or InMemoryInboxRepository()
        self._committed: bool = False
        self._in_tx: bool = False
        self._prior_signals: list[Signal] = []
        self._prior_current: dict[tuple[str, str], Signal] = {}
        self._prior_snapshots: dict[uuid.UUID, FeatureSnapshot] = {}
        self._prior_outbox: list[dict[str, Any]] = []

    async def __aenter__(self) -> "InMemoryUnitOfWork":
        self._in_tx = True
        self._committed = False
        self._prior_signals = list(self.signals._signals)
        self._prior_current = dict(self.signals._current)
        self._prior_snapshots = dict(self.snapshots._snapshots)
        self._prior_outbox = list(self.outbox._entries) if self.outbox else []
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None or not self._committed:
            await self.rollback()
        self._in_tx = False

    async def commit(self) -> None:
        if not self._in_tx:
            raise RuntimeError("Cannot commit outside an active Unit of Work transaction")
        self._committed = True

    async def rollback(self) -> None:
        self.signals._signals = list(self._prior_signals)
        self.signals._current = dict(self._prior_current)
        self.snapshots._snapshots = dict(self._prior_snapshots)
        if self.outbox is not None:
            self.outbox._entries = list(self._prior_outbox)
        self._committed = False
