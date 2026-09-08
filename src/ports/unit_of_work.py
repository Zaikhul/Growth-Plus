"""Unit of Work port ensuring atomic multi-repository operations."""

from abc import ABC, abstractmethod
from types import TracebackType

from src.ports.repositories import (
    InboxRepository,
    OutboxRepository,
    SignalRepository,
    SnapshotRepository,
)


class UnitOfWork(ABC):
    """Abstract Unit of Work interface guaranteeing transaction atomicity."""

    signals: SignalRepository
    snapshots: SnapshotRepository
    outbox: OutboxRepository | None
    inbox: InboxRepository | None

    @abstractmethod
    async def __aenter__(self) -> "UnitOfWork":
        """Begin atomic transaction."""
        ...

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Roll back on error, or clean up transaction resources."""
        ...

    @abstractmethod
    async def commit(self) -> None:
        """Commit all staging operations atomically."""
        ...

    @abstractmethod
    async def rollback(self) -> None:
        """Roll back active transaction."""
        ...
