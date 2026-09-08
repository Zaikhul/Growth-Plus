"""PostgreSQL Unit of Work coordinating atomic operations via AsyncSession transactions."""

from types import TracebackType

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.adapters.persistence.postgres.repositories import (
    PostgresInboxRepository,
    PostgresOutboxRepository,
    PostgresSignalRepository,
    PostgresSnapshotRepository,
)
from src.ports.unit_of_work import UnitOfWork


class PostgresUnitOfWork(UnitOfWork):
    """PostgreSQL Unit of Work implementing atomic commits and rollbacks."""

    def __init__(self, session_or_factory: AsyncSession | async_sessionmaker[AsyncSession]) -> None:
        if isinstance(session_or_factory, AsyncSession):
            self._session: AsyncSession | None = session_or_factory
            self._session_factory: async_sessionmaker[AsyncSession] | None = None
            self._owns_session = False
        else:
            self._session = None
            self._session_factory = session_or_factory
            self._owns_session = True

    async def __aenter__(self) -> "PostgresUnitOfWork":
        if self._session is None and self._session_factory is not None:
            self._session = self._session_factory()
        assert self._session is not None
        if not self._session.in_transaction():
            await self._session.begin()

        self.signals = PostgresSignalRepository(self._session)
        self.snapshots = PostgresSnapshotRepository(self._session)
        self.outbox = PostgresOutboxRepository(self._session)
        self.inbox = PostgresInboxRepository(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._session is not None:
            try:
                if exc_type is not None:
                    await self._session.rollback()
            finally:
                if self._owns_session:
                    await self._session.close()
                    self._session = None

    async def commit(self) -> None:
        if self._session is not None:
            await self._session.commit()

    async def rollback(self) -> None:
        if self._session is not None:
            await self._session.rollback()
