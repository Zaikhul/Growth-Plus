"""PostgreSQL persistence adapters package."""

from src.adapters.persistence.postgres.models import (
    Base,
    CurrentSignalModel,
    EtfFlowObservationModel,
    FeatureSnapshotModel,
    InboxModel,
    MacroObservationModel,
    MarketBarModel,
    MarketTradeModel,
    OutboxModel,
    SignalModel,
)
from src.adapters.persistence.postgres.repositories import (
    PostgresBarRepository,
    PostgresInboxRepository,
    PostgresOutboxRepository,
    PostgresSignalRepository,
    PostgresSnapshotRepository,
    PostgresTradeRepository,
)

__all__ = [
    "Base",
    "CurrentSignalModel",
    "EtfFlowObservationModel",
    "FeatureSnapshotModel",
    "InboxModel",
    "MacroObservationModel",
    "MarketBarModel",
    "MarketTradeModel",
    "OutboxModel",
    "PostgresBarRepository",
    "PostgresInboxRepository",
    "PostgresOutboxRepository",
    "PostgresSignalRepository",
    "PostgresSnapshotRepository",
    "PostgresTradeRepository",
    "SignalModel",
]
