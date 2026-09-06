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
    PostgresObservationRepository,
    PostgresOutboxRepository,
    PostgresSignalRepository,
    PostgresSnapshotRepository,
    PostgresTradeRepository,
)
from src.adapters.persistence.postgres.tenant_context import (
    set_session_tenant,
    tenant_transaction,
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
    "PostgresObservationRepository",
    "PostgresOutboxRepository",
    "PostgresSignalRepository",
    "PostgresSnapshotRepository",
    "PostgresTradeRepository",
    "SignalModel",
    "set_session_tenant",
    "tenant_transaction",
]
