"""SQLAlchemy 2 Declarative Models for PostgreSQL 17 Persistence.

Conforms to PRD Section 4.4:
- Partitioned time-series tables: trades, bars, snapshots, signals
- Non-partitioned tables: outbox, inbox, current signal pointer, catalog
- JSONB storage for manifests and scalar feature maps
"""

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all persistent entities."""

    type_annotation_map = {
        dict[str, Any]: JSONB,
        uuid.UUID: UUID(as_uuid=True),
    }


# ==============================================================================
# 1. Market Domain Models (Partitioned)
# ==============================================================================


class MarketTradeModel(Base):
    """Partitioned trades table (daily partitions)."""

    __tablename__ = "trades"
    __table_args__ = (
        PrimaryKeyConstraint("market_id", "event_time", "trade_id"),
        {"schema": "market", "postgresql_partition_by": "RANGE (event_time)"},
    )

    trade_id: Mapped[str] = mapped_column(String(64), nullable=False)
    market_id: Mapped[str] = mapped_column(String(64), nullable=False)
    price_usd: Mapped[float] = mapped_column(Float, nullable=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # BUY, SELL
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MarketBarModel(Base):
    """Partitioned 1m bars table (monthly partitions)."""

    __tablename__ = "bars"
    __table_args__ = (
        PrimaryKeyConstraint("market_id", "bar_close_at"),
        Index("idx_bars_market_close", "market_id", "bar_close_at"),
        {"schema": "market", "postgresql_partition_by": "RANGE (bar_close_at)"},
    )

    market_id: Mapped[str] = mapped_column(String(64), nullable=False)
    bar_start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    bar_close_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open_usd: Mapped[float] = mapped_column(Float, nullable=False)
    high_usd: Mapped[float] = mapped_column(Float, nullable=False)
    low_usd: Mapped[float] = mapped_column(Float, nullable=False)
    close_usd: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False)
    vwap_usd: Mapped[float] = mapped_column(Float, nullable=False)


# ==============================================================================
# 2. Observations Models (Macro & Flows)
# ==============================================================================


class MacroObservationModel(Base):
    """Macroeconomic series observations with Point-in-Time available_at."""

    __tablename__ = "observations"
    __table_args__ = (
        PrimaryKeyConstraint("record_id"),
        Index("idx_macro_series_avail", "series_id", "available_at"),
        {"schema": "macro"},
    )

    record_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    series_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_id: Mapped[str] = mapped_column(String(64), nullable=False)
    reference_period: Mapped[str] = mapped_column(String(32), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(32), nullable=False)
    revision_seq: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    supersedes_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_digest: Mapped[str] = mapped_column(String(64), nullable=False)


class EtfFlowObservationModel(Base):
    """U.S. Spot ETF daily flow observations."""

    __tablename__ = "etf_observations"
    __table_args__ = (
        PrimaryKeyConstraint("record_id"),
        UniqueConstraint(
            "asset_id", "fund_id", "session_date", "revision_seq", name="uq_etf_session"
        ),
        Index("idx_etf_asset_avail", "asset_id", "available_at"),
        {"schema": "flows"},
    )

    record_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(16), nullable=False)
    fund_id: Mapped[str] = mapped_column(String(32), nullable=False)
    session_date: Mapped[date] = mapped_column(Date, nullable=False)
    flow_usd: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    covered_funds: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_funds: Mapped[int] = mapped_column(Integer, nullable=False)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# ==============================================================================
# 3. Features and Signals Models (Partitioned)
# ==============================================================================


class FeatureSnapshotModel(Base):
    """Partitioned immutable feature snapshots (monthly partitions)."""

    __tablename__ = "snapshots"
    __table_args__ = (
        PrimaryKeyConstraint("snapshot_id", "cutoff_at"),
        Index("idx_snapshots_market_cutoff", "market_id", "horizon", "cutoff_at"),
        {"schema": "features", "postgresql_partition_by": "RANGE (cutoff_at)"},
    )

    snapshot_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    market_id: Mapped[str] = mapped_column(String(64), nullable=False)
    horizon: Mapped[str] = mapped_column(String(32), nullable=False)
    cutoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    feature_set_version: Mapped[str] = mapped_column(String(32), nullable=False)
    scalars: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    active_pillars: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)


class SignalModel(Base):
    """Partitioned historical signal decisions (monthly partitions)."""

    __tablename__ = "signals"
    __table_args__ = (
        PrimaryKeyConstraint("signal_id", "issued_at"),
        Index("idx_signals_market_horizon_seq", "market_id", "horizon", "sequence"),
        {"schema": "signals", "postgresql_partition_by": "RANGE (issued_at)"},
    )

    signal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    market_id: Mapped[str] = mapped_column(String(64), nullable=False)
    horizon: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    cutoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_event: Mapped[str | None] = mapped_column(String(64), nullable=True)
    data_quality: Mapped[float] = mapped_column(Float, nullable=False)
    probabilities: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    reasons: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    is_replay: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class CurrentSignalModel(Base):
    """Authoritative latest signal pointer per market and horizon (non-partitioned)."""

    __tablename__ = "current"
    __table_args__ = (
        PrimaryKeyConstraint("market_id", "horizon"),
        {"schema": "signals"},
    )

    market_id: Mapped[str] = mapped_column(String(64), nullable=False)
    horizon: Mapped[str] = mapped_column(String(32), nullable=False)
    signal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)


# ==============================================================================
# 4. Ops Models: Transactional Outbox and Idempotent Inbox (Non-Partitioned)
# ==============================================================================


class OutboxModel(Base):
    """Transactional outbox for reliable event dispatch."""

    __tablename__ = "outbox"
    __table_args__ = (
        PrimaryKeyConstraint("id"),
        UniqueConstraint("dedupe_key", name="uq_outbox_dedupe"),
        Index("idx_outbox_pending", "status", "created_at"),
        {"schema": "ops"},
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    dedupe_key: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), default="PENDING", nullable=False
    )  # PENDING, DISPATCHED, FAILED


class InboxModel(Base):
    """Idempotent consumer inbox preventing duplicate processing."""

    __tablename__ = "inbox"
    __table_args__ = (
        PrimaryKeyConstraint("consumer_name", "event_id"),
        {"schema": "ops"},
    )

    consumer_name: Mapped[str] = mapped_column(String(64), nullable=False)
    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
