"""SQLAlchemy 2 Async Repositories for PostgreSQL.

Implements ports in src/ports/repositories.py for production workloads.
"""

import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.persistence.postgres.models import (
    CurrentSignalModel,
    FeatureSnapshotModel,
    InboxModel,
    MarketBarModel,
    MarketTradeModel,
    OutboxModel,
    SignalModel,
)
from src.domain.features import FeatureSnapshot, PillarType, SourceCoverageMode
from src.domain.identity import HorizonId, MarketId
from src.domain.market import Bar1m, Trade, TradeSide
from src.domain.signals import Signal, SignalLabel, SignalStatus


class PostgresTradeRepository:
    """PostgreSQL partitioned trade repository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def append_trade(self, trade: Trade) -> None:
        row = MarketTradeModel(
            trade_id=trade.trade_id,
            market_id=str(trade.market_id),
            price_usd=trade.price_usd,
            size=trade.size,
            side=trade.side.value,
            event_time=trade.event_time,
        )
        self._session.add(row)
        await self._session.flush()

    async def get_trades_window(
        self,
        market_id: MarketId,
        start_time: datetime,
        end_time: datetime,
    ) -> Sequence[Trade]:
        stmt = (
            select(MarketTradeModel)
            .where(
                MarketTradeModel.market_id == str(market_id),
                MarketTradeModel.event_time >= start_time,
                MarketTradeModel.event_time <= end_time,
            )
            .order_by(MarketTradeModel.event_time)
        )
        res = await self._session.execute(stmt)
        return [
            Trade(
                trade_id=r.trade_id,
                market_id=market_id,
                price_usd=r.price_usd,
                size=r.size,
                side=TradeSide(r.side),
                event_time=r.event_time,
            )
            for r in res.scalars()
        ]

    async def get_latest_trade(self, market_id: MarketId) -> Trade | None:
        stmt = (
            select(MarketTradeModel)
            .where(MarketTradeModel.market_id == str(market_id))
            .order_by(desc(MarketTradeModel.event_time))
            .limit(1)
        )
        res = await self._session.execute(stmt)
        r = res.scalar_one_or_none()
        if r is None:
            return None
        return Trade(
            trade_id=r.trade_id,
            market_id=market_id,
            price_usd=r.price_usd,
            size=r.size,
            side=TradeSide(r.side),
            event_time=r.event_time,
        )


class PostgresBarRepository:
    """PostgreSQL partitioned 1m bar repository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def append_bar(self, bar: Bar1m) -> None:
        row = MarketBarModel(
            market_id=str(bar.market_id),
            bar_start_at=bar.bar_start_at,
            bar_close_at=bar.bar_close_at,
            open_usd=bar.open_usd,
            high_usd=bar.high_usd,
            low_usd=bar.low_usd,
            close_usd=bar.close_usd,
            volume=bar.volume,
            trade_count=bar.trade_count,
            vwap_usd=bar.vwap_usd,
        )
        self._session.add(row)
        await self._session.flush()

    async def get_bars(
        self,
        market_id: MarketId,
        start_time: datetime,
        end_time: datetime,
    ) -> Sequence[Bar1m]:
        stmt = (
            select(MarketBarModel)
            .where(
                MarketBarModel.market_id == str(market_id),
                MarketBarModel.bar_close_at >= start_time,
                MarketBarModel.bar_close_at <= end_time,
            )
            .order_by(MarketBarModel.bar_close_at)
        )
        res = await self._session.execute(stmt)
        return [
            Bar1m(
                market_id=market_id,
                open_usd=r.open_usd,
                high_usd=r.high_usd,
                low_usd=r.low_usd,
                close_usd=r.close_usd,
                volume=r.volume,
                bar_start_at=r.bar_start_at,
                bar_close_at=r.bar_close_at,
                trade_count=r.trade_count,
                vwap_usd=r.vwap_usd,
            )
            for r in res.scalars()
        ]

    async def get_latest_closed_bar(
        self,
        market_id: MarketId,
        as_of_time: datetime,
    ) -> Bar1m | None:
        # PRD Section 3.12: Closed bars are eligible only after bar_close_at + 2s <= as_of_time
        stmt = (
            select(MarketBarModel)
            .where(
                MarketBarModel.market_id == str(market_id),
                MarketBarModel.bar_close_at <= as_of_time,
            )
            .order_by(desc(MarketBarModel.bar_close_at))
            .limit(1)
        )
        res = await self._session.execute(stmt)
        r = res.scalar_one_or_none()
        if r is None:
            return None
        bar = Bar1m(
            market_id=market_id,
            open_usd=r.open_usd,
            high_usd=r.high_usd,
            low_usd=r.low_usd,
            close_usd=r.close_usd,
            volume=r.volume,
            bar_start_at=r.bar_start_at,
            bar_close_at=r.bar_close_at,
            trade_count=r.trade_count,
            vwap_usd=r.vwap_usd,
        )
        if bar.available_for_decision_at > as_of_time:
            return None
        return bar


class PostgresSnapshotRepository:
    """PostgreSQL feature snapshot repository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def append_snapshot(self, snapshot: FeatureSnapshot) -> None:
        row = FeatureSnapshotModel(
            snapshot_id=snapshot.snapshot_id,
            market_id=str(snapshot.market_id),
            horizon=snapshot.horizon.value,
            cutoff_at=snapshot.cutoff_at,
            computed_at=snapshot.computed_at,
            feature_set_version=snapshot.feature_set_version,
            scalars=dict(snapshot.scalars),
            active_pillars={"pillars": [p.value for p in snapshot.active_pillars]},
            mode=snapshot.mode.value,
        )
        self._session.add(row)
        await self._session.flush()

    async def get_snapshot(self, snapshot_id: uuid.UUID) -> FeatureSnapshot | None:
        stmt = select(FeatureSnapshotModel).where(FeatureSnapshotModel.snapshot_id == snapshot_id)
        res = await self._session.execute(stmt)
        r = res.scalar_one_or_none()
        if r is None:
            return None
        return FeatureSnapshot(
            snapshot_id=r.snapshot_id,
            market_id=MarketId(r.market_id),
            horizon=HorizonId(r.horizon),
            cutoff_at=r.cutoff_at,
            computed_at=r.computed_at,
            feature_set_version=r.feature_set_version,
            scalars=r.scalars,
            active_pillars=tuple(PillarType(p) for p in r.active_pillars.get("pillars", [])),
            mode=SourceCoverageMode(r.mode),
        )


class PostgresSignalRepository:
    """PostgreSQL signal repository maintaining ledger and current pointer."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def append_signal(self, signal: Signal) -> None:
        row = SignalModel(
            signal_id=signal.signal_id,
            sequence=signal.sequence,
            market_id=str(signal.market_id),
            horizon=signal.horizon.value,
            status=signal.status.value,
            mode=signal.mode.value,
            cutoff_at=signal.cutoff_at,
            issued_at=signal.issued_at,
            expires_at=signal.expires_at,
            label=signal.label.value if signal.label else None,
            confidence=signal.confidence,
            confidence_event=signal.confidence_event,
            data_quality=signal.data_quality,
            probabilities=None,
            reasons={"reasons": []},
            is_replay=signal.is_replay,
        )
        self._session.add(row)

        # Update current pointer
        curr = await self._session.get(
            CurrentSignalModel,
            (str(signal.market_id), signal.horizon.value),
        )
        if curr is None:
            curr = CurrentSignalModel(
                market_id=str(signal.market_id),
                horizon=signal.horizon.value,
                signal_id=signal.signal_id,
                sequence=signal.sequence,
                issued_at=signal.issued_at,
                expires_at=signal.expires_at,
                label=signal.label.value if signal.label else None,
                status=signal.status.value,
            )
            self._session.add(curr)
        elif signal.sequence >= curr.sequence:
            curr.signal_id = signal.signal_id
            curr.sequence = signal.sequence
            curr.issued_at = signal.issued_at
            curr.expires_at = signal.expires_at
            curr.label = signal.label.value if signal.label else None
            curr.status = signal.status.value

        await self._session.flush()

    async def get_latest_signal(
        self,
        market_id: MarketId,
        horizon: HorizonId,
    ) -> Signal | None:
        curr = await self._session.get(
            CurrentSignalModel,
            (str(market_id), horizon.value),
        )
        if curr is None:
            return None
        return await self.get_signal(curr.signal_id)

    async def get_signal(self, signal_id: uuid.UUID) -> Signal | None:
        stmt = select(SignalModel).where(SignalModel.signal_id == signal_id)
        res = await self._session.execute(stmt)
        r = res.scalar_one_or_none()
        if r is None:
            return None
        return Signal(
            signal_id=r.signal_id,
            sequence=r.sequence,
            market_id=MarketId(r.market_id),
            horizon=HorizonId(r.horizon),
            status=SignalStatus(r.status),
            mode=SourceCoverageMode(r.mode),
            cutoff_at=r.cutoff_at,
            issued_at=r.issued_at,
            expires_at=r.expires_at,
            label=SignalLabel(r.label) if r.label else None,
            confidence=r.confidence,
            confidence_event=r.confidence_event,
            data_quality=r.data_quality,
            is_replay=r.is_replay,
        )

    async def list_signals(
        self,
        market_id: MarketId,
        horizon: HorizonId,
        limit: int = 100,
    ) -> Sequence[Signal]:
        stmt = (
            select(SignalModel)
            .where(
                SignalModel.market_id == str(market_id),
                SignalModel.horizon == horizon.value,
            )
            .order_by(desc(SignalModel.sequence))
            .limit(limit)
        )
        res = await self._session.execute(stmt)
        return [
            Signal(
                signal_id=r.signal_id,
                sequence=r.sequence,
                market_id=MarketId(r.market_id),
                horizon=HorizonId(r.horizon),
                status=SignalStatus(r.status),
                mode=SourceCoverageMode(r.mode),
                cutoff_at=r.cutoff_at,
                issued_at=r.issued_at,
                expires_at=r.expires_at,
                label=SignalLabel(r.label) if r.label else None,
                confidence=r.confidence,
                confidence_event=r.confidence_event,
                data_quality=r.data_quality,
                is_replay=r.is_replay,
            )
            for r in res.scalars()
        ]


class PostgresOutboxRepository:
    """PostgreSQL transactional outbox repository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def append_outbox(
        self,
        event_id: uuid.UUID,
        event_type: str,
        dedupe_key: str,
        payload: Mapping[str, Any],
    ) -> None:
        row = OutboxModel(
            id=event_id,
            event_type=event_type,
            dedupe_key=dedupe_key,
            payload=dict(payload),
            created_at=datetime.now(UTC),
            status="PENDING",
        )
        self._session.add(row)
        await self._session.flush()

    async def get_pending(self, limit: int = 50) -> Sequence[Mapping[str, Any]]:
        stmt = (
            select(OutboxModel)
            .where(OutboxModel.status == "PENDING")
            .order_by(OutboxModel.created_at)
            .limit(limit)
        )
        res = await self._session.execute(stmt)
        return [
            {
                "id": r.id,
                "event_type": r.event_type,
                "dedupe_key": r.dedupe_key,
                "payload": r.payload,
                "created_at": r.created_at,
            }
            for r in res.scalars()
        ]

    async def mark_dispatched(self, event_ids: Sequence[uuid.UUID]) -> None:
        now = datetime.now(UTC)
        stmt = (
            update(OutboxModel)
            .where(OutboxModel.id.in_(event_ids))
            .values(status="DISPATCHED", dispatched_at=now)
        )
        await self._session.execute(stmt)
        await self._session.flush()


class PostgresInboxRepository:
    """PostgreSQL idempotent consumer inbox repository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def is_processed(self, consumer_name: str, event_id: uuid.UUID) -> bool:
        row = await self._session.get(InboxModel, (consumer_name, event_id))
        return row is not None

    async def mark_processed(self, consumer_name: str, event_id: uuid.UUID) -> bool:
        if await self.is_processed(consumer_name, event_id):
            return False
        row = InboxModel(
            consumer_name=consumer_name,
            event_id=event_id,
            processed_at=datetime.now(UTC),
        )
        self._session.add(row)
        await self._session.flush()
        return True
