"""SQLAlchemy 2 Async Repositories for PostgreSQL.

Implements ports in src/ports/repositories.py for production workloads.
"""

import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import desc, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.persistence.postgres.models import (
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
from src.domain.features import FeatureSnapshot, PillarType, SourceCoverageMode
from src.domain.flows import EtfFlowObservation, EtfProductStatus
from src.domain.identity import AssetId, DatasetId, HorizonId, MarketId, SourceId
from src.domain.macro import MacroObservation, MacroSeriesId
from src.domain.market import Bar1m, Trade, TradeSide
from src.domain.observations import ObservationEnvelope
from src.domain.predictions import ProbabilityVector
from src.domain.signals import (
    CohortReliabilityInterval95,
    Signal,
    SignalDeployment,
    SignalExplanationFactor,
    SignalLabel,
    SignalReasonCode,
    SignalStatus,
)
from src.domain.time import TimeEnvelope, ensure_utc


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
        as_of = ensure_utc(as_of_time)
        effective_cutoff = as_of - timedelta(seconds=2)
        stmt = (
            select(MarketBarModel)
            .where(
                MarketBarModel.market_id == str(market_id),
                MarketBarModel.bar_close_at <= effective_cutoff,
            )
            .order_by(desc(MarketBarModel.bar_close_at))
            .limit(1)
        )
        res = await self._session.execute(stmt)
        r = res.scalar_one_or_none()
        if r is None:
            return None
        return Bar1m(
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
            is_closed=True,
        )


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
            lineage_record_ids=[str(lid) for lid in snapshot.lineage_record_ids],
        )
        self._session.add(row)
        await self._session.flush()

    async def get_snapshot(self, snapshot_id: uuid.UUID) -> FeatureSnapshot | None:
        stmt = select(FeatureSnapshotModel).where(FeatureSnapshotModel.snapshot_id == snapshot_id)
        res = await self._session.execute(stmt)
        r = res.scalar_one_or_none()
        if r is None:
            return None
        lineage_ids = tuple(
            uuid.UUID(str(lid)) for lid in getattr(r, "lineage_record_ids", []) or []
        )
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
            lineage_record_ids=lineage_ids,
        )


def _to_macro_pair(r: MacroObservationModel) -> tuple[MacroObservation, ObservationEnvelope]:
    obs = MacroObservation(
        series_id=MacroSeriesId(r.series_id),
        source_id=SourceId(r.source_id),
        reference_period=r.reference_period,
        value=r.value,
        unit=r.unit,
        revision_seq=r.revision_seq,
        supersedes_id=str(r.supersedes_id) if r.supersedes_id else None,
        is_seasonally_adjusted=getattr(r, "is_seasonally_adjusted", False),
    )
    env = ObservationEnvelope(
        record_id=r.record_id,
        source_id=SourceId(r.source_id),
        dataset_id=DatasetId("macro_indicators"),
        source_record_key=getattr(r, "source_record_key", f"{r.series_id}:{r.reference_period}"),
        schema_version=getattr(r, "schema_version", "macro_v1"),
        parser_version=getattr(r, "parser_version", "parser_v1"),
        time_envelope=TimeEnvelope(
            event_time=r.available_at,
            reference_period=r.reference_period,
            first_seen_at=r.available_at,
            available_at=r.available_at,
        ),
        raw_digest=r.raw_digest,
        canonical_digest=r.canonical_digest,
        rights_policy_id=getattr(r, "rights_policy_id", "rights_public"),
        rights_version=getattr(r, "rights_version", "v1"),
        revision_seq=r.revision_seq,
        supersedes_id=r.supersedes_id,
    )
    return obs, env


def _to_etf_pair(r: EtfFlowObservationModel) -> tuple[EtfFlowObservation, ObservationEnvelope]:
    flow = EtfFlowObservation(
        asset_id=AssetId(r.asset_id),
        fund_id=r.fund_id,
        ticker_at_time=getattr(r, "ticker_at_time", r.fund_id) or r.fund_id,
        issuer=getattr(r, "issuer", "") or "",
        jurisdiction=getattr(r, "jurisdiction", "US") or "US",
        session_date=r.session_date,
        flow_usd=r.flow_usd,
        status=EtfProductStatus(r.status)
        if r.status in EtfProductStatus.__members__.values()
        else EtfProductStatus.PRELIMINARY,
        covered_funds=r.covered_funds,
        expected_funds=r.expected_funds,
        is_seed_or_conversion=getattr(r, "is_seed_or_conversion", False),
        revision_seq=r.revision_seq,
    )
    raw_digest = getattr(r, "raw_digest", "na") or "na"
    canonical_digest = getattr(r, "canonical_digest", "na") or "na"
    source_id_str = getattr(r, "source_id", "farside") or "farside"
    source_record_key = (
        getattr(r, "source_record_key", f"{r.asset_id}:{r.fund_id}:{r.session_date}")
        or f"{r.asset_id}:{r.fund_id}:{r.session_date}"
    )
    schema_version = getattr(r, "schema_version", "flows_v1") or "flows_v1"
    parser_version = getattr(r, "parser_version", "parser_v1") or "parser_v1"
    rights_policy_id = getattr(r, "rights_policy_id", "rights_licensed") or "rights_licensed"
    rights_version = getattr(r, "rights_version", "v1") or "v1"

    env = ObservationEnvelope(
        record_id=r.record_id,
        source_id=SourceId(source_id_str),
        dataset_id=DatasetId("etf_flows"),
        source_record_key=source_record_key,
        schema_version=schema_version,
        parser_version=parser_version,
        time_envelope=TimeEnvelope(
            event_time=r.available_at,
            reference_period=str(r.session_date),
            first_seen_at=r.available_at,
            available_at=r.available_at,
        ),
        raw_digest=raw_digest,
        canonical_digest=canonical_digest,
        rights_policy_id=rights_policy_id,
        rights_version=rights_version,
        revision_seq=r.revision_seq,
    )
    return flow, env


class PostgresObservationRepository:
    """PostgreSQL macro and ETF flow observation repository with strict PIT filtering."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def append_macro_observation(
        self, obs: MacroObservation, envelope: ObservationEnvelope
    ) -> None:
        supersedes_uuid: uuid.UUID | None = None
        if envelope.supersedes_id is not None:
            supersedes_uuid = (
                envelope.supersedes_id
                if isinstance(envelope.supersedes_id, uuid.UUID)
                else uuid.UUID(str(envelope.supersedes_id))
            )
        elif obs.supersedes_id is not None:
            try:
                supersedes_uuid = uuid.UUID(str(obs.supersedes_id))
            except ValueError:
                supersedes_uuid = None

        s_id = str(obs.series_id.value if hasattr(obs.series_id, "value") else obs.series_id)
        src_id = str(
            envelope.source_id.value if hasattr(envelope.source_id, "value") else envelope.source_id
        )
        row = MacroObservationModel(
            record_id=envelope.record_id,
            series_id=s_id,
            source_id=src_id,
            reference_period=envelope.time_envelope.valid_from.isoformat()
            if envelope.time_envelope.valid_from
            else obs.reference_period,
            value=obs.value,
            unit=obs.unit,
            revision_seq=envelope.revision_seq,
            supersedes_id=supersedes_uuid,
            available_at=envelope.time_envelope.available_at,
            raw_digest=envelope.raw_digest,
            canonical_digest=envelope.canonical_digest,
            is_seasonally_adjusted=obs.is_seasonally_adjusted,
            source_record_key=envelope.source_record_key,
            schema_version=envelope.schema_version,
            parser_version=envelope.parser_version,
            rights_policy_id=envelope.rights_policy_id,
            rights_version=envelope.rights_version,
        )
        self._session.add(row)
        await self._session.flush()

    async def get_macro_observations(
        self, series_id: str, as_of_time: datetime
    ) -> Sequence[tuple[MacroObservation, ObservationEnvelope]]:
        as_of = ensure_utc(as_of_time)
        stmt = (
            select(MacroObservationModel)
            .where(
                MacroObservationModel.series_id == series_id,
                MacroObservationModel.available_at <= as_of,
            )
            .order_by(
                MacroObservationModel.available_at,
                MacroObservationModel.revision_seq,
            )
        )
        res = await self._session.execute(stmt)
        return [_to_macro_pair(r) for r in res.scalars()]

    async def append_etf_flow(
        self, flow: EtfFlowObservation, envelope: ObservationEnvelope
    ) -> None:
        src_id = str(
            envelope.source_id.value if hasattr(envelope.source_id, "value") else envelope.source_id
        )
        row = EtfFlowObservationModel(
            record_id=envelope.record_id,
            asset_id=str(flow.asset_id.value if hasattr(flow.asset_id, "value") else flow.asset_id),
            fund_id=flow.fund_id,
            ticker_at_time=flow.ticker_at_time,
            issuer=flow.issuer,
            jurisdiction=flow.jurisdiction,
            session_date=flow.session_date,
            flow_usd=flow.flow_usd,
            status=str(flow.status.value if hasattr(flow.status, "value") else flow.status),
            covered_funds=flow.covered_funds,
            expected_funds=flow.expected_funds,
            is_seed_or_conversion=flow.is_seed_or_conversion,
            revision_seq=flow.revision_seq,
            available_at=envelope.time_envelope.available_at,
            raw_digest=envelope.raw_digest,
            canonical_digest=envelope.canonical_digest,
            source_id=src_id,
            source_record_key=envelope.source_record_key,
            schema_version=envelope.schema_version,
            parser_version=envelope.parser_version,
            rights_policy_id=envelope.rights_policy_id,
            rights_version=envelope.rights_version,
        )
        self._session.add(row)
        await self._session.flush()

    async def get_latest_etf_flow(
        self, asset_id: AssetId, as_of_time: datetime
    ) -> tuple[EtfFlowObservation, ObservationEnvelope] | None:
        a_id = str(asset_id.value if hasattr(asset_id, "value") else asset_id)
        as_of = ensure_utc(as_of_time)
        stmt = (
            select(EtfFlowObservationModel)
            .where(
                EtfFlowObservationModel.asset_id == a_id,
                EtfFlowObservationModel.available_at <= as_of,
            )
            .order_by(
                desc(EtfFlowObservationModel.available_at),
                desc(EtfFlowObservationModel.revision_seq),
            )
            .limit(1)
        )
        res = await self._session.execute(stmt)
        r = res.scalar_one_or_none()
        return None if r is None else _to_etf_pair(r)


def _rehydrate_signal(r: SignalModel) -> Signal:
    probs = (
        ProbabilityVector(
            p_up=float(r.probabilities["p_up"]),
            p_flat=float(r.probabilities["p_flat"]),
            p_down=float(r.probabilities["p_down"]),
        )
        if r.probabilities and isinstance(r.probabilities, dict)
        else None
    )
    cohort = (
        CohortReliabilityInterval95(
            lower=float(r.cohort_reliability["lower"]),
            upper=float(r.cohort_reliability["upper"]),
            raw_observations=int(r.cohort_reliability["raw_observations"]),
            independent_blocks=int(r.cohort_reliability["independent_blocks"]),
            method=str(r.cohort_reliability.get("method", "stationary_block_bootstrap")),
        )
        if r.cohort_reliability and isinstance(r.cohort_reliability, dict)
        else None
    )
    reasons_list = (
        tuple(
            SignalExplanationFactor(
                code=SignalReasonCode(item["code"])
                if item.get("code") in SignalReasonCode.__members__.values()
                else str(item.get("code", "")),
                direction=str(item.get("direction", "FLAT")),
                attribution_weight=float(item.get("attribution_weight", 0.0)),
                pillar=str(item.get("pillar", "technical")),
            )
            for item in r.reasons.get("reasons", [])
        )
        if r.reasons and isinstance(r.reasons, dict)
        else ()
    )
    deployment_enum = (
        SignalDeployment(r.deployment)
        if hasattr(r, "deployment") and r.deployment in SignalDeployment.__members__.values()
        else SignalDeployment.PROMOTED
    )
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
        probabilities=probs,
        label=SignalLabel(r.label) if r.label else None,
        confidence=r.confidence,
        confidence_event=r.confidence_event,
        data_quality=r.data_quality,
        outcome_hurdle_log_return=float(getattr(r, "outcome_hurdle_log_return", 0.0)),
        reason_code=SignalReasonCode(r.reason_code)
        if r.reason_code and r.reason_code in SignalReasonCode.__members__.values()
        else None,
        reasons=reasons_list,
        cohort_reliability=cohort,
        model_bundle=str(getattr(r, "model_bundle", "")),
        policy_version=str(getattr(r, "policy_version", "labels_1.0.0")),
        feature_set_version=str(getattr(r, "feature_set_version", "features_1.0.0")),
        snapshot_id=getattr(r, "snapshot_id", None),
        rights_policy_version=str(getattr(r, "rights_policy_version", "rights_1.0.0")),
        deployment=deployment_enum,
        is_replay=r.is_replay,
    )


class PostgresSignalRepository:
    """PostgreSQL signal repository maintaining ledger and current pointer."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def append_signal(self, signal: Signal) -> None:
        probs_dict = (
            {
                "p_up": signal.probabilities.p_up,
                "p_flat": signal.probabilities.p_flat,
                "p_down": signal.probabilities.p_down,
            }
            if signal.probabilities is not None
            else None
        )
        cohort_dict = (
            {
                "lower": signal.cohort_reliability.lower,
                "upper": signal.cohort_reliability.upper,
                "raw_observations": signal.cohort_reliability.raw_observations,
                "independent_blocks": signal.cohort_reliability.independent_blocks,
                "method": signal.cohort_reliability.method,
            }
            if signal.cohort_reliability is not None
            else None
        )
        reasons_dict = {
            "reasons": [
                {
                    "code": str(f.code.value if hasattr(f.code, "value") else f.code),
                    "direction": f.direction,
                    "attribution_weight": f.attribution_weight,
                    "pillar": getattr(f, "pillar", "technical"),
                }
                for f in signal.reasons
            ]
        }
        deployment_val = (
            signal.deployment.value
            if hasattr(signal.deployment, "value")
            else str(getattr(signal, "deployment", "PROMOTED"))
        )
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
            probabilities=probs_dict,
            reason_code=signal.reason_code.value if signal.reason_code else None,
            reasons=reasons_dict,
            deployment=deployment_val,
            outcome_hurdle_log_return=signal.outcome_hurdle_log_return,
            cohort_reliability=cohort_dict,
            model_bundle=signal.model_bundle,
            policy_version=signal.policy_version,
            feature_set_version=signal.feature_set_version,
            snapshot_id=signal.snapshot_id,
            rights_policy_version=signal.rights_policy_version,
            is_replay=signal.is_replay,
        )
        self._session.add(row)

        # Atomic current pointer update with monotonic cutoff_at
        current_stmt = (
            pg_insert(CurrentSignalModel)
            .values(
                market_id=str(signal.market_id),
                horizon=signal.horizon.value,
                signal_id=signal.signal_id,
                sequence=signal.sequence,
                cutoff_at=signal.cutoff_at,
                issued_at=signal.issued_at,
                expires_at=signal.expires_at,
                label=signal.label.value if signal.label else None,
                status=signal.status.value,
            )
            .on_conflict_do_update(
                index_elements=["market_id", "horizon"],
                set_={
                    "signal_id": signal.signal_id,
                    "sequence": signal.sequence,
                    "cutoff_at": signal.cutoff_at,
                    "issued_at": signal.issued_at,
                    "expires_at": signal.expires_at,
                    "label": signal.label.value if signal.label else None,
                    "status": signal.status.value,
                },
                where=(
                    (CurrentSignalModel.cutoff_at < signal.cutoff_at)
                    | (
                        (CurrentSignalModel.cutoff_at == signal.cutoff_at)
                        & (CurrentSignalModel.sequence <= signal.sequence)
                    )
                ),
            )
        )
        await self._session.execute(current_stmt)
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
        return await self.get_signal(curr.signal_id, issued_at=curr.issued_at)

    async def get_signal(
        self, signal_id: uuid.UUID, issued_at: datetime | None = None
    ) -> Signal | None:
        stmt = select(SignalModel).where(SignalModel.signal_id == signal_id)
        if issued_at is not None:
            stmt = stmt.where(SignalModel.issued_at == issued_at)
        res = await self._session.execute(stmt)
        r = res.scalar_one_or_none()
        if r is None:
            return None
        return _rehydrate_signal(r)

    async def list_signals(
        self,
        market_id: MarketId,
        horizon: HorizonId,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Signal]:
        stmt = (
            select(SignalModel)
            .where(
                SignalModel.market_id == str(market_id),
                SignalModel.horizon == horizon.value,
            )
            .order_by(desc(SignalModel.sequence))
            .offset(offset)
            .limit(limit)
        )
        res = await self._session.execute(stmt)
        return [_rehydrate_signal(r) for r in res.scalars()]

    async def count_signals(
        self,
        market_id: MarketId,
        horizon: HorizonId,
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(SignalModel)
            .where(
                SignalModel.market_id == str(market_id),
                SignalModel.horizon == horizon.value,
            )
        )
        res = await self._session.execute(stmt)
        return int(res.scalar_one())

    async def ping(self) -> bool:
        """Active health check probe."""
        await self._session.execute(text("SELECT 1"))
        return True


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
            .with_for_update(skip_locked=True)
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
        stmt = (
            pg_insert(InboxModel)
            .values(
                consumer_name=consumer_name,
                event_id=event_id,
                processed_at=datetime.now(UTC),
            )
            .on_conflict_do_nothing(index_elements=["consumer_name", "event_id"])
        )
        res = await self._session.execute(stmt)
        await self._session.flush()
        rc = getattr(res, "rowcount", 0)
        return bool(rc and rc > 0)
