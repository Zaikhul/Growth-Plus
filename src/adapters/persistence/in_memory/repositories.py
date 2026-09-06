"""Hermetic, thread-safe in-memory repository implementations for testing.

Fully implements the ports defined in src/ports/repositories.py, enabling
exhaustive Hypothesis property-based tests without live PostgreSQL dependencies.
"""

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from src.domain.features import FeatureSnapshot
from src.domain.flows import EtfFlowObservation
from src.domain.identity import AssetId, HorizonId, MarketId
from src.domain.macro import MacroObservation
from src.domain.market import Bar1m, Trade
from src.domain.observations import ObservationEnvelope
from src.domain.signals import Signal
from src.domain.time import ensure_utc


class InMemoryTradeRepository:
    """In-memory trade repository."""

    def __init__(self) -> None:
        self._trades: list[Trade] = []
        self._lock = asyncio.Lock()

    async def append_trade(self, trade: Trade) -> None:
        async with self._lock:
            self._trades.append(trade)

    async def get_trades_window(
        self,
        market_id: MarketId,
        start_time: datetime,
        end_time: datetime,
    ) -> Sequence[Trade]:
        st = ensure_utc(start_time)
        et = ensure_utc(end_time)
        async with self._lock:
            return [
                t for t in self._trades if t.market_id == market_id and st <= t.event_time <= et
            ]

    async def get_latest_trade(self, market_id: MarketId) -> Trade | None:
        async with self._lock:
            matches = [t for t in self._trades if t.market_id == market_id]
            if not matches:
                return None
            return max(matches, key=lambda t: t.event_time)


class InMemoryBarRepository:
    """In-memory 1m bar repository with Point-in-Time watermark filtering."""

    def __init__(self) -> None:
        self._bars: list[Bar1m] = []
        self._lock = asyncio.Lock()

    async def append_bar(self, bar: Bar1m) -> None:
        async with self._lock:
            self._bars.append(bar)

    async def get_bars(
        self,
        market_id: MarketId,
        start_time: datetime,
        end_time: datetime,
    ) -> Sequence[Bar1m]:
        st = ensure_utc(start_time)
        et = ensure_utc(end_time)
        async with self._lock:
            return [
                b for b in self._bars if b.market_id == market_id and st <= b.bar_close_at <= et
            ]

    async def get_latest_closed_bar(
        self,
        market_id: MarketId,
        as_of_time: datetime,
    ) -> Bar1m | None:
        as_of = ensure_utc(as_of_time)
        async with self._lock:
            # Bar must be closed and available watermark (bar_close_at + 2s) <= as_of
            eligible = [
                b
                for b in self._bars
                if b.market_id == market_id and b.is_closed and b.available_for_decision_at <= as_of
            ]
            if not eligible:
                return None
            return max(eligible, key=lambda b: b.bar_close_at)


class InMemoryObservationRepository:
    """In-memory observation repository with strict Point-in-Time available_at filtering."""

    def __init__(self) -> None:
        self._macro_records: list[tuple[MacroObservation, ObservationEnvelope]] = []
        self._etf_records: list[tuple[EtfFlowObservation, ObservationEnvelope]] = []
        self._lock = asyncio.Lock()

    async def append_macro_observation(
        self,
        obs: MacroObservation,
        envelope: ObservationEnvelope,
    ) -> None:
        async with self._lock:
            self._macro_records.append((obs, envelope))

    async def get_macro_observations(
        self,
        series_id: str,
        as_of_time: datetime,
    ) -> Sequence[tuple[MacroObservation, ObservationEnvelope]]:
        as_of = ensure_utc(as_of_time)
        async with self._lock:
            # Must satisfy available_at <= as_of_time
            return [
                (obs, env)
                for (obs, env) in self._macro_records
                if obs.series_id == series_id and env.time_envelope.available_at <= as_of
            ]

    async def append_etf_flow(
        self,
        flow: EtfFlowObservation,
        envelope: ObservationEnvelope,
    ) -> None:
        async with self._lock:
            self._etf_records.append((flow, envelope))

    async def get_latest_etf_flow(
        self,
        asset_id: AssetId,
        as_of_time: datetime,
    ) -> tuple[EtfFlowObservation, ObservationEnvelope] | None:
        as_of = ensure_utc(as_of_time)
        async with self._lock:
            eligible = [
                (flow, env)
                for (flow, env) in self._etf_records
                if flow.asset_id == asset_id and env.time_envelope.available_at <= as_of
            ]
            if not eligible:
                return None
            return max(eligible, key=lambda pair: pair[1].time_envelope.available_at)


class InMemorySnapshotRepository:
    """In-memory feature snapshot repository."""

    def __init__(self) -> None:
        self._snapshots: dict[uuid.UUID, FeatureSnapshot] = {}
        self._lock = asyncio.Lock()

    async def append_snapshot(self, snapshot: FeatureSnapshot) -> None:
        async with self._lock:
            self._snapshots[snapshot.snapshot_id] = snapshot

    async def get_snapshot(self, snapshot_id: uuid.UUID) -> FeatureSnapshot | None:
        async with self._lock:
            return self._snapshots.get(snapshot_id)


class InMemorySignalRepository:
    """In-memory signal repository managing append-only ledger and current pointers."""

    def __init__(self) -> None:
        self._signals: list[Signal] = []
        self._current: dict[tuple[str, str], Signal] = {}
        self._lock = asyncio.Lock()

    async def append_signal(self, signal: Signal) -> None:
        async with self._lock:
            self._signals.append(signal)
            key = (str(signal.market_id), str(signal.horizon))
            curr = self._current.get(key)
            if (
                curr is None
                or signal.cutoff_at > curr.cutoff_at
                or (signal.cutoff_at == curr.cutoff_at and signal.sequence >= curr.sequence)
            ):
                self._current[key] = signal

    async def get_signal(self, signal_id: uuid.UUID) -> Signal | None:
        async with self._lock:
            for s in self._signals:
                if s.signal_id == signal_id:
                    return s
            return None

    async def get_latest_signal(
        self,
        market_id: MarketId,
        horizon: HorizonId,
    ) -> Signal | None:
        async with self._lock:
            return self._current.get((str(market_id), str(horizon)))

    async def list_signals(
        self,
        market_id: MarketId,
        horizon: HorizonId,
        limit: int = 100,
    ) -> Sequence[Signal]:
        async with self._lock:
            matched = [
                s for s in self._signals if s.market_id == market_id and s.horizon == horizon
            ]
            matched.sort(key=lambda s: s.sequence, reverse=True)
            return matched[:limit]


class InMemoryOutboxRepository:
    """In-memory transactional outbox repository."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def append_outbox(
        self,
        event_id: uuid.UUID,
        event_type: str,
        dedupe_key: str,
        payload: Mapping[str, Any],
    ) -> None:
        async with self._lock:
            # Enforce unique dedupe_key
            for entry in self._entries:
                if entry["dedupe_key"] == dedupe_key:
                    # Already exists: skip duplicate insertion
                    return
            self._entries.append(
                {
                    "id": event_id,
                    "event_type": event_type,
                    "dedupe_key": dedupe_key,
                    "payload": dict(payload),
                    "created_at": datetime.now(UTC),
                    "dispatched_at": None,
                    "status": "PENDING",
                }
            )

    async def get_pending(self, limit: int = 50) -> Sequence[Mapping[str, Any]]:
        async with self._lock:
            pending = [e for e in self._entries if e["status"] == "PENDING"]
            return pending[:limit]

    async def mark_dispatched(self, event_ids: Sequence[uuid.UUID]) -> None:
        id_set = set(event_ids)
        now = datetime.now(UTC)
        async with self._lock:
            for e in self._entries:
                if e["id"] in id_set:
                    e["status"] = "DISPATCHED"
                    e["dispatched_at"] = now


class InMemoryInboxRepository:
    """In-memory idempotent inbox repository."""

    def __init__(self) -> None:
        self._processed: set[tuple[str, uuid.UUID]] = set()
        self._lock = asyncio.Lock()

    async def is_processed(self, consumer_name: str, event_id: uuid.UUID) -> bool:
        async with self._lock:
            return (consumer_name, event_id) in self._processed

    async def mark_processed(self, consumer_name: str, event_id: uuid.UUID) -> bool:
        key = (consumer_name, event_id)
        async with self._lock:
            if key in self._processed:
                return False  # Already processed (duplicate)
            self._processed.add(key)
            return True
