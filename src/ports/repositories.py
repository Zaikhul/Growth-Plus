"""Abstract repository protocols for Growth+ persistence."""

import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol

from src.domain.features import FeatureSnapshot
from src.domain.flows import EtfFlowObservation
from src.domain.identity import AssetId, HorizonId, MarketId
from src.domain.macro import MacroObservation
from src.domain.market import Bar1m, Trade
from src.domain.observations import ObservationEnvelope
from src.domain.signals import Signal


class TradeRepository(Protocol):
    """Port for market trade persistence."""

    async def append_trade(self, trade: Trade) -> None:
        """Append a new market trade."""
        ...

    async def get_trades_window(
        self,
        market_id: MarketId,
        start_time: datetime,
        end_time: datetime,
    ) -> Sequence[Trade]:
        """Fetch trades within a time range."""
        ...

    async def get_latest_trade(self, market_id: MarketId) -> Trade | None:
        """Fetch the most recent trade for a market."""
        ...


class BarRepository(Protocol):
    """Port for canonical 1m bar persistence."""

    async def append_bar(self, bar: Bar1m) -> None:
        """Append a canonical 1-minute closed bar."""
        ...

    async def get_bars(
        self,
        market_id: MarketId,
        start_time: datetime,
        end_time: datetime,
    ) -> Sequence[Bar1m]:
        """Fetch bars within a time range."""
        ...

    async def get_latest_closed_bar(
        self,
        market_id: MarketId,
        as_of_time: datetime,
    ) -> Bar1m | None:
        """Fetch the latest closed bar whose available watermark is <= as_of_time."""
        ...


class ObservationRepository(Protocol):
    """Port for macroeconomic and ETF flow observations."""

    async def append_macro_observation(
        self,
        obs: MacroObservation,
        envelope: ObservationEnvelope,
    ) -> None:
        """Append a macroeconomic series observation with its envelope."""
        ...

    async def get_macro_observations(
        self,
        series_id: str,
        as_of_time: datetime,
    ) -> Sequence[tuple[MacroObservation, ObservationEnvelope]]:
        """Fetch macro observations whose available_at <= as_of_time."""
        ...

    async def append_etf_flow(
        self,
        flow: EtfFlowObservation,
        envelope: ObservationEnvelope,
    ) -> None:
        """Append an ETF flow record with its envelope."""
        ...

    async def get_latest_etf_flow(
        self,
        asset_id: AssetId,
        as_of_time: datetime,
    ) -> tuple[EtfFlowObservation, ObservationEnvelope] | None:
        """Fetch the latest verified ETF flow whose available_at <= as_of_time."""
        ...


class SnapshotRepository(Protocol):
    """Port for immutable feature snapshot persistence."""

    async def append_snapshot(self, snapshot: FeatureSnapshot) -> None:
        """Append an immutable feature snapshot."""
        ...

    async def get_snapshot(self, snapshot_id: uuid.UUID) -> FeatureSnapshot | None:
        """Fetch snapshot by unique ID."""
        ...


class SignalRepository(Protocol):
    """Port for signal decisions and current pointers."""

    async def append_signal(self, signal: Signal) -> None:
        """Append an evaluated signal decision."""
        ...

    async def get_signal(self, signal_id: uuid.UUID) -> Signal | None:
        """Fetch signal by unique ID."""
        ...

    async def get_latest_signal(
        self,
        market_id: MarketId,
        horizon: HorizonId,
    ) -> Signal | None:
        """Fetch the current latest signal for a market and horizon."""
        ...

    async def list_signals(
        self,
        market_id: MarketId,
        horizon: HorizonId,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Signal]:
        """Fetch historical signals ordered by sequence descending."""
        ...

    async def count_signals(
        self,
        market_id: MarketId,
        horizon: HorizonId,
    ) -> int:
        """Count total historical signals for a market and horizon."""
        ...


class OutboxRepository(Protocol):
    """Port for transactional outbox persistence."""

    async def append_outbox(
        self,
        event_id: uuid.UUID,
        event_type: str,
        dedupe_key: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Insert outbox record within the current transaction."""
        ...

    async def get_pending(self, limit: int = 50) -> Sequence[Mapping[str, Any]]:
        """Fetch pending outbox records ready for relay."""
        ...

    async def mark_dispatched(self, event_ids: Sequence[uuid.UUID]) -> None:
        """Mark outbox records as successfully dispatched to broker."""
        ...


class InboxRepository(Protocol):
    """Port for idempotent consumer inbox deduplication."""

    async def is_processed(self, consumer_name: str, event_id: uuid.UUID) -> bool:
        """Check if event has already been processed by consumer."""
        ...

    async def mark_processed(self, consumer_name: str, event_id: uuid.UUID) -> bool:
        """Record event processing. Returns True if recorded, False if duplicate."""
        ...
