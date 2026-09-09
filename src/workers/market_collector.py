"""Market collector worker ingesting raw trades and aggregating 1-minute bars.

Enforces PRD Section 3.6 & 4.2:
- Subscribes to trades or ingests venue feed batches
- Computes running VWAP and aggregates closed Bar1m candles
- Enforces +2-second watermark on bar availability
- Commits bars to BarRepository and emits event to growth.raw.market.v1
"""

import uuid
from collections.abc import Sequence
from datetime import datetime

from src.adapters.messaging.envelope import EventEnvelope
from src.domain.identity import MarketId, SourceId
from src.domain.market import Bar1m, Trade
from src.domain.rights import DataOperation
from src.features.technical.bars import BarAggregator
from src.ports.event_bus import EventBus
from src.ports.repositories import BarRepository, OutboxRepository
from src.ports.rights_authorizer import RightsAuthorizer


class MarketCollectorWorker:
    """Worker aggregating real-time trades into authoritative closed bars."""

    def __init__(
        self,
        market_id: MarketId,
        bar_repo: BarRepository,
        event_bus: EventBus | None = None,
        rights_authorizer: RightsAuthorizer | None = None,
        source_id: SourceId | None = None,
        outbox_repo: OutboxRepository | None = None,
    ) -> None:
        self._market_id = market_id
        self._bar_repo = bar_repo
        self._event_bus = event_bus
        self._authorizer = rights_authorizer
        self._source_id = source_id or SourceId("binance_live")
        self._outbox_repo = outbox_repo
        self._aggregator = BarAggregator()
        self._running = False

    async def ingest_trades(
        self,
        trades: Sequence[Trade],
        watermark_time: datetime | None = None,
        force_close: bool = False,
    ) -> list[Bar1m]:
        """Ingest a batch of incoming trades and commit any newly closed bars."""
        # 1. Authorize ingestion operation if authorizer configured
        if self._authorizer is not None:
            decision = self._authorizer.authorize(
                source_id=self._source_id.value,
                dataset_id="spot_trades",
                operation=DataOperation.COLLECT,
            )
            decision.assert_permitted()

        closed_bars = self._aggregator.aggregate_trades(
            self._market_id,
            trades,
            watermark_time=watermark_time,
            force_close=force_close,
        )
        for new_bar in closed_bars:
            # 2. Commit closed bar to repository
            await self._bar_repo.append_bar(new_bar)

            payload = {
                "market_id": self._market_id.value,
                "open_time": new_bar.bar_start_at.isoformat(),
                "close_time": new_bar.bar_close_at.isoformat(),
                "available_at": new_bar.available_for_decision_at.isoformat(),
                "close_usd": new_bar.close_usd,
                "volume": new_bar.volume,
                "vwap_usd": new_bar.vwap_usd,
            }
            dedupe_key = f"{self._market_id.value}-{new_bar.bar_close_at.isoformat()}"
            event_id = uuid.uuid4()

            # 3. Commit to outbox repository for atomic reliability
            if self._outbox_repo is not None:
                await self._outbox_repo.append_outbox(
                    event_id=event_id,
                    event_type="market.bar.closed",
                    dedupe_key=dedupe_key,
                    payload=payload,
                )

            # 4. Publish durable event on message bus if wired
            if self._event_bus is not None:
                envelope = EventEnvelope(
                    event_id=event_id,
                    subject=f"growth.raw.market.v1.{self._market_id.value}",
                    occurred_at=new_bar.bar_close_at,
                    dedupe_key=dedupe_key,
                    source="market-collector",
                    payload=payload,
                )
                await self._event_bus.publish(envelope)

        return closed_bars
