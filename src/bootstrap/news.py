"""Composition root: explicit rights and approved model dependencies, no permissive defaults."""

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncEngine

from src.adapters.persistence.postgres.news import PostgresNewsStore
from src.adapters.providers.gdelt.parser import GdeltParser
from src.adapters.providers.gdelt.transport import PostgresBudgetedGdeltTransport
from src.application.news_collection import NewsCollector
from src.domain.contracts.news_evidence import NewsWindow
from src.ports.news import NewsAnnotator, NewsRights


class AssetNewsReader:
    def __init__(self, stores: Mapping[str, PostgresNewsStore]) -> None:
        self.stores = dict(stores)

    async def read(self, asset: str, cutoff: datetime) -> NewsWindow:
        return await self.stores[asset].read(asset, cutoff)


def build_news_services(
    engine: AsyncEngine, rights: NewsRights, annotator: NewsAnnotator, contact_user_agent: str
) -> tuple[AssetNewsReader, dict[str, NewsCollector]]:
    transport = PostgresBudgetedGdeltTransport(engine, contact_user_agent)
    stores = {
        asset: PostgresNewsStore(engine, asset, rights, annotator.approved_model_version)
        for asset in ("BTC", "ETH")
    }
    collectors = {
        asset: NewsCollector(
            transport, store, rights, GdeltParser(), annotator, lambda: datetime.now(UTC)
        )
        for asset, store in stores.items()
    }
    return AssetNewsReader(stores), collectors


async def run_news_schedule(collectors: Mapping[str, NewsCollector], stop: asyncio.Event) -> None:
    while not stop.is_set():
        end = datetime.now(UTC)
        for asset, collector in collectors.items():
            await collector.collect(asset, end - timedelta(minutes=30), end)
        # Failure propagates to supervisor; no false healthy poll is recorded.
        next_slot = (int(end.timestamp()) // 900 + 1) * 900
        delay = max(0, next_slot - datetime.now(UTC).timestamp())
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
        except TimeoutError:
            continue
