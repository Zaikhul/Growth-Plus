"""News use case depends only on ports and immutable contracts."""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta

from src.domain.contracts.news_evidence import NewsEvidence
from src.ports.news import GdeltTransport, NewsAnnotator, NewsParser, NewsRights, NewsStore


class NewsCollector:
    def __init__(
        self,
        transport: GdeltTransport,
        store: NewsStore,
        rights: NewsRights,
        parser: NewsParser,
        annotator: NewsAnnotator,
        clock: Callable[[], datetime],
    ) -> None:
        self.transport, self.store, self.rights = transport, store, rights
        self.parser, self.annotator, self.clock = parser, annotator, clock

    async def _window(
        self, asset: str, start: datetime, end: datetime
    ) -> tuple[tuple[NewsEvidence, ...], bool]:
        self.rights.assert_collection()
        body = await self.transport.fetch(asset, start, end)
        seen = self.clock()
        records, capped = self.parser.parse(
            body,
            asset=asset,
            first_seen_at=seen,
            available_at=seen,
            rights_version=self.rights.assert_retention(),
            annotator=self.annotator,
        )
        if capped and end - start > timedelta(minutes=1):
            midpoint = start + (end - start) / 2
            left, left_cap = await self._window(asset, start, midpoint)
            right, right_cap = await self._window(asset, midpoint, end)
            return left + right, left_cap or right_cap
        completed = self.clock()
        return tuple(replace(item, available_at=completed) for item in records), capped

    async def collect(self, asset: str, start: datetime, end: datetime) -> None:
        if asset not in ("BTC", "ETH") or not start < end <= self.clock():
            raise ValueError("Invalid news collection window")
        records, capped = await self._window(asset, start, end)
        self.rights.assert_retention()
        await self.store.append(
            records, self.clock(), capped, self.annotator.approved_model_version
        )
