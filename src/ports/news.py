"""News boundaries: provider I/O, storage and approved text interpretation."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from src.domain.contracts.news_evidence import NewsEvidence, NewsStatus, NewsWindow


class NewsRights(Protocol):
    def assert_collection(self) -> str: ...
    def assert_retention(self) -> str: ...
    def assert_feature_use(self) -> str: ...


class NewsAnnotator(Protocol):
    @property
    def approved_model_version(self) -> str: ...
    def annotate(
        self, title: str, asset: str, url: str, published_at: datetime
    ) -> tuple[str, float, float, NewsStatus]:
        """Return originating event ID, sentiment, relevance, supported event status."""
        ...


class GdeltTransport(Protocol):
    async def fetch(self, asset: str, start: datetime, end: datetime) -> bytes:
        """Admitted DOC ArtList JSON only; global one-at-a-time 12/minute budget."""
        ...


class NewsStore(Protocol):
    async def append(
        self,
        records: Sequence[NewsEvidence],
        checked_at: datetime,
        coverage_capped: bool,
        model_version: str,
    ) -> None: ...
    async def read(self, asset: str, cutoff: datetime) -> NewsWindow: ...


class NewsParser(Protocol):
    def parse(
        self,
        body: bytes,
        *,
        asset: str,
        first_seen_at: datetime,
        available_at: datetime,
        rights_version: str,
        annotator: NewsAnnotator,
    ) -> tuple[tuple[NewsEvidence, ...], bool]: ...


class NewsReader(Protocol):
    async def read(self, asset: str, cutoff: datetime) -> NewsWindow: ...
