"""News, announcements, and sentiment domain entities.

Enforces Section 3.5:
- Syndication clustering and deduplication tracking
- Title and entity metadata boundaries (no raw copyrighted full-article retention)
- Authoritative regulatory vs third-party source tagging
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from src.domain.identity import AssetId, SourceId
from src.domain.time import ensure_utc


class NewsSourceCategory(StrEnum):
    """Categorization of news and event publishers."""

    REGULATORY_OFFICIAL = "REGULATORY_OFFICIAL"  # SEC, Fed, CFTC
    CRYPTO_NATIVE_MEDIA = "CRYPTO_NATIVE_MEDIA"  # CoinDesk, Cointelegraph
    MAINSTREAM_FINANCIAL = "MAINSTREAM_FINANCIAL"  # Reuters, Bloomberg
    AGGREGATOR = "AGGREGATOR"  # GDELT, CryptoPanic


@dataclass(frozen=True, slots=True)
class NewsItem:
    """Canonical news/event metadata item."""

    item_id: str
    source_id: SourceId
    category: NewsSourceCategory
    title: str
    published_at: datetime
    raw_digest: str
    cluster_id: str | None = None
    asset_id: AssetId | None = None
    entity_tags: tuple[str, ...] = field(default_factory=tuple)
    is_syndicated_duplicate: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "published_at", ensure_utc(self.published_at))
