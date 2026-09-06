"""Immutable permitted news metadata; text/model decisions remain explicit inputs."""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

NewsStatus = Literal["RUMOR", "ANNOUNCED", "EFFECTIVE", "DENIED", "CORRECTED"]


@dataclass(frozen=True, slots=True)
class NewsEvidence:
    record_id: str
    source_key: str
    revision: int
    asset: str
    publisher: str
    url: str
    title: str
    title_truncated: bool
    published_at: datetime | None
    source_observed_at: datetime
    first_seen_at: datetime
    available_at: datetime
    event_id: str
    sentiment: float
    relevance: float
    status: NewsStatus | None
    rights_version: str
    model_version: str
    language: str = "English"

    def __post_init__(self) -> None:
        for value in (self.source_observed_at, self.first_seen_at, self.available_at):
            if value.tzinfo is None or value.utcoffset() != timedelta(0):
                raise ValueError("News clocks require UTC")
        if self.source_observed_at > self.first_seen_at or self.first_seen_at > self.available_at:
            raise ValueError("Invalid news publication/availability clocks")
        if (
            len(self.title) > 8000
            or self.revision < 0
            or self.asset not in ("BTC", "ETH")
            or not self.event_id
            or not self.publisher
            or not self.rights_version
            or not self.model_version
            or self.status not in ("RUMOR", "ANNOUNCED", "EFFECTIVE", "DENIED", "CORRECTED", None)
        ):
            raise ValueError("Invalid news metadata")
        if (
            not math.isfinite(self.sentiment)
            or not -1 <= self.sentiment <= 1
            or not math.isfinite(self.relevance)
            or not 0 <= self.relevance <= 1
        ):
            raise ValueError("Invalid news annotation")


@dataclass(frozen=True, slots=True)
class NewsWindow:
    records: tuple[NewsEvidence, ...]
    checked_at: datetime
    coverage_capped: bool
    model_approved: bool
    rights_allowed: bool
