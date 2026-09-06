"""Point-in-Time (PIT) time semantics and clock structures for Growth+.

Enforces microsecond UTC precision and strict anti-leakage invariants:
- available_at <= decision_cutoff
- first_seen_at <= available_at (live captures)
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from src.domain.errors import PointInTimeViolationError


class AvailabilityEvidenceTier(StrEnum):
    """Availability evidence tier for datasets (PRD Section 3.1)."""

    CAPTURED_LIVE = "CAPTURED_LIVE"
    RELEASE_ARCHIVE_VERIFIED = "RELEASE_ARCHIVE_VERIFIED"
    ASSUMED_LAG = "ASSUMED_LAG"


def ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware and converted to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class TimeEnvelope:
    """Canonical Point-in-Time timestamp envelope matching PRD Section 3.1.

    Separates physical occurrence, publisher publication assertion, local ingestion,
    and verified decision availability.
    """

    event_time: datetime
    reference_period: str
    first_seen_at: datetime
    available_at: datetime
    published_at: datetime | None = None
    ingested_at: datetime | None = None
    validated_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    def __post_init__(self) -> None:
        # Validate all datetimes are UTC aware
        object.__setattr__(self, "event_time", ensure_utc(self.event_time))
        object.__setattr__(self, "first_seen_at", ensure_utc(self.first_seen_at))
        object.__setattr__(self, "available_at", ensure_utc(self.available_at))

        if self.published_at is not None:
            object.__setattr__(self, "published_at", ensure_utc(self.published_at))
        if self.ingested_at is not None:
            object.__setattr__(self, "ingested_at", ensure_utc(self.ingested_at))
        if self.validated_at is not None:
            object.__setattr__(self, "validated_at", ensure_utc(self.validated_at))
        if self.valid_from is not None:
            object.__setattr__(self, "valid_from", ensure_utc(self.valid_from))
        if self.valid_to is not None:
            object.__setattr__(self, "valid_to", ensure_utc(self.valid_to))

        # Basic invariant: available_at cannot precede first_seen_at in live feeds
        # (For historical archives, available_at is documented by release archive timestamp)
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise PointInTimeViolationError(
                "valid_from cannot be strictly after valid_to",
                details={"valid_from": str(self.valid_from), "valid_to": str(self.valid_to)},
            )


@dataclass(frozen=True, slots=True)
class PointInTimeCutoff:
    """Strict Point-in-Time decision cutoff filter.

    Assures that any observation joined or used in computation has
    available_at <= cutoff_at.
    """

    cutoff_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoff_at", ensure_utc(self.cutoff_at))

    def is_available(self, available_at: datetime) -> bool:
        """Check if an item available at `available_at` can be observed at cutoff."""
        return ensure_utc(available_at) <= self.cutoff_at

    def assert_available(self, available_at: datetime, record_id: str | None = None) -> None:
        """Raise PointInTimeViolationError if availability leaks future information."""
        utc_avail = ensure_utc(available_at)
        if utc_avail > self.cutoff_at:
            raise PointInTimeViolationError(
                f"PIT Leakage detected: available_at ({utc_avail.isoformat()}) > "
                f"decision_cutoff ({self.cutoff_at.isoformat()})",
                details={
                    "available_at": utc_avail.isoformat(),
                    "decision_cutoff": self.cutoff_at.isoformat(),
                    "record_id": record_id or "unknown",
                },
            )
