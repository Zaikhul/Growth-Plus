"""Macroeconomic entities, series definitions, and event types.

Enforces:
- Canonical series identities
- Month-on-month, year-on-year, and basis point transformations
- Scheduled vs unexpected economic release tags
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from src.domain.identity import SourceId
from src.domain.time import ensure_utc


class MacroSeriesId(StrEnum):
    """Canonical macroeconomic series identifiers (PRD Section 3.3)."""

    US_FED_TARGET_LOWER = "US_FED_TARGET_LOWER"
    US_FED_TARGET_UPPER = "US_FED_TARGET_UPPER"
    US_EFFR = "US_EFFR"
    US_CPI_HEADLINE_SA = "US_CPI_HEADLINE_SA"
    US_CPI_HEADLINE_NSA = "US_CPI_HEADLINE_NSA"
    US_CPI_CORE_SA = "US_CPI_CORE_SA"
    US_CPI_CORE_NSA = "US_CPI_CORE_NSA"
    US_UST_2Y = "US_UST_2Y"
    US_UST_10Y = "US_UST_10Y"


class MacroEventType(StrEnum):
    """Official macroeconomic calendar events."""

    FOMC_DECISION = "FOMC_DECISION"
    FOMC_MINUTES = "FOMC_MINUTES"
    CPI_RELEASE = "CPI_RELEASE"


@dataclass(frozen=True, slots=True)
class MacroObservation:
    """Standardized macroeconomic series observation record."""

    series_id: MacroSeriesId
    source_id: SourceId
    reference_period: str  # e.g. "2026-08" or "2026-09-04"
    value: float
    unit: str  # e.g. "INDEX", "PERCENT", "BASIS_POINTS"
    revision_seq: int = 0
    supersedes_id: str | None = None
    is_seasonally_adjusted: bool = False

    def basis_point_change(self, prior_value: float | None) -> float | None:
        """Calculate basis point change (1.00% = 100 bps) from prior value."""
        if prior_value is None:
            return None
        return (self.value - prior_value) * 100.0

    def percent_change(self, prior_value: float | None) -> float | None:
        """Calculate percentage change 100 * (val / prior - 1)."""
        if prior_value is None or prior_value == 0.0:
            return None
        return 100.0 * ((self.value / prior_value) - 1.0)


@dataclass(frozen=True, slots=True)
class MacroReleaseCalendarEvent:
    """Scheduled macroeconomic release event for event blackout policy."""

    event_type: MacroEventType
    scheduled_at: datetime
    first_seen_at: datetime | None = None
    is_unexpected: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "scheduled_at", ensure_utc(self.scheduled_at))
        if self.first_seen_at is not None:
            object.__setattr__(self, "first_seen_at", ensure_utc(self.first_seen_at))
