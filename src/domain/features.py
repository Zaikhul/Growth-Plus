"""Feature store domain entities, pillars, and snapshot definitions.

Enforces Section 3.7 & 5.3:
- Four distinct feature pillars: TECHNICAL, MACRO, ETF, NEWS
- Maximum 192 feature scalars
- Immutable snapshot associated with decision cutoff and input lineage
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from src.domain.errors import InvariantViolationError
from src.domain.identity import HorizonId, MarketId
from src.domain.time import ensure_utc


class PillarType(StrEnum):
    """The four core signal intelligence pillars."""

    TECHNICAL = "technical"
    MACRO = "macro"
    ETF = "etf"
    NEWS = "news"


class CoverageMode(StrEnum):
    """Approved source coverage modes (PRD Section 3.13)."""

    FULL = "FULL"  # All 4 pillars active
    CORE_NO_ETF = "CORE_NO_ETF"  # Technical, Macro, News
    TECH_MACRO = "TECH_MACRO"  # Technical and Macro only
    RESEARCH = "RESEARCH"  # Unvalidated or exploratory masks


# Canonical backward-compatibility alias
SourceCoverageMode = CoverageMode


MAX_FEATURE_SCALARS = 192


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """Immutable vector of calculated feature scalars at a decision cutoff."""

    snapshot_id: uuid.UUID
    market_id: MarketId
    horizon: HorizonId
    cutoff_at: datetime
    computed_at: datetime
    feature_set_version: str
    scalars: Mapping[str, float]
    active_pillars: tuple[PillarType, ...]
    mode: SourceCoverageMode
    lineage_record_ids: tuple[uuid.UUID, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoff_at", ensure_utc(self.cutoff_at))
        object.__setattr__(self, "computed_at", ensure_utc(self.computed_at))

        if len(self.scalars) > MAX_FEATURE_SCALARS:
            msg = f"Feature count ({len(self.scalars)}) exceeds maximum ({MAX_FEATURE_SCALARS})"
            raise InvariantViolationError(
                msg,
                details={"feature_count": len(self.scalars)},
            )

        # Computation cannot complete before cutoff
        if self.computed_at < self.cutoff_at:
            raise InvariantViolationError("computed_at cannot precede decision cutoff_at")

    @property
    def features(self) -> Mapping[str, float]:
        """Alias for scalars mapping."""
        return self.scalars

    @property
    def decision_cutoff(self) -> datetime:
        """Alias for cutoff_at timestamp."""
        return self.cutoff_at

    @property
    def digest(self) -> str:
        """Deterministic cryptographic digest of snapshot contents."""
        import hashlib
        import json

        payload = {
            "market_id": self.market_id.value,
            "horizon": self.horizon.value,
            "cutoff_at": self.cutoff_at.isoformat(),
            "version": self.feature_set_version,
            "scalars": sorted(self.scalars.items()),
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:32]
