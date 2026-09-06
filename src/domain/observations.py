"""Universal canonical observation envelope for Growth+.

Enforces Section 3.1:
- Explicit record identity and provenance tracking
- Raw byte digest and canonical digest (SHA-256)
- Complete TimeEnvelope with microsecond UTC timestamps
- Monotonic revision sequencing and append-only invariants
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.domain.identity import AssetId, DatasetId, MarketId, SourceId
from src.domain.time import TimeEnvelope


class QualityFlag(StrEnum):
    """Quality and degradation diagnostic flags."""

    OK = "OK"
    PARTIAL = "PARTIAL"
    REVISED = "REVISED"
    CLOCK_SKEW = "CLOCK_SKEW"
    GAP = "GAP"
    STALE = "STALE"
    UNIT_UNVERIFIED = "UNIT_UNVERIFIED"
    QUARANTINED = "QUARANTINED"


@dataclass(frozen=True, slots=True)
class ObservationEnvelope:
    """Canonical observation record wrapper."""

    record_id: uuid.UUID
    source_id: SourceId
    dataset_id: DatasetId
    source_record_key: str
    schema_version: str
    parser_version: str
    time_envelope: TimeEnvelope
    raw_digest: str  # SHA-256 over exact raw bytes
    canonical_digest: str  # SHA-256 over normalized canonical representation
    rights_policy_id: str
    rights_version: str
    asset_id: AssetId | None = None
    market_id: MarketId | None = None
    revision_seq: int = 0
    supersedes_id: uuid.UUID | None = None
    quality_flags: tuple[QualityFlag, ...] = field(default_factory=lambda: (QualityFlag.OK,))
    payload: Mapping[str, Any] = field(default_factory=dict)
