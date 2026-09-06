"""ETF institutional product flows domain entities.

Enforces Section 3.4:
- Separation of U.S. spot daily net flows from weekly or global products
- Status lifecycle: PRELIMINARY -> PARTIAL -> SOURCE_COMPLETE -> RECONCILED (or DISPUTED)
- Coverage ratios and constituent fund reconciliation
"""

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from src.domain.errors import InvariantViolationError
from src.domain.identity import AssetId


class EtfProductStatus(StrEnum):
    """ETF flow session data status."""

    PRELIMINARY = "PRELIMINARY"
    PARTIAL = "PARTIAL"
    SOURCE_COMPLETE = "SOURCE_COMPLETE"
    RECONCILED = "RECONCILED"
    DISPUTED = "DISPUTED"


@dataclass(frozen=True, slots=True)
class EtfFlowObservation:
    """Canonical ETF daily flow observation record."""

    asset_id: AssetId
    fund_id: str  # e.g. "IBIT", "FBTC", "GBTC", "TOTAL"
    ticker_at_time: str
    issuer: str
    jurisdiction: str  # e.g. "US"
    session_date: date
    flow_usd: float
    status: EtfProductStatus
    covered_funds: int
    expected_funds: int
    is_seed_or_conversion: bool = False
    revision_seq: int = 0

    def __post_init__(self) -> None:
        if self.expected_funds <= 0:
            raise InvariantViolationError("expected_funds must be strictly positive")
        if self.covered_funds < 0 or self.covered_funds > self.expected_funds:
            raise InvariantViolationError(
                f"covered_funds ({self.covered_funds}) must be in [0, {self.expected_funds}]"
            )

    @property
    def coverage_ratio(self) -> float:
        return float(self.covered_funds) / float(self.expected_funds)

    @property
    def is_complete(self) -> bool:
        return self.status in (EtfProductStatus.SOURCE_COMPLETE, EtfProductStatus.RECONCILED)
