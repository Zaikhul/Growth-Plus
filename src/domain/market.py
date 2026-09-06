"""Domain entities and invariants for market trades and canonical bars.

Enforces:
- Closed-bar availability watermarks (bar_close_at + 2 seconds)
- Non-negative volumes and prices
- OHLC consistency (high >= max(open, close), low <= min(open, close))
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from src.domain.errors import InvariantViolationError
from src.domain.identity import MarketId
from src.domain.time import ensure_utc


class TradeSide(StrEnum):
    """Trade execution side."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class Trade:
    """Individual trade event on a venue-scoped market."""

    trade_id: str
    market_id: MarketId
    price_usd: float
    size: float
    side: TradeSide
    event_time: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_time", ensure_utc(self.event_time))
        if self.price_usd <= 0.0:
            raise InvariantViolationError(
                f"Trade price must be strictly positive, got {self.price_usd}",
                details={"price_usd": self.price_usd, "trade_id": self.trade_id},
            )
        if self.size <= 0.0:
            raise InvariantViolationError(
                f"Trade size must be strictly positive, got {self.size}",
                details={"size": self.size, "trade_id": self.trade_id},
            )


@dataclass(frozen=True, slots=True)
class Bar1m:
    """Canonical 1-minute aggregated market bar."""

    market_id: MarketId
    open_usd: float
    high_usd: float
    low_usd: float
    close_usd: float
    volume: float
    bar_start_at: datetime
    bar_close_at: datetime
    trade_count: int
    vwap_usd: float
    is_closed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "bar_start_at", ensure_utc(self.bar_start_at))
        object.__setattr__(self, "bar_close_at", ensure_utc(self.bar_close_at))

        if self.bar_start_at >= self.bar_close_at:
            raise InvariantViolationError(
                "bar_start_at must be strictly before bar_close_at",
                details={
                    "bar_start_at": self.bar_start_at.isoformat(),
                    "bar_close_at": self.bar_close_at.isoformat(),
                },
            )
        if min(self.open_usd, self.high_usd, self.low_usd, self.close_usd) <= 0.0:
            raise InvariantViolationError("All OHLC prices must be strictly positive")

        if self.high_usd < max(self.open_usd, self.close_usd):
            raise InvariantViolationError("High price must be >= max(open, close)")

        if self.low_usd > min(self.open_usd, self.close_usd):
            raise InvariantViolationError("Low price must be <= min(open, close)")

        if self.volume < 0.0:
            raise InvariantViolationError("Volume must be non-negative")

    @property
    def available_for_decision_at(self) -> datetime:
        """Closed 1m bars are eligible for decisions after bar_close_at + 2s (PRD Section 3.12)."""
        return self.bar_close_at + timedelta(seconds=2)
