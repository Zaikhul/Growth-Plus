"""Canonical identity types and market identifiers for Growth+.

Enforces venue-scoped market identities (e.g. binance:BTCUSDT vs coinbase:BTC-USD).
"""

import re
import uuid
from dataclasses import dataclass
from enum import StrEnum

from src.domain.errors import InvalidEntityIdError


class AssetId(StrEnum):
    """Supported crypto asset identities."""

    BTC = "BTC"
    ETH = "ETH"


class QuoteCurrency(StrEnum):
    """Supported quote currency identities."""

    USD = "USD"
    USDT = "USDT"


class VenueId(StrEnum):
    """Supported execution and data venues."""

    BINANCE = "binance"
    COINBASE = "coinbase"


class HorizonId(StrEnum):
    """Supported forecasting horizon families."""

    SCALP_15M = "scalp_15m"
    SWING_24H = "swing_24h"
    POSITION_30D = "position_30d"

    @property
    def forecast_seconds(self) -> int:
        match self:
            case HorizonId.SCALP_15M:
                return 900
            case HorizonId.SWING_24H:
                return 86400
            case HorizonId.POSITION_30D:
                return 2592000

    @property
    def action_delay_seconds(self) -> int:
        match self:
            case HorizonId.SCALP_15M:
                return 3
            case HorizonId.SWING_24H:
                return 10
            case HorizonId.POSITION_30D:
                return 60

    @property
    def baseline_hurdle_b(self) -> float:
        match self:
            case HorizonId.SCALP_15M:
                return 0.001
            case HorizonId.SWING_24H:
                return 0.003
            case HorizonId.POSITION_30D:
                return 0.015

    @property
    def volatility_multiplier_k(self) -> float:
        match self:
            case HorizonId.SCALP_15M:
                return 0.25
            case HorizonId.SWING_24H:
                return 0.35
            case HorizonId.POSITION_30D:
                return 0.50


_MARKET_ID_PATTERN = re.compile(r"^[a-z0-9_-]+:[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class MarketId:
    """Venue-scoped market identity (e.g. 'binance:BTCUSDT' or 'coinbase:BTC-USD').

    In accordance with PRD Section 1.3:
    BTC/USDT must never be relabeled BTC/USD without an independently validated conversion.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value or not _MARKET_ID_PATTERN.match(self.value):
            raise InvalidEntityIdError(
                f"Market ID '{self.value}' must follow format '<venue>:<symbol>'",
                details={"value": self.value},
            )

    @classmethod
    def from_parts(cls, venue: VenueId | str, symbol: str) -> "MarketId":
        venue_str = venue.value if isinstance(venue, VenueId) else str(venue)
        return cls(f"{venue_str}:{symbol}")

    @property
    def venue(self) -> str:
        return self.value.split(":", 1)[0]

    @property
    def symbol(self) -> str:
        return self.value.split(":", 1)[1]

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class SourceId:
    """Registered data provider source identity."""

    value: str

    def __post_init__(self) -> None:
        if not self.value or len(self.value) > 64:
            raise InvalidEntityIdError(
                f"Invalid source identifier: {self.value}",
                details={"value": self.value},
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class DatasetId:
    """Registered dataset identity under a source."""

    value: str

    def __post_init__(self) -> None:
        if not self.value or len(self.value) > 64:
            raise InvalidEntityIdError(
                f"Invalid dataset identifier: {self.value}",
                details={"value": self.value},
            )

    def __str__(self) -> str:
        return self.value


def generate_uuid() -> uuid.UUID:
    """Generate a random UUIDv4."""
    return uuid.uuid4()
