"""Deterministic, rights-aware long-only spot paper execution. Never places orders."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal

D = Decimal
Asset = Literal["BTC", "ETH"]
Side = Literal["BUY", "SELL"]
Label = Literal["Strong Buy", "Buy", "Neutral", "Sell", "Strong Sell"]


@dataclass(frozen=True, slots=True)
class CostPolicy:
    fee_bps: Decimal = D("20")
    impact_bps: Decimal = D("5")
    scenario: str = "PRIMARY_CONSERVATIVE"
    spread_multiplier: Decimal = D("1")
    spread_floor_bps: Decimal = D("0")
    extra_delay_seconds: int = 0

    def __post_init__(self) -> None:
        values = (self.fee_bps, self.impact_bps, self.spread_multiplier, self.spread_floor_bps)
        if any(not value.is_finite() or value < 0 for value in values):
            raise ValueError("Invalid cost policy")
        if self.spread_multiplier < 1 or self.extra_delay_seconds < 0:
            raise ValueError("Invalid spread/delay policy")


LOW_COST = CostPolicy(D("2"), D("1"), "LOW_COST_SENSITIVITY")
HORIZONS = {"scalp_15m": (900, 3), "swing_24h": (86400, 10), "position_30d": (2592000, 60)}


def severe_cost(horizon: str) -> CostPolicy:
    return CostPolicy(
        D("60"),
        D("20"),
        "SEVERE_COST",
        D("2"),
        D("20"),
        {"scalp_15m": 30, "swing_24h": 300, "position_30d": 3600}[horizon],
    )


@dataclass(frozen=True, slots=True)
class Quote:
    asset: Asset
    observed_at: datetime
    available_at: datetime
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    preceding_minute_notional: Decimal
    book_id: str
    rights_allowed: bool = True

    def __post_init__(self) -> None:
        for instant in (self.observed_at, self.available_at):
            if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
                raise ValueError("Quote timestamps must be UTC")
        values = (self.bid, self.ask, self.bid_size, self.ask_size, self.preceding_minute_notional)
        if any(not value.is_finite() or value < 0 for value in values):
            raise ValueError("Invalid quote/depth")
        if not D("0") < self.bid <= self.ask or self.observed_at > self.available_at:
            raise ValueError("Invalid quote clock/spread")
        if self.asset not in ("BTC", "ETH") or not self.book_id:
            raise ValueError("Invalid asset/book identity")


@dataclass(frozen=True, slots=True)
class Signal:
    asset: Asset
    label: Label | None
    issued_at: datetime
    expires_at: datetime
    unavailable: bool = False

    def __post_init__(self) -> None:
        if self.asset not in ("BTC", "ETH") or self.label not in (
            "Strong Buy",
            "Buy",
            "Neutral",
            "Sell",
            "Strong Sell",
            None,
        ):
            raise ValueError("Invalid paper signal")
        for value in (self.issued_at, self.expires_at):
            if value.tzinfo is None or value.utcoffset() != timedelta(0):
                raise ValueError("Signal timestamps require UTC")
        if self.expires_at <= self.issued_at:
            raise ValueError("Invalid signal validity interval")


@dataclass(frozen=True, slots=True)
class Fill:
    asset: Asset
    side: Side
    at: datetime
    quantity: Decimal
    price: Decimal
    fee: Decimal
    book_id: str


@dataclass(slots=True)
class Position:
    quantity: Decimal
    entered_at: datetime


@dataclass(slots=True)
class Order:
    asset: Asset
    side: Side
    eligible_at: datetime
    expires_at: datetime | None
    budget: Decimal = D("0")
    remaining: Decimal = D("0")
    first_quote_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PaperReport:
    simulated: bool
    horizon: str
    initial_equity: Decimal
    cash: Decimal
    final_equity: Decimal | None
    fee_bps_per_side: Decimal
    impact_bps_per_side: Decimal
    fill_model: str
    scenario: str
    action_delay_seconds: int
    fills: tuple[Fill, ...]
    censored_assets: tuple[Asset, ...]
    unfilled_orders: int
    assumptions: tuple[str, ...]
    pending_orders: int
    valuation_ages_seconds: tuple[tuple[str, float], ...]


@dataclass
class PaperSimulator:
    horizon: str
    costs: CostPolicy = field(default_factory=CostPolicy)
    cash: Decimal = field(init=False, default=D("100000"))
    positions: dict[Asset, Position] = field(init=False, default_factory=dict)
    orders: dict[Asset, Order] = field(init=False, default_factory=dict)
    quotes: dict[Asset, Quote] = field(init=False, default_factory=dict)
    fills: list[Fill] = field(init=False, default_factory=list)
    denied: set[Asset] = field(init=False, default_factory=set)
    used_books: dict[tuple[Asset, str, Side], Decimal] = field(init=False, default_factory=dict)
    recent_fills: list[tuple[datetime, Asset, Decimal]] = field(init=False, default_factory=list)
    unfilled_orders: int = field(init=False, default=0)
    previous_at: datetime | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.horizon not in HORIZONS:
            raise ValueError("Unsupported paper horizon")

    def equity(self) -> Decimal | None:
        if any(asset in self.denied or asset not in self.quotes for asset in self.positions):
            return None
        return self.cash + sum(
            (
                position.quantity * self.quotes[asset].bid
                for asset, position in self.positions.items()
            ),
            D("0"),
        )

    def _exit(self, asset: Asset, eligible: datetime) -> None:
        position = self.positions.get(asset)
        if position is None:
            return
        existing = self.orders.get(asset)
        if existing is not None and existing.side == "SELL":
            return
        self.orders[asset] = Order(asset, "SELL", eligible, None, remaining=position.quantity)

    def step(
        self,
        at: datetime,
        *,
        quotes: tuple[Quote, ...] = (),
        signals: tuple[Signal, ...] = (),
        rights_denied: tuple[Asset, ...] = (),
    ) -> None:
        if at.tzinfo is None or at.utcoffset() != timedelta(0):
            raise ValueError("Simulation uses UTC")
        if self.previous_at is not None and at <= self.previous_at:
            raise ValueError(
                "Frames must have strictly increasing times; batch simultaneous events"
            )
        self.previous_at = at
        hold_seconds, delay = HORIZONS[self.horizon]
        delay += self.costs.extra_delay_seconds
        for asset in rights_denied:
            self.denied.add(asset)
            self.quotes.pop(asset, None)
            self.orders.pop(asset, None)
        for quote in quotes:
            if quote.available_at > at or quote.observed_at > at:
                raise ValueError("Future quote leakage")
            if quote.asset in self.denied or not quote.rights_allowed:
                continue
            self.quotes[quote.asset] = quote
        for asset, position in self.positions.items():
            expiry = position.entered_at + timedelta(seconds=hold_seconds)
            if at >= expiry:
                self._exit(asset, expiry)
        for signal in signals:
            if signal.asset not in ("BTC", "ETH") or signal.issued_at > at:
                raise ValueError("Invalid signal asset/clock")
            if signal.asset in self.denied or (not signal.unavailable and at >= signal.expires_at):
                continue
            eligible = signal.issued_at + timedelta(seconds=delay)
            if signal.unavailable or signal.label in ("Sell", "Strong Sell"):
                if self.orders.get(signal.asset) and self.orders[signal.asset].side == "BUY":
                    self.orders.pop(signal.asset)
                self._exit(signal.asset, eligible)
            elif (
                signal.label in ("Buy", "Strong Buy")
                and at < signal.expires_at
                and signal.asset not in self.positions
                and signal.asset not in self.orders
            ):
                equity = self.equity()
                if equity is not None:
                    reserved = sum(
                        (order.budget for order in self.orders.values() if order.side == "BUY"),
                        D("0"),
                    )
                    budget = min(equity * D(".5"), max(D("0"), self.cash - reserved))
                    self.orders[signal.asset] = Order(
                        signal.asset, "BUY", eligible, signal.expires_at, budget=budget
                    )
        self.recent_fills = [
            row for row in self.recent_fills if row[0] > at - timedelta(seconds=60)
        ]
        for asset, order in tuple(self.orders.items()):
            if asset in self.denied:
                continue
            if order.expires_at is not None and at >= order.expires_at:
                self.unfilled_orders += 1
                del self.orders[asset]
                continue
            if order.first_quote_at is not None and at > order.first_quote_at + timedelta(
                seconds=60
            ):
                self.unfilled_orders += 1
                del self.orders[asset]
                continue
            target_quote: Quote | None = next(
                (
                    q
                    for q in quotes
                    if q.asset == asset and q.rights_allowed and q.observed_at >= order.eligible_at
                ),
                None,
            )
            if target_quote is None or at < order.eligible_at:
                continue  # Venue outage cannot produce a fill at a cached favorable price.
            if order.first_quote_at is None:
                order.first_quote_at = at
            mid = (target_quote.bid + target_quote.ask) / 2
            spread = max(
                (target_quote.ask - target_quote.bid) * self.costs.spread_multiplier,
                mid * self.costs.spread_floor_bps / D("10000"),
            )
            executable = mid + spread / 2 if order.side == "BUY" else mid - spread / 2
            impact = self.costs.impact_bps / D("10000")
            price = executable * (1 + impact if order.side == "BUY" else 1 - impact)
            fee_rate = self.costs.fee_bps / D("10000")
            if price <= 0:
                raise ValueError("Nonpositive stressed executable price")
            book_key = (asset, target_quote.book_id, order.side)
            depth = target_quote.ask_size if order.side == "BUY" else target_quote.bid_size
            depth = max(D("0"), depth - self.used_books.get(book_key, D("0")))
            participation = max(
                D("0"),
                target_quote.preceding_minute_notional * D(".01")
                - sum((notional for _, a, notional in self.recent_fills if a == asset), D("0")),
            )
            wanted = (
                min(order.budget, self.cash) / (price * (1 + fee_rate))
                if order.side == "BUY"
                else order.remaining
            )
            quantity = min(wanted, depth, participation / price)
            if quantity <= 0:
                continue
            notional, fee = quantity * price, quantity * price * fee_rate
            self.used_books[book_key] = self.used_books.get(book_key, D("0")) + quantity
            self.recent_fills.append((at, asset, notional))
            self.fills.append(
                Fill(asset, order.side, at, quantity, price, fee, target_quote.book_id)
            )
            if order.side == "BUY":
                self.cash -= notional + fee
                order.budget -= notional + fee
                curr_pos = self.positions.get(asset)
                if curr_pos:
                    curr_pos.quantity += quantity
                else:
                    self.positions[asset] = Position(quantity, at)
                done = order.budget <= D("0.00000001")
            else:
                self.cash += notional - fee
                order.remaining -= quantity
                self.positions[asset].quantity -= quantity
                done = order.remaining <= D("0.00000001")
                if done:
                    del self.positions[asset]
            if done:
                del self.orders[asset]
            if self.cash < 0:
                raise AssertionError("Paper execution cannot borrow cash")

    def report(self) -> PaperReport:
        return PaperReport(
            True,
            self.horizon,
            D("100000"),
            self.cash,
            self.equity(),
            self.costs.fee_bps,
            self.costs.impact_bps,
            "NEXT_ELIGIBLE_BID_ASK_WITH_DEPTH",
            self.costs.scenario,
            HORIZONS[self.horizon][1] + self.costs.extra_delay_seconds,
            tuple(self.fills),
            tuple(sorted(self.denied)),
            self.unfilled_orders,
            (
                "Long-only spot; no leverage; 50% asset/100% gross entry allocation caps",
                "Zero exchange-cash yield; bid/ask spread is paid through execution prices once",
                "Depth and trailing 60-second notional required; OHLC-only scalp unsupported",
                "1% participation; partial execution window 60 seconds after first eligible quote",
                "Quotes/decisions must carry PIT-verified rights and availability",
                "Terminal open positions are marked at latest entitled observable bid; report gaps",
                "Primary defaults are simulation fixtures, not current venue fee quotations",
            ),
            len(self.orders),
            tuple(
                (asset, (self.previous_at - quote.available_at).total_seconds())
                for asset, quote in sorted(self.quotes.items())
                if self.previous_at is not None
            ),
        )
