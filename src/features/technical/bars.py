"""Canonical 1-minute bar aggregator from tick trades.

Enforces:
- Exact minute boundary alignment
- Closed bar semantics with trade count and VWAP calculation
"""

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta

from src.domain.identity import MarketId
from src.domain.market import Bar1m, Trade
from src.domain.time import ensure_utc


class BarAggregator:
    """Aggregates trades into discrete 1-minute OHLCV bars."""

    @staticmethod
    def _minute_floor(dt: datetime) -> datetime:
        """Truncate datetime to the nearest preceding minute boundary."""
        utc_dt = ensure_utc(dt)
        return utc_dt.replace(second=0, microsecond=0)

    def aggregate_trades(
        self,
        market_id: MarketId,
        trades: Sequence[Trade],
    ) -> list[Bar1m]:
        """Aggregate trades into sorted 1-minute bars."""
        if not trades:
            return []

        # Group trades by minute boundary
        buckets: dict[datetime, list[Trade]] = defaultdict(list)
        for t in trades:
            if t.market_id == market_id:
                m_start = self._minute_floor(t.event_time)
                buckets[m_start].append(t)

        bars: list[Bar1m] = []
        for m_start in sorted(buckets.keys()):
            bucket_trades = buckets[m_start]
            bucket_trades.sort(key=lambda t: t.event_time)

            open_usd = bucket_trades[0].price_usd
            close_usd = bucket_trades[-1].price_usd
            high_usd = max(t.price_usd for t in bucket_trades)
            low_usd = min(t.price_usd for t in bucket_trades)

            total_volume = sum(t.size for t in bucket_trades)
            dollar_volume = sum(t.price_usd * t.size for t in bucket_trades)
            vwap = dollar_volume / total_volume if total_volume > 0 else close_usd

            bar_close_at = m_start + timedelta(minutes=1)
            bars.append(
                Bar1m(
                    market_id=market_id,
                    open_usd=open_usd,
                    high_usd=high_usd,
                    low_usd=low_usd,
                    close_usd=close_usd,
                    volume=total_volume,
                    bar_start_at=m_start,
                    bar_close_at=bar_close_at,
                    trade_count=len(bucket_trades),
                    vwap_usd=vwap,
                    is_closed=True,
                )
            )

        return bars
