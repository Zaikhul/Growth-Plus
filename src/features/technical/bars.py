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
    """Aggregates trades into discrete 1-minute OHLCV bars.

    Retains open minute buckets across batches and only closes a bar
    once watermark (event_time >= bar_close_at + 2s) is reached.
    """

    def __init__(self, watermark_delay_seconds: float = 2.0) -> None:
        self._watermark_delay = timedelta(seconds=watermark_delay_seconds)
        self._open_buckets: dict[tuple[str, datetime], list[Trade]] = defaultdict(list)

    @staticmethod
    def _minute_floor(dt: datetime) -> datetime:
        """Truncate datetime to the nearest preceding minute boundary."""
        utc_dt = ensure_utc(dt)
        return utc_dt.replace(second=0, microsecond=0)

    def aggregate_trades(
        self,
        market_id: MarketId,
        trades: Sequence[Trade],
        watermark_time: datetime | None = None,
        force_close: bool = False,
    ) -> list[Bar1m]:
        """Aggregate trades into sorted 1-minute bars using watermark windowing."""
        m_id_str = str(market_id.value if hasattr(market_id, "value") else market_id)
        latest_trade_time: datetime | None = None

        # 1. Accumulate trades into open minute buckets
        for t in trades:
            t_market_str = str(t.market_id.value if hasattr(t.market_id, "value") else t.market_id)
            if t_market_str == m_id_str:
                m_start = self._minute_floor(t.event_time)
                self._open_buckets[(m_id_str, m_start)].append(t)
                t_event = ensure_utc(t.event_time)
                if latest_trade_time is None or t_event > latest_trade_time:
                    latest_trade_time = t_event

        effective_watermark = (
            ensure_utc(watermark_time) if watermark_time is not None else latest_trade_time
        )

        # 2. Identify and close eligible buckets
        bars: list[Bar1m] = []
        matching_keys = [k for k in self._open_buckets.keys() if k[0] == m_id_str]
        for key in sorted(matching_keys, key=lambda k: k[1]):
            _, m_start = key
            bar_close_at = m_start + timedelta(minutes=1)
            watermark_threshold = bar_close_at + self._watermark_delay

            if force_close or (
                effective_watermark is not None and effective_watermark >= watermark_threshold
            ):
                bucket_trades = self._open_buckets.pop(key)
                if not bucket_trades:
                    continue
                bucket_trades.sort(key=lambda t: t.event_time)

                open_usd = bucket_trades[0].price_usd
                close_usd = bucket_trades[-1].price_usd
                high_usd = max(t.price_usd for t in bucket_trades)
                low_usd = min(t.price_usd for t in bucket_trades)

                total_volume = sum(t.size for t in bucket_trades)
                dollar_volume = sum(t.price_usd * t.size for t in bucket_trades)
                vwap = dollar_volume / total_volume if total_volume > 0 else close_usd

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
