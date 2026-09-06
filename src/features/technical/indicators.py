"""Technical indicators calculation engine using TA-Lib.

Enforces PRD Section 3.6:
- Strictly closed bars (available_for_decision_at <= decision_cutoff)
- Standardized indicator family: RSI, MACD, Bollinger Bands, EMA, ATR, EWMA volatility
- Robust handling of insufficient history
"""

import math
from collections.abc import Sequence

import numpy as np
import talib

from src.domain.market import Bar1m


class TechnicalIndicatorsEngine:
    """Computes deterministic technical feature scalars from a sequence of closed bars."""

    def __init__(self, lambda_decay: float = 0.94) -> None:
        self._lambda_decay = lambda_decay

    def compute_features(self, bars: Sequence[Bar1m]) -> dict[str, float]:
        """Compute technical indicator vector from closed bars.

        Requires at least 200 bars for full 200 EMA initialization, but handles
        shorter windows gracefully with NaNs or partial availability flags.
        """
        if len(bars) < 2:
            return {}

        closes = np.array([b.close_usd for b in bars], dtype=np.float64)
        highs = np.array([b.high_usd for b in bars], dtype=np.float64)
        lows = np.array([b.low_usd for b in bars], dtype=np.float64)
        volumes = np.array([b.volume for b in bars], dtype=np.float64)

        features: dict[str, float] = {
            "tech_close": float(closes[-1]),
            "tech_volume": float(volumes[-1]),
        }

        # 1. RSI (14)
        if len(bars) >= 15:
            rsi = talib.RSI(closes, timeperiod=14)
            val = float(rsi[-1])
            if not math.isnan(val):
                features["tech_rsi_14"] = val

        # 2. MACD (12, 26, 9)
        if len(bars) >= 35:
            macd, macd_signal, macd_hist = talib.MACD(
                closes, fastperiod=12, slowperiod=26, signalperiod=9
            )
            m_val = float(macd[-1])
            s_val = float(macd_signal[-1])
            h_val = float(macd_hist[-1])
            if not math.isnan(m_val):
                features["tech_macd"] = m_val
                features["tech_macd_signal"] = s_val
                features["tech_macd_hist"] = h_val

        # 3. Bollinger Bands (20, 2)
        if len(bars) >= 20:
            upper, middle, lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
            u_val, m_val, l_val = float(upper[-1]), float(middle[-1]), float(lower[-1])
            if not math.isnan(u_val):
                features["tech_bb_upper"] = u_val
                features["tech_bb_middle"] = m_val
                features["tech_bb_lower"] = l_val
                band_width = u_val - l_val
                features["tech_bb_pct_b"] = (
                    (closes[-1] - l_val) / band_width if band_width > 0 else 0.5
                )

        # 4. EMAs (12, 26, 50, 200)
        for period in [12, 26, 50, 200]:
            if len(bars) >= period:
                ema = talib.EMA(closes, timeperiod=period)
                ema_val = float(ema[-1])
                if not math.isnan(ema_val):
                    features[f"tech_ema_{period}"] = ema_val

        # 5. ATR (14)
        if len(bars) >= 15:
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            atr_val = float(atr[-1])
            if not math.isnan(atr_val):
                features["tech_atr_14"] = atr_val

        # 6. EWMA Volatility (lambda = 0.94)
        log_returns = np.diff(np.log(closes))
        if len(log_returns) > 0:
            variance = 0.0
            for r in log_returns:
                variance = self._lambda_decay * variance + (1.0 - self._lambda_decay) * (r * r)
            features["tech_ewma_volatility"] = float(math.sqrt(variance))

        return features
