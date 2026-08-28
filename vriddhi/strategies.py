"""Rule-based signal strategies.

Each strategy maps a symbol's OHLCV history to a daily target position in
[-1, +1] (fraction of that symbol's capital sleeve; sign = long/flat/short-
as-reduce). Crypto spot is long-only in practice, so outputs are clipped to
[0, 1] by the engine — strategies may still express "get out" with 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import ema, rsi, realized_vol


class Strategy:
    name = "base"

    def signals(self, ohlcv: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class Momentum(Strategy):
    """Dual-EMA trend filter with a 30d momentum gate, vol-scaled."""
    name = "momentum"

    def __init__(self, fast: int = 20, slow: int = 60, mom: int = 30):
        self.fast, self.slow, self.mom = fast, slow, mom

    def signals(self, ohlcv: pd.DataFrame) -> pd.Series:
        c = ohlcv["Close"]
        trend = (ema(c, self.fast) > ema(c, self.slow)).astype(float)
        gate = (c.pct_change(self.mom) > 0).astype(float)
        vol = realized_vol(c, 21).replace(0, np.nan)
        scale = (0.25 / vol).clip(0.25, 1.0)  # size down in high-vol regimes
        return (trend * gate * scale).fillna(0.0)


class MeanReversion(Strategy):
    """Buy RSI dips inside an uptrend; exit when stretched."""
    name = "meanrev"

    def __init__(self, rsi_n: int = 14, lo: float = 35, hi: float = 65):
        self.rsi_n, self.lo, self.hi = rsi_n, lo, hi

    def signals(self, ohlcv: pd.DataFrame) -> pd.Series:
        c = ohlcv["Close"]
        r = rsi(c, self.rsi_n)
        uptrend = (c > ema(c, 60)).astype(float)
        pos = pd.Series(0.0, index=c.index)
        holding = 0.0
        for i in range(len(c)):
            if holding == 0 and uptrend.iloc[i] == 1 and r.iloc[i] < self.lo:
                holding = 1.0
            elif holding == 1 and r.iloc[i] > self.hi:
                holding = 0.0
            pos.iloc[i] = holding
        return pos


class Breakout(Strategy):
    """Donchian breakout: long on N-day high close, exit on M-day low."""
    name = "breakout"

    def __init__(self, entry_n: int = 30, exit_n: int = 15):
        self.entry_n, self.exit_n = entry_n, exit_n

    def signals(self, ohlcv: pd.DataFrame) -> pd.Series:
        c, h, l = ohlcv["Close"], ohlcv["High"], ohlcv["Low"]
        hi = h.rolling(self.entry_n).max().shift(1)
        lo = l.rolling(self.exit_n).min().shift(1)
        pos = pd.Series(0.0, index=c.index)
        holding = 0.0
        for i in range(len(c)):
            if holding == 0 and c.iloc[i] > hi.iloc[i]:
                holding = 1.0
            elif holding == 1 and c.iloc[i] < lo.iloc[i]:
                holding = 0.0
            pos.iloc[i] = holding
        return pos


RULE_STRATEGIES = [Momentum(), MeanReversion(), Breakout()]
