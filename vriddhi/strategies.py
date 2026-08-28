"""Signal strategies.

Two families:
  * Per-symbol strategies take one symbol's OHLCV and return a position in
    [0,1] per bar (momentum, mean-reversion, breakout).
  * Universe-aware strategies see the whole close matrix and return a
    (dates x symbols) position frame. These are what make the ensemble
    DECORRELATED: cross-sectional momentum longs the relative winners while
    spread reversion longs the relative laggards — they disagree by design,
    which is exactly what the Hedge allocator needs to have something to
    adapt between.

Crypto spot is long-only, so every position is clipped to [0,1] downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import ema, rsi, realized_vol


class Strategy:
    name = "base"
    universe_aware = False

    def signals(self, ohlcv: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Per-symbol rule strategies
# ---------------------------------------------------------------------------
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
        scale = (0.25 / vol).clip(0.25, 1.0)
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


# ---------------------------------------------------------------------------
# Universe-aware (decorrelating) strategies
# ---------------------------------------------------------------------------
class CrossSectionalMomentum(Strategy):
    """Long the top-half relative performers over `lookback` days.

    Always half-invested, always rotated into the strongest assets. This is
    RELATIVE value: it can be long ETH and flat BTC in the same bar, which no
    per-symbol absolute strategy can express.
    """
    name = "xsmom"
    universe_aware = True

    def __init__(self, lookback: int = 30, n_long: int | None = None):
        self.lookback, self.n_long = lookback, n_long

    def signals_universe(self, universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
        closes = pd.DataFrame({s: df["Close"] for s, df in universe.items()})
        n_long = self.n_long or max(1, closes.shape[1] // 2)
        mom = closes.pct_change(self.lookback)
        ranks = mom.rank(axis=1, ascending=False, method="first")
        return (ranks <= n_long).astype(float).fillna(0.0)


class SpreadReversion(Strategy):
    """Long the bottom-half relative laggards (fade relative strength).

    The cross-sectional mean-reversion twin of CrossSectionalMomentum: where
    xsmom chases the leaders, this buys the losers expecting the spread to
    close. Negatively correlated with xsmom by construction — the two give
    the allocator a genuine choice to adapt over.
    """
    name = "spreadrev"
    universe_aware = True

    def __init__(self, lookback: int = 14, n_long: int | None = None):
        self.lookback, self.n_long = lookback, n_long

    def signals_universe(self, universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
        closes = pd.DataFrame({s: df["Close"] for s, df in universe.items()})
        n_long = self.n_long or max(1, closes.shape[1] // 2)
        rel = closes.pct_change(self.lookback)
        mu, sd = rel.mean(axis=1), rel.std(axis=1)
        z = rel.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
        ranks = z.rank(axis=1, ascending=True, method="first")  # 1 = most lagging
        return (ranks <= n_long).astype(float).fillna(0.0)


class Defensive(Strategy):
    """Always flat (hold cash / stablecoin). Returns ~0 with ~0 vol.

    This is the strategy that makes adaptation MEANINGFUL in a long-only,
    correlated market: it gives the allocator a genuine risk-off choice. When
    every directional strategy is losing, Defensive wins the trailing-PnL
    contest and the allocator rotates to cash — i.e. self-improvement learns
    "be out of the market" instead of being forced to pick the least-bad long.
    """
    name = "defensive"
    universe_aware = True

    def signals_universe(self, universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
        closes = pd.DataFrame({s: df["Close"] for s, df in universe.items()})
        return pd.DataFrame(0.0, index=closes.index, columns=closes.columns)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
RULE_STRATEGIES = [Momentum(), MeanReversion(), Breakout()]
UNIVERSE_STRATEGIES = [CrossSectionalMomentum(), SpreadReversion(), Defensive()]


def all_strategies() -> list[Strategy]:
    """Canonical order; ML is appended by the caller when present."""
    return RULE_STRATEGIES + UNIVERSE_STRATEGIES
