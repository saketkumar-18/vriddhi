"""Feature engineering: indicators used by strategies and the ML model."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def realized_vol(close: pd.Series, n: int = 21) -> pd.Series:
    return close.pct_change().rolling(n).std() * np.sqrt(365)


def features_for_symbol(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for one symbol (rows aligned to ohlcv index)."""
    c, h, l = ohlcv["Close"], ohlcv["High"], ohlcv["Low"]
    ret = c.pct_change()
    f = pd.DataFrame(index=ohlcv.index)
    for n in (3, 7, 14, 30, 90):
        f[f"ret_{n}"] = c.pct_change(n)
    f["ema_ratio_10_30"] = ema(c, 10) / ema(c, 30) - 1
    f["ema_ratio_20_60"] = ema(c, 20) / ema(c, 60) - 1
    f["rsi_14"] = rsi(c, 14) / 100 - 0.5
    f["vol_21"] = realized_vol(c, 21)
    f["vol_ratio"] = realized_vol(c, 7) / realized_vol(c, 42).replace(0, np.nan)
    f["atr_pct"] = atr(h, l, c, 14) / c
    f["dist_high_30"] = c / h.rolling(30).max() - 1
    f["dist_low_30"] = c / l.rolling(30).min() - 1
    f["bb_pos"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() * 2).replace(0, np.nan)
    f["vol_z"] = (ohlcv["Volume"] / ohlcv["Volume"].rolling(20).mean()) - 1
    f["ret_lag1"] = ret.shift(1)
    f["ret_lag2"] = ret.shift(2)
    return f.replace([np.inf, -np.inf], np.nan)
