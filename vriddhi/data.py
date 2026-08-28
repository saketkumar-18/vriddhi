"""Data pipeline: fetch OHLCV from yfinance with a local parquet/csv cache."""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from . import config


def fetch_ohlcv(symbol: str, start: str = config.START,
                end: str | None = None) -> pd.DataFrame:
    """Daily OHLCV for one symbol, adjusted close, indexed by date."""
    df = yf.download(symbol, start=start, end=end, interval="1d",
                     progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"no data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna(subset=["Close"])


def load_universe(symbols: list[str] | None = None,
                  start: str = config.START) -> dict[str, pd.DataFrame]:
    """Fetch every symbol; returns {symbol: ohlcv df} on a common date index."""
    symbols = symbols or config.UNIVERSE
    frames = {}
    for s in symbols:
        try:
            frames[s] = fetch_ohlcv(s, start=start)
        except Exception as e:  # keep going on single-symbol failure
            print(f"[data] WARN {s}: {e}")
    if not frames:
        raise RuntimeError("no symbols fetched")
    common = None
    for df in frames.values():
        common = df.index if common is None else common.intersection(df.index)
    return {s: df.loc[common] for s, df in frames.items()}


def close_matrix(universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Close prices as a (dates x symbols) matrix."""
    return pd.DataFrame({s: df["Close"] for s, df in universe.items()})
