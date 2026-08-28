"""Unit tests for Vriddhi core components (no network needed)."""
import numpy as np
import pandas as pd
import pytest

from vriddhi.indicators import atr, ema, features_for_symbol, realized_vol, rsi
from vriddhi.meta import HedgeAllocator
from vriddhi.metrics import metrics
from vriddhi.risk import RiskManager
from vriddhi.strategies import RULE_STRATEGIES, Breakout, MeanReversion, Momentum


def make_ohlcv(n=400, seed=7, drift=0.0008):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.03, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    vol = rng.uniform(1e3, 1e4, n)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({"Open": close, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=idx)


# ---------- indicators ----------
def test_rsi_bounds():
    r = rsi(make_ohlcv()["Close"], 14)
    assert r.dropna().between(0, 100).all()


def test_ema_converges_to_constant():
    s = pd.Series(np.full(100, 5.0))
    assert abs(ema(s, 10).iloc[-1] - 5.0) < 1e-9


def test_realized_vol_positive():
    v = realized_vol(make_ohlcv()["Close"], 21).dropna()
    assert (v > 0).all()


def test_atr_positive():
    df = make_ohlcv()
    a = atr(df["High"], df["Low"], df["Close"], 14).dropna()
    assert (a > 0).all()


def test_features_no_inf():
    f = features_for_symbol(make_ohlcv())
    assert np.isfinite(f.values[~f.isna().values]).all()
    assert f.shape[1] >= 10


# ---------- strategies ----------
@pytest.mark.parametrize("stg", RULE_STRATEGIES)
def test_strategy_signal_range(stg):
    sig = stg.signals(make_ohlcv())
    assert len(sig) == 400
    assert sig.between(-0.01, 1.01).all()


def test_momentum_flat_in_downtrend():
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(-0.004, 0.02, 300)))
    idx = pd.date_range("2023-01-01", periods=300, freq="D")
    df = pd.DataFrame({"Open": close, "High": close * 1.01,
                       "Low": close * 0.99, "Close": close,
                       "Volume": np.full(300, 1e3)}, index=idx)
    sig = Momentum().signals(df)
    assert sig.iloc[-60:].mean() < 0.3  # mostly out of a clear downtrend


def test_breakout_enters_on_high():
    df = make_ohlcv(200, drift=0.004)
    sig = Breakout(entry_n=30, exit_n=15).signals(df)
    assert sig.sum() > 0  # an uptrending series must trigger entries


# ---------- hedge allocator ----------
def test_hedge_concentrates_on_winner():
    alloc = HedgeAllocator(["a", "b"], eta=1.0, smooth=0.0, min_weight=0.0)
    for _ in range(100):
        alloc.update(np.array([0.02, -0.01]))
    w = alloc.weights()
    assert w[0] > 0.9


def test_hedge_weights_sum_to_one():
    alloc = HedgeAllocator(["a", "b", "c"])
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = alloc.update(rng.normal(0, 0.02, 3))
        assert abs(w.sum() - 1) < 1e-9
        assert (w >= 0.05 - 1e-9).all()


def test_hedge_min_weight_floor():
    alloc = HedgeAllocator(["a", "b"], eta=1.0, smooth=0.0, min_weight=0.1)
    for _ in range(30):
        w = alloc.update(np.array([-0.3, 0.3]))
    assert w[0] >= 0.1 - 1e-9


# ---------- risk ----------
def test_vol_targeting_scales_down():
    rm = RiskManager(vol_target=0.25)
    rets = np.random.default_rng(2).normal(0, 0.05, 30)  # ~95% ann vol
    m = rm.exposure_multiplier(rets, 10_000)
    assert 0 < m < 1


def test_drawdown_breaker_trips_and_recovers():
    rm = RiskManager(dd_soft=0.10, dd_hard=0.20, dd_recovery=0.5)
    eq = 100.0
    m = rm.exposure_multiplier(np.array([]), eq)
    assert m == 1.0
    m = rm.exposure_multiplier(np.array([]), 75.0)   # -25% DD -> trip
    assert m == 0.0
    m = rm.exposure_multiplier(np.array([]), 88.0)   # recovers >50% of trough
    assert m > 0.0


# ---------- metrics ----------
def test_metrics_sane():
    idx = pd.date_range("2023-01-01", periods=365, freq="D")
    eq = pd.Series(np.linspace(100, 150, 365), index=idx)
    m = metrics(eq)
    assert m["cagr"] > 0.4 and m["sharpe"] > 1 and m["max_dd"] == 0.0


def test_metrics_drawdown():
    idx = pd.date_range("2023-01-01", periods=100, freq="D")
    eq = pd.Series([100, 120, 60, 90] + [90] * 96, index=idx)
    m = metrics(eq)
    assert abs(m["max_dd"] - (-0.5)) < 1e-9
