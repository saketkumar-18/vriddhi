"""Unit tests for Vriddhi core components (no network needed)."""
import numpy as np
import pandas as pd
import pytest

from vriddhi.indicators import atr, ema, features_for_symbol, realized_vol, rsi
from vriddhi.meta import HedgeAllocator
from vriddhi.metrics import metrics
from vriddhi.risk import RiskManager
from vriddhi.strategies import (RULE_STRATEGIES, UNIVERSE_STRATEGIES, Breakout,
                                CrossSectionalMomentum, MeanReversion, Momentum,
                                SpreadReversion, all_strategies)


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


def make_universe(n=400, seed=3, k=6):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    uni = {}
    for i in range(k):
        drift = rng.normal(0.0005, 0.001)
        rets = rng.normal(drift, 0.03, n)
        close = 100 * np.exp(np.cumsum(rets))
        uni[f"S{i}"] = pd.DataFrame({"Open": close, "High": close * 1.01,
                                     "Low": close * 0.99, "Close": close,
                                     "Volume": rng.uniform(1e3, 1e4, n)},
                                    index=idx)
    return uni


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


# ---------- per-symbol strategies ----------
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
    assert sig.iloc[-60:].mean() < 0.3


def test_breakout_enters_on_high():
    df = make_ohlcv(200, drift=0.004)
    sig = Breakout(entry_n=30, exit_n=15).signals(df)
    assert sig.sum() > 0


# ---------- universe-aware strategies ----------
def test_xsmom_half_invested_and_no_lookahead():
    uni = make_universe()
    sig = CrossSectionalMomentum(lookback=30).signals_universe(uni)
    assert sig.shape == (400, 6)
    # after warmup, exactly half the universe is longed each bar
    row_sums = sig.iloc[40:].sum(axis=1)
    assert (row_sums == 3).all()
    # no lookahead: signal at t depends only on closes up to t
    sig_trunc = CrossSectionalMomentum(lookback=30).signals_universe(
        {s: df.iloc[:200] for s, df in uni.items()})
    assert np.allclose(sig_trunc.iloc[-1].values, sig.iloc[199].values)


def test_spreadrev_picks_laggards():
    uni = make_universe(n=200)
    # force one extreme laggard (-13% over the last 14d, unambiguous)
    uni["S0"]["Close"] = uni["S0"]["Close"] * np.linspace(1.0, 0.2, 200)
    sig = SpreadReversion(lookback=14).signals_universe(uni)
    assert sig["S0"].iloc[-1] == 1.0  # the engineered loser is longed


def test_xsmom_spreadrev_disagree():
    uni = make_universe(n=300, seed=11)
    a = CrossSectionalMomentum(30).signals_universe(uni)
    b = SpreadReversion(14).signals_universe(uni)
    # they should rarely pick the same assets (decorrelation by design)
    overlap = (a * b).sum(axis=1).iloc[60:]
    assert overlap.mean() < 1.5  # <1.5 of 3 shared on average


def test_registry_complete():
    names = [s.name for s in all_strategies()]
    assert names == ["momentum", "meanrev", "breakout", "xsmom", "spreadrev",
                     "defensive"]
    assert sum(s.universe_aware for s in all_strategies()) == 3


def test_defensive_always_flat():
    uni = make_universe()
    from vriddhi.strategies import Defensive
    sig = Defensive().signals_universe(uni)
    assert (sig == 0.0).all().all()
    assert sig.shape == (400, 6)


# ---------- hedge allocator ----------
def test_hedge_concentrates_on_winner():
    # cumulative mode: unbounded memory, concentration grows without limit
    alloc = HedgeAllocator(["a", "b"], eta=1.0, smooth=0.0, min_weight=0.0,
                           meta_learn=False, windowed=False)
    for _ in range(100):
        alloc.update(np.array([0.02, -0.01]))
    assert alloc.weights()[0] > 0.9


def test_windowed_hedge_concentrates_and_forgets():
    # windowed mode: concentration bounded by window, but real
    alloc = HedgeAllocator(["a", "b"], eta=1.0, smooth=0.0, min_weight=0.0,
                           meta_learn=False, windowed=True, lookback=63)
    for _ in range(100):
        alloc.update(np.array([0.02, -0.01]))
    assert alloc.weights()[0] > 0.8
    # now the regime FLIPS: windowed Hedge must forget the old winner
    for _ in range(100):
        alloc.update(np.array([-0.01, 0.02]))
    assert alloc.weights()[1] > 0.8  # rotated to the new winner


def test_hedge_weights_sum_to_one():
    alloc = HedgeAllocator(["a", "b", "c"], meta_learn=False)
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = alloc.update(rng.normal(0, 0.02, 3))
        assert abs(w.sum() - 1) < 1e-9
        assert (w >= 0.05 - 1e-9).all()


def test_hedge_min_weight_floor():
    alloc = HedgeAllocator(["a", "b"], eta=1.0, smooth=0.0, min_weight=0.1,
                           meta_learn=False)
    for _ in range(30):
        w = alloc.update(np.array([-0.3, 0.3]))
    assert w[0] >= 0.1 - 1e-9


def test_meta_learning_eta_moves():
    """With a persistent winner, eta should rise toward eta_max."""
    alloc = HedgeAllocator(["a", "b"], meta_learn=True, eta_min=0.05,
                           eta_max=0.6, rank_window=20)
    rng = np.random.default_rng(5)
    for _ in range(120):
        # 'a' persistently beats 'b' with small noise -> stable ranking
        alloc.update(np.array([0.02, -0.01]) + rng.normal(0, 0.002, 2))
    assert alloc.eta > 0.3  # learned to chase a stable regime


def test_meta_learning_eta_stays_low_on_noise():
    """With randomly flipping rankings, eta should stay near eta_min."""
    alloc = HedgeAllocator(["a", "b", "c"], meta_learn=True, eta_min=0.05,
                           eta_max=0.6, rank_window=20)
    rng = np.random.default_rng(7)
    for _ in range(120):
        alloc.update(rng.normal(0, 0.05, 3))  # pure noise, no persistence
    assert alloc.eta < 0.35


# ---------- risk ----------
def test_vol_targeting_scales_down():
    rm = RiskManager(vol_target=0.25)
    rets = np.random.default_rng(2).normal(0, 0.05, 30)
    m = rm.exposure_multiplier(rets, 10_000)
    assert 0 < m < 1


def test_drawdown_breaker_trips_and_recovers():
    rm = RiskManager(dd_soft=0.10, dd_hard=0.20, dd_recovery=0.5)
    assert rm.exposure_multiplier(np.array([]), 100.0) == 1.0
    assert rm.exposure_multiplier(np.array([]), 75.0) == 0.0
    assert rm.exposure_multiplier(np.array([]), 88.0) > 0.0


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
