"""Integration test: full backtest on synthetic data, no network."""
import numpy as np
import pandas as pd

from vriddhi.backtest import buy_and_hold, run_backtest
from vriddhi.ml_strategy import WalkForwardML


def make_universe(n=700, seed=3):
    rng = np.random.default_rng(seed)
    uni = {}
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    for i, sym in enumerate(["AAA", "BBB"]):
        drift = 0.001 if i == 0 else 0.0002
        rets = rng.normal(drift, 0.03, n)
        close = 100 * np.exp(np.cumsum(rets))
        uni[sym] = pd.DataFrame({
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": rng.uniform(1e3, 1e4, n)}, index=idx)
    return uni


def test_backtest_runs_and_equity_positive():
    uni = make_universe()
    res = run_backtest(uni, ml_signals=None, adaptive=True, use_risk=True)
    eq = res["equity"].dropna()
    assert len(eq) > 600
    assert (eq > 0).all()
    assert set(res["metrics"]) >= {"cagr", "sharpe", "max_dd"}


def test_backtest_weights_sum_to_one():
    uni = make_universe()
    res = run_backtest(uni, ml_signals=None, adaptive=True)
    w = res["weights"]
    assert len(w) > 0
    assert np.allclose(w.sum(axis=1), 1.0)


def test_static_vs_adaptive_differ():
    uni = make_universe()
    a = run_backtest(uni, adaptive=True)["equity"]
    s = run_backtest(uni, adaptive=False)["equity"]
    assert not np.allclose(a.values, s.values)


def test_buy_and_hold_matches_prices():
    uni = make_universe()
    bh = buy_and_hold(uni, initial_capital=10_000)
    assert abs(bh.iloc[0] - 10_000) < 1e-6
    assert (bh > 0).all()


def test_ml_walk_forward_no_lookahead():
    uni = make_universe(n=600)
    ml = WalkForwardML(retrain_every=50, train_window=300, valid_window=30)
    sig = ml.signals(uni["AAA"])
    assert len(sig) == 600
    assert sig.between(-0.01, 1.01).all()
    # early bars (before enough history) must be flat
    assert sig.iloc[:330].abs().sum() == 0
    assert len(ml.retrain_log) >= 2
