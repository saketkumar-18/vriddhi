"""Walk-forward backtest engine.

Simulates the full agent day by day:
  1. Each base strategy emits a target position per symbol (from history only).
     Per-symbol strategies see one OHLCV; universe-aware strategies (xsmom,
     spreadrev) see the whole close matrix.
  2. The Hedge allocator blends strategies using trailing per-strategy PnL,
     and (with meta-learning) tunes its own learning rate.
  3. The risk overlay scales gross exposure (vol targeting + DD breaker).
  4. Positions are executed at the close with fees + slippage on turnover.

Everything at time t uses information up to and including t's close; PnL is
earned on t -> t+1. No lookahead: ML retrains only on data strictly before t,
and universe-aware signals at row t use closes up to t only (rolling ops).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .meta import HedgeAllocator
from .metrics import metrics
from .risk import RiskManager
from .strategies import all_strategies


def build_strategy_positions(universe: dict[str, pd.DataFrame],
                             ml_signals: dict[str, pd.Series] | None = None
                             ) -> dict[str, dict[str, pd.Series]]:
    """Normalize every strategy's output to {name: {symbol: Series}}."""
    symbols = list(universe.keys())
    strat_pos: dict[str, dict[str, pd.Series]] = {}
    for st in all_strategies():
        if st.universe_aware:
            frame = st.signals_universe(universe)
            strat_pos[st.name] = {s: frame[s] for s in symbols}
        else:
            strat_pos[st.name] = {s: st.signals(universe[s]) for s in symbols}
    if ml_signals:
        strat_pos["ml"] = {s: ml_signals[s] for s in symbols if s in ml_signals}
    return strat_pos


def run_backtest(universe: dict[str, pd.DataFrame],
                 ml_signals: dict[str, pd.Series] | None = None,
                 adaptive: bool = True,
                 use_risk: bool = True,
                 meta_learn: bool = config.META_LEARN,
                 initial_capital: float = config.INITIAL_CAPITAL) -> dict:
    """Returns dict with equity curve, per-strategy PnL, weights, metrics."""
    closes = pd.DataFrame({s: df["Close"] for s, df in universe.items()})
    rets = closes.pct_change().fillna(0.0)
    symbols = list(closes.columns)
    dates = closes.index
    n = len(dates)

    strat_pos = build_strategy_positions(universe, ml_signals)
    names = list(strat_pos.keys())
    alloc = HedgeAllocator(names, meta_learn=meta_learn)
    risk = RiskManager() if use_risk else None
    cost = (config.FEE_BPS + config.SLIPPAGE_BPS) / 1e4

    # pre-stack positions for speed: (n, n_strat, n_sym)
    pos_stack = np.zeros((n, len(names), len(symbols)))
    for k, name in enumerate(names):
        for j, s in enumerate(symbols):
            pos_stack[:, k, j] = strat_pos[name][s].values

    equity = np.full(n, np.nan)
    equity[0] = initial_capital
    port_rets: list[float] = []
    prev_pos = np.zeros(len(symbols))
    weight_rows, pnl_rows, eta_rows = [], [], []

    for t in range(1, n):
        # 1) per-strategy return earned over t-1 -> t (positions set at t-1)
        r_t = rets.iloc[t].values
        p_prev = pos_stack[t - 1]  # (n_strat, n_sym)
        strat_rets = (p_prev @ r_t) / len(symbols)

        # 2) blend with hedge weights (update with realized PnL first)
        if adaptive and t > config.WARMUP:
            w = alloc.update(strat_rets)
        else:
            w = alloc.weights()
        target = np.clip(w @ p_prev, 0.0, 1.0)  # long-only spot

        # 3) risk overlay
        mult = risk.exposure_multiplier(np.array(port_rets), equity[t - 1]) if risk else 1.0
        target = target * mult

        # 4) execute at t-1 close -> earn r_t, pay costs on turnover
        turnover = np.abs(target - prev_pos).sum() / len(symbols)
        port_ret = float(np.dot(target, r_t) / len(symbols)) - turnover * cost
        equity[t] = equity[t - 1] * (1 + port_ret)
        port_rets.append(port_ret)
        prev_pos = target
        weight_rows.append(w)
        pnl_rows.append(strat_rets)
        eta_rows.append(alloc.eta)

    eq = pd.Series(equity, index=dates).ffill()
    return {
        "equity": eq,
        "metrics": metrics(eq),
        "weights": pd.DataFrame(weight_rows, index=dates[1:n], columns=names),
        "strategy_pnl": pd.DataFrame(pnl_rows, index=dates[1:n], columns=names),
        "eta": pd.Series(eta_rows, index=dates[1:n]),
        "positions": None,
    }


def buy_and_hold(universe: dict[str, pd.DataFrame],
                 initial_capital: float = config.INITIAL_CAPITAL) -> pd.Series:
    closes = pd.DataFrame({s: df["Close"] for s, df in universe.items()})
    eq = initial_capital * (closes / closes.iloc[0]).mean(axis=1)
    return eq
