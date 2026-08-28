"""Live paper-trading tick.

Each run: fetch latest daily bars, regenerate every strategy's signals via the
SAME registry the backtest uses, update the allocator with yesterday's
realized per-strategy PnL, apply risk, and persist state (equity, positions,
weights, trade log) to state/live_state.json.

This is the same decision pipeline as the backtest, stepped forward one day
at a time in the real world. No real orders are ever placed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .backtest import build_strategy_positions
from .data import load_universe
from .meta import HedgeAllocator
from .ml_strategy import WalkForwardML
from .risk import RiskManager

STATE_PATH = config.STATE_DIR / "live_state.json"


def _fresh_state() -> dict:
    return {
        "started": datetime.now(timezone.utc).isoformat(),
        "equity": config.INITIAL_CAPITAL,
        "peak": config.INITIAL_CAPITAL,
        "cash": config.INITIAL_CAPITAL,
        "positions": {},          # symbol -> {"qty": float, "entry": float}
        "last_date": None,
        "log_weights": [],        # per-strategy Hedge log-weights (cumulative)
        "smoothed": [],           # current smoothed weights (windowed mode)
        "reward_buf": [],         # trailing per-strategy returns for Hedge
        "eta": config.HEDGE_ETA,  # allocator's current (meta-learned) eta
        "returns": [],            # daily portfolio returns
        "trades": [],
        "ticks": 0,
    }


def _load_state(n_strat: int) -> dict:
    if STATE_PATH.exists():
        st = json.loads(STATE_PATH.read_text())
        # strategy set changed (or corrupt) -> reset rather than misalign
        if len(st.get("log_weights", [])) != n_strat:
            return _fresh_state()
        return st
    return _fresh_state()


def _save_state(st: dict) -> None:
    config.STATE_DIR.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, indent=2, default=str))


def tick() -> dict:
    """Advance the paper portfolio by one day. Returns a summary dict."""
    universe = load_universe()
    symbols = list(universe.keys())
    closes = pd.DataFrame({s: df["Close"] for s, df in universe.items()})
    today = str(closes.index[-1].date())

    # --- build every strategy's signals (walk-forward ML included) ---
    ml = WalkForwardML()
    ml_signals = {s: ml.signals(universe[s]) for s in symbols}
    strat_pos = build_strategy_positions(universe, ml_signals)
    names = list(strat_pos.keys())
    n_strat = len(names)

    st = _load_state(n_strat)
    if st["last_date"] == today:
        return {"status": "already_ran", "date": today, "equity": st["equity"]}

    alloc = HedgeAllocator(names)
    alloc.log_weights = np.array(st["log_weights"], dtype=float)
    if st.get("smoothed"):
        alloc.smoothed = np.array(st["smoothed"], dtype=float)
        alloc.smoothed /= alloc.smoothed.sum()
    else:
        alloc.smoothed = np.exp(alloc.log_weights)
        alloc.smoothed /= alloc.smoothed.sum()
    alloc._buf = [np.array(r, dtype=float) for r in st.get("reward_buf", [])]
    alloc.eta = st.get("eta", config.HEDGE_ETA)

    # --- mark-to-market existing positions at today's close ---
    equity = st["cash"]
    for sym, pos in list(st["positions"].items()):
        equity += pos["qty"] * closes[sym].iloc[-1]
    prev_equity = equity

    # --- realize yesterday's per-strategy PnL and update weights ---
    if st["last_date"] is not None and len(closes) > 1:
        r = (closes.iloc[-1] / closes.iloc[-2] - 1).values
        p_prev = np.array([[strat_pos[nm][s].iloc[-2] for s in symbols]
                           for nm in names])
        strat_rets = (p_prev @ r) / len(symbols)
        w = alloc.update(strat_rets)
        st["log_weights"] = alloc.log_weights.tolist()
        st["smoothed"] = alloc.smoothed.tolist()
        st["reward_buf"] = [x.tolist() for x in alloc._buf]
        st["eta"] = float(alloc.eta)
    else:
        w = alloc.weights()

    # --- fresh targets for tomorrow (positions held from today's close) ---
    p_today = np.array([[strat_pos[nm][s].iloc[-1] for s in symbols]
                        for nm in names])
    targets = np.clip(w @ p_today, 0.0, 1.0)

    risk = RiskManager()
    risk.peak = st["peak"]
    mult = risk.exposure_multiplier(np.array(st["returns"]), equity)
    targets = targets * mult

    # --- execute rebalance at today's close (paper) ---
    cost = (config.FEE_BPS + config.SLIPPAGE_BPS) / 1e4
    new_positions = {}
    turnover = 0.0
    for j, s in enumerate(symbols):
        px = float(closes[s].iloc[-1])
        want_notional = equity * targets[j] / len(symbols)
        have = st["positions"].get(s, {}).get("qty", 0.0) * px
        turnover += abs(want_notional - have) / max(equity, 1e-9)
        if want_notional > 1.0:
            new_positions[s] = {"qty": want_notional / px, "entry": px}
    fees = turnover * cost * equity
    st["cash"] = equity - sum(p["qty"] * p["entry"]
                              for p in new_positions.values()) - fees
    st["positions"] = new_positions
    st["peak"] = max(st["peak"], equity)
    st["last_date"] = today
    st["equity"] = equity - fees
    st["returns"].append(float(st["equity"] / max(prev_equity, 1e-9) - 1))
    st["ticks"] += 1
    st["trades"].append({"date": today,
                         "weights": {n: round(float(x), 4)
                                     for n, x in zip(names, w)},
                         "targets": {s: round(float(t), 4)
                                     for s, t in zip(symbols, targets)},
                         "eta": round(float(alloc.eta), 4),
                         "equity": round(st["equity"], 2)})
    _save_state(st)
    return {"status": "ok", "date": today, "equity": round(st["equity"], 2),
            "weights": {n: round(float(x), 4) for n, x in zip(names, w)},
            "targets": {s: round(float(t), 4) for s, t in zip(symbols, targets)},
            "eta": round(float(alloc.eta), 4),
            "risk_mult": round(mult, 3)}
