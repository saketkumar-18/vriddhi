"""Live paper-trading tick.

Each run: fetch latest daily bars, regenerate signals, update the allocator
with yesterday's realized per-strategy PnL, apply risk, and persist state
(equity, positions, weights, trade log) to state/live_state.json.

This is the same decision pipeline as the backtest, stepped forward one day
at a time in the real world. No real orders are ever placed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .data import load_universe
from .meta import HedgeAllocator
from .ml_strategy import WalkForwardML
from .risk import RiskManager
from .strategies import RULE_STRATEGIES

STATE_PATH = config.STATE_DIR / "live_state.json"


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "started": datetime.now(timezone.utc).isoformat(),
        "equity": config.INITIAL_CAPITAL,
        "peak": config.INITIAL_CAPITAL,
        "cash": config.INITIAL_CAPITAL,
        "positions": {},          # symbol -> {"qty": float, "entry": float}
        "last_date": None,
        "log_weights": [0.0] * 4, # momentum, meanrev, breakout, ml
        "reward_buf": [],         # trailing per-strategy returns for Hedge
        "returns": [],            # daily portfolio returns
        "trades": [],
        "ticks": 0,
    }


def _save_state(st: dict) -> None:
    config.STATE_DIR.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, indent=2, default=str))


def tick() -> dict:
    """Advance the paper portfolio by one day. Returns a summary dict."""
    st = _load_state()
    universe = load_universe()
    symbols = list(universe.keys())
    closes = pd.DataFrame({s: df["Close"] for s, df in universe.items()})
    today = str(closes.index[-1].date())

    if st["last_date"] == today:
        return {"status": "already_ran", "date": today,
                "equity": st["equity"]}

    names = ["momentum", "meanrev", "breakout", "ml"]
    alloc = HedgeAllocator(names)
    alloc.log_weights = np.array(st["log_weights"], dtype=float)
    alloc.smoothed = np.exp(alloc.log_weights)
    alloc.smoothed /= alloc.smoothed.sum()
    alloc._buf = [np.array(r, dtype=float) for r in st.get("reward_buf", [])]

    # --- mark-to-market existing positions at today's close
    equity = st["cash"]
    for sym, pos in list(st["positions"].items()):
        equity += pos["qty"] * closes[sym].iloc[-1]
    prev_equity = equity

    # --- realize yesterday's per-strategy PnL and update weights
    if st["last_date"] is not None and len(closes) > 1:
        r = closes.iloc[-1] / closes.iloc[-2] - 1
        strat_rets = []
        for name in names:
            if name == "ml":
                ml = WalkForwardML()
                sig = {s: ml.signals(universe[s].iloc[:-1]) for s in symbols}
                p = np.array([sig[s].iloc[-1] for s in symbols])
            else:
                stg = next(x for x in RULE_STRATEGIES if x.name == name)
                p = np.array([stg.signals(universe[s].iloc[:-1]).iloc[-1]
                              for s in symbols])
            strat_rets.append(float(np.dot(p, r.values) / len(symbols)))
        w = alloc.update(np.array(strat_rets))
        st["log_weights"] = alloc.log_weights.tolist()
        st["reward_buf"] = [r.tolist() for r in alloc._buf]
    else:
        w = alloc.weights()

    # --- fresh targets for tomorrow
    targets = np.zeros(len(symbols))
    ml = WalkForwardML()
    for k, name in enumerate(names):
        for j, s in enumerate(symbols):
            if name == "ml":
                sig = ml.signals(universe[s])
            else:
                stg = next(x for x in RULE_STRATEGIES if x.name == name)
                sig = stg.signals(universe[s])
            targets[j] += w[k] * float(sig.iloc[-1])
    targets = np.clip(targets, 0.0, 1.0)

    risk = RiskManager()
    risk.peak = st["peak"]
    mult = risk.exposure_multiplier(np.array(st["returns"]), equity)
    targets = targets * mult

    # --- execute rebalance at today's close (paper)
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
    st["cash"] = equity - sum(p["qty"] * p["entry"] for p in new_positions.values()) - fees
    st["positions"] = new_positions
    st["peak"] = max(st["peak"], equity)
    st["last_date"] = today
    st["equity"] = equity - fees
    st["returns"].append(float(st["equity"] / max(prev_equity, 1e-9) - 1))
    st["ticks"] += 1
    st["trades"].append({"date": today, "weights": {n: round(float(x), 4)
                        for n, x in zip(names, w)},
                         "targets": {s: round(float(t), 4)
                                     for s, t in zip(symbols, targets)},
                         "equity": round(st["equity"], 2)})
    _save_state(st)
    return {"status": "ok", "date": today, "equity": round(st["equity"], 2),
            "weights": {n: round(float(x), 4) for n, x in zip(names, w)},
            "targets": {s: round(float(t), 4) for s, t in zip(symbols, targets)},
            "risk_mult": round(mult, 3)}
