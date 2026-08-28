"""Run the full experiment: ML signals, ablation backtests, report dump.

Ablation design (v2 — decorrelated strategies + meta-learning):
  * adaptive_full      — full agent on the whole sample
  * static_full        — equal-weight ensemble, same strategies (isolates the
                         value of weight adaptation)
  * adaptive_no_meta   — adaptive but eta fixed (isolates meta-learning)
  * same three on a trailing-2-year OUT-OF-TIME holdout, so no variant can
    be credited with performance it was tuned on.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from . import config
from .backtest import buy_and_hold, run_backtest
from .data import load_universe
from .metrics import metrics as M
from .ml_strategy import WalkForwardML

HOLDOUT_DAYS = 730


def main() -> None:
    config.REPORT_DIR.mkdir(exist_ok=True)
    print("[exp] fetching universe...")
    universe = load_universe()
    symbols = list(universe.keys())
    n_bars = len(next(iter(universe.values())))
    print(f"[exp] symbols: {symbols}, bars: {n_bars}")

    print("[exp] walk-forward ML signals...")
    ml = WalkForwardML()
    ml_signals = {}
    for s in symbols:
        ml_signals[s] = ml.signals(universe[s])
        print(f"  {s}: retrains={len(ml.retrain_log)}, "
              f"accepted={sum(r['accepted'] for r in ml.retrain_log)}")
    accept_rate = (sum(r['accepted'] for r in ml.retrain_log)
                   / max(len(ml.retrain_log), 1))

    def run_suite(tag: str, uni: dict, mls: dict) -> dict:
        print(f"[exp] backtest suite: {tag}...")
        res = {
            "adaptive": run_backtest(uni, mls, adaptive=True, use_risk=True,
                                     meta_learn=True),
            "static": run_backtest(uni, mls, adaptive=False, use_risk=True),
            "adaptive_no_meta": run_backtest(uni, mls, adaptive=True,
                                             use_risk=True, meta_learn=False),
        }
        res["buy_hold"] = buy_and_hold(uni)
        return res

    full = run_suite("full sample", universe, ml_signals)

    # out-of-time holdout: trailing 2 years, signals recomputed on the slice
    cut = next(iter(universe.values())).index[-HOLDOUT_DAYS]
    uni_h = {s: df.loc[df.index >= cut] for s, df in universe.items()}
    ml_h = WalkForwardML()
    mls_h = {s: ml_h.signals(uni_h[s]) for s in symbols}
    hold = run_suite("trailing-2yr holdout", uni_h, mls_h)

    def mrow(res: dict) -> dict:
        return {
            "adaptive": res["adaptive"]["metrics"],
            "static": res["static"]["metrics"],
            "adaptive_no_meta": res["adaptive_no_meta"]["metrics"],
            "buy_hold": M(res["buy_hold"]),
        }

    rows_full, rows_hold = mrow(full), mrow(hold)
    print("\n=== FULL SAMPLE ===")
    for k, m in rows_full.items():
        print(f"{k:20s} CAGR={m['cagr']*100:7.2f}%  Sharpe={m['sharpe']:6.2f}  "
              f"MaxDD={m['max_dd']*100:7.2f}%")
    print("\n=== TRAILING-2YR HOLDOUT (out-of-time) ===")
    for k, m in rows_hold.items():
        print(f"{k:20s} CAGR={m['cagr']*100:7.2f}%  Sharpe={m['sharpe']:6.2f}  "
              f"MaxDD={m['max_dd']*100:7.2f}%")

    # adaptation gap: how much did weights actually move this time?
    wf = full["adaptive"]["weights"]
    wstats = {c: {"start": round(float(wf[c].iloc[0]), 3),
                  "end": round(float(wf[c].iloc[-1]), 3),
                  "min": round(float(wf[c].min()), 3),
                  "max": round(float(wf[c].max()), 3)} for c in wf.columns}
    leader_changes = int((wf.values.argmax(axis=1)[1:]
                          != wf.values.argmax(axis=1)[:-1]).sum())

    out = {
        "generated": str(pd.Timestamp.now("UTC")),
        "universe": symbols,
        "metrics": rows_full,
        "metrics_holdout": rows_hold,
        "ml_accept_rate": accept_rate,
        "weight_stats": wstats,
        "leader_changes": leader_changes,
        "eta": {str(i): round(float(v), 4)
                for i, v in full["adaptive"]["eta"].items()},
        "equity": {},
        "weights": {},
        "strategy_pnl_cum": {},
        "ml_retrain_log": ml.retrain_log[-20:],
    }
    for label, res in (("full", full), ("holdout", hold)):
        eq = {"agent": res["adaptive"]["equity"],
              "static": res["static"]["equity"],
              "buy_hold": res["buy_hold"]}
        out["equity"][label] = {k: {str(i): str(round(float(v), 2))
                                    for i, v in d.items()}
                                for k, d in eq.items()}
    w = full["adaptive"]["weights"]
    for i, row in w.iterrows():
        out["weights"][str(i)] = {c: round(float(v), 4) for c, v in row.items()}
    pnl = full["adaptive"]["strategy_pnl"].cumsum()
    out["strategy_pnl_cum"] = {c: {str(i): round(float(v), 4)
                                   for i, v in pnl[c].items()}
                               for c in pnl.columns}
    (config.REPORT_DIR / "report.json").write_text(json.dumps(out))
    print(f"\n[exp] weight stats: {json.dumps(wstats, indent=1)}")
    print(f"[exp] leader changes: {leader_changes}")
    print(f"[exp] report -> {config.REPORT_DIR / 'report.json'}")


if __name__ == "__main__":
    sys.exit(main())
