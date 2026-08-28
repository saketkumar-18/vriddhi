"""Run the full experiment: ML signals, ablation backtests, report dump."""
from __future__ import annotations

import json
import sys

import pandas as pd

from . import config
from .backtest import buy_and_hold, run_backtest
from .data import load_universe
from .ml_strategy import WalkForwardML


def main() -> None:
    config.REPORT_DIR.mkdir(exist_ok=True)
    print("[exp] fetching universe...")
    universe = load_universe()
    symbols = list(universe.keys())
    print(f"[exp] symbols: {symbols}, bars: {len(next(iter(universe.values())))}")

    print("[exp] walk-forward ML signals (this retrains ~90 times)...")
    ml = WalkForwardML()
    ml_signals = {}
    for s in symbols:
        ml_signals[s] = ml.signals(universe[s])
        print(f"  {s}: retrains={len(ml.retrain_log)}, "
              f"accepted={sum(r['accepted'] for r in ml.retrain_log)}")
    accept_rate = (sum(r['accepted'] for r in ml.retrain_log)
                   / max(len(ml.retrain_log), 1))

    print("[exp] backtest: adaptive ensemble (full agent)...")
    full = run_backtest(universe, ml_signals, adaptive=True, use_risk=True)
    print("[exp] backtest: static equal-weight ensemble (no self-improvement)...")
    static = run_backtest(universe, ml_signals, adaptive=False, use_risk=True)
    print("[exp] backtest: adaptive, no risk overlay...")
    norisk = run_backtest(universe, ml_signals, adaptive=True, use_risk=False)
    bh = buy_and_hold(universe)
    from .metrics import metrics as M
    bh_m = M(bh)

    rows = {"agent_adaptive": full["metrics"],
            "static_ensemble": static["metrics"],
            "adaptive_no_risk": norisk["metrics"],
            "buy_and_hold": bh_m}
    print("\n=== RESULTS ===")
    for k, m in rows.items():
        print(f"{k:20s} CAGR={m['cagr']*100:7.2f}%  Sharpe={m['sharpe']:6.2f}  "
              f"MaxDD={m['max_dd']*100:7.2f}%  Win={m['win_rate']*100:5.1f}%")

    # persist everything the dashboard needs
    out = {
        "generated": str(pd.Timestamp.now("UTC")),
        "universe": symbols,
        "metrics": rows,
        "ml_accept_rate": accept_rate,
        "equity": {
            "agent": full["equity"].round(2).astype(str).to_dict(),
            "static": static["equity"].round(2).astype(str).to_dict(),
            "buy_hold": bh.round(2).astype(str).to_dict(),
        },
        "weights": full["weights"].round(4).to_dict(),
        "strategy_pnl_cum": full["strategy_pnl"].cumsum().round(4).to_dict(),
        "ml_retrain_log": ml.retrain_log[-20:],
    }
    # convert timestamps in keys to strings
    for key in ("equity",):
        out[key] = {k: {str(i): v for i, v in d.items()}
                    for k, d in out[key].items()}
    w_cols = out["weights"]  # {strategy: {Timestamp: weight}}
    out["weights"] = {}
    for c, inner in w_cols.items():
        for i, v in inner.items():
            out["weights"].setdefault(str(i), {})[c] = v
    out["strategy_pnl_cum"] = {c: {str(i): v for i, v in inner.items()}
                               for c, inner in out["strategy_pnl_cum"].items()}
    (config.REPORT_DIR / "report.json").write_text(json.dumps(out))
    print(f"\n[exp] report -> {config.REPORT_DIR / 'report.json'}")


if __name__ == "__main__":
    sys.exit(main())
