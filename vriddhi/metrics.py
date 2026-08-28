"""Performance metrics for equity curves."""
from __future__ import annotations

import numpy as np
import pandas as pd

YEAR = 365


def metrics(equity: pd.Series) -> dict:
    eq = equity.dropna()
    rets = eq.pct_change().dropna()
    n = len(rets)
    if n < 2:
        return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan,
                "calmar": np.nan, "win_rate": np.nan, "n_days": n}
    total = eq.iloc[-1] / eq.iloc[0]
    cagr = total ** (YEAR / n) - 1
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(YEAR)
    dd = (eq / eq.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    return {"cagr": float(cagr), "sharpe": float(sharpe),
            "max_dd": float(dd), "calmar": float(calmar),
            "win_rate": float((rets > 0).mean()), "n_days": int(n)}
