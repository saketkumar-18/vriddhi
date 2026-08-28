"""Diagnose WHY adaptation isn't helping: measure the raw material it has.

If per-strategy returns are highly correlated and their Sharpe ratios sit
within noise of each other, NO allocator — however clever — can add value.
The regret bound is vs the best FIXED strategy in hindsight; if that best
isn't much better than equal weight, there's little regret to remove.
"""
import json
import numpy as np
import pandas as pd

from vriddhi import config
from vriddhi.backtest import build_strategy_positions
from vriddhi.data import load_universe
from vriddhi.ml_strategy import WalkForwardML

universe = load_universe()
symbols = list(universe.keys())
closes = pd.DataFrame({s: df["Close"] for s, df in universe.items()})
rets = closes.pct_change().fillna(0.0)

ml = WalkForwardML()
ml_signals = {s: ml.signals(universe[s]) for s in symbols}
strat_pos = build_strategy_positions(universe, ml_signals)
names = list(strat_pos.keys())

n = len(closes)
pos_stack = np.zeros((n, len(names), len(symbols)))
for k, nm in enumerate(names):
    for j, s in enumerate(symbols):
        pos_stack[:, k, j] = strat_pos[nm][s].values

# per-strategy daily return (positions from t-1 earn r_t)
strat_daily = np.zeros((n - 1, len(names)))
for t in range(1, n):
    strat_daily[t - 1] = (pos_stack[t - 1] @ rets.iloc[t].values) / len(symbols)
sd = pd.DataFrame(strat_daily, index=closes.index[1:], columns=names)

print("=== per-strategy Sharpe (daily, annualized) ===")
sharpe = sd.mean() / sd.std() * np.sqrt(365)
for nm in names:
    print(f"  {nm:12s} sharpe={sharpe[nm]:6.2f}  "
          f"ann_ret={sd[nm].mean()*365*100:6.1f}%  ann_vol={sd[nm].std()*np.sqrt(365)*100:5.1f}%")

print("\n=== pairwise correlation of daily strategy returns ===")
corr = sd.corr()
print(corr.round(2).to_string())
off = corr.values[np.triu_indices_from(corr.values, k=1)]
print(f"\n  mean off-diagonal corr: {off.mean():.3f}")

print("\n=== dispersion of trailing-63d cum returns across strategies ===")
trail = sd.rolling(63).sum()
disp = trail.max(axis=1) - trail.min(axis=1)
print(f"  mean spread (best-worst, 63d): {disp.mean()*100:5.1f}%")
print(f"  median spread: {disp.median()*100:5.1f}%")
print(f"  spread > 20% on {(disp>0.20).mean()*100:.0f}% of days")

print("\n=== how often does the 63d LEADER persist? ===")
trail_valid = trail.dropna()
leader = trail_valid.idxmax(axis=1)
persist = (leader == leader.shift(1)).mean()
print(f"  P(same leader as yesterday): {persist*100:.0f}%")
lead_codes = leader.astype("category").cat.codes
lead21 = lead_codes.rolling(21).apply(lambda x: (x == x.iloc[-1]).mean(), raw=False)
print(f"  mean fraction of last-21d days sharing today's leader: {lead21.mean()*100:.0f}%")
print(f"  leader value counts:\n{leader.value_counts().to_string()}")

print("\n=== equal-weight vs best-fixed-in-hindsight (the ceiling) ===")
ew = sd.mean(axis=1)
ew_sharpe = ew.mean()/ew.std()*np.sqrt(365)
best = sharpe.max()
print(f"  equal-weight sharpe: {ew_sharpe:.2f}")
print(f"  best single strategy sharpe: {best:.2f}  ({sharpe.idxmax()})")
print(f"  -> max possible gain from perfect selection: {best-ew_sharpe:.2f} sharpe")
