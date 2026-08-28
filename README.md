# Vriddhi (वृद्धि) — a self-improving AI trading agent

> **Paper trading only.** This is a research artifact. No real orders are
> ever placed. Read [RISK.md](RISK.md) before believing anything here.

Vriddhi is an ensemble trading agent whose allocation **improves itself over
time** through three concrete, testable mechanisms — not vibes:

## The three self-improvement loops

1. **Hedge multiplicative-weights meta-allocator.** Four strategies
   (momentum, mean-reversion, Donchian breakout, ML predictor) each propose
   positions. A meta-allocator holds a weight over them and updates daily:
   `w ← w · exp(η · strategy_return)`, renormalized, with a floor so no
   strategy is ever fully eliminated, plus EMA smoothing to curb churn.
   Hedge provides a regret bound vs the best fixed strategy in hindsight —
   capital flows automatically toward whatever regime is currently working.

2. **Walk-forward ML retraining with a validation gate.** The ML strategy
   (HistGradientBoosting over ~17 technical features predicting next-day
   return) retrains every 21 bars on a rolling 365-bar window. Each fresh
   model must clear an out-of-sample rank-IC gate on a held-out 42-bar tail;
   if it fails, the agent keeps the last *trusted* model. The agent learns
   only from validated experience — it never deploys a model that didn't
   prove itself out-of-sample first.

3. **Adaptive risk overlay.** Volatility targeting (scale exposure to ~25%
   annualized) plus a drawdown circuit breaker: at −10% DD exposure halves,
   at −20% it goes to zero until half the trough is recovered.

## Results (walk-forward backtest, Jan 2021 → Aug 2026)

Crypto spot universe: BTC, ETH, SOL, BNB, XRP, ADA. Daily bars, long-only,
10 bps fee + 5 bps slippage per side, $10k start.

| variant | CAGR | Sharpe | Max DD |
|---|---|---|---|
| **adaptive agent (full)** | **16.6%** | **1.06** | **−19.0%** |
| static ensemble (no adaptation) | 16.4% | 1.07 | −18.6% |
| adaptive, no risk overlay | 18.6% | 0.98 | −26.1% |
| buy & hold (equal-weight) | 61.9% | 1.00 | **−90.8%** |

Honest reading: the four strategies are correlated trend-followers, so the
Hedge allocator's CAGR edge over static weights is small — adaptation shines
when strategy returns diverge across regimes. The **risk overlay is where
the agent earns its keep**: it cuts max drawdown from −91% (buy & hold) to
−19% at equal Sharpe. The ML validation gate rejected ~44% of retrain
candidates (888/2025 accepted) — the agent declined to deploy models that
failed out-of-sample, which is self-improvement behaving as designed.

## Repo layout

```
vriddhi/
  config.py       universe, costs, all hyperparameters
  data.py         yfinance fetch (no API keys)
  indicators.py   features: returns, EMAs, RSI, ATR, vol, Donchian, BB
  strategies.py   momentum / mean-reversion / breakout
  ml_strategy.py  walk-forward GBM with validation gate
  meta.py         Hedge multiplicative-weights allocator
  risk.py         vol targeting + drawdown circuit breaker
  backtest.py     walk-forward engine (no lookahead)
  metrics.py      CAGR / Sharpe / maxDD / Calmar / win rate
  experiment.py   full ablation + report.json for the dashboard
  live.py         daily paper-trading tick, state in state/live_state.json
dashboard/        static Chart.js dashboard (deployed on Netlify)
tests/            22 pytest tests (unit + integration, no network)
```

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install numpy pandas scikit-learn yfinance pytest
python -m pytest tests/ -q          # 22 tests, no network
python -m vriddhi.experiment        # full ablation -> reports/report.json
python -m vriddhi.live              # one paper-trading tick (idempotent/day)
```

A scheduled job runs `vriddhi.live` daily; the dashboard shows the live
paper portfolio alongside the backtest evidence.

## Honesty notes

- Signals use the close of day *t*; PnL is earned *t → t+1*. ML trains only
  on data strictly before the prediction date (enforced by construction and
  by test).
- Backtests assume perfect execution at daily closes. Real fills are worse.
- Crypto regimes shift; a model that passed validation last month can stop
  working. The validation gate and weight floor are the defenses, not
  guarantees.
- Nothing here is financial advice. See RISK.md.
