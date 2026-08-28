# Vriddhi (वृद्धि) — a self-improving AI trading agent

> **Paper trading only.** This is a research artifact. No real orders are
> ever placed. Read [RISK.md](RISK.md) before believing anything here.

Vriddhi is an ensemble trading agent whose allocation **improves itself over
time** through three concrete, testable mechanisms — not vibes.

## The three self-improvement loops

1. **Hedge multiplicative-weights meta-allocator over *decorrelated*
   strategies.** Seven strategies propose positions: three per-symbol
   trend/mean-reversion rules (momentum, mean-reversion, Donchian breakout),
   a walk-forward ML predictor, and three *universe-aware* strategies that
   read the whole close matrix — cross-sectional momentum (long relative
   winners), spread reversion (long relative laggards), and a defensive
   sleeve. The last group is **negatively correlated with the trend-followers
   by construction**, which is deliberate: an allocator can only add value if
   its strategies genuinely disagree. A meta-allocator holds a weight over
   them and updates daily: `w ← w · exp(η · trailing_return)`, renormalized
   with a floor so no strategy is ever fully eliminated, plus EMA smoothing.
   This is **windowed Hedge** — it scores each strategy on a trailing 63-bar
   window so it *forgets* stale performance and rotates when regimes flip.
   Hedge provides a regret bound vs the best fixed strategy in hindsight.

   **Meta-learning:** the allocator also tunes its *own* learning rate η. It
   measures how persistent the strategy ranking is (rank autocorrelation over
   42 bars); when the same strategies stay on top it chases harder (raises η),
   and when rankings flip constantly it backs off (lowers η). So the agent
   learns not just *which* strategy to trust, but *how fast* to shift trust.

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

**Full sample:**

| variant | CAGR | Sharpe | Max DD |
|---|---|---|---|
| **adaptive agent (full)** | **9.3%** | **0.88** | **−21.1%** |
| static ensemble (no adaptation) | 9.2% | 0.88 | −20.1% |
| adaptive, no meta-learning | 9.3% | 0.88 | −21.0% |
| buy & hold (equal-weight) | 62.0% | 1.00 | **−90.8%** |

**Trailing-2-year holdout (out-of-time — no variant was tuned on this):**

| variant | CAGR | Sharpe | Max DD |
|---|---|---|---|
| **adaptive agent** | **6.0%** | **0.48** | **−17.3%** |
| static ensemble | 5.8% | 0.47 | −17.2% |
| adaptive, no meta-learning | 6.0% | 0.48 | −17.3% |
| buy & hold | 12.6% | 0.50 | **−65.6%** |

Honest reading: the **risk overlay is where the agent earns its keep** — it
cuts max drawdown from −91% (buy & hold) to −21% on the full sample and from
−66% to −17% on the holdout, at comparable Sharpe. The Hedge allocator's CAGR
edge over static weights is small here because, even with decorrelated
strategies, the ensemble's members still move together in crypto's dominant
regime. What *did* change vs the v1 build: the lead strategy now rotates
**11 times** over the sample (vs 3 before), weights spread across a wider
range, and the self-tuned learning rate η moved between 0.25 and 0.42 as
ranking persistence shifted — the machinery is adapting, the market just
isn't paying a large premium for it in this universe. The ML validation gate
accepted ~56% of retrain candidates and refused the rest — the agent declined
to deploy models that failed out-of-sample, which is self-improvement
behaving as designed.

## Repo layout

```
vriddhi/
  config.py       universe, costs, all hyperparameters
  data.py         yfinance fetch (no API keys)
  indicators.py   features: returns, EMAs, RSI, ATR, vol, Donchian, BB
  strategies.py   momentum / mean-reversion / breakout + universe-aware
                  xsmom / spreadrev / defensive (decorrelated)
  ml_strategy.py  walk-forward GBM with validation gate
  meta.py         windowed Hedge allocator + self-tuned eta (meta-learning)
  risk.py         vol targeting + drawdown circuit breaker
  backtest.py     walk-forward engine (no lookahead)
  metrics.py      CAGR / Sharpe / maxDD / Calmar / win rate
  experiment.py   full ablation + out-of-time holdout -> reports/report.json
  live.py         daily paper-trading tick, state in state/live_state.json
dashboard/        static Chart.js dashboard (deployed on Vercel)
tests/            33 pytest tests (unit + integration, no network)
```

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install numpy pandas scikit-learn yfinance pytest scipy
python -m pytest tests/ -q          # 33 tests, no network
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
