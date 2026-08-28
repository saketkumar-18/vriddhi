# RISK.md — what this is and isn't

## What this is
A **research artifact and learning project**: a paper-trading agent that
demonstrates online learning (Hedge multiplicative weights), walk-forward
model validation, and adaptive risk management on public crypto price data.

## What this is NOT
- **Not financial advice.** Nothing in this repo is a recommendation to buy
  or sell anything.
- **Not a money printer.** Backtest Sharpe ratios are hypothetical. They
  assume perfect daily-close execution, no market impact, and stable
  statistical structure — none of which survive contact with reality.
- **Not connected to any broker.** There is no execution adapter and there
  should not be one without deliberate, separate, adult-supervised work.

## Specific risks if anyone were to trade anything like this
1. **Overfitting.** Walk-forward validation reduces it; it does not remove
   it. Multiple testing across strategy/hyperparameter choices inflates
   apparent performance.
2. **Regime change.** Crypto microstructure and macro drivers shift without
   notice. Models trained on 2021–2024 data may be meaningless in 2027.
3. **Execution gap.** Fees modeled here (10 bps) are optimistic in thin
   markets; slippage on real orders can be multiples of that.
4. **Total loss.** Crypto assets can go to zero. Position sizing and
   circuit breakers limit simulated drawdowns; they cannot limit a
   counterparty or protocol failure.
5. **Data quality.** Free data feeds have gaps, restatements, and survivor
   bias (this universe was picked knowing all six assets still exist).

## Ethics
- No real capital is at risk and no third parties are affected by this
  system's actions.
- The project is published with full methodology so claims can be audited
  and reproduced.
- If this ever approached real execution, the prerequisites would be:
  regulatory/KYC compliance, explicit risk capital limits, human approval
  on every order, and kill switches. None of that exists here by design.

## The one-sentence version
This agent trades imaginary money to test whether adaptive ensembles beat
static ones; treat every number it produces as a hypothesis, not a promise.
