"""Vriddhi — a self-improving paper-trading agent.

Modules:
  config      — universe, costs, hyperparameters
  data        — yfinance fetch + cache
  indicators  — feature engineering
  strategies  — rule-based signal strategies
  ml_strategy — walk-forward gradient-boosting predictor
  meta        — Hedge multiplicative-weights meta-allocator
  risk        — vol targeting + drawdown circuit breaker
  backtest    — walk-forward engine with fees/slippage
  metrics     — performance statistics
  live        — daily paper-trading tick with state persistence
"""

__version__ = "0.1.0"
