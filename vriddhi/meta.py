"""Hedge multiplicative-weights meta-allocator — the self-improving core.

Maintains a weight over base strategies. Each day, weights are multiplied by
exp(eta * strategy_return) and renormalized (with a floor so no strategy is
ever fully eliminated), then EMA-smoothed to curb churn. Hedge gives a
regret bound vs the best fixed strategy in hindsight: the allocator
concentrates capital in whatever is currently working, automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


class HedgeAllocator:
    def __init__(self, strategy_names: list[str], eta: float = config.HEDGE_ETA,
                 lookback: int = config.HEDGE_LOOKBACK,
                 min_weight: float = config.MIN_WEIGHT,
                 smooth: float = config.WEIGHT_SMOOTH):
        self.names = list(strategy_names)
        self.eta, self.lookback = eta, lookback
        self.min_weight, self.smooth = min_weight, smooth
        n = len(self.names)
        self.log_weights = np.zeros(n)
        self.smoothed = np.full(n, 1.0 / n)
        self.history: list[np.ndarray] = []
        self._buf: list[np.ndarray] = []

    def weights(self) -> np.ndarray:
        return self.smoothed.copy()

    def update(self, strategy_returns: np.ndarray) -> np.ndarray:
        """strategy_returns: per-strategy return realized over the last bar.

        Reward estimate is the trailing-mean return over `lookback` bars —
        Hedge on smoothed rewards, which chases regimes instead of noise.
        """
        r = np.clip(np.asarray(strategy_returns, dtype=float), -0.5, 0.5)
        self._buf.append(r)
        if len(self._buf) > self.lookback:
            self._buf.pop(0)
        reward = np.mean(self._buf, axis=0)
        self.log_weights += self.eta * reward
        self.log_weights -= self.log_weights.max()  # numerical stability
        w = np.exp(self.log_weights)
        w = (1 - self.min_weight * len(w)) * (w / w.sum()) + self.min_weight
        self.smoothed = self.smooth * self.smoothed + (1 - self.smooth) * w
        self.smoothed /= self.smoothed.sum()
        self.history.append(self.smoothed.copy())
        return self.smoothed.copy()

    def weight_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history, columns=self.names)
