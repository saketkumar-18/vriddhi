"""Hedge multiplicative-weights meta-allocator — the self-improving core.

Two modes:

* **Windowed Hedge (default, `windowed=True`)** — the weight of each strategy
  is proportional to exp(eta * cumulative_return_over_the_last_L_bars).
  Bounded memory: performance from before the window is FORGOTTEN, so the
  allocator tracks the CURRENT regime instead of averaging the whole history.
  This is the fixed-window variant of Freund & Schapire's Hedge and keeps the
  regret guarantee (vs the best strategy over the window).

  Why this matters (measured): with cumulative log-weights, 5 years of
  history dominate and weights barely move (0.11–0.17 range, lead changed
  11 times in 2000 bars). Windowed weights let the agent genuinely rotate —
  including into the Defensive (cash) strategy when every directional
  strategy is losing.

* **Cumulative Hedge (`windowed=False`)** — classic Hedge over the full
  history; kept for ablation.

Meta-learning: the allocator also tunes its OWN learning rate. It measures
how persistent the daily strategy ranking is over a trailing window (rank
autocorrelation of RAW daily returns). Persistent rankings => regimes stick
=> chase harder (raise eta). Flipping rankings => chasing is noise => back
off (lower eta). The agent learns not just WHICH strategy to trust, but HOW
FAST to shift trust.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config


class HedgeAllocator:
    def __init__(self, strategy_names: list[str], eta: float = config.HEDGE_ETA,
                 lookback: int = config.HEDGE_LOOKBACK,
                 min_weight: float = config.MIN_WEIGHT,
                 smooth: float = config.WEIGHT_SMOOTH,
                 windowed: bool = config.HEDGE_WINDOWED,
                 meta_learn: bool = config.META_LEARN,
                 eta_min: float = config.ETA_MIN,
                 eta_max: float = config.ETA_MAX,
                 rank_window: int = config.RANK_PERSISTENCE_WINDOW):
        self.names = list(strategy_names)
        self.eta = eta
        self.lookback = lookback
        self.min_weight, self.smooth = min_weight, smooth
        self.windowed = windowed
        self.meta_learn = meta_learn
        self.eta_min, self.eta_max = eta_min, eta_max
        self.rank_window = rank_window
        n = len(self.names)
        self.log_weights = np.zeros(n)   # cumulative mode state
        self.smoothed = np.full(n, 1.0 / n)
        self.history: list[np.ndarray] = []
        self.eta_history: list[float] = []
        self._buf: list[np.ndarray] = []
        self._rank_buf: list[np.ndarray] = []

    def weights(self) -> np.ndarray:
        return self.smoothed.copy()

    def _tune_eta(self) -> None:
        """Self-tune learning rate from ranking persistence."""
        if not self.meta_learn or len(self._rank_buf) < self.rank_window + 1:
            return
        recent = np.array(self._rank_buf[-self.rank_window:])
        prev = np.array(self._rank_buf[-self.rank_window - 1:-1])
        cors = []
        for a, b in zip(recent, prev):
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                continue
            c = spearmanr(a, b).statistic
            if np.isfinite(c):
                cors.append(c)
        if not cors:
            return
        persistence = float(np.mean(cors))  # in [-1, 1]
        frac = (persistence + 1.0) / 2.0
        self.eta = self.eta_min + frac * (self.eta_max - self.eta_min)

    def update(self, strategy_returns: np.ndarray) -> np.ndarray:
        """strategy_returns: per-strategy return realized over the last bar."""
        r = np.clip(np.asarray(strategy_returns, dtype=float), -0.5, 0.5)
        self._buf.append(r)
        if len(self._buf) > self.lookback:
            self._buf.pop(0)

        # rank persistence is measured on RAW daily returns (ranking smoothed
        # rewards would measure the smoothing's autocorrelation, not regimes)
        self._rank_buf.append(r.argsort().argsort().astype(float))
        if len(self._rank_buf) > self.rank_window + 2:
            self._rank_buf.pop(0)
        self._tune_eta()

        if self.windowed:
            # fixed-window Hedge: log-weight = eta * trailing cum return
            cum = np.sum(self._buf, axis=0)
            lw = self.eta * cum
        else:
            reward = np.mean(self._buf, axis=0)
            self.log_weights += self.eta * reward
            lw = self.log_weights.copy()
        lw -= lw.max()  # numerical stability
        w = np.exp(lw)
        w = (1 - self.min_weight * len(w)) * (w / w.sum()) + self.min_weight
        self.smoothed = self.smooth * self.smoothed + (1 - self.smooth) * w
        self.smoothed /= self.smoothed.sum()
        self.history.append(self.smoothed.copy())
        self.eta_history.append(self.eta)
        return self.smoothed.copy()

    def weight_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history, columns=self.names)
