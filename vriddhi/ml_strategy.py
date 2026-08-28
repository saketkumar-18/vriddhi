"""Walk-forward ML strategy.

A HistGradientBoosting regressor predicts next-day forward return from
technical features. It is retrained every ML_RETRAIN_EVERY bars on a rolling
window and gated by an out-of-sample validation check (rank-IC on a held-out
tail). If the fresh model fails the gate, the strategy falls back to its last
trusted model — this is the "learn only from validated experience" half of
self-improvement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config
from .indicators import features_for_symbol


class WalkForwardML:
    name = "ml"

    def __init__(self, retrain_every: int = config.ML_RETRAIN_EVERY,
                 train_window: int = config.ML_TRAIN_WINDOW,
                 valid_window: int = config.ML_VALID_WINDOW,
                 min_valid_ic: float = config.ML_MIN_VALID_IC):
        self.retrain_every = retrain_every
        self.train_window = train_window
        self.valid_window = valid_window
        self.min_valid_ic = min_valid_ic
        self.model = None
        self.feature_cols: list[str] = []
        self.retrain_log: list[dict] = []

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Train on window, validate on held-out tail; keep if IC passes."""
        df = pd.concat([X, y.rename("y")], axis=1).dropna()
        if len(df) < self.train_window + self.valid_window:
            return False
        train = df.iloc[: self.train_window]
        valid = df.iloc[self.train_window:]
        m = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, max_depth=3,
            l2_regularization=1.0, min_samples_leaf=30, random_state=0)
        m.fit(train[self.feature_cols], train["y"])
        pred = m.predict(valid[self.feature_cols])
        ic = spearmanr(pred, valid["y"]).statistic
        ok = np.isfinite(ic) and ic >= self.min_valid_ic
        self.retrain_log.append({"n_train": len(train), "valid_ic": float(ic),
                                 "accepted": bool(ok)})
        if ok:
            self.model = m
        return ok

    def signals(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Walk-forward positions: model score -> position in [0,1]."""
        feats = features_for_symbol(ohlcv)
        self.feature_cols = list(feats.columns)
        fwd_ret = ohlcv["Close"].pct_change().shift(-1)  # label: next-day return
        pos = pd.Series(0.0, index=ohlcv.index)
        last_train = -10**9
        for i in range(len(ohlcv)):
            if i - last_train >= self.retrain_every:
                window = slice(max(0, i - self.train_window - self.valid_window), i)
                if self._fit(feats.iloc[window], fwd_ret.iloc[window]):
                    pass  # accepted
                last_train = i
            if self.model is None:
                continue
            row = feats.iloc[[i]].dropna(axis=1)
            if row.shape[1] < len(self.feature_cols):
                continue
            x = feats.iloc[[i]][self.feature_cols]
            if x.isna().any(axis=1).iloc[0]:
                continue
            score = float(self.model.predict(x)[0])
            # squash score to [0,1]: positive expected return -> long
            pos.iloc[i] = float(np.clip(0.5 + score / 0.02, 0.0, 1.0))
        return pos
