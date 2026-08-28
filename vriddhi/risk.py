"""Portfolio-level risk overlays: volatility targeting + drawdown breaker."""
from __future__ import annotations

import numpy as np

from . import config


class RiskManager:
    def __init__(self, vol_target: float = config.VOL_TARGET,
                 vol_lookback: int = config.VOL_LOOKBACK,
                 max_gross: float = config.MAX_GROSS,
                 dd_soft: float = config.DD_SOFT,
                 dd_hard: float = config.DD_HARD,
                 dd_recovery: float = config.DD_RECOVERY):
        self.vol_target, self.vol_lookback = vol_target, vol_lookback
        self.max_gross = max_gross
        self.dd_soft, self.dd_hard, self.dd_recovery = dd_soft, dd_hard, dd_recovery
        self.peak = 1.0
        self.tripped = False
        self.trough_dd = 0.0

    def exposure_multiplier(self, port_ret_history: np.ndarray,
                            equity: float) -> float:
        """Scalar in [0, max_gross] applied to all target positions."""
        rets = np.asarray(port_ret_history, dtype=float)
        mult = 1.0
        if len(rets) >= self.vol_lookback:
            vol = rets[-self.vol_lookback:].std() * np.sqrt(365)
            if vol > 1e-8:
                mult = min(mult, self.vol_target / vol)
        # drawdown state machine
        self.peak = max(self.peak, equity)
        dd = 1 - equity / self.peak
        if self.tripped:
            self.trough_dd = max(self.trough_dd, dd)
            if dd <= self.trough_dd * (1 - self.dd_recovery):
                self.tripped = False
                self.trough_dd = 0.0
            mult = 0.0 if self.tripped else mult
        else:
            if dd >= self.dd_hard:
                self.tripped = True
                self.trough_dd = dd
                mult = 0.0
            elif dd >= self.dd_soft:
                mult *= 0.5
        return float(np.clip(mult, 0.0, self.max_gross))
