"""Global configuration for Vriddhi."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATE_DIR = ROOT / "state"
REPORT_DIR = ROOT / "reports"

# --- Universe (crypto spot via yfinance, no API keys) ---
UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD"]
BENCHMARK = "BTC-USD"
START = "2021-01-01"

# --- Trading costs (conservative for crypto spot) ---
FEE_BPS = 10          # 0.10% per side (typical taker fee)
SLIPPAGE_BPS = 5      # 0.05% per side

# --- Backtest ---
INITIAL_CAPITAL = 10_000.0
WARMUP = 60           # bars needed before strategies emit signals
REBALANCE_EVERY = 1   # days

# --- Meta-allocator (Hedge) ---
HEDGE_ETA = 0.25      # base learning rate for multiplicative weights
HEDGE_LOOKBACK = 63   # window (bars) of strategy PnL driving the weights
HEDGE_WINDOWED = True # fixed-window Hedge: forget performance before window
MIN_WEIGHT = 0.05     # floor so no strategy is fully killed
WEIGHT_SMOOTH = 0.3   # EMA smoothing applied to raw hedge weights

# --- Meta-learning: allocator self-tunes eta from ranking persistence ---
META_LEARN = True
ETA_MIN = 0.05        # when strategy rankings flip constantly, chase softly
ETA_MAX = 0.60        # when rankings persist, chase hard
RANK_PERSISTENCE_WINDOW = 42  # bars over which rank autocorrelation is measured

# --- ML strategy ---
ML_RETRAIN_EVERY = 21       # bars between walk-forward retrains
ML_TRAIN_WINDOW = 365       # bars of training data
ML_VALID_WINDOW = 42        # out-of-sample validation gate
ML_MIN_VALID_IC = 0.0       # min validation rank-IC to trust the model

# --- Risk ---
VOL_TARGET = 0.25           # annualized portfolio vol target
VOL_LOOKBACK = 21
MAX_GROSS = 1.0             # no leverage
DD_SOFT = 0.10              # drawdown at which exposure halves
DD_HARD = 0.20              # drawdown at which exposure -> 0 (circuit breaker)
DD_RECOVERY = 0.5           # recover when DD retraces to this fraction of trough
