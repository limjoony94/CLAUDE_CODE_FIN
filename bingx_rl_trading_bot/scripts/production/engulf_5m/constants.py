"""
Engulf 5m Bot - Constants Module
All magic numbers and configuration constants extracted for better maintainability.
"""

# ============================================================
# BOT IDENTIFICATION
# ============================================================
BOT_NAME = "engulf_5m_bot"
BOT_VERSION = "2.3"

# ============================================================
# FILE PATHS (relative to bot root)
# ============================================================
CONFIG_FILE = "config/engulf_5m_config.yaml"
STATE_FILE = "results/engulf_5m_bot_state.json"
LOCK_FILE = "results/engulf_5m_bot.lock"
METRICS_FILE = "results/engulf_5m_metrics.json"
LOG_DIR = "logs"
API_KEYS_FILE = "config/api_keys.yaml"

# ============================================================
# TRADING PARAMETERS
# ============================================================
SLIPPAGE_BUFFER_PCT = 0.02
FEE_PCT = 0.05

# ============================================================
# API & PERFORMANCE SETTINGS
# ============================================================
CACHE_TTL_SECONDS = 5
MAX_OHLCV_CANDLES = 100
METRICS_SAVE_INTERVAL = 10

# ============================================================
# PRICE TOLERANCE (for TP/SL detection)
# ============================================================
PRICE_TOLERANCE_PCT = 0.003  # 0.3% tolerance for price matching
TP_LOWER_MULT = 0.999  # TP detected if price >= TP * 0.999
TP_UPPER_MULT = 1.001  # TP detected if price <= TP * 1.001
SL_LOWER_MULT = 0.999  # SL tolerance
SL_UPPER_MULT = 1.001  # SL tolerance

# ============================================================
# QUANTITY TOLERANCES
# ============================================================
QTY_TOLERANCE = 0.001  # Quantity mismatch tolerance
QTY_TOLERANCE_SCALEOUT = 0.0001  # Scale-out stage detection
QTY_REDUCTION_THRESHOLD = 0.99  # Detect quantity reduction

# ============================================================
# TIMING CONSTANTS
# ============================================================
CANDLE_DURATION_MS = 300000  # 5 minutes in milliseconds
CANDLE_SETTLE_SECONDS = 15  # Wait time for candle data to settle
DEFAULT_SLEEP_INTERVAL = 30  # Main loop sleep (seconds)
POSITION_CHECK_SLEEP = 30  # Sleep when position exists (seconds)
DAILY_LOSS_PAUSE_SECONDS = 300  # Pause when daily loss limit hit

# Entry price fetch retry
ENTRY_PRICE_FETCH_DELAY = 0.5
EXIT_PRICE_FETCH_DELAY = 0.5
EXIT_PRICE_RETRY_DELAY = 0.5
EXIT_PRICE_INITIAL_DELAY = 1.0
MAX_EXIT_PRICE_RETRIES = 3

# ============================================================
# INTERVAL CONSTANTS
# ============================================================
TP_SL_CHECK_INTERVAL = 20  # Iterations between TP/SL verification
DEFAULT_HEALTH_CHECK_INTERVAL = 50  # Iterations between health checks
LOG_STATUS_INTERVAL = 10  # Iterations between status logs

# ============================================================
# BACKUP SETTINGS
# ============================================================
MAX_STATE_BACKUPS = 3

# ============================================================
# CIRCUIT BREAKER DEFAULTS
# ============================================================
CB_FAILURE_THRESHOLD = 5
CB_RESET_TIMEOUT = 60.0

# ============================================================
# API RETRY DEFAULTS
# ============================================================
API_MAX_ATTEMPTS = 3
API_BASE_DELAY = 2
API_MAX_DELAY = 30

# ============================================================
# METRICS DEFAULTS (from backtest)
# ============================================================
EXPECTED_WIN_RATE = 60.0
EXPECTED_AVG_WIN = 2.5
EXPECTED_AVG_LOSS = 2.0
EXPECTED_EDGE = 78.02
METRICS_WINDOW_SIZE = 50  # Recent trades to keep for rolling stats
MIN_TRADES_FOR_COMPARISON = 5

# ============================================================
# STATE FILE AGE THRESHOLD
# ============================================================
STATE_STALE_THRESHOLD_SECONDS = 300

# ============================================================
# POSITION SYNC
# ============================================================
POSITION_SYNC_INTERVAL_MINUTES = 5

# ============================================================
# PRECISION
# ============================================================
PRICE_ROUND_DECIMALS = 1
QUANTITY_ROUND_DECIMALS = 4

# ============================================================
# DEFAULT CONFIGURATION
# ============================================================
DEFAULT_CONFIG = {
    'symbol': 'BTC-USDT',
    'timeframe': '5m',
    'leverage': 3,
    'exchange_leverage': 10,
    'position_mode': 'one-way',
    'margin_mode': 'crossed',
    'position_size_pct': 95,
    'strategy': {
        'tp_pct': 1.2,
        'sl_pct': 1.5,
        'cooldown_candles': 0,
        'use_body_filter': False,
        'min_body_ratio': 0.5,
        'use_prev_body_filter': False,
        'min_prev_body_ratio': 0.3,
        'use_volume_filter': False,
        'volume_ma_period': 20,
        'min_volume_ratio': 1.0,
    },
    'risk': {
        'max_daily_loss_pct': 10,
        'max_position_size_usd': 10000,
    },
    'api': {
        'retry_attempts': API_MAX_ATTEMPTS,
        'retry_delay': API_BASE_DELAY,
        'max_retry_delay': API_MAX_DELAY,
    },
    'debug_mode': False,
    'json_logging': False,
    'health_check_interval': DEFAULT_HEALTH_CHECK_INTERVAL,
}
