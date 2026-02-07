"""
Pattern 5m Bot - Constants and Magic Numbers
All configurable values and pattern definitions.
"""

import os
from enum import Enum
from typing import List

# ============================================================
# BOT IDENTIFICATION
# ============================================================
BOT_NAME = "pattern_5m_bot"
BOT_VERSION = "1.25.5"  # v1.25.5: Fix CWD-dependent path bug (absolute paths)
# 23 patterns (6L+17S), validated via quality pruning analysis (2026-02-07)

# ============================================================
# PROJECT ROOT (absolute path, CWD-independent)
# ============================================================
# __file__ = .../bingx_rl_trading_bot/scripts/production/pattern_5m/constants.py
# _THIS_DIR = .../bingx_rl_trading_bot/scripts/production/pattern_5m/
# parent(1) = .../bingx_rl_trading_bot/scripts/production/
# parent(2) = .../bingx_rl_trading_bot/scripts/
# parent(3) = .../bingx_rl_trading_bot/  ← PROJECT_ROOT
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))

# ============================================================
# FILE PATHS (absolute, CWD-independent)
# ============================================================
CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "pattern_5m_config.yaml")
STATE_FILE = os.path.join(PROJECT_ROOT, "results", "pattern_5m_bot_state.json")
LOCK_FILE = os.path.join(PROJECT_ROOT, "results", "pattern_5m_bot.lock")
METRICS_FILE = os.path.join(PROJECT_ROOT, "results", "pattern_5m_metrics.json")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
API_KEYS_FILE = os.path.join(PROJECT_ROOT, "config", "api_keys.yaml")

# ============================================================
# Candle Type Classification
# ============================================================

class CandleType(Enum):
    """12-type candle classification system."""
    DOJI = "D"
    DRAGONFLY = "DF"
    GRAVESTONE = "GS"
    HAMMER = "H"
    INV_HAMMER = "IH"
    SPINNING_TOP = "ST"
    MARUBOZU_UP = "MU"
    MARUBOZU_DOWN = "MD"
    BIG_UP = "BU"
    BIG_DOWN = "BD"
    MED_UP = "U"
    MED_DOWN = "DN"


# ============================================================
# Market Regime Detection (v1.18)
# Research: regime_adaptive_research.py, regime_adaptive_validation.py
# WF 5/5 (vs v1.17: 0/5), Total PnL: +415% (vs v1.17: -1081%)
# ============================================================

class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"

# Regime detection parameters
# DEPRECATED: Regime disabled since v1.19.0 (tight TP/SL proved regime-independent)
REGIME_DETECTION_ENABLED = False  # v1.19.0: Disabled - tight TP/SL is regime-independent
REGIME_LOOKBACK_BARS = 100          # 100 bars = ~8.3 hours
REGIME_TREND_THRESHOLD = 1.5        # % price change for BULL/BEAR (optimized from 2.0)
REGIME_VOL_THRESHOLD = 0.15         # ATR% threshold for HIGH/LOW volatility

# Regime-specific patterns and TP/SL (v1.18)
# DEPRECATED: Regime disabled since v1.19.0 - kept for reference only
# Each regime has its own optimal patterns based on backtest validation
REGIME_PATTERNS = {
    "BULL": {
        # Counter-trend SHORT patterns work best in BULL market
        "MU-ST-DN": ("SHORT", 3.0, 1.5),   # EV=+5.25%, WR=100%
        "IH-DN-DN": ("SHORT", 2.5, 1.5),   # EV=+5.08%, WR=83.3%
        "BU-U-DN": ("SHORT", 1.5, 2.0),    # EV=+1.86%, WR=87.5%
    },
    "BEAR": {
        # Mix of trend-following SHORT and counter-trend LONG
        "BD-ST-DN": ("SHORT", 1.5, 1.5),   # EV=+2.41%, WR=75.0%
        "BU-U-DN": ("SHORT", 3.0, 1.5),    # EV=+1.97%, WR=55.6%
        "U-ST-U": ("LONG", 1.5, 3.0),      # EV=+1.80%, WR=75.0% (bounce)
        "DN-DN-BD": ("SHORT", 2.0, 2.5),   # EV=+1.93%, WR=53.3%
    },
    "SIDEWAYS": {
        # Range-bound patterns (most common regime: 85.8% of time)
        "BD-BD-BD": ("SHORT", 3.0, 2.5),   # EV=+1.84%, WR=61.3%
        "ST-BD-DN": ("LONG", 3.0, 3.0),    # EV=+1.53%, WR=62.2%
        "MU-ST-DN": ("SHORT", 3.0, 2.0),   # EV=+1.52%, WR=57.1%
        "DN-DN-BD": ("SHORT", 1.5, 2.0),   # EV=+0.42%, WR=50.0%
    },
}

# Default regime to use when detection fails
# DEPRECATED: Regime disabled since v1.19.0 - kept for reference only
DEFAULT_REGIME = MarketRegime.SIDEWAYS

# ============================================================
# Validated Patterns (v1.25.4 Quality-Filtered Portfolio)
# Research: comprehensive_pattern_evaluation.py (2026-02-07)
# Validation: MC < 0.05, WF >= 4/5, WR >= 85% (quality filter)
# Removed v1.25.4: DN-IH-ST (76.3%), U-U-DN (84.1%), U-U-U (84.2%) - WR < 85%
# ============================================================

# LONG patterns - v1.25.4: 6 total (WR >= 85% only)
VALIDATED_LONG_PATTERNS: List[str] = [
    "MD-BU-U",     # WR 94.4%, 54t, MC=0.0004, WF 5/5, PnL +14.1%
    "MU-MU-U",     # WR 93.3%, 30t, MC=0.0011, WF 5/5, PnL +12.6%
    "MU-U-MU",     # WR 96.3%, 54t, MC=0.0000, WF 4/5, PnL +16.6%
    "BU-BU-BD",    # WR 90.5%, 42t, MC=0.0038, WF 5/5, PnL +14.4%
    "ST-MU-U",     # WR 91.4%, 81t, MC=0.0000, WF 5/5, PnL +29.7%
    "IH-DN-DN",    # WR 87.5%, 56t, MC=0.0179, WF 4/5, PnL +14.7%
    # REMOVED v1.25.4: "DN-IH-ST" (WR 76.3% < 85%)
    # REMOVED v1.25.4: "U-U-DN" (WR 84.1% < 85%)
    # REMOVED v1.25.4: "U-U-U" (WR 84.2% < 85%)
]

# SHORT patterns - v1.25.4: 17 total (WR >= 85% only)
VALIDATED_SHORT_PATTERNS: List[str] = [
    "MD-ST-ST",    # WR 98.5%, 65t, MC=0.0000, WF 5/5, PnL +23.5%
    "U-MU-BU",     # WR 98.1%, 54t, MC=0.0000, WF 5/5, PnL +19.1%
    "MU-BU-DN",    # WR 97.8%, 45t, MC=0.0000, WF 5/5, PnL +24.3%
    "ST-H-U",      # WR 97.1%, 34t, MC=0.0000, WF 5/5, PnL +11.1%
    "ST-DN-H",     # WR 93.6%, 47t, MC=0.0000, WF 5/5, PnL +20.1%
    "MD-MU-U",     # WR 90.5%, 42t, MC=0.0000, WF 5/5, PnL +25.8%
    "BU-U-ST",     # WR 91.7%, 72t, MC=0.0001, WF 5/5, PnL +27.0%
    "H-DN-ST",     # WR 97.3%, 37t, MC=0.0000, WF 5/5, PnL +19.5%
    "DN-BD-BU",    # WR 94.4%, 71t, MC=0.0000, WF 5/5, PnL +31.8%
    "DN-BU-U",     # WR 88.9%, 90t, MC=0.0000, WF 5/5, PnL +51.0%
    "DN-U-U",      # WR 91.1%, 644t, MC=0.0000, WF 5/5, PnL +115.1% ★★ TOP
    "ST-DN-U",     # WR 88.1%, 286t, MC=0.0000, WF 5/5, PnL +79.8%
    "ST-DN-DN",    # WR 88.1%, 269t, MC=0.0000, WF 5/5, PnL +75.0%
    "U-ST-DN",     # WR 87.8%, 270t, MC=0.0000, WF 5/5, PnL +72.9%
    "U-U-ST",      # WR 87.6%, 266t, MC=0.0000, WF 5/5, PnL +70.5%
    "ST-U-DN",     # WR 87.7%, 260t, MC=0.0000, WF 5/5, PnL +69.6%
    "ST-ST-U",     # WR 91.8%, 159t, MC=0.0000, WF 5/5, PnL +60.3%
    # REMOVED v1.25.4: "DN-DN-DN" (WR 80.6% < 85%)
]

# ============================================================
# Pattern Historical Statistics (for confidence calculation)
# Source: v1.25.4 quality-filtered portfolio (2026-02-07)
# ============================================================
PATTERN_STATS = {
    # LONG patterns (6) - v1.25.4: WR >= 85% only
    "MD-BU-U": {"direction": "LONG", "trades": 54, "wr": 94.4, "mc": 0.0004, "wf": 5, "periods": 3},
    "MU-MU-U": {"direction": "LONG", "trades": 30, "wr": 93.3, "mc": 0.0011, "wf": 5, "periods": 3},
    "MU-U-MU": {"direction": "LONG", "trades": 54, "wr": 96.3, "mc": 0.0000, "wf": 4, "periods": 3},
    "BU-BU-BD": {"direction": "LONG", "trades": 42, "wr": 90.5, "mc": 0.0038, "wf": 5, "periods": 3},
    "ST-MU-U": {"direction": "LONG", "trades": 81, "wr": 91.4, "mc": 0.0000, "wf": 5, "periods": 3},
    "IH-DN-DN": {"direction": "LONG", "trades": 56, "wr": 87.5, "mc": 0.0179, "wf": 4, "periods": 3},
    # SHORT patterns (17) - v1.25.4: WR >= 85% only
    "MD-ST-ST": {"direction": "SHORT", "trades": 65, "wr": 98.5, "mc": 0.0000, "wf": 5, "periods": 3},
    "U-MU-BU": {"direction": "SHORT", "trades": 54, "wr": 98.1, "mc": 0.0000, "wf": 5, "periods": 3},
    "MU-BU-DN": {"direction": "SHORT", "trades": 45, "wr": 97.8, "mc": 0.0000, "wf": 5, "periods": 3},
    "ST-H-U": {"direction": "SHORT", "trades": 34, "wr": 97.1, "mc": 0.0000, "wf": 5, "periods": 3},
    "ST-DN-H": {"direction": "SHORT", "trades": 47, "wr": 93.6, "mc": 0.0000, "wf": 5, "periods": 3},
    "MD-MU-U": {"direction": "SHORT", "trades": 42, "wr": 90.5, "mc": 0.0000, "wf": 5, "periods": 3},
    "BU-U-ST": {"direction": "SHORT", "trades": 72, "wr": 91.7, "mc": 0.0001, "wf": 5, "periods": 3},
    "H-DN-ST": {"direction": "SHORT", "trades": 37, "wr": 97.3, "mc": 0.0000, "wf": 5, "periods": 3},
    "DN-BD-BU": {"direction": "SHORT", "trades": 71, "wr": 94.4, "mc": 0.0000, "wf": 5, "periods": 3},
    "DN-BU-U": {"direction": "SHORT", "trades": 90, "wr": 88.9, "mc": 0.0000, "wf": 5, "periods": 3},
    "DN-U-U": {"direction": "SHORT", "trades": 644, "wr": 91.1, "mc": 0.0000, "wf": 5, "periods": 3},
    "ST-DN-U": {"direction": "SHORT", "trades": 286, "wr": 88.1, "mc": 0.0000, "wf": 5, "periods": 3},
    "ST-DN-DN": {"direction": "SHORT", "trades": 269, "wr": 88.1, "mc": 0.0000, "wf": 5, "periods": 3},
    "U-ST-DN": {"direction": "SHORT", "trades": 270, "wr": 87.8, "mc": 0.0000, "wf": 5, "periods": 3},
    "U-U-ST": {"direction": "SHORT", "trades": 266, "wr": 87.6, "mc": 0.0000, "wf": 5, "periods": 3},
    "ST-U-DN": {"direction": "SHORT", "trades": 260, "wr": 87.7, "mc": 0.0000, "wf": 5, "periods": 3},
    "ST-ST-U": {"direction": "SHORT", "trades": 159, "wr": 91.8, "mc": 0.0000, "wf": 5, "periods": 3},
}

# Confidence calculation weights
CONFIDENCE_WEIGHT_CLARITY = 0.40      # Candle classification clarity
CONFIDENCE_WEIGHT_HISTORICAL = 0.30   # Historical pattern win rate
CONFIDENCE_WEIGHT_REGIME = 0.30       # DEPRECATED: Regime disabled since v1.19.0 (placeholder)

# Confidence logging file
CONFIDENCE_LOG_FILE = os.path.join(PROJECT_ROOT, "results", "pattern_5m_confidence_log.csv")


# ============================================================
# Pattern TP/SL (v1.25.4 Per-Pattern Optimized)
# Research: quality-filtered portfolio (2026-02-07)
# Individually optimized TP/SL for each pattern
# ============================================================
PATTERN_OPTIMAL_TPSL = {
    # LONG patterns (6) - format: (tp_pct, sl_pct)
    "MD-BU-U": (0.5, 2.0),
    "MU-MU-U": (0.7, 2.0),
    "MU-U-MU": (0.5, 2.0),
    "BU-BU-BD": (0.7, 2.0),
    "ST-MU-U": (0.7, 2.0),
    "IH-DN-DN": (0.7, 2.0),
    # SHORT patterns (17)
    "MD-ST-ST": (0.5, 2.0),
    "U-MU-BU": (0.5, 2.0),
    "MU-BU-DN": (0.7, 2.0),
    "ST-H-U": (0.5, 2.0),
    "ST-DN-H": (0.7, 2.0),
    "MD-MU-U": (1.0, 2.0),
    "BU-U-ST": (0.7, 2.0),
    "H-DN-ST": (0.7, 2.0),
    "DN-BD-BU": (0.7, 2.0),
    "DN-BU-U": (1.0, 2.0),
    "DN-U-U": (0.5, 2.0),   # WR 91.1%, PnL +115.1% ★★ TOP
    "ST-DN-U": (0.7, 2.0),
    "ST-DN-DN": (0.7, 2.0),
    "U-ST-DN": (0.7, 2.0),
    "U-U-ST": (0.7, 2.0),
    "ST-U-DN": (0.7, 2.0),
    "ST-ST-U": (0.7, 2.0),
}

# ============================================================
# Pattern Context Filters (v1.19.0)
# v1.19.0: All old filters removed (old patterns replaced with tight TP/SL patterns)
# New patterns don't have context filter research yet
# ============================================================

# Context filter configuration
PATTERN_CONTEXT_FILTERS = {}

# Context filter settings
CONTEXT_FILTER_ENABLED = True  # Master switch for context filters
CONTEXT_PREFERRED_BONUS = 0.10  # Confidence bonus for preferred context match

# RSI thresholds for zone classification
RSI_OVERSOLD_THRESHOLD = 30
RSI_OVERBOUGHT_THRESHOLD = 70

# Volatility quantiles (calculated dynamically from ATR)
VOL_LOW_QUANTILE = 0.33
VOL_HIGH_QUANTILE = 0.66

# Trend lookback period
TREND_LOOKBACK_BARS = 20


# ============================================================
# Classification Thresholds
# ============================================================

# Doji detection
DOJI_BODY_RATIO_THRESHOLD = 0.10  # body < 10% of range

# Dragonfly/Gravestone wick threshold
WICK_DOMINANCE_THRESHOLD = 0.70  # wick > 70% of range

# Marubozu (no wicks)
MARUBOZU_WICK_RATIO_THRESHOLD = 0.15  # total wicks < 15% of range

# Hammer/Inverted Hammer
HAMMER_WICK_TO_BODY_RATIO = 2.0  # wick > 2x body
HAMMER_OPPOSITE_WICK_RATIO = 0.3  # opposite wick < 0.3x body

# Spinning Top
SPINNING_TOP_BODY_NORM = 0.5  # normalized body < 0.5
SPINNING_TOP_WICK_RATIO = 0.5  # both wicks > 0.5x body

# Big vs Medium candle threshold
BIG_CANDLE_NORM_THRESHOLD = 1.5  # normalized body > 1.5

# Rolling average window
AVG_BODY_WINDOW = 20


# ============================================================
# Trading Parameters (v1.5 optimized from comprehensive research)
# Research: claudedocs/PATTERN_V14_COMPREHENSIVE_RESEARCH_20260124.md
# ============================================================

# TP/SL (v1.19.0: tight TP/SL for regime-independent edge)
DEFAULT_TP_PCT = 1.0  # v1.19.0: Tight TP (was 1.5)
DEFAULT_SL_PCT = 1.0  # v1.19.0: Tight SL (was 3.0)

# Double Exit (scale-out)
TP1_RATIO = 0.8   # First TP at 80% of full TP
TP1_QTY_PCT = 50  # Close 50% at TP1

# Leverage
DEFAULT_LEVERAGE = 3

# Position sizing
DEFAULT_POSITION_PCT = 5.0  # % of balance per trade


# ============================================================
# Early Exit Signal Configuration (v1.13)
# v1.13: confirm_candles 2→3 (baseline +125% → +146.7%, WF 4/5)
# Research: early_exit_deep_analysis.py (2026-01-25)
# Finding: 2-candle confirm HARMFUL (-71.2% vs baseline)
#          3-candle confirm BENEFICIAL (+21.7% vs baseline)
# ============================================================

EARLY_EXIT_CONFIG = {
    'enabled': True,
    # Bearish reversal types → exit LONG positions
    'bearish_types': ['BD'],  # Big Down only (most conservative)
    # Bullish reversal types → exit SHORT positions
    'bullish_types': ['BU'],  # Big Up only (most conservative)
    # Require N consecutive reversal candles before exit
    'confirm_candles': 3,  # v1.13: 2→3 (극단적 반전에서만 청산)
    # Minimum unrealized profit (%) required to trigger early exit
    'min_profit_pct': 0.3,
}

# Early exit type codes (from 12-type classification)
EARLY_EXIT_BEARISH_TYPES = EARLY_EXIT_CONFIG['bearish_types']
EARLY_EXIT_BULLISH_TYPES = EARLY_EXIT_CONFIG['bullish_types']
EARLY_EXIT_CONFIRM_CANDLES = EARLY_EXIT_CONFIG['confirm_candles']
EARLY_EXIT_MIN_PROFIT_PCT = EARLY_EXIT_CONFIG['min_profit_pct']


# ============================================================
# API and System
# ============================================================

# API cache TTL (seconds)
CACHE_TTL_TICKER = 5
CACHE_TTL_BALANCE = 5
CACHE_TTL_POSITIONS = 5

# Circuit breaker
CIRCUIT_BREAKER_THRESHOLD = 5  # consecutive failures
CIRCUIT_BREAKER_TIMEOUT = 60   # seconds

# Polling interval
CANDLE_POLL_INTERVAL = 60  # seconds (check every minute)

# Min data bars for classification
MIN_BARS_FOR_CLASSIFICATION = 25  # Need 20 for avg_body + 5 buffer


# ============================================================
# Logging
# ============================================================

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================
# TRADING PARAMETERS
# ============================================================
SLIPPAGE_BUFFER_PCT = 0.02
FEE_PCT = 0.05

# ============================================================
# PRICE TOLERANCE (for TP/SL detection)
# ============================================================
PRICE_TOLERANCE_PCT = 0.003
TP_LOWER_MULT = 0.999
TP_UPPER_MULT = 1.001
SL_LOWER_MULT = 0.999
SL_UPPER_MULT = 1.001

# ============================================================
# QUANTITY TOLERANCES
# ============================================================
QTY_TOLERANCE = 0.001
QTY_TOLERANCE_SCALEOUT = 0.0001
QTY_REDUCTION_THRESHOLD = 0.99

# ============================================================
# TIMING CONSTANTS
# ============================================================
CANDLE_DURATION_MS = 300000  # 5 minutes
CANDLE_SETTLE_SECONDS = 5  # v1.25.1: Reduced from 15 (BingX delivers BTC 5m in 2-3s)
DEFAULT_SLEEP_INTERVAL = 10  # v1.25.1: Error fallback only (candle-aligned loop replaces polling)
POSITION_CHECK_SLEEP = 30  # Deprecated: kept for backward compat

# Candle-aligned loop timing (v1.25.1)
TRADING_WINDOW_SECONDS = 30     # First 30s after candle close = trading window
POSITION_MONITOR_INTERVAL = 15  # Check position status every 15s during maintenance
MAX_MAINTENANCE_SLEEP = 120     # Max sleep when no position (maintenance window)
DAILY_LOSS_PAUSE_SECONDS = 300
ENTRY_PRICE_FETCH_DELAY = 0.5
EXIT_PRICE_FETCH_DELAY = 0.5
EXIT_PRICE_RETRY_DELAY = 0.5
EXIT_PRICE_INITIAL_DELAY = 1.0
MAX_EXIT_PRICE_RETRIES = 3

# ============================================================
# INTERVAL CONSTANTS
# ============================================================
# Iteration-based intervals (deprecated: kept for backward compat in tests)
TP_SL_CHECK_INTERVAL = 20
DEFAULT_HEALTH_CHECK_INTERVAL = 50
LOG_STATUS_INTERVAL = 10
MAX_OHLCV_CANDLES = 150  # v1.18.2: Increased from 100 for regime detection (needs 114+)
METRICS_SAVE_INTERVAL = 10
CACHE_TTL_SECONDS = 5

# Time-based intervals (v1.25.1: replaces iteration-based in main loop)
TP_SL_VERIFY_INTERVAL_SECONDS = 600   # Every 10 minutes
LOG_STATUS_INTERVAL_SECONDS = 300     # Every 5 minutes
METRICS_SAVE_INTERVAL_SECONDS = 300   # Every 5 minutes

# ============================================================
# BACKUP & STATE
# ============================================================
MAX_STATE_BACKUPS = 3
STATE_STALE_THRESHOLD_SECONDS = 300
POSITION_SYNC_INTERVAL_MINUTES = 5

# ============================================================
# CIRCUIT BREAKER DEFAULTS
# ============================================================
CB_FAILURE_THRESHOLD = 5
CB_RESET_TIMEOUT = 60.0

# Exponential backoff parameters
CB_INITIAL_TIMEOUT = 60.0      # 1st open: 60 seconds
CB_MAX_TIMEOUT = 600.0         # Max timeout: 600 seconds (10 min)
CB_BACKOFF_MULTIPLIER = 2.0    # 2x each failure

# ============================================================
# API RETRY DEFAULTS
# ============================================================
API_MAX_ATTEMPTS = 3
API_BASE_DELAY = 2
API_MAX_DELAY = 30

# ============================================================
# METRICS DEFAULTS (from v1.15 regime-validated backtest)
# ============================================================
EXPECTED_WIN_RATE = 76.0  # v1.19.0: Average across 15 tight TP/SL patterns
EXPECTED_AVG_WIN = 1.0    # Tight TP targets (0.3-1.0%)
EXPECTED_AVG_LOSS = 1.0   # Tight SL (0.3-1.0%)
EXPECTED_EDGE = 50.0      # Conservative estimate for tight TP/SL
METRICS_WINDOW_SIZE = 50
MIN_TRADES_FOR_COMPARISON = 5

# ============================================================
# PRECISION
# ============================================================
PRICE_ROUND_DECIMALS = 1
QUANTITY_ROUND_DECIMALS = 4

# ============================================================
# ROTATION SETTINGS
# ============================================================
ROTATION_ENABLED = False  # Disabled for pattern bot
ROTATION_MAX_SIZE = 1.0
ROTATION_MIN_PARTIAL_PCT = 0.6
ROTATION_REFILL_TO_FULL = True

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
        'tp_pct': 1.0,  # v1.19.0: Tight TP (was 1.5)
        'sl_pct': 1.0,  # v1.19.0: Tight SL (was 3.0)
        'cooldown_candles': 0,
        'long_patterns': VALIDATED_LONG_PATTERNS,
        'short_patterns': VALIDATED_SHORT_PATTERNS,
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