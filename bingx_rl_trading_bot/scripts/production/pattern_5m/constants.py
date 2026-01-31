"""
Pattern 5m Bot - Constants and Magic Numbers
All configurable values and pattern definitions.
"""

from enum import Enum
from typing import List

# ============================================================
# BOT IDENTIFICATION
# ============================================================
BOT_NAME = "pattern_5m_bot"
BOT_VERSION = "1.20.1"  # v1.20.1: Improved early-bar classification (default avg_body_20=1.0)
# 13 patterns (7L+6S), 226t, WR 73.5%, MDD 14.6%, PF 2.59, WF 5/5, MC=0.0000

# ============================================================
# FILE PATHS (relative to bot root)
# ============================================================
CONFIG_FILE = "config/pattern_5m_config.yaml"
STATE_FILE = "results/pattern_5m_bot_state.json"
LOCK_FILE = "results/pattern_5m_bot.lock"
METRICS_FILE = "results/pattern_5m_metrics.json"
LOG_DIR = "logs"
API_KEYS_FILE = "config/api_keys.yaml"

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
REGIME_DETECTION_ENABLED = False  # v1.19.0: Disabled - tight TP/SL is regime-independent
REGIME_LOOKBACK_BARS = 100          # 100 bars = ~8.3 hours
REGIME_TREND_THRESHOLD = 1.5        # % price change for BULL/BEAR (optimized from 2.0)
REGIME_VOL_THRESHOLD = 0.15         # ATR% threshold for HIGH/LOW volatility

# Regime-specific patterns and TP/SL (v1.18)
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
DEFAULT_REGIME = MarketRegime.SIDEWAYS

# ============================================================
# Validated Patterns (v1.19.1 Tight TP/SL)
# Research: tight_tpsl_validation.py (2026-01-30)
# 270-day validation: 66 validated → 21 selected (10 LONG + 11 SHORT)
# Criteria: excess_wr > 20%, MC < 0.05, WF >= 3/5, trades >= 15, 3/3 periods profitable
# Key insight: Tight TP/SL (0.3-1.0%) eliminates regime dependency, enables genuine SHORT edge
# v1.19.1: +6 Tier 1 patterns (3/3 periods, excess>27%) → +290% PnL, PnL/MDD 80→105
# ============================================================

# LONG patterns - v1.19.1: 10 total (tight TP/SL, regime-independent)
VALIDATED_LONG_PATTERNS: List[str] = [
    # v1.20.1: Validated with improved classification (default avg_body_20=1.0 for early bars)
    # Tier 1: WF>=4, MC<0.01, excess>15, uniform 1.0/1.0 TP/SL
    "U-MU-H",     # WR 68.4%, 57t, excess +16.7%, MC=0.0036, WF 4/5, PP 3/3
    "MD-ST-MD",   # WR 70.8%, 48t, excess +19.1%, MC=0.0020, WF 4/5, PP 3/3
    "GS-U-BD",    # WR 76.0%, 25t, excess +24.3%, MC=0.0073, WF 4/5, PP 2/3
    "MD-MD-ST",   # WR 71.1%, 38t, excess +19.3%, MC=0.0055, WF 5/5, PP 3/3
    "BU-IH-DN",   # WR 76.0%, 25t, excess +24.3%, MC=0.0065, WF 4/5, PP 3/3
    "MD-H-MD",    # WR 83.3%, 18t, excess +31.6%, MC=0.0038, WF 5/5, PP 3/3
    "IH-MD-MD",   # WR 86.7%, 15t, excess +34.9%, MC=0.0043, WF 4/5, PP 2/3
]

# SHORT patterns - v1.20.0: Tier 1.5 (WF>=4, MC<0.03, excess>15)
VALIDATED_SHORT_PATTERNS: List[str] = [
    "DN-D-BD",    # WR 67.4%, 46t, excess +19.1%, MC=0.0131, WF 5/5, PP 3/3
    "BD-U-GS",    # WR 76.5%, 17t, excess +28.2%, MC=0.0259, WF 4/5, PP 3/3
    "DN-GS-H",    # WR 80.0%, 15t, excess +31.7%, MC=0.0165, WF 4/5, PP 2/3
    "U-DF-BU",    # WR 76.5%, 17t, excess +28.2%, MC=0.0258, WF 4/5, PP 2/3
    "BD-GS-BD",   # WR 76.5%, 17t, excess +28.2%, MC=0.0269, WF 4/5, PP 3/3
    "DN-IH-IH",   # WR 80.0%, 15t, excess +31.7%, MC=0.0164, WF 5/5, PP 3/3
]

# ============================================================
# Pattern Historical Statistics (for confidence calculation)
# Source: v1.20.0 unified_pattern_discovery.py (production classification)
# ============================================================
PATTERN_STATS = {
    # LONG patterns (7) - Tier 1 (WF>=4, MC<0.01, excess>15)
    "U-MU-H":   {"wr": 0.684, "count": 57, "avg_conf": 0.680},
    "MD-ST-MD": {"wr": 0.708, "count": 48, "avg_conf": 0.710},
    "GS-U-BD":  {"wr": 0.760, "count": 25, "avg_conf": 0.760},
    "MD-MD-ST": {"wr": 0.711, "count": 38, "avg_conf": 0.710},
    "BU-IH-DN": {"wr": 0.760, "count": 25, "avg_conf": 0.760},
    "MD-H-MD":  {"wr": 0.833, "count": 18, "avg_conf": 0.830},
    "IH-MD-MD": {"wr": 0.867, "count": 15, "avg_conf": 0.870},
    # SHORT patterns (6) - Tier 1.5 (WF>=4, MC<0.03, excess>15)
    "DN-D-BD":  {"wr": 0.674, "count": 46, "avg_conf": 0.670},
    "BD-U-GS":  {"wr": 0.765, "count": 17, "avg_conf": 0.770},
    "DN-GS-H":  {"wr": 0.800, "count": 15, "avg_conf": 0.800},
    "U-DF-BU":  {"wr": 0.765, "count": 17, "avg_conf": 0.770},
    "BD-GS-BD": {"wr": 0.765, "count": 17, "avg_conf": 0.770},
    "DN-IH-IH": {"wr": 0.800, "count": 15, "avg_conf": 0.800},
}

# Confidence calculation weights
CONFIDENCE_WEIGHT_CLARITY = 0.40      # Candle classification clarity
CONFIDENCE_WEIGHT_HISTORICAL = 0.30   # Historical pattern win rate
CONFIDENCE_WEIGHT_REGIME = 0.30       # Regime alignment (placeholder)

# Confidence logging file
CONFIDENCE_LOG_FILE = "results/pattern_5m_confidence_log.csv"


# ============================================================
# Pattern TP/SL (v1.20.0 Uniform)
# Research: unified_pattern_discovery.py (production-consistent classification)
# v1.20.0: Uniform 1.0/1.0 TP/SL, 13 patterns (7L+6S)
# Validated: 353t, WR 73.7%, MDD 14.8%, PF 2.62, WF 5/5, MC=0.0000
# ============================================================
PATTERN_OPTIMAL_TPSL = {
    # LONG patterns (7)
    'U-MU-H':   (1.0, 1.0),
    'MD-ST-MD': (1.0, 1.0),
    'GS-U-BD':  (1.0, 1.0),
    'MD-MD-ST': (1.0, 1.0),
    'BU-IH-DN': (1.0, 1.0),
    'MD-H-MD':  (1.0, 1.0),
    'IH-MD-MD': (1.0, 1.0),
    # SHORT patterns (6)
    'DN-D-BD':  (1.0, 1.0),
    'BD-U-GS':  (1.0, 1.0),
    'DN-GS-H':  (1.0, 1.0),
    'U-DF-BU':  (1.0, 1.0),
    'BD-GS-BD': (1.0, 1.0),
    'DN-IH-IH': (1.0, 1.0),
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
CANDLE_SETTLE_SECONDS = 15
DEFAULT_SLEEP_INTERVAL = 30
POSITION_CHECK_SLEEP = 30
DAILY_LOSS_PAUSE_SECONDS = 300
ENTRY_PRICE_FETCH_DELAY = 0.5
EXIT_PRICE_FETCH_DELAY = 0.5
EXIT_PRICE_RETRY_DELAY = 0.5
EXIT_PRICE_INITIAL_DELAY = 1.0
MAX_EXIT_PRICE_RETRIES = 3

# ============================================================
# INTERVAL CONSTANTS
# ============================================================
TP_SL_CHECK_INTERVAL = 20
DEFAULT_HEALTH_CHECK_INTERVAL = 50
LOG_STATUS_INTERVAL = 10
MAX_OHLCV_CANDLES = 150  # v1.18.2: Increased from 100 for regime detection (needs 114+)
METRICS_SAVE_INTERVAL = 10
CACHE_TTL_SECONDS = 5

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