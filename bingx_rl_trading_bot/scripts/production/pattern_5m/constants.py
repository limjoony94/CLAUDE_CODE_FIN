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
BOT_VERSION = "1.17.0"  # v1.17: Remove D-DN-BD (only 6 trades, p=1.0 - statistically invalid)

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
# Validated Patterns (v1.17 Statistical Validation)
# v1.17: production_pattern_validation.py (2026-01-26)
# Changes: Removed D-DN-BD (only 6 trades, p=1.0 - statistically invalid)
# Criteria: WR >= 60%, WF >= 4/5, Trades >= 10, p < 0.10 or Edge > 0
# Report: claudedocs/PRODUCTION_VALIDATION_REPORT_20260126.md
# ============================================================

# LONG patterns - v1.16: 8 total (2 existing + 6 new)
VALIDATED_LONG_PATTERNS: List[str] = [
    # Existing v1.15
    "U-BU-U",     # WR 93.8%, 32 trades, WF 4/5
    "ST-BD-DN",   # WR 70.6%, 34 trades, WF 3/5
    # NEW v1.16 - Pattern Discovery
    "DN-DN-DN",   # WR 87.4%, 87 trades, WF 5/5 - Mean Reversion
    "DN-U-U",     # WR 87.3%, 79 trades, WF 5/5 - Trend Confirmation
    "DN-DN-U",    # WR 83.1%, 89 trades, WF 4/5 - Reversal
    "DN-ST-U",    # WR 93.6%, 47 trades, WF 5/5 - Support Bounce
    "U-ST-U",     # WR 89.8%, 49 trades, WF 5/5 - Consolidation Break
    "U-U-U",      # WR 92.1%, 38 trades, WF 5/5 - Momentum
]

# SHORT patterns - v1.17: 10 total (removed D-DN-BD)
VALIDATED_SHORT_PATTERNS: List[str] = [
    # Existing v1.15
    "BD-BD-BD",   # WR 84.6%, 13 trades, WF 5/5, p=0.002
    "DN-DN-BD",   # WR 89.5%, 38 trades, WF 4/5, p<0.001
    "MU-ST-DN",   # WR 93.9%, 33 trades, WF 5/5, p<0.001
    "IH-DN-DN",   # WR 88.2%, 17 trades, WF 4/5, p=0.072
    "BD-ST-DN",   # WR 92.9%, 14 trades, WF 5/5, p=0.002
    "BU-U-DN",    # WR 83.3%, 36 trades, WF 4/5, p=0.002
    # REMOVED v1.17: "D-DN-BD" - only 6 trades, p=1.0 (statistically invalid)
    # NEW v1.16 - Pattern Discovery
    "U-DN-DN",    # WR 90.1%, 172 trades, WF 4/5, p<0.001 - Reversal Confirmation
    "DN-U-DN",    # WR 75.8%, 66 trades, WF 4/5, p=0.003 - Lower High
    "DN-DN-ST",   # WR 83.0%, 53 trades, WF 5/5, p=0.002 - Continuation
    "U-U-DN",     # WR 74.0%, 77 trades, WF 4/5, p=0.005 - Exhaustion
]

# ============================================================
# Pattern Historical Statistics (for confidence calculation)
# Source: v1.17 production validation (2026-01-26)
# ============================================================
PATTERN_STATS = {
    # LONG patterns (8)
    "U-BU-U": {"wr": 0.704, "count": 27, "avg_conf": 0.700},
    "ST-BD-DN": {"wr": 0.909, "count": 11, "avg_conf": 0.850},
    "DN-DN-DN": {"wr": 0.878, "count": 148, "avg_conf": 0.850},
    "DN-U-U": {"wr": 0.800, "count": 145, "avg_conf": 0.800},
    "DN-DN-U": {"wr": 0.834, "count": 145, "avg_conf": 0.820},
    "DN-ST-U": {"wr": 0.851, "count": 94, "avg_conf": 0.830},
    "U-ST-U": {"wr": 0.847, "count": 85, "avg_conf": 0.830},
    "U-U-U": {"wr": 0.719, "count": 89, "avg_conf": 0.720},
    # SHORT patterns (10) - D-DN-BD REMOVED v1.17
    "BD-BD-BD": {"wr": 0.846, "count": 13, "avg_conf": 0.820},
    "DN-DN-BD": {"wr": 0.895, "count": 38, "avg_conf": 0.860},
    "MU-ST-DN": {"wr": 0.939, "count": 33, "avg_conf": 0.900},
    "IH-DN-DN": {"wr": 0.882, "count": 17, "avg_conf": 0.850},
    "BD-ST-DN": {"wr": 0.929, "count": 14, "avg_conf": 0.890},
    "BU-U-DN": {"wr": 0.833, "count": 36, "avg_conf": 0.810},
    # REMOVED v1.17: "D-DN-BD" - only 6 trades, p=1.0
    "U-DN-DN": {"wr": 0.901, "count": 172, "avg_conf": 0.870},
    "DN-U-DN": {"wr": 0.758, "count": 66, "avg_conf": 0.760},
    "DN-DN-ST": {"wr": 0.830, "count": 53, "avg_conf": 0.810},
    "U-U-DN": {"wr": 0.740, "count": 77, "avg_conf": 0.750},
}

# Confidence calculation weights
CONFIDENCE_WEIGHT_CLARITY = 0.40      # Candle classification clarity
CONFIDENCE_WEIGHT_HISTORICAL = 0.30   # Historical pattern win rate
CONFIDENCE_WEIGHT_REGIME = 0.30       # Regime alignment (placeholder)

# Confidence logging file
CONFIDENCE_LOG_FILE = "results/pattern_5m_confidence_log.csv"


# ============================================================
# Pattern-Specific Optimal TP/SL (v1.17)
# v1.17: Removed D-DN-BD (statistically invalid - only 6 trades, p=1.0)
# Research: production_pattern_validation.py (2026-01-26)
# Methodology: Grid search + WF 5-fold validation + statistical significance testing
# ============================================================
PATTERN_OPTIMAL_TPSL = {
    # LONG patterns (8)
    'U-BU-U': (1.5, 2.0),     # WR 70.4%, 27 trades, WF 4/5, p=0.091
    'ST-BD-DN': (2.0, 3.0),   # WR 90.9%, 11 trades, WF 4/5, p=0.004
    'DN-DN-DN': (1.0, 3.0),   # WR 87.8%, 148 trades, WF 5/5, p<0.001 ⭐
    'DN-U-U': (1.0, 3.0),     # WR 80.0%, 145 trades, WF 5/5, p=0.107
    'DN-DN-U': (1.0, 3.0),    # WR 83.4%, 145 trades, WF 4/5, p=0.008
    'DN-ST-U': (1.0, 3.0),    # WR 85.1%, 94 trades, WF 5/5, p=0.007
    'U-ST-U': (1.0, 3.0),     # WR 84.7%, 85 trades, WF 5/5, p=0.013
    'U-U-U': (1.5, 3.0),      # WR 71.9%, 89 trades, WF 4/5, p=0.175

    # SHORT patterns (10) - D-DN-BD REMOVED v1.17
    'BD-BD-BD': (3.0, 2.5),   # WR 84.6%, 13 trades, WF 5/5, p=0.002
    'DN-DN-BD': (1.5, 3.0),   # WR 89.5%, 38 trades, WF 4/5, p<0.001 ⭐
    'MU-ST-DN': (1.0, 2.5),   # WR 93.9%, 33 trades, WF 5/5, p<0.001 ⭐
    'IH-DN-DN': (1.0, 3.0),   # WR 88.2%, 17 trades, WF 4/5, p=0.072
    'BD-ST-DN': (1.5, 3.0),   # WR 92.9%, 14 trades, WF 5/5, p=0.002
    'BU-U-DN': (1.5, 2.5),    # WR 83.3%, 36 trades, WF 4/5, p=0.002
    # REMOVED v1.17: 'D-DN-BD': (2.5, 2.0), - only 6 trades, p=1.0 (statistically invalid)
    'U-DN-DN': (1.0, 3.0),    # WR 90.1%, 172 trades, WF 4/5, p<0.001 ⭐
    'DN-U-DN': (2.0, 3.0),    # WR 75.8%, 66 trades, WF 4/5, p=0.003
    'DN-DN-ST': (1.5, 3.0),   # WR 83.0%, 53 trades, WF 5/5, p=0.002
    'U-U-DN': (2.0, 3.0),     # WR 74.0%, 77 trades, WF 4/5, p=0.005
}


# ============================================================
# Pattern Context Filters (v1.14)
# v1.14: Added MU-ST-DN and BD-BD-BD filters from context research
# Research: pattern_context_comprehensive_research.py (2026-01-26)
# ============================================================

# Context filter configuration (v1.17)
# 'required': Must match to take signal (strict filter)
# 'preferred': Adds confidence bonus if matched (soft filter)
# 'excluded': Must NOT match to take signal (exclusion filter)
PATTERN_CONTEXT_FILTERS = {
    # v1.14 filters
    'DN-DN-BD': {
        'required': {'rsi_zone': ['OS']},  # RSI_OS filter, WR +43%
    },
    'U-BU-U': {
        'preferred': {'trend': ['DN']},  # Bonus if downtrend
    },
    'IH-DN-DN': {
        'excluded': {'vol': ['H']},  # Avoid high volatility, WR +23%
    },
    'MU-ST-DN': {
        'preferred': {'position_zone': ['L']},  # +36.2% WR improvement at low position
    },
    'BD-BD-BD': {
        'preferred': {'session': ['ASIA']},  # +29.8% WR improvement in Asia session
    },
    # REMOVED v1.17: 'D-DN-BD' filter (pattern itself removed due to insufficient trades)
}

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

# TP/SL (v1.5 optimized: WR 64.7% → 87.6%)
DEFAULT_TP_PCT = 1.5  # Changed from 2.5 (smaller TP = more frequent hits)
DEFAULT_SL_PCT = 3.0  # Changed from 2.0 (wider SL = noise filtering)

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
MAX_OHLCV_CANDLES = 100
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
EXPECTED_WIN_RATE = 88.0  # From v1.15 validation (tighter TP, wider SL)
EXPECTED_AVG_WIN = 1.5    # Smaller TP targets hit more frequently
EXPECTED_AVG_LOSS = 2.8   # Wider SL prevents premature stop-outs
EXPECTED_EDGE = 85.0      # Conservative estimate based on regime-adjusted PnL
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
        'tp_pct': 1.5,  # v1.15: Tighter TP for higher WR
        'sl_pct': 3.0,  # v1.15: Wider SL to filter market noise
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
