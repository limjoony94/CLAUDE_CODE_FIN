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
BOT_VERSION = "1.26.1"  # v1.26.1: T5_Optimized pruned portfolio (58 patterns, 35L+23S)
# R:R >= 0.75, MC < 0.01, WF >= 4/5, bias-validated (excess WR +22.6pp avg over random baseline)

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
# Validated Patterns (v1.26.1: T5_Optimized pruned portfolio)
# Source: portfolio_pruning_v4.py leave-one-out optimization
# Criteria: R:R >= 0.75, MC < 0.01, WF >= 4/5
# Validated: tp_sl_bias_research.py — 57/58 excess WR > random (avg +22.6pp)
# Portfolio: PnL +963.8%, MDD 19.8%, PnL/MDD 48.68x, PF 3.53, WF 5/5
# ============================================================

# LONG patterns (35) — v1.26.1 T5_Optimized
VALIDATED_LONG_PATTERNS: List[str] = [
    "BD-BD-U",     # R:R 1.00, 101t, WR 63.4%, Exp +1.10%, PnL +111.4%
    "BD-MU-BD",    # R:R 1.43,  23t, WR 69.6%, Exp +1.35%, PnL  +31.0%
    "BD-ST-U",     # R:R 1.00, 122t, WR 63.1%, Exp +1.08%, PnL +131.8%
    "BU-BU-BD",    # R:R 1.20,  38t, WR 65.8%, Exp +3.26%, PnL +123.7%
    "BU-U-GS",     # R:R 0.75,  21t, WR 81.0%, Exp +2.40%, PnL  +50.4%
    "D-MU-U",      # R:R 0.75,  35t, WR 80.0%, Exp +2.30%, PnL  +80.5%
    "DN-BD-BD",    # R:R 2.00, 107t, WR 48.6%, Exp +1.27%, PnL +136.3%
    "DN-DF-MU",    # R:R 1.00,  15t, WR 86.7%, Exp +3.20%, PnL  +48.0%
    "DN-DF-ST",    # R:R 1.33,  29t, WR 69.0%, Exp +2.64%, PnL  +76.6%
    "DN-DN-H",     # R:R 1.00, 117t, WR 63.2%, Exp +0.70%, PnL  +81.3%
    "DN-MD-DN",    # R:R 0.75, 184t, WR 68.5%, Exp +1.09%, PnL +200.6%
    "GS-ST-ST",    # R:R 0.75,  18t, WR 83.3%, Exp +2.65%, PnL  +47.7%
    "GS-U-MU",     # R:R 0.75,  18t, WR 83.3%, Exp +2.65%, PnL  +47.7%
    "H-BU-BU",     # R:R 1.50,  17t, WR 76.5%, Exp +2.64%, PnL  +44.8%
    "H-MU-MD",     # R:R 1.40,  29t, WR 69.0%, Exp +0.88%, PnL  +25.6%
    "IH-MD-MD",    # R:R 2.00,  21t, WR 66.7%, Exp +1.40%, PnL  +29.4%
    "IH-ST-MU",    # R:R 1.67,  25t, WR 68.0%, Exp +0.63%, PnL  +15.8%
    "MD-BU-MD",    # R:R 1.33,  17t, WR 76.5%, Exp +3.43%, PnL  +58.3%
    "MD-DN-MU",    # R:R 1.00, 130t, WR 62.3%, Exp +0.64%, PnL  +83.0%
    "MD-H-MD",     # R:R 2.00,  24t, WR 62.5%, Exp +1.21%, PnL  +29.1%
    "MD-MD-ST",    # R:R 1.50,  35t, WR 62.9%, Exp +1.61%, PnL  +56.5%
    "MD-ST-BD",    # R:R 2.00,  33t, WR 57.6%, Exp +0.99%, PnL  +32.7%
    "MD-ST-MD",    # R:R 1.00,  30t, WR 73.3%, Exp +2.70%, PnL  +81.0%
    "MU-BD-ST",    # R:R 0.83,  18t, WR 83.3%, Exp +4.65%, PnL  +83.7%
    "MU-DF-U",     # R:R 2.14,  19t, WR 73.7%, Exp +2.66%, PnL  +50.6%
    "MU-H-MU",     # R:R 2.14,  28t, WR 60.7%, Exp +1.81%, PnL  +50.6%
    "MU-IH-DN",    # R:R 1.33,  38t, WR 65.8%, Exp +2.31%, PnL  +87.7%
    "MU-MU-IH",    # R:R 1.00,  30t, WR 73.3%, Exp +2.00%, PnL  +60.0%
    "MU-U-H",      # R:R 0.83,  27t, WR 77.8%, Exp +3.73%, PnL +100.8%
    "U-H-MU",      # R:R 0.75,  38t, WR 84.2%, Exp +2.74%, PnL +104.2%
    "U-MD-GS",     # R:R 1.67,  23t, WR 69.6%, Exp +0.67%, PnL  +15.4%
    "U-MD-MD",     # R:R 2.14, 104t, WR 45.2%, Exp +0.78%, PnL  +81.4%
    "U-MU-H",      # R:R 4.00,  58t, WR 37.9%, Exp +1.25%, PnL  +72.2%
    "U-MU-IH",     # R:R10.00,  40t, WR 35.0%, Exp +2.47%, PnL  +98.6%
    "U-ST-DF",     # R:R 0.80,  21t, WR 81.0%, Exp +3.33%, PnL  +69.9%
]

# SHORT patterns (23) — v1.26.1 T5_Optimized
VALIDATED_SHORT_PATTERNS: List[str] = [
    "BD-BU-DN",    # R:R 1.00,  47t, WR 68.1%, Exp +3.16%, PnL +148.3%
    "BD-D-D",      # R:R 1.00,  15t, WR 86.7%, Exp +3.20%, PnL  +48.0%
    "BD-U-H",      # R:R 0.83,  20t, WR 85.0%, Exp +4.93%, PnL  +98.5%
    "BU-MD-MD",    # R:R 1.50,  16t, WR 75.0%, Exp +5.15%, PnL  +82.4%
    "BU-ST-GS",    # R:R 1.00,  15t, WR 86.7%, Exp +1.00%, PnL  +15.0%
    "D-BD-ST",     # R:R 0.83,  18t, WR 83.3%, Exp +4.65%, PnL  +83.7%
    "D-DN-DN",     # R:R 0.83,  99t, WR 67.7%, Exp +2.07%, PnL +204.6%
    "DN-BD-BU",    # R:R 0.83,  56t, WR 73.2%, Exp +2.98%, PnL +166.9%
    "DN-D-BD",     # R:R 8.33,  45t, WR 28.9%, Exp +1.43%, PnL  +64.2%
    "DN-DF-DN",    # R:R 1.00,  48t, WR 68.8%, Exp +2.15%, PnL +103.2%
    "DN-IH-U",     # R:R 0.83,  65t, WR 69.2%, Exp +2.32%, PnL +151.0%
    "GS-ST-U",     # R:R 1.40,  32t, WR 65.6%, Exp +0.76%, PnL  +24.4%
    "H-U-BD",      # R:R 1.50,  20t, WR 70.0%, Exp +4.40%, PnL  +88.0%
    "IH-ST-ST",    # R:R 0.80,  28t, WR 78.6%, Exp +3.01%, PnL  +84.2%
    "MD-MD-MD",    # R:R 1.00,  21t, WR 85.7%, Exp +6.33%, PnL +132.9%
    "MD-MU-U",     # R:R 0.75,  67t, WR 71.6%, Exp +1.42%, PnL  +95.3%
    "ST-BD-BU",    # R:R 1.00,  22t, WR 77.3%, Exp +4.81%, PnL +105.8%
    "ST-DN-BU",    # R:R 0.80,  62t, WR 72.6%, Exp +2.20%, PnL +136.3%
    "ST-DN-U",     # R:R 1.00, 234t, WR 60.7%, Exp +1.82%, PnL +426.6% ★ TOP
    "ST-MU-ST",    # R:R 0.80,  46t, WR 73.9%, Exp +2.38%, PnL +109.4%
    "U-GS-DN",     # R:R 1.00,  26t, WR 84.6%, Exp +6.13%, PnL +159.4%
    "U-H-BU",      # R:R 3.33,  42t, WR 45.2%, Exp +0.76%, PnL  +32.1%
    "U-ST-DN",     # R:R 1.00, 234t, WR 60.7%, Exp +1.82%, PnL +426.6%
]

# ============================================================
# Pattern Historical Statistics (for confidence calculation)
# Source: v1.26.1 — T5_Optimized (58 patterns, 35L+23S)
# ============================================================
PATTERN_STATS = {
    # LONG patterns (35)
    "BD-BD-U":   {"direction": "LONG",  "trades": 101, "wr": 63.4, "mc": 0.0065, "wf": 4, "periods": 2},
    "BD-MU-BD":  {"direction": "LONG",  "trades":  23, "wr": 69.6, "mc": 0.0086, "wf": 5, "periods": 3},
    "BD-ST-U":   {"direction": "LONG",  "trades": 122, "wr": 63.1, "mc": 0.0044, "wf": 5, "periods": 3},
    "BU-BU-BD":  {"direction": "LONG",  "trades":  38, "wr": 65.8, "mc": 0.0072, "wf": 5, "periods": 3},
    "BU-U-GS":   {"direction": "LONG",  "trades":  21, "wr": 81.0, "mc": 0.0120, "wf": 4, "periods": 3},
    "D-MU-U":    {"direction": "LONG",  "trades":  35, "wr": 80.0, "mc": 0.0016, "wf": 5, "periods": 3},
    "DN-BD-BD":  {"direction": "LONG",  "trades": 107, "wr": 48.6, "mc": 0.0025, "wf": 4, "periods": 3},
    "DN-DF-MU":  {"direction": "LONG",  "trades":  15, "wr": 86.7, "mc": 0.0026, "wf": 5, "periods": 3},
    "DN-DF-ST":  {"direction": "LONG",  "trades":  29, "wr": 69.0, "mc": 0.0044, "wf": 4, "periods": 3},
    "DN-DN-H":   {"direction": "LONG",  "trades": 117, "wr": 63.2, "mc": 0.0050, "wf": 5, "periods": 3},
    "DN-MD-DN":  {"direction": "LONG",  "trades": 184, "wr": 68.5, "mc": 0.0013, "wf": 4, "periods": 3},
    "GS-ST-ST":  {"direction": "LONG",  "trades":  18, "wr": 83.3, "mc": 0.0089, "wf": 5, "periods": 3},
    "GS-U-MU":   {"direction": "LONG",  "trades":  18, "wr": 83.3, "mc": 0.0102, "wf": 4, "periods": 3},
    "H-BU-BU":   {"direction": "LONG",  "trades":  17, "wr": 76.5, "mc": 0.0043, "wf": 4, "periods": 3},
    "H-MU-MD":   {"direction": "LONG",  "trades":  29, "wr": 69.0, "mc": 0.0053, "wf": 5, "periods": 3},
    "IH-MD-MD":  {"direction": "LONG",  "trades":  21, "wr": 66.7, "mc": 0.0060, "wf": 5, "periods": 2},
    "IH-ST-MU":  {"direction": "LONG",  "trades":  25, "wr": 68.0, "mc": 0.0089, "wf": 5, "periods": 2},
    "MD-BU-MD":  {"direction": "LONG",  "trades":  17, "wr": 76.5, "mc": 0.0054, "wf": 5, "periods": 3},
    "MD-DN-MU":  {"direction": "LONG",  "trades": 130, "wr": 62.3, "mc": 0.0073, "wf": 4, "periods": 3},
    "MD-H-MD":   {"direction": "LONG",  "trades":  24, "wr": 62.5, "mc": 0.0059, "wf": 4, "periods": 3},
    "MD-MD-ST":  {"direction": "LONG",  "trades":  35, "wr": 62.9, "mc": 0.0086, "wf": 5, "periods": 3},
    "MD-ST-BD":  {"direction": "LONG",  "trades":  33, "wr": 57.6, "mc": 0.0085, "wf": 5, "periods": 3},
    "MD-ST-MD":  {"direction": "LONG",  "trades":  30, "wr": 73.3, "mc": 0.0097, "wf": 4, "periods": 3},
    "MU-BD-ST":  {"direction": "LONG",  "trades":  18, "wr": 83.3, "mc": 0.0039, "wf": 5, "periods": 3},
    "MU-DF-U":   {"direction": "LONG",  "trades":  19, "wr": 73.7, "mc": 0.0016, "wf": 5, "periods": 3},
    "MU-H-MU":   {"direction": "LONG",  "trades":  28, "wr": 60.7, "mc": 0.0051, "wf": 5, "periods": 3},
    "MU-IH-DN":  {"direction": "LONG",  "trades":  38, "wr": 65.8, "mc": 0.0042, "wf": 4, "periods": 3},
    "MU-MU-IH":  {"direction": "LONG",  "trades":  30, "wr": 73.3, "mc": 0.0100, "wf": 4, "periods": 3},
    "MU-U-H":    {"direction": "LONG",  "trades":  27, "wr": 77.8, "mc": 0.0078, "wf": 4, "periods": 3},
    "U-H-MU":    {"direction": "LONG",  "trades":  38, "wr": 84.2, "mc": 0.0001, "wf": 4, "periods": 3},
    "U-MD-GS":   {"direction": "LONG",  "trades":  23, "wr": 69.6, "mc": 0.0057, "wf": 5, "periods": 3},
    "U-MD-MD":   {"direction": "LONG",  "trades": 104, "wr": 45.2, "mc": 0.0074, "wf": 4, "periods": 2},
    "U-MU-H":    {"direction": "LONG",  "trades":  58, "wr": 37.9, "mc": 0.0046, "wf": 5, "periods": 3},
    "U-MU-IH":   {"direction": "LONG",  "trades":  40, "wr": 35.0, "mc": 0.0007, "wf": 4, "periods": 3},
    "U-ST-DF":   {"direction": "LONG",  "trades":  21, "wr": 81.0, "mc": 0.0051, "wf": 4, "periods": 3},
    # SHORT patterns (23)
    "BD-BU-DN":  {"direction": "SHORT", "trades":  47, "wr": 68.1, "mc": 0.0090, "wf": 4, "periods": 3},
    "BD-D-D":    {"direction": "SHORT", "trades":  15, "wr": 86.7, "mc": 0.0040, "wf": 4, "periods": 3},
    "BD-U-H":    {"direction": "SHORT", "trades":  20, "wr": 85.0, "mc": 0.0013, "wf": 5, "periods": 3},
    "BU-MD-MD":  {"direction": "SHORT", "trades":  16, "wr": 75.0, "mc": 0.0064, "wf": 5, "periods": 3},
    "BU-ST-GS":  {"direction": "SHORT", "trades":  15, "wr": 86.7, "mc": 0.0032, "wf": 4, "periods": 3},
    "D-BD-ST":   {"direction": "SHORT", "trades":  18, "wr": 83.3, "mc": 0.0040, "wf": 5, "periods": 2},
    "D-DN-DN":   {"direction": "SHORT", "trades":  99, "wr": 67.7, "mc": 0.0059, "wf": 5, "periods": 3},
    "DN-BD-BU":  {"direction": "SHORT", "trades":  56, "wr": 73.2, "mc": 0.0017, "wf": 5, "periods": 3},
    "DN-D-BD":   {"direction": "SHORT", "trades":  45, "wr": 28.9, "mc": 0.0089, "wf": 5, "periods": 3},
    "DN-DF-DN":  {"direction": "SHORT", "trades":  48, "wr": 68.8, "mc": 0.0074, "wf": 5, "periods": 3},
    "DN-IH-U":   {"direction": "SHORT", "trades":  65, "wr": 69.2, "mc": 0.0084, "wf": 4, "periods": 2},
    "GS-ST-U":   {"direction": "SHORT", "trades":  32, "wr": 65.6, "mc": 0.0119, "wf": 4, "periods": 3},
    "H-U-BD":    {"direction": "SHORT", "trades":  20, "wr": 70.0, "mc": 0.0084, "wf": 5, "periods": 3},
    "IH-ST-ST":  {"direction": "SHORT", "trades":  28, "wr": 78.6, "mc": 0.0061, "wf": 5, "periods": 3},
    "MD-MD-MD":  {"direction": "SHORT", "trades":  21, "wr": 85.7, "mc": 0.0007, "wf": 4, "periods": 3},
    "MD-MU-U":   {"direction": "SHORT", "trades":  67, "wr": 71.6, "mc": 0.0110, "wf": 5, "periods": 3},
    "ST-BD-BU":  {"direction": "SHORT", "trades":  22, "wr": 77.3, "mc": 0.0095, "wf": 4, "periods": 3},
    "ST-DN-BU":  {"direction": "SHORT", "trades":  62, "wr": 72.6, "mc": 0.0032, "wf": 4, "periods": 2},
    "ST-DN-U":   {"direction": "SHORT", "trades": 234, "wr": 60.7, "mc": 0.0018, "wf": 4, "periods": 2},
    "ST-MU-ST":  {"direction": "SHORT", "trades":  46, "wr": 73.9, "mc": 0.0056, "wf": 5, "periods": 3},
    "U-GS-DN":   {"direction": "SHORT", "trades":  26, "wr": 84.6, "mc": 0.0003, "wf": 5, "periods": 3},
    "U-H-BU":    {"direction": "SHORT", "trades":  42, "wr": 45.2, "mc": 0.0076, "wf": 5, "periods": 3},
    "U-ST-DN":   {"direction": "SHORT", "trades": 234, "wr": 60.7, "mc": 0.0010, "wf": 4, "periods": 2},
}

# Confidence calculation weights
CONFIDENCE_WEIGHT_CLARITY = 0.40      # Candle classification clarity
CONFIDENCE_WEIGHT_HISTORICAL = 0.30   # Historical pattern win rate
CONFIDENCE_WEIGHT_REGIME = 0.30       # DEPRECATED: Regime disabled since v1.19.0 (placeholder)

# Confidence logging file
CONFIDENCE_LOG_FILE = os.path.join(PROJECT_ROOT, "results", "pattern_5m_confidence_log.csv")


# ============================================================
# Pattern TP/SL (v1.26.1 T5_Optimized Per-Pattern)
# All R:R >= 0.75, bias-validated against random baseline
# ============================================================
PATTERN_OPTIMAL_TPSL = {
    # LONG patterns (35) — format: (tp_pct, sl_pct)
    "BD-BD-U":   (1.5, 1.5),  # R:R=1.00
    "BD-MU-BD":  (1.0, 0.7),  # R:R=1.43
    "BD-ST-U":   (1.5, 1.5),  # R:R=1.00
    "BU-BU-BD":  (3.0, 2.5),  # R:R=1.20
    "BU-U-GS":   (1.5, 2.0),  # R:R=0.75
    "D-MU-U":    (1.5, 2.0),  # R:R=0.75
    "DN-BD-BD":  (2.0, 1.0),  # R:R=2.00
    "DN-DF-MU":  (1.5, 1.5),  # R:R=1.00
    "DN-DF-ST":  (2.0, 1.5),  # R:R=1.33
    "DN-DN-H":   (1.0, 1.0),  # R:R=1.00
    "DN-MD-DN":  (1.5, 2.0),  # R:R=0.75
    "GS-ST-ST":  (1.5, 2.0),  # R:R=0.75
    "GS-U-MU":   (1.5, 2.0),  # R:R=0.75
    "H-BU-BU":   (1.5, 1.0),  # R:R=1.50
    "H-MU-MD":   (0.7, 0.5),  # R:R=1.40
    "IH-MD-MD":  (1.0, 0.5),  # R:R=2.00
    "IH-ST-MU":  (0.5, 0.3),  # R:R=1.67
    "MD-BU-MD":  (2.0, 1.5),  # R:R=1.33
    "MD-DN-MU":  (1.0, 1.0),  # R:R=1.00
    "MD-H-MD":   (1.0, 0.5),  # R:R=2.00
    "MD-MD-ST":  (1.5, 1.0),  # R:R=1.50
    "MD-ST-BD":  (1.0, 0.5),  # R:R=2.00
    "MD-ST-MD":  (2.0, 2.0),  # R:R=1.00
    "MU-BD-ST":  (2.5, 3.0),  # R:R=0.83
    "MU-DF-U":   (1.5, 0.7),  # R:R=2.14
    "MU-H-MU":   (1.5, 0.7),  # R:R=2.14
    "MU-IH-DN":  (2.0, 1.5),  # R:R=1.33
    "MU-MU-IH":  (1.5, 1.5),  # R:R=1.00
    "MU-U-H":    (2.5, 3.0),  # R:R=0.83
    "U-H-MU":    (1.5, 2.0),  # R:R=0.75
    "U-MD-GS":   (0.5, 0.3),  # R:R=1.67
    "U-MD-MD":   (1.5, 0.7),  # R:R=2.14
    "U-MU-H":    (2.0, 0.5),  # R:R=4.00
    "U-MU-IH":   (3.0, 0.3),  # R:R=10.0
    "U-ST-DF":   (2.0, 2.5),  # R:R=0.80
    # SHORT patterns (23)
    "BD-BU-DN":  (3.0, 3.0),  # R:R=1.00
    "BD-D-D":    (1.5, 1.5),  # R:R=1.00
    "BD-U-H":    (2.5, 3.0),  # R:R=0.83
    "BU-MD-MD":  (3.0, 2.0),  # R:R=1.50
    "BU-ST-GS":  (0.5, 0.5),  # R:R=1.00
    "D-BD-ST":   (2.5, 3.0),  # R:R=0.83
    "D-DN-DN":   (2.5, 3.0),  # R:R=0.83
    "DN-BD-BU":  (2.5, 3.0),  # R:R=0.83
    "DN-D-BD":   (2.5, 0.3),  # R:R=8.33
    "DN-DF-DN":  (2.0, 2.0),  # R:R=1.00
    "DN-IH-U":   (2.5, 3.0),  # R:R=0.83
    "GS-ST-U":   (0.7, 0.5),  # R:R=1.40
    "H-U-BD":    (3.0, 2.0),  # R:R=1.50
    "IH-ST-ST":  (2.0, 2.5),  # R:R=0.80
    "MD-MD-MD":  (3.0, 3.0),  # R:R=1.00
    "MD-MU-U":   (1.5, 2.0),  # R:R=0.75
    "ST-BD-BU":  (3.0, 3.0),  # R:R=1.00
    "ST-DN-BU":  (2.0, 2.5),  # R:R=0.80
    "ST-DN-U":   (3.0, 3.0),  # R:R=1.00
    "ST-MU-ST":  (2.0, 2.5),  # R:R=0.80
    "U-GS-DN":   (3.0, 3.0),  # R:R=1.00
    "U-H-BU":    (1.0, 0.3),  # R:R=3.33
    "U-ST-DN":   (3.0, 3.0),  # R:R=1.00
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

# DEPRECATED: Individual TTL constants unused — use CACHE_TTL_SECONDS instead
# CACHE_TTL_TICKER = 5
# CACHE_TTL_BALANCE = 5
# CACHE_TTL_POSITIONS = 5

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
POSITION_CHECK_SLEEP = 30  # DEPRECATED (v1.25.1): candle-aligned loop replaces polling

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
# DEPRECATED: Iteration-based intervals replaced by time-based (v1.25.1).
# Kept only for DEFAULT_CONFIG and backward compat.
TP_SL_CHECK_INTERVAL = 20           # DEPRECATED: use TP_SL_VERIFY_INTERVAL_SECONDS
DEFAULT_HEALTH_CHECK_INTERVAL = 50  # Used in DEFAULT_CONFIG
LOG_STATUS_INTERVAL = 10            # DEPRECATED: use LOG_STATUS_INTERVAL_SECONDS
MAX_OHLCV_CANDLES = 150  # v1.18.2: Increased from 100 for regime detection (needs 114+)
METRICS_SAVE_INTERVAL = 10          # DEPRECATED: use METRICS_SAVE_INTERVAL_SECONDS
CACHE_TTL_SECONDS = 5               # Used in models.py APICache

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
EXPECTED_WIN_RATE = 73.5  # v1.26.1: T5_Optimized 58-pattern portfolio avg WR
EXPECTED_AVG_WIN = 5.38   # R:R >= 0.75, TP range 0.5-3.0%
EXPECTED_AVG_LOSS = 4.24  # SL range 0.7-2.0%
EXPECTED_EDGE = 50.0      # Conservative estimate
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