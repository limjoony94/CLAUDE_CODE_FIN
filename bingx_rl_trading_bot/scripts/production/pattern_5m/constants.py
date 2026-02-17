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
BOT_VERSION = "1.28.23"  # v1.28.23: extend EXCHANGE_MANAGED to initial TP/SL placement (recovery path)
# Base: v1.27.1 + low-WR pattern review (low_wr_pattern_review.py)
# Result: PnL +966%, WR 84.9%, MDD 16.2%, PnL/MDD 59.6x, portfolio MC p=0.0000
# Changes: U-H-BU removed (SL 0.3% < 0.5% min, effective SL 0.23% after spread/slippage)

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
DYNAMIC_PATTERNS_FILE = os.path.join(PROJECT_ROOT, "results", "dynamic_patterns.json")
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
# Validated Patterns (v1.27.0: Uniform TP 70%)
# Base: v1.26.4 deep-validated patterns (52 = 32L+20S)
# Change: All TP * 0.7, SL unchanged
# Research: uniform_tp_validation.py (8-phase deep validation)
#   - Portfolio MC: p=0.0000, WF: 5/5
#   - Edge preserved: avg +18.7pp (all patterns >= +7.1pp)
#   - MDD P99: 47.2% → 39.9%, max consec loss: 3 → 2
# Portfolio: PnL +911.1%, WR 83.7%, MDD 16.2%, PF 3.62, PnL/MDD 56.2x
# ============================================================

# LONG patterns (32) — v1.26.4 full TP/SL optimization (23 LONG optimized)
VALIDATED_LONG_PATTERNS: List[str] = [
    "BD-BD-U",     # R:R 0.67,  72t, WR 76.4%, Exp +2.36%, PnL +169.8%
    "BD-MU-BD",    # R:R 1.43,  23t, WR 69.6%, Exp +1.35%, PnL  +31.0%
    "BD-ST-U",     # R:R 1.00, 106t, WR 63.2%, Exp +1.49%, PnL +157.4%
    "BU-BU-BD",    # R:R 1.00,  35t, WR 71.4%, Exp +3.76%, PnL +131.5%
    "D-MU-U",      # R:R 0.75,  35t, WR 80.0%, Exp +2.30%, PnL  +80.5%
    "DN-BD-BD",    # R:R 1.33, 103t, WR 59.2%, Exp +1.62%, PnL +166.7%
    "DN-DF-MU",    # R:R 0.50,  14t, WR 92.9%, Exp +3.44%, PnL  +48.1%
    "DN-DF-ST",    # R:R 0.67,  24t, WR 83.3%, Exp +3.40%, PnL  +81.6%
    "DN-DN-H",     # R:R 0.40, 103t, WR 83.5%, Exp +1.17%, PnL +120.2%
    "DN-MD-DN",    # R:R 0.75, 184t, WR 68.5%, Exp +1.09%, PnL +200.6%
    "GS-ST-ST",    # R:R 0.75,  18t, WR 83.3%, Exp +2.65%, PnL  +47.7%
    "H-BU-BU",     # R:R 0.67,  11t, WR 100 %, Exp +5.90%, PnL  +64.9%
    "H-MU-MD",     # R:R 0.50,  23t, WR 91.3%, Exp +2.12%, PnL  +48.7%
    "IH-MD-MD",    # R:R 0.50,  19t, WR 94.7%, Exp +2.43%, PnL  +46.1%
    "IH-ST-MU",    # R:R 0.20,  24t, WR 100 %, Exp +1.40%, PnL  +33.6%
    "MD-BU-MD",    # R:R 0.80,  16t, WR 87.5%, Exp +4.21%, PnL  +67.4%
    "MD-DN-MU",    # R:R 1.00, 130t, WR 62.3%, Exp +0.64%, PnL  +83.0%
    "MD-H-MD",     # R:R 1.43,  24t, WR 70.8%, Exp +1.41%, PnL  +33.9%
    "MD-MD-ST",    # R:R 0.75,  30t, WR 80.0%, Exp +2.30%, PnL  +69.0%
    "MD-ST-BD",    # R:R 2.00,  33t, WR 57.6%, Exp +0.99%, PnL  +32.7%
    "MD-ST-MD",    # R:R 1.00,  30t, WR 73.3%, Exp +2.70%, PnL  +81.0%
    "MU-BD-ST",    # R:R 0.67,  24t, WR 87.5%, Exp +4.03%, PnL  +96.6%
    "MU-DF-U",     # R:R 0.75,  18t, WR 88.9%, Exp +3.23%, PnL  +58.2%
    "MU-H-MU",     # R:R 2.14,  28t, WR 60.7%, Exp +1.81%, PnL  +50.6%
    "MU-IH-DN",    # R:R 1.00,  32t, WR 81.2%, Exp +3.65%, PnL +116.8%
    "MU-U-H",      # R:R 0.83,  27t, WR 77.8%, Exp +3.73%, PnL +100.8%
    "U-H-MU",      # R:R 0.75,  38t, WR 84.2%, Exp +2.74%, PnL +104.2%
    "U-MD-GS",     # R:R 0.20,  22t, WR 95.5%, Exp +0.99%, PnL  +21.8%
    "U-MD-MD",     # R:R 2.14, 104t, WR 45.2%, Exp +0.78%, PnL  +81.4%
    "U-MU-H",      # R:R 0.50,  48t, WR 87.5%, Exp +2.71%, PnL +130.2%
    "U-MU-IH",     # R:R 1.20,  25t, WR 76.0%, Exp +4.94%, PnL +123.5%
    "U-ST-DF",     # R:R 0.67,  20t, WR 85.0%, Exp +3.65%, PnL  +73.0%
]

# SHORT patterns (19) — v1.26.4 full TP/SL optimization (8 SHORT optimized)
VALIDATED_SHORT_PATTERNS: List[str] = [
    "BD-BU-DN",    # R:R 1.00,  47t, WR 68.1%, Exp +3.16%, PnL +148.3%
    "BD-D-D",      # R:R 0.50,  13t, WR 100 %, Exp +4.40%, PnL  +57.2%
    "BD-U-H",      # R:R 0.83,  20t, WR 85.0%, Exp +4.93%, PnL  +98.5%
    "BU-MD-MD",    # R:R 1.50,  16t, WR 75.0%, Exp +5.15%, PnL  +82.4%
    "BU-ST-GS",    # R:R 0.40,  13t, WR 100 %, Exp +2.90%, PnL  +37.7%
    "D-BD-ST",     # R:R 0.83,  18t, WR 83.3%, Exp +4.65%, PnL  +83.7%
    "D-DN-DN",     # R:R 0.83,  99t, WR 67.7%, Exp +2.07%, PnL +204.6%
    "DN-BD-BU",    # R:R 0.83,  56t, WR 73.2%, Exp +2.98%, PnL +166.9%
    "DN-D-BD",     # R:R 2.50,  43t, WR 48.8%, Exp +2.03%, PnL  +87.2%
    "DN-DF-DN",    # R:R 1.00,  48t, WR 68.8%, Exp +2.15%, PnL +103.2%
    "H-U-BD",      # R:R 1.50,  20t, WR 70.0%, Exp +4.40%, PnL  +88.0%
    "IH-ST-ST",    # R:R 0.67,  26t, WR 84.6%, Exp +3.59%, PnL  +93.4%
    "MD-MD-MD",    # R:R 1.00,  21t, WR 85.7%, Exp +6.33%, PnL +132.9%
    "ST-BD-BU",    # R:R 1.00,  22t, WR 77.3%, Exp +4.81%, PnL +105.8%
    "ST-DN-BU",    # R:R 0.67,  59t, WR 76.3%, Exp +2.34%, PnL +138.1%
    "ST-DN-U",     # R:R 1.00, 234t, WR 60.7%, Exp +1.82%, PnL +426.6%
    "ST-MU-ST",    # R:R 0.67,  43t, WR 79.1%, Exp +2.76%, PnL +118.7%
    "U-GS-DN",     # R:R 1.00,  26t, WR 84.6%, Exp +6.13%, PnL +159.4%
    # U-H-BU removed v1.27.2: SL 0.3% execution infeasible (effective SL 0.23% after costs)
    "U-ST-DN",     # R:R 0.67, 292t, WR 70.9%, Exp +1.53%, PnL +447.8%
]

# ============================================================
# Pattern Historical Statistics (for confidence calculation)
# Source: v1.27.0 — Uniform TP 70% (52 patterns, 32L+20S)
# WR values updated to reflect TP*0.7 performance
# ============================================================
PATTERN_STATS = {
    # LONG patterns (32)
    "BD-BD-U":   {"direction": "LONG",  "trades":  72, "wr": 81.2, "mc": 0.0027, "wf": 4, "periods": 3},
    "BD-MU-BD":  {"direction": "LONG",  "trades":  23, "wr": 79.2, "mc": 0.0024, "wf": 3, "periods": 3},
    "BD-ST-U":   {"direction": "LONG",  "trades": 106, "wr": 72.2, "mc": 0.0019, "wf": 4, "periods": 3},
    "BU-BU-BD":  {"direction": "LONG",  "trades":  35, "wr": 79.5, "mc": 0.0017, "wf": 5, "periods": 3},
    "D-MU-U":    {"direction": "LONG",  "trades":  35, "wr": 86.8, "mc": 0.0002, "wf": 4, "periods": 3},
    "DN-BD-BD":  {"direction": "LONG",  "trades": 103, "wr": 63.9, "mc": 0.0099, "wf": 4, "periods": 3},
    "DN-DF-MU":  {"direction": "LONG",  "trades":  14, "wr": 93.8, "mc": 0.0017, "wf": 4, "periods": 3},
    "DN-DF-ST":  {"direction": "LONG",  "trades":  24, "wr": 85.2, "mc": 0.0122, "wf": 4, "periods": 3},
    "DN-DN-H":   {"direction": "LONG",  "trades": 103, "wr": 89.7, "mc": 0.0003, "wf": 4, "periods": 3},
    "DN-MD-DN":  {"direction": "LONG",  "trades": 167, "wr": 77.2, "mc": 0.0007, "wf": 4, "periods": 3},
    "GS-ST-ST":  {"direction": "LONG",  "trades":  18, "wr": 84.2, "mc": 0.0174, "wf": 5, "periods": 3},
    "H-BU-BU":   {"direction": "LONG",  "trades":  11, "wr": 100.0,"mc": 0.0001, "wf": 5, "periods": 3},
    "H-MU-MD":   {"direction": "LONG",  "trades":  23, "wr": 92.9, "mc": 0.0007, "wf": 4, "periods": 3},
    "IH-MD-MD":  {"direction": "LONG",  "trades":  19, "wr": 95.2, "mc": 0.0003, "wf": 5, "periods": 2},
    "IH-ST-MU":  {"direction": "LONG",  "trades":  24, "wr": 100.0,"mc": 0.0000, "wf": 5, "periods": 3},
    "MD-BU-MD":  {"direction": "LONG",  "trades":  16, "wr": 90.0, "mc": 0.0014, "wf": 5, "periods": 3},
    "MD-DN-MU":  {"direction": "LONG",  "trades": 130, "wr": 62.3, "mc": 0.0090, "wf": 4, "periods": 3},
    "MD-H-MD":   {"direction": "LONG",  "trades":  24, "wr": 75.0, "mc": 0.0110, "wf": 3, "periods": 3},
    "MD-MD-ST":  {"direction": "LONG",  "trades":  30, "wr": 83.8, "mc": 0.0059, "wf": 5, "periods": 3},
    "MD-ST-BD":  {"direction": "LONG",  "trades":  33, "wr": 57.6, "mc": 0.0088, "wf": 5, "periods": 3},
    "MD-ST-MD":  {"direction": "LONG",  "trades":  30, "wr": 73.3, "mc": 0.0083, "wf": 4, "periods": 3},
    "MU-BD-ST":  {"direction": "LONG",  "trades":  24, "wr": 89.3, "mc": 0.0010, "wf": 5, "periods": 3},
    "MU-DF-U":   {"direction": "LONG",  "trades":  18, "wr": 89.5, "mc": 0.0056, "wf": 4, "periods": 3},
    "MU-H-MU":   {"direction": "LONG",  "trades":  28, "wr": 60.7, "mc": 0.0068, "wf": 5, "periods": 3},
    "MU-IH-DN":  {"direction": "LONG",  "trades":  32, "wr": 83.3, "mc": 0.0004, "wf": 4, "periods": 3},
    "MU-U-H":    {"direction": "LONG",  "trades":  27, "wr": 81.8, "mc": 0.0055, "wf": 4, "periods": 3},
    "U-H-MU":    {"direction": "LONG",  "trades":  38, "wr": 84.2, "mc": 0.0001, "wf": 4, "periods": 3},
    "U-MD-GS":   {"direction": "LONG",  "trades":  22, "wr": 95.7, "mc": 0.0707, "wf": 4, "periods": 3},
    "U-MD-MD":   {"direction": "LONG",  "trades": 104, "wr": 45.2, "mc": 0.0066, "wf": 4, "periods": 2},
    "U-MU-H":    {"direction": "LONG",  "trades":  48, "wr": 88.9, "mc": 0.0006, "wf": 5, "periods": 3},
    "U-MU-IH":   {"direction": "LONG",  "trades":  25, "wr": 80.6, "mc": 0.0011, "wf": 4, "periods": 3},
    "U-ST-DF":   {"direction": "LONG",  "trades":  20, "wr": 86.4, "mc": 0.0131, "wf": 4, "periods": 3},
    # SHORT patterns (19)
    "BD-BU-DN":  {"direction": "SHORT", "trades":  47, "wr": 68.1, "mc": 0.0089, "wf": 4, "periods": 3},
    "BD-D-D":    {"direction": "SHORT", "trades":  13, "wr": 100.0,"mc": 0.0001, "wf": 5, "periods": 3},
    "BD-U-H":    {"direction": "SHORT", "trades":  20, "wr": 85.0, "mc": 0.0011, "wf": 5, "periods": 3},
    "BU-MD-MD":  {"direction": "SHORT", "trades":  16, "wr": 75.0, "mc": 0.0089, "wf": 5, "periods": 3},
    "BU-ST-GS":  {"direction": "SHORT", "trades":  13, "wr": 100.0,"mc": 0.0003, "wf": 5, "periods": 3},
    "D-BD-ST":   {"direction": "SHORT", "trades":  18, "wr": 83.3, "mc": 0.0260, "wf": 5, "periods": 2},
    "D-DN-DN":   {"direction": "SHORT", "trades":  99, "wr": 67.7, "mc": 0.0058, "wf": 5, "periods": 3},
    "DN-BD-BU":  {"direction": "SHORT", "trades":  56, "wr": 79.2, "mc": 0.0014, "wf": 5, "periods": 3},
    "DN-D-BD":   {"direction": "SHORT", "trades":  43, "wr": 54.5, "mc": 0.0177, "wf": 4, "periods": 3},
    "DN-DF-DN":  {"direction": "SHORT", "trades":  48, "wr": 68.8, "mc": 0.0064, "wf": 5, "periods": 3},
    "H-U-BD":    {"direction": "SHORT", "trades":  20, "wr": 75.0, "mc": 0.0038, "wf": 5, "periods": 3},
    "IH-ST-ST":  {"direction": "SHORT", "trades":  26, "wr": 90.0, "mc": 0.0014, "wf": 5, "periods": 3},
    "MD-MD-MD":  {"direction": "SHORT", "trades":  21, "wr": 85.7, "mc": 0.0004, "wf": 4, "periods": 3},
    "ST-BD-BU":  {"direction": "SHORT", "trades":  22, "wr": 82.1, "mc": 0.0027, "wf": 5, "periods": 3},
    "ST-DN-BU":  {"direction": "SHORT", "trades":  59, "wr": 80.6, "mc": 0.0080, "wf": 5, "periods": 2},
    "ST-DN-U":   {"direction": "SHORT", "trades": 234, "wr": 69.3, "mc": 0.0000, "wf": 5, "periods": 2},
    "ST-MU-ST":  {"direction": "SHORT", "trades":  43, "wr": 83.7, "mc": 0.0047, "wf": 4, "periods": 3},
    "U-GS-DN":   {"direction": "SHORT", "trades":  26, "wr": 84.6, "mc": 0.0002, "wf": 5, "periods": 3},
    # U-H-BU removed v1.27.2: SL 0.3% execution infeasible
    "U-ST-DN":   {"direction": "SHORT", "trades": 292, "wr": 77.4, "mc": 0.0002, "wf": 5, "periods": 3},
}

# Confidence calculation weights
CONFIDENCE_WEIGHT_CLARITY = 0.40      # Candle classification clarity
CONFIDENCE_WEIGHT_HISTORICAL = 0.30   # Historical pattern win rate
CONFIDENCE_WEIGHT_REGIME = 0.30       # DEPRECATED: Regime disabled since v1.19.0 (placeholder)

# Confidence logging file
CONFIDENCE_LOG_FILE = os.path.join(PROJECT_ROOT, "results", "pattern_5m_confidence_log.csv")


# ============================================================
# Pattern TP/SL (v1.27.1: Legacy pattern re-optimization)
# Base: v1.27.0 Uniform TP 70%, 15 legacy patterns re-optimized
# Research: tp_sl_lineage_analysis.py, tp_sl_reoptimization_v1270.py
# ============================================================
PATTERN_OPTIMAL_TPSL = {
    # LONG patterns (32) — format: (tp_pct, sl_pct)
    "BD-BD-U":   (1.4, 3.0),   # R:R=0.47  (was 2.0)
    "BD-MU-BD":  (0.7, 0.7),   # R:R=1.00  (was 1.0)
    "BD-ST-U":   (1.4, 2.0),   # R:R=0.70  (was 2.0)
    "BU-BU-BD":  (2.1, 3.0),   # R:R=0.70  (was 3.0)
    "D-MU-U":    (1.05, 2.0),  # R:R=0.53  (was 1.5)
    "DN-BD-BD":  (1.4, 1.5),   # R:R=0.93  (was 2.0)
    "DN-DF-MU":  (1.05, 3.0),  # R:R=0.35  (was 1.5)
    "DN-DF-ST":  (1.4, 3.0),   # R:R=0.47  (was 2.0)
    "DN-DN-H":   (0.7, 2.5),   # R:R=0.28  (was 1.0)
    "DN-MD-DN":  (1.5, 3.0),   # R:R=0.50  (v1.27.1 reopt, was 1.05/2.0)
    "GS-ST-ST":  (1.05, 2.0),  # R:R=0.53  (was 1.5)
    "H-BU-BU":   (1.4, 3.0),   # R:R=0.47  (was 2.0)
    "H-MU-MD":   (0.7, 2.0),   # R:R=0.35  (was 1.0)
    "IH-MD-MD":  (0.7, 2.0),   # R:R=0.35  (was 1.0)
    "IH-ST-MU":  (0.35, 2.5),  # R:R=0.14  (was 0.5)
    "MD-BU-MD":  (1.4, 2.5),   # R:R=0.56  (was 2.0)
    "MD-DN-MU":  (1.0, 1.0),   # R:R=1.00  (v1.27.1 reopt, was 0.7/1.0)
    "MD-H-MD":   (0.7, 0.7),   # R:R=1.00  (was 1.0)
    "MD-MD-ST":  (1.05, 2.0),  # R:R=0.53  (was 1.5)
    "MD-ST-BD":  (1.0, 0.5),   # R:R=2.00  (v1.27.1 reopt, was 0.7/0.5)
    "MD-ST-MD":  (2.0, 2.0),   # R:R=1.00  (v1.27.1 reopt, was 1.4/2.0)
    "MU-BD-ST":  (1.4, 3.0),   # R:R=0.47  (was 2.0)
    "MU-DF-U":   (1.05, 2.0),  # R:R=0.53  (was 1.5)
    "MU-H-MU":   (1.5, 0.7),   # R:R=2.14  (v1.27.1 reopt, was 1.05/0.7)
    "MU-IH-DN":  (1.4, 2.0),   # R:R=0.70  (was 2.0)
    "MU-U-H":    (1.75, 3.0),  # R:R=0.58  (was 2.5)
    "U-H-MU":    (1.5, 2.0),   # R:R=0.75  (v1.27.1 reopt, was 1.05/2.0)
    "U-MD-GS":   (0.35, 2.5),  # R:R=0.14  (was 0.5)
    "U-MD-MD":   (1.5, 0.7),   # R:R=2.14  (v1.27.1 reopt, was 1.05/0.7)
    "U-MU-H":    (1.05, 3.0),  # R:R=0.35  (was 1.5)
    "U-MU-IH":   (2.1, 2.5),   # R:R=0.84  (was 3.0)
    "U-ST-DF":   (1.4, 3.0),   # R:R=0.47  (was 2.0)
    # SHORT patterns (19)
    "BD-BU-DN":  (3.0, 3.0),   # R:R=1.00  (v1.27.1 reopt, was 2.1/3.0)
    "BD-D-D":    (1.05, 3.0),  # R:R=0.35  (was 1.5)
    "BD-U-H":    (2.5, 3.0),   # R:R=0.83  (v1.27.1 reopt, was 1.75/3.0)
    "BU-MD-MD":  (3.0, 2.0),   # R:R=1.50  (v1.27.1 reopt, was 2.1/2.0)
    "BU-ST-GS":  (0.7, 2.5),   # R:R=0.28  (was 1.0)
    "D-BD-ST":   (1.75, 3.0),  # R:R=0.58  (was 2.5)
    "D-DN-DN":   (2.5, 3.0),   # R:R=0.83  (v1.27.1 reopt, was 1.75/3.0)
    "DN-BD-BU":  (1.75, 3.0),  # R:R=0.58  (was 2.5)
    "DN-D-BD":   (1.75, 1.0),  # R:R=1.75  (was 2.5)
    "DN-DF-DN":  (2.0, 2.0),   # R:R=1.00  (v1.27.1 reopt, was 1.4/2.0)
    "H-U-BD":    (2.1, 2.0),   # R:R=1.05  (was 3.0)
    "IH-ST-ST":  (1.4, 3.0),   # R:R=0.47  (was 2.0)
    "MD-MD-MD":  (3.0, 3.0),   # R:R=1.00  (v1.27.1 reopt, was 2.1/3.0)
    "ST-BD-BU":  (2.1, 3.0),   # R:R=0.70  (was 3.0)
    "ST-DN-BU":  (1.4, 3.0),   # R:R=0.47  (was 2.0)
    "ST-DN-U":   (2.1, 3.0),   # R:R=0.70  (was 3.0)
    "ST-MU-ST":  (1.4, 3.0),   # R:R=0.47  (was 2.0)
    "U-GS-DN":   (3.0, 3.0),   # R:R=1.00  (v1.27.1 reopt, was 2.1/3.0)
    # U-H-BU removed v1.27.2: SL 0.3% execution infeasible
    "U-ST-DN":   (1.4, 3.0),   # R:R=0.47  (was 2.0)
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

# Candle-aligned loop timing (v1.25.1)
TRADING_WINDOW_SECONDS = 30     # First 30s after candle close = trading window
POSITION_MONITOR_INTERVAL = 15  # Check position status every 15s during maintenance
MAX_MAINTENANCE_SLEEP = 120     # Max sleep when no position (maintenance window)
DAILY_LOSS_PAUSE_SECONDS = 300
CONSECUTIVE_LOSS_PAUSE_SECONDS = 600  # v1.27.0: 10min pause after 3 consecutive losses
MAX_CONSECUTIVE_LOSSES = 3            # v1.27.0: pause threshold
ENTRY_PRICE_FETCH_DELAY = 0.5
EXIT_PRICE_FETCH_DELAY = 0.5
EXIT_PRICE_RETRY_DELAY = 0.5
EXIT_PRICE_INITIAL_DELAY = 1.0
MAX_EXIT_PRICE_RETRIES = 3

# ============================================================
# INTERVAL CONSTANTS
# ============================================================
MAX_OHLCV_CANDLES = 150  # v1.18.2: Increased from 100 for regime detection (needs 114+)
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
EXPECTED_WIN_RATE = 68.0  # v1.27.3: genuine forward WR 68.5% (strategy_options_evaluation.py)
EXPECTED_AVG_WIN = 5.44   # v1.28.24: 112pat (288bars) TP mean 1.85% × 3x - 0.10%
EXPECTED_AVG_LOSS = 10.73  # v1.28.24: 112pat (288bars) SL mean 3.54% × 3x + 0.10%
EXPECTED_EDGE = 15.0      # v1.27.3: genuine edge ~10-20% (was 50% — inflated by look-ahead)
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
        'max_daily_loss_pct': 13,  # v1.28.5: 10% → 13% (Per-pattern SL max 4.0% × 3x = 12.1% per loss)
        'max_position_size_usd': 10000,
    },
    'api': {
        'retry_attempts': API_MAX_ATTEMPTS,
        'retry_delay': API_BASE_DELAY,
        'max_retry_delay': API_MAX_DELAY,
    },
    'debug_mode': False,
    'json_logging': False,
}