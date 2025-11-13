"""
Opportunity Gating Trading Bot - 4x Leverage + Dynamic Sizing + ML Exit
=======================================================================

Strategy: Only enter SHORT when EV(SHORT) > EV(LONG) + gate_threshold
Entry: 30% Entry Rate Models (2025-11-07, High Frequency Configuration)
Exit: 30% Exit Rate Models (2025-11-07, High Frequency Configuration)

Validated Performance: +12.36% per 28-day window, 60.75% win rate, 9.46 trades/day
Entry Models: 30% entry rate (vs 15% Optimal) for increased trade frequency
Exit Models: 30% exit rate (vs 15% Optimal) for increased trade frequency

Configuration:
  LONG Threshold: 0.60 (HIGH FREQUENCY 2025-11-07)
  SHORT Threshold: 0.60 (HIGH FREQUENCY 2025-11-07)
  Gate Threshold: 0.001
  Leverage: 4x
  Position Sizing: Dynamic (20-95%)

  ML Exit: XGBoost models (30% EXIT RATE 2025-11-07)
    - LONG: prob >= 0.75 (30% exit rate)
    - SHORT: prob >= 0.75 (30% exit rate)
  Emergency Stop Loss: -3% (optimized)
  Emergency Max Hold: 10 hours
"""

import sys
from pathlib import Path
import time
import pickle
import joblib
import logging
from datetime import datetime, timedelta, timezone
UTC = timezone.utc  # For feature logging timestamps
import json
import pandas as pd
import numpy as np
import yaml
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOCK_FILE = RESULTS_DIR / "opportunity_gating_bot_4x.lock"
STATE_FILE = RESULTS_DIR / "opportunity_gating_bot_4x_state.json"
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient
# 🛡️ PRODUCTION-ONLY FEATURE MODULES (2025-11-03)
# Separated from experiments/ to prevent accidental modifications affecting live trading
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer
from src.utils.exchange_reconciliation_v2 import reconcile_state_from_exchange_v2

# =============================================================================
# CONFIGURATION
# =============================================================================

# Strategy Parameters (validated in backtest)
# UPDATED 2025-11-13: Critical Fix for LONG Stop Loss issue (33% SL rate)
# Issue: Production showed 5/15 LONG hitting SL (-$43 loss)
# Solution: Increase LONG threshold 0.60→0.70 to filter low-quality entries
# Models: 30% Entry + 30% Exit models (20251107) - High Frequency Configuration
# Performance: Backtest +12.36% (28 days), 60.75% WR, 9.46 trades/day
# Thresholds: Entry 0.70 (LONG), 0.55 (SHORT), Exit 0.75 (both)
LONG_THRESHOLD = 0.70  # 🔴 CRITICAL FIX: Increased from 0.60 to filter low-quality LONG (reduce SL rate)
SHORT_THRESHOLD = 0.55  # 🟡 OPTIONAL: Decreased from 0.60 to increase SHORT opportunities
GATE_THRESHOLD = 0.001  # Opportunity cost gate

# Leverage & Position Sizing
LEVERAGE = 4  # 4x leverage (validated)
# Dynamic sizing: 20-95% based on signal strength only
# (Volatility, regime, and streak factors available in module but not currently used)

# Expected Returns (for opportunity gating)
LONG_AVG_RETURN = 0.0041  # Average expected return for LONG trades
SHORT_AVG_RETURN = 0.0047  # Average expected return for SHORT trades

# Legacy Exit Parameters (preserved for state compatibility)
FIXED_TAKE_PROFIT = 0.03  # Not used (ML Exit handles profit-taking)
TRAILING_TP_ACTIVATION = 0.02  # Not used
TRAILING_TP_DRAWDOWN = 0.1  # Not used
VOLATILITY_HIGH = 0.02  # Not used (adaptive threshold disabled)
VOLATILITY_LOW = 0.01  # Not used (adaptive threshold disabled)
ML_THRESHOLD_HIGH_VOL = 0.65  # Not used (adaptive threshold disabled)
ML_THRESHOLD_LOW_VOL = 0.75  # Not used (adaptive threshold disabled)
EXIT_STRATEGY = "COMBINED"  # Current strategy

# Exit Parameters (30% EXIT RATE 2025-11-07: High Frequency Configuration)
# Configuration: Triple Barrier (1:1 R/R, P&L-weighted, 30% exit rate)
# Models: xgboost_{long|short}_exit_30pct_20251107_171927.pkl (171 features)
# Backtest (Oct 9 - Nov 6): +12.36% return, 60.75% WR, 1.04× PF, 265 trades (9.46/day)
# Previous: 0.75 (15% exit rate, Optimal models) → +60.44%, 90% WR, 0.37 trades/day
# Trade-off: Higher frequency (25× more trades) vs lower quality (29% lower WR)
ML_EXIT_THRESHOLD_LONG = 0.75  # ML Exit threshold for LONG (30% Exit Rate 2025-11-07)
ML_EXIT_THRESHOLD_SHORT = 0.75  # ML Exit threshold for SHORT (30% Exit Rate 2025-11-07)
ML_EXIT_THRESHOLD_BASE_LONG = ML_EXIT_THRESHOLD_LONG  # Alias for compatibility
ML_EXIT_THRESHOLD_BASE_SHORT = ML_EXIT_THRESHOLD_SHORT  # Alias for compatibility
EMERGENCY_STOP_LOSS = 0.03  # -3% total balance loss (OPTIMIZED 2025-10-22: grid search)
EMERGENCY_MAX_HOLD_TIME = 120  # 120 candles = 10 hours
EMERGENCY_MAX_HOLD_HOURS = EMERGENCY_MAX_HOLD_TIME * 5 / 60  # Convert 5-min candles to hours

# Trading Symbol
SYMBOL = "BTC/USDT:USDT"  # Trading pair
CANDLE_INTERVAL = "5m"  # Candle timeframe
MAX_DATA_CANDLES = 1000  # Number of candles to use for feature calculation

# Data Source (CHANGED 2025-10-26: Use CSV for exact backtest alignment)
DATA_SOURCE = "API"  # "CSV" or "API" - API is default, UTC timezone maintained for features

# Timing
WARMUP_PERIOD_MINUTES = 5  # Bot warmup period (ignore entry signals)

# Reconciliation Settings
BALANCE_CHECK_INTERVAL = 3600  # 1 hour in seconds
BALANCE_RECONCILIATION_THRESHOLD_PCT = 0.005  # 0.5% threshold
FEE_RECONCILIATION_ENABLED = True  # Enable automatic fee reconciliation
FEE_RECONCILIATION_INTERVAL = 1800  # 30 minutes in seconds
FEE_CHECK_INTERVAL = FEE_RECONCILIATION_INTERVAL  # Alias for display
FEE_RECONCILIATION_SYMBOL = "BTC/USDT:USDT"  # Symbol for fee reconciliation

# =============================================================================
# API CONFIGURATION
# =============================================================================

def load_api_keys():
    """Load API keys from config/api_keys.yaml"""
    api_keys_file = CONFIG_DIR / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('mainnet', {})
    return {}

_api_config = load_api_keys()
API_KEY = _api_config.get('api_key', os.getenv("BINGX_API_KEY", ""))
API_SECRET = _api_config.get('secret_key', os.getenv("BINGX_API_SECRET", ""))
USE_TESTNET = False  # ⚠️ MAINNET - REAL MONEY TRADING

# =============================================================================
# LOGGING SETUP
# =============================================================================

log_file = LOGS_DIR / f"opportunity_gating_bot_4x_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# =============================================================================
# MODEL LOADING
# =============================================================================

logger.info("="*80)
logger.info("OPPORTUNITY GATING BOT (4x LEVERAGE) - STARTING")
logger.info("="*80)

logger.info("Loading models...")

# ==============================================================================
# ENTRY MODELS: THRESHOLD 0.80 (Trained 2025-10-27 23:57) ✅ DEPLOYED
# ==============================================================================
# ROLLBACK 2025-10-30: Enhanced Entry Models (Proven +25% Performance)
# Training Script: 5-Fold CV training with enhanced features
# Methodology: Cross-validation on full dataset
#   - Enhanced feature engineering (85 LONG, 79 SHORT features)
#   - Validated performance: +25.21% per 5-day window
# Performance: 72.3% WR, High quality signals, Proven track record
# Timestamp: 20251024_012445
# NOTE: Rolled back from Walk-Forward due to -9.69% backtest performance

# LONG Entry Model - 30% Entry Rate (High Frequency Configuration)
# Training: Aug 9 - Oct 8, 2025 (60 days, 17,455 candles)
# Validation: Oct 9 - Nov 6, 2025 (Max prob: 99.53%, 34.04% at >0.60)
# Entry Rate: 30% (vs 15% Optimal) for 2-10 trades/day target
# Timestamp: 20251107_173027
long_entry_model_path = MODELS_DIR / "xgboost_long_entry_30pct_20251107_173027.pkl"
with open(long_entry_model_path, 'rb') as f:
    long_entry_model = pickle.load(f)

long_entry_scaler_path = MODELS_DIR / "xgboost_long_entry_30pct_20251107_173027_scaler.pkl"
long_entry_scaler = joblib.load(long_entry_scaler_path)

long_entry_features_path = MODELS_DIR / "xgboost_long_entry_30pct_20251107_173027_features.txt"
with open(long_entry_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

logger.info(f"  ✅ LONG Entry loaded: 30% Entry Rate (20251107_173027)")
logger.info(f"     Features: {len(long_feature_columns)} | Entry Rate: 30% | Threshold: 0.60")

# SHORT Entry Model - 30% Entry Rate (High Frequency Configuration)
# Training: Aug 9 - Oct 8, 2025 (60 days, 17,455 candles)
# Validation: Oct 9 - Nov 6, 2025 (Max prob: 96.81%, 16.31% at >0.60)
# Entry Rate: 30% (vs 15% Optimal) for 2-10 trades/day target
# Timestamp: 20251107_173027
short_entry_model_path = MODELS_DIR / "xgboost_short_entry_30pct_20251107_173027.pkl"
with open(short_entry_model_path, 'rb') as f:
    short_entry_model = pickle.load(f)

short_entry_scaler_path = MODELS_DIR / "xgboost_short_entry_30pct_20251107_173027_scaler.pkl"
short_entry_scaler = joblib.load(short_entry_scaler_path)

short_entry_features_path = MODELS_DIR / "xgboost_short_entry_30pct_20251107_173027_features.txt"
with open(short_entry_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

logger.info(f"  ✅ SHORT Entry loaded: 30% Entry Rate (20251107_173027)")
logger.info(f"     Features: {len(short_feature_columns)} | Entry Rate: 30% | Threshold: 0.60")

# LONG Exit Model - 30% Exit Rate (High Frequency Configuration)
# Methodology: 1:1 R/R ATR barriers, P&L-weighted scoring, 30th percentile
# Training: Aug 9 - Oct 8, 2025 (60 days, 17,455 candles)
# Validation: Oct 9 - Nov 6, 2025 (Best CV: 0.8364, 23.48% predictions at >0.75)
# Features: 171, Exit Rate: 30% (vs 15% Optimal) for 2-10 trades/day target
# Exit Threshold: 0.75
# Timestamp: 20251107_171927
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_30pct_20251107_171927.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_30pct_20251107_171927_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_30pct_20251107_171927_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

logger.info(f"  ✅ LONG Exit loaded: 30% Exit Rate (171 features, 20251107_171927)")
logger.info(f"     Features: {len(long_exit_feature_columns)} | Exit Rate: 30% | Threshold: 0.75")

# SHORT Exit Model - 30% Exit Rate (High Frequency Configuration)
# Methodology: 1:1 R/R ATR barriers, P&L-weighted scoring, 30th percentile
# Training: Aug 9 - Oct 8, 2025 (60 days, 17,455 candles)
# Validation: Oct 9 - Nov 6, 2025 (Best CV: 0.8295, 43.73% predictions at >0.75)
# Features: 171, Exit Rate: 30% (vs 15% Optimal) for 2-10 trades/day target
# Exit Threshold: 0.75
# Timestamp: 20251107_171927
short_exit_model_path = MODELS_DIR / "xgboost_short_exit_30pct_20251107_171927.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_30pct_20251107_171927_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_30pct_20251107_171927_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

logger.info(f"  ✅ SHORT Exit loaded: 30% Exit Rate (171 features, 20251107_171927)")
logger.info(f"     Features: {len(short_exit_feature_columns)} | Exit Rate: 30% | Threshold: 0.75")

logger.info(f"  ✅ All models loaded")
logger.info(f"     LONG Entry: {len(long_feature_columns)} features")
logger.info(f"     SHORT Entry: {len(short_feature_columns)} features")
logger.info(f"     LONG Exit: {len(long_exit_feature_columns)} features")
logger.info(f"     SHORT Exit: {len(short_exit_feature_columns)} features")

# Initialize Dynamic Position Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)
logger.info(f"  ✅ Dynamic Position Sizer initialized")

# =============================================================================
# WEIGHTED ENSEMBLE PREDICTION
# =============================================================================

# NOTE: This function is NO LONGER USED as of 2025-10-30
# Bot now uses Enhanced Baseline single models (20251024_012445)
# Keeping for reference only - can be removed in future cleanup
def weighted_ensemble_predict(features, ensemble_models, ensemble_scalers, weights):
    """
    [DEPRECATED - NO LONGER USED]
    Weighted ensemble prediction using Top-3 fold models

    Args:
        features: numpy array of features (1 sample)
        ensemble_models: list of XGBoost models (Top-3)
        ensemble_scalers: list of scalers (Top-3)
        weights: numpy array of normalized weights (sum to 1)

    Returns:
        float: Weighted average probability (positive class)
    """
    predictions = []

    for model, scaler in zip(ensemble_models, ensemble_scalers):
        # Scale features
        features_scaled = scaler.transform(features)

        # Predict probability
        prob_both = model.predict_proba(features_scaled)[0]
        prob = prob_both[1]  # Positive class probability

        predictions.append(prob)

    # Weighted average
    weighted_pred = np.average(predictions, weights=weights)

    return weighted_pred

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_state():
    """Load bot state from file"""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            # Ensure all required fields exist (for backward compatibility)
            if 'session_start' not in state:
                state['session_start'] = datetime.now().isoformat()
            if 'initial_balance' not in state:
                state['initial_balance'] = 100000.0
            if 'current_balance' not in state:
                state['current_balance'] = 100000.0
            if 'timestamp' not in state:
                state['timestamp'] = datetime.now().isoformat()
            if 'latest_signals' not in state:
                state['latest_signals'] = {'entry': {}, 'exit': {}}
            if 'closed_trades' not in state:
                state['closed_trades'] = len([t for t in state.get('trades', []) if t.get('exit_time')])

            # Ensure configuration section exists (backward compatibility)
            # If missing, it will be added on next save_state()
            if 'configuration' not in state:
                state['configuration'] = {
                    'long_threshold': LONG_THRESHOLD,
                    'short_threshold': SHORT_THRESHOLD,
                    'gate_threshold': GATE_THRESHOLD,
                    'ml_exit_threshold_base_long': ML_EXIT_THRESHOLD_BASE_LONG,
                    'ml_exit_threshold_base_short': ML_EXIT_THRESHOLD_BASE_SHORT,
                    'emergency_stop_loss': EMERGENCY_STOP_LOSS,
                    'emergency_max_hold_hours': EMERGENCY_MAX_HOLD_HOURS,
                    'leverage': LEVERAGE,
                    'long_avg_return': LONG_AVG_RETURN,
                    'short_avg_return': SHORT_AVG_RETURN,
                    'fixed_take_profit': FIXED_TAKE_PROFIT,
                    'trailing_tp_activation': TRAILING_TP_ACTIVATION,
                    'trailing_tp_drawdown': TRAILING_TP_DRAWDOWN,
                    'volatility_high': VOLATILITY_HIGH,
                    'volatility_low': VOLATILITY_LOW,
                    'ml_threshold_high_vol': ML_THRESHOLD_HIGH_VOL,
                    'ml_threshold_low_vol': ML_THRESHOLD_LOW_VOL,
                    'exit_strategy': 'COMBINED'
                }

            return state
    return {
        'session_start': datetime.now().isoformat(),
        'initial_balance': 100000.0,
        'current_balance': 100000.0,
        'timestamp': datetime.now().isoformat(),
        'position': None,
        'trades': [],
        'closed_trades': 0,
        'latest_signals': {
            'entry': {},
            'exit': {}
        },
        'stats': {
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl_usd': 0.0,
            'total_pnl_pct': 0.0
        },
        'configuration': {
            'long_threshold': LONG_THRESHOLD,
            'short_threshold': SHORT_THRESHOLD,
            'gate_threshold': GATE_THRESHOLD,
            'ml_exit_threshold_base_long': ML_EXIT_THRESHOLD_BASE_LONG,
            'ml_exit_threshold_base_short': ML_EXIT_THRESHOLD_BASE_SHORT,
            'emergency_stop_loss': EMERGENCY_STOP_LOSS,
            'emergency_max_hold_hours': EMERGENCY_MAX_HOLD_HOURS,
            'leverage': LEVERAGE,
            'long_avg_return': LONG_AVG_RETURN,
            'short_avg_return': SHORT_AVG_RETURN,
            # COMBINED Strategy parameters
            'fixed_take_profit': FIXED_TAKE_PROFIT,
            'trailing_tp_activation': TRAILING_TP_ACTIVATION,
            'trailing_tp_drawdown': TRAILING_TP_DRAWDOWN,
            'volatility_high': VOLATILITY_HIGH,
            'volatility_low': VOLATILITY_LOW,
            'ml_threshold_high_vol': ML_THRESHOLD_HIGH_VOL,
            'ml_threshold_low_vol': ML_THRESHOLD_LOW_VOL,
            'exit_strategy': 'COMBINED'
        }
    }

def reconcile_balance_with_exchange(client, state, threshold_pct=BALANCE_RECONCILIATION_THRESHOLD_PCT):
    """
    Auto-detect deposits/withdrawals by comparing exchange balance with state balance.
    Adjusts state balances and logs reconciliation event.

    Args:
        client: BingXClient instance
        state: Current bot state dictionary
        threshold_pct: Minimum difference percentage to trigger reconciliation (default: 0.5%)

    Returns:
        bool: True if reconciliation was performed, False otherwise
    """
    try:
        # Get current balance from exchange API
        balance_info = client.get_balance()
        exchange_balance = float(balance_info['balance']['balance'])

        # Get state balance
        state_balance = state.get('current_balance', 0)

        # Calculate difference (absolute and percentage)
        balance_diff = exchange_balance - state_balance
        balance_diff_pct = (abs(balance_diff) / state_balance * 100) if state_balance > 0 else 0

        # Check if difference exceeds threshold (percentage-based, adaptive to capital size)
        if balance_diff_pct < threshold_pct:
            # No significant difference - no reconciliation needed
            return False

        # Significant difference detected - determine cause
        logger.info(f"💰 Balance Reconciliation Triggered")
        logger.info(f"   Exchange Balance: ${exchange_balance:,.2f}")
        logger.info(f"   State Balance: ${state_balance:,.2f}")
        logger.info(f"   Difference: ${balance_diff:+,.2f} ({balance_diff_pct:+.2f}%)")

        # Check if we have open position (skip reconciliation during active trading)
        if state.get('position') and state['position'].get('status') == 'OPEN':
            logger.warning(f"⚠️  Open position detected - skipping auto-reconciliation")
            logger.warning(f"   Please close position before deposit/withdrawal")
            return False

        # ✅ FIXED 2025-10-25: NEVER adjust initial_balance
        # Initial balance is the baseline set at session start or manual reset
        # It should NEVER be automatically adjusted by reconciliation
        # Only current_balance should be updated to match exchange

        # Determine reconciliation type
        reconciliation_type = 'deposit' if balance_diff > 0 else 'withdrawal'

        # Get current values
        old_initial_balance = state.get('initial_balance', 0)
        old_current_balance = state.get('current_balance', 0)
        old_realized_balance = state.get('realized_balance', 0)

        # ✅ CRITICAL FIX: NEVER adjust initial_balance or realized_balance
        # These are baseline values that must remain constant
        state['initial_balance'] = old_initial_balance  # PRESERVE
        state['realized_balance'] = old_realized_balance  # PRESERVE

        # ONLY update current_balance to match exchange
        state['current_balance'] = exchange_balance

        # Add reconciliation log entry
        reconciliation_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'auto_balance_reconciliation',
            'type': reconciliation_type,
            'amount': abs(balance_diff),
            'amount_pct': balance_diff_pct,
            'reason': f'Auto-detected {reconciliation_type} (${abs(balance_diff):.2f}, {balance_diff_pct:.2f}%)',
            'exchange_balance': exchange_balance,
            'previous_state_balance': old_current_balance,
            'previous_initial_balance': old_initial_balance,
            'new_initial_balance': state['initial_balance'],  # Same as old (never adjusted)
            'new_current_balance': state['current_balance'],
            'initial_balance_adjusted': False,  # Always False - never adjusted
            'notes': f'Automatic reconciliation - {reconciliation_type} detected via exchange API (initial_balance NEVER adjusted)'
        }

        # Initialize reconciliation_log if not exists
        if 'reconciliation_log' not in state:
            state['reconciliation_log'] = []

        state['reconciliation_log'].append(reconciliation_entry)

        # Log reconciliation
        logger.info(f"✅ Balance Reconciliation Complete:")
        logger.info(f"   Type: {reconciliation_type.upper()}")
        logger.info(f"   Amount: ${abs(balance_diff):+,.2f} ({balance_diff_pct:.2f}%)")
        logger.info(f"   Initial Balance: ${old_initial_balance:,.2f} (PRESERVED - never auto-adjusted)")
        logger.info(f"   Current Balance: ${old_current_balance:,.2f} → ${state['current_balance']:,.2f}")
        logger.info(f"   Manual baseline always respected")

        return True

    except Exception as e:
        logger.error(f"❌ Balance reconciliation failed: {e}")
        return False

def get_order_fee_from_exchange(client, order_id, symbol=FEE_RECONCILIATION_SYMBOL):
    """
    Get actual fee from exchange order (ground truth).
    Returns fee in USDT.
    """
    try:
        order = client.exchange.fetch_order(order_id, symbol)
        fee_info = order.get('fee', {})
        fee_cost = fee_info.get('cost', 0)
        return float(fee_cost) if fee_cost else 0.0
    except Exception as e:
        logger.warning(f"⚠️  Could not fetch fee for order {order_id}: {e}")
        return None  # Return None to indicate fetch failure (don't overwrite with 0)

def reconcile_trade_fees_from_exchange(client, state):
    """
    Reconcile trade fees from exchange orders (ground truth).
    Updates state file with actual fees from exchange API.

    This function:
    1. Queries exchange API for each trade order fees
    2. Updates entry_fee, exit_fee, total_fee in state
    3. Recalculates pnl_usd_net with actual fees
    4. Marks trades as fees_reconciled

    Returns: (reconciled_count, failed_count)
    """
    if not FEE_RECONCILIATION_ENABLED:
        return (0, 0)

    reconciled_count = 0
    failed_count = 0

    try:
        # Process all closed trades
        for trade in state.get('trades', []):
            if trade.get('status') != 'CLOSED':
                continue

            # Skip if already reconciled
            if trade.get('fees_reconciled', False):
                continue

            trade_id = trade.get('order_id', 'unknown')
            fees_updated = False

            # Get entry fee
            entry_order_id = trade.get('order_id')
            if entry_order_id and entry_order_id != 'N/A':
                entry_fee = get_order_fee_from_exchange(client, entry_order_id)
                if entry_fee is not None:
                    trade['entry_fee'] = entry_fee
                    fees_updated = True
                else:
                    failed_count += 1
                    logger.warning(f"⚠️  Failed to fetch entry fee for trade {trade_id}")

            # Get exit fee
            exit_order_id = trade.get('close_order_id')
            if exit_order_id and exit_order_id != 'N/A':
                exit_fee = get_order_fee_from_exchange(client, exit_order_id)
                if exit_fee is not None:
                    trade['exit_fee'] = exit_fee
                    fees_updated = True
                else:
                    # Exit order might not exist for stop loss triggers
                    # This is okay, just log it
                    logger.info(f"ℹ️  No exit fee for trade {trade_id} (exit order: {exit_order_id})")
                    trade['exit_fee'] = 0.0

            # Calculate total fee and net P&L
            if fees_updated:
                entry_fee_val = trade.get('entry_fee', 0.0)
                exit_fee_val = trade.get('exit_fee', 0.0)
                total_fee = entry_fee_val + exit_fee_val

                trade['total_fee'] = total_fee
                trade['pnl_usd_net'] = trade.get('pnl_usd', 0.0) - total_fee
                trade['fees_reconciled'] = True

                reconciled_count += 1
                logger.info(f"✅ Trade {trade_id}: fees reconciled (Entry: ${entry_fee_val:.2f}, "
                           f"Exit: ${exit_fee_val:.2f}, Total: ${total_fee:.2f})")

        if reconciled_count > 0:
            logger.info(f"💰 Fee reconciliation complete: {reconciled_count} trades updated, "
                       f"{failed_count} failures")

        return (reconciled_count, failed_count)

    except Exception as e:
        logger.error(f"❌ Fee reconciliation failed: {e}")
        return (reconciled_count, failed_count)

def recalculate_stats(trades):
    """
    Recalculate stats from bot trades (excludes manual trades)
    
    Args:
        trades: List of all trades (bot + manual)
        
    Returns:
        dict: Updated stats
    """
    # Filter bot trades only (exclude manual reconciled)
    bot_closed = [t for t in trades if t.get('status') == 'CLOSED' and not t.get('manual_trade', False)]
    
    # Calculate stats
    long_trades = [t for t in bot_closed if t.get('side') in ['LONG', 'BUY']]
    short_trades = [t for t in bot_closed if t.get('side') in ['SHORT', 'SELL']]
    wins = [t for t in bot_closed if t.get('pnl_usd_net', 0) > 0]
    losses = [t for t in bot_closed if t.get('pnl_usd_net', 0) <= 0]
    
    total_pnl_usd = sum([t.get('pnl_usd_net', 0) for t in bot_closed])
    total_pnl_pct = sum([t.get('pnl_pct', 0) for t in bot_closed])
    
    return {
        'total_trades': len(bot_closed),
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'wins': len(wins),
        'losses': len(losses),
        'total_pnl_usd': total_pnl_usd,
        'total_pnl_pct': total_pnl_pct
    }

def remove_duplicate_trades(trades):
    """
    Remove duplicate trades (same order_id)
    
    Args:
        trades: List of trades
        
    Returns:
        tuple: (cleaned_trades, num_removed)
    """
    seen_order_ids = set()
    unique_trades = []
    duplicates_removed = 0
    
    for trade in trades:
        order_id = trade.get('order_id')
        
        if order_id in seen_order_ids:
            duplicates_removed += 1
            logger.warning(f"🗑️  Removed duplicate trade: {order_id}")
            continue
        
        seen_order_ids.add(order_id)
        unique_trades.append(trade)
    
    if duplicates_removed > 0:
        logger.info(f"✅ Removed {duplicates_removed} duplicate trade(s)")
    
    return unique_trades, duplicates_removed

def remove_stale_trades(trades):
    """
    Remove stale trades (position not found errors)
    
    Args:
        trades: List of trades
        
    Returns:
        tuple: (cleaned_trades, num_removed)
    """
    clean_trades = []
    stale_removed = 0
    
    for trade in trades:
        exit_reason = trade.get('exit_reason', '')
        
        # Check for stale trade indicators
        if 'position not found' in exit_reason.lower() or 'stale trade' in exit_reason.lower():
            stale_removed += 1
            logger.warning(f"🗑️  Removed stale trade: {trade.get('order_id')} - {exit_reason[:50]}")
            continue
        
        clean_trades.append(trade)
    
    if stale_removed > 0:
        logger.info(f"✅ Removed {stale_removed} stale trade(s)")
    
    return clean_trades, stale_removed

def save_state(state):
    """Save bot state to file"""
    # IMPORTANT: Preserve manual trades AND fee-reconciled data from existing state file
    # This prevents bot from overwriting manually tracked trades and fee data
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                existing_state = json.load(f)

            # Find manual trades in existing state
            existing_trades = existing_state.get('trades', [])
            manual_trades = [t for t in existing_trades if t.get('manual_trade', False)]

            # Also preserve fee-reconciled data for ALL trades
            current_trades = state.get('trades', [])
            for current_trade in current_trades:
                # Skip manual trades (they'll be added from file later)
                if current_trade.get('manual_trade', False):
                    continue

                # Find matching trade in existing state by order_id
                order_id = current_trade.get('order_id')
                if order_id:
                    matching_existing = next((t for t in existing_trades if t.get('order_id') == order_id), None)

                    # If found and it has reconciled fees, preserve them
                    if matching_existing and matching_existing.get('fees_reconciled', False):
                        current_trade['entry_fee'] = matching_existing.get('entry_fee', 0.0)
                        current_trade['exit_fee'] = matching_existing.get('exit_fee', 0.0)
                        current_trade['total_fee'] = matching_existing.get('total_fee', 0.0)
                        current_trade['pnl_usd_net'] = matching_existing.get('pnl_usd_net', current_trade.get('pnl_usd', 0))
                        current_trade['fees_reconciled'] = True

            if manual_trades:
                # Keep all current trades (including reconciled manual trades)
                # Only add old manual trades that aren't already in current_trades
                current_ids = set(t.get('order_id') for t in current_trades)
                old_manual_new = [t for t in manual_trades
                                  if t.get('order_id') not in current_ids]

                # Merge: current trades (all bot + reconciled) + old manual (not yet in current)
                state['trades'] = current_trades + old_manual_new

                reconciled_count = len([t for t in current_trades if t.get('exchange_reconciled', False)])
                logger.info(f"💾 Preserved {len(current_trades)} trades ({reconciled_count} reconciled, {len(old_manual_new)} old manual)")
        except Exception as e:
            logger.warning(f"⚠️  Could not preserve manual trades and fees: {e}")

    # === AUTOMATIC STATE CLEANUP (Auto-update: 2025-10-22) ===
    # Runs on every save to maintain state file integrity
    trades = state.get('trades', [])
    if trades:
        # 1. Remove duplicate trades (same order_id)
        trades, duplicates_removed = remove_duplicate_trades(trades)
        if duplicates_removed > 0:
            logger.info(f"🗑️  Removed {duplicates_removed} duplicate trade(s)")
        state['trades'] = trades

        # 2. Remove stale trades (position not found errors)
        trades, stale_removed = remove_stale_trades(trades)
        if stale_removed > 0:
            logger.info(f"🗑️  Removed {stale_removed} stale trade(s)")
        state['trades'] = trades

        # 3. Recalculate stats from bot trades (excludes manual trades)
        # 🔧 FIX 2025-11-13: Use trading_history instead of trades to include all closed positions
        # trades array may have some positions removed during cleanup, but trading_history
        # is the permanent record of all bot trades (including Stop Loss exits)
        updated_stats = recalculate_stats(state.get('trading_history', []))
        state['stats'] = updated_stats
        logger.debug(f"📊 Stats auto-updated: {updated_stats['total_trades']} trades, "
                    f"{updated_stats['wins']}W/{updated_stats['losses']}L, "
                    f"P&L: ${updated_stats['total_pnl_usd']:.2f}")

    # 🔧 FIX 2025-11-10: Ensure initial_wallet_balance is synced with initial_balance
    # This prevents Monitor from showing incorrect withdrawal amounts
    if 'initial_balance' in state:
        state['initial_wallet_balance'] = state['initial_balance']

    # Add configuration section (Single Source of Truth for monitor)
    state['configuration'] = {
        'long_threshold': LONG_THRESHOLD,
        'short_threshold': SHORT_THRESHOLD,
        'gate_threshold': GATE_THRESHOLD,
        'ml_exit_threshold_base_long': ML_EXIT_THRESHOLD_BASE_LONG,
        'ml_exit_threshold_base_short': ML_EXIT_THRESHOLD_BASE_SHORT,
        'emergency_stop_loss': EMERGENCY_STOP_LOSS,
        'emergency_max_hold_hours': EMERGENCY_MAX_HOLD_HOURS,
        'leverage': LEVERAGE,
        'long_avg_return': LONG_AVG_RETURN,
        'short_avg_return': SHORT_AVG_RETURN,
        # COMBINED Strategy parameters
        'fixed_take_profit': FIXED_TAKE_PROFIT,
        'trailing_tp_activation': TRAILING_TP_ACTIVATION,
        'trailing_tp_drawdown': TRAILING_TP_DRAWDOWN,
        'volatility_high': VOLATILITY_HIGH,
        'volatility_low': VOLATILITY_LOW,
        'ml_threshold_high_vol': ML_THRESHOLD_HIGH_VOL,
        'ml_threshold_low_vol': ML_THRESHOLD_LOW_VOL,
        'exit_strategy': 'COMBINED'
    }

    # Unrealized P&L: Clear logic based on position state
    if state.get('position') is None:
        # No position = no unrealized P&L
        state['unrealized_pnl'] = 0.0
    # else: keep the unrealized_pnl already stored in state from position calculation

    # Calculate realized balance: Current balance (from exchange) - Unrealized P&L
    # This accurately accounts for fees (already deducted from current_balance by exchange)
    # current_balance includes: initial - fees - realized_pnl - unrealized_pnl
    # realized_balance should be: initial - fees - realized_pnl
    unrealized_pnl = state.get('unrealized_pnl', 0.0)
    realized_balance = state['current_balance'] - unrealized_pnl
    state['realized_balance'] = realized_balance

    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj

    state_converted = convert_numpy_types(state)
    with open(STATE_FILE, 'w') as f:
        json.dump(state_converted, f, indent=2)

# =============================================================================
# TIME SYNCHRONIZATION & CANDLE VALIDATION
# =============================================================================

def wait_for_next_candle(interval_minutes=5, post_delay_seconds=1):
    """
    다음 5분 정각 +1초까지 대기

    Args:
        interval_minutes: 캔들 간격 (기본 5분)
        post_delay_seconds: 정각 후 추가 대기 시간 (초 단위)

    Returns:
        datetime: 대기 완료 시간
    """
    now = datetime.now()

    # 다음 정각 계산
    minutes_to_next = interval_minutes - (now.minute % interval_minutes)
    seconds_to_next = minutes_to_next * 60 - now.second

    # 정각 + post_delay_seconds
    total_wait = seconds_to_next + post_delay_seconds

    next_candle_time = now + timedelta(seconds=seconds_to_next)
    wait_until = next_candle_time + timedelta(seconds=post_delay_seconds)

    logger.info(f"⏰ 다음 캔들까지 대기:")
    logger.info(f"   현재 (KST): {now.strftime('%H:%M:%S')}")
    logger.info(f"   정각 (KST): {next_candle_time.strftime('%H:%M:%S')}")
    logger.info(f"   요청 (KST): {wait_until.strftime('%H:%M:%S')} (정각 +{post_delay_seconds}초)")
    logger.info(f"   대기: {total_wait:.0f}초")

    time.sleep(total_wait)

    return datetime.now()

def load_from_csv(csv_file, limit, current_time):
    """
    CSV 파일에서 최신 캔들 데이터 로드

    Args:
        csv_file: CSV 파일 경로
        limit: 로드할 캔들 개수
        current_time: 현재 시간 (최신성 검증용)

    Returns:
        DataFrame or None: 검증된 캔들 데이터
    """
    try:
        # OPTIMIZED: Load only last N rows (메모리 97% 절약, 8배 속도 향상)
        # Count total lines efficiently
        with open(csv_file, 'r') as f:
            total_lines = sum(1 for _ in f) - 1  # Exclude header

        # Calculate rows to skip
        rows_needed = limit + 10  # +10 buffer for filtering
        skip_rows = max(0, total_lines - rows_needed)

        # Load only needed rows
        if skip_rows > 0:
            df = pd.read_csv(csv_file, skiprows=range(1, skip_rows + 1))  # Keep header (row 0)
        else:
            df = pd.read_csv(csv_file)  # File smaller than needed, load all

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Convert UTC to KST (CSV is in UTC)
        import pytz
        kst = pytz.timezone('Asia/Seoul')
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(kst).dt.tz_localize(None)

        # Convert price columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Verify data freshness
        latest_candle = df.iloc[-1]['timestamp']
        data_age_minutes = (current_time - latest_candle).total_seconds() / 60

        # Data should be within 10 minutes (2 candles)
        if data_age_minutes > 10:
            logger.warning(f"   ⚠️ CSV data is stale: {data_age_minutes:.1f} minutes old")
            logger.warning(f"   Latest CSV candle: {latest_candle.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.warning(f"   Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            return None

        logger.info(f"   ✅ CSV loaded: {len(df)} candles")
        logger.info(f"   Latest candle: {latest_candle.strftime('%H:%M:%S')} KST")
        logger.info(f"   Data age: {data_age_minutes:.1f} minutes")

        return df

    except Exception as e:
        logger.error(f"   ❌ CSV load error: {e}")
        return None

def update_csv_if_needed(csv_file, update_script, current_time):
    """
    CSV가 오래되었으면 업데이트 시도

    Args:
        csv_file: CSV 파일 경로
        update_script: 업데이트 스크립트 경로
        current_time: 현재 시간

    Returns:
        bool: 업데이트 성공 여부
    """
    try:
        # Check CSV age
        if not csv_file.exists():
            logger.warning("   CSV file not found - attempting update")
        else:
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            latest = df['timestamp'].max()

            import pytz
            kst = pytz.timezone('Asia/Seoul')
            latest_kst = latest.tz_localize('UTC').tz_convert(kst).tz_localize(None)

            age_minutes = (current_time - latest_kst).total_seconds() / 60

            if age_minutes < 6:  # CSV is fresh enough
                logger.info(f"   ✅ CSV is fresh ({age_minutes:.1f} min old) - no update needed")
                return True

            logger.info(f"   📅 CSV is {age_minutes:.1f} min old - updating...")

        # Run update script
        import subprocess
        result = subprocess.run(
            ['python', str(update_script)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            logger.info("   ✅ CSV updated successfully")
            return True
        else:
            logger.error(f"   ❌ CSV update failed: {result.stderr[:200]}")
            return False

    except Exception as e:
        logger.error(f"   ❌ CSV update error: {e}")
        return False

def fetch_and_validate_candles(client, symbol, interval, limit, expected_candle_time, max_retries=5):
    """
    캔들 데이터 가져오기 + 예상 캔들 검증 + 재시도

    Args:
        client: BingXClient 인스턴스
        symbol: 심볼 (예: "BTC-USDT")
        interval: 캔들 간격 (예: "5m")
        limit: 가져올 캔들 개수
        expected_candle_time: 예상되는 최신 캔들 시간
        max_retries: 최대 재시도 횟수

    Returns:
        DataFrame or None: 검증된 캔들 데이터
    """

    for attempt in range(max_retries):
        try:
            # 데이터 요청
            klines = client.get_klines(symbol, interval, limit=limit)

            if klines is None or len(klines) == 0:
                logger.warning(f"   재시도 {attempt+1}/{max_retries}: 데이터 없음")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None

            # DataFrame 변환
            df = pd.DataFrame(klines)

            # 타임존 처리: UTC 유지 (FIXED 2025-10-26: Feature calculation과 동일하게)
            # ⚠️ CRITICAL: UTC로 유지해야 백테스트와 동일한 feature 계산됨
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            # 최신 캔들 시간 (UTC)
            # Note: This may include in-progress candle, but filter_completed_candles() will handle it
            latest_candle_time_utc = df.iloc[-1]['timestamp']

            # KST 변환 (로깅용 - feature calculation에는 영향 없음)
            import pytz
            kst = pytz.timezone('Asia/Seoul')
            latest_candle_time_kst = pd.to_datetime(latest_candle_time_utc).tz_localize('UTC').tz_convert(kst).tz_localize(None)
            expected_candle_time_kst = pd.to_datetime(expected_candle_time).tz_localize('UTC').tz_convert(kst).tz_localize(None)

            # ⚠️ STRICT VALIDATION (2025-10-22): Exact timestamp match required
            # Compare in UTC (data timezone)
            if latest_candle_time_utc == expected_candle_time:
                logger.info(f"   ✅ 예상 캔들 정확히 일치: {latest_candle_time_kst.strftime('%H:%M:%S')} KST ({latest_candle_time_utc.strftime('%H:%M:%S')} UTC)")
                return df
            else:
                time_diff_seconds = (latest_candle_time_utc - expected_candle_time).total_seconds()
                logger.warning(f"   ⚠️  재시도 {attempt+1}/{max_retries}: 예상 캔들 불일치")
                logger.warning(f"      예상 (KST): {expected_candle_time_kst.strftime('%H:%M:%S')} / UTC: {expected_candle_time.strftime('%H:%M:%S')}")
                logger.warning(f"      실제 (KST): {latest_candle_time_kst.strftime('%H:%M:%S')} / UTC: {latest_candle_time_utc.strftime('%H:%M:%S')}")
                logger.warning(f"      시간차: {time_diff_seconds:+.0f}초")

                if attempt < max_retries - 1:
                    logger.info(f"      1초 후 재시도...")
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"   ❌ 최대 재시도 횟수 초과")
                    return None

        except Exception as e:
            logger.error(f"   재시도 {attempt+1}/{max_retries} 오류: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None

    return None

def filter_completed_candles(df, current_time, interval_minutes=5, max_age_minutes=2):
    """
    완성된 캔들만 필터링 + 최신성 검증

    ⚠️ KEY LOGIC (2025-10-23): Time-based filtering for completed candles
    This is the PRIMARY mechanism to exclude in-progress candles.

    Args:
        df: 캔들 DataFrame (may include in-progress candle as last element)
        current_time: 현재 시간
        interval_minutes: 캔들 간격 (기본 5분)
        max_age_minutes: 최대 허용 지연 시간 (기본 2분 - 캔들 완성 후 경과 시간)

    Returns:
        DataFrame: 완성된 캔들만 포함된 DataFrame
        bool: 최신 데이터 여부
    """

    # ⚠️ CORRECTED (2025-10-23): BingX API는 캔들의 시작 시간(start time)을 타임스탬프로 사용
    # 예: 23:15:00 타임스탬프 = 23:15:00~23:20:00 구간의 캔들 (23:20:00에 완성)
    #     23:20:00 타임스탬프 = 23:20:00~23:25:00 구간의 캔들 (23:25:00에 완성 예정)
    # 현재 시각 23:24:03인 경우:
    #   - 23:15:00 캔들: 완성됨 (< 23:20:00) ✅
    #   - 23:20:00 캔들: 진행 중 (>= 23:20:00) ❌ 제외

    # FIXED 2025-10-26: 모든 계산은 UTC로 수행 (백테스트와 동일)
    # ⚠️ CRITICAL: current_time과 df['timestamp'] 모두 UTC임

    # 현재 진행중 캔들의 시작 시간 계산 (UTC)
    current_candle_start = current_time.replace(second=0, microsecond=0)
    current_candle_start = current_candle_start - timedelta(minutes=current_candle_start.minute % interval_minutes)

    # 완성된 캔들만 필터링 (진행중 캔들 제외)
    # timestamp < current_candle_start 인 캔들만 완성됨
    df_completed = df[df['timestamp'] < current_candle_start].copy()

    if len(df_completed) == 0:
        logger.error("   ❌ 완성된 캔들 없음")
        return df_completed, False

    # 최신 완성 캔들 (시작 시간, UTC)
    latest_completed_start = df_completed.iloc[-1]['timestamp']
    # Expected latest completed candle = current candle start - 1 interval
    # Example: Current time 23:24:03 → current_candle_start 23:20:00
    #          → expected_latest 23:15:00 (last completed candle)
    expected_latest = current_candle_start - timedelta(minutes=interval_minutes)

    # 최신성 검증 (캔들 완성 시간 기준, UTC)
    # ⚠️ CORRECTED (2025-10-23): 데이터 나이 = 캔들 완성 후 경과 시간
    # 예: 00:25:00 시작 캔들 → 00:30:00 완성 → 현재 00:30:06 → 데이터 나이 0.1분 ✅
    candle_completion_time = latest_completed_start + timedelta(minutes=interval_minutes)
    age_minutes = (current_time - candle_completion_time).total_seconds() / 60
    is_fresh = age_minutes <= max_age_minutes

    # KST 변환 (로깅용 - feature calculation에는 영향 없음)
    import pytz
    kst = pytz.timezone('Asia/Seoul')
    latest_completed_start_kst = pd.to_datetime(latest_completed_start).tz_localize('UTC').tz_convert(kst).tz_localize(None)
    candle_completion_time_kst = pd.to_datetime(candle_completion_time).tz_localize('UTC').tz_convert(kst).tz_localize(None)
    expected_latest_kst = pd.to_datetime(expected_latest).tz_localize('UTC').tz_convert(kst).tz_localize(None)

    logger.info(f"   📊 캔들 필터링 결과:")
    logger.info(f"      전체 캔들: {len(df)} → 완성 캔들: {len(df_completed)}")
    logger.info(f"      최신 완성 시작: {latest_completed_start_kst.strftime('%H:%M:%S')} KST ({latest_completed_start.strftime('%H:%M:%S')} UTC)")
    logger.info(f"      최신 완성 종료: {candle_completion_time_kst.strftime('%H:%M:%S')} KST ({candle_completion_time.strftime('%H:%M:%S')} UTC)")
    logger.info(f"      예상 최신: {expected_latest_kst.strftime('%H:%M:%S')} KST ({expected_latest.strftime('%H:%M:%S')} UTC)")
    logger.info(f"      데이터 나이 (완성 후): {age_minutes:.1f}분")

    if not is_fresh:
        logger.warning(f"   ⚠️  데이터가 오래됨 ({age_minutes:.1f}분 > {max_age_minutes}분)")
        logger.warning(f"      API 지연 또는 네트워크 문제 가능성")
    else:
        logger.info(f"   ✅ 최신 데이터 확인 (완성 후 {age_minutes:.1f}분)")

    # ⚠️ STRICT VALIDATION (2025-10-22): Exact timestamp match required
    # No tolerance - must match exactly
    if latest_completed_start == expected_latest:
        logger.info(f"   ✅ 예상 캔들과 정확히 일치")
    else:
        time_diff = (latest_completed_start - expected_latest).total_seconds()
        logger.warning(f"   ⚠️  예상 캔들과 불일치 ({time_diff:+.0f}초 차이)")
        logger.warning(f"      예상: {expected_latest_kst.strftime('%H:%M:%S')} KST ({expected_latest.strftime('%H:%M:%S')} UTC)")
        logger.warning(f"      실제: {latest_completed_start_kst.strftime('%H:%M:%S')} KST ({latest_completed_start.strftime('%H:%M:%S')} UTC)")
        # Note: This is now a warning, not accepted as "close enough"

    return df_completed, is_fresh

# =============================================================================
# SIGNAL GENERATION (OPTIMIZED WITH FEATURE CACHING)
# =============================================================================

# Global feature cache for performance optimization
_cached_features = None
_last_candle_timestamp = None
_cache_stats = {'hits': 0, 'misses': 0, 'full_calcs': 0}

def get_signals(df):
    """
    Calculate LONG and SHORT signals with feature caching optimization

    Performance:
    - First run: Full calculation (1000 candles)
    - Same candle: Cache hit (0 calculations, ~1000x faster)
    - New candle: Incremental update (optimized)

    Returns:
        (long_prob, short_prob, df_features)
    """
    global _cached_features, _last_candle_timestamp, _cache_stats

    try:
        current_candle_time = df.iloc[-1]['timestamp']

        # Check if we can reuse cached features (same candle)
        if _last_candle_timestamp is not None and current_candle_time == _last_candle_timestamp:
            # Cache HIT: Same candle, reuse features
            _cache_stats['hits'] += 1
            logger.debug(f"Feature cache HIT (candle: {current_candle_time}) [hits: {_cache_stats['hits']}, misses: {_cache_stats['misses']}]")
            df_features = _cached_features
        else:
            # Cache MISS: New candle detected
            _cache_stats['misses'] += 1

            # For now, recalculate all features
            # TODO: Implement incremental update for further optimization
            _cache_stats['full_calcs'] += 1
            logger.debug(f"Feature cache MISS - calculating features for {len(df)} candles")

            df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
            df_features = prepare_exit_features(df_features)  # Add EXIT-specific features (25 features)

            # 🔍 DEBUG: Verify feature DataFrame order
            logger.info(f"🔍 DEBUG - Feature DataFrame Order:")
            logger.info(f"   Feature rows: {len(df_features)}")
            if 'timestamp' in df_features.columns:
                logger.info(f"   FIRST feature timestamp: {df_features.iloc[0]['timestamp']}")
                logger.info(f"   LAST feature timestamp: {df_features.iloc[-1]['timestamp']}")
                feat_sorted = df_features['timestamp'].is_monotonic_increasing
                logger.info(f"   Feature timestamps ascending: {feat_sorted} {'✅' if feat_sorted else '❌ ERROR!'}")
            else:
                logger.warning(f"   ⚠️  No timestamp column in features - cannot verify order")

            # Update cache
            _cached_features = df_features
            _last_candle_timestamp = current_candle_time

            logger.debug(f"Features calculated and cached (candle: {current_candle_time})")

        # Get latest candle features
        latest = df_features.iloc[-1:].copy()

        # 🔍 DEBUG: Verify we're using the LATEST candle
        logger.info(f"🔍 DEBUG - Latest Candle Selection:")
        logger.info(f"   Selected index: -1 (last row)")
        if 'timestamp' in latest.columns:
            logger.info(f"   Selected candle timestamp: {latest.iloc[0]['timestamp']}")
        if 'close' in latest.columns:
            logger.info(f"   Selected candle close: ${latest.iloc[0]['close']:,.1f}")

        # LONG signal (Top-3 Weighted Ensemble)
        try:
            expected_long_features = len(long_feature_columns)
            long_feat_df = latest[long_feature_columns]

            # Verify we have all required features
            if long_feat_df.shape[1] != expected_long_features:
                logger.warning(f"LONG features: expected {expected_long_features}, got {long_feat_df.shape[1]}")
                logger.warning(f"  Missing features: {set(long_feature_columns) - set(long_feat_df.columns)}")
                long_prob = 0.0
                raise ValueError(f"Feature mismatch: need {expected_long_features} features")

            long_feat = long_feat_df.values

            # Enhanced Baseline single model prediction
            long_scaled = long_entry_scaler.transform(long_feat)
            long_prob = long_entry_model.predict_proba(long_scaled)[0][1]

        except Exception as e:
            logger.warning(f"LONG signal error: {e}")
            long_prob = 0.0

        # SHORT signal (Top-3 Weighted Ensemble)
        try:
            expected_short_features = len(short_feature_columns)
            short_feat_df = latest[short_feature_columns]

            # Verify we have all required features
            if short_feat_df.shape[1] != expected_short_features:
                logger.warning(f"SHORT features: expected {expected_short_features}, got {short_feat_df.shape[1]}")
                logger.warning(f"  Missing features: {set(short_feature_columns) - set(short_feat_df.columns)}")
                short_prob = 0.0
                raise ValueError(f"Feature mismatch: need {expected_short_features} features")

            short_feat = short_feat_df.values

            # Enhanced Baseline single model prediction
            short_scaled = short_entry_scaler.transform(short_feat)
            short_prob = short_entry_model.predict_proba(short_scaled)[0][1]

        except Exception as e:
            logger.warning(f"SHORT signal error: {e}")
            short_prob = 0.0

        return long_prob, short_prob, df_features

    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        return 0.0, 0.0, None

# =============================================================================
# TRADING LOGIC
# =============================================================================

def check_entry_signal(long_prob, short_prob, position, balance, recent_trades):
    """
    Check for entry signal using Opportunity Gating strategy

    Returns:
        (should_enter, side, reason, sizing_result)
    """
    if position is not None:
        return False, None, "Already in position", None

    # LONG entry (standard)
    if long_prob >= LONG_THRESHOLD:
        sizing_result = sizer.get_position_size_simple(
            capital=balance,
            signal_strength=long_prob,
            leverage=LEVERAGE
        )
        return True, "LONG", f"LONG signal (prob={long_prob:.4f})", sizing_result

    # SHORT entry (gated)
    if short_prob >= SHORT_THRESHOLD:
        # Calculate expected values
        long_ev = long_prob * LONG_AVG_RETURN
        short_ev = short_prob * SHORT_AVG_RETURN
        opportunity_cost = short_ev - long_ev

        # Gate check
        if opportunity_cost > GATE_THRESHOLD:
            sizing_result = sizer.get_position_size_simple(
                capital=balance,
                signal_strength=short_prob,
                leverage=LEVERAGE
            )
            return True, "SHORT", f"SHORT signal (prob={short_prob:.4f}, opp_cost={opportunity_cost:.6f})", sizing_result
        else:
            return False, None, f"SHORT blocked by gate (opp_cost={opportunity_cost:.6f} < {GATE_THRESHOLD})", None

    return False, None, "No signal", None


def calculate_market_volatility(df_features, lookback=20):
    """
    Calculate recent market volatility (rolling std of returns)

    Args:
        df_features: DataFrame with 'close' column
        lookback: Number of candles to look back (default 20 = 1.67 hours for 5min candles)

    Returns:
        volatility: Standard deviation of returns (float)
    """
    try:
        if len(df_features) < lookback:
            lookback = len(df_features)

        if lookback < 2:
            return 0.015  # Default mid-range volatility

        recent_prices = df_features['close'].iloc[-lookback:]
        returns = recent_prices.pct_change().dropna()

        if len(returns) < 2:
            return 0.015

        volatility = float(returns.std())
        return volatility

    except Exception as e:
        logger.warning(f"Volatility calculation error: {e}")
        return 0.015  # Default mid-range volatility


def check_exit_signal(position, current_price, current_time, df_features):
    """
    Check for exit signal using ML Exit + Max Hold Strategy

    Exit Conditions:
    1. ML Exit Model (LONG 0.70, SHORT 0.72) - Primary intelligent exit
    2. Emergency Max Hold (8h) - Capital efficiency

    Note: Emergency Stop Loss (-1.5%) is handled by exchange-level STOP_MARKET order
          No program-level SL check needed (exchange monitors 24/7)

    Returns:
        (should_exit, reason, pnl_info, exit_prob)
    """
    if position is None:
        return False, "No position", None, None

    # Calculate time in position
    # CRITICAL FIX: Convert pandas Timestamp to datetime for comparison
    if isinstance(current_time, pd.Timestamp):
        current_time_dt = current_time.to_pydatetime()
    else:
        current_time_dt = current_time

    try:
        # Try ISO format first (2025-10-17T07:05:00)
        entry_time = datetime.fromisoformat(position['entry_time'])
    except (ValueError, TypeError):
        try:
            # Try space-separated format (2025-10-17 07:05:00)
            entry_time = datetime.strptime(position['entry_time'], '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            logger.error(f"❌ Failed to parse entry_time: {position.get('entry_time')}")
            hours_held = 0
            entry_time = None

    if entry_time:
        try:
            time_delta = current_time_dt - entry_time
            hours_held = time_delta.total_seconds() / 3600
            logger.info(f"⏱️ Time in position: {hours_held:.2f}h (entry: {entry_time}, current: {current_time_dt})")
        except Exception as e:
            logger.error(f"❌ Time calculation error: {e}")
            logger.error(f"   Entry time: {entry_time} (type: {type(entry_time)})")
            logger.error(f"   Current time: {current_time_dt} (type: {type(current_time_dt)})")
            hours_held = 0
    else:
        hours_held = 0

    # Calculate P&L (direct notional value difference)
    entry_notional = position['quantity'] * position['entry_price']
    current_notional = position['quantity'] * current_price

    if position['side'] == "LONG":
        pnl_usd = current_notional - entry_notional
        price_change_pct = (current_price - position['entry_price']) / position['entry_price']
    else:  # SHORT
        pnl_usd = entry_notional - current_notional
        price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

    # Calculate leveraged P&L % (for display/logging)
    leveraged_pnl_pct = price_change_pct * LEVERAGE

    pnl_info = {
        'price_change_pct': price_change_pct,
        'leveraged_pnl_pct': leveraged_pnl_pct,
        'pnl_usd': pnl_usd
    }

    # =========================================================================
    # BACKTEST-ALIGNED EXIT LOGIC (ML + Emergency)
    # =========================================================================

    # 1. ML Exit Model (FIXED THRESHOLDS - matching backtest)
    exit_prob = None
    try:
        # Select appropriate exit model
        if position['side'] == "LONG":
            exit_model = long_exit_model
            exit_scaler = long_exit_scaler
            exit_features_list = long_exit_feature_columns
        else:  # SHORT
            exit_model = short_exit_model
            exit_scaler = short_exit_scaler
            exit_features_list = short_exit_feature_columns

        # Get latest features
        latest = df_features.iloc[-1:].copy()

        # Extract exit features
        exit_features_values = latest[exit_features_list].values
        exit_features_scaled = exit_scaler.transform(exit_features_values)

        # Get exit probability
        exit_prob = exit_model.predict_proba(exit_features_scaled)[0][1]

        # Use FIXED threshold based on position side (matching backtest)
        if position['side'] == 'LONG':
            ml_threshold = ML_EXIT_THRESHOLD_LONG  # 0.70
        else:  # SHORT
            ml_threshold = ML_EXIT_THRESHOLD_SHORT  # 0.72

        logger.info(f"   ML Exit Signal ({position['side']}): {exit_prob:.3f} (exit if >= {ml_threshold:.2f})")

        # Exit if probability exceeds threshold
        if exit_prob >= ml_threshold:
            return True, f"ML Exit ({exit_prob:.3f})", pnl_info, exit_prob

    except Exception as e:
        logger.warning(f"ML Exit error: {e}")

    # 2. Emergency Stop Loss: Handled by Exchange-Level STOP_MARKET order
    # STOP_MARKET order set at entry to trigger at -4% leveraged P&L
    # - LONG 4x: price -1% → leveraged P&L -4% → STOP_MARKET triggers
    # - SHORT 4x: price +1% → leveraged P&L -4% → STOP_MARKET triggers
    # No program-level check needed - exchange monitors 24/7

    # 3. Emergency Max Hold
    logger.info(f"⏱️ Emergency Max Hold Check: {hours_held:.2f}h (trigger: {EMERGENCY_MAX_HOLD_HOURS}h)")
    if hours_held >= EMERGENCY_MAX_HOLD_HOURS:
        logger.warning(f"🚨 EMERGENCY MAX HOLD TRIGGERED: {hours_held:.1f}h >= {EMERGENCY_MAX_HOLD_HOURS}h")
        return True, f"Max Hold ({hours_held:.1f}h)", pnl_info, exit_prob

    # Continue holding
    return False, f"Holding ({leveraged_pnl_pct*100:+.2f}%, {hours_held:.1f}h)", pnl_info, exit_prob


def sync_position_with_exchange(client, state):
    """
    Synchronize bot state with actual exchange position

    Detects Stop Loss triggers, manual closes, and other desync scenarios

    Returns:
        (bool, str): (desync_detected, reason)
    """
    try:
        # Fetch actual position from exchange
        positions = client.exchange.fetch_positions([SYMBOL])
        open_positions = [p for p in positions if float(p.get('contracts', 0)) != 0]

        has_exchange_position = len(open_positions) > 0
        has_state_position = (state.get('position') is not None and
                             state['position'].get('status') == 'OPEN')

        # Case 1: State says OPEN, but exchange has no position (CRITICAL)
        if has_state_position and not has_exchange_position:
            logger.warning("🚨 POSITION DESYNC DETECTED!")
            logger.warning("   State: OPEN | Exchange: CLOSED")
            logger.warning("   Likely cause: Stop Loss triggered, Manual close, or Exchange issue")

            position = state['position']
            position_id = position.get('position_id_exchange')
            order_id = position.get('order_id')

            # Try to get exact close details from Position History API
            # PRIORITY: Use position_id if available (reliable), fallback to order_id (won't work)
            close_details = client.get_position_close_details(
                position_id=position_id,
                order_id=order_id,  # Fallback (won't match, but logged)
                symbol=SYMBOL
            )

            if close_details:
                # Got exact close data from exchange
                logger.info(f"   ✅ Retrieved close details from Position History API")
                logger.info(f"      Exit Price: ${close_details['exit_price']:,.2f}")
                logger.info(f"      Net P&L: ${close_details['net_profit']:,.2f}")
                logger.info(f"      Close Time: {datetime.fromtimestamp(close_details['close_time']/1000)}")

                # Update trade record with exact data
                exit_price = close_details['exit_price']
                pnl_usd = close_details['realized_pnl']
                pnl_usd_net = close_details['net_profit']
                exit_time = datetime.fromtimestamp(close_details['close_time'] / 1000).isoformat()

                # Calculate percentages
                price_diff = exit_price - position.get('entry_price', exit_price)
                price_change_pct = price_diff / position.get('entry_price') if position.get('entry_price') else 0
                leveraged_pnl_pct = price_change_pct * LEVERAGE

                exit_reason = 'Stop Loss Triggered (detected via position sync)'

            else:
                # No close details available - use stored Stop Loss price
                logger.warning(f"   ⚠️  Could not retrieve close details from API")
                logger.warning(f"      Using stored Stop Loss price for estimation")

                # Use stored SL price as better estimate (instead of entry price)
                exit_price = position.get('stop_loss_price', position.get('entry_price', 0))
                entry_price = position.get('entry_price', 0)

                # Calculate P&L using SL price
                if exit_price != entry_price and entry_price > 0:
                    price_diff = exit_price - entry_price
                    price_change_pct = price_diff / entry_price

                    # Adjust sign for SHORT positions
                    if position.get('side') == 'SHORT':
                        price_change_pct = -price_change_pct

                    leveraged_pnl_pct = price_change_pct * position.get('leverage', LEVERAGE)
                    position_value = position.get('position_value', 0)
                    pnl_usd = position_value * leveraged_pnl_pct

                    # Estimate fees (0.05% entry + 0.05% exit)
                    estimated_fees = position_value * 0.001
                    pnl_usd_net = pnl_usd - estimated_fees
                else:
                    # No SL price stored or prices are equal
                    price_diff = 0
                    price_change_pct = 0
                    pnl_usd = 0
                    pnl_usd_net = 0
                    leveraged_pnl_pct = 0

                exit_time = datetime.now().isoformat()
                exit_reason = 'Stop Loss Triggered (estimated from stored SL price)'

            # Update position record
            position['status'] = 'CLOSED'
            position['exit_time'] = exit_time
            position['exit_price'] = exit_price
            position['exit_fee'] = close_details.get('fee', 0) if close_details else 0
            position['price_change_pct'] = price_change_pct
            position['leveraged_pnl_pct'] = leveraged_pnl_pct
            position['pnl_pct'] = leveraged_pnl_pct
            position['pnl_usd'] = pnl_usd
            position['pnl_usd_net'] = pnl_usd_net
            position['exit_reason'] = exit_reason
            position['close_order_id'] = close_details.get('close_order_id', 'N/A') if close_details else 'N/A'

            # 🔧 FIX 2025-11-04: Update existing OPEN trade OR add new trade
            # First, try to find and update existing OPEN trade with matching entry_time or order_id
            trade_updated = False
            for i, trade in enumerate(state.get('trades', [])):
                # Match by order_id (preferred) or entry_time (fallback)
                is_match = (
                    (position.get('order_id') and trade.get('order_id') == position.get('order_id')) or
                    (position.get('entry_time') and trade.get('entry_time') == position.get('entry_time'))
                )

                if is_match and trade.get('status') == 'OPEN':
                    # Update existing OPEN trade to CLOSED
                    state['trades'][i].update({
                        'status': 'CLOSED',
                        'exit_time': exit_time,
                        'close_time': exit_time,  # For monitor compatibility
                        'exit_price': exit_price,
                        'exit_fee': position.get('exit_fee', 0),
                        'price_change_pct': price_change_pct,
                        'leveraged_pnl_pct': leveraged_pnl_pct,
                        'pnl_pct': leveraged_pnl_pct,
                        'pnl_usd': pnl_usd,
                        'pnl_usd_net': pnl_usd_net,
                        'exit_reason': exit_reason,
                        'close_order_id': position.get('close_order_id', 'N/A'),
                        'exchange_reconciled': True  # Mark as reconciled
                    })
                    trade_updated = True
                    logger.info(f"   ✅ Updated existing OPEN trade to CLOSED (order_id: {position.get('order_id')})")

                    # Add to trading_history for monitor
                    if 'trading_history' not in state:
                        state['trading_history'] = []
                    state['trading_history'].append(state['trades'][i].copy())
                    logger.info(f"   📝 Trade added to history (monitor will show it)")
                    break

            # If no existing trade found, add as new trade
            if not trade_updated:
                state.setdefault('trades', []).append(position.copy())
                logger.info(f"   ✅ New trade added to trades array (order_id: {position.get('order_id')})")

                # Add to trading_history
                if 'trading_history' not in state:
                    state['trading_history'] = []
                state['trading_history'].append(position.copy())
                logger.info(f"   📝 Trade added to history (monitor will show it)")

            # Update closed trades count
            state['closed_trades'] = state.get('closed_trades', 0) + 1

            # Update balance accounting
            state['realized_balance'] = state.get('current_balance', 0) + pnl_usd_net
            state['unrealized_pnl'] = 0

            # Clear current position
            state['position'] = None

            # Log trade summary
            logger.info("")
            logger.info("📊 TRADE CLOSED (via desync detection):")
            logger.info(f"   Side: {position.get('side')}")
            logger.info(f"   Entry: ${position.get('entry_price', 0):,.2f} @ {position.get('entry_time', 'N/A')}")
            logger.info(f"   Exit: ${exit_price:,.2f} @ {exit_time}")
            logger.info(f"   P&L: ${pnl_usd_net:+,.2f} ({leveraged_pnl_pct*100:+.2f}%)")
            logger.info(f"   Reason: {exit_reason}")
            logger.info("")

            save_state(state)
            logger.info("✅ Position state synchronized with exchange")

            return True, exit_reason

        # Case 2: State says CLOSED, but exchange has position (RARE)
        elif not has_state_position and has_exchange_position:
            logger.warning("⚠️  ORPHAN POSITION DETECTED on exchange!")
            logger.warning("   State: CLOSED | Exchange: OPEN")
            logger.warning("   This position is not managed by bot - syncing for monitoring")

            pos = open_positions[0]
            side = pos.get('side', 'unknown').upper()
            contracts = float(pos.get('contracts', 0))
            entry_price = float(pos.get('entryPrice', 0))
            notional = float(pos.get('notional', 0))
            unrealized_pnl = float(pos.get('unrealizedPnl', 0))
            leverage_on_exchange = float(pos.get('leverage', 1))

            # ✅ CRITICAL: Get position_id from exchange (for future Position History lookup)
            position_id_exchange = pos.get('id')  # Correct path: pos['id'], not pos['info']['positionId']

            logger.info(f"   Syncing {side} position: {contracts:.6f} BTC @ ${entry_price:,.2f}")
            if position_id_exchange:
                logger.info(f"   Position ID: {position_id_exchange} (will use for close details)")
            else:
                logger.warning(f"   ⚠️  Position ID not available from exchange")

            # Try to fetch actual order details from exchange
            actual_order_id = 'ORPHAN_FROM_EXCHANGE'
            entry_fee = 0.0
            
            try:
                logger.info("   🔍 Attempting to fetch order details from exchange...")
                recent_orders = client.exchange.fetch_closed_orders(SYMBOL, limit=50)
                
                # Search for matching entry order
                for order in reversed(recent_orders):
                    order_contracts = float(order.get('amount', 0))
                    order_price = float(order.get('price', 0))
                    order_side = order.get('side', '').upper()
                    
                    # Match by: quantity, price (within 1%), and side
                    if (abs(order_contracts - contracts) < 0.0001 and
                        abs(order_price - entry_price) / entry_price < 0.01 and
                        order_side == side):
                        
                        actual_order_id = order.get('id', actual_order_id)
                        fee_info = order.get('fee', {})
                        entry_fee = float(fee_info.get('cost', 0))
                        
                        logger.info(f"   ✅ Found matching order!")
                        logger.info(f"      Order ID: {actual_order_id}")
                        logger.info(f"      Entry Fee: ${entry_fee:.4f}")
                        break
                else:
                    logger.warning("   ⚠️  Could not find matching order in recent history")
                    logger.warning("      Using placeholder order_id and fee=0")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not fetch order details: {e}")
                logger.warning("      Using placeholder order_id and fee=0")

            # Reconstruct position for state
            # Calculate position size percentage (leveraged position / balance)
            position_value = notional / leverage_on_exchange if leverage_on_exchange > 0 else notional
            current_balance = state.get('current_balance', 0)  # ✅ FIXED: Use correct key
            # Store as decimal (0.0-1.0) to match codebase convention (multiply by 100 for display only)
            position_size_pct = (notional / current_balance) if current_balance > 0 else 0.0

            position_data = {
                'status': 'OPEN',
                'side': side,
                'entry_time': datetime.now().isoformat(),
                'entry_candle_time': None,
                'entry_price': entry_price,
                'entry_fee': entry_fee,  # ✅ Fee from order history if found
                'entry_candle_idx': 0,
                'entry_long_prob': 0.0,
                'entry_short_prob': 0.0,
                'probability': 0.0,
                'position_size_pct': position_size_pct,  # ✅ FIXED: Calculate from notional and balance
                'position_value': position_value,
                'leveraged_value': notional,
                'quantity': contracts,
                'order_id': actual_order_id,
                'position_id_exchange': position_id_exchange,  # ✅ CRITICAL: Store for future lookup
                'stop_loss_order_id': 'N/A',
                'synced_from_exchange': True
            }

            state['position'] = position_data

            # ✅ CRITICAL FIX: Add to trades array for data consistency
            state.setdefault('trades', []).append(position_data.copy())
            logger.info(f"   ✅ Position added to trades array (position_id: {position_id_exchange or 'N/A'})")

            save_state(state)
            logger.info("✅ Orphan position synced to state AND trades array")
            logger.info(f"   🔑 position_id stored: {position_id_exchange}")
            logger.info("   📊 When position closes, will use position_id for accurate close details")

            return True, 'Orphan position synced'

        # Case 3: Both agree - no desync
        else:
            return False, 'Positions in sync'

    except Exception as e:
        logger.error(f"❌ Position sync error: {e}")
        logger.error(f"   Continuing bot operation")
        return False, f'Sync error: {e}'


# =============================================================================
# MAIN TRADING LOOP
# =============================================================================

def main():
    """Main trading loop"""

    # ===========================================================================
    # DUPLICATE INSTANCE PREVENTION (Enhanced with process name check)
    # ===========================================================================
    logger.info("🔍 Checking for duplicate bot instances...")

    if LOCK_FILE.exists():
        logger.info(f"   Lock file found: {LOCK_FILE}")
        try:
            with open(LOCK_FILE, 'r') as f:
                lock_content = f.read().strip()

            # Parse lock file (format: PID|timestamp or just PID for legacy)
            if '|' in lock_content:
                existing_pid_str, lock_timestamp = lock_content.split('|')
                existing_pid = int(existing_pid_str)
                lock_age = time.time() - float(lock_timestamp)
                logger.info(f"   Lock file PID: {existing_pid} (age: {lock_age:.1f}s)")
            else:
                existing_pid = int(lock_content)
                logger.info(f"   Lock file PID: {existing_pid} (legacy format)")

            # Check if process exists with detailed validation
            is_duplicate = False
            try:
                import psutil
                logger.info(f"   Checking if PID {existing_pid} is running...")

                if psutil.pid_exists(existing_pid):
                    try:
                        proc = psutil.Process(existing_pid)
                        cmdline = " ".join(proc.cmdline())
                        logger.info(f"   Process found: {proc.name()}")
                        logger.info(f"   Command line: {cmdline}")

                        # Verify it's actually our bot
                        if "opportunity_gating_bot_4x.py" in cmdline:
                            is_duplicate = True
                            logger.warning("="*80)
                            logger.warning(f"⚠️ DUPLICATE INSTANCE DETECTED - TERMINATING OLD BOT")
                            logger.warning(f"   Old bot details:")
                            logger.warning(f"   - PID: {existing_pid}")
                            logger.warning(f"   - Process: {proc.name()}")
                            logger.warning(f"   - Command: {cmdline}")
                            logger.warning(f"   - Lock file: {LOCK_FILE}")
                            logger.warning("="*80)

                            try:
                                # Terminate old bot gracefully
                                logger.info(f"   Attempting graceful termination...")
                                proc.terminate()
                                proc.wait(timeout=5)  # Wait up to 5 seconds
                                logger.info(f"✅ Old bot terminated successfully (PID {existing_pid})")
                            except psutil.TimeoutExpired:
                                # Force kill if graceful termination fails
                                logger.warning(f"   Graceful termination timed out, forcing kill...")
                                proc.kill()
                                proc.wait(timeout=2)
                                logger.info(f"✅ Old bot killed successfully (PID {existing_pid})")
                            except Exception as e:
                                logger.error(f"   Failed to terminate old bot: {e}")
                                logger.error(f"   Continuing with new bot anyway...")

                            # Remove old lock file
                            if LOCK_FILE.exists():
                                LOCK_FILE.unlink()
                                logger.info(f"✅ Old lock file removed")

                            # Continue with new bot (do NOT exit)
                        else:
                            logger.info(f"   PID {existing_pid} is not our bot (safe to continue)")
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        logger.info(f"   Cannot access process {existing_pid}: {e}")
                        logger.info(f"   Treating as stale lock file")
                else:
                    logger.info(f"   PID {existing_pid} not running (stale lock file)")

            except ImportError:
                # psutil not available, check using os.kill
                logger.info(f"   psutil not available, using os.kill check...")
                import signal
                try:
                    os.kill(existing_pid, 0)  # Signal 0 = check if process exists

                    # Duplicate found - terminate old bot
                    logger.warning("="*80)
                    logger.warning(f"⚠️ DUPLICATE INSTANCE DETECTED - TERMINATING OLD BOT")
                    logger.warning(f"   Old bot PID: {existing_pid}")
                    logger.warning(f"   Lock file: {LOCK_FILE}")
                    logger.warning("="*80)

                    try:
                        # Terminate old bot (Windows: only SIGTERM/SIGKILL available)
                        logger.info(f"   Terminating old bot (PID {existing_pid})...")

                        # On Windows, use os.kill with signal 0 to check, then terminate
                        # Note: Windows doesn't support SIGTERM gracefully, so we use psutil if available
                        # Since psutil is not available here, we'll use direct kill
                        if sys.platform == 'win32':
                            # Windows: use taskkill for graceful termination
                            import subprocess
                            try:
                                subprocess.run(['taskkill', '/PID', str(existing_pid), '/T', '/F'],
                                             check=True, capture_output=True, timeout=5)
                                logger.info(f"✅ Old bot terminated successfully (PID {existing_pid})")
                            except subprocess.TimeoutExpired:
                                logger.warning(f"   Termination timed out")
                            except subprocess.CalledProcessError as e:
                                logger.error(f"   Termination failed: {e}")
                        else:
                            # Unix: use SIGTERM then SIGKILL
                            os.kill(existing_pid, signal.SIGTERM)
                            time.sleep(2)  # Wait for graceful shutdown
                            try:
                                os.kill(existing_pid, 0)  # Check if still running
                                logger.warning(f"   Process still running, forcing kill...")
                                os.kill(existing_pid, signal.SIGKILL)
                            except OSError:
                                pass  # Process already terminated
                            logger.info(f"✅ Old bot terminated successfully (PID {existing_pid})")
                    except Exception as e:
                        logger.error(f"   Failed to terminate old bot: {e}")
                        logger.error(f"   Continuing with new bot anyway...")

                    # Remove old lock file
                    if LOCK_FILE.exists():
                        LOCK_FILE.unlink()
                        logger.info(f"✅ Old lock file removed")

                    # Continue with new bot (do NOT exit)

                except OSError:
                    logger.info(f"   PID {existing_pid} not running (stale lock file)")

        except Exception as e:
            logger.warning(f"   Lock file check failed: {e}")
            logger.warning(f"   Continuing with caution...")
    else:
        logger.info(f"   No lock file found (first instance)")

    # Create lock file with current PID and timestamp
    current_pid = os.getpid()
    current_time = time.time()
    with open(LOCK_FILE, 'w') as f:
        f.write(f"{current_pid}|{current_time}")
    logger.info(f"✅ Lock file created: PID {current_pid} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info("\n" + "="*80)
    logger.info("BOT CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Strategy: Opportunity Gating + 4x Leverage + COMBINED Exit")
    logger.info(f"")
    logger.info(f"Entry:")
    logger.info(f"  LONG Threshold: {LONG_THRESHOLD}")
    logger.info(f"  SHORT Threshold: {SHORT_THRESHOLD}")
    logger.info(f"  Gate Threshold: {GATE_THRESHOLD}")
    logger.info(f"  Leverage: {LEVERAGE}x")
    logger.info(f"  Position Sizing: Dynamic (20-95%)")
    logger.info(f"")
    logger.info(f"Exit Strategy (ML Exit + Max Hold + Exchange SL):")
    logger.info(f"")
    logger.info(f"  Primary Exits (Program-Level):")
    logger.info(f"    1. ML Exit Model:")
    logger.info(f"       - LONG threshold: {ML_EXIT_THRESHOLD_LONG:.2f}")
    logger.info(f"       - SHORT threshold: {ML_EXIT_THRESHOLD_SHORT:.2f}")
    logger.info(f"    2. Emergency Max Hold: {EMERGENCY_MAX_HOLD_HOURS:.1f}h")
    logger.info(f"")
    logger.info(f"  Emergency Protection (Exchange-Level):")
    logger.info(f"    3. Stop Loss: {abs(EMERGENCY_STOP_LOSS)*100:.1f}% total balance (Balance-Based STOP_MARKET)")
    logger.info(f"       - Dynamic Price SL: Varies by position size (6% / (size × leverage))")
    logger.info(f"       - Example: 50% position → 3.0% price SL, 95% → 1.6% price SL")
    logger.info(f"       - Monitoring: Exchange server 24/7")
    logger.info(f"       - Protection: Survives bot crashes & network failures")
    logger.info(f"")
    logger.info(f"  Note: Fixed TP removed - ML Exit handles all profit-taking")
    logger.info("="*80 + "\n")

    # Validate API keys
    if not API_KEY or not API_SECRET:
        logger.error("❌ API keys not found!")
        logger.error("   Please set keys in config/api_keys.yaml or environment variables")
        logger.error("   BINGX_API_KEY and BINGX_API_SECRET")
        return

    # Initialize
    logger.info(f"Initializing BingX client (Testnet: {USE_TESTNET})...")
    client = BingXClient(api_key=API_KEY, secret_key=API_SECRET, testnet=USE_TESTNET)
    logger.info(f"✅ Client initialized\n")

    state = load_state()

    # Initialize balance tracking on first run, and update on restarts
    try:
        balance_info = client.get_balance()
        current_balance = float(balance_info.get('balance', {}).get('balance', 0))

        # 🔧 FIX 2025-11-10: Improved new session detection
        # Check if this is a new session by comparing session_start and timestamp
        is_new_session = False

        # Case 1: initial_balance is missing or default value (100000.0)
        if 'initial_balance' not in state or state.get('initial_balance') == 100000.0:
            is_new_session = True
            logger.info(f"   🆕 New session detected (initial_balance missing or default)")
        else:
            # Case 2: session_start and timestamp are very close (within 5 seconds)
            try:
                session_start = datetime.fromisoformat(state.get('session_start'))
                timestamp = datetime.fromisoformat(state.get('timestamp'))
                time_diff = abs((timestamp - session_start).total_seconds())
                if time_diff < 5:
                    is_new_session = True
                    logger.info(f"   🆕 New session detected (session just started, {time_diff:.1f}s ago)")
            except:
                pass

        # Set initial_balance from exchange on new sessions
        if is_new_session:
            state['initial_balance'] = current_balance
            # 🔧 FIX 2025-11-10: Also update initial_wallet_balance on new sessions
            state['initial_wallet_balance'] = current_balance
            logger.info(f"   💰 initial_balance set from exchange: ${current_balance:,.2f}")
            logger.info(f"   💰 initial_wallet_balance set from exchange: ${current_balance:,.2f}")
        else:
            logger.info(f"   ♻️  Existing session - preserving initial_balance: ${state.get('initial_balance', 0):,.2f}")

        # ✅ FIXED 2025-10-26: Ensure baseline values exist for reverse calculation
        # If initial_wallet_balance or initial_unrealized_pnl are missing, set to initial_balance
        if 'initial_wallet_balance' not in state:
            state['initial_wallet_balance'] = state.get('initial_balance', current_balance)
            logger.info(f"   💰 initial_wallet_balance initialized: ${state['initial_wallet_balance']:,.2f}")
        if 'initial_unrealized_pnl' not in state:
            state['initial_unrealized_pnl'] = 0  # Default to no position baseline

        # Always update current_balance on startup (includes restarts)
        state['current_balance'] = current_balance
        save_state(state)  # Save immediately to reflect corrected balance
        logger.info(f"✅ Balance initialized: ${current_balance:,.2f}")
        logger.info(f"   Baseline: initial_balance=${state.get('initial_balance', 0):,.2f}, initial_unrealized=${state.get('initial_unrealized_pnl', 0):,.2f}")
    except Exception as e:
        logger.warning(f"Could not get initial balance: {e}")

    # ===========================================================================
    # EXCHANGE RECONCILIATION V2 (Position History API - Ground Truth)
    # ===========================================================================
    logger.info(f"\n🔄 Reconciling state from exchange (V2 - Position History API)...")
    try:
        # ✅ FIXED 2025-10-25: Use 'session_start' not 'start_time' (which doesn't exist)
        # session_start is set on first run and updated by trading_history_reset
        updated_count, new_count = reconcile_state_from_exchange_v2(
            state=state,
            api_client=client,
            bot_start_time=state.get('session_start'),
            days=7
        )
        if updated_count > 0 or new_count > 0:
            save_state(state)
            logger.info(f"✅ State reconciled: {updated_count} updated, {new_count} new trades")
        else:
            logger.info(f"ℹ️  No reconciliation needed (all trades up to date)")
    except Exception as e:
        logger.warning(f"⚠️  Reconciliation failed: {e}")
        logger.warning(f"⚠️  Continuing without reconciliation...")

    # Note: We do NOT set exchange leverage settings
    # Instead, we calculate position size using available_margin × 4x internally
    # This prevents modifying exchange settings and uses margin-based calculation
    logger.info(f"💡 Using {LEVERAGE}x internal leverage calculation (no exchange setting modification)")

    # ===========================================================================
    # CHECK EXISTING POSITIONS ON EXCHANGE
    # ===========================================================================
    logger.info(f"\n🔍 Checking for existing positions on exchange...")
    try:
        positions = client.exchange.fetch_positions([SYMBOL])
        open_positions = [p for p in positions if float(p.get('contracts', 0)) != 0]

        if open_positions:
            # Found existing position - sync with State
            pos = open_positions[0]
            side = pos.get('side', 'unknown').upper()
            contracts = float(pos.get('contracts', 0))
            entry_price = float(pos.get('entryPrice', 0))
            notional = float(pos.get('notional', 0))
            unrealized_pnl = float(pos.get('unrealizedPnl', 0))
            leverage_on_exchange = float(pos.get('leverage', 1))

            # ✅ CRITICAL: Get position_id from exchange (for future Position History lookup)
            position_id_exchange = pos.get('id')  # Correct path: pos['id'], not pos['info']['positionId']

            logger.warning(f"⚠️  EXISTING {side} POSITION FOUND ON EXCHANGE!")
            logger.info(f"   Contracts: {contracts:.6f} BTC")
            logger.info(f"   Entry Price: ${entry_price:,.2f}")
            logger.info(f"   Notional: ${notional:,.2f}")
            logger.info(f"   Leverage (Exchange): {leverage_on_exchange}x")
            logger.info(f"   Unrealized P&L: ${unrealized_pnl:+,.2f}")
            if position_id_exchange:
                logger.info(f"   Position ID: {position_id_exchange} (will use for close details)")
            else:
                logger.warning(f"   ⚠️  Position ID not available from exchange")

            # Reconstruct position data for State
            # Note: We don't have all original data, so we reconstruct what we can

            # Calculate position value and size percentage
            position_value = notional / leverage_on_exchange if leverage_on_exchange > 0 else notional
            current_balance = state.get('current_balance', 0)  # ✅ FIXED: Use correct key
            # Store as decimal (0.0-1.0) to match codebase convention (multiply by 100 for display only)
            position_size_pct = (notional / current_balance) if current_balance > 0 else 0.0

            # Log calculated values for verification
            logger.info(f"   Margin (Position Value): ${position_value:,.2f}")
            logger.info(f"   Leveraged Position: ${notional:,.2f}")
            logger.info(f"   Position Size: {position_size_pct*100:.2f}% of balance")  # Display as percentage

            # ✅ FIXED 2025-11-07: Get actual entry time from exchange if available
            # Try to get entry time from CCXT position fields (timestamp, datetime)
            entry_time_actual = pos.get('datetime')  # ISO8601 string
            if not entry_time_actual and pos.get('timestamp'):
                # Convert milliseconds timestamp to datetime
                from datetime import datetime as dt
                entry_time_actual = dt.fromtimestamp(pos['timestamp'] / 1000).isoformat()

            if entry_time_actual:
                logger.info(f"   ✅ Entry Time (from exchange): {entry_time_actual}")
                entry_time = entry_time_actual
            else:
                logger.warning(f"   ⚠️  Entry time not available from exchange, using bot start time (approximate)")
                entry_time = datetime.now().isoformat()  # Fallback

            position_data = {
                'status': 'OPEN',
                'side': side,
                'entry_time': entry_time,  # ✅ FIXED: Use actual entry time if available
                'entry_candle_time': None,  # Unknown
                'entry_price': entry_price,
                'entry_fee': 0.0,  # Unknown
                'entry_candle_idx': 0,  # Unknown
                'entry_long_prob': 0.0,  # Unknown
                'entry_short_prob': 0.0,  # Unknown
                'probability': 0.0,  # Unknown
                'position_size_pct': position_size_pct,  # ✅ FIXED 2025-10-24: Calculate from notional (leveraged value) and balance
                'position_value': position_value,
                'leveraged_value': notional,
                'quantity': contracts,
                'order_id': 'EXISTING_FROM_EXCHANGE',
                'position_id_exchange': position_id_exchange,  # ✅ CRITICAL: Store for future lookup
                'synced_from_exchange': True  # Mark as synced from exchange
            }

            state['position'] = position_data

            # Clean up stale OPEN trades in trades array
            # (trades that don't match the synced position)
            if state.get('trades'):
                open_trades = [t for t in state['trades'] if t.get('status') == 'OPEN']
                for trade in open_trades:
                    # Skip manual trades (don't modify them)
                    if trade.get('manual_trade', False):
                        logger.info(f"   ⏭️  Skipping manual trade (order: {trade.get('order_id')})")
                        continue

                    # If trade doesn't match synced position → close it as stale
                    if trade.get('quantity') != contracts:
                        logger.warning(f"   ⚠️  Found stale OPEN trade (qty: {trade.get('quantity')} ≠ synced: {contracts})")

                        # Try to get exact close details from API
                        # PRIORITY: Use position_id_exchange if available (reliable), fallback to order_id (won't work)
                        close_details = client.get_position_close_details(
                            position_id=trade.get('position_id_exchange'),
                            order_id=trade.get('order_id'),  # Fallback (won't match, but logged)
                            symbol=SYMBOL
                        )

                        # Check if fees are already reconciled - preserve them if so
                        fees_already_reconciled = trade.get('fees_reconciled', False)
                        if fees_already_reconciled:
                            # Preserve fee fields
                            preserved_entry_fee = trade.get('entry_fee', 0.0)
                            preserved_exit_fee = trade.get('exit_fee', 0.0)
                            preserved_total_fee = trade.get('total_fee', 0.0)
                            logger.info(f"   💾 Preserving reconciled fees (${preserved_total_fee:.2f})")

                        if close_details:
                            # Use exact data from Position History API ✅
                            trade['exit_price'] = close_details['exit_price']
                            trade['pnl_usd'] = close_details['realized_pnl']

                            # Preserve fee-reconciled pnl_usd_net or use API data
                            if fees_already_reconciled:
                                trade['pnl_usd_net'] = trade['pnl_usd'] - preserved_total_fee
                            else:
                                trade['pnl_usd_net'] = close_details['net_profit']

                            exit_time_dt = datetime.fromtimestamp(close_details['close_time'] / 1000)
                            trade['exit_time'] = exit_time_dt.isoformat()

                            # Calculate percentage changes from exact prices
                            price_diff = close_details['exit_price'] - trade.get('entry_price', close_details['exit_price'])
                            trade['price_change_pct'] = price_diff / trade.get('entry_price') if trade.get('entry_price') else 0
                            trade['leveraged_pnl_pct'] = trade['price_change_pct'] * leverage_on_exchange
                            trade['pnl_pct'] = trade['leveraged_pnl_pct']

                            logger.info(f"   ✅ Got exact close data from Position History API")
                            logger.info(f"      Exit: ${close_details['exit_price']:,.2f} | P&L: ${close_details['net_profit']:,.2f}")
                        else:
                            # Fallback: Use current price as estimate
                            trade['exit_price'] = entry_price
                            price_diff = entry_price - trade.get('entry_price', entry_price)
                            trade['price_change_pct'] = price_diff / trade.get('entry_price', entry_price) if trade.get('entry_price') else 0
                            trade['pnl_usd'] = price_diff * trade.get('quantity', 0) if trade.get('side') == 'LONG' else -price_diff * trade.get('quantity', 0)

                            # Preserve fee-reconciled pnl_usd_net or use gross pnl
                            if fees_already_reconciled:
                                trade['pnl_usd_net'] = trade['pnl_usd'] - preserved_total_fee
                            else:
                                trade['pnl_usd_net'] = trade.get('pnl_usd', 0)

                            trade['leveraged_pnl_pct'] = trade.get('price_change_pct', 0) * leverage_on_exchange
                            trade['pnl_pct'] = trade.get('leveraged_pnl_pct', 0)
                            trade['exit_time'] = datetime.now().isoformat()
                            logger.warning(f"   ⚠️ Using estimated close price (API data not available)")

                        # Restore fee fields if they were reconciled
                        if fees_already_reconciled:
                            trade['entry_fee'] = preserved_entry_fee
                            trade['exit_fee'] = preserved_exit_fee
                            trade['total_fee'] = preserved_total_fee
                            trade['fees_reconciled'] = True

                        trade['status'] = 'CLOSED'
                        trade['exit_reason'] = 'Position not found on exchange (stale trade removed during sync)'
                        trade['close_order_id'] = 'N/A'
                        state['closed_trades'] = state.get('closed_trades', 0) + 1
                        logger.info(f"   ✅ Closed stale trade (order: {trade.get('order_id')})")

            # ✅ CRITICAL FIX: Add synced position to trades array for data consistency
            state.setdefault('trades', []).append(position_data.copy())
            logger.info(f"   ✅ Position added to trades array (position_id: {position_id_exchange or 'N/A'})")
            logger.info(f"   📊 When position closes, will use position_id for accurate close details")

            # ==================================================================
            # ✅ CRITICAL: Ensure Stop Loss protection is active
            # ==================================================================
            logger.info(f"\n🛡️  Checking Stop Loss protection...")

            try:
                # Check if Stop Loss order exists
                open_orders = client.exchange.fetch_open_orders(SYMBOL)
                stop_loss_exists = False

                for order in open_orders:
                    # Check multiple ways (CCXT transforms BingX API response):
                    # 1. stopLossPrice field (CCXT conversion)
                    # 2. info.type == 'STOP_MARKET' (raw API)
                    # 3. info.stopPrice (raw API)

                    stop_price = None
                    is_stop_loss = False

                    # Method 1: CCXT stopLossPrice field
                    if order.get('stopLossPrice'):
                        stop_price = order.get('stopLossPrice')
                        is_stop_loss = True

                    # Method 2: Check raw API type
                    raw_type = order.get('info', {}).get('type', '')
                    if 'STOP' in raw_type.upper():
                        is_stop_loss = True
                        if not stop_price:
                            stop_price = float(order.get('info', {}).get('stopPrice', 0))

                    # Method 3: CCXT stopPrice (fallback)
                    if not stop_price and order.get('stopPrice'):
                        stop_price = order.get('stopPrice')
                        is_stop_loss = True

                    if is_stop_loss and stop_price:
                        stop_loss_exists = True
                        logger.info(f"   ✅ Stop Loss order found: ${stop_price:,.2f}")
                        logger.info(f"      Order ID: {order.get('id', 'N/A')}")
                        logger.info(f"      Detection: {raw_type} / stopLossPrice={order.get('stopLossPrice')}")

                        # Store in position data
                        position_data['stop_loss_price'] = stop_price
                        state['position']['stop_loss_price'] = stop_price
                        break

                if not stop_loss_exists:
                    logger.warning(f"   ⚠️  No Stop Loss found - Setting up protection now...")

                    # Calculate Balance-Based Stop Loss
                    # 🔴 CRITICAL FIX 2025-11-13: Minimum 2.5% SL distance
                    # Previous: 1.08-2.10% (too tight, 33% LONG hit SL)
                    # Solution: max(2.5%, calculated) to prevent tight SLs on large positions
                    price_sl_pct = max(0.025, abs(EMERGENCY_STOP_LOSS) / position_size_pct)

                    # Calculate stop price based on side
                    if side == "LONG":
                        stop_loss_price = entry_price * (1 - price_sl_pct)
                        stop_order_side = "SELL"
                    else:  # SHORT
                        stop_loss_price = entry_price * (1 + price_sl_pct)
                        stop_order_side = "BUY"

                    logger.info(f"   📊 Stop Loss calculation:")
                    logger.info(f"      Balance SL: {abs(EMERGENCY_STOP_LOSS)*100:.1f}% of balance")
                    logger.info(f"      Position Size: {position_size_pct*100:.2f}% of balance")
                    logger.info(f"      Price SL: {price_sl_pct*100:.3f}%")
                    logger.info(f"      Stop Price: ${stop_loss_price:,.2f}")

                    # Create STOP_MARKET order
                    stop_loss_order = client.create_order(
                        symbol=SYMBOL,
                        side=stop_order_side,
                        position_side="BOTH",
                        order_type="STOP_MARKET",
                        quantity=contracts,
                        stop_price=stop_loss_price
                    )

                    logger.info(f"   ✅ Stop Loss created: ${stop_loss_price:,.2f}")
                    logger.info(f"      Order ID: {stop_loss_order.get('id', 'N/A')}")
                    logger.info(f"   🛡️  Exchange-level protection now active (24/7 monitoring)")

                    # Store in position data
                    position_data['stop_loss_price'] = stop_loss_price
                    state['position']['stop_loss_price'] = stop_loss_price

            except Exception as e:
                logger.error(f"   ❌ Failed to setup Stop Loss: {e}")
                logger.error(f"   ⚠️  MANUAL INTERVENTION REQUIRED - Position has no Stop Loss protection!")

            logger.info("")
            # ==================================================================

            save_state(state)
            logger.info(f"✅ Position synced to State - Bot will manage this position")
        else:
            logger.info(f"   ✅ No existing positions on exchange")
            # Ensure State reflects no position
            if state.get('position'):
                logger.info(f"   ⚠️  State had a position but exchange doesn't - clearing State")
                state['position'] = None

            # Clean up ALL OPEN trades when no position exists on exchange
            if state.get('trades'):
                open_trades = [t for t in state['trades'] if t.get('status') == 'OPEN']
                if open_trades:
                    logger.warning(f"   ⚠️  Found {len(open_trades)} stale OPEN trades (no position on exchange)")
                    for trade in open_trades:
                        # Skip manual trades (don't modify them)
                        if trade.get('manual_trade', False):
                            logger.info(f"   ⏭️  Skipping manual trade (order: {trade.get('order_id')})")
                            continue
                        # Try to get exact close details from API
                        # PRIORITY: Use position_id_exchange if available (reliable), fallback to order_id (won't work)
                        close_details = client.get_position_close_details(
                            position_id=trade.get('position_id_exchange'),
                            order_id=trade.get('order_id'),  # Fallback (won't match, but logged)
                            symbol=SYMBOL
                        )

                        # Check if fees are already reconciled - preserve them if so
                        fees_already_reconciled = trade.get('fees_reconciled', False)
                        if fees_already_reconciled:
                            # Preserve fee fields
                            preserved_entry_fee = trade.get('entry_fee', 0.0)
                            preserved_exit_fee = trade.get('exit_fee', 0.0)
                            preserved_total_fee = trade.get('total_fee', 0.0)
                            logger.info(f"   💾 Preserving reconciled fees (${preserved_total_fee:.2f})")

                        if close_details:
                            # Use exact data from Position History API ✅
                            trade['exit_price'] = close_details['exit_price']
                            trade['pnl_usd'] = close_details['realized_pnl']

                            # Preserve fee-reconciled pnl_usd_net or use API data
                            if fees_already_reconciled:
                                trade['pnl_usd_net'] = trade['pnl_usd'] - preserved_total_fee
                            else:
                                trade['pnl_usd_net'] = close_details['net_profit']

                            exit_time_dt = datetime.fromtimestamp(close_details['close_time'] / 1000)
                            trade['exit_time'] = exit_time_dt.isoformat()

                            # Calculate percentage changes from exact prices
                            price_diff = close_details['exit_price'] - trade.get('entry_price', close_details['exit_price'])
                            trade['price_change_pct'] = price_diff / trade.get('entry_price') if trade.get('entry_price') else 0
                            trade['leveraged_pnl_pct'] = trade['price_change_pct'] * leverage_on_exchange
                            trade['pnl_pct'] = trade['leveraged_pnl_pct']

                            logger.info(f"   ✅ Got exact close data from Position History API")
                            logger.info(f"      Exit: ${close_details['exit_price']:,.2f} | P&L: ${close_details['net_profit']:,.2f}")
                        else:
                            # Fallback: Assume breakeven
                            trade['exit_price'] = trade.get('entry_price', 0)
                            trade['price_change_pct'] = 0
                            trade['pnl_usd'] = 0

                            # Preserve fee-reconciled pnl_usd_net or use zero
                            if fees_already_reconciled:
                                trade['pnl_usd_net'] = -preserved_total_fee  # Only fee cost
                            else:
                                trade['pnl_usd_net'] = 0

                            trade['leveraged_pnl_pct'] = 0
                            trade['pnl_pct'] = 0
                            trade['exit_time'] = datetime.now().isoformat()
                            logger.warning(f"   ⚠️ Using breakeven estimate (API data not available)")

                        # Restore fee fields if they were reconciled
                        if fees_already_reconciled:
                            trade['entry_fee'] = preserved_entry_fee
                            trade['exit_fee'] = preserved_exit_fee
                            trade['total_fee'] = preserved_total_fee
                            trade['fees_reconciled'] = True

                        trade['status'] = 'CLOSED'
                        trade['exit_reason'] = 'No position on exchange (stale trade cleaned up)'
                        trade['close_order_id'] = 'N/A'
                        state['closed_trades'] = state.get('closed_trades', 0) + 1
                        logger.info(f"   ✅ Closed stale trade (order: {trade.get('order_id')})")

            save_state(state)

    except Exception as e:
        logger.error(f"   ❌ Error checking positions: {e}")
        logger.error(f"   Continuing, but position sync may be inaccurate")

    logger.info("")

    candle_counter = 0
    BOT_START_TIME = datetime.now()  # Track bot start time for warmup period
    logger.info(f"⏰ Bot start time: {BOT_START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"⏸️  Warmup period: {WARMUP_PERIOD_MINUTES} minutes (entry signals will be ignored)")
    logger.info(f"🕐 Timing: 정각 동기화 (5분 정각 +1초) - 완성된 캔들만 사용")

    # Balance reconciliation tracking
    last_balance_check = time.time()  # Initialize balance check timer
    logger.info(f"💰 Balance reconciliation enabled (check interval: {BALANCE_CHECK_INTERVAL/3600:.1f}h, threshold: {BALANCE_RECONCILIATION_THRESHOLD_PCT}%)")

    # Fee reconciliation tracking
    last_fee_check = time.time()  # Initialize fee check timer
    if FEE_RECONCILIATION_ENABLED:
        logger.info(f"💵 Fee reconciliation enabled (check interval: {FEE_CHECK_INTERVAL/60:.0f}min, symbol: {FEE_RECONCILIATION_SYMBOL})")

    try:
        while True:
            try:
                # =================================================================
                # STEP 1: 정각 동기화 - 다음 5분 정각 +1초까지 대기
                # =================================================================
                wait_for_next_candle(interval_minutes=5, post_delay_seconds=1)

                # =================================================================
                # STEP 2: 캔들 데이터 가져오기 + 예상 캔들 검증 + 재시도
                # =================================================================
                # FIXED 2025-10-26: UTC 사용 (백테스트와 동일)
                current_time = datetime.utcnow()  # UTC

                # 예상되는 최신 캔들 시간 계산 (시작 시간 기준, UTC)
                # ⚠️ CORRECTED (2025-10-22): BingX API는 캔들의 시작 시간을 타임스탬프로 사용
                # 예: 14:24:03 UTC 요청 시
                #   - API는 14:20:00 UTC 캔들을 반환 (진행 중, 14:25:00에 완성 예정)
                #   - 이것이 현재 시점에서 가장 최신 캔들
                #   - filter_completed_candles()에서 진행 중 캔들 제거
                # 따라서 예상 최신 = 현재 5분 간격의 시작 (진행 중 캔들 포함)
                expected_candle = current_time.replace(second=0, microsecond=0)
                expected_candle = expected_candle - timedelta(minutes=expected_candle.minute % 5)

                # KST 변환 (로깅용)
                import pytz
                kst = pytz.timezone('Asia/Seoul')
                current_time_kst = pytz.utc.localize(current_time).astimezone(kst).replace(tzinfo=None)
                expected_candle_kst = pytz.utc.localize(expected_candle).astimezone(kst).replace(tzinfo=None)

                logger.info(f"\n{'='*80}")
                logger.info(f"📊 캔들 데이터 요청 ({DATA_SOURCE} 모드)")
                logger.info(f"{'='*80}")
                logger.info(f"   현재 시간: {current_time_kst.strftime('%H:%M:%S')} KST ({current_time.strftime('%H:%M:%S')} UTC)")
                logger.info(f"   예상 최신 캔들: {expected_candle_kst.strftime('%H:%M:%S')} KST ({expected_candle.strftime('%H:%M:%S')} UTC)")

                # =================================================================
                # STEP 2.5: 데이터 가져오기 (CSV 또는 API)
                # =================================================================
                # 데이터 가져오기: CSV 우선, API fallback
                df = None
                if DATA_SOURCE == "CSV":
                    # CSV 업데이트 시도
                    update_csv_if_needed(CSV_DATA_FILE, CSV_UPDATE_SCRIPT, current_time)

                    # CSV 로드 시도
                    df = load_from_csv(CSV_DATA_FILE, MAX_DATA_CANDLES, current_time)

                    if df is None:
                        logger.warning("   ⚠️ CSV load failed - falling back to API")

                # API 모드 또는 CSV fallback
                if df is None:
                    logger.info("   📡 Fetching from API...")
                    df = fetch_and_validate_candles(
                        client=client,
                        symbol=SYMBOL,
                        interval=CANDLE_INTERVAL,
                        limit=MAX_DATA_CANDLES,
                        expected_candle_time=expected_candle,
                        max_retries=5
                    )

                if df is None:
                    logger.error("   ❌ 데이터 가져오기 실패 (CSV + API 모두 실패)")
                    logger.info("   다음 정각까지 대기 후 재시도...\n")
                    continue

                # =================================================================
                # STEP 3: 완성된 캔들 필터링 + 최신성 검증
                # =================================================================
                df_completed, is_fresh = filter_completed_candles(
                    df=df,
                    current_time=current_time,
                    interval_minutes=5,
                    max_age_minutes=30
                )

                if len(df_completed) == 0:
                    logger.error("   ❌ 완성된 캔들 없음")
                    logger.info("   다음 정각까지 대기 후 재시도...\n")
                    continue

                if not is_fresh:
                    logger.warning("   ⚠️  데이터가 오래되었지만 계속 진행")

                # 완성된 캔들 사용
                df = df_completed

                # 🔍 데이터 검증
                logger.info(f"\n{'='*80}")
                logger.info(f"🔍 최종 데이터 검증")
                logger.info(f"{'='*80}")
                logger.info(f"   총 캔들 수: {len(df)}")
                logger.info(f"   첫 캔들 (KST): {df.iloc[0]['timestamp'].strftime('%Y-%m-%d %H:%M')} | ${df.iloc[0]['close']:,.1f}")
                logger.info(f"   마지막 캔들 (KST): {df.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M')} | ${df.iloc[-1]['close']:,.1f}")

                # Verify timestamps are ascending
                is_sorted = df['timestamp'].is_monotonic_increasing
                logger.info(f"   시간순 정렬: {is_sorted} {'✅' if is_sorted else '❌ ERROR!'}")

                if not is_sorted:
                    logger.error(f"   ❌ CRITICAL: 데이터 순서 오류!")
                    logger.error(f"   특징 계산이 잘못될 수 있음")
                    continue

                current_price = float(df.iloc[-1]['close'])
                latest_candle_time = df.iloc[-1]['timestamp']
                candle_counter = len(df)

                logger.info(f"   현재 가격: ${current_price:,.1f}")
                logger.info(f"   ✅ 데이터 검증 완료\n")

                # Get balance
                balance_info = client.get_balance()
                balance = float(balance_info.get('balance', {}).get('balance', 0))

                # Update state with current info
                state['current_balance'] = balance
                state['timestamp'] = datetime.now().isoformat()

                # 🔄 Position Sync: Detect Stop Loss triggers and desync scenarios
                try:
                    desync_detected, reason = sync_position_with_exchange(client, state)
                    if desync_detected:
                        logger.info(f"🔄 Position sync result: {reason}")
                        # Save state after sync changes
                        save_state(state)
                except Exception as e:
                    logger.error(f"❌ Position sync error: {e}")
                    logger.error(f"   Continuing bot operation despite sync failure")

                # Get signals (with feature caching)
                long_prob, short_prob, df_features = get_signals(df)

                # 🔧 CRITICAL FIX 2025-11-03: Log calculated features for backtest validation
                # This enables exact backtest-production comparison by logging feature values
                try:
                    features_log_dir = Path("logs/production_features")
                    features_log_dir.mkdir(parents=True, exist_ok=True)

                    # Log latest candle features to daily CSV
                    if df_features is not None and len(df_features) > 0:
                        latest_features = df_features.iloc[[-1]].copy()  # Keep as DataFrame
                        # Use timezone-naive datetime to avoid Windows CSV errors
                        latest_features['logged_at_utc'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                        latest_features['logged_at_kst'] = latest_candle_time.strftime('%Y-%m-%d %H:%M:%S')

                        today_str = datetime.utcnow().strftime('%Y%m%d')
                        features_file = features_log_dir / f"features_{today_str}.csv"

                        # Append to daily file
                        if features_file.exists():
                            latest_features.to_csv(features_file, mode='a', header=False, index=False)
                        else:
                            latest_features.to_csv(features_file, mode='w', header=True, index=False)

                        logger.debug(f"✅ Features logged: {len(df_features.columns)} features @ {latest_candle_time.strftime('%H:%M:%S')} KST")
                except Exception as e:
                    logger.warning(f"⚠️  Feature logging failed (non-critical): {e}")

                # Store latest signals
                state['latest_signals'] = {
                    'entry': {
                        'long_prob': float(long_prob),
                        'short_prob': float(short_prob),
                        'long_threshold': LONG_THRESHOLD,
                        'short_threshold': SHORT_THRESHOLD
                    },
                    'exit': {}
                }

                # Log current status
                logger.info(f"[Candle {latest_candle_time.strftime('%H:%M:%S')} KST] Price: ${current_price:,.1f} | Balance: ${balance:,.2f} | LONG: {long_prob:.4f} | SHORT: {short_prob:.4f}")

                # Check for exit first
                if state['position'] is not None:
                    # Use actual current time for emergency max hold check, not candle timestamp
                    actual_current_time = datetime.now()
                    should_exit, exit_reason, pnl_info, exit_prob = check_exit_signal(state['position'], current_price, actual_current_time, df_features)

                    # Store exit signal information for monitor (with dynamic threshold info)
                    if exit_prob is not None:
                        # Calculate current volatility for display
                        try:
                            current_vol = calculate_market_volatility(df_features)
                            # Get side-specific base threshold
                            if state['position']['side'] == 'LONG':
                                base_threshold = ML_EXIT_THRESHOLD_BASE_LONG
                            else:
                                base_threshold = ML_EXIT_THRESHOLD_BASE_SHORT

                            # ⚠️ DISABLED 2025-10-25: Adaptive threshold disabled
                            # Always use base threshold (0.725) regardless of volatility
                            current_threshold = base_threshold

                            # Old adaptive logic (disabled):
                            # if current_vol > VOLATILITY_HIGH:
                            #     current_threshold = ML_THRESHOLD_HIGH_VOL
                            # elif current_vol < VOLATILITY_LOW:
                            #     current_threshold = ML_THRESHOLD_LOW_VOL
                            # else:
                            #     current_threshold = base_threshold
                        except:
                            # Default to LONG threshold on error
                            current_threshold = ML_EXIT_THRESHOLD_BASE_LONG

                        state['latest_signals']['exit'] = {
                            'exit_prob': float(exit_prob),
                            'exit_threshold_base_long': ML_EXIT_THRESHOLD_BASE_LONG,
                            'exit_threshold_base_short': ML_EXIT_THRESHOLD_BASE_SHORT,
                            'exit_threshold_current': current_threshold,
                            'volatility': current_vol if 'current_vol' in locals() else 0.015,
                            'position_side': state['position']['side'],
                            'strategy': 'COMBINED'
                        }

                    # Store unrealized P&L from actual position calculation
                    if pnl_info:
                        state['unrealized_pnl'] = pnl_info['pnl_usd']

                    # Save state even if not exiting (to update balance, timestamp, signals for monitor)
                    if not should_exit or not pnl_info:
                        save_state(state)

                    if should_exit and pnl_info:
                        # Execute real position close via BingX API
                        position = state['position']

                        try:
                            logger.info(f"📡 Closing {position['side']} position: {position['quantity']:.6f} BTC @ market price")

                            # STEP 1: Cancel pending Stop Loss order (if ML Exit or Max Hold)
                            if position.get('stop_loss_order_id') and position['stop_loss_order_id'] != 'N/A':
                                logger.info(f"🗑️ Cancelling Stop Loss order...")
                                try:
                                    cancel_result = client.cancel_position_orders(
                                        symbol=SYMBOL,
                                        order_ids=[position['stop_loss_order_id']]
                                    )
                                    if cancel_result['cancelled']:
                                        logger.info(f"✅ SL order cancelled: {cancel_result['cancelled'][0]}")
                                    elif cancel_result['failed']:
                                        logger.warning(f"⚠️ SL cancel failed (order may be filled): {cancel_result['failed'][0]}")
                                        logger.info(f"ℹ️ Continuing with position close anyway")
                                except Exception as e:
                                    logger.error(f"❌ SL cancel error: {e}")
                                    logger.info(f"ℹ️ Continuing with position close (SL may already be filled)")
                            else:
                                logger.info(f"ℹ️ No Stop Loss order to cancel (may have been triggered by exchange)")

                            # STEP 2: Close position via API
                            close_result = client.close_position(
                                symbol=SYMBOL,
                                position_side=position['side'],
                                quantity=position['quantity']
                            )

                            close_order_id = close_result.get('id', 'N/A')
                            logger.info(f"✅ Position closed: {close_order_id}")

                            # IMPROVED: Use Exchange Ground Truth via Position History API
                            actual_exit_price = current_price  # Default fallback
                            actual_exit_time = datetime.now()  # Default fallback
                            entry_fee = float(position.get('entry_fee', 0))  # From position data
                            exit_fee = 0  # Default fallback
                            actual_pnl_usd = 0  # Will get from API
                            actual_leveraged_pnl_pct = 0
                            exchange_ground_truth = False  # Track if we got exchange data

                            try:
                                # Wait for exchange to record position closure
                                from time import sleep
                                sleep(2.0)  # Increased from 0.5s to 2.0s for reliability

                                # PRIMARY: Use Position History API for ground truth
                                logger.info(f"📊 Fetching position close details from exchange...")
                                close_details = client.get_position_close_details(
                                    position_id=position.get('position_id_exchange'),
                                    symbol=SYMBOL
                                )

                                if close_details and close_details.get('exit_price'):
                                    # ✅ Got exchange ground truth!
                                    actual_exit_price = close_details['exit_price']
                                    actual_pnl_usd = close_details['realized_pnl']  # From exchange (includes slippage)
                                    pnl_usd_net = close_details['net_profit']  # After fees (from exchange)
                                    actual_exit_time = datetime.fromtimestamp(close_details['close_time'] / 1000)

                                    # Calculate fees from exchange data
                                    total_fee = actual_pnl_usd - pnl_usd_net
                                    exit_fee = total_fee - entry_fee if total_fee > entry_fee else 0

                                    # Calculate actual price change and leveraged return
                                    if position['side'] == "LONG":
                                        actual_price_change = (actual_exit_price - position['entry_price']) / position['entry_price']
                                    else:  # SHORT
                                        actual_price_change = (position['entry_price'] - actual_exit_price) / position['entry_price']

                                    actual_leveraged_pnl_pct = actual_price_change * LEVERAGE
                                    exchange_ground_truth = True

                                    logger.info(f"✅ Exchange ground truth: Price=${actual_exit_price:,.2f}")
                                    logger.info(f"   Realized P&L: ${actual_pnl_usd:+.2f} (gross)")
                                    logger.info(f"   Net P&L: ${pnl_usd_net:+.2f} (after fees)")
                                    logger.info(f"   Total Fees: ${total_fee:.4f} (Entry: ${entry_fee:.4f}, Exit: ${exit_fee:.4f})")

                                else:
                                    # FALLBACK: Use fetch_my_trades (old method)
                                    logger.warning(f"⚠️  Position History not available, using fetch_my_trades fallback")

                                    recent_fills = client.exchange.fetch_my_trades(
                                        symbol='BTC/USDT:USDT',
                                        limit=10
                                    )

                                    # Find exit trade matching our close order ID
                                    exit_trade_found = False
                                    for fill in recent_fills:
                                        if fill.get('order') == close_order_id:
                                            # Extract exit details from filled trade
                                            actual_exit_price = float(fill.get('price', actual_exit_price))
                                            actual_exit_time = datetime.fromtimestamp(fill.get('timestamp', 0) / 1000)

                                            # Extract exit fee
                                            if 'fee' in fill and isinstance(fill['fee'], dict):
                                                exit_fee = float(fill['fee'].get('cost', 0))
                                            elif 'info' in fill and 'commission' in fill['info']:
                                                exit_fee = abs(float(fill['info']['commission']))

                                            exit_trade_found = True
                                            logger.info(f"📊 Exit fill: Price=${actual_exit_price:,.2f}, Fee=${exit_fee:.4f}")
                                            break

                                    if not exit_trade_found:
                                        logger.warning(f"⚠️  Could not find exit trade for order {close_order_id}")

                                    # Calculate P&L using actual prices and fees from API
                                    entry_notional = position['quantity'] * position['entry_price']
                                    exit_notional = position['quantity'] * actual_exit_price

                                    if position['side'] == "LONG":
                                        gross_pnl = exit_notional - entry_notional
                                        actual_price_change = (actual_exit_price - position['entry_price']) / position['entry_price']
                                    else:  # SHORT
                                        gross_pnl = entry_notional - exit_notional
                                        actual_price_change = (position['entry_price'] - actual_exit_price) / position['entry_price']

                                    # P&L BEFORE fees (for leverage calculation)
                                    actual_pnl_usd = gross_pnl

                                    # Calculate leveraged return % (based on gross P&L)
                                    actual_leveraged_pnl_pct = actual_price_change * LEVERAGE

                                    # Calculate total fees
                                    total_fee = entry_fee + exit_fee
                                    pnl_usd_net = actual_pnl_usd - total_fee

                                    logger.info(f"📊 Trade summary: Entry Fee=${entry_fee:.4f}, Exit Fee=${exit_fee:.4f}")
                                    logger.info(f"📊 Gross P&L=${gross_pnl:+.2f} (before fees)")

                            except Exception as e:
                                logger.warning(f"⚠️ Error fetching trade details: {e}")
                                logger.warning(f"   Using calculated values")

                                # Fallback calculation
                                entry_notional = position['quantity'] * position['entry_price']
                                exit_notional = position['quantity'] * actual_exit_price

                                if position['side'] == "LONG":
                                    actual_pnl_usd = exit_notional - entry_notional
                                    actual_price_change = (actual_exit_price - position['entry_price']) / position['entry_price']
                                else:
                                    actual_pnl_usd = entry_notional - exit_notional
                                    actual_price_change = (position['entry_price'] - actual_exit_price) / position['entry_price']

                                actual_leveraged_pnl_pct = actual_price_change * LEVERAGE
                                total_fee = entry_fee + exit_fee
                                pnl_usd_net = actual_pnl_usd - total_fee

                            # Ensure total_fee and pnl_usd_net are set (may already be set by exchange ground truth path)
                            if 'total_fee' not in locals():
                                total_fee = entry_fee + exit_fee
                                pnl_usd_net = actual_pnl_usd - total_fee

                            # Calculate hold time using actual exit time from exchange
                            entry_time_obj = datetime.fromisoformat(position['entry_time'])
                            hold_seconds = (actual_exit_time - entry_time_obj).total_seconds()
                            hold_minutes = hold_seconds / 60
                            hold_candles = int(hold_minutes / 5)  # 5-minute candles

                            # Calculate ROI (Return on Investment) - fee-inclusive actual return
                            roi = (pnl_usd_net / float(position['position_value'])) * 100 if position.get('position_value', 0) > 0 else 0

                            # Find and update the open trade
                            for i, trade in enumerate(state['trades']):
                                if trade.get('status') == 'OPEN' and trade.get('entry_time') == position['entry_time']:
                                    # Update with exit information from EXCHANGE API
                                    state['trades'][i].update({
                                        'status': 'CLOSED',
                                        'exit_time': actual_exit_time.isoformat(),  # ← ACTUAL time from exchange
                                        'close_time': actual_exit_time.isoformat(),  # 🔧 FIX 2025-11-03: Add close_time for monitor compatibility
                                        'exit_price': float(actual_exit_price),  # ← ACTUAL price from exchange
                                        'exit_fee': float(exit_fee),  # ← ACTUAL fee from exchange
                                        'total_fee': float(total_fee),  # Entry + Exit fees
                                        'price_change_pct': actual_price_change,  # ← ACTUAL price change
                                        'leveraged_pnl_pct': actual_leveraged_pnl_pct,  # ← ACTUAL leveraged P&L (fee excluded)
                                        'pnl_usd': actual_pnl_usd,  # ← ACTUAL P&L (before fees)
                                        'pnl_usd_net': float(pnl_usd_net),  # ← ACTUAL P&L after fees
                                        'pnl_pct': actual_leveraged_pnl_pct,  # ← Legacy: leveraged P&L % (keep for compatibility)
                                        'roi': roi,  # ← NEW: Actual return on investment (fee-inclusive)
                                        'hold_candles': hold_candles,  # ← ACTUAL hold time from exchange timestamps
                                        'exit_reason': exit_reason,
                                        'close_order_id': close_order_id,
                                        'exchange_reconciled': exchange_ground_truth  # 🔧 FIX 2025-11-03: Mark as exchange-reconciled
                                    })
                                    break

                            # Update stats (use net P&L after fees)
                            state['stats']['total_trades'] += 1
                            state['closed_trades'] = state.get('closed_trades', 0) + 1
                            if position['side'] == "LONG":
                                state['stats']['long_trades'] += 1
                            else:
                                state['stats']['short_trades'] += 1

                            # Win/Loss based on net P&L (after fees)
                            if pnl_usd_net > 0:
                                state['stats']['wins'] += 1
                            else:
                                state['stats']['losses'] += 1

                            # Track net P&L (after fees) for accurate statistics
                            state['stats']['total_pnl_usd'] += pnl_usd_net
                            state['stats']['total_pnl_pct'] += actual_leveraged_pnl_pct

                            # Log exit (using ACTUAL values from exchange API)
                            logger.info(f"🚪 EXIT {position['side']}: {exit_reason}")
                            logger.info(f"   Entry: ${position['entry_price']:,.1f} @ {position['entry_time']}")
                            logger.info(f"   Exit: ${actual_exit_price:,.1f} @ {actual_exit_time.isoformat()}")
                            logger.info(f"   Hold Time: {hold_candles} candles ({hold_minutes:.1f} minutes)")
                            logger.info(f"   P&L: {actual_leveraged_pnl_pct*100:+.2f}% (${actual_pnl_usd:+,.2f})")
                            logger.info(f"   Fees: ${total_fee:+,.2f} (Entry: ${entry_fee:.2f} + Exit: ${exit_fee:.2f})")
                            logger.info(f"   Net P&L: ${pnl_usd_net:+,.2f}")
                            logger.info(f"   Stats: {state['stats']['wins']}/{state['stats']['total_trades']} wins ({state['stats']['wins']/max(state['stats']['total_trades'],1)*100:.1f}%)")
                            logger.info(f"   Total Net P&L: ${state['stats']['total_pnl_usd']:+,.2f}")
                            logger.info(f"   Close Order ID: {close_order_id}")

                            # Add closed trade to trading_history for monitor
                            if 'trading_history' not in state:
                                state['trading_history'] = []

                            # Find the closed trade and add to history
                            for trade in state['trades']:
                                if trade.get('status') == 'CLOSED' and trade.get('entry_time') == position['entry_time']:
                                    state['trading_history'].append(trade.copy())
                                    logger.info(f"📝 Trade added to history (monitor will show it)")
                                    break

                            # Clear position
                            state['position'] = None
                            save_state(state)

                        except Exception as e:
                            logger.error(f"❌ Position close failed: {e}")
                            logger.error(f"   Will retry on next cycle")
                            logger.error(f"   Position: {position['side']}, Quantity: {position['quantity']:.6f}, Reason: {exit_reason}")
                            # Don't clear position - will retry next iteration

                # Check for entry
                else:
                    # No position, unrealized P&L is 0
                    state['unrealized_pnl'] = 0.0

                    # Check if still in warmup period
                    time_since_start = (datetime.now() - BOT_START_TIME).total_seconds() / 60
                    if time_since_start < WARMUP_PERIOD_MINUTES:
                        logger.info(f"⏸️  Warmup period: {time_since_start:.1f}/{WARMUP_PERIOD_MINUTES} min - ignoring entry signals")
                        save_state(state)  # Save state to update monitor
                        continue  # 정각 동기화: 다음 루프에서 wait_for_next_candle이 대기

                    recent_trades = state['trades'][-10:] if len(state['trades']) > 0 else []
                    should_enter, side, entry_reason, sizing_result = check_entry_signal(
                        long_prob, short_prob, state['position'], balance, recent_trades
                    )

                    if should_enter and sizing_result:
                        # Calculate quantity
                        quantity = sizing_result['leveraged_value'] / current_price

                        # Execute real order via BingX API with Stop Loss protection
                        try:
                            logger.info(f"📡 Placing {side} order: {quantity:.6f} BTC @ market price")
                            logger.info(f"🛡️ With exchange-level Stop Loss protection (ML Exit for TP)")

                            # Place entry order WITH Stop Loss protection (Balance-Based SL)
                            protection_result = client.enter_position_with_protection(
                                symbol=SYMBOL,
                                side=side,
                                quantity=quantity,
                                entry_price=current_price,
                                leverage=LEVERAGE,
                                balance_sl_pct=EMERGENCY_STOP_LOSS,  # -6% total balance
                                current_balance=state['current_balance'],
                                position_size_pct=sizing_result['position_size_pct']
                            )

                            # Extract results
                            order_result = protection_result['entry_order']
                            stop_loss_order = protection_result['stop_loss_order']

                            logger.info(f"✅ Entry executed: {order_result.get('id', 'N/A')}")
                            logger.info(f"✅ Stop Loss order: {stop_loss_order.get('id', 'N/A')}")

                            # Extract actual fill information from API response
                            # Market orders may not have 'price' in initial response - use current price as fallback
                            actual_entry_price = order_result.get('price')
                            if actual_entry_price is None:
                                actual_entry_price = current_price
                            # Ensure we have a valid price
                            if actual_entry_price is None:
                                raise ValueError("Cannot determine entry price from order response or current market data")

                            # FIXED: Fee handling - fetch actual fee from trade history
                            # create_order() response doesn't always include fee, so fetch from trades
                            # FIXED 2025-11-03: Increased wait time (0.5s → 2.0s) and limit (5 → 100)
                            #                   Added retry logic with exponential backoff
                            entry_fee = 0

                            # Wait for trade to be recorded (increased from 0.5s for reliability)
                            time.sleep(2.0)

                            # Retry logic for fee fetching
                            max_retries = 3
                            retry_delay = 1.0

                            for attempt in range(max_retries):
                                try:
                                    # Fetch recent trades with larger history window
                                    recent_fills = client.exchange.fetch_my_trades(
                                        symbol='BTC/USDT:USDT',
                                        limit=100  # Increased from 5
                                    )

                                    logger.info(f"📊 Attempt {attempt+1}/{max_retries}: Retrieved {len(recent_fills)} trades")

                                    # Find the trade matching our order ID
                                    order_id = order_result.get('id', 'N/A')
                                    for fill in recent_fills:
                                        if fill.get('order') == order_id:
                                            # Extract fee from filled trade
                                            if 'fee' in fill and isinstance(fill['fee'], dict):
                                                entry_fee = float(fill['fee'].get('cost', 0))
                                            elif 'info' in fill and 'commission' in fill['info']:
                                                entry_fee = abs(float(fill['info']['commission']))

                                            # Also update actual_entry_price from fill if available
                                            if fill.get('price'):
                                                actual_entry_price = float(fill['price'])

                                            logger.info(f"📊 Trade fill details: Price=${actual_entry_price:,.2f}, Fee=${entry_fee:.4f}")
                                            break

                                    # If fee found, exit retry loop
                                    if entry_fee > 0:
                                        logger.info(f"✅ Entry fee fetched successfully: ${entry_fee:.4f}")
                                        break
                                    else:
                                        if attempt < max_retries - 1:
                                            logger.warning(f"⚠️  Fee not found in {len(recent_fills)} trades, retrying in {retry_delay}s...")
                                            time.sleep(retry_delay)
                                            retry_delay *= 2  # Exponential backoff
                                        else:
                                            logger.warning(f"⚠️  Could not fetch fee from trade history after {max_retries} attempts")
                                            logger.warning(f"      Order ID: {order_id}")
                                            logger.warning(f"      This may indicate an API issue or the trade is not yet recorded")

                                except Exception as e:
                                    logger.warning(f"⚠️  Error fetching trade fee (attempt {attempt+1}): {e}")
                                    if attempt < max_retries - 1:
                                        time.sleep(retry_delay)
                                        retry_delay *= 2
                                    else:
                                        entry_fee = 0
                                        break

                            # Create position tracking
                            # IMPORTANT: Use actual entry time (datetime.now()), not candle timestamp!
                            actual_entry_time = datetime.now().isoformat()

                            position_data = {
                                'status': 'OPEN',  # Monitor compatibility
                                'side': side,
                                'entry_time': actual_entry_time,  # Fixed: actual entry time
                                'entry_candle_time': str(current_time),  # Candle timestamp (for reference)
                                'entry_price': actual_entry_price,  # ← API actual fill price
                                'entry_fee': float(entry_fee),  # ← API actual fee (USD)
                                'entry_candle_idx': candle_counter,
                                'entry_long_prob': long_prob,
                                'entry_short_prob': short_prob,
                                'probability': float(long_prob if side == 'LONG' else short_prob),
                                'position_size_pct': sizing_result['position_size_pct'],
                                'position_value': sizing_result['position_value'],
                                'leveraged_value': sizing_result['leveraged_value'],
                                'quantity': quantity,
                                'order_id': order_result.get('id', 'N/A'),
                                'position_id_exchange': protection_result.get('position_id'),  # CRITICAL: For reconciliation
                                # Protection order (exchange-level Stop Loss only)
                                'stop_loss_order_id': stop_loss_order.get('id', 'N/A'),
                                'stop_loss_price': protection_result['stop_loss_price']
                            }

                            state['position'] = position_data
                            # Also add to trades array for monitor (as OPEN trade)
                            state['trades'].append(position_data.copy())

                            # Log entry
                            logger.info(f"🚀 ENTER {side}: {entry_reason}")
                            logger.info(f"   Entry Price: ${actual_entry_price:,.1f}")
                            logger.info(f"   Entry Fee: ${entry_fee:.2f}")
                            logger.info(f"   Time: {current_time}")
                            logger.info(f"   Position Size: {sizing_result['position_size_pct']*100:.1f}%")
                            logger.info(f"   Position Value: ${sizing_result['position_value']:,.2f}")
                            logger.info(f"   Leveraged ({LEVERAGE}x): ${sizing_result['leveraged_value']:,.2f}")
                            logger.info(f"   Quantity: {quantity:.6f} BTC")
                            logger.info(f"   Order ID: {order_result.get('id', 'N/A')}")
                            logger.info(f"   🛡️ Protection:")
                            logger.info(f"      Stop Loss: ${protection_result['stop_loss_price']:,.2f} (-{EMERGENCY_STOP_LOSS*100:.1f}% balance) [Balance-Based]")
                            logger.info(f"      SL Price Change: {protection_result['price_sl_pct']*100:.2f}% (dynamic by position size)")
                            logger.info(f"      SL Order ID: {stop_loss_order.get('id', 'N/A')}")
                            logger.info(f"      Exit Strategy: ML Exit Model + Max Hold (8h)")

                            save_state(state)

                        except Exception as e:
                            logger.error(f"❌ Order failed: {e}")
                            logger.error(f"   Signal was valid but order execution failed")
                            logger.error(f"   Side: {side}, Quantity: {quantity:.6f}, Reason: {entry_reason}")
                    else:
                        # No entry signal, but still save state for monitor
                        # (balance, timestamp, signals were already updated above)

                        # Check if it's time for balance reconciliation
                        current_time_seconds = time.time()
                        time_since_last_check = current_time_seconds - last_balance_check

                        if time_since_last_check >= BALANCE_CHECK_INTERVAL:
                            logger.info(f"⏰ Balance reconciliation check ({time_since_last_check/3600:.1f}h since last check)")

                            # Perform balance reconciliation
                            reconciliation_performed = reconcile_balance_with_exchange(client, state)

                            if reconciliation_performed:
                                # Save state immediately after reconciliation
                                save_state(state)
                                logger.info(f"✅ State saved after balance reconciliation")

                            # Update last check time
                            last_balance_check = current_time_seconds

                        # Check if it's time for fee reconciliation
                        time_since_last_fee_check = current_time_seconds - last_fee_check

                        if FEE_RECONCILIATION_ENABLED and time_since_last_fee_check >= FEE_CHECK_INTERVAL:
                            logger.info(f"⏰ Fee reconciliation check ({time_since_last_fee_check/60:.0f}min since last check)")

                            # Perform fee reconciliation from exchange orders (ground truth)
                            reconciled_count, failed_count = reconcile_trade_fees_from_exchange(client, state)

                            if reconciled_count > 0:
                                # Save state immediately after fee reconciliation
                                save_state(state)
                                logger.info(f"✅ State saved after fee reconciliation")

                            # Update last check time
                            last_fee_check = current_time_seconds

                        save_state(state)

                # 정각 동기화: 다음 루프에서 wait_for_next_candle이 대기
                # (별도 sleep 불필요)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                logger.info("5초 대기 후 재시도...")
                time.sleep(5)  # 에러 시 짧은 대기 (정각 동기화가 나머지 처리)

    except KeyboardInterrupt:
        logger.info("\n" + "="*80)
        logger.info("BOT STOPPED BY USER")
        logger.info("="*80)

        # Final statistics
        logger.info("\nFinal Statistics:")
        logger.info(f"  Total Trades: {state['stats']['total_trades']}")
        logger.info(f"  LONG Trades: {state['stats']['long_trades']}")
        logger.info(f"  SHORT Trades: {state['stats']['short_trades']}")
        logger.info(f"  Wins: {state['stats']['wins']}")
        logger.info(f"  Losses: {state['stats']['losses']}")
        logger.info(f"  Win Rate: {state['stats']['wins']/max(state['stats']['total_trades'],1)*100:.1f}%")
        logger.info(f"  Total P&L: ${state['stats']['total_pnl_usd']:+,.2f}")
        logger.info(f"  Avg P&L per trade: {state['stats']['total_pnl_pct']/max(state['stats']['total_trades'],1)*100:+.2f}%")

        save_state(state)

    finally:
        # Remove lock file on exit (regardless of how the bot stops)
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
            logger.info("✅ Lock file removed")


if __name__ == "__main__":
    main()