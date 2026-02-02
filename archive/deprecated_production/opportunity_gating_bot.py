"""
Opportunity Gating Trading Bot - Production
===========================================

Strategy: Only enter SHORT when EV(SHORT) > EV(LONG) + gate_threshold
Validated Performance: 2.73% per window (105 days), 72% win rate

Configuration:
  LONG Threshold: 0.65
  SHORT Threshold: 0.70
  Gate Threshold: 0.001
  Max Hold: 240 candles (4 hours)
  Take Profit: 3%
  Stop Loss: -1.5%
"""

import sys
from pathlib import Path
import time
import pickle
import logging
from datetime import datetime
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.exchange.bingx_client import BingXClient
from scripts.experiments.calculate_all_features import calculate_all_features

# =============================================================================
# CONFIGURATION
# =============================================================================

# Strategy Parameters (validated in backtest)
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001  # Opportunity cost gate

# Exit Parameters
MAX_HOLD_TIME = 240  # 4 hours (candles)
TAKE_PROFIT = 0.03   # 3%
STOP_LOSS = -0.015   # -1.5%

# Expected Values (from backtest)
LONG_AVG_RETURN = 0.0041  # 0.41%
SHORT_AVG_RETURN = 0.0047  # 0.47%

# Trading Parameters
SYMBOL = "BTC-USDT"
LEVERAGE = 1
POSITION_SIZE_PCT = 0.70  # Use 70% of available balance

# Bot Parameters
CANDLE_INTERVAL = "5m"
CHECK_INTERVAL_SECONDS = 60  # Check every minute
MAX_DATA_CANDLES = 5000  # Keep last 5000 candles for feature calculation

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_state.json"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True, parents=True)
STATE_FILE.parent.mkdir(exist_ok=True, parents=True)

# =============================================================================
# LOGGING SETUP
# =============================================================================

log_file = LOGS_DIR / f"opportunity_gating_bot_{datetime.now().strftime('%Y%m%d')}.log"

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
logger.info("OPPORTUNITY GATING BOT - STARTING")
logger.info("="*80)

logger.info("Loading models...")

# LONG Model
long_model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
with open(long_scaler_path, 'rb') as f:
    long_scaler = pickle.load(f)

long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

# SHORT Model
short_model_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl"
with open(short_scaler_path, 'rb') as f:
    short_scaler = pickle.load(f)

short_features_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

logger.info(f"  ✅ Models loaded")
logger.info(f"     LONG features: {len(long_feature_columns)}")
logger.info(f"     SHORT features: {len(short_feature_columns)}")

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_state():
    """Load bot state from file"""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        'position': None,
        'trades': [],
        'stats': {
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0
        }
    }

def save_state(state):
    """Save bot state to file"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def get_signals(df):
    """
    Calculate LONG and SHORT signals

    Returns:
        (long_prob, short_prob)
    """
    try:
        # Calculate features
        df_features = calculate_all_features(df.copy())

        # Get latest candle features
        latest = df_features.iloc[-1:].copy()

        # LONG signal
        try:
            long_feat = latest[long_feature_columns].values
            long_feat_scaled = long_scaler.transform(long_feat)
            long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
        except Exception as e:
            logger.warning(f"LONG signal error: {e}")
            long_prob = 0.0

        # SHORT signal
        try:
            short_feat = latest[short_feature_columns].values
            short_feat_scaled = short_scaler.transform(short_feat)
            short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
        except Exception as e:
            logger.warning(f"SHORT signal error: {e}")
            short_prob = 0.0

        return long_prob, short_prob

    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        return 0.0, 0.0

# =============================================================================
# TRADING LOGIC
# =============================================================================

def check_entry_signal(long_prob, short_prob, position):
    """
    Check for entry signal using Opportunity Gating strategy

    Returns:
        (should_enter, side, reason)
    """
    if position is not None:
        return False, None, "Already in position"

    # LONG entry (standard)
    if long_prob >= LONG_THRESHOLD:
        return True, "LONG", f"LONG signal (prob={long_prob:.4f})"

    # SHORT entry (gated)
    if short_prob >= SHORT_THRESHOLD:
        # Calculate expected values
        long_ev = long_prob * LONG_AVG_RETURN
        short_ev = short_prob * SHORT_AVG_RETURN
        opportunity_cost = short_ev - long_ev

        # Gate check
        if opportunity_cost > GATE_THRESHOLD:
            return True, "SHORT", f"SHORT signal (prob={short_prob:.4f}, opp_cost={opportunity_cost:.6f})"
        else:
            return False, None, f"SHORT blocked by gate (opp_cost={opportunity_cost:.6f} < {GATE_THRESHOLD})"

    return False, None, "No signal"


def check_exit_signal(position, current_price, current_candle_idx):
    """
    Check for exit signal

    Returns:
        (should_exit, reason)
    """
    if position is None:
        return False, "No position"

    time_in_position = current_candle_idx - position['entry_candle_idx']

    # Calculate P&L
    if position['side'] == "LONG":
        pnl_pct = (current_price - position['entry_price']) / position['entry_price']
    else:  # SHORT
        pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

    # Exit conditions
    if time_in_position >= MAX_HOLD_TIME:
        return True, f"Max hold time ({MAX_HOLD_TIME} candles)"

    if pnl_pct >= TAKE_PROFIT:
        return True, f"Take profit ({pnl_pct*100:.2f}%)"

    if pnl_pct <= STOP_LOSS:
        return True, f"Stop loss ({pnl_pct*100:.2f}%)"

    return False, f"Holding ({pnl_pct*100:.2f}%)"


def calculate_position_size(balance):
    """Calculate position size based on available balance"""
    return balance * POSITION_SIZE_PCT


# =============================================================================
# MAIN TRADING LOOP
# =============================================================================

def main():
    """Main trading loop"""

    logger.info("\n" + "="*80)
    logger.info("BOT CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Strategy: Opportunity Gating")
    logger.info(f"LONG Threshold: {LONG_THRESHOLD}")
    logger.info(f"SHORT Threshold: {SHORT_THRESHOLD}")
    logger.info(f"Gate Threshold: {GATE_THRESHOLD}")
    logger.info(f"Max Hold: {MAX_HOLD_TIME} candles (4 hours)")
    logger.info(f"Take Profit: {TAKE_PROFIT*100}%")
    logger.info(f"Stop Loss: {STOP_LOSS*100}%")
    logger.info(f"Position Size: {POSITION_SIZE_PCT*100}% of balance")
    logger.info("="*80 + "\n")

    # Initialize
    client = BingXClient()
    state = load_state()

    candle_counter = 0

    try:
        while True:
            try:
                # Get latest data
                df = client.get_klines(SYMBOL, CANDLE_INTERVAL, limit=MAX_DATA_CANDLES)

                if df is None or len(df) == 0:
                    logger.warning("Failed to fetch data, retrying...")
                    time.sleep(CHECK_INTERVAL_SECONDS)
                    continue

                current_price = float(df.iloc[-1]['close'])
                current_time = df.iloc[-1]['timestamp']
                candle_counter = len(df)

                # Get signals
                long_prob, short_prob = get_signals(df)

                # Log current status
                logger.info(f"[{current_time}] Price: ${current_price:,.1f} | LONG: {long_prob:.4f} | SHORT: {short_prob:.4f}")

                # Check for exit first
                if state['position'] is not None:
                    should_exit, exit_reason = check_exit_signal(state['position'], current_price, candle_counter)

                    if should_exit:
                        # Calculate final P&L
                        position = state['position']
                        if position['side'] == "LONG":
                            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                        else:
                            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

                        # Record trade
                        trade = {
                            'side': position['side'],
                            'entry_time': position['entry_time'],
                            'exit_time': str(current_time),
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'pnl_pct': pnl_pct,
                            'hold_candles': candle_counter - position['entry_candle_idx'],
                            'exit_reason': exit_reason
                        }

                        # Update stats
                        state['stats']['total_trades'] += 1
                        if position['side'] == "LONG":
                            state['stats']['long_trades'] += 1
                        else:
                            state['stats']['short_trades'] += 1

                        if pnl_pct > 0:
                            state['stats']['wins'] += 1
                        else:
                            state['stats']['losses'] += 1

                        state['stats']['total_pnl'] += pnl_pct

                        # Log exit
                        logger.info(f"🚪 EXIT {position['side']}: {exit_reason}")
                        logger.info(f"   Entry: ${position['entry_price']:,.1f} @ {position['entry_time']}")
                        logger.info(f"   Exit: ${current_price:,.1f} @ {current_time}")
                        logger.info(f"   P&L: {pnl_pct*100:+.2f}%")
                        logger.info(f"   Stats: {state['stats']['wins']}/{state['stats']['total_trades']} wins ({state['stats']['wins']/max(state['stats']['total_trades'],1)*100:.1f}%)")

                        # Clear position
                        state['trades'].append(trade)
                        state['position'] = None
                        save_state(state)

                # Check for entry
                else:
                    should_enter, side, entry_reason = check_entry_signal(long_prob, short_prob, state['position'])

                    if should_enter:
                        # Create position
                        state['position'] = {
                            'side': side,
                            'entry_time': str(current_time),
                            'entry_price': current_price,
                            'entry_candle_idx': candle_counter,
                            'entry_long_prob': long_prob,
                            'entry_short_prob': short_prob
                        }

                        # Log entry
                        logger.info(f"🚀 ENTER {side}: {entry_reason}")
                        logger.info(f"   Price: ${current_price:,.1f}")
                        logger.info(f"   Time: {current_time}")
                        logger.info(f"   TP: ${current_price * (1 + (TAKE_PROFIT if side == 'LONG' else -TAKE_PROFIT)):,.1f}")
                        logger.info(f"   SL: ${current_price * (1 + (STOP_LOSS if side == 'LONG' else -STOP_LOSS)):,.1f}")

                        save_state(state)

                # Sleep until next check
                time.sleep(CHECK_INTERVAL_SECONDS)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(CHECK_INTERVAL_SECONDS)

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
        logger.info(f"  Total P&L: {state['stats']['total_pnl']*100:+.2f}%")

        save_state(state)


if __name__ == "__main__":
    main()
