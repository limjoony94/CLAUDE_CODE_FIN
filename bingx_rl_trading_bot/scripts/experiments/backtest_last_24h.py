#!/usr/bin/env python3
"""
Backtest Last 24 Hours - Production Settings
=============================================
Backtests the last 24 hours using current production models and settings.
Shows detailed signal information and trade execution.

Production Settings:
  - LONG Threshold: 0.65
  - SHORT Threshold: 0.70
  - Gate Threshold: 0.001
  - ML Exit Threshold: 0.75 (both LONG/SHORT)
  - Emergency Stop Loss: -3% of balance
  - Emergency Max Hold: 120 candles (10 hours)
  - Leverage: 4x

Usage:
    python scripts/experiments/backtest_last_24h.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from api.bingx_client import BingXClient

# ============================================================================
# Configuration (PRODUCTION SETTINGS)
# ============================================================================

LOOKBACK_HOURS = 24
CANDLE_INTERVAL = '5m'
INITIAL_BALANCE = 10000  # Standard backtest starting balance

# Opportunity Gating Parameters (PRODUCTION)
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Exit Parameters (FULLY OPTIMIZED 2025-10-22)
ML_EXIT_THRESHOLD_LONG = 0.75
ML_EXIT_THRESHOLD_SHORT = 0.75
EMERGENCY_STOP_LOSS = 0.03  # -3% of balance
EMERGENCY_MAX_HOLD_CANDLES = 120  # 10 hours

LEVERAGE = 4

# Model paths (PRODUCTION MODELS)
MODELS_DIR = PROJECT_ROOT / 'models'
LONG_ENTRY_MODEL = MODELS_DIR / 'xgboost_long_trade_outcome_full_20251018_233146.pkl'
SHORT_ENTRY_MODEL = MODELS_DIR / 'xgboost_short_trade_outcome_full_20251018_233146.pkl'
LONG_EXIT_MODEL = MODELS_DIR / 'xgboost_long_exit_oppgating_improved_20251017_151624.pkl'
SHORT_EXIT_MODEL = MODELS_DIR / 'xgboost_short_exit_oppgating_improved_20251017_152440.pkl'

LONG_SCALER = MODELS_DIR / 'xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl'
SHORT_SCALER = MODELS_DIR / 'xgboost_short_trade_outcome_full_20251018_233146_scaler.pkl'
LONG_EXIT_SCALER = MODELS_DIR / 'xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl'
SHORT_EXIT_SCALER = MODELS_DIR / 'xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl'


# ============================================================================
# Feature Calculation
# ============================================================================

def calculate_all_features(df):
    """Calculate all features needed for models"""
    # Import calculate_all_features function
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'experiments'))
    from calculate_all_features import calculate_all_features as calc_features
    return calc_features(df)


def load_models():
    """Load all ML models and scalers"""
    # Load models and scalers
    models = {
        'long_entry': joblib.load(LONG_ENTRY_MODEL),
        'short_entry': joblib.load(SHORT_ENTRY_MODEL),
        'long_exit': joblib.load(LONG_EXIT_MODEL),
        'short_exit': joblib.load(SHORT_EXIT_MODEL),
        'long_scaler': joblib.load(LONG_SCALER),
        'short_scaler': joblib.load(SHORT_SCALER),
        'long_exit_scaler': joblib.load(LONG_EXIT_SCALER),
        'short_exit_scaler': joblib.load(SHORT_EXIT_SCALER),
    }

    # Load feature lists
    long_features_path = MODELS_DIR / 'xgboost_long_trade_outcome_full_20251018_233146_features.txt'
    with open(long_features_path, 'r') as f:
        models['long_features'] = [line.strip() for line in f.readlines() if line.strip()]

    short_features_path = MODELS_DIR / 'xgboost_short_trade_outcome_full_20251018_233146_features.txt'
    with open(short_features_path, 'r') as f:
        models['short_features'] = [line.strip() for line in f.readlines() if line.strip()]

    long_exit_features_path = MODELS_DIR / 'xgboost_long_exit_oppgating_improved_20251017_151624_features.txt'
    with open(long_exit_features_path, 'r') as f:
        models['long_exit_features'] = [line.strip() for line in f.readlines() if line.strip()]

    short_exit_features_path = MODELS_DIR / 'xgboost_short_exit_oppgating_improved_20251017_152440_features.txt'
    with open(short_exit_features_path, 'r') as f:
        models['short_exit_features'] = [line.strip() for line in f.readlines() if line.strip()]

    return models


def get_entry_signals(features, models, idx):
    """Get LONG and SHORT entry signals for a given candle"""
    # LONG signal
    long_features = features[models['long_features']].iloc[idx:idx+1].values
    long_features_scaled = models['long_scaler'].transform(long_features)
    long_prob = models['long_entry'].predict_proba(long_features_scaled)[0][1]

    # SHORT signal
    short_features = features[models['short_features']].iloc[idx:idx+1].values
    short_features_scaled = models['short_scaler'].transform(short_features)
    short_prob = models['short_entry'].predict_proba(short_features_scaled)[0][1]

    return long_prob, short_prob


def get_exit_signal(features, models, idx, direction):
    """Get exit signal for current position"""
    # Try to get exit signal, fall back to 0.0 if features are missing
    try:
        if direction == 'LONG':
            # Check if all required features exist
            missing = [f for f in models['long_exit_features'] if f not in features.columns]
            if missing:
                return 0.0  # Features not available, cannot calculate exit signal

            exit_features = features[models['long_exit_features']].iloc[idx:idx+1].values
            exit_features_scaled = models['long_exit_scaler'].transform(exit_features)
            exit_prob = models['long_exit'].predict_proba(exit_features_scaled)[0][1]
        else:  # SHORT
            missing = [f for f in models['short_exit_features'] if f not in features.columns]
            if missing:
                return 0.0

            exit_features = features[models['short_exit_features']].iloc[idx:idx+1].values
            exit_features_scaled = models['short_exit_scaler'].transform(exit_features)
            exit_prob = models['short_exit'].predict_proba(exit_features_scaled)[0][1]

        return exit_prob
    except Exception as e:
        print(f"    Warning: Could not calculate exit signal - {e}")
        return 0.0


def check_opportunity_gate(long_prob, short_prob):
    """Check if SHORT should be allowed through opportunity gate"""
    if short_prob < SHORT_THRESHOLD:
        return False, 0.0

    long_ev = long_prob * LONG_AVG_RETURN
    short_ev = short_prob * SHORT_AVG_RETURN
    opportunity_cost = short_ev - long_ev

    return opportunity_cost > GATE_THRESHOLD, opportunity_cost


# ============================================================================
# Backtest Engine
# ============================================================================

def run_backtest(df, features, models):
    """Run backtest on 24-hour data"""
    balance = INITIAL_BALANCE
    position = None
    trades = []
    signals_log = []

    print("\n" + "="*100)
    print("📊 BACKTEST: LAST 24 HOURS - PRODUCTION SETTINGS")
    print("="*100)
    print(f"Period: {df.iloc[0]['timestamp']} → {df.iloc[-1]['timestamp']}")
    print(f"Candles: {len(df)}")
    print(f"Initial Balance: ${balance:.2f}")
    print("="*100 + "\n")

    for idx in range(len(df)):
        candle = df.iloc[idx]
        timestamp = candle['timestamp']
        price = candle['close']

        # Get entry signals
        long_prob, short_prob = get_entry_signals(features, models, idx)

        # Check if position should be exited
        if position:
            hold_time = idx - position['entry_idx']
            current_pnl_pct = (price - position['entry_price']) / position['entry_price'] * LEVERAGE
            if position['direction'] == 'SHORT':
                current_pnl_pct = -current_pnl_pct

            current_pnl_usd = current_pnl_pct * position['size_usd']

            # Get exit signal
            exit_prob = get_exit_signal(features, models, idx, position['direction'])

            # Check exit conditions
            exit_reason = None

            # 1. ML Exit
            if exit_prob >= (ML_EXIT_THRESHOLD_LONG if position['direction'] == 'LONG' else ML_EXIT_THRESHOLD_SHORT):
                exit_reason = f'ML_EXIT (prob={exit_prob:.3f})'

            # 2. Stop Loss (balance-based)
            elif current_pnl_usd / balance <= -EMERGENCY_STOP_LOSS:
                exit_reason = f'STOP_LOSS (loss={current_pnl_usd/balance:.2%})'

            # 3. Max Hold Time
            elif hold_time >= EMERGENCY_MAX_HOLD_CANDLES:
                exit_reason = f'MAX_HOLD ({hold_time} candles)'

            if exit_reason:
                # Close position
                position['exit_price'] = price
                position['exit_time'] = timestamp
                position['exit_idx'] = idx
                position['hold_candles'] = hold_time
                position['pnl_pct'] = current_pnl_pct
                position['pnl_usd'] = current_pnl_usd
                position['exit_reason'] = exit_reason
                position['exit_prob'] = exit_prob

                balance += current_pnl_usd
                trades.append(position)

                print(f"🔴 EXIT  [{timestamp}] {position['direction']:5s} @ ${price:,.1f}")
                print(f"    Reason: {exit_reason}")
                print(f"    P&L: ${current_pnl_usd:+.2f} ({current_pnl_pct:+.2%}) | Hold: {hold_time} candles")
                print(f"    Balance: ${balance:.2f}")
                print()

                position = None

        # Check entry signals (only if no position)
        if not position:
            gate_passed = False
            opportunity_cost = 0.0

            # Check LONG entry
            if long_prob >= LONG_THRESHOLD:
                direction = 'LONG'
                entry_signal = True

            # Check SHORT entry with gate
            elif short_prob >= SHORT_THRESHOLD:
                gate_passed, opportunity_cost = check_opportunity_gate(long_prob, short_prob)
                if gate_passed:
                    direction = 'SHORT'
                    entry_signal = True
                else:
                    entry_signal = False
                    direction = None
            else:
                entry_signal = False
                direction = None

            # Enter position
            if entry_signal:
                # Dynamic position sizing (simplified - use 50% for demo)
                position_pct = 0.50
                size_usd = balance * position_pct

                position = {
                    'direction': direction,
                    'entry_price': price,
                    'entry_time': timestamp,
                    'entry_idx': idx,
                    'size_usd': size_usd,
                    'size_pct': position_pct,
                    'entry_long_prob': long_prob,
                    'entry_short_prob': short_prob,
                    'opportunity_cost': opportunity_cost if direction == 'SHORT' else None,
                }

                print(f"🟢 ENTRY [{timestamp}] {direction:5s} @ ${price:,.1f}")
                print(f"    Size: ${size_usd:.2f} ({position_pct:.1%}) | Leverage: {LEVERAGE}x")
                print(f"    Signals - LONG: {long_prob:.3f} | SHORT: {short_prob:.3f}")
                if direction == 'SHORT':
                    print(f"    Gate: PASSED (opp_cost={opportunity_cost:.4f})")
                print()

        # Log signals
        signals_log.append({
            'timestamp': timestamp,
            'price': price,
            'long_prob': long_prob,
            'short_prob': short_prob,
            'position': position['direction'] if position else None,
            'exit_prob': get_exit_signal(features, models, idx, position['direction']) if position else None,
        })

    # Close any open position at end
    if position:
        price = df.iloc[-1]['close']
        timestamp = df.iloc[-1]['timestamp']
        hold_time = len(df) - 1 - position['entry_idx']
        current_pnl_pct = (price - position['entry_price']) / position['entry_price'] * LEVERAGE
        if position['direction'] == 'SHORT':
            current_pnl_pct = -current_pnl_pct
        current_pnl_usd = current_pnl_pct * position['size_usd']

        position['exit_price'] = price
        position['exit_time'] = timestamp
        position['exit_idx'] = len(df) - 1
        position['hold_candles'] = hold_time
        position['pnl_pct'] = current_pnl_pct
        position['pnl_usd'] = current_pnl_usd
        position['exit_reason'] = 'END_OF_PERIOD'
        position['exit_prob'] = get_exit_signal(features, models, len(df)-1, position['direction'])

        balance += current_pnl_usd
        trades.append(position)

        print(f"🔴 EXIT  [{timestamp}] {position['direction']:5s} @ ${price:,.1f}")
        print(f"    Reason: END_OF_PERIOD")
        print(f"    P&L: ${current_pnl_usd:+.2f} ({current_pnl_pct:+.2%}) | Hold: {hold_time} candles")
        print(f"    Balance: ${balance:.2f}")
        print()

    return trades, signals_log, balance


def print_results(trades, initial_balance, final_balance):
    """Print backtest results"""
    print("\n" + "="*100)
    print("📈 BACKTEST RESULTS - 24 HOURS")
    print("="*100)

    if not trades:
        print("No trades executed in this period.")
        return

    # Calculate metrics
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]

    total_pnl = sum(t['pnl_usd'] for t in trades)
    total_return = (final_balance - initial_balance) / initial_balance

    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean([t['pnl_usd'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_usd'] for t in losses]) if losses else 0

    long_trades = [t for t in trades if t['direction'] == 'LONG']
    short_trades = [t for t in trades if t['direction'] == 'SHORT']

    print(f"Total Trades: {len(trades)}")
    print(f"  LONG: {len(long_trades)} ({len(long_trades)/len(trades)*100:.1f}%)")
    print(f"  SHORT: {len(short_trades)} ({len(short_trades)/len(trades)*100:.1f}%)")
    print()

    print(f"Performance:")
    print(f"  Total P&L: ${total_pnl:+.2f}")
    print(f"  Total Return: {total_return:+.2%}")
    print(f"  Initial Balance: ${initial_balance:.2f}")
    print(f"  Final Balance: ${final_balance:.2f}")
    print()

    print(f"Win Rate: {win_rate:.1%} ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg Win: ${avg_win:.2f}")
    print(f"  Avg Loss: ${avg_loss:.2f}")
    if avg_loss != 0:
        print(f"  Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}")
    print()

    # Exit reasons
    print("Exit Reasons:")
    exit_reasons = {}
    for t in trades:
        reason = t['exit_reason'].split('(')[0].strip()
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count/len(trades)*100:.1f}%)")

    # Trade details
    print("\n" + "-"*100)
    print("Trade Details:")
    print("-"*100)
    for i, t in enumerate(trades, 1):
        print(f"{i}. {t['direction']:5s} | Entry: {t['entry_time']} @ ${t['entry_price']:,.1f}")
        print(f"   Exit: {t['exit_time']} @ ${t['exit_price']:,.1f} | {t['exit_reason']}")
        print(f"   P&L: ${t['pnl_usd']:+.2f} ({t['pnl_pct']:+.2%}) | Hold: {t['hold_candles']} candles")
        print()

    print("="*100)


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n🚀 BACKTEST LAST 24 HOURS - PRODUCTION SETTINGS")
    print("="*100)
    print("Loading configuration...")

    # Load API keys
    config_path = PROJECT_ROOT / 'config' / 'api_keys.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize client
    api_key = config['bingx']['mainnet']['api_key']
    secret_key = config['bingx']['mainnet']['secret_key']
    client = BingXClient(api_key, secret_key, testnet=False)

    # Load models
    print("🧠 Loading ML models...")
    models = load_models()

    # Fetch data (last 24 hours + buffer for features)
    print(f"📥 Fetching last 24 hours of data...")
    limit = 24 * 60 // 5 + 200  # 24 hours + buffer for features (288 + 200 = 488)
    ohlcv = client.get_klines('BTC-USDT', CANDLE_INTERVAL, limit=limit)

    # Convert to DataFrame (get_klines returns list of dicts)
    df = pd.DataFrame(ohlcv)
    df = df.rename(columns={'time': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Convert string columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    print(f"  Total candles fetched: {len(df)}")
    print(f"  Period: {df.iloc[0]['timestamp']} → {df.iloc[-1]['timestamp']}")

    # Calculate features
    print("🔧 Calculating features...")
    features = calculate_all_features(df)

    # Filter to last 24 hours for backtest
    backtest_start_idx = len(df) - (24 * 60 // 5)
    df_backtest = df.iloc[backtest_start_idx:].reset_index(drop=True)
    features_backtest = features.iloc[backtest_start_idx:].reset_index(drop=True)

    print(f"  Backtest candles: {len(df_backtest)}")
    print(f"  Backtest period: {df_backtest.iloc[0]['timestamp']} → {df_backtest.iloc[-1]['timestamp']}")

    # Run backtest
    trades, signals_log, final_balance = run_backtest(df_backtest, features_backtest, models)

    # Print results
    print_results(trades, INITIAL_BALANCE, final_balance)

    # Save signals log
    signals_df = pd.DataFrame(signals_log)
    output_path = PROJECT_ROOT / 'results' / 'backtest_last_24h_signals.csv'
    signals_df.to_csv(output_path, index=False)
    print(f"\n💾 Signals saved to: {output_path}")

    # Save trade log
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_output_path = PROJECT_ROOT / 'results' / 'backtest_last_24h_trades.csv'
        trades_df.to_csv(trades_output_path, index=False)
        print(f"💾 Trades saved to: {trades_output_path}")


if __name__ == '__main__':
    main()
