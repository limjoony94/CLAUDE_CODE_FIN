#!/usr/bin/env python3
"""
LONG Entry Model Clean Validation Backtest
==========================================

Purpose: Recalculate LONG Entry model performance on CLEAN validation period only
Issue: Original backtest included 73.1% training data (data leakage)

Model: xgboost_long_entry_enhanced_20251024_012445.pkl
Training: Jul 14 - Sep 28, 2025 (76 days, 21,940 candles)
Validation: Sep 28 - Oct 26, 2025 (28 days, 8,064 candles) ← ONLY THIS PERIOD

This script validates performance on 100% out-of-sample data (NO training overlap)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Configuration (Production settings)
LONG_THRESHOLD = 0.80  # Original production threshold (before Nov 5 increase)
SHORT_THRESHOLD = 0.80
GATE_THRESHOLD = 0.001
LEVERAGE = 4
STOP_LOSS_PCT = 0.03  # -3% balance
MAX_HOLD_CANDLES = 120  # 10 hours
POSITION_SIZE_PCT = 0.95  # 95% of balance
INITIAL_BALANCE = 10000.0

# Validation period (CLEAN - no training overlap)
VALIDATION_START = "2025-09-28 00:00:00"
VALIDATION_END = "2025-10-26 23:59:59"

def load_models():
    """Load LONG Entry model (Enhanced 5-Fold CV)"""
    print("=" * 80)
    print("LOADING LONG ENTRY MODEL")
    print("=" * 80)

    # LONG Entry
    long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl")
    long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
    with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
        long_entry_features = [line.strip() for line in f.readlines()]

    print(f"✅ LONG Entry: 20251024_012445")
    print(f"   Features: {len(long_entry_features)}")

    # SHORT Entry (for comparison)
    short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl")
    short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
    with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
        short_entry_features = [line.strip() for line in f.readlines()]

    print(f"✅ SHORT Entry: 20251024_012445")
    print(f"   Features: {len(short_entry_features)}")

    return (long_entry_model, long_entry_scaler, long_entry_features,
            short_entry_model, short_entry_scaler, short_entry_features)


def load_clean_validation_data():
    """Load CLEAN validation period only (Sep 28 - Oct 26)"""
    print("\n" + "=" * 80)
    print("LOADING CLEAN VALIDATION DATA")
    print("=" * 80)

    # Load features
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Full Dataset: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total Rows: {len(df):,}")

    # Extract CLEAN validation period only
    val_start = pd.to_datetime(VALIDATION_START)
    val_end = pd.to_datetime(VALIDATION_END)

    df_val = df[(df['timestamp'] >= val_start) & (df['timestamp'] <= val_end)].copy()
    df_val = df_val.reset_index(drop=True)

    print(f"\n✅ CLEAN Validation Period (NO training overlap):")
    print(f"   Period: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
    print(f"   Rows: {len(df_val):,}")
    print(f"   Days: {(df_val['timestamp'].max() - df_val['timestamp'].min()).days}")
    print(f"   Expected: Sep 28 - Oct 26, 2025 (28 days, ~8,064 candles)")

    return df_val


def backtest_clean_validation(df, long_entry_model, long_entry_scaler, long_entry_features,
                               short_entry_model, short_entry_scaler, short_entry_features):
    """Run backtest on CLEAN validation period"""
    print("\n" + "=" * 80)
    print("BACKTESTING CLEAN VALIDATION PERIOD")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Entry Threshold: {LONG_THRESHOLD}/{SHORT_THRESHOLD} (LONG/SHORT)")
    print(f"  Gate Threshold: {GATE_THRESHOLD}")
    print(f"  Leverage: {LEVERAGE}×")
    print(f"  Stop Loss: -{STOP_LOSS_PCT * 100}% balance")
    print(f"  Max Hold: {MAX_HOLD_CANDLES} candles ({MAX_HOLD_CANDLES / 12:.1f} hours)")
    print(f"  Position Size: {POSITION_SIZE_PCT * 100}% balance")
    print(f"  Initial Balance: ${INITIAL_BALANCE:,.2f}")

    balance = INITIAL_BALANCE
    position = None
    trades = []

    # Predict probabilities (convert to numpy to avoid pandas/XGBoost issues)
    X_long = df[long_entry_features].fillna(0).replace([np.inf, -np.inf], 0).values
    X_short = df[short_entry_features].fillna(0).replace([np.inf, -np.inf], 0).values

    X_long_scaled = long_entry_scaler.transform(X_long)
    X_short_scaled = short_entry_scaler.transform(X_short)

    long_probs = long_entry_model.predict_proba(X_long_scaled)[:, 1]
    short_probs = short_entry_model.predict_proba(X_short_scaled)[:, 1]

    # Add to dataframe for analysis
    df['long_prob'] = long_probs
    df['short_prob'] = short_probs

    print(f"\nProbability Statistics:")
    print(f"  LONG >= 0.80: {(long_probs >= 0.80).sum()} candles ({(long_probs >= 0.80).mean() * 100:.2f}%)")
    print(f"  SHORT >= 0.80: {(short_probs >= 0.80).sum()} candles ({(short_probs >= 0.80).mean() * 100:.2f}%)")

    # Backtest loop
    for i in range(len(df)):
        row = df.iloc[i]
        current_price = row['close']
        current_time = row['timestamp']

        # Check existing position
        if position:
            # Calculate hold time
            hold_candles = i - position['entry_idx']

            # Check Stop Loss
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:  # SHORT
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl_pct = pnl_pct * LEVERAGE
            balance_pnl_pct = leveraged_pnl_pct * position['position_size_pct']

            # Stop Loss hit
            if balance_pnl_pct <= -STOP_LOSS_PCT:
                # Close position
                exit_price = current_price
                pnl = balance * balance_pnl_pct
                balance += pnl

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'entry_prob': position['probability'],
                    'hold_candles': hold_candles,
                    'pnl': pnl,
                    'balance': balance,
                    'exit_reason': 'Stop Loss'
                })

                position = None
                continue

            # Max Hold
            if hold_candles >= MAX_HOLD_CANDLES:
                # Close position
                exit_price = current_price
                pnl = balance * balance_pnl_pct
                balance += pnl

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'entry_prob': position['probability'],
                    'hold_candles': hold_candles,
                    'pnl': pnl,
                    'balance': balance,
                    'exit_reason': 'Max Hold'
                })

                position = None
                continue

        # No position - check entry signals
        if not position:
            long_prob = row['long_prob']
            short_prob = row['short_prob']

            # Opportunity Gating
            should_long = long_prob >= LONG_THRESHOLD
            should_short = short_prob >= SHORT_THRESHOLD and short_prob > long_prob + GATE_THRESHOLD

            if should_long or should_short:
                side = 'SHORT' if should_short else 'LONG'
                probability = short_prob if should_short else long_prob

                # Enter position
                position = {
                    'side': side,
                    'entry_time': current_time,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'probability': probability,
                    'position_size_pct': POSITION_SIZE_PCT
                }

    # Close any remaining position at end
    if position:
        row = df.iloc[-1]
        current_price = row['close']
        current_time = row['timestamp']
        hold_candles = len(df) - 1 - position['entry_idx']

        if position['side'] == 'LONG':
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

        leveraged_pnl_pct = pnl_pct * LEVERAGE
        balance_pnl_pct = leveraged_pnl_pct * position['position_size_pct']
        pnl = balance * balance_pnl_pct
        balance += pnl

        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': current_time,
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'entry_prob': position['probability'],
            'hold_candles': hold_candles,
            'pnl': pnl,
            'balance': balance,
            'exit_reason': 'Force Close'
        })

    return trades, balance


def analyze_results(trades, final_balance):
    """Analyze backtest results"""
    print("\n" + "=" * 80)
    print("CLEAN VALIDATION BACKTEST RESULTS")
    print("=" * 80)

    if len(trades) == 0:
        print("❌ No trades executed")
        return

    df_trades = pd.DataFrame(trades)

    # Overall metrics
    total_return = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    print(f"\n💰 Performance Metrics:")
    print(f"   Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"   Final Balance: ${final_balance:,.2f}")
    print(f"   Total Return: {total_return:+.2f}%")
    print(f"   Total Trades: {len(trades)}")

    # Trade breakdown
    long_trades = df_trades[df_trades['side'] == 'LONG']
    short_trades = df_trades[df_trades['side'] == 'SHORT']

    print(f"\n📊 Trade Breakdown:")
    print(f"   LONG: {len(long_trades)} ({len(long_trades) / len(trades) * 100:.1f}%)")
    print(f"   SHORT: {len(short_trades)} ({len(short_trades) / len(trades) * 100:.1f}%)")

    # Win/Loss
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[df_trades['pnl'] <= 0]
    win_rate = len(wins) / len(trades) * 100

    print(f"\n✅ Win/Loss:")
    print(f"   Wins: {len(wins)} ({win_rate:.2f}%)")
    print(f"   Losses: {len(losses)} ({100 - win_rate:.2f}%)")

    if len(wins) > 0:
        avg_win = wins['pnl'].mean()
        print(f"   Avg Win: ${avg_win:.2f}")

    if len(losses) > 0:
        avg_loss = losses['pnl'].mean()
        print(f"   Avg Loss: ${avg_loss:.2f}")

    # Profit Factor
    if len(wins) > 0 and len(losses) > 0:
        profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum())
        print(f"   Profit Factor: {profit_factor:.2f}×")

    # Exit reasons
    print(f"\n🚪 Exit Reasons:")
    for reason, count in df_trades['exit_reason'].value_counts().items():
        pct = count / len(trades) * 100
        print(f"   {reason}: {count} ({pct:.1f}%)")

    # Hold time
    avg_hold = df_trades['hold_candles'].mean()
    print(f"\n⏱️ Hold Time:")
    print(f"   Average: {avg_hold:.1f} candles ({avg_hold / 12:.2f} hours)")

    # Trading frequency
    first_trade = pd.to_datetime(df_trades['entry_time'].min())
    last_trade = pd.to_datetime(df_trades['entry_time'].max())
    trading_days = (last_trade - first_trade).days + 1
    trades_per_day = len(trades) / trading_days

    print(f"\n📈 Trading Frequency:")
    print(f"   Trades/Day: {trades_per_day:.2f}")
    print(f"   Trading Period: {trading_days} days")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"backtest_long_clean_validation_{timestamp}.csv"
    df_trades.to_csv(output_file, index=False)
    print(f"\n💾 Results saved: {output_file.name}")

    return df_trades


def compare_with_reported():
    """Compare with originally reported backtest (with data leakage)"""
    print("\n" + "=" * 80)
    print("COMPARISON: CLEAN vs REPORTED (with data leakage)")
    print("=" * 80)

    print(f"\nReported Backtest (104 days, Jul 14 - Oct 26):")
    print(f"  Period: Jul 14 - Oct 26, 2025 (104 days)")
    print(f"  Total Return: +1,209.26%")
    print(f"  Win Rate: 56.41%")
    print(f"  Total Trades: 4,135")
    print(f"  Trades/Day: ~40")
    print(f"  ⚠️ DATA LEAKAGE: 73.1% (76 days training included)")

    print(f"\nCLEAN Validation (28 days, Sep 28 - Oct 26):")
    print(f"  Period: Sep 28 - Oct 26, 2025 (28 days)")
    print(f"  ✅ 100% Out-of-Sample (NO training overlap)")
    print(f"  Results: See above")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LONG ENTRY MODEL - CLEAN VALIDATION BACKTEST")
    print("=" * 80)
    print(f"\nObjective: Calculate true out-of-sample performance")
    print(f"Model: xgboost_long_entry_enhanced_20251024_012445.pkl")
    print(f"Training: Jul 14 - Sep 28, 2025 (76 days) ← NOT USED IN BACKTEST")
    print(f"Validation: Sep 28 - Oct 26, 2025 (28 days) ← ONLY THIS PERIOD")
    print(f"\nThis corrects the 73.1% data leakage in original backtest")

    # Load models
    models = load_models()

    # Load clean validation data
    df_val = load_clean_validation_data()

    # Run backtest
    trades, final_balance = backtest_clean_validation(df_val, *models)

    # Analyze results
    df_trades = analyze_results(trades, final_balance)

    # Compare with reported
    compare_with_reported()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
