#!/usr/bin/env python3
"""
Compare Stop Loss Strategies: Fixed Price SL vs Balance-Based SL

Tests different SL approaches to find optimal configuration:
1. Fixed Price SL (current): leveraged_pnl_pct = price_pct × leverage (고정 4)
2. Balance-Based SL: price_sl_pct = balance_sl_pct / (position_size_pct × leverage)

Compares across multiple balance SL thresholds: 2%, 3%, 4%, 5%, 6%
"""

print("🚀 Starting script...", flush=True)

import sys
print("✅ sys imported", flush=True)
import pandas as pd
print("✅ pandas imported", flush=True)
import numpy as np
print("✅ numpy imported", flush=True)
import joblib
print("✅ joblib imported", flush=True)
from pathlib import Path
print("✅ Path imported", flush=True)
from datetime import datetime
print("✅ datetime imported", flush=True)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
print(f"✅ PROJECT_ROOT: {PROJECT_ROOT}", flush=True)

print("📦 Importing custom modules...", flush=True)
from scripts.experiments.calculate_all_features import calculate_all_features
print("✅ calculate_all_features imported", flush=True)
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
print("✅ prepare_exit_features imported", flush=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data
DATA_DIR = PROJECT_ROOT / "data" / "historical"

# Models (Trade-Outcome Full Dataset - Oct 19)
MODEL_DIR = PROJECT_ROOT / "models"
LONG_ENTRY_MODEL = MODEL_DIR / "xgboost_long_trade_outcome_full_20251019_101219.pkl"
SHORT_ENTRY_MODEL = MODEL_DIR / "xgboost_short_trade_outcome_full_20251019_101219.pkl"
LONG_EXIT_MODEL = MODEL_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl"
SHORT_EXIT_MODEL = MODEL_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"

# Scalers
LONG_ENTRY_SCALER = MODEL_DIR / "xgboost_long_trade_outcome_full_20251019_101219_scaler.pkl"
SHORT_ENTRY_SCALER = MODEL_DIR / "xgboost_short_trade_outcome_full_20251019_101219_scaler.pkl"
LONG_EXIT_SCALER = MODEL_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl"
SHORT_EXIT_SCALER = MODEL_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl"

# Features
LONG_ENTRY_FEATURES = MODEL_DIR / "xgboost_long_trade_outcome_full_20251019_101219_features.txt"
SHORT_ENTRY_FEATURES = MODEL_DIR / "xgboost_short_trade_outcome_full_20251019_101219_features.txt"
LONG_EXIT_FEATURES = MODEL_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt"
SHORT_EXIT_FEATURES = MODEL_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt"

# Backtest Parameters
WINDOW_SIZE = 1440  # 5 days
STEP_SIZE = 72  # 6 hours
INITIAL_BALANCE = 10000.0
LEVERAGE = 4
FEE_RATE = 0.0005  # 0.05%

# Entry Thresholds
LONG_ENTRY_THRESHOLD = 0.65
SHORT_ENTRY_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001

# Exit Parameters
ML_EXIT_THRESHOLD_LONG = 0.70
ML_EXIT_THRESHOLD_SHORT = 0.72
EMERGENCY_MAX_HOLD = 96  # 8 hours

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# SL Strategies to Test
SL_STRATEGIES = {
    'fixed_price_4pct': {
        'type': 'fixed_price',
        'leveraged_sl_pct': 0.04,
        'description': 'Fixed Price SL (Current): -4% leveraged P&L = -1% price'
    },
    'balance_2pct': {
        'type': 'balance_based',
        'balance_sl_pct': 0.02,
        'description': 'Balance-Based SL: -2% total balance'
    },
    'balance_3pct': {
        'type': 'balance_based',
        'balance_sl_pct': 0.03,
        'description': 'Balance-Based SL: -3% total balance'
    },
    'balance_4pct': {
        'type': 'balance_based',
        'balance_sl_pct': 0.04,
        'description': 'Balance-Based SL: -4% total balance'
    },
    'balance_5pct': {
        'type': 'balance_based',
        'balance_sl_pct': 0.05,
        'description': 'Balance-Based SL: -5% total balance'
    },
    'balance_6pct': {
        'type': 'balance_based',
        'balance_sl_pct': 0.06,
        'description': 'Balance-Based SL: -6% total balance'
    }
}


def calculate_stop_loss_price(strategy_config, entry_price, side, position_size_pct):
    """
    Calculate stop loss price based on strategy

    Args:
        strategy_config: SL strategy configuration
        entry_price: Position entry price
        side: 'LONG' or 'SHORT'
        position_size_pct: Position size as % of balance

    Returns:
        stop_loss_price: Price at which SL triggers
    """
    if strategy_config['type'] == 'fixed_price':
        # Fixed price SL: leveraged_pnl_pct = price_pct × leverage
        # Example: -4% / 4 = -1% price
        leveraged_sl_pct = strategy_config['leveraged_sl_pct']
        price_sl_pct = leveraged_sl_pct / LEVERAGE

    elif strategy_config['type'] == 'balance_based':
        # Balance-based SL: price_sl_pct = balance_sl_pct / (position_size_pct × leverage)
        # Example: -4% / (0.5 × 4) = -2% price
        balance_sl_pct = strategy_config['balance_sl_pct']
        price_sl_pct = balance_sl_pct / (position_size_pct * LEVERAGE)

    # Calculate stop loss price
    if side == 'LONG':
        stop_loss_price = entry_price * (1 - price_sl_pct)
    else:  # SHORT
        stop_loss_price = entry_price * (1 + price_sl_pct)

    return stop_loss_price


def check_stop_loss_triggered(current_price, stop_loss_price, side):
    """Check if stop loss is triggered"""
    if side == 'LONG':
        return current_price <= stop_loss_price
    else:  # SHORT
        return current_price >= stop_loss_price


def run_backtest_with_strategy(df_features, strategy_name, strategy_config, models, scalers, features_dict):
    """
    Run backtest with specific SL strategy

    Returns:
        dict: Backtest results
    """
    print(f"\n{'='*80}")
    print(f"Strategy: {strategy_name}")
    print(f"Description: {strategy_config['description']}")
    print(f"{'='*80}")

    # Sliding window backtest
    num_windows = (len(df_features) - WINDOW_SIZE) // STEP_SIZE + 1
    print(f"📊 Processing {num_windows} windows...", flush=True)

    window_results = []

    for window_idx in range(num_windows):
        start_idx = window_idx * STEP_SIZE
        end_idx = start_idx + WINDOW_SIZE

        if end_idx > len(df_features):
            break

        window_df = df_features.iloc[start_idx:end_idx].copy()
        window_df.reset_index(drop=True, inplace=True)

        # Progress update every 20 windows
        if (window_idx + 1) % 20 == 0 or window_idx == 0:
            print(f"  Processed {window_idx + 1}/{num_windows} windows...", flush=True)

        # Initialize window state
        balance = INITIAL_BALANCE
        position = None
        trades = []

        # Simulate trading
        for i in range(len(window_df)):
            current_price = window_df['close'].iloc[i]

            # Exit check
            if position is not None:
                hold_time = i - position['entry_idx']

                # Calculate P&L
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

                leveraged_pnl_pct = pnl_pct * LEVERAGE

                exit_triggered = False
                exit_reason = None

                # 1. ML Exit
                try:
                    if position['side'] == 'LONG':
                        exit_feat = window_df[features_dict['long_exit']].iloc[i:i+1].values
                        exit_scaled = scalers['long_exit'].transform(exit_feat)
                        exit_prob = models['long_exit'].predict_proba(exit_scaled)[0][1]
                        threshold = ML_EXIT_THRESHOLD_LONG
                    else:
                        exit_feat = window_df[features_dict['short_exit']].iloc[i:i+1].values
                        exit_scaled = scalers['short_exit'].transform(exit_feat)
                        exit_prob = models['short_exit'].predict_proba(exit_scaled)[0][1]
                        threshold = ML_EXIT_THRESHOLD_SHORT

                    if exit_prob >= threshold:
                        exit_triggered = True
                        exit_reason = 'ml_exit'
                except:
                    pass

                # 2. Stop Loss (Strategy-Specific)
                if check_stop_loss_triggered(current_price, position['stop_loss_price'], position['side']):
                    exit_triggered = True
                    exit_reason = 'stop_loss'

                # 3. Emergency Max Hold
                if hold_time >= EMERGENCY_MAX_HOLD:
                    exit_triggered = True
                    exit_reason = 'emergency_max_hold'

                if exit_triggered:
                    # Close position
                    pnl_usd = balance * position['position_size_pct'] * leveraged_pnl_pct

                    # Calculate fees
                    position_value = balance * position['position_size_pct']
                    entry_fee = position_value * FEE_RATE
                    exit_fee = position_value * FEE_RATE
                    total_fees = entry_fee + exit_fee

                    # Subtract fees
                    pnl_usd_after_fees = pnl_usd - total_fees
                    balance += pnl_usd_after_fees

                    trades.append({
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'leveraged_pnl_pct': leveraged_pnl_pct,
                        'pnl_usd': pnl_usd_after_fees,
                        'position_size_pct': position['position_size_pct'],
                        'exit_reason': exit_reason,
                        'hold_time': hold_time
                    })

                    position = None

            # Entry check
            if position is None:
                try:
                    # LONG signal
                    long_feat = window_df[features_dict['long_entry']].iloc[i:i+1].values
                    long_scaled = scalers['long_entry'].transform(long_feat)
                    long_prob = models['long_entry'].predict_proba(long_scaled)[0][1]

                    # SHORT signal
                    short_feat = window_df[features_dict['short_entry']].iloc[i:i+1].values
                    short_scaled = scalers['short_entry'].transform(short_feat)
                    short_prob = models['short_entry'].predict_proba(short_scaled)[0][1]

                    # Entry decision
                    should_enter = False
                    side = None

                    if long_prob >= LONG_ENTRY_THRESHOLD:
                        should_enter = True
                        side = 'LONG'

                    if short_prob >= SHORT_ENTRY_THRESHOLD:
                        long_ev = long_prob * LONG_AVG_RETURN
                        short_ev = short_prob * SHORT_AVG_RETURN
                        opportunity_cost = short_ev - long_ev

                        if opportunity_cost > GATE_THRESHOLD:
                            should_enter = True
                            side = 'SHORT'

                    if should_enter:
                        # Dynamic position sizing (20-95%)
                        prob = long_prob if side == 'LONG' else short_prob
                        position_size_pct = 0.20 + (prob - 0.65) * (0.95 - 0.20) / (1.0 - 0.65)
                        position_size_pct = max(0.20, min(0.95, position_size_pct))

                        # Calculate stop loss price
                        stop_loss_price = calculate_stop_loss_price(
                            strategy_config,
                            current_price,
                            side,
                            position_size_pct
                        )

                        position = {
                            'side': side,
                            'entry_price': current_price,
                            'entry_idx': i,
                            'position_size_pct': position_size_pct,
                            'stop_loss_price': stop_loss_price
                        }
                except:
                    pass

        # Window summary
        if len(trades) > 0:
            total_pnl = sum(t['pnl_usd'] for t in trades)
            returns_pct = (total_pnl / INITIAL_BALANCE) * 100
            wins = sum(1 for t in trades if t['pnl_usd'] > 0)
            win_rate = wins / len(trades) * 100

            long_trades = [t for t in trades if t['side'] == 'LONG']
            short_trades = [t for t in trades if t['side'] == 'SHORT']

            sl_trades = [t for t in trades if t['exit_reason'] == 'stop_loss']

            window_results.append({
                'returns_pct': returns_pct,
                'total_trades': len(trades),
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'win_rate': win_rate,
                'sl_triggers': len(sl_trades),
                'sl_rate': len(sl_trades) / len(trades) * 100 if len(trades) > 0 else 0
            })

    # Aggregate results
    if len(window_results) > 0:
        avg_return = np.mean([w['returns_pct'] for w in window_results])
        avg_win_rate = np.mean([w['win_rate'] for w in window_results])
        avg_trades = np.mean([w['total_trades'] for w in window_results])
        avg_long_pct = np.mean([w['long_trades'] / w['total_trades'] * 100 for w in window_results])
        avg_sl_rate = np.mean([w['sl_rate'] for w in window_results])

        return {
            'strategy': strategy_name,
            'description': strategy_config['description'],
            'avg_return': avg_return,
            'avg_win_rate': avg_win_rate,
            'avg_trades': avg_trades,
            'avg_long_pct': avg_long_pct,
            'avg_sl_rate': avg_sl_rate,
            'num_windows': len(window_results)
        }
    else:
        return None


def main():
    import sys
    sys.stdout.flush()
    print("="*80, flush=True)
    print("STOP LOSS STRATEGY COMPARISON", flush=True)
    print("="*80, flush=True)
    sys.stdout.flush()

    # Load data
    print("\nStep 1: Loading historical data...")
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} candles")
    print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Calculate features
    print("\nStep 2: Calculating features...")
    df = calculate_all_features(df)
    df_features = prepare_exit_features(df)
    print(f"✅ All features calculated ({len(df_features.columns)} columns)")

    # Load models
    print("\nStep 3: Loading models...")
    models = {
        'long_entry': joblib.load(LONG_ENTRY_MODEL),
        'short_entry': joblib.load(SHORT_ENTRY_MODEL),
        'long_exit': joblib.load(LONG_EXIT_MODEL),
        'short_exit': joblib.load(SHORT_EXIT_MODEL)
    }

    scalers = {
        'long_entry': joblib.load(LONG_ENTRY_SCALER),
        'short_entry': joblib.load(SHORT_ENTRY_SCALER),
        'long_exit': joblib.load(LONG_EXIT_SCALER),
        'short_exit': joblib.load(SHORT_EXIT_SCALER)
    }

    # Load feature columns from files
    with open(LONG_ENTRY_FEATURES, 'r') as f:
        long_entry_features = [line.strip() for line in f.readlines() if line.strip()]
    with open(SHORT_ENTRY_FEATURES, 'r') as f:
        short_entry_features = [line.strip() for line in f.readlines() if line.strip()]
    with open(LONG_EXIT_FEATURES, 'r') as f:
        long_exit_features = [line.strip() for line in f.readlines() if line.strip()]
    with open(SHORT_EXIT_FEATURES, 'r') as f:
        short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

    features_dict = {
        'long_entry': long_entry_features,
        'short_entry': short_entry_features,
        'long_exit': long_exit_features,
        'short_exit': short_exit_features
    }

    print(f"  ✅ LONG Entry: {len(features_dict['long_entry'])} features")
    print(f"  ✅ SHORT Entry: {len(features_dict['short_entry'])} features")
    print(f"  ✅ LONG Exit: {len(features_dict['long_exit'])} features")
    print(f"  ✅ SHORT Exit: {len(features_dict['short_exit'])} features")

    # Run backtests
    print("\nStep 4: Running backtests...")
    results = []

    for strategy_name, strategy_config in SL_STRATEGIES.items():
        result = run_backtest_with_strategy(
            df_features,
            strategy_name,
            strategy_config,
            models,
            scalers,
            features_dict
        )

        if result:
            results.append(result)
            print(f"\n✅ {strategy_name} complete:")
            print(f"   Return: {result['avg_return']:.2f}%")
            print(f"   Win Rate: {result['avg_win_rate']:.1f}%")
            print(f"   Trades: {result['avg_trades']:.1f}")
            print(f"   SL Rate: {result['avg_sl_rate']:.1f}%")

    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('avg_return', ascending=False)

    print("\n" + results_df.to_string(index=False))

    # Best strategy
    best = results_df.iloc[0]
    print("\n" + "="*80)
    print("🏆 BEST STRATEGY")
    print("="*80)
    print(f"Strategy: {best['strategy']}")
    print(f"Description: {best['description']}")
    print(f"Average Return: {best['avg_return']:.2f}%")
    print(f"Win Rate: {best['avg_win_rate']:.1f}%")
    print(f"Average Trades: {best['avg_trades']:.1f}")
    print(f"SL Trigger Rate: {best['avg_sl_rate']:.1f}%")

    # Save results
    output_file = PROJECT_ROOT / "results" / f"sl_strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved: {output_file}")


if __name__ == "__main__":
    main()
