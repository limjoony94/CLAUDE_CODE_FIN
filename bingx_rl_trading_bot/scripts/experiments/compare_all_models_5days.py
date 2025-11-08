"""
Compare All Major Model Versions - Latest 5 Days
================================================

Compares performance of all major model versions on the latest 5 days:
1. Walk-Forward Decoupled (Production)
2. Reduced Features (Latest - overfitting prevention)
3. Trade-Outcome Full (Previous version)

Period: 2025-10-25 to 2025-10-30 (latest 5 days)

Created: 2025-10-30
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta

from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

# Configuration
ENTRY_THRESHOLD = 0.75
ML_EXIT_THRESHOLD = 0.75
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Model configurations to test
MODEL_CONFIGS = {
    "WalkForward_Decoupled": {
        "long_entry": "xgboost_long_entry_walkforward_decoupled_20251027_194313",
        "short_entry": "xgboost_short_entry_walkforward_decoupled_20251027_194313",
        "long_exit": "xgboost_long_exit_threshold_075_20251027_190512",
        "short_exit": "xgboost_short_exit_threshold_075_20251027_190512",
        "description": "Production - Walk-Forward Decoupled (85/79 features)"
    },
    "Reduced_Features": {
        "long_entry": "xgboost_long_entry_walkforward_reduced_20251028_230817",
        "short_entry": "xgboost_short_entry_walkforward_reduced_20251028_230817",
        "long_exit": "xgboost_long_exit_threshold_075_20251027_190512",
        "short_exit": "xgboost_short_exit_threshold_075_20251027_190512",
        "description": "Latest - Reduced Features (73/74 features)"
    },
    "TradeOutcome_Full": {
        "long_entry": "xgboost_long_trade_outcome_full_20251018_233146",
        "short_entry": "xgboost_short_trade_outcome_full_20251018_233146",
        "long_exit": "xgboost_long_exit_oppgating_improved_20251024_043527",
        "short_exit": "xgboost_short_exit_oppgating_improved_20251024_044510",
        "description": "Previous - Trade-Outcome Full Dataset"
    }
}

print("="*100)
print("MODEL COMPARISON: LATEST 5 DAYS (2025-10-25 to 2025-10-30)")
print("="*100)
print()
print(f"Configuration:")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Stop Loss: {EMERGENCY_STOP_LOSS}")
print(f"  Max Hold: {EMERGENCY_MAX_HOLD} candles")
print()

# Load Data
print("-"*100)
print("STEP 1: Loading Data (Latest 5 Days)")
print("-"*100)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter to latest 5 days
end_time = df['timestamp'].max()
start_time = end_time - timedelta(days=5)
df = df[df['timestamp'] >= start_time].copy()

print(f"✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"   Period: Latest 5 days ({(end_time - start_time).days} days)")

# Prepare Exit features
print("\nPreparing Exit features...")
df = prepare_exit_features(df)
print(f"✅ Exit features added ({len(df.columns)} total columns)")
print()


def load_model_set(config_name, config):
    """Load a complete model set (Entry + Exit for LONG/SHORT)"""
    print(f"\nLoading {config_name}...")
    print(f"  Description: {config['description']}")

    models = {}

    # LONG Entry
    with open(MODELS_DIR / f"{config['long_entry']}.pkl", 'rb') as f:
        models['long_entry_model'] = pickle.load(f)
    models['long_entry_scaler'] = joblib.load(MODELS_DIR / f"{config['long_entry']}_scaler.pkl")
    with open(MODELS_DIR / f"{config['long_entry']}_features.txt", 'r') as f:
        models['long_entry_features'] = [line.strip() for line in f]

    # SHORT Entry
    with open(MODELS_DIR / f"{config['short_entry']}.pkl", 'rb') as f:
        models['short_entry_model'] = pickle.load(f)
    models['short_entry_scaler'] = joblib.load(MODELS_DIR / f"{config['short_entry']}_scaler.pkl")
    with open(MODELS_DIR / f"{config['short_entry']}_features.txt", 'r') as f:
        models['short_entry_features'] = [line.strip() for line in f]

    # LONG Exit
    with open(MODELS_DIR / f"{config['long_exit']}.pkl", 'rb') as f:
        models['long_exit_model'] = pickle.load(f)
    models['long_exit_scaler'] = joblib.load(MODELS_DIR / f"{config['long_exit']}_scaler.pkl")
    with open(MODELS_DIR / f"{config['long_exit']}_features.txt", 'r') as f:
        models['long_exit_features'] = [line.strip() for line in f]

    # SHORT Exit
    with open(MODELS_DIR / f"{config['short_exit']}.pkl", 'rb') as f:
        models['short_exit_model'] = pickle.load(f)
    models['short_exit_scaler'] = joblib.load(MODELS_DIR / f"{config['short_exit']}_scaler.pkl")
    with open(MODELS_DIR / f"{config['short_exit']}_features.txt", 'r') as f:
        models['short_exit_features'] = [line.strip() for line in f]

    print(f"  ✅ LONG Entry: {len(models['long_entry_features'])} features")
    print(f"  ✅ SHORT Entry: {len(models['short_entry_features'])} features")
    print(f"  ✅ LONG Exit: {len(models['long_exit_features'])} features")
    print(f"  ✅ SHORT Exit: {len(models['short_exit_features'])} features")

    return models


def run_backtest(df, models, config_name):
    """Run backtest with given models"""
    print(f"\n{'='*100}")
    print(f"BACKTEST: {config_name}")
    print(f"{'='*100}")

    balance = 10000.0
    position = None
    trades = []

    for i in range(len(df)):
        candle = df.iloc[i]

        # Check exit if in position
        if position is not None:
            position['hold_time'] += 1
            current_price = candle['close']

            # Calculate P&L
            if position['side'] == 'LONG':
                pnl_pct = ((current_price - position['entry_price']) / position['entry_price']) * LEVERAGE
            else:
                pnl_pct = ((position['entry_price'] - current_price) / position['entry_price']) * LEVERAGE

            leveraged_pnl = position['size'] * pnl_pct

            # Emergency exits
            emergency_exit = False
            exit_reason = None

            # Stop Loss
            if leveraged_pnl / balance <= EMERGENCY_STOP_LOSS:
                emergency_exit = True
                exit_reason = 'Stop Loss'

            # Max Hold
            elif position['hold_time'] >= EMERGENCY_MAX_HOLD:
                emergency_exit = True
                exit_reason = 'Max Hold'

            # ML Exit
            else:
                if position['side'] == 'LONG':
                    exit_feats = candle[models['long_exit_features']].values.reshape(1, -1)
                    exit_feats_scaled = models['long_exit_scaler'].transform(exit_feats)
                    exit_prob = models['long_exit_model'].predict_proba(exit_feats_scaled)[0][1]
                else:
                    exit_feats = candle[models['short_exit_features']].values.reshape(1, -1)
                    exit_feats_scaled = models['short_exit_scaler'].transform(exit_feats)
                    exit_prob = models['short_exit_model'].predict_proba(exit_feats_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    emergency_exit = True
                    exit_reason = 'ML Exit'

            # Exit
            if emergency_exit:
                balance += leveraged_pnl
                position['exit_price'] = current_price
                position['pnl'] = leveraged_pnl
                position['pnl_pct'] = leveraged_pnl / position['size']
                position['exit_reason'] = exit_reason
                trades.append(position)
                position = None

        # Entry signals (if not in position)
        if position is None:
            # LONG signal
            long_feats = candle[models['long_entry_features']].values.reshape(1, -1)
            long_feats_scaled = models['long_entry_scaler'].transform(long_feats)
            long_prob = models['long_entry_model'].predict_proba(long_feats_scaled)[0][1]

            # SHORT signal
            short_feats = candle[models['short_entry_features']].values.reshape(1, -1)
            short_feats_scaled = models['short_entry_scaler'].transform(short_feats)
            short_prob = models['short_entry_model'].predict_proba(short_feats_scaled)[0][1]

            # Entry logic with Opportunity Gating
            if long_prob >= ENTRY_THRESHOLD:
                # Enter LONG
                position = {
                    'side': 'LONG',
                    'entry_price': candle['close'],
                    'size': balance * 0.95,  # 95% position
                    'hold_time': 0,
                    'entry_time': candle['timestamp']
                }

            elif short_prob >= ENTRY_THRESHOLD:
                # Opportunity Gating
                long_ev = long_prob * 0.0041
                short_ev = short_prob * 0.0047
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > 0.001:
                    # Enter SHORT
                    position = {
                        'side': 'SHORT',
                        'entry_price': candle['close'],
                        'size': balance * 0.95,
                        'hold_time': 0,
                        'entry_time': candle['timestamp']
                    }

    # Calculate results
    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)
    total_return = (balance - 10000) / 10000
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]

    results = {
        'config_name': config_name,
        'total_return': total_return * 100,
        'final_balance': balance,
        'total_trades': len(trades),
        'win_rate': len(winning_trades) / len(trades) * 100,
        'avg_trade': trades_df['pnl'].mean(),
        'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
        'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
        'long_trades': len(trades_df[trades_df['side'] == 'LONG']),
        'short_trades': len(trades_df[trades_df['side'] == 'SHORT']),
        'ml_exits': len(trades_df[trades_df['exit_reason'] == 'ML Exit']),
        'sl_exits': len(trades_df[trades_df['exit_reason'] == 'Stop Loss']),
        'maxhold_exits': len(trades_df[trades_df['exit_reason'] == 'Max Hold'])
    }

    return results


# Run backtests for all models
print("\n" + "="*100)
print("RUNNING BACKTESTS")
print("="*100)

all_results = []

for config_name, config in MODEL_CONFIGS.items():
    try:
        models = load_model_set(config_name, config)
        results = run_backtest(df.copy(), models, config_name)

        if results is not None:
            all_results.append(results)

            # Print results
            print(f"\nResults for {config_name}:")
            print(f"  Total Return: {results['total_return']:+.2f}%")
            print(f"  Final Balance: ${results['final_balance']:,.2f}")
            print(f"  Total Trades: {results['total_trades']}")
            print(f"  Win Rate: {results['win_rate']:.1f}%")
            print(f"  Avg Trade: ${results['avg_trade']:+.2f}")
            print(f"  LONG/SHORT: {results['long_trades']}/{results['short_trades']}")
            print(f"  Exits: ML={results['ml_exits']}, SL={results['sl_exits']}, MaxHold={results['maxhold_exits']}")
        else:
            print(f"\n⚠️ No trades for {config_name}")

    except Exception as e:
        print(f"\n❌ Error with {config_name}: {str(e)}")
        import traceback
        traceback.print_exc()

# Summary comparison
if len(all_results) > 0:
    print("\n" + "="*100)
    print("SUMMARY COMPARISON")
    print("="*100)
    print()

    # Create comparison table
    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df.sort_values('total_return', ascending=False)

    print(comparison_df[['config_name', 'total_return', 'win_rate', 'total_trades',
                         'long_trades', 'short_trades', 'ml_exits']].to_string(index=False))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"model_comparison_5days_{timestamp}.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")

    # Winner
    winner = comparison_df.iloc[0]
    print(f"\n🏆 WINNER: {winner['config_name']}")
    print(f"   Return: {winner['total_return']:+.2f}%")
    print(f"   Win Rate: {winner['win_rate']:.1f}%")
    print(f"   Trades: {winner['total_trades']}")

print("\n" + "="*100)
print("BACKTEST COMPLETE")
print("="*100)
