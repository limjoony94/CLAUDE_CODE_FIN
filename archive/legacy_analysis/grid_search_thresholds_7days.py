"""
Grid Search for Entry and Exit Thresholds - 7 Day Period
=========================================================

Tests all combinations of Entry and Exit thresholds (0.1 step):
- Entry Thresholds: [0.60, 0.65, 0.70, 0.75, 0.80]
- Exit Thresholds: [0.60, 0.65, 0.70, 0.75, 0.80]
- Total Combinations: 25 (5×5)

Period: Last 7 days
Configuration: Current production models (2025-10-24)
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
import time
from datetime import datetime
from itertools import product

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("GRID SEARCH: ENTRY & EXIT THRESHOLDS - 7 DAY TEST")
print("="*80)
print(f"\nSearching for optimal threshold combination\n")

# =============================================================================
# FIXED CONFIGURATION
# =============================================================================

LEVERAGE = 4
GATE_THRESHOLD = 0.001
EMERGENCY_MAX_HOLD_TIME = 120
EMERGENCY_STOP_LOSS = 0.03
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047
INITIAL_CAPITAL = 10000.0
TAKER_FEE = 0.0005

# Grid Search Parameters
ENTRY_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]
EXIT_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]

print(f"📊 GRID SEARCH PARAMETERS:")
print(f"  Entry Thresholds: {ENTRY_THRESHOLDS}")
print(f"  Exit Thresholds: {EXIT_THRESHOLDS}")
print(f"  Total Combinations: {len(ENTRY_THRESHOLDS) * len(EXIT_THRESHOLDS)}\n")

# =============================================================================
# LOAD MODELS
# =============================================================================

print("Loading models...")
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ Entry Models: LONG {len(long_feature_columns)} features, SHORT {len(short_feature_columns)} features")
print(f"  ✅ Exit Models: LONG {len(long_exit_feature_columns)} features, SHORT {len(short_exit_feature_columns)} features\n")

sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

print("Loading and preparing data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)

SEVEN_DAYS_CANDLES = 7 * 288
BUFFER_CANDLES = 200
TOTAL_CANDLES_NEEDED = SEVEN_DAYS_CANDLES + BUFFER_CANDLES

df_recent = df_full.tail(TOTAL_CANDLES_NEEDED).reset_index(drop=True)

start_time = time.time()
df = calculate_all_features_enhanced_v2(df_recent)
df = prepare_exit_features(df)
df = df.dropna().reset_index(drop=True)
df_test = df.tail(SEVEN_DAYS_CANDLES).reset_index(drop=True)
feature_time = time.time() - start_time

print(f"  ✅ Data prepared: {len(df_test):,} candles ({feature_time:.1f}s)")
print(f"     Period: {df_test['timestamp'].iloc[0]} to {df_test['timestamp'].iloc[-1]}\n")

# Pre-calculate signals
print("Pre-calculating signals...")
long_feat = df_test[long_feature_columns].values
long_feat_scaled = long_scaler.transform(long_feat)
long_probs_array = long_model.predict_proba(long_feat_scaled)[:, 1]
df_test['long_prob'] = long_probs_array

short_feat = df_test[short_feature_columns].values
short_feat_scaled = short_scaler.transform(short_feat)
short_probs_array = short_model.predict_proba(short_feat_scaled)[:, 1]
df_test['short_prob'] = short_probs_array
print(f"  ✅ Signals pre-calculated\n")


# =============================================================================
# BACKTEST FUNCTION
# =============================================================================

def backtest_threshold_combination(test_df, long_threshold, short_threshold,
                                   ml_exit_long, ml_exit_short):
    """Run backtest for specific threshold combination"""
    trades = []
    position = None
    capital = INITIAL_CAPITAL

    for i in range(len(test_df) - 1):
        current_price = test_df['close'].iloc[i]

        # Entry logic
        if position is None:
            long_prob = test_df['long_prob'].iloc[i]
            short_prob = test_df['short_prob'].iloc[i]

            # LONG entry
            if long_prob >= long_threshold:
                sizing_result = sizer.get_position_size_simple(
                    capital=capital,
                    signal_strength=long_prob,
                    leverage=LEVERAGE
                )
                position_size_pct = sizing_result['position_size_pct']
                position_value = capital * position_size_pct
                entry_fee = position_value * LEVERAGE * TAKER_FEE

                position = {
                    'type': 'LONG',
                    'entry_price': current_price,
                    'entry_candle': i,
                    'position_value': position_value,
                    'entry_fee': entry_fee,
                    'position_size_pct': position_size_pct
                }

            # SHORT entry (Opportunity Gating)
            elif short_prob >= short_threshold:
                long_ev = long_prob * LONG_AVG_RETURN
                short_ev = short_prob * SHORT_AVG_RETURN
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > GATE_THRESHOLD:
                    sizing_result = sizer.get_position_size_simple(
                        capital=capital,
                        signal_strength=short_prob,
                        leverage=LEVERAGE
                    )
                    position_size_pct = sizing_result['position_size_pct']
                    position_value = capital * position_size_pct
                    entry_fee = position_value * LEVERAGE * TAKER_FEE

                    position = {
                        'type': 'SHORT',
                        'entry_price': current_price,
                        'entry_candle': i,
                        'position_value': position_value,
                        'entry_fee': entry_fee,
                        'position_size_pct': position_size_pct
                    }

        # Exit logic
        else:
            candles_held = i - position['entry_candle']

            if position['type'] == 'LONG':
                price_change_pct = (current_price - position['entry_price']) / position['entry_price']
                leveraged_pnl_pct = price_change_pct * LEVERAGE
            else:  # SHORT
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
                leveraged_pnl_pct = price_change_pct * LEVERAGE

            balance_pnl_pct = leveraged_pnl_pct * position['position_size_pct']

            # Emergency Stop Loss
            if balance_pnl_pct <= -EMERGENCY_STOP_LOSS:
                pnl = capital * balance_pnl_pct
                exit_fee = position['position_value'] * LEVERAGE * TAKER_FEE
                net_pnl = pnl - position['entry_fee'] - exit_fee
                capital += net_pnl

                trades.append({
                    'type': position['type'],
                    'pnl': net_pnl,
                    'exit_reason': 'EMERGENCY_STOP_LOSS',
                    'candles_held': candles_held
                })
                position = None
                continue

            # Emergency Max Hold
            if candles_held >= EMERGENCY_MAX_HOLD_TIME:
                pnl = capital * balance_pnl_pct
                exit_fee = position['position_value'] * LEVERAGE * TAKER_FEE
                net_pnl = pnl - position['entry_fee'] - exit_fee
                capital += net_pnl

                trades.append({
                    'type': position['type'],
                    'pnl': net_pnl,
                    'exit_reason': 'EMERGENCY_MAX_HOLD',
                    'candles_held': candles_held
                })
                position = None
                continue

            # ML Exit
            try:
                if position['type'] == 'LONG':
                    exit_features_values = test_df[long_exit_feature_columns].iloc[i:i+1].values
                    exit_features_scaled = long_exit_scaler.transform(exit_features_values)
                    exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0][1]
                    exit_threshold = ml_exit_long
                else:  # SHORT
                    exit_features_values = test_df[short_exit_feature_columns].iloc[i:i+1].values
                    exit_features_scaled = short_exit_scaler.transform(exit_features_values)
                    exit_prob = short_exit_model.predict_proba(exit_features_scaled)[0][1]
                    exit_threshold = ml_exit_short

                if exit_prob >= exit_threshold:
                    pnl = capital * balance_pnl_pct
                    exit_fee = position['position_value'] * LEVERAGE * TAKER_FEE
                    net_pnl = pnl - position['entry_fee'] - exit_fee
                    capital += net_pnl

                    trades.append({
                        'type': position['type'],
                        'pnl': net_pnl,
                        'exit_reason': 'ML_EXIT',
                        'candles_held': candles_held
                    })
                    position = None
            except:
                pass

    # Calculate metrics
    if len(trades) == 0:
        return {
            'long_threshold': long_threshold,
            'short_threshold': short_threshold,
            'ml_exit_long': ml_exit_long,
            'ml_exit_short': ml_exit_short,
            'total_return': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_trade': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'final_capital': INITIAL_CAPITAL
        }

    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades['pnl'] > 0]
    win_rate = (len(wins) / len(df_trades)) * 100

    total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    avg_trade = df_trades['pnl'].mean()

    # Max Drawdown
    returns_array = df_trades['pnl'].values / INITIAL_CAPITAL * 100
    max_drawdown = 0
    peak = 0
    cumulative = 0

    for ret in returns_array:
        cumulative += ret
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Sharpe Ratio
    sharpe_ratio = 0
    if len(returns_array) > 1 and returns_array.std() != 0:
        sharpe_ratio = (returns_array.mean() / returns_array.std()) * np.sqrt(len(returns_array))

    return {
        'long_threshold': long_threshold,
        'short_threshold': short_threshold,
        'ml_exit_long': ml_exit_long,
        'ml_exit_short': ml_exit_short,
        'total_return': total_return,
        'total_trades': len(df_trades),
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'final_capital': capital,
        'ml_exit_pct': (df_trades['exit_reason'] == 'ML_EXIT').sum() / len(df_trades) * 100,
        'avg_hold_time': df_trades['candles_held'].mean()
    }


# =============================================================================
# RUN GRID SEARCH
# =============================================================================

print("="*80)
print("RUNNING GRID SEARCH...")
print("="*80)
print()

results = []
total_combinations = len(ENTRY_THRESHOLDS) * len(EXIT_THRESHOLDS)
current = 0

start_time = time.time()

# Test all combinations
for entry_threshold in ENTRY_THRESHOLDS:
    for exit_threshold in EXIT_THRESHOLDS:
        current += 1

        # Run backtest
        result = backtest_threshold_combination(
            df_test,
            long_threshold=entry_threshold,
            short_threshold=entry_threshold,  # Same for LONG/SHORT
            ml_exit_long=exit_threshold,
            ml_exit_short=exit_threshold
        )

        results.append(result)

        # Progress update
        if current % 5 == 0 or current == total_combinations:
            elapsed = time.time() - start_time
            avg_time = elapsed / current
            remaining = (total_combinations - current) * avg_time

            print(f"  [{current:2d}/{total_combinations}] "
                  f"Entry={entry_threshold:.2f}, Exit={exit_threshold:.2f} → "
                  f"Return={result['total_return']:+.2f}%, "
                  f"WR={result['win_rate']:.1f}%, "
                  f"Trades={result['total_trades']} "
                  f"({remaining:.0f}s remaining)")

total_time = time.time() - start_time
print(f"\n✅ Grid search complete ({total_time:.1f}s)\n")


# =============================================================================
# ANALYZE RESULTS
# =============================================================================

df_results = pd.DataFrame(results)

# Sort by composite score
df_results['composite_score'] = (
    df_results['total_return'] / 100 * 0.4 +
    df_results['win_rate'] / 100 * 0.2 +
    df_results['sharpe_ratio'] / 2 * 0.2 +
    (100 - df_results['max_drawdown']) / 100 * 0.2
)

df_results = df_results.sort_values('composite_score', ascending=False)

print("="*80)
print("GRID SEARCH RESULTS")
print("="*80)
print()

# Top 10 Results
print("🏆 TOP 10 CONFIGURATIONS:")
print()
print(f"{'Rank':<6} {'Entry':<7} {'Exit':<7} {'Return':<10} {'WR':<8} {'Trades':<8} {'Sharpe':<8} {'MDD':<8} {'Score':<8}")
print("-" * 80)

for idx, row in df_results.head(10).iterrows():
    print(f"{df_results.index.get_loc(idx)+1:<6} "
          f"{row['long_threshold']:<7.2f} "
          f"{row['ml_exit_long']:<7.2f} "
          f"{row['total_return']:+8.2f}% "
          f"{row['win_rate']:6.1f}% "
          f"{row['total_trades']:<8.0f} "
          f"{row['sharpe_ratio']:7.3f} "
          f"{row['max_drawdown']:6.2f}% "
          f"{row['composite_score']:7.3f}")

# Best by individual metrics
print("\n" + "="*80)
print("BEST BY INDIVIDUAL METRICS:")
print("="*80)
print()

best_return = df_results.loc[df_results['total_return'].idxmax()]
print(f"📈 HIGHEST RETURN ({best_return['total_return']:+.2f}%):")
print(f"   Entry={best_return['long_threshold']:.2f}, Exit={best_return['ml_exit_long']:.2f}")
print(f"   Win Rate: {best_return['win_rate']:.1f}%, Trades: {best_return['total_trades']:.0f}")
print()

best_winrate = df_results.loc[df_results['win_rate'].idxmax()]
print(f"🎯 HIGHEST WIN RATE ({best_winrate['win_rate']:.1f}%):")
print(f"   Entry={best_winrate['long_threshold']:.2f}, Exit={best_winrate['ml_exit_long']:.2f}")
print(f"   Return: {best_winrate['total_return']:+.2f}%, Trades: {best_winrate['total_trades']:.0f}")
print()

best_sharpe = df_results.loc[df_results['sharpe_ratio'].idxmax()]
print(f"📊 HIGHEST SHARPE RATIO ({best_sharpe['sharpe_ratio']:.3f}):")
print(f"   Entry={best_sharpe['long_threshold']:.2f}, Exit={best_sharpe['ml_exit_long']:.2f}")
print(f"   Return: {best_sharpe['total_return']:+.2f}%, WR: {best_sharpe['win_rate']:.1f}%")
print()

best_dd = df_results.loc[df_results['max_drawdown'].idxmin()]
print(f"🛡️ LOWEST MAX DRAWDOWN ({best_dd['max_drawdown']:.2f}%):")
print(f"   Entry={best_dd['long_threshold']:.2f}, Exit={best_dd['ml_exit_long']:.2f}")
print(f"   Return: {best_dd['total_return']:+.2f}%, WR: {best_dd['win_rate']:.1f}%")
print()

# Recommended Configuration
best_overall = df_results.iloc[0]
print("="*80)
print("🎯 RECOMMENDED CONFIGURATION (Rank 1 by Composite Score):")
print("="*80)
print()
print(f"  Entry Threshold: {best_overall['long_threshold']:.2f}")
print(f"  Exit Threshold: {best_overall['ml_exit_long']:.2f}")
print()
print(f"  Expected Performance:")
print(f"    Return: {best_overall['total_return']:+.2f}%")
print(f"    Win Rate: {best_overall['win_rate']:.1f}%")
print(f"    Trades: {best_overall['total_trades']:.0f}")
print(f"    Sharpe Ratio: {best_overall['sharpe_ratio']:.3f}")
print(f"    Max Drawdown: {best_overall['max_drawdown']:.2f}%")
print(f"    ML Exit Usage: {best_overall['ml_exit_pct']:.1f}%")
print(f"    Avg Hold Time: {best_overall['avg_hold_time']:.0f} candles")
print()

# Save results
output_file = RESULTS_DIR / f"grid_search_thresholds_7days_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df_results.to_csv(output_file, index=False)
print(f"💾 Results saved to: {output_file}")
print()

print("="*80)
print("GRID SEARCH COMPLETE")
print("="*80)
