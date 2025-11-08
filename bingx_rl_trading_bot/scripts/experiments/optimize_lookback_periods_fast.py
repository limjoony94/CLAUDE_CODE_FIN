"""
Lookback Period Optimization - Fast Version (12 combinations)
==============================================================

Quick test of key lookback period variations:
- Baseline: RSI=14, MACD=12/26/9, BB=20, ATR=14, EMA=12
- Test each parameter variation independently
- Test COMBINED optimization (ATR21 + BB25 + EMA8)

Total Combinations: 12 (1 baseline + 2×5 variations + 1 combined)
Expected Time: ~1-2 hours

Created: 2025-10-23
Updated: 2025-10-23 (Added COMBINED test)
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.dynamic_position_sizing import DynamicPositionSizer

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("LOOKBACK PERIOD OPTIMIZATION - FAST VERSION")
print("="*80)
print(f"\nTesting 12 parameter combinations (11 individual + 1 combined)")
print(f"Expected time: ~2 hours\n")

# ============================================================================
# Test Combinations (Baseline + Individual Variations)
# ============================================================================

BASELINE = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'atr_period': 14,
    'ema_period': 12
}

TEST_COMBINATIONS = [
    # Baseline
    ('Baseline', BASELINE),

    # RSI variations
    ('RSI_10', {**BASELINE, 'rsi_period': 10}),
    ('RSI_20', {**BASELINE, 'rsi_period': 20}),

    # MACD variations
    ('MACD_8_17', {**BASELINE, 'macd_fast': 8, 'macd_slow': 17}),
    ('MACD_16_35', {**BASELINE, 'macd_fast': 16, 'macd_slow': 35}),

    # Bollinger Bands variations
    ('BB_15', {**BASELINE, 'bb_period': 15}),
    ('BB_25', {**BASELINE, 'bb_period': 25}),

    # ATR variations
    ('ATR_7', {**BASELINE, 'atr_period': 7}),
    ('ATR_21', {**BASELINE, 'atr_period': 21}),

    # EMA variations
    ('EMA_8', {**BASELINE, 'ema_period': 8}),
    ('EMA_20', {**BASELINE, 'ema_period': 20}),

    # COMBINED: Best individual optimizations
    ('COMBINED (ATR21+BB25+EMA8)', {
        'rsi_period': 14,      # Keep baseline (best)
        'macd_fast': 12,       # Keep baseline (best)
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 25,       # Optimized
        'atr_period': 21,      # Optimized
        'ema_period': 8        # Optimized
    }),
]

print(f"Test Combinations:")
for i, (name, params) in enumerate(TEST_COMBINATIONS, 1):
    print(f"  {i}. {name}: RSI={params['rsi_period']}, MACD={params['macd_fast']}/{params['macd_slow']}, BB={params['bb_period']}, ATR={params['atr_period']}, EMA={params['ema_period']}")

# ============================================================================
# Load Production Models
# ============================================================================
print(f"\n{'='*80}")
print("Loading Production Models")
print(f"{'='*80}")

timestamp = "20251018_233146"

# LONG Entry
with open(MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}.pkl", 'rb') as f:
    long_model = pickle.load(f)
long_scaler = joblib.load(MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}_features.txt", 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

# SHORT Entry
with open(MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}.pkl", 'rb') as f:
    short_model = pickle.load(f)
short_scaler = joblib.load(MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}_features.txt", 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

# Exit Models
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt", 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt", 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines()]

print(f"✅ Models loaded")

# ============================================================================
# Load Base Data
# ============================================================================
print(f"\n{'='*80}")
print("Loading Base Data")
print(f"{'='*80}")

df_base = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
print(f"✅ Loaded {len(df_base):,} candles")

# ============================================================================
# Minimal Feature Calculation (only optimized features)
# ============================================================================

def calculate_minimal_features(df, params):
    """
    Calculate only the features affected by lookback period changes
    Reuse original features for everything else
    """
    df = df.copy()

    # Extract parameters
    rsi_period = params['rsi_period']
    macd_fast = params['macd_fast']
    macd_slow = params['macd_slow']
    macd_signal = params['macd_signal']
    bb_period = params['bb_period']
    atr_period = params['atr_period']
    ema_period = params['ema_period']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df['close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
    bb_std = df['close'].rolling(window=bb_period).std()
    df['bb_high'] = df['bb_mid'] + (2 * bb_std)
    df['bb_low'] = df['bb_mid'] - (2 * bb_std)
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=atr_period).mean()
    df['atr_pct'] = df['atr'] / df['close']

    # EMA
    df['ema_12'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=ema_period*2, adjust=False).mean()

    return df

# Load full features ONCE (for baseline)
print(f"\n{'='*80}")
print("Calculating Full Feature Set (One Time)")
print(f"{'='*80}")

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

df_full = calculate_all_features(df_base.copy())
df_full = prepare_exit_features(df_full)
df_full = df_full.dropna()

print(f"✅ Full features calculated: {len(df_full):,} candles")

# Store baseline feature values
baseline_features = {}
for col in df_full.columns:
    if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']:
        baseline_features[col] = df_full[col].copy()

# ============================================================================
# Backtest Function
# ============================================================================

def run_backtest(df, name, params):
    """Run backtest with given parameters"""

    # Trading parameters
    LEVERAGE = 4
    LONG_THRESHOLD = 0.65
    SHORT_THRESHOLD = 0.70
    GATE_THRESHOLD = 0.001
    ML_EXIT_THRESHOLD_LONG = 0.75
    ML_EXIT_THRESHOLD_SHORT = 0.75
    EMERGENCY_MAX_HOLD_TIME = 120
    EMERGENCY_STOP_LOSS = -0.03
    LONG_AVG_RETURN = 0.0041
    SHORT_AVG_RETURN = 0.0047
    INITIAL_CAPITAL = 10000.0
    TAKER_FEE = 0.0005

    position_sizer = DynamicPositionSizer(min_position_pct=0.20, max_position_pct=0.95, base_position_pct=0.50)
    balance = INITIAL_CAPITAL
    position = None
    trades = []

    for idx in range(len(df)):
        current_price = df['close'].iloc[idx]

        # Position Management
        if position is not None:
            direction = position['direction']
            entry_price = position['entry_price']
            size_pct = position['size_pct']
            entry_idx = position['entry_idx']
            candles_held = idx - entry_idx

            if direction == 'LONG':
                price_change_pct = (current_price - entry_price) / entry_price
            else:
                price_change_pct = (entry_price - current_price) / entry_price

            leveraged_pnl_pct = price_change_pct * LEVERAGE
            balance_pnl_pct = leveraged_pnl_pct * size_pct

            should_exit = False
            exit_reason = None

            if balance_pnl_pct <= EMERGENCY_STOP_LOSS:
                should_exit, exit_reason = True, 'STOP_LOSS'
            elif candles_held >= EMERGENCY_MAX_HOLD_TIME:
                should_exit, exit_reason = True, 'MAX_HOLD'
            else:
                try:
                    exit_features = df[long_exit_feature_columns if direction == 'LONG' else short_exit_feature_columns].iloc[idx].values.reshape(1, -1)
                    exit_scaler_obj = long_exit_scaler if direction == 'LONG' else short_exit_scaler
                    exit_model_obj = long_exit_model if direction == 'LONG' else short_exit_model
                    threshold = ML_EXIT_THRESHOLD_LONG if direction == 'LONG' else ML_EXIT_THRESHOLD_SHORT

                    exit_features_scaled = exit_scaler_obj.transform(exit_features)
                    exit_prob = exit_model_obj.predict_proba(exit_features_scaled)[0][1]

                    if exit_prob >= threshold:
                        should_exit, exit_reason = True, 'ML'
                except:
                    pass

            if should_exit:
                position_value = balance * size_pct
                leveraged_position = position_value * LEVERAGE
                total_fees = leveraged_position * TAKER_FEE * 2
                gross_pnl = balance * balance_pnl_pct
                net_pnl = gross_pnl - total_fees
                balance += net_pnl

                trades.append({
                    'direction': direction,
                    'net_pnl': net_pnl,
                    'balance_pnl_pct': balance_pnl_pct,
                    'exit_reason': exit_reason,
                    'balance_after': balance
                })

                position = None

        # Entry Signals
        if position is None:
            try:
                long_features = df[long_feature_columns].iloc[idx].values.reshape(1, -1)
                long_features_scaled = long_scaler.transform(long_features)
                long_prob = long_model.predict_proba(long_features_scaled)[0][1]

                short_features = df[short_feature_columns].iloc[idx].values.reshape(1, -1)
                short_features_scaled = short_scaler.transform(short_features)
                short_prob = short_model.predict_proba(short_features_scaled)[0][1]

                if long_prob >= LONG_THRESHOLD:
                    sizing_result = position_sizer.get_position_size_simple(balance, long_prob, LEVERAGE)
                    position = {
                        'direction': 'LONG',
                        'entry_price': current_price,
                        'size_pct': sizing_result['position_size_pct'],
                        'entry_idx': idx
                    }
                elif short_prob >= SHORT_THRESHOLD:
                    long_ev = long_prob * LONG_AVG_RETURN
                    short_ev = short_prob * SHORT_AVG_RETURN
                    if (short_ev - long_ev) > GATE_THRESHOLD:
                        sizing_result = position_sizer.get_position_size_simple(balance, short_prob, LEVERAGE)
                        position = {
                            'direction': 'SHORT',
                            'entry_price': current_price,
                            'size_pct': sizing_result['position_size_pct'],
                            'entry_idx': idx
                        }
            except:
                pass

    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)
    total_return = (balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate = len(trades_df[trades_df['net_pnl'] > 0]) / len(trades_df) * 100

    cumulative = trades_df['balance_after'].values
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    returns = trades_df['balance_pnl_pct'].values
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 12) if returns.std() > 0 else 0

    return {
        'name': name,
        **params,
        'total_return': total_return,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'num_trades': len(trades_df),
        'final_balance': balance
    }

# ============================================================================
# Run Tests
# ============================================================================

print(f"\n{'='*80}")
print("Running Tests")
print(f"{'='*80}\n")

results = []
start_time = time.time()

for i, (name, params) in enumerate(TEST_COMBINATIONS, 1):
    print(f"[{i}/11] Testing {name}...", end=' ', flush=True)

    test_start = time.time()

    try:
        # Start with full baseline features
        df_test = df_full.copy()

        # Override only the optimized features
        df_optimized = calculate_minimal_features(df_base.copy(), params)

        # Replace optimized features
        for col in ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_mid', 'bb_high', 'bb_low', 'bb_width', 'atr', 'atr_pct', 'ema_12', 'ema_26']:
            if col in df_optimized.columns and col in df_test.columns:
                df_test[col] = df_optimized[col]

        # Clean NaN
        df_test = df_test.dropna()

        # Run backtest
        result = run_backtest(df_test, name, params)

        if result is not None:
            results.append(result)
            test_time = time.time() - test_start
            print(f"✅ {test_time:.1f}s | Return: {result['total_return']:+.2f}% | WR: {result['win_rate']:.1f}%")
        else:
            print(f"❌ No trades")

    except Exception as e:
        print(f"❌ Error: {e}")

elapsed = time.time() - start_time
print(f"\n✅ Tests complete in {elapsed/60:.1f} minutes")

# ============================================================================
# Results
# ============================================================================

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}\n")

if len(results) > 0:
    results_df = pd.DataFrame(results)

    # Calculate scores
    results_df['return_score'] = (results_df['total_return'] - results_df['total_return'].min()) / (results_df['total_return'].max() - results_df['total_return'].min()) if results_df['total_return'].max() != results_df['total_return'].min() else 0.5
    results_df['wr_score'] = (results_df['win_rate'] - results_df['win_rate'].min()) / (results_df['win_rate'].max() - results_df['win_rate'].min()) if results_df['win_rate'].max() != results_df['win_rate'].min() else 0.5
    results_df['sharpe_score'] = (results_df['sharpe_ratio'] - results_df['sharpe_ratio'].min()) / (results_df['sharpe_ratio'].max() - results_df['sharpe_ratio'].min()) if results_df['sharpe_ratio'].max() != results_df['sharpe_ratio'].min() else 0.5
    results_df['dd_score'] = (results_df['max_drawdown'].max() - results_df['max_drawdown']) / (results_df['max_drawdown'].max() - results_df['max_drawdown'].min()) if results_df['max_drawdown'].max() != results_df['max_drawdown'].min() else 0.5

    results_df['composite_score'] = (
        results_df['return_score'] * 0.30 +
        results_df['wr_score'] * 0.20 +
        results_df['sharpe_score'] * 0.40 +
        results_df['dd_score'] * 0.10
    )

    results_df = results_df.sort_values('composite_score', ascending=False)

    print("All Results (sorted by composite score):")
    print("="*100)
    print(f"{'Rank':<5} {'Name':<15} {'Return':<9} {'WR%':<7} {'Sharpe':<8} {'MDD%':<8} {'Trades':<7} {'Score':<7}")
    print("="*100)

    for idx, row in results_df.iterrows():
        print(f"{list(results_df.index).index(idx)+1:<5} {row['name']:<15} {row['total_return']:>8.2f}% {row['win_rate']:>6.1f}% {row['sharpe_ratio']:>7.3f} {row['max_drawdown']:>7.2f}% {int(row['num_trades']):<7} {row['composite_score']:>6.4f}")

    # Save
    results_path = RESULTS_DIR / f"lookback_optimization_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path.name}")

    # Best vs Baseline
    best = results_df.iloc[0]
    baseline_result = results_df[results_df['name'] == 'Baseline']

    if len(baseline_result) > 0:
        baseline = baseline_result.iloc[0]
        print(f"\n{'='*80}")
        print("COMPARISON: Best vs Baseline")
        print(f"{'='*80}")
        print(f"""
BASELINE (Standard Periods):
  Return:    {baseline['total_return']:+.2f}%
  Win Rate:  {baseline['win_rate']:.1f}%
  Sharpe:    {baseline['sharpe_ratio']:.3f}
  Max DD:    {baseline['max_drawdown']:.2f}%

BEST ({best['name']}):
  Return:    {best['total_return']:+.2f}%  ({(best['total_return']-baseline['total_return']):+.2f}pp)
  Win Rate:  {best['win_rate']:.1f}%  ({(best['win_rate']-baseline['win_rate']):+.1f}pp)
  Sharpe:    {best['sharpe_ratio']:.3f}  ({(best['sharpe_ratio']-baseline['sharpe_ratio']):+.3f})
  Max DD:    {best['max_drawdown']:.2f}%  ({(best['max_drawdown']-baseline['max_drawdown']):+.2f}pp)

Improvement: {(best['composite_score']/baseline['composite_score']-1)*100:+.1f}%
""")

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")

    if best['name'] == 'Baseline':
        print("\n✅ BASELINE IS OPTIMAL")
        print("   Standard periods (RSI=14, MACD=12/26, BB=20, ATR=14, EMA=12) perform best")
        print("   No need to change lookback periods")
    else:
        improvement = (best['composite_score'] / baseline['composite_score'] - 1) * 100
        if improvement > 5:
            print(f"\n✅ SIGNIFICANT IMPROVEMENT FOUND ({improvement:+.1f}%)")
            print(f"   Optimal: {best['name']}")
            print(f"   Recommend: Deploy optimized parameters")
        else:
            print(f"\n⚠️ MARGINAL IMPROVEMENT ({improvement:+.1f}%)")
            print(f"   Optimal: {best['name']}")
            print(f"   Recommend: Keep baseline (difference too small)")
else:
    print("❌ No successful results")

print(f"\n{'='*80}")
print("OPTIMIZATION COMPLETE")
print(f"{'='*80}\n")
