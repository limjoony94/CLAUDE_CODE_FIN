"""
Full Optimization System - 6가지 전략 통합
목표: LONG+SHORT > LONG-only (+10.14%)

전략:
1. Dynamic Position Sizing - 신호 강도 기반
2. Adaptive Exit - 변동성 기반 SL/TP
3. Threshold Optimization - Grid search
4. SHORT Timing Filter - Regime 기반
5. Multi-Timeframe Confirmation - 5m+15m+1h
6. Window Size Optimization - 최적 크기
"""

import pandas as pd
import numpy as np
import pickle
import talib
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.experiments.feature_utils import calculate_short_features_optimized

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Base parameters
INITIAL_CAPITAL = 10000.0
LEVERAGE = 4
TRANSACTION_COST = 0.0002

def get_dynamic_position_size(signal_prob):
    """Strategy 1: Dynamic Position Sizing"""
    if signal_prob >= 0.85:
        return 0.95  # Very strong signal
    elif signal_prob >= 0.75:
        return 0.80  # Strong signal
    elif signal_prob >= 0.65:
        return 0.65  # Medium signal
    else:
        return 0.50  # Weak signal

def get_adaptive_exit_params(current_atr, current_price, base_sl=0.01, base_tp=0.02):
    """Strategy 2: Adaptive Exit based on volatility"""
    atr_pct = current_atr / current_price
    volatility_multiplier = max(0.5, min(2.0, atr_pct / 0.01))  # 0.5x to 2.0x

    stop_loss = base_sl * volatility_multiplier
    take_profit = base_tp * volatility_multiplier
    max_hold_hours = int(4 * (2 - volatility_multiplier))  # 2-6 hours

    return stop_loss, take_profit, max_hold_hours

def classify_market_regime(df, idx, lookback=20):
    """Strategy 4: Market Regime Classification"""
    if idx < lookback:
        return 'SIDEWAYS'

    returns = df['close'].iloc[idx] / df['close'].iloc[idx-lookback] - 1

    if returns > 0.02:
        return 'BULL'
    elif returns < -0.02:
        return 'BEAR'
    else:
        return 'SIDEWAYS'

def get_multiframe_confirmation(df, idx, long_feat_cols, short_feat_cols,
                                long_model, long_scaler, short_model, short_scaler):
    """Strategy 5: Multi-Timeframe Confirmation (simplified - using trend)"""
    # Use EMA trends as proxy for higher timeframes
    ema_12 = df['ema_12'].iloc[idx] if 'ema_12' in df.columns else df['close'].iloc[idx]
    ema_26 = df['close'].ewm(span=26).mean().iloc[idx]
    trend = 1 if ema_12 > ema_26 else -1

    # Get 5m signals
    long_feat = df[long_feat_cols].iloc[idx:idx+1].values
    if not np.isnan(long_feat).any():
        long_feat_scaled = long_scaler.transform(long_feat)
        long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
    else:
        long_prob = 0.0

    short_feat = df[short_feat_cols].iloc[idx:idx+1].values
    if not np.isnan(short_feat).any():
        short_feat_scaled = short_scaler.transform(short_feat)
        short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
    else:
        short_prob = 0.0

    # Apply trend confirmation
    if trend > 0:
        long_prob *= 1.1  # Boost LONG in uptrend
        short_prob *= 0.9  # Reduce SHORT in uptrend
    else:
        long_prob *= 0.9  # Reduce LONG in downtrend
        short_prob *= 1.1  # Boost SHORT in downtrend

    return min(1.0, long_prob), min(1.0, short_prob)

def backtest_full_optimization(df, long_model, long_scaler, short_model, short_scaler,
                               long_features, short_features,
                               long_threshold=0.65, short_threshold=0.75,
                               use_dynamic_sizing=True, use_adaptive_exit=True,
                               use_regime_filter=True, use_multiframe=True):
    """
    Full Optimization Backtest

    Integrates all 6 strategies:
    1. Dynamic Position Sizing
    2. Adaptive Exit
    3. Threshold (parameter)
    4. Regime Filter
    5. Multi-Timeframe
    6. Window Size (external parameter)
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        current_atr = df['atr'].iloc[i] if 'atr' in df.columns else current_price * 0.01

        # Manage existing position
        if position is not None:
            side = position['side']
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            max_hold = position['max_hold_hours']

            hours_held = (i - entry_idx) / 12  # 5-min candles

            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # Exit logic with adaptive params
            exit_reason = None
            if pnl_pct <= -stop_loss:
                exit_reason = "SL"
            elif pnl_pct >= take_profit:
                exit_reason = "TP"
            elif hours_held >= max_hold:
                exit_reason = "MH"

            if exit_reason:
                quantity = position['quantity']
                pnl_usd = pnl_pct * (entry_price * quantity)
                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                pnl_usd -= (entry_cost + exit_cost)
                capital += pnl_usd

                trades.append({
                    'side': side,
                    'pnl_usd': pnl_usd,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'hours_held': hours_held
                })

                position = None

        # Entry logic with all optimizations
        if position is None and i < len(df) - 1:
            # Get signals (with multi-timeframe if enabled)
            if use_multiframe:
                long_prob, short_prob = get_multiframe_confirmation(
                    df, i, long_features, short_features,
                    long_model, long_scaler, short_model, short_scaler
                )
            else:
                # Standard signal
                long_feat = df[long_features].iloc[i:i+1].values
                if not np.isnan(long_feat).any():
                    long_feat_scaled = long_scaler.transform(long_feat)
                    long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
                else:
                    long_prob = 0.0

                short_feat = df[short_features].iloc[i:i+1].values
                if not np.isnan(short_feat).any():
                    short_feat_scaled = short_scaler.transform(short_feat)
                    short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
                else:
                    short_prob = 0.0

            # Regime filter for SHORT
            if use_regime_filter:
                regime = classify_market_regime(df, i)
                if regime != 'BEAR':
                    short_prob *= 0.5  # Significantly reduce SHORT in non-BEAR

            # Priority: LONG first
            side = None
            signal_prob = 0.0

            if long_prob >= long_threshold:
                side = 'LONG'
                signal_prob = long_prob
            elif short_prob >= short_threshold:
                side = 'SHORT'
                signal_prob = short_prob

            if side is not None:
                # Dynamic position sizing
                if use_dynamic_sizing:
                    position_size_pct = get_dynamic_position_size(signal_prob)
                else:
                    position_size_pct = 0.95

                # Adaptive exit parameters
                if use_adaptive_exit:
                    stop_loss, take_profit, max_hold_hours = get_adaptive_exit_params(
                        current_atr, current_price
                    )
                else:
                    stop_loss, take_profit, max_hold_hours = 0.01, 0.02, 4

                position_value = capital * position_size_pct
                quantity = position_value / current_price

                position = {
                    'side': side,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'max_hold_hours': max_hold_hours,
                    'signal_prob': signal_prob
                }

    # Calculate metrics
    if len(trades) == 0:
        return {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'num_long': 0,
            'num_short': 0,
            'long_wr': 0.0,
            'short_wr': 0.0,
            'avg_pnl': 0.0
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']

    return {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'num_long': len(long_trades),
        'num_short': len(short_trades),
        'long_wr': (len([t for t in long_trades if t['pnl_usd'] > 0]) / len(long_trades) * 100) if long_trades else 0.0,
        'short_wr': (len([t for t in short_trades if t['pnl_usd'] > 0]) / len(short_trades) * 100) if short_trades else 0.0,
        'avg_pnl': np.mean([t['pnl_pct'] for t in trades]) * 100,
        'final_capital': capital
    }

print("="*80)
print("🚀 FULL OPTIMIZATION SYSTEM")
print("="*80)
print("\n목표: LONG+SHORT > LONG-only (+10.14%)")
print("전략: 6가지 최적화 통합\n")

# Load models
print("Loading models...")
long_model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
with open(long_scaler_path, 'rb') as f:
    long_scaler = pickle.load(f)

long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl"
with open(short_scaler_path, 'rb') as f:
    short_scaler = pickle.load(f)

short_features_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print(f"  ✅ Models loaded\n")

# Load and prepare data
print("Loading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)

print("Calculating features (optimized)...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = calculate_short_features_optimized(df)
print(f"  ✅ Data ready: {len(df)} rows\n")

# Strategy 3 & 6: Grid Search - Threshold + Window Size
print("="*80)
print("GRID SEARCH: Threshold × Window Size")
print("="*80)

threshold_configs = [
    {'long': 0.60, 'short': 0.75},
    {'long': 0.60, 'short': 0.80},
    {'long': 0.65, 'short': 0.75},
    {'long': 0.65, 'short': 0.80},
    {'long': 0.70, 'short': 0.75},
]

window_sizes = [1440, 2160, 2880]  # 5 days, 7.5 days, 10 days
step_size = 288  # 1 day

all_results = []

for window_size in window_sizes:
    print(f"\n{'='*80}")
    print(f"Window Size: {window_size} candles ({window_size/288:.1f} days)")
    print(f"{'='*80}")

    for config in threshold_configs:
        long_thresh = config['long']
        short_thresh = config['short']

        print(f"\nTesting LONG {long_thresh:.2f}, SHORT {short_thresh:.2f}...", end='')

        # Run backtest on all windows
        windows = []
        start_idx = 0

        while start_idx + window_size <= len(df):
            end_idx = start_idx + window_size
            window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

            metrics = backtest_full_optimization(
                window_df, long_model, long_scaler, short_model, short_scaler,
                long_feature_columns, short_feature_columns,
                long_threshold=long_thresh,
                short_threshold=short_thresh,
                use_dynamic_sizing=True,
                use_adaptive_exit=True,
                use_regime_filter=True,
                use_multiframe=True
            )

            windows.append(metrics)
            start_idx += step_size

        results_df = pd.DataFrame(windows)

        summary = {
            'window_size': window_size,
            'window_days': window_size / 288,
            'long_threshold': long_thresh,
            'short_threshold': short_thresh,
            'avg_return': results_df['total_return_pct'].mean(),
            'win_windows': (results_df['total_return_pct'] > 0).sum() / len(results_df) * 100,
            'avg_trades': results_df['num_trades'].mean(),
            'avg_long': results_df['num_long'].mean(),
            'avg_short': results_df['num_short'].mean(),
            'long_wr': results_df['long_wr'].mean(),
            'short_wr': results_df['short_wr'].mean(),
            'avg_pnl': results_df['avg_pnl'].mean()
        }

        all_results.append(summary)

        status = "✅" if summary['avg_return'] > 10.14 else "⚠️" if summary['avg_return'] > 8.0 else "❌"
        print(f" {status} {summary['avg_return']:.2f}% | LONG: {summary['avg_long']:.1f}, SHORT: {summary['avg_short']:.1f}")

# Analyze results
print(f"\n{'='*80}")
print("OPTIMIZATION RESULTS")
print(f"{'='*80}\n")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('avg_return', ascending=False)

print("Top 10 Configurations:")
print(results_df.head(10)[['window_days', 'long_threshold', 'short_threshold', 'avg_return', 'avg_long', 'avg_short', 'long_wr', 'short_wr']].to_string(index=False))

best = results_df.iloc[0]

print(f"\n{'='*80}")
print("BEST CONFIGURATION")
print(f"{'='*80}")
print(f"\nWindow Size: {best['window_size']:.0f} candles ({best['window_days']:.1f} days)")
print(f"LONG Threshold: {best['long_threshold']:.2f}")
print(f"SHORT Threshold: {best['short_threshold']:.2f}")
print(f"\nPerformance:")
print(f"  Average Return: {best['avg_return']:.2f}% per window")
print(f"  Win Rate (windows): {best['win_windows']:.1f}%")
print(f"  Trades/window: {best['avg_trades']:.1f} (LONG: {best['avg_long']:.1f}, SHORT: {best['avg_short']:.1f})")
print(f"  LONG WR: {best['long_wr']:.1f}%")
print(f"  SHORT WR: {best['short_wr']:.1f}%")
print(f"  Avg P&L: {best['avg_pnl']:.2f}%")

print(f"\n{'='*80}")
print("COMPARISON vs BASELINE")
print(f"{'='*80}")
print(f"\nLONG-only Baseline: +10.14% per window")
print(f"Full Optimization:  +{best['avg_return']:.2f}% per window")

if best['avg_return'] > 10.14:
    improvement = best['avg_return'] - 10.14
    pct_improvement = (improvement / 10.14) * 100
    print(f"\n✅ SUCCESS! LONG+SHORT > LONG-only")
    print(f"   Improvement: +{improvement:.2f}% ({pct_improvement:+.1f}%)")
    print(f"   🎯 목표 달성! 🎉")
else:
    gap = 10.14 - best['avg_return']
    print(f"\n⚠️ Almost there! Gap: -{gap:.2f}%")
    print(f"   Progress: {best['avg_return']/10.14*100:.1f}% of target")

# Save results
output_file = RESULTS_DIR / "full_optimization_results.csv"
results_df.to_csv(output_file, index=False)
print(f"\n💾 Results saved: {output_file}")

print(f"\n{'='*80}")
print("OPTIMIZATION COMPLETE!")
print(f"{'='*80}\n")
