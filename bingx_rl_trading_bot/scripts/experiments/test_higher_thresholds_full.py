"""
Higher Threshold Testing for Enhanced SHORT Entry (Full Dataset)

Goal: Find threshold where LONG+SHORT > LONG-only (+10.14%)
Test: 0.70, 0.75, 0.80, 0.85, 0.90, 0.95

Strategy: Higher threshold → Fewer SHORT trades → More selective
"""

import pandas as pd
import numpy as np
import pickle
import talib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters
WINDOW_SIZE = 1440
STEP_SIZE = 288  # 1 day (same as LONG-only for fair comparison)
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
LEVERAGE = 4
STOP_LOSS = 0.01
TAKE_PROFIT = 0.02
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002

LONG_ENTRY_THRESHOLD = 0.70
SHORT_ENTRY_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]  # Test higher thresholds

def calculate_rsi_macd_features(df):
    """Calculate RSI/MACD features"""
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_slope'] = df['rsi'].diff(3)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)

    price_change = df['close'].diff(5)
    rsi_change = df['rsi'].diff(5)
    df['rsi_divergence'] = (
        ((price_change > 0) & (rsi_change < 0)) |
        ((price_change < 0) & (rsi_change > 0))
    ).astype(float)

    macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_histogram_slope'] = pd.Series(macd_hist).diff(3)
    df['macd_crossover'] = ((macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))).astype(float)
    df['macd_crossunder'] = ((macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))).astype(float)

    return df

def calculate_enhanced_short_features(df):
    """Calculate other 22 SELL features"""
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20
    df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50
    df['price_acceleration'] = df['close'].diff().diff()

    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    volatility_median = df['volatility_20'].median()
    df['volatility_regime'] = (df['volatility_20'] > volatility_median).astype(float)

    df['higher_high'] = ((df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))).astype(float)
    df['lower_low'] = ((df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))).astype(float)

    resistance = df['high'].rolling(50).max()
    support = df['low'].rolling(50).min()
    df['near_resistance'] = (df['close'] > resistance * 0.98).astype(float)
    df['near_support'] = (df['close'] < support * 1.02).astype(float)

    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_high = bb_mid + 2 * bb_std
    bb_low = bb_mid - 2 * bb_std
    bb_range = bb_high - bb_low
    df['bb_position'] = np.where(bb_range != 0, (df['close'] - bb_low) / bb_range, 0.5)

    return df.ffill().fillna(0)

def classify_market_regime(df_window):
    """Classify market regime"""
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100
    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"

def backtest_with_threshold(window_df, long_model, long_scaler, short_model, short_scaler,
                            long_features, short_features, short_threshold):
    """Backtest with specific SHORT threshold"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(window_df)):
        current_price = window_df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            side = position['side']
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            exit_reason = None
            if pnl_pct <= -STOP_LOSS:
                exit_reason = "Stop Loss"
            elif pnl_pct >= TAKE_PROFIT:
                exit_reason = "Take Profit"
            elif hours_held >= MAX_HOLDING_HOURS:
                exit_reason = "Max Holding"

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
                    'probability': position['probability']
                })

                position = None

        # Look for entry
        if position is None and i < len(window_df) - 1:
            long_feat = window_df[long_features].iloc[i:i+1].values
            if not np.isnan(long_feat).any():
                long_feat_scaled = long_scaler.transform(long_feat)
                long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
            else:
                long_prob = 0.0

            short_feat = window_df[short_features].iloc[i:i+1].values
            if not np.isnan(short_feat).any():
                short_feat_scaled = short_scaler.transform(short_feat)
                short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
            else:
                short_prob = 0.0

            side = None
            signal_prob = 0.0

            if long_prob >= LONG_ENTRY_THRESHOLD:
                side = 'LONG'
                signal_prob = long_prob
            elif short_prob >= short_threshold:  # Use variable threshold
                side = 'SHORT'
                signal_prob = short_prob

            if side is not None:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price
                position = {
                    'side': side,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'probability': signal_prob
                }

    # Calculate metrics
    if len(trades) == 0:
        return {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'num_long': 0,
            'num_short': 0,
            'win_rate': 0.0,
            'win_rate_long': 0.0,
            'win_rate_short': 0.0,
            'avg_pnl_long': 0.0,
            'avg_pnl_short': 0.0
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100

    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']

    win_rate_long = (len([t for t in long_trades if t['pnl_usd'] > 0]) / len(long_trades) * 100) if long_trades else 0.0
    win_rate_short = (len([t for t in short_trades if t['pnl_usd'] > 0]) / len(short_trades) * 100) if short_trades else 0.0

    avg_pnl_long = np.mean([t['pnl_pct'] for t in long_trades]) * 100 if long_trades else 0.0
    avg_pnl_short = np.mean([t['pnl_pct'] for t in short_trades]) * 100 if short_trades else 0.0

    return {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'num_long': len(long_trades),
        'num_short': len(short_trades),
        'win_rate': win_rate,
        'win_rate_long': win_rate_long,
        'win_rate_short': win_rate_short,
        'avg_pnl_long': avg_pnl_long,
        'avg_pnl_short': avg_pnl_short
    }

print("="*80)
print("Higher Threshold Testing - Full Dataset (101 windows)")
print("="*80)
print(f"\n🎯 Goal: LONG+SHORT > LONG-only (+10.14%)")
print(f"   Testing thresholds: {SHORT_ENTRY_THRESHOLDS}")
print(f"   Same windows as LONG-only (101 windows, step=288)")
print()

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

short_model_path = MODELS_DIR / "xgboost_short_entry_enhanced_rsimacd_20251016_223048.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_entry_enhanced_rsimacd_20251016_223048_scaler.pkl"
with open(short_scaler_path, 'rb') as f:
    short_scaler = pickle.load(f)

short_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_rsimacd_20251016_223048_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print("  ✅ Models loaded")

# Load and prepare data
print("\nLoading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print("Calculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = calculate_rsi_macd_features(df)
df = calculate_enhanced_short_features(df)
df = df.ffill().dropna()
print(f"  ✅ Data ready: {len(df)} rows")

# Test each threshold
print(f"\n{'='*80}")
print("Testing Multiple Thresholds (Full Dataset)")
print(f"{'='*80}\n")

all_results = []

for threshold in SHORT_ENTRY_THRESHOLDS:
    print(f"Testing threshold {threshold:.2f}...")

    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        regime = classify_market_regime(window_df)

        metrics = backtest_with_threshold(
            window_df, long_model, long_scaler, short_model, short_scaler,
            long_feature_columns, short_feature_columns, threshold
        )

        bh_start = window_df['close'].iloc[0]
        bh_end = window_df['close'].iloc[-1]
        bh_return = ((bh_end - bh_start) / bh_start) * 100 - 2 * TRANSACTION_COST * 100

        windows.append({
            'start_idx': start_idx,
            'regime': regime,
            'return': metrics['total_return_pct'],
            'bh_return': bh_return,
            'num_trades': metrics['num_trades'],
            'num_long': metrics['num_long'],
            'num_short': metrics['num_short'],
            'win_rate': metrics['win_rate'],
            'long_win_rate': metrics['win_rate_long'],
            'short_win_rate': metrics['win_rate_short'],
            'avg_pnl_long': metrics['avg_pnl_long'],
            'avg_pnl_short': metrics['avg_pnl_short']
        })

        start_idx += STEP_SIZE

    results_df = pd.DataFrame(windows)

    # Summary
    summary = {
        'threshold': threshold,
        'avg_return': results_df['return'].mean(),
        'total_return': results_df['return'].sum(),
        'win_windows': (results_df['return'] > 0).sum() / len(results_df) * 100,
        'avg_trades': results_df['num_trades'].mean(),
        'avg_long': results_df['num_long'].mean(),
        'avg_short': results_df['num_short'].mean(),
        'overall_wr': results_df['win_rate'].mean(),
        'long_wr': results_df['long_win_rate'].mean(),
        'short_wr': results_df['short_win_rate'].mean(),
        'avg_pnl_long': results_df['avg_pnl_long'].mean(),
        'avg_pnl_short': results_df['avg_pnl_short'].mean(),
        'num_windows': len(results_df)
    }

    all_results.append(summary)

    status = "✅" if summary['avg_return'] > 10.14 else "❌"
    print(f"  {status} Avg Return: {summary['avg_return']:.2f}% | SHORT WR: {summary['short_wr']:.1f}% | "
          f"SHORT Trades: {summary['avg_short']:.1f} | Avg PnL SHORT: {summary['avg_pnl_short']:+.2f}%")

# Compare all thresholds
print(f"\n{'='*80}")
print("COMPARISON: All Thresholds vs LONG-only")
print(f"{'='*80}\n")

comparison_df = pd.DataFrame(all_results)
print(comparison_df.to_string(index=False))

# Find best threshold
best_return = comparison_df.loc[comparison_df['avg_return'].idxmax()]

print(f"\n{'='*80}")
print("OPTIMAL THRESHOLD vs LONG-ONLY")
print(f"{'='*80}")

print(f"\nBest LONG+SHORT Performance:")
print(f"  Threshold: {best_return['threshold']:.2f}")
print(f"  Avg Return: {best_return['avg_return']:.2f}% per window")
print(f"  SHORT WR: {best_return['short_wr']:.1f}%")
print(f"  LONG WR: {best_return['long_wr']:.1f}%")
print(f"  SHORT Trades: {best_return['avg_short']:.1f} per window")
print(f"  LONG Trades: {best_return['avg_long']:.1f} per window")
print(f"  Avg PnL LONG: {best_return['avg_pnl_long']:+.2f}%")
print(f"  Avg PnL SHORT: {best_return['avg_pnl_short']:+.2f}%")

print(f"\nLONG-only Baseline:")
print(f"  Avg Return: +10.14% per window")
print(f"  Trades: ~20.9 per window")
print(f"  Win Rate: 61.0%")

print(f"\n{'='*80}")
print("FINAL DECISION")
print(f"{'='*80}")

if best_return['avg_return'] > 10.14:
    improvement = ((best_return['avg_return'] - 10.14) / 10.14) * 100
    print(f"\n  ✅ SUCCESS! LONG+SHORT > LONG-only")
    print(f"     LONG+SHORT: {best_return['avg_return']:.2f}% per window")
    print(f"     LONG-only:  10.14% per window")
    print(f"     Improvement: +{improvement:.1f}%")
    print(f"\n  🎯 RECOMMENDATION: Deploy with threshold {best_return['threshold']:.2f}")
else:
    gap = 10.14 - best_return['avg_return']
    print(f"\n  ❌ FAILED: LONG+SHORT < LONG-only")
    print(f"     LONG+SHORT: {best_return['avg_return']:.2f}% per window")
    print(f"     LONG-only:  10.14% per window")
    print(f"     Gap: -{gap:.2f}% per window")
    print(f"\n  💡 Analysis:")
    print(f"     - SHORT Avg PnL: {best_return['avg_pnl_short']:+.2f}%")
    print(f"     - LONG Avg PnL: {best_return['avg_pnl_long']:+.2f}%")
    if best_return['avg_pnl_short'] < 0:
        print(f"     - SHORT is net negative → Consider disabling SHORT")
    else:
        print(f"     - SHORT reduces LONG opportunities → Position sizing issue")

# Save comparison
output_file = RESULTS_DIR / "threshold_comparison_full_dataset.csv"
comparison_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved: {output_file}")

print(f"\n{'='*80}")
print("Testing Complete!")
print(f"{'='*80}\n")
