"""
Calibrated Models - Threshold Grid Search (3-8 trades/day target)
==================================================================

Goal: Find threshold that achieves 3-8 trades/day with positive returns

Tested Thresholds: 0.16, 0.17, 0.18, 0.19, 0.20, 0.22, 0.25, 0.30
Period: 14-day holdout

Created: 2025-10-29
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Configuration
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
ML_EXIT_THRESHOLD = 0.75
HOLDOUT_DAYS = 14
INITIAL_CAPITAL = 10000

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Thresholds to test
THRESHOLDS = [0.16, 0.17, 0.18, 0.19, 0.20, 0.22, 0.25, 0.30]

print("="*80)
print("CALIBRATED MODELS - THRESHOLD GRID SEARCH")
print("="*80)
print()
print(f"Target: 3-8 trades/day")
print(f"Period: {HOLDOUT_DAYS} days")
print(f"Thresholds: {THRESHOLDS}")
print()

# Load data
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
holdout_candles = HOLDOUT_DAYS * 24 * 12
df = df.iloc[-holdout_candles:].copy().reset_index(drop=True)

# Prepare exit features
def prepare_exit_features(df):
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)
    if 'ma_50' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['ma_50']) / df['ma_50']
    else:
        ma_20 = df['close'].rolling(20).mean()
        df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20
    if 'ma_200' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['ma_200']) / df['ma_200']
    else:
        ma_50 = df['close'].rolling(50).mean()
        df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(100).median()).astype(float)
    df['rsi_slope'] = df['rsi'].diff(3) / 3
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    df['rsi_divergence'] = 0
    df['macd_histogram_slope'] = df['macd_hist'].diff(3) / 3 if 'macd_hist' in df.columns else 0
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    df['price_acceleration'] = df['close'].diff().diff()
    df['near_resistance'] = 0
    df['near_support'] = 0
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    else:
        df['bb_position'] = 0.5
    df = df.ffill().bfill()
    return df

df = prepare_exit_features(df)

# Load models
long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_calibrated_20251029_183259.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_calibrated_20251029_183259_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_calibrated_20251029_183259_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f if line.strip()]

short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_entry_calibrated_20251029_183259.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_calibrated_20251029_183259_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_calibrated_20251029_183259_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f if line.strip()]

long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f if line.strip()]

short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f if line.strip()]

print(f"✅ Models loaded")
print()

# Run grid search
results = []

for ENTRY_THRESHOLD in THRESHOLDS:
    print(f"Testing threshold {ENTRY_THRESHOLD:.2f}...")

    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(100, len(df) - EMERGENCY_MAX_HOLD):
        row = df.iloc[i]
        timestamp = row['timestamp']
        price = row['close']

        # Exit logic
        if position is not None:
            side = position['side']
            entry_price = position['entry_price']
            position_size_pct = position['position_size_pct']

            if side == 'LONG':
                price_change_pct = (price - entry_price) / entry_price
            else:
                price_change_pct = (entry_price - price) / entry_price

            leveraged_pnl_pct = price_change_pct * LEVERAGE

            should_exit = False
            exit_reason = None

            if leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = "STOP_LOSS"
            elif i - position['entry_index'] >= EMERGENCY_MAX_HOLD:
                should_exit = True
                exit_reason = "MAX_HOLD"
            else:
                if side == 'LONG':
                    X_exit = row[long_exit_features].values.reshape(1, -1)
                    X_exit_scaled = long_exit_scaler.transform(X_exit)
                    exit_prob = long_exit_model.predict_proba(X_exit_scaled)[0][1]
                    if exit_prob >= ML_EXIT_THRESHOLD:
                        should_exit = True
                        exit_reason = "ML_EXIT"
                else:
                    X_exit = row[short_exit_features].values.reshape(1, -1)
                    X_exit_scaled = short_exit_scaler.transform(X_exit)
                    exit_prob = short_exit_model.predict_proba(X_exit_scaled)[0][1]
                    if exit_prob >= ML_EXIT_THRESHOLD:
                        should_exit = True
                        exit_reason = "ML_EXIT"

            if should_exit:
                pnl_usd = capital * position_size_pct * leveraged_pnl_pct
                capital += pnl_usd
                hold_time = i - position['entry_index']

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'position_size_pct': position_size_pct,
                    'pnl_pct': leveraged_pnl_pct * 100,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'hold_time': hold_time
                })
                position = None

        # Entry logic
        if position is None:
            X_long = row[long_entry_features].values.reshape(1, -1)
            X_long_scaled = long_entry_scaler.transform(X_long)
            long_prob = long_entry_model.predict_proba(X_long_scaled)[0][1]

            X_short = row[short_entry_features].values.reshape(1, -1)
            X_short_scaled = short_entry_scaler.transform(X_short)
            short_prob = short_entry_model.predict_proba(X_short_scaled)[0][1]

            if long_prob >= ENTRY_THRESHOLD:
                side = 'LONG'
                entry_prob = long_prob
            elif short_prob >= ENTRY_THRESHOLD:
                side = 'SHORT'
                entry_prob = short_prob
            else:
                continue

            if entry_prob < 0.65:
                position_size_pct = 0.20
            elif entry_prob >= 0.85:
                position_size_pct = 0.95
            else:
                position_size_pct = 0.20 + (0.95 - 0.20) * ((entry_prob - 0.65) / (0.85 - 0.65))

            position = {
                'side': side,
                'entry_price': price,
                'entry_time': timestamp,
                'entry_index': i,
                'entry_prob': entry_prob,
                'position_size_pct': position_size_pct
            }

    # Calculate results
    total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    trade_freq = len(trades) / HOLDOUT_DAYS if trades else 0

    result = {
        'threshold': ENTRY_THRESHOLD,
        'trades': len(trades),
        'frequency': trade_freq,
        'return': total_return,
        'final_capital': capital
    }

    if trades:
        df_trades = pd.DataFrame(trades)
        wins = df_trades[df_trades['pnl_usd'] > 0]
        result['win_rate'] = len(wins) / len(df_trades) * 100
        result['long_pct'] = len(df_trades[df_trades['side'] == 'LONG']) / len(df_trades) * 100
        result['short_pct'] = len(df_trades[df_trades['side'] == 'SHORT']) / len(df_trades) * 100
    else:
        result['win_rate'] = 0
        result['long_pct'] = 0
        result['short_pct'] = 0

    results.append(result)

    # Print result
    in_target = "✅" if 3 <= trade_freq <= 8 else "❌"
    print(f"  Trades: {len(trades)} ({trade_freq:.2f}/day) {in_target}")
    print(f"  Return: {total_return:+.2f}%")
    if trades:
        print(f"  Win Rate: {result['win_rate']:.1f}%")
    print()

# Summary
print("="*80)
print("GRID SEARCH RESULTS")
print("="*80)
print()

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print()

# Find best threshold in target range
target_results = df_results[(df_results['frequency'] >= 3) & (df_results['frequency'] <= 8)]

if len(target_results) > 0:
    print("="*80)
    print("THRESHOLDS IN TARGET RANGE (3-8 trades/day)")
    print("="*80)
    print()
    print(target_results.to_string(index=False))
    print()

    # Best by return
    best = target_results.loc[target_results['return'].idxmax()]
    print(f"BEST THRESHOLD: {best['threshold']:.2f}")
    print(f"  Frequency: {best['frequency']:.2f}/day ✅")
    print(f"  Return: {best['return']:+.2f}%")
    print(f"  Win Rate: {best['win_rate']:.1f}%")
    print(f"  LONG/SHORT: {best['long_pct']:.1f}%/{best['short_pct']:.1f}%")
else:
    print("⚠️  No threshold found in target range (3-8 trades/day)")

# Save results
output_file = RESULTS_DIR / f"calibrated_threshold_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df_results.to_csv(output_file, index=False)
print()
print(f"✅ Results saved: {output_file.name}")
print()
