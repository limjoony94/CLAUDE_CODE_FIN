"""
LONG Priority Strategy + Asymmetric Thresholds

전략: LONG 우선, SHORT는 극히 선별적
목표: LONG+SHORT > LONG-only (+10.14%)
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
STEP_SIZE = 288
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
LEVERAGE = 4
STOP_LOSS = 0.01
TAKE_PROFIT = 0.02
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002

def calculate_short_features(df):
    """Calculate all SHORT-specific features"""
    # Symmetric features
    df['rsi_raw'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_deviation'] = np.abs(df['rsi_raw'] - 50)
    df['rsi_direction'] = np.sign(df['rsi_raw'] - 50)
    df['rsi_extreme'] = ((df['rsi_raw'] > 70) | (df['rsi_raw'] < 30)).astype(float)

    macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd_strength'] = np.abs(macd_hist)
    df['macd_direction'] = np.sign(macd_hist)
    df['macd_divergence_abs'] = np.abs(macd - macd_signal)

    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    df['price_distance_ma20'] = np.abs(df['close'] - ma_20) / ma_20
    df['price_direction_ma20'] = np.sign(df['close'] - ma_20)
    df['price_distance_ma50'] = np.abs(df['close'] - ma_50) / ma_50
    df['price_direction_ma50'] = np.sign(df['close'] - ma_50)

    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['atr_pct'] = df['atr'] / df['close']

    # Inverse features
    df['negative_momentum'] = -df['close'].pct_change(5).clip(upper=0)
    df['negative_acceleration'] = -df['close'].diff().diff().clip(upper=0)

    df['down_candle'] = (df['close'] < df['open']).astype(float)
    df['down_candle_ratio'] = df['down_candle'].rolling(10).mean()
    df['down_candle_body'] = (df['open'] - df['close']).clip(lower=0)

    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    df['lower_low_streak'] = df['lower_low'].rolling(5).sum()

    resistance = df['high'].rolling(20).max()
    df['near_resistance'] = (df['high'] > resistance.shift(1) * 0.99).astype(float)
    df['rejection_from_resistance'] = ((df['near_resistance'] == 1) & (df['close'] < df['open'])).astype(float)
    df['resistance_rejection_count'] = df['rejection_from_resistance'].rolling(10).sum()

    price_change_5 = df['close'].diff(5)
    rsi_change_5 = df['rsi_raw'].diff(5) if 'rsi_raw' in df.columns else 0
    df['bearish_divergence'] = ((price_change_5 > 0) & (rsi_change_5 < 0)).astype(float)

    df['price_up'] = (df['close'] > df['close'].shift(1)).astype(float)
    df['volume_on_decline'] = df['volume'] * (1 - df['price_up'])
    df['volume_on_advance'] = df['volume'] * df['price_up']
    df['volume_decline_ratio'] = df['volume_on_decline'].rolling(10).sum() / (df['volume_on_advance'].rolling(10).sum() + 1e-10)

    price_range_20 = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
    volume_surge = df['volume'] / df['volume'].rolling(20).mean()
    df['distribution_signal'] = ((price_range_20 < 0.05) & (volume_surge > 1.5)).astype(float)

    # Opportunity cost features
    returns_20 = df['close'].pct_change(20)
    df['bear_market_strength'] = (-returns_20).clip(lower=0)

    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['trend_strength'] = (df['ema_12'] - df['ema_26']) / df['close']
    df['downtrend_confirmed'] = (df['trend_strength'] < -0.01).astype(float)

    returns = df['close'].pct_change()
    upside_vol = returns[returns > 0].rolling(20).std()
    downside_vol = returns[returns < 0].rolling(20).std()
    df['downside_volatility'] = downside_vol.ffill()
    df['upside_volatility'] = upside_vol.ffill()
    df['volatility_asymmetry'] = df['downside_volatility'] / (df['upside_volatility'] + 1e-10)

    support = df['low'].rolling(50).min()
    df['below_support'] = (df['close'] < support.shift(1) * 1.01).astype(float)
    df['support_breakdown'] = ((df['close'].shift(1) >= support.shift(1)) & (df['close'] < support.shift(1))).astype(float)

    df['panic_selling'] = ((df['down_candle'] == 1) & (df['volume'] > df['volume'].rolling(10).mean() * 1.5)).astype(float)

    return df

def backtest_with_priority(window_df, long_model, long_scaler, short_model, short_scaler,
                           long_features, short_features, long_threshold, short_threshold):
    """
    Backtest with LONG priority strategy

    Rules:
    1. If LONG signal >= long_threshold → LONG
    2. Else if SHORT signal >= short_threshold → SHORT
    3. Else → No trade
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(window_df)):
        current_price = window_df['close'].iloc[i]

        # Manage position
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
                exit_reason = "SL"
            elif pnl_pct >= TAKE_PROFIT:
                exit_reason = "TP"
            elif hours_held >= MAX_HOLDING_HOURS:
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
                    'pnl_pct': pnl_pct
                })

                position = None

        # Entry logic with LONG priority
        if position is None and i < len(window_df) - 1:
            # Get LONG signal first
            long_feat = window_df[long_features].iloc[i:i+1].values
            if not np.isnan(long_feat).any():
                long_feat_scaled = long_scaler.transform(long_feat)
                long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
            else:
                long_prob = 0.0

            # Rule 1: LONG priority
            if long_prob >= long_threshold:
                side = 'LONG'
            else:
                # Only check SHORT if LONG is not good enough
                short_feat = window_df[short_features].iloc[i:i+1].values
                if not np.isnan(short_feat).any():
                    short_feat_scaled = short_scaler.transform(short_feat)
                    short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
                else:
                    short_prob = 0.0

                # Rule 2: SHORT only if signal is very strong
                if short_prob >= short_threshold:
                    side = 'SHORT'
                else:
                    side = None

            if side is not None:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price
                position = {
                    'side': side,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity
                }

    # Calculate metrics
    if len(trades) == 0:
        return {'total_return_pct': 0.0, 'num_trades': 0, 'num_long': 0, 'num_short': 0}

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']

    return {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'num_long': len(long_trades),
        'num_short': len(short_trades),
        'long_wr': (len([t for t in long_trades if t['pnl_usd'] > 0]) / len(long_trades) * 100) if long_trades else 0.0,
        'short_wr': (len([t for t in short_trades if t['pnl_usd'] > 0]) / len(short_trades) * 100) if short_trades else 0.0
    }

print("="*80)
print("LONG Priority Strategy + Asymmetric Thresholds")
print("="*80)
print("\n전략: LONG 우선 + SHORT 극히 선별적")
print("목표: LONG+SHORT > LONG-only (+10.14%)\n")

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

print(f"  ✅ Models loaded")

# Load and prepare data
print("\nLoading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)

print("Calculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = calculate_short_features(df)
df = df.ffill().bfill().fillna(0)
print(f"  ✅ Data ready: {len(df)} rows")

# Test different threshold combinations
print(f"\n{'='*80}")
print("Testing Threshold Combinations")
print(f"{'='*80}\n")

# Strategy: LONG priority
threshold_combinations = [
    {'long': 0.70, 'short': 0.70, 'desc': 'Baseline (Equal thresholds)'},
    {'long': 0.65, 'short': 0.75, 'desc': 'LONG easier, SHORT harder'},
    {'long': 0.65, 'short': 0.80, 'desc': 'LONG easier, SHORT much harder'},
    {'long': 0.65, 'short': 0.85, 'desc': 'LONG easier, SHORT ultra-selective'},
    {'long': 0.60, 'short': 0.85, 'desc': 'LONG very easy, SHORT ultra-selective'},
]

all_results = []

for config in threshold_combinations:
    long_thresh = config['long']
    short_thresh = config['short']
    desc = config['desc']

    print(f"Testing: {desc} (LONG {long_thresh:.2f}, SHORT {short_thresh:.2f})...")

    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        metrics = backtest_with_priority(
            window_df, long_model, long_scaler, short_model, short_scaler,
            long_feature_columns, short_feature_columns, long_thresh, short_thresh
        )

        windows.append(metrics)
        start_idx += STEP_SIZE

    results_df = pd.DataFrame(windows)

    summary = {
        'long_threshold': long_thresh,
        'short_threshold': short_thresh,
        'description': desc,
        'avg_return': results_df['total_return_pct'].mean(),
        'win_windows': (results_df['total_return_pct'] > 0).sum() / len(results_df) * 100,
        'avg_trades': results_df['num_trades'].mean(),
        'avg_long': results_df['num_long'].mean(),
        'avg_short': results_df['num_short'].mean(),
        'long_wr': results_df['long_wr'].mean(),
        'short_wr': results_df['short_wr'].mean()
    }

    all_results.append(summary)

    status = "✅" if summary['avg_return'] > 10.14 else "❌"
    print(f"  {status} Avg: {summary['avg_return']:.2f}% | LONG: {summary['avg_long']:.1f}, SHORT: {summary['avg_short']:.1f}")

# Results
print(f"\n{'='*80}")
print("RESULTS vs LONG-ONLY")
print(f"{'='*80}\n")

comparison_df = pd.DataFrame(all_results)
print(comparison_df[['description', 'avg_return', 'avg_long', 'avg_short', 'long_wr', 'short_wr']].to_string(index=False))

best = comparison_df.loc[comparison_df['avg_return'].idxmax()]

print(f"\n{'='*80}")
print("FINAL VERDICT")
print(f"{'='*80}")

print(f"\nBest LONG+SHORT (Priority Strategy):")
print(f"  Configuration: {best['description']}")
print(f"  Avg Return: {best['avg_return']:.2f}% per window")
print(f"  LONG: {best['avg_long']:.1f} trades/window, {best['long_wr']:.1f}% WR")
print(f"  SHORT: {best['avg_short']:.1f} trades/window, {best['short_wr']:.1f}% WR")

print(f"\nLONG-only Baseline:")
print(f"  Avg Return: +10.14% per window")

if best['avg_return'] > 10.14:
    improvement = ((best['avg_return'] - 10.14) / 10.14) * 100
    print(f"\n  ✅ SUCCESS! LONG+SHORT > LONG-only")
    print(f"     Improvement: +{improvement:.1f}%")
    print(f"     Priority strategy worked!")
else:
    gap = 10.14 - best['avg_return']
    print(f"\n  ❌ Still below LONG-only")
    print(f"     Gap: -{gap:.2f}% per window")

# Save
output_file = RESULTS_DIR / "long_priority_strategy_results.csv"
comparison_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved: {output_file}")

print(f"\n{'='*80}")
print("Testing Complete!")
print(f"{'='*80}\n")
