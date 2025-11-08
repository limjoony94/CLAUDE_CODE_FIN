"""
Market Regime Filter Strategy

전략: LONG-only를 기본으로, 특정 시장 조건에서만 SHORT 허용
목표: LONG+SHORT > LONG-only 달성
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "historical"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("Market Regime Filter Strategy Test")
print("="*80)
print("\n전략: LONG-only 기본 + 조건부 SHORT")
print("목표: LONG+SHORT > LONG-only (+10.14%)")
print()

# Load models
long_model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
long_scaler_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"

short_model_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl"
short_scaler_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl"
short_features_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"

with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)
with open(long_scaler_path, 'rb') as f:
    long_scaler = pickle.load(f)
with open(long_features_path, 'r') as f:
    long_features = [line.strip() for line in f.readlines()]

with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)
with open(short_scaler_path, 'rb') as f:
    short_scaler = pickle.load(f)
with open(short_features_path, 'r') as f:
    short_features = [line.strip() for line in f.readlines()]

print(f"✅ Models loaded")

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)

# Calculate features
print("\nCalculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)

# Calculate SHORT-specific features using same functions as test_redesigned_model_full.py
import talib

def calculate_symmetric_features(df):
    """Symmetric features - same as training"""
    df['rsi_raw'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_deviation'] = np.abs(df['rsi_raw'] - 50)
    df['rsi_direction'] = np.sign(df['rsi_raw'] - 50)
    df['rsi_extreme'] = ((df['rsi_raw'] > 70) | (df['rsi_raw'] < 30)).astype(float)

    macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd_strength'] = np.abs(macd_hist)
    df['macd_direction'] = np.sign(macd_hist)
    df['macd_divergence_abs'] = np.abs(macd - macd_signal)  # MISSING FEATURE

    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    df['price_distance_ma20'] = np.abs(df['close'] - ma_20) / ma_20
    df['price_direction_ma20'] = np.sign(df['close'] - ma_20)
    df['price_distance_ma50'] = np.abs(df['close'] - ma_50) / ma_50
    df['price_direction_ma50'] = np.sign(df['close'] - ma_50)

    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)  # MISSING FEATURE
    df['atr_pct'] = df['atr'] / df['close']  # MISSING FEATURE

    return df

def calculate_inverse_features(df):
    """Inverse features - same as training"""
    df['negative_momentum'] = -df['close'].pct_change(5).clip(upper=0)
    df['negative_acceleration'] = -df['close'].diff().diff().clip(upper=0)

    df['down_candle'] = (df['close'] < df['open']).astype(float)
    df['down_candle_ratio'] = df['down_candle'].rolling(10).mean()
    df['down_candle_body'] = (df['open'] - df['close']).clip(lower=0)

    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    df['lower_low_streak'] = df['lower_low'].rolling(5).sum()

    resistance = df['high'].rolling(20).max()
    df['near_resistance'] = (df['high'] > resistance.shift(1) * 0.99).astype(float)
    df['rejection_from_resistance'] = (
        (df['near_resistance'] == 1) & (df['close'] < df['open'])
    ).astype(float)
    df['resistance_rejection_count'] = df['rejection_from_resistance'].rolling(10).sum()

    price_change_5 = df['close'].diff(5)
    rsi_change_5 = df['rsi_raw'].diff(5) if 'rsi_raw' in df.columns else 0
    df['bearish_divergence'] = (  # MISSING FEATURE
        (price_change_5 > 0) & (rsi_change_5 < 0)
    ).astype(float)

    df['price_up'] = (df['close'] > df['close'].shift(1)).astype(float)
    df['volume_on_decline'] = df['volume'] * (1 - df['price_up'])
    df['volume_on_advance'] = df['volume'] * df['price_up']
    df['volume_decline_ratio'] = (
        df['volume_on_decline'].rolling(10).sum() /
        (df['volume_on_advance'].rolling(10).sum() + 1e-10)
    )

    price_range_20 = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
    volume_surge = df['volume'] / df['volume'].rolling(20).mean()
    df['distribution_signal'] = (
        (price_range_20 < 0.05) &
        (volume_surge > 1.5)
    ).astype(float)

    return df

def calculate_opportunity_cost_features(df):
    """Opportunity cost features - same as training"""
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
    df['support_breakdown'] = (
        (df['close'].shift(1) >= support.shift(1)) &
        (df['close'] < support.shift(1))
    ).astype(float)

    df['panic_selling'] = (
        (df['down_candle'] == 1) &
        (df['volume'] > df['volume'].rolling(10).mean() * 1.5)
    ).astype(float)

    return df

# Apply all feature calculations
df = calculate_symmetric_features(df)
df = calculate_inverse_features(df)
df = calculate_opportunity_cost_features(df)
df = df.ffill().bfill().fillna(0)
print(f"✅ Features calculated: {len(df)} rows")

# Backtest parameters
WINDOW_SIZE = 1440
STEP_SIZE = 288
INITIAL_CAPITAL = 10000.0
STOP_LOSS = 0.01
TAKE_PROFIT = 0.03
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002
LONG_THRESHOLD = 0.70
SHORT_THRESHOLD = 0.70
LEVERAGE = 4

def classify_regime(df, idx, lookback=20):
    """Classify market regime"""
    if idx < lookback:
        return None

    recent_data = df.iloc[idx-lookback:idx]

    start_price = recent_data['close'].iloc[0]
    end_price = recent_data['close'].iloc[-1]
    price_change_pct = ((end_price / start_price) - 1) * 100

    returns = recent_data['close'].pct_change()
    volatility = returns.std()

    # Regime classification
    if price_change_pct > 3.0:
        regime = "Strong Bull"
    elif price_change_pct > 1.0:
        regime = "Mild Bull"
    elif price_change_pct < -2.0:
        regime = "Strong Bear"
    elif price_change_pct < -0.5:
        regime = "Mild Bear"
    else:
        regime = "Sideways"

    # Volatility
    avg_vol = 0.02
    if volatility > avg_vol * 1.5:
        vol_tag = "High Vol"
    elif volatility < avg_vol * 0.7:
        vol_tag = "Low Vol"
    else:
        vol_tag = "Normal Vol"

    return {
        'regime': regime,
        'volatility_tag': vol_tag,
        'price_change_pct': price_change_pct,
        'volatility': volatility
    }

def is_short_allowed(regime_info):
    """
    Determine if SHORT is allowed based on market regime

    SHORT allowed ONLY when:
    1. Market is bearish (Strong Bear or Mild Bear)
    2. Volatility is not too low (Normal or High)
    """
    if regime_info is None:
        return False

    regime = regime_info['regime']
    vol_tag = regime_info['volatility_tag']

    # Condition 1: Bearish market
    is_bearish = regime in ['Strong Bear', 'Mild Bear']

    # Condition 2: Not low volatility (low vol = choppy consolidation)
    not_low_vol = vol_tag in ['Normal Vol', 'High Vol']

    return is_bearish and not_low_vol

def backtest_with_regime_filter(window_df, leverage, filter_config):
    """
    Backtest with regime filter

    filter_config:
      - 'none': No filter (LONG+SHORT default)
      - 'regime': Regime-based filter
      - 'strict': Stricter regime filter (Strong Bear only)
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    sizer = DynamicPositionSizer(
        base_position_pct=0.50,
        max_position_pct=0.95,
        min_position_pct=0.20,
        signal_weight=0.4,
        volatility_weight=0.3,
        regime_weight=0.2,
        streak_weight=0.1
    )

    for i in range(len(window_df)):
        current_price = window_df['close'].iloc[i]

        # Exit logic
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            if position['side'] == 'LONG':
                price_change_pct = (current_price - entry_price) / entry_price
            else:
                price_change_pct = (entry_price - current_price) / entry_price

            leveraged_pnl_pct = price_change_pct * leverage
            leveraged_pnl_usd = leveraged_pnl_pct * position['base_value']

            exit_reason = None
            if leveraged_pnl_pct <= -STOP_LOSS:
                exit_reason = "SL"
            elif leveraged_pnl_pct >= TAKE_PROFIT:
                exit_reason = "TP"
            elif hours_held >= MAX_HOLDING_HOURS:
                exit_reason = "MH"

            if exit_reason:
                entry_cost = position['leveraged_value'] * TRANSACTION_COST
                exit_cost = (current_price / entry_price) * position['leveraged_value'] * TRANSACTION_COST
                total_cost = entry_cost + exit_cost
                net_pnl_usd = leveraged_pnl_usd - total_cost

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'side': position['side'],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_usd_net': net_pnl_usd,
                    'pnl_pct': leveraged_pnl_pct,
                    'exit_reason': exit_reason,
                    'probability': position['probability'],
                    'regime': position['regime']
                })

                capital += net_pnl_usd
                position = None

        # Entry logic
        if position is None and i < len(window_df) - 1:
            if capital <= 0:
                break

            # Get signals
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

            long_signal = long_prob >= LONG_THRESHOLD
            short_signal = short_prob >= SHORT_THRESHOLD

            # Classify regime
            regime_info = classify_regime(window_df, i, 20)

            if regime_info is None:
                continue

            # Apply filter
            if filter_config == 'none':
                # No filter - default LONG+SHORT
                short_allowed = True
            elif filter_config == 'regime':
                # Regime-based filter
                short_allowed = is_short_allowed(regime_info)
            elif filter_config == 'strict':
                # Strict filter - Strong Bear only
                short_allowed = (regime_info['regime'] == 'Strong Bear')
            else:
                short_allowed = False

            # Choose side
            chosen_side = None
            chosen_prob = 0.0

            if long_signal and short_signal:
                if short_allowed:
                    # Both signals and SHORT allowed - choose stronger
                    if long_prob >= short_prob:
                        chosen_side = 'LONG'
                        chosen_prob = long_prob
                    else:
                        chosen_side = 'SHORT'
                        chosen_prob = short_prob
                else:
                    # SHORT not allowed - LONG only
                    chosen_side = 'LONG'
                    chosen_prob = long_prob
            elif long_signal:
                chosen_side = 'LONG'
                chosen_prob = long_prob
            elif short_signal and short_allowed:
                chosen_side = 'SHORT'
                chosen_prob = short_prob

            # Enter position
            if chosen_side is not None:
                current_volatility = window_df['atr_pct'].iloc[i] if 'atr_pct' in window_df.columns else 0.01
                avg_volatility = window_df['atr_pct'].iloc[max(0, i-50):i].mean() if 'atr_pct' in window_df.columns else 0.01

                sizing_result = sizer.calculate_position_size(
                    capital=capital,
                    signal_strength=chosen_prob,
                    current_volatility=current_volatility,
                    avg_volatility=avg_volatility,
                    market_regime=regime_info['regime'],
                    recent_trades=trades[-10:] if len(trades) > 0 else [],
                    leverage=leverage
                )

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'side': chosen_side,
                    'base_value': sizing_result['position_value'],
                    'leveraged_value': sizing_result['leveraged_value'],
                    'position_size_pct': sizing_result['position_size_pct'],
                    'probability': chosen_prob,
                    'regime': regime_info['regime']
                }

    return trades, capital

# Test different filter configurations
print(f"\n{'='*80}")
print("Testing Filter Configurations (101 windows)")
print(f"{'='*80}\n")

filter_configs = {
    'none': "No filter (default LONG+SHORT)",
    'regime': "Regime filter (Bear markets only)",
    'strict': "Strict filter (Strong Bear only)"
}

results = []

for config_name, config_desc in filter_configs.items():
    print(f"Testing: {config_desc}...")

    all_trades = []
    all_capital = []

    for w in range(101):
        start_idx = w * STEP_SIZE
        end_idx = start_idx + WINDOW_SIZE

        if end_idx > len(df):
            break

        window_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        trades, final_capital = backtest_with_regime_filter(window_df, LEVERAGE, config_name)

        all_trades.extend(trades)

        window_return = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        all_capital.append(window_return)

    # Calculate metrics
    df_trades = pd.DataFrame(all_trades)

    if len(df_trades) > 0:
        avg_return = np.mean(all_capital)
        win_windows = (np.array(all_capital) > 0).sum() / len(all_capital) * 100

        num_long = len(df_trades[df_trades['side'] == 'LONG'])
        num_short = len(df_trades[df_trades['side'] == 'SHORT'])
        total_trades = len(df_trades) / 101

        long_trades = df_trades[df_trades['side'] == 'LONG']
        short_trades = df_trades[df_trades['side'] == 'SHORT']

        long_wr = (long_trades['pnl_usd_net'] > 0).mean() * 100 if len(long_trades) > 0 else 0
        short_wr = (short_trades['pnl_usd_net'] > 0).mean() * 100 if len(short_trades) > 0 else 0

        overall_wr = (df_trades['pnl_usd_net'] > 0).mean() * 100

        results.append({
            'config': config_name,
            'description': config_desc,
            'avg_return': avg_return,
            'win_windows': win_windows,
            'total_trades': total_trades,
            'long_trades': num_long / 101,
            'short_trades': num_short / 101,
            'long_wr': long_wr,
            'short_wr': short_wr,
            'overall_wr': overall_wr
        })

        print(f"  ✅ {config_name}: {avg_return:+.2f}% per window, {total_trades:.1f} trades/window")
    else:
        print(f"  ❌ {config_name}: No trades")

# Results comparison
print(f"\n{'='*80}")
print("Results Comparison")
print(f"{'='*80}\n")

df_results = pd.DataFrame(results)

print(f"{'Config':<15} {'Avg Return':>12} {'LONG':>8} {'SHORT':>8} {'LONG WR':>10} {'SHORT WR':>10} {'Overall WR':>12}")
print(f"{'-'*15} {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")

for _, row in df_results.iterrows():
    print(f"{row['config']:<15} {row['avg_return']:>11.2f}% {row['long_trades']:>7.1f} {row['short_trades']:>7.1f} {row['long_wr']:>9.1f}% {row['short_wr']:>9.1f}% {row['overall_wr']:>11.1f}%")

print(f"\n{'='*80}")
print("Comparison with Baselines")
print(f"{'='*80}\n")

print("Baselines:")
print(f"  LONG-only: +10.14% per window (target to beat)")
print(f"  LONG+SHORT (no filter): +4.55% per window (previous best)")
print()

best_config = df_results.loc[df_results['avg_return'].idxmax()]

print(f"Best Configuration: {best_config['description']}")
print(f"  Avg Return: {best_config['avg_return']:+.2f}% per window")
print(f"  vs LONG-only: {best_config['avg_return'] - 10.14:+.2f}% ({(best_config['avg_return'] / 10.14 - 1) * 100:+.1f}%)")

if best_config['avg_return'] > 10.14:
    print(f"\n  ✅ SUCCESS! LONG+SHORT > LONG-only")
    print(f"     Goal achieved with regime filter!")
else:
    gap = 10.14 - best_config['avg_return']
    print(f"\n  ❌ Still below LONG-only")
    print(f"     Gap: {gap:.2f}% ({gap / 10.14 * 100:.1f}%)")

# Save
output_file = RESULTS_DIR / "regime_filter_strategy_results.csv"
df_results.to_csv(output_file, index=False)
print(f"\n✅ Saved: {output_file.name}")

print(f"\n{'='*80}")
print("Analysis Complete!")
print(f"{'='*80}")
