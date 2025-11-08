"""
Market Regime별 LONG vs SHORT 성능 분석

목표: 어떤 시장 환경에서 LONG/SHORT가 더 유리한지 발견
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
print("Market Regime 분석: LONG vs SHORT 성능")
print("="*80)

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

# Calculate all features
print("\nCalculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)

# Calculate SHORT-specific features
import talib

# Symmetric features
df['rsi_raw'] = talib.RSI(df['close'], timeperiod=14)
df['rsi_deviation'] = np.abs(df['rsi_raw'] - 50)
df['rsi_direction'] = np.sign(df['rsi_raw'] - 50)
df['rsi_extreme'] = ((df['rsi_raw'] > 70) | (df['rsi_raw'] < 30)).astype(float)

macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['macd_strength'] = np.abs(macd_hist)
df['macd_direction'] = np.sign(macd_hist)

ma_20 = df['close'].rolling(20).mean()
df['price_distance_ma20'] = np.abs(df['close'] - ma_20) / ma_20
df['price_direction_ma20'] = np.sign(df['close'] - ma_20)

ma_50 = df['close'].rolling(50).mean()
df['price_distance_ma50'] = np.abs(df['close'] - ma_50) / ma_50
df['price_direction_ma50'] = np.sign(df['close'] - ma_50)

volume_ma = df['volume'].rolling(20).mean()
df['volume_deviation'] = np.abs(df['volume'] - volume_ma) / volume_ma
df['volume_direction'] = np.sign(df['volume'] - volume_ma)

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

df['price_up'] = (df['close'] > df['close'].shift(1)).astype(float)
df['volume_on_decline'] = df['volume'] * (1 - df['price_up'])
df['volume_on_advance'] = df['volume'] * df['price_up']
df['volume_decline_ratio'] = df['volume_on_decline'].rolling(10).sum() / (df['volume_on_advance'].rolling(10).sum() + 1e-10)

df['sell_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
df['buy_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
df['pressure_imbalance'] = df['sell_pressure'] - df['buy_pressure']

df['distribution_signal'] = ((df['high'] == df['high'].rolling(10).max()) & (df['close'] < df['open'])).astype(float)

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

df['panic_selling'] = ((df['volume'] > volume_ma * 1.5) & (df['close'] < df['open']) & (df['close'] < df['close'].shift(1))).astype(float)

# Fill NaN (don't drop - will handle in backtest loop)
df = df.ffill()

print(f"✅ Features calculated: {len(df)} rows")

# Analyze market regimes
print(f"\n{'='*80}")
print("Market Regime Classification")
print(f"{'='*80}\n")

# Calculate market regime indicators
lookback_regime = 20  # 20 candles = ~1.7 hours

def classify_regime(df, idx, lookback=20):
    """Classify market regime at given index"""
    if idx < lookback:
        return None

    recent_data = df.iloc[idx-lookback:idx]

    # Price change
    start_price = recent_data['close'].iloc[0]
    end_price = recent_data['close'].iloc[-1]
    price_change_pct = ((end_price / start_price) - 1) * 100

    # Volatility
    returns = recent_data['close'].pct_change()
    volatility = returns.std()

    # Trend strength (EMA crossover)
    ema_12 = recent_data['close'].ewm(span=12).mean().iloc[-1]
    ema_26 = recent_data['close'].ewm(span=26).mean().iloc[-1]
    trend = (ema_12 - ema_26) / end_price

    # Classify
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

    # Add volatility tag
    avg_vol = 0.02  # approximate average
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
        'volatility': volatility,
        'trend_strength': trend
    }

# Simulate trades with regime tracking
print("Simulating trades with regime classification...")

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

def backtest_window_with_regime(window_df, leverage):
    """Backtest with regime tracking"""
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

                # Classify exit regime
                exit_regime = classify_regime(window_df, i, lookback_regime)

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'side': position['side'],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'entry_regime': position['entry_regime'],
                    'entry_regime_vol': position['entry_regime_vol'],
                    'entry_price_change': position['entry_price_change'],
                    'exit_regime': exit_regime['regime'] if exit_regime else 'Unknown',
                    'exit_regime_vol': exit_regime['volatility_tag'] if exit_regime else 'Unknown',
                    'pnl_usd_net': net_pnl_usd,
                    'pnl_pct': leveraged_pnl_pct,
                    'exit_reason': exit_reason,
                    'probability': position['probability']
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

            # Classify entry regime
            entry_regime_info = classify_regime(window_df, i, lookback_regime)

            if entry_regime_info is None:
                continue

            chosen_side = None
            chosen_prob = 0.0

            if long_signal and short_signal:
                if long_prob >= short_prob:
                    chosen_side = 'LONG'
                    chosen_prob = long_prob
                else:
                    chosen_side = 'SHORT'
                    chosen_prob = short_prob
            elif long_signal:
                chosen_side = 'LONG'
                chosen_prob = long_prob
            elif short_signal:
                chosen_side = 'SHORT'
                chosen_prob = short_prob

            if chosen_side is not None:
                current_volatility = window_df['atr_pct'].iloc[i] if 'atr_pct' in window_df.columns else 0.01
                avg_volatility = window_df['atr_pct'].iloc[max(0, i-50):i].mean() if 'atr_pct' in window_df.columns else 0.01

                sizing_result = sizer.calculate_position_size(
                    capital=capital,
                    signal_strength=chosen_prob,
                    current_volatility=current_volatility,
                    avg_volatility=avg_volatility,
                    market_regime=entry_regime_info['regime'],
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
                    'entry_regime': entry_regime_info['regime'],
                    'entry_regime_vol': entry_regime_info['volatility_tag'],
                    'entry_price_change': entry_regime_info['price_change_pct']
                }

    return trades, capital

# Backtest first 30 windows
print("Running backtest on 30 windows...")
all_trades = []

for w in range(30):
    start_idx = w * STEP_SIZE
    end_idx = start_idx + WINDOW_SIZE

    if end_idx > len(df):
        break

    window_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    trades, final_capital = backtest_window_with_regime(window_df, LEVERAGE)
    all_trades.extend(trades)

df_trades = pd.DataFrame(all_trades)

# Analysis
print(f"\n{'='*80}")
print(f"Regime Analysis Results ({len(df_trades)} trades)")
print(f"{'='*80}\n")

if len(df_trades) > 0:
    # By entry regime
    print("📊 Performance by Entry Regime:\n")

    for regime in df_trades['entry_regime'].unique():
        regime_trades = df_trades[df_trades['entry_regime'] == regime]

        long_trades = regime_trades[regime_trades['side'] == 'LONG']
        short_trades = regime_trades[regime_trades['side'] == 'SHORT']

        print(f"  {regime}:")
        print(f"    Total: {len(regime_trades)} trades")

        if len(long_trades) > 0:
            long_wr = (long_trades['pnl_usd_net'] > 0).mean() * 100
            long_avg = long_trades['pnl_usd_net'].mean()
            print(f"    LONG: {len(long_trades)} trades, {long_wr:.1f}% WR, ${long_avg:.2f} avg")

        if len(short_trades) > 0:
            short_wr = (short_trades['pnl_usd_net'] > 0).mean() * 100
            short_avg = short_trades['pnl_usd_net'].mean()
            print(f"    SHORT: {len(short_trades)} trades, {short_wr:.1f}% WR, ${short_avg:.2f} avg")

        print()

    # By volatility
    print("\n📊 Performance by Volatility:\n")

    for vol_tag in df_trades['entry_regime_vol'].unique():
        vol_trades = df_trades[df_trades['entry_regime_vol'] == vol_tag]

        long_trades = vol_trades[vol_trades['side'] == 'LONG']
        short_trades = vol_trades[vol_trades['side'] == 'SHORT']

        print(f"  {vol_tag}:")
        print(f"    Total: {len(vol_trades)} trades")

        if len(long_trades) > 0:
            long_wr = (long_trades['pnl_usd_net'] > 0).mean() * 100
            long_avg = long_trades['pnl_usd_net'].mean()
            print(f"    LONG: {len(long_trades)} trades, {long_wr:.1f}% WR, ${long_avg:.2f} avg")

        if len(short_trades) > 0:
            short_wr = (short_trades['pnl_usd_net'] > 0).mean() * 100
            short_avg = short_trades['pnl_usd_net'].mean()
            print(f"    SHORT: {len(short_trades)} trades, {short_wr:.1f}% WR, ${short_avg:.2f} avg")

        print()

    # Key insight: When is SHORT better?
    print("\n💡 Key Insights:\n")

    long_trades = df_trades[df_trades['side'] == 'LONG']
    short_trades = df_trades[df_trades['side'] == 'SHORT']

    if len(long_trades) > 0 and len(short_trades) > 0:
        long_avg_pnl = long_trades['pnl_usd_net'].mean()
        short_avg_pnl = short_trades['pnl_usd_net'].mean()

        print(f"  Overall:")
        print(f"    LONG: {len(long_trades)} trades, ${long_avg_pnl:.2f} avg PnL")
        print(f"    SHORT: {len(short_trades)} trades, ${short_avg_pnl:.2f} avg PnL")
        print(f"    Difference: ${short_avg_pnl - long_avg_pnl:.2f}")

        if short_avg_pnl > long_avg_pnl:
            print(f"\n  ✅ SHORT is more profitable on average!")
        else:
            print(f"\n  ❌ LONG is more profitable on average")

    # Find best regime for SHORT
    short_by_regime = df_trades[df_trades['side'] == 'SHORT'].groupby('entry_regime')['pnl_usd_net'].agg(['mean', 'count'])
    short_by_regime = short_by_regime[short_by_regime['count'] >= 3]  # at least 3 trades

    if len(short_by_regime) > 0:
        best_regime = short_by_regime['mean'].idxmax()
        best_pnl = short_by_regime.loc[best_regime, 'mean']
        print(f"\n  🎯 Best regime for SHORT: {best_regime} (${best_pnl:.2f} avg PnL)")

# Save
output_file = RESULTS_DIR / "market_regime_analysis.csv"
df_trades.to_csv(output_file, index=False)
print(f"\n✅ Saved: {output_file.name}")

print(f"\n{'='*80}")
print("Analysis Complete!")
print(f"{'='*80}")
