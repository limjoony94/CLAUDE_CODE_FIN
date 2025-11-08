"""
Capital Lock 문제 정량적 분석

목표: Trade-by-trade 레벨에서 SHORT가 LONG 기회를 차단하는 영향을 정량화
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
print("Capital Lock 정량적 분석")
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

print(f"✅ Models loaded: LONG={len(long_features)} features, SHORT={len(short_features)} features")

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)

# Calculate all features needed
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

def simulate_single_window_with_analysis(window_df, leverage):
    """
    단일 window 시뮬레이션 with detailed analysis

    Returns:
        - trades: 실제 수행된 거래
        - missed_opportunities: 놓친 LONG 기회
        - capital_lock_analysis: capital lock 분석
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []
    missed_opportunities = []

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

        # Manage existing position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # Calculate P&L
            if position['side'] == 'LONG':
                price_change_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                price_change_pct = (entry_price - current_price) / entry_price

            leveraged_pnl_pct = price_change_pct * leverage
            leveraged_pnl_usd = leveraged_pnl_pct * position['base_value']

            # Exit logic
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
                    'base_value': position['base_value'],
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

            # Get LONG signal
            long_feat = window_df[long_features].iloc[i:i+1].values
            if not np.isnan(long_feat).any():
                long_feat_scaled = long_scaler.transform(long_feat)
                long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
            else:
                long_prob = 0.0

            # Get SHORT signal
            short_feat = window_df[short_features].iloc[i:i+1].values
            if not np.isnan(short_feat).any():
                short_feat_scaled = short_scaler.transform(short_feat)
                short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
            else:
                short_prob = 0.0

            # Decision: which side to take?
            long_signal = long_prob >= LONG_THRESHOLD
            short_signal = short_prob >= SHORT_THRESHOLD

            # Calculate regime
            lookback = 20
            recent_data = window_df.iloc[max(0, i-lookback):i+1]
            if len(recent_data) >= lookback:
                start_price = recent_data['close'].iloc[0]
                end_price = recent_data['close'].iloc[-1]
                price_change_pct = ((end_price / start_price) - 1) * 100

                if price_change_pct > 3.0:
                    regime = "Bull"
                elif price_change_pct < -2.0:
                    regime = "Bear"
                else:
                    regime = "Sideways"
            else:
                regime = "Unknown"

            current_volatility = window_df['atr_pct'].iloc[i] if 'atr_pct' in window_df.columns else 0.01
            avg_volatility = window_df['atr_pct'].iloc[max(0, i-50):i].mean() if 'atr_pct' in window_df.columns else 0.01

            # Choose best signal
            chosen_side = None
            chosen_prob = 0.0

            if long_signal and short_signal:
                # Both signals - choose stronger
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

            # If SHORT chosen, record missed LONG opportunity
            if chosen_side == 'SHORT' and long_signal:
                missed_opportunities.append({
                    'idx': i,
                    'long_prob': long_prob,
                    'short_prob': short_prob,
                    'chose': 'SHORT',
                    'regime': regime
                })

            # Enter position
            if chosen_side is not None:
                sizing_result = sizer.calculate_position_size(
                    capital=capital,
                    signal_strength=chosen_prob,
                    current_volatility=current_volatility,
                    avg_volatility=avg_volatility,
                    market_regime=regime,
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
                    'regime': regime
                }

    return trades, missed_opportunities, capital

# Run analysis on first 10 windows
print(f"\n{'='*80}")
print("Analyzing first 10 windows for capital lock impact")
print(f"{'='*80}\n")

all_missed = []
all_trades = []

for w in range(10):
    start_idx = w * STEP_SIZE
    end_idx = start_idx + WINDOW_SIZE

    if end_idx > len(df):
        break

    window_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    trades, missed, final_capital = simulate_single_window_with_analysis(window_df, LEVERAGE)

    all_trades.extend(trades)
    all_missed.extend(missed)

    num_long = len([t for t in trades if t['side'] == 'LONG'])
    num_short = len([t for t in trades if t['side'] == 'SHORT'])

    print(f"Window {w+1}: {num_long} LONG, {num_short} SHORT trades, {len(missed)} missed LONG opportunities")

# Analysis
print(f"\n{'='*80}")
print("Capital Lock Impact Analysis")
print(f"{'='*80}\n")

df_trades = pd.DataFrame(all_trades)
df_missed = pd.DataFrame(all_missed)

if len(df_missed) > 0:
    print(f"📊 Missed LONG Opportunities: {len(df_missed)}")
    print(f"   Average LONG prob of missed: {df_missed['long_prob'].mean():.3f}")
    print(f"   Average SHORT prob chosen: {df_missed['short_prob'].mean():.3f}")
    print(f"\n   Regime breakdown:")
    for regime in df_missed['regime'].unique():
        count = len(df_missed[df_missed['regime'] == regime])
        pct = (count / len(df_missed)) * 100
        print(f"     {regime}: {count} ({pct:.1f}%)")
else:
    print("No missed LONG opportunities (all SHORT were chosen over lower LONG signals)")

print(f"\n📈 Actual Trades: {len(df_trades)}")
if len(df_trades) > 0:
    long_trades = df_trades[df_trades['side'] == 'LONG']
    short_trades = df_trades[df_trades['side'] == 'SHORT']

    print(f"   LONG: {len(long_trades)} trades")
    if len(long_trades) > 0:
        print(f"     Win Rate: {(long_trades['pnl_usd_net'] > 0).mean() * 100:.1f}%")
        print(f"     Avg PnL: ${long_trades['pnl_usd_net'].mean():.2f}")

    print(f"   SHORT: {len(short_trades)} trades")
    if len(short_trades) > 0:
        print(f"     Win Rate: {(short_trades['pnl_usd_net'] > 0).mean() * 100:.1f}%")
        print(f"     Avg PnL: ${short_trades['pnl_usd_net'].mean():.2f}")

# Save detailed analysis
output_file = RESULTS_DIR / "capital_lock_analysis.csv"
if len(df_missed) > 0:
    df_missed.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file.name}")

print(f"\n{'='*80}")
print("Analysis Complete!")
print(f"{'='*80}")
