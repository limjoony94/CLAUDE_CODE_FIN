"""
Deep Signal Quality Analysis
==============================

근본적 문제 파악:
1. LONG vs SHORT 신호 품질 비교
2. 신호 타이밍과 시장 상황의 정합성
3. 확률 vs 실제 수익성 상관관계
4. 시장 구조적 편향 분석
5. 신호 충돌 분석
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

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("DEEP SIGNAL QUALITY ANALYSIS")
print("="*80)
print("\n근본적 문제 파악을 위한 심층 분석\n")

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
print(f"  LONG features: {len(long_feature_columns)}")
print(f"  SHORT features: {len(short_feature_columns)}")

# Load data (sample for speed)
print("\nLoading data (sample for speed)...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)

# Sample: last 50,000 candles (~174 days)
df = df.tail(50000).reset_index(drop=True)
print(f"  ✅ Data loaded: {len(df)} candles (~{len(df)//288:.0f} days)")

# Calculate features
print("\nCalculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)

# Calculate SHORT features (complete set)
import talib

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

df = calculate_short_features(df)
df = df.ffill().bfill().fillna(0)
print(f"  ✅ Features calculated")

# Calculate market trend
print("\nCalculating market trend...")
df['price_change_20'] = df['close'].pct_change(20)
df['trend'] = 'sideways'
df.loc[df['price_change_20'] > 0.02, 'trend'] = 'bull'
df.loc[df['price_change_20'] < -0.02, 'trend'] = 'bear'

# Generate signals
print("\nGenerating signals...")
long_probs = []
short_probs = []

for i in range(len(df)):
    # LONG signal
    long_feat = df[long_feature_columns].iloc[i:i+1].values
    if not np.isnan(long_feat).any():
        long_feat_scaled = long_scaler.transform(long_feat)
        long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
    else:
        long_prob = 0.0
    long_probs.append(long_prob)

    # SHORT signal
    short_feat = df[short_feature_columns].iloc[i:i+1].values
    if not np.isnan(short_feat).any():
        short_feat_scaled = short_scaler.transform(short_feat)
        short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
    else:
        short_prob = 0.0
    short_probs.append(short_prob)

df['long_prob'] = long_probs
df['short_prob'] = short_probs
print(f"  ✅ Signals generated for {len(df)} candles")

# Calculate actual future returns (for validation)
df['future_return_5'] = df['close'].pct_change(5).shift(-5)  # 5 candles ahead

print(f"\n{'='*80}")
print("ANALYSIS 1: Signal Frequency Distribution")
print(f"{'='*80}\n")

# Signal frequency by threshold
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
for thresh in thresholds:
    long_count = (df['long_prob'] >= thresh).sum()
    short_count = (df['short_prob'] >= thresh).sum()
    overlap = ((df['long_prob'] >= thresh) & (df['short_prob'] >= thresh)).sum()

    print(f"Threshold {thresh:.1f}:")
    print(f"  LONG signals: {long_count:,} ({long_count/len(df)*100:.2f}%)")
    print(f"  SHORT signals: {short_count:,} ({short_count/len(df)*100:.2f}%)")
    print(f"  Overlap (both high): {overlap:,} ({overlap/len(df)*100:.2f}%)")
    print()

print(f"\n{'='*80}")
print("ANALYSIS 2: Signal Quality vs Market Trend")
print(f"{'='*80}\n")

# Group by market trend
trend_analysis = []
for trend in ['bull', 'sideways', 'bear']:
    trend_df = df[df['trend'] == trend]

    long_high = (trend_df['long_prob'] >= 0.7).sum()
    short_high = (trend_df['short_prob'] >= 0.7).sum()

    # Average future return when signal is high
    long_signal_df = trend_df[trend_df['long_prob'] >= 0.7]
    short_signal_df = trend_df[trend_df['short_prob'] >= 0.7]

    long_avg_return = long_signal_df['future_return_5'].mean() if len(long_signal_df) > 0 else 0
    short_avg_return = short_signal_df['future_return_5'].mean() if len(short_signal_df) > 0 else 0

    trend_analysis.append({
        'trend': trend,
        'candles': len(trend_df),
        'long_signals': long_high,
        'short_signals': short_high,
        'long_avg_return': long_avg_return,
        'short_avg_return_raw': short_avg_return,
        'short_avg_return_inverted': -short_avg_return  # SHORT profits from decline
    })

    print(f"{trend.upper()} Market:")
    print(f"  Candles: {len(trend_df):,} ({len(trend_df)/len(df)*100:.1f}%)")
    print(f"  LONG signals (≥0.7): {long_high} → Avg future return: {long_avg_return*100:.3f}%")
    print(f"  SHORT signals (≥0.7): {short_high} → Avg future return: {short_avg_return*100:.3f}% (raw)")
    print(f"                                    → SHORT P&L: {-short_avg_return*100:.3f}%")
    print()

print(f"\n{'='*80}")
print("ANALYSIS 3: Probability vs Actual Performance")
print(f"{'='*80}\n")

# Binned probability analysis
prob_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print("LONG Model:")
for i in range(len(prob_bins)-1):
    bin_mask = (df['long_prob'] >= prob_bins[i]) & (df['long_prob'] < prob_bins[i+1])
    bin_df = df[bin_mask]

    if len(bin_df) > 0:
        avg_return = bin_df['future_return_5'].mean()
        win_rate = (bin_df['future_return_5'] > 0).sum() / len(bin_df) * 100
        print(f"  P({prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}): {len(bin_df):,} signals | "
              f"Avg return: {avg_return*100:.3f}% | WR: {win_rate:.1f}%")

print("\nSHORT Model:")
for i in range(len(prob_bins)-1):
    bin_mask = (df['short_prob'] >= prob_bins[i]) & (df['short_prob'] < prob_bins[i+1])
    bin_df = df[bin_mask]

    if len(bin_df) > 0:
        avg_return_raw = bin_df['future_return_5'].mean()
        avg_return_short = -avg_return_raw  # SHORT profits from decline
        win_rate = (bin_df['future_return_5'] < 0).sum() / len(bin_df) * 100  # SHORT wins when price falls
        print(f"  P({prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}): {len(bin_df):,} signals | "
              f"SHORT P&L: {avg_return_short*100:.3f}% | WR: {win_rate:.1f}%")

print(f"\n{'='*80}")
print("ANALYSIS 4: Market Structural Bias")
print(f"{'='*80}\n")

# Overall market direction
total_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
bull_candles = (df['trend'] == 'bull').sum()
bear_candles = (df['trend'] == 'bear').sum()
sideways_candles = (df['trend'] == 'sideways').sum()

print(f"Period analyzed: {len(df):,} candles (~{len(df)//288:.0f} days)")
print(f"Total price change: {total_return:+.2f}%")
print(f"Market bias:")
print(f"  Bull: {bull_candles:,} candles ({bull_candles/len(df)*100:.1f}%)")
print(f"  Bear: {bear_candles:,} candles ({bear_candles/len(df)*100:.1f}%)")
print(f"  Sideways: {sideways_candles:,} candles ({sideways_candles/len(df)*100:.1f}%)")

avg_bull_return = df[df['trend'] == 'bull']['future_return_5'].mean()
avg_bear_return = df[df['trend'] == 'bear']['future_return_5'].mean()
avg_sideways_return = df[df['trend'] == 'sideways']['future_return_5'].mean()

print(f"\nAverage 5-candle returns by regime:")
print(f"  Bull market: {avg_bull_return*100:.3f}%")
print(f"  Bear market: {avg_bear_return*100:.3f}%")
print(f"  Sideways: {avg_sideways_return*100:.3f}%")

print(f"\n{'='*80}")
print("ANALYSIS 5: Signal Conflict (High LONG + High SHORT)")
print(f"{'='*80}\n")

# When both signals are high
conflict_mask = (df['long_prob'] >= 0.7) & (df['short_prob'] >= 0.7)
conflict_df = df[conflict_mask]

print(f"Conflicts (both ≥0.7): {len(conflict_df):,} candles ({len(conflict_df)/len(df)*100:.2f}%)")

if len(conflict_df) > 0:
    print(f"\nIn conflict situations:")
    print(f"  Avg LONG prob: {conflict_df['long_prob'].mean():.3f}")
    print(f"  Avg SHORT prob: {conflict_df['short_prob'].mean():.3f}")
    print(f"  Avg future return: {conflict_df['future_return_5'].mean()*100:.3f}%")

    # Which direction wins?
    up_count = (conflict_df['future_return_5'] > 0).sum()
    down_count = (conflict_df['future_return_5'] < 0).sum()
    print(f"  Actual direction: UP {up_count} ({up_count/len(conflict_df)*100:.1f}%) | "
          f"DOWN {down_count} ({down_count/len(conflict_df)*100:.1f}%)")

print(f"\n{'='*80}")
print("KEY FINDINGS")
print(f"{'='*80}\n")

# Calculate key metrics
long_signals_70 = (df['long_prob'] >= 0.7).sum()
short_signals_70 = (df['short_prob'] >= 0.7).sum()

long_roi = df[df['long_prob'] >= 0.7]['future_return_5'].mean() * 100
short_roi = -df[df['short_prob'] >= 0.7]['future_return_5'].mean() * 100  # SHORT P&L

print(f"1. Signal Frequency (threshold 0.7):")
print(f"   LONG: {long_signals_70:,} signals ({long_signals_70/len(df)*100:.2f}%)")
print(f"   SHORT: {short_signals_70:,} signals ({short_signals_70/len(df)*100:.2f}%)")
print(f"   Ratio: LONG is {long_signals_70/short_signals_70:.1f}x more frequent")

print(f"\n2. Average ROI per signal:")
print(f"   LONG: {long_roi:+.3f}% per signal")
print(f"   SHORT: {short_roi:+.3f}% per signal")

print(f"\n3. Market Structural Bias:")
print(f"   Total return: {total_return:+.2f}%")
print(f"   Bull/Bear ratio: {bull_candles/bear_candles:.2f}")
print(f"   → Market has {'bullish' if total_return > 0 else 'bearish'} bias")

print(f"\n4. Model Appropriateness:")
bull_long_signals = (df[df['trend'] == 'bull']['long_prob'] >= 0.7).sum()
bear_short_signals = (df[df['trend'] == 'bear']['short_prob'] >= 0.7).sum()
bull_short_signals = (df[df['trend'] == 'bull']['short_prob'] >= 0.7).sum()

print(f"   LONG signals in bull market: {bull_long_signals} (appropriate)")
print(f"   SHORT signals in bear market: {bear_short_signals} (appropriate)")
print(f"   SHORT signals in bull market: {bull_short_signals} (inappropriate)")

inappropriate_short_pct = bull_short_signals / (df['short_prob'] >= 0.7).sum() * 100 if (df['short_prob'] >= 0.7).sum() > 0 else 0
print(f"   → {inappropriate_short_pct:.1f}% of SHORT signals occur in WRONG market condition")

print(f"\n{'='*80}")
print("Analysis Complete!")
print(f"{'='*80}\n")
