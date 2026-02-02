"""
Reduced Feature Calculator - 중복 제거 버전

Purpose:
- 기존 calculate_all_features.py를 수정하지 않고
- 중복 feature를 제거한 새 버전 생성
- correlation 분석 결과 기반 최적화

Changes from Original:
- LONG: 44 → 37 features (-7, -15.9%)
- SHORT: 38 → 29 features (-9, -23.7%)
- Exit: 25 → 23 features (-2, -8.0%)

Author: Claude Code
Date: 2025-10-23
"""

import pandas as pd
import numpy as np
import talib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures


# LONG Entry Features (Reduced) - 37개
LONG_ENTRY_REDUCED_FEATURES = [
    # Basic (2)
    'close_change_1',
    'close_change_3',

    # Volume (1)
    'volume_ma_ratio',  # 중복 제거 (1개만)

    # Momentum (3)
    'rsi',
    'macd',
    # 'macd_signal',  # REMOVED (macd와 0.95 상관)
    'macd_diff',

    # Bollinger Bands (1)
    # 'bb_high',  # REMOVED
    'bb_mid',
    # 'bb_low',   # REMOVED

    # Support/Resistance (4)
    'distance_to_support_pct',
    'distance_to_resistance_pct',
    'num_support_touches',
    'num_resistance_touches',

    # Trendline (3)
    'upper_trendline_slope',
    # 'lower_trendline_slope',  # REMOVED (upper와 0.98 상관)
    'price_vs_upper_trendline_pct',
    # 'price_vs_lower_trendline_pct',  # REMOVED (upper와 0.92 상관)

    # Divergence (4)
    'rsi_bullish_divergence',
    'rsi_bearish_divergence',
    'macd_bullish_divergence',
    'macd_bearish_divergence',

    # Chart Patterns (4)
    'double_top',
    'double_bottom',
    'higher_highs_lows',
    'lower_highs_lows',

    # Volume Profile (2)
    'volume_price_correlation',
    'price_volume_trend',

    # Price Action (7)
    'body_to_range_ratio',
    'upper_shadow_ratio',
    'lower_shadow_ratio',
    'bullish_engulfing',
    'bearish_engulfing',
    'hammer',
    'shooting_star',
    'doji',

    # SHORT-specific (5)
    'distance_from_recent_high_pct',
    'bearish_candle_count',
    'red_candle_volume_ratio',
    # 'strong_selling_pressure',  # REMOVED (shooting_star와 0.81 상관)
    'price_momentum_near_resistance',
    'rsi_from_recent_peak',
    'consecutive_up_candles',
]


# SHORT Entry Features (Reduced) - 29개
SHORT_ENTRY_REDUCED_FEATURES = [
    # Symmetric (11)
    'rsi_deviation',
    'rsi_direction',
    'rsi_extreme',
    'macd_strength',
    # 'macd_divergence_abs',  # REMOVED (완전 중복 1.0)
    'macd_direction',
    'price_distance_ma20',
    # 'price_direction_ma20',  # REMOVED (rsi_direction과 0.82 상관)
    # 'price_distance_ma50',  # REMOVED (ma20과 0.81 상관)
    'price_direction_ma50',  # 방향은 유지 (장기 추세)
    'volatility',
    'atr_pct',
    # 'atr',  # REMOVED (atr_pct와 0.998 상관)

    # Inverse (13)
    'negative_momentum',
    'negative_acceleration',
    'down_candle_ratio',
    'down_candle_body',
    'lower_low_streak',
    # 'resistance_rejection_count',  # REMOVED (down_candle_ratio와 0.80 상관)
    'bearish_divergence',
    'volume_decline_ratio',
    'distribution_signal',
    'down_candle',
    'lower_low',
    'near_resistance',
    # 'rejection_from_resistance',  # REMOVED (down_candle과 0.95 상관)
    'volume_on_decline',
    'volume_on_advance',

    # Opportunity Cost (8)
    'bear_market_strength',
    'trend_strength',
    'downtrend_confirmed',
    'volatility_asymmetry',
    'below_support',
    'support_breakdown',
    'panic_selling',
    # 'downside_volatility',  # REMOVED (upside_vol과 0.88 상관)
    # 'upside_volatility',    # REMOVED (volatility로 충분)
    'ema_12',
]


# Exit Features (Reduced) - 23개
EXIT_REDUCED_FEATURES = [
    'rsi',
    'macd',
    # 'macd_signal',  # REMOVED (macd와 0.95 상관)
    'atr',
    'ema_12',
    # 'trend_strength',  # REMOVED (macd와 0.999 상관!)
    'volatility_regime',
    'volume_surge',
    'price_acceleration',
    'volume_ratio',
    'price_vs_ma20',
    'price_vs_ma50',
    'volatility_20',
    'rsi_slope',
    'rsi_overbought',
    'rsi_oversold',
    'rsi_divergence',
    'macd_histogram_slope',
    'macd_crossover',
    'macd_crossunder',
    'higher_high',
    'lower_low',
    'near_resistance',
    'near_support',
    'bb_position',
]


def calculate_reduced_features(df):
    """
    Calculate reduced feature set (중복 제거 버전)

    기존 calculate_all_features()를 호출한 후
    필요한 features만 선택

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with reduced features
    """
    print("Calculating REDUCED features (중복 제거 버전)...")

    # Step 1: LONG model features (기존 계산 사용)
    print("  1/4 Calculating LONG basic features...")
    df = calculate_features(df)

    print("  2/4 Calculating LONG advanced features...")
    adv_features = AdvancedTechnicalFeatures(lookback_sr=200, lookback_trend=50)
    df = adv_features.calculate_all_features(df)

    # Step 2: SHORT model features
    print("  3/4 Calculating SHORT features...")
    df = calculate_symmetric_features(df)
    df = calculate_inverse_features(df)
    df = calculate_opportunity_cost_features(df)

    # Step 3: Exit model features (추가 계산)
    print("  3.5/4 Calculating Exit model features...")
    df = calculate_exit_features(df)

    # Step 4: Clean NaN
    print("  4/4 Cleaning NaN values...")
    df = df.ffill().bfill().fillna(0)

    print(f"  ✅ Reduced features calculated ({len(df)} rows)")

    return df


def calculate_symmetric_features(df):
    """
    Symmetric features for SHORT model
    (기존 코드와 동일 - 변경 없음)
    """
    features = {}

    # RSI
    features['rsi_raw'] = talib.RSI(df['close'], timeperiod=14)
    features['rsi_deviation'] = np.abs(features['rsi_raw'] - 50)
    features['rsi_direction'] = np.sign(features['rsi_raw'] - 50)
    features['rsi_extreme'] = ((features['rsi_raw'] > 70) | (features['rsi_raw'] < 30)).astype(float)

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    features['macd_strength'] = np.abs(macd_hist)
    features['macd_direction'] = np.sign(macd_hist)
    features['macd_divergence_abs'] = np.abs(macd - macd_signal)  # 계산은 하되 모델에서 제외

    # Price vs MA
    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    features['price_distance_ma20'] = np.abs(df['close'] - ma_20) / ma_20
    features['price_direction_ma20'] = np.sign(df['close'] - ma_20)
    features['price_distance_ma50'] = np.abs(df['close'] - ma_50) / ma_50
    features['price_direction_ma50'] = np.sign(df['close'] - ma_50)

    # Volatility
    features['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    features['atr_pct'] = features['atr'] / df['close']

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def calculate_inverse_features(df):
    """
    Inverse features for SHORT model
    (기존 코드와 동일 - 변경 없음)
    """
    features = {}

    # Negative momentum
    features['negative_momentum'] = -df['close'].pct_change(5).clip(upper=0)
    features['negative_acceleration'] = -df['close'].diff().diff().clip(upper=0)

    # Down candle
    features['down_candle'] = (df['close'] < df['open']).astype(float)
    features['down_candle_ratio'] = features['down_candle'].rolling(10).mean()
    features['down_candle_body'] = (df['open'] - df['close']).clip(lower=0)

    # Lower lows
    features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    features['lower_low_streak'] = features['lower_low'].rolling(5).sum()

    # Resistance rejection
    resistance = df['high'].rolling(20).max()
    features['near_resistance'] = (df['high'] > resistance.shift(1) * 0.99).astype(float)
    features['rejection_from_resistance'] = (
        (features['near_resistance'] == 1) & (df['close'] < df['open'])
    ).astype(float)
    features['resistance_rejection_count'] = features['rejection_from_resistance'].rolling(10).sum()

    # Bearish divergence
    price_change_5 = df['close'].diff(5)
    rsi_change_5 = df['rsi_raw'].diff(5) if 'rsi_raw' in df.columns else 0
    features['bearish_divergence'] = (
        (price_change_5 > 0) & (rsi_change_5 < 0)
    ).astype(float)

    # Volume on decline
    features['price_up'] = (df['close'] > df['close'].shift(1)).astype(float)
    features['volume_on_decline'] = df['volume'] * (1 - features['price_up'])
    features['volume_on_advance'] = df['volume'] * features['price_up']
    features['volume_decline_ratio'] = (
        features['volume_on_decline'].rolling(10).sum() /
        (features['volume_on_advance'].rolling(10).sum() + 1e-10)
    )

    # Distribution signal
    price_range_20 = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
    volume_surge = df['volume'] / df['volume'].rolling(20).mean()
    features['distribution_signal'] = (
        (price_range_20 < 0.05) &
        (volume_surge > 1.5)
    ).astype(float)

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def calculate_opportunity_cost_features(df):
    """
    Opportunity cost features for SHORT model
    (기존 코드와 동일 - 변경 없음)
    """
    features = {}

    # Bear market strength
    returns_20 = df['close'].pct_change(20)
    features['bear_market_strength'] = (-returns_20).clip(lower=0)

    # Trend strength
    features['ema_12'] = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    features['trend_strength'] = (features['ema_12'] - ema_26) / df['close']
    features['downtrend_confirmed'] = (features['trend_strength'] < -0.01).astype(float)

    # Volatility asymmetry
    returns = df['close'].pct_change()
    upside_vol = returns[returns > 0].rolling(20).std()
    downside_vol = returns[returns < 0].rolling(20).std()
    features['downside_volatility'] = downside_vol.ffill()
    features['upside_volatility'] = upside_vol.ffill()
    features['volatility_asymmetry'] = features['downside_volatility'] / (features['upside_volatility'] + 1e-10)

    # Support breakdown
    support = df['low'].rolling(50).min()
    features['below_support'] = (df['close'] < support.shift(1) * 1.01).astype(float)
    features['support_breakdown'] = (
        (df['close'].shift(1) >= support.shift(1)) &
        (df['close'] < support.shift(1))
    ).astype(float)

    # Panic selling
    features['panic_selling'] = (
        (df['down_candle'] == 1) &
        (df['volume'] > df['volume'].rolling(10).mean() * 1.5)
    ).astype(float)

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def calculate_exit_features(df):
    """
    Exit model 추가 features
    (기존에 missing이었던 16개 계산)
    """
    features = {}

    # Volatility regime
    volatility_median = df['volatility'].rolling(100).median()
    features['volatility_regime'] = np.where(
        df['volatility'] > volatility_median * 1.2, 1,
        np.where(df['volatility'] < volatility_median * 0.8, -1, 0)
    )

    # Volume surge
    features['volume_surge'] = (
        df['volume'] > df['volume'].rolling(20).mean() * 1.5
    ).astype(float)

    # Price acceleration
    features['price_acceleration'] = df['close'].diff().diff()

    # Price vs MA
    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    features['price_vs_ma20'] = (df['close'] - ma_20) / ma_20
    features['price_vs_ma50'] = (df['close'] - ma_50) / ma_50

    # Volatility 20
    features['volatility_20'] = df['close'].pct_change().rolling(20).std()

    # RSI slope
    if 'rsi' in df.columns:
        features['rsi_slope'] = df['rsi'].diff(3) / 3
        features['rsi_overbought'] = (df['rsi'] > 70).astype(float)
        features['rsi_oversold'] = (df['rsi'] < 30).astype(float)
        features['rsi_divergence'] = 0  # Placeholder

    # MACD histogram slope
    if 'macd_diff' in df.columns:
        features['macd_histogram_slope'] = df['macd_diff'].diff(3) / 3

    # MACD crossovers
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        features['macd_crossover'] = (
            (df['macd'] > df['macd_signal']) &
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(float)
        features['macd_crossunder'] = (
            (df['macd'] < df['macd_signal']) &
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        ).astype(float)

    # Price patterns
    features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)

    # Support/Resistance proximity (이미 계산되어 있으면 사용)
    if 'near_support' not in df.columns:
        features['near_support'] = 0

    # BB position
    if 'bb_high' in df.columns and 'bb_low' in df.columns:
        features['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-10)
    else:
        features['bb_position'] = 0.5

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


if __name__ == "__main__":
    # Test feature calculation
    print("="*80)
    print("Testing Reduced Feature Calculation")
    print("="*80)

    # Load sample data
    data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file).tail(1000)

    print(f"\nOriginal data: {len(df)} rows, {len(df.columns)} columns")

    # Calculate reduced features
    df = calculate_reduced_features(df)

    print(f"After feature calculation: {len(df)} rows, {len(df.columns)} columns")

    # Check for reduced features
    print("\n" + "="*80)
    print("Feature Counts")
    print("="*80)

    long_available = [f for f in LONG_ENTRY_REDUCED_FEATURES if f in df.columns]
    long_missing = [f for f in LONG_ENTRY_REDUCED_FEATURES if f not in df.columns]

    print(f"\nLONG Entry: {len(long_available)}/{len(LONG_ENTRY_REDUCED_FEATURES)} features")
    if long_missing:
        print(f"  Missing: {long_missing}")

    short_available = [f for f in SHORT_ENTRY_REDUCED_FEATURES if f in df.columns]
    short_missing = [f for f in SHORT_ENTRY_REDUCED_FEATURES if f not in df.columns]

    print(f"\nSHORT Entry: {len(short_available)}/{len(SHORT_ENTRY_REDUCED_FEATURES)} features")
    if short_missing:
        print(f"  Missing: {short_missing}")

    exit_available = [f for f in EXIT_REDUCED_FEATURES if f in df.columns]
    exit_missing = [f for f in EXIT_REDUCED_FEATURES if f not in df.columns]

    print(f"\nExit: {len(exit_available)}/{len(EXIT_REDUCED_FEATURES)} features")
    if exit_missing:
        print(f"  Missing: {exit_missing}")

    print("\n" + "="*80)
    print("Test Complete")
    print("="*80)
