"""
Unified Feature Calculation for LONG + SHORT Models
====================================================

Combines ALL features needed by both models:
- LONG model features (basic + advanced)
- SHORT model features (symmetric + inverse + opportunity cost)

Usage:
    from calculate_all_features import calculate_all_features
    df = calculate_all_features(df)
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


def calculate_symmetric_features(df):
    """
    Symmetric features - LONG/SHORT equally weighted

    Key insight: Measure DISTANCE and DIRECTION separately
    (13 features)
    """
    # Calculate all features at once to avoid fragmentation
    features = {}

    # RSI: Deviation from neutral (50)
    features['rsi_raw'] = talib.RSI(df['close'], timeperiod=14)
    features['rsi_deviation'] = np.abs(features['rsi_raw'] - 50)
    features['rsi_direction'] = np.sign(features['rsi_raw'] - 50)
    features['rsi_extreme'] = ((features['rsi_raw'] > 70) | (features['rsi_raw'] < 30)).astype(float)

    # MACD: Absolute strength + direction
    macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    features['macd_strength'] = np.abs(macd_hist)
    features['macd_direction'] = np.sign(macd_hist)
    features['macd_divergence_abs'] = np.abs(macd - macd_signal)

    # Price vs MA: Distance + direction
    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    features['price_distance_ma20'] = np.abs(df['close'] - ma_20) / ma_20
    features['price_direction_ma20'] = np.sign(df['close'] - ma_20)
    features['price_distance_ma50'] = np.abs(df['close'] - ma_50) / ma_50
    features['price_direction_ma50'] = np.sign(df['close'] - ma_50)

    # Volatility (direction-agnostic)
    # Note: 'volatility' already exists from LONG features, only add ATR metrics
    features['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    features['atr_pct'] = features['atr'] / df['close']

    # Join all at once
    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def calculate_inverse_features(df):
    """
    Inverse features - SHORT-specific decline detection

    Focus on downward movement, bearish patterns, weakness
    (15 features)
    """
    # Calculate all features at once to avoid fragmentation
    features = {}

    # Negative momentum
    features['negative_momentum'] = -df['close'].pct_change(5).clip(upper=0)
    features['negative_acceleration'] = -df['close'].diff().diff().clip(upper=0)

    # Down candle analysis
    features['down_candle'] = (df['close'] < df['open']).astype(float)
    features['down_candle_ratio'] = features['down_candle'].rolling(10).mean()
    features['down_candle_body'] = (df['open'] - df['close']).clip(lower=0)

    # Lower lows (weakness)
    features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    features['lower_low_streak'] = features['lower_low'].rolling(5).sum()

    # Resistance rejection (failed breakouts)
    resistance = df['high'].rolling(20).max()
    features['near_resistance'] = (df['high'] > resistance.shift(1) * 0.99).astype(float)
    features['rejection_from_resistance'] = (
        (features['near_resistance'] == 1) & (df['close'] < df['open'])
    ).astype(float)
    features['resistance_rejection_count'] = features['rejection_from_resistance'].rolling(10).sum()

    # Bearish divergence (price up but momentum down)
    price_change_5 = df['close'].diff(5)
    rsi_change_5 = df['rsi_raw'].diff(5) if 'rsi_raw' in df.columns else 0
    features['bearish_divergence'] = (
        (price_change_5 > 0) & (rsi_change_5 < 0)
    ).astype(float)

    # Volume on decline vs advance
    features['price_up'] = (df['close'] > df['close'].shift(1)).astype(float)
    features['volume_on_decline'] = df['volume'] * (1 - features['price_up'])
    features['volume_on_advance'] = df['volume'] * features['price_up']
    features['volume_decline_ratio'] = (
        features['volume_on_decline'].rolling(10).sum() /
        (features['volume_on_advance'].rolling(10).sum() + 1e-10)
    )

    # Distribution phase detection (high volume, price stagnant)
    price_range_20 = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
    volume_surge = df['volume'] / df['volume'].rolling(20).mean()
    features['distribution_signal'] = (
        (price_range_20 < 0.05) &  # Narrow range
        (volume_surge > 1.5)  # High volume
    ).astype(float)

    # Join all at once
    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def calculate_opportunity_cost_features(df):
    """
    Opportunity cost features - SHORT vs LONG comparison

    Label SHORT only when it's BETTER than LONG
    (10 features)
    """
    # Calculate all features at once to avoid fragmentation
    features = {}

    # Bear market strength
    returns_20 = df['close'].pct_change(20)
    features['bear_market_strength'] = (-returns_20).clip(lower=0)

    # Trend strength (negative for downtrend)
    features['ema_12'] = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()  # Intermediate calculation only
    features['trend_strength'] = (features['ema_12'] - ema_26) / df['close']
    features['downtrend_confirmed'] = (features['trend_strength'] < -0.01).astype(float)

    # Downside vs upside volatility asymmetry
    returns = df['close'].pct_change()
    upside_vol = returns[returns > 0].rolling(20).std()
    downside_vol = returns[returns < 0].rolling(20).std()
    features['downside_volatility'] = downside_vol.ffill()
    features['upside_volatility'] = upside_vol.ffill()
    features['volatility_asymmetry'] = features['downside_volatility'] / (features['upside_volatility'] + 1e-10)

    # Support breakdown (bearish)
    support = df['low'].rolling(50).min()
    features['below_support'] = (df['close'] < support.shift(1) * 1.01).astype(float)
    features['support_breakdown'] = (
        (df['close'].shift(1) >= support.shift(1)) &
        (df['close'] < support.shift(1))
    ).astype(float)

    # Weak hands exit (high volume down candles)
    features['panic_selling'] = (
        (df['down_candle'] == 1) &
        (df['volume'] > df['volume'].rolling(10).mean() * 1.5)
    ).astype(float)

    # Join all at once
    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def calculate_all_features(df):
    """
    Calculate ALL features needed by LONG + SHORT models

    Pipeline:
    1. LONG model features (basic + advanced)
    2. SHORT model features (symmetric + inverse + opportunity cost)
    3. Clean NaN values

    Returns:
        DataFrame with ALL features calculated
    """
    print("Calculating ALL features (LONG + SHORT)...")

    # Step 1: LONG model features
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

    # Step 3: Clean NaN
    print("  4/4 Cleaning NaN values...")
    df = df.ffill().bfill().fillna(0)

    print(f"  ✅ All features calculated ({len(df)} rows)")

    return df


# SHORT model feature list (38 features)
SHORT_FEATURE_COLUMNS = [
    # Symmetric (13)
    'rsi_deviation', 'rsi_direction', 'rsi_extreme',
    'macd_strength', 'macd_direction', 'macd_divergence_abs',
    'price_distance_ma20', 'price_direction_ma20',
    'price_distance_ma50', 'price_direction_ma50',
    'volatility', 'atr_pct', 'atr',

    # Inverse (15)
    'negative_momentum', 'negative_acceleration',
    'down_candle_ratio', 'down_candle_body',
    'lower_low_streak', 'resistance_rejection_count',
    'bearish_divergence', 'volume_decline_ratio',
    'distribution_signal', 'down_candle',
    'lower_low', 'near_resistance', 'rejection_from_resistance',
    'volume_on_decline', 'volume_on_advance',

    # Opportunity Cost (10) - Note: ema_26 not used by model, only ema_12
    'bear_market_strength', 'trend_strength', 'downtrend_confirmed',
    'volatility_asymmetry', 'below_support', 'support_breakdown',
    'panic_selling', 'downside_volatility', 'upside_volatility', 'ema_12'
]


if __name__ == "__main__":
    # Test feature calculation
    import pandas as pd
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "historical"

    print("="*80)
    print("Testing Unified Feature Calculation")
    print("="*80)

    # Load sample data
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file).tail(1000)

    print(f"\nOriginal data: {len(df)} rows, {len(df.columns)} columns")

    # Calculate all features
    df = calculate_all_features(df)

    print(f"After feature calculation: {len(df)} rows, {len(df.columns)} columns")

    # Check for SHORT features
    missing_features = [f for f in SHORT_FEATURE_COLUMNS if f not in df.columns]

    if missing_features:
        print(f"\n❌ Missing {len(missing_features)} SHORT features:")
        for f in missing_features:
            print(f"   - {f}")
    else:
        print(f"\n✅ All {len(SHORT_FEATURE_COLUMNS)} SHORT features present!")

    print("\n" + "="*80)
    print("Test Complete")
    print("="*80)
