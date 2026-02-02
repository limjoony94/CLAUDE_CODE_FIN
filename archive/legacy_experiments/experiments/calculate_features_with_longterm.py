"""
Enhanced Feature Calculation with Long-Term Indicators
=======================================================

Adds long-term (200-period) indicators to existing features:
- MA/EMA 200 (Golden Cross / Death Cross)
- Volume MA 200 (Accumulation / Distribution)
- ATR 200 (Volatility Regime)
- RSI 200 (Long-term Trend)
- Support/Resistance 200 (Major Levels)

Total Features: ~127 (107 baseline + 20 long-term)

Usage:
    from calculate_features_with_longterm import calculate_all_features_enhanced
    df = calculate_all_features_enhanced(df)

Created: 2025-10-23
"""

import pandas as pd
import numpy as np
import talib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features


def calculate_long_term_features(df):
    """
    Calculate long-term (200-period) indicators

    Adds 20 new features:
    - MA/EMA 200 + cross signals (5 features)
    - Volume MA 200 + regime (2 features)
    - ATR 200 + regime (2 features)
    - RSI 200 + trend (2 features)
    - Bollinger Bands 200 (4 features)
    - Support/Resistance 200 (3 features)
    - Momentum 200 (2 features)

    Returns: DataFrame with 20 additional features
    """

    # ========================================================================
    # Priority 1: Moving Averages (MA/EMA 200) - Golden/Death Cross
    # ========================================================================

    # MA 200 - Most important long-term trendline
    df['ma_200'] = df['close'].rolling(window=200).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # MA Cross Signals (단기 MA20 vs 장기 MA200)
    # Assumes ma_20 already exists from baseline features
    if 'ma_20' not in df.columns:
        df['ma_20'] = df['close'].rolling(window=20).mean()

    # Distance from MA 200 (relative position)
    df['price_vs_ma200'] = (df['close'] - df['ma_200']) / df['ma_200']

    # Golden Cross / Death Cross signals
    df['golden_cross'] = (
        (df['ma_20'] > df['ma_200']) &
        (df['ma_20'].shift(1) <= df['ma_200'].shift(1))
    ).astype(float)

    df['death_cross'] = (
        (df['ma_20'] < df['ma_200']) &
        (df['ma_20'].shift(1) >= df['ma_200'].shift(1))
    ).astype(float)

    # MA trend strength (단기가 장기보다 얼마나 위/아래)
    df['ma_cross_strength'] = (df['ma_20'] - df['ma_200']) / df['ma_200']


    # ========================================================================
    # Priority 2: Volume MA 200 - Accumulation/Distribution
    # ========================================================================

    # Volume MA 200 - Long-term average volume
    df['volume_ma_200'] = df['volume'].rolling(window=200).mean()

    # Volume Regime (현재 거래량 vs 장기 평균)
    df['volume_regime'] = df['volume'] / df['volume_ma_200']
    # > 1.5: High volume (accumulation/distribution)
    # < 0.7: Low volume (consolidation)


    # ========================================================================
    # Priority 3: ATR 200 - Volatility Regime
    # ========================================================================

    # ATR 200 - Long-term baseline volatility
    df['atr_200'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=200)

    # Volatility Regime (현재 변동성 vs 장기 baseline)
    # Assumes atr already exists from baseline
    if 'atr' not in df.columns:
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    df['volatility_regime'] = df['atr'] / df['atr_200']
    # > 1.5: High volatility (risk increases)
    # < 0.7: Low volatility (potential breakout after squeeze)


    # ========================================================================
    # Priority 4: RSI 200 - Long-term Trend
    # ========================================================================

    # RSI 200 - Long-term momentum
    df['rsi_200'] = talib.RSI(df['close'], timeperiod=200)

    # RSI Trend (단기 RSI vs 장기 RSI)
    # Assumes rsi already exists from baseline
    if 'rsi' not in df.columns:
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)

    df['rsi_trend'] = df['rsi'] - df['rsi_200']
    # Positive: 단기가 장기보다 강함 (uptrend)
    # Negative: 단기가 장기보다 약함 (downtrend)


    # ========================================================================
    # Priority 5: Bollinger Bands 200
    # ========================================================================

    # BB 200 - Long-term volatility bands
    bb_upper_200, bb_mid_200, bb_lower_200 = talib.BBANDS(
        df['close'],
        timeperiod=200,
        nbdevup=2,
        nbdevdn=2
    )

    df['bb_mid_200'] = bb_mid_200
    df['bb_width_200'] = (bb_upper_200 - bb_lower_200) / bb_mid_200

    # BB Position (price position within bands)
    df['bb_position_200'] = (df['close'] - bb_lower_200) / (bb_upper_200 - bb_lower_200)
    # 0-1 range: 0 = lower band, 0.5 = middle, 1 = upper band

    # BB Squeeze Detection (단기 BB vs 장기 BB)
    # Assumes bb_width already exists from baseline
    if 'bb_width' not in df.columns:
        bb_upper_20, bb_mid_20, bb_lower_20 = talib.BBANDS(df['close'], timeperiod=20)
        df['bb_width'] = (bb_upper_20 - bb_lower_20) / bb_mid_20

    df['bb_squeeze'] = (df['bb_width'] < df['bb_width_200'] * 0.5).astype(float)
    # 단기 BB가 장기 BB 대비 매우 좁음 → 큰 움직임 임박


    # ========================================================================
    # Bonus: Support/Resistance 200
    # ========================================================================

    # Major Support/Resistance levels (200-candle window)
    df['support_200'] = df['low'].rolling(window=200).min()
    df['resistance_200'] = df['high'].rolling(window=200).max()

    # Distance to major levels
    df['distance_to_support_200'] = (df['close'] - df['support_200']) / df['close']
    df['distance_to_resistance_200'] = (df['resistance_200'] - df['close']) / df['close']

    # Major level confluence (단기 + 장기 레벨 겹침)
    # Assumes support/resistance from baseline advanced features
    if 'distance_to_support_pct' in df.columns:
        df['major_support_confluence'] = (
            (abs(df['distance_to_support_pct']) < 0.01) &  # 단기 지지선 근처
            (abs(df['distance_to_support_200']) < 0.01)     # 장기 지지선도 근처
        ).astype(float)
    else:
        df['major_support_confluence'] = 0.0


    # ========================================================================
    # Bonus: Momentum 200
    # ========================================================================

    # Long-term momentum (200-candle return)
    df['momentum_200'] = df['close'].pct_change(200)

    # ROC (Rate of Change) 200
    df['roc_200'] = talib.ROC(df['close'], timeperiod=200)


    return df


def calculate_all_features_enhanced(df):
    """
    Calculate ALL features (baseline + long-term)

    Process:
    1. Calculate baseline features (107 features)
    2. Add long-term features (20 features)
    3. Total: 127 features

    Returns: DataFrame with enhanced features
    """

    print("Calculating enhanced features (baseline + long-term)...")

    # Step 1: Calculate baseline features (107 features)
    print("  1/2 Calculating baseline features (107)...")
    df_enhanced = calculate_all_features(df)

    # Step 2: Add long-term features (20 features)
    print("  2/2 Adding long-term features (20)...")
    df_enhanced = calculate_long_term_features(df_enhanced)

    # Clean NaN values
    df_enhanced = df_enhanced.dropna().reset_index(drop=True)

    print(f"✅ Enhanced features calculated: {len(df_enhanced.columns)} total features")
    print(f"   - Baseline features: 107")
    print(f"   - Long-term features: 20")
    print(f"   - Available candles: {len(df_enhanced):,}")

    return df_enhanced


def get_long_term_feature_list():
    """
    Return list of long-term feature names for model training
    """
    return [
        # MA/EMA 200 (5 features)
        'ma_200',
        'ema_200',
        'price_vs_ma200',
        'golden_cross',
        'death_cross',
        'ma_cross_strength',

        # Volume (2 features)
        'volume_ma_200',
        'volume_regime',

        # ATR/Volatility (2 features)
        'atr_200',
        'volatility_regime',

        # RSI (2 features)
        'rsi_200',
        'rsi_trend',

        # Bollinger Bands (4 features)
        'bb_mid_200',
        'bb_width_200',
        'bb_position_200',
        'bb_squeeze',

        # Support/Resistance (3 features)
        'support_200',
        'resistance_200',
        'distance_to_support_200',
        'distance_to_resistance_200',
        'major_support_confluence',

        # Momentum (2 features)
        'momentum_200',
        'roc_200',
    ]


if __name__ == "__main__":
    """
    Test feature calculation
    """
    print("="*80)
    print("ENHANCED FEATURE CALCULATION TEST")
    print("="*80)
    print()

    # Load sample data
    data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} candles")
    print()

    # Calculate enhanced features
    df_enhanced = calculate_all_features_enhanced(df)

    print()
    print("="*80)
    print("FEATURE SUMMARY")
    print("="*80)

    # Show long-term feature stats
    longterm_features = get_long_term_feature_list()

    print(f"\nLong-term features (20):")
    for feature in longterm_features:
        if feature in df_enhanced.columns:
            mean_val = df_enhanced[feature].mean()
            std_val = df_enhanced[feature].std()
            print(f"  ✅ {feature}: mean={mean_val:.4f}, std={std_val:.4f}")
        else:
            print(f"  ❌ {feature}: NOT FOUND")

    print()
    print(f"Total features: {len(df_enhanced.columns)}")
    print(f"Available candles: {len(df_enhanced):,}")
    print()
    print("✅ Enhanced feature calculation complete!")
