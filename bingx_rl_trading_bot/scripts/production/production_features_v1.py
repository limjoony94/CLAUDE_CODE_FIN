"""
Enhanced Feature Calculation Pipeline V2
=========================================

Complete feature calculation including:
1. Baseline features (LONG + SHORT)
2. Long-term features (200-period)
3. Advanced indicators (Volume Profile + VWAP)
4. Ratio & normalization features (production pattern)

Total Features: ~165
- Baseline: 107 features
- Long-term: 23 features
- Advanced: 11 features (VP 7 + VWAP 4)
- Engineered ratios: ~24 features

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
from scripts.experiments.calculate_features_with_longterm import calculate_long_term_features
from scripts.indicators.advanced_indicators import calculate_all_advanced_indicators


def calculate_short_specific_features(df):
    """
    Calculate SHORT-specific features to reduce VWAP dependency

    Features designed to detect downtrends and bearish conditions
    independently of VWAP positioning.

    NEW FEATURES (10):
    1. downtrend_strength: EMA slope + consecutive lower highs
    2. bearish_momentum: Red candles + volume on down days
    3. price_distance_from_high_pct: % from recent 50-candle high
    4. price_below_ma200_pct: % below MA200
    5. price_below_ema12_pct: % below EMA12
    6. volatility_expansion_down: ATR increasing while price falls
    7. consecutive_red_candles: Count of consecutive down candles
    8. volume_on_down_days_ratio: Volume bias toward down days
    9. lower_highs_pattern: Detection of lower high pattern
    10. below_multiple_mas: Count of MAs price is below

    Created: 2025-11-04
    Purpose: Fix SHORT signal disappearance in Nov falling market
    """

    df = df.copy()

    # 1. Downtrend Strength (EMA slope)
    # Negative slope = downtrend
    if 'ema_12' in df.columns:
        df['ema12_slope'] = (df['ema_12'] - df['ema_12'].shift(5)) / df['ema_12'].shift(5)
        df['ema12_slope'] = df['ema12_slope'].fillna(0)
    else:
        df['ema12_slope'] = 0

    # 2. Bearish Momentum (consecutive red candles)
    df['is_red'] = (df['close'] < df['open']).astype(int)
    df['consecutive_red_candles'] = 0

    count = 0
    for i in range(len(df)):
        if df.iloc[i]['is_red'] == 1:
            count += 1
        else:
            count = 0
        df.iloc[i, df.columns.get_loc('consecutive_red_candles')] = count

    # 3. Price Distance from Recent High
    df['high_50'] = df['high'].rolling(window=50).max()
    df['price_distance_from_high_pct'] = (df['close'] - df['high_50']) / df['high_50']
    df['price_distance_from_high_pct'] = df['price_distance_from_high_pct'].fillna(0)

    # 4. Price Below MA200
    if 'ma_200' in df.columns:
        df['price_below_ma200_pct'] = (df['close'] - df['ma_200']) / df['ma_200']
        df['price_below_ma200_pct'] = df['price_below_ma200_pct'].fillna(0)
        df['is_below_ma200'] = (df['close'] < df['ma_200']).astype(int)
    else:
        df['price_below_ma200_pct'] = 0
        df['is_below_ma200'] = 0

    # 5. Price Below EMA12
    if 'ema_12' in df.columns:
        df['price_below_ema12_pct'] = (df['close'] - df['ema_12']) / df['ema_12']
        df['price_below_ema12_pct'] = df['price_below_ema12_pct'].fillna(0)
        df['is_below_ema12'] = (df['close'] < df['ema_12']).astype(int)
    else:
        df['price_below_ema12_pct'] = 0
        df['is_below_ema12'] = 0

    # 6. Volatility Expansion on Downside
    if 'atr' in df.columns:
        df['atr_change'] = df['atr'].pct_change(periods=5)
        df['price_change_5'] = df['close'].pct_change(periods=5)
        # Positive when ATR increases while price falls
        df['volatility_expansion_down'] = df['atr_change'] * (-df['price_change_5'])
        df['volatility_expansion_down'] = df['volatility_expansion_down'].fillna(0).clip(lower=0, upper=10)
    else:
        df['volatility_expansion_down'] = 0

    # 7. Volume on Down Days Ratio
    df['volume_on_red'] = df['volume'] * df['is_red']
    df['volume_on_red_sum'] = df['volume_on_red'].rolling(window=20).sum()
    df['volume_total_sum'] = df['volume'].rolling(window=20).sum()
    df['volume_on_down_days_ratio'] = df['volume_on_red_sum'] / df['volume_total_sum'].replace(0, 1)
    df['volume_on_down_days_ratio'] = df['volume_on_down_days_ratio'].fillna(0.5)

    # 8. Lower Highs Pattern (last 3 highs are descending)
    df['high_1'] = df['high'].shift(1)
    df['high_2'] = df['high'].shift(2)
    df['high_3'] = df['high'].shift(3)
    df['lower_highs_pattern'] = ((df['high'] < df['high_1']) &
                                  (df['high_1'] < df['high_2']) &
                                  (df['high_2'] < df['high_3'])).astype(int)

    # 9. Below Multiple MAs (count)
    df['below_multiple_mas'] = 0
    for ma_col in ['ma_50', 'ma_100', 'ma_200', 'ema_12', 'ema_26']:
        if ma_col in df.columns:
            df['below_multiple_mas'] += (df['close'] < df[ma_col]).astype(int)

    # 10. Downtrend Strength Score (composite)
    # Combine multiple bearish signals
    df['downtrend_strength'] = (
        df['ema12_slope'].clip(lower=-1, upper=0) * -10 +  # EMA slope
        (df['consecutive_red_candles'] / 10).clip(upper=1) +  # Red candles
        (df['price_distance_from_high_pct'] * -10).clip(upper=1) +  # Distance from high
        df['lower_highs_pattern'] * 0.3 +  # Lower highs
        (df['below_multiple_mas'] / 5)  # Below MAs
    )
    df['downtrend_strength'] = df['downtrend_strength'].clip(lower=0, upper=5)

    # Drop temporary columns
    df = df.drop(columns=['is_red', 'volume_on_red', 'volume_on_red_sum', 'volume_total_sum',
                          'high_1', 'high_2', 'high_3', 'high_50'], errors='ignore')

    return df


def calculate_advanced_ratio_features(df):
    """
    Calculate ratio and normalized features from advanced indicators

    Following production pattern:
    - Ratios for relative measurements
    - Normalized values (0-1 scale)
    - Momentum derivatives

    Returns 24 engineered features
    """
    features = {}

    # ========================================================================
    # VOLUME PROFILE RATIOS (8 features)
    # ========================================================================

    # 1. Value Area width (relative to POC)
    va_width = df['vp_value_area_high'] - df['vp_value_area_low']
    features['vp_value_area_width_pct'] = va_width / df['vp_poc']

    # 2. Price position within Value Area (0-1)
    # Already have vp_percentile, but this is VA-specific
    vp_position = np.where(
        df['vp_in_value_area'] == 1,
        (df['close'] - df['vp_value_area_low']) / (va_width + 1e-10),
        0.5  # Outside VA = neutral position
    )
    features['vp_price_in_va_position'] = vp_position

    # 3. POC stability (POC change rate)
    features['vp_poc_momentum'] = df['vp_poc'].pct_change(5)

    # 4. Value Area shift (trend detection)
    va_midpoint = (df['vp_value_area_high'] + df['vp_value_area_low']) / 2
    features['vp_va_midpoint_momentum'] = va_midpoint.pct_change(5)

    # 5. Volume imbalance extremes (binary flags)
    features['vp_strong_buy_pressure'] = (df['vp_volume_imbalance'] > 0.2).astype(float)
    features['vp_strong_sell_pressure'] = (df['vp_volume_imbalance'] < -0.2).astype(float)

    # 6. Distance to POC normalized by volatility
    atr_pct = df['atr'] / df['close'] if 'atr' in df.columns else 0.01
    features['vp_poc_distance_normalized'] = df['vp_distance_to_poc_pct'] / (atr_pct + 1e-10)

    # 7. Value Area breakout detection
    features['vp_above_va_breakout'] = (
        (df['close'] > df['vp_value_area_high']) &
        (df['close'].shift(1) <= df['vp_value_area_high'].shift(1))
    ).astype(float)

    features['vp_below_va_breakout'] = (
        (df['close'] < df['vp_value_area_low']) &
        (df['close'].shift(1) >= df['vp_value_area_low'].shift(1))
    ).astype(float)

    # ========================================================================
    # VWAP RATIOS (8 features)
    # ========================================================================

    # 1. VWAP momentum (rate of change)
    features['vwap_momentum'] = df['vwap'].pct_change(5)

    # 2. VWAP vs MA comparison
    if 'ma_20' in df.columns:
        features['vwap_vs_ma20'] = (df['vwap'] - df['ma_20']) / df['ma_20']
    else:
        ma_20 = df['close'].rolling(20).mean()
        features['vwap_vs_ma20'] = (df['vwap'] - ma_20) / ma_20

    # 3. VWAP distance normalized by volatility
    features['vwap_distance_normalized'] = df['vwap_distance_pct'] / (atr_pct + 1e-10)

    # 4. VWAP crossover signals
    features['vwap_cross_up'] = (
        (df['vwap_above'] == 1) &
        (df['vwap_above'].shift(1) == 0)
    ).astype(float)

    features['vwap_cross_down'] = (
        (df['vwap_above'] == 0) &
        (df['vwap_above'].shift(1) == 1)
    ).astype(float)

    # 5. VWAP band extremes (overbought/oversold)
    features['vwap_overbought'] = (df['vwap_band_position'] > 0.8).astype(float)
    features['vwap_oversold'] = (df['vwap_band_position'] < 0.2).astype(float)

    # 6. Price-VWAP divergence with volume
    price_change_5 = df['close'].pct_change(5)
    vwap_change_5 = df['vwap'].pct_change(5)
    volume_surge = df['volume'] / df['volume'].rolling(20).mean()

    features['vwap_bullish_divergence'] = (
        (price_change_5 < 0) &  # Price down
        (vwap_change_5 > 0) &   # VWAP up
        (volume_surge > 1.2)    # High volume
    ).astype(float)

    features['vwap_bearish_divergence'] = (
        (price_change_5 > 0) &  # Price up
        (vwap_change_5 < 0) &   # VWAP down
        (volume_surge > 1.2)    # High volume
    ).astype(float)

    # ========================================================================
    # COMBINED VP + VWAP SIGNALS (8 features)
    # ========================================================================

    # 1. Confluence: Both VP and VWAP agree on direction
    features['vp_vwap_bullish_confluence'] = (
        (df['close'] < df['vp_value_area_low']) &  # Below VP Value Area
        (df['vwap_above'] == 0) &                   # Below VWAP
        (df['vp_volume_imbalance'] > 0.1)           # Buy pressure
    ).astype(float)

    features['vp_vwap_bearish_confluence'] = (
        (df['close'] > df['vp_value_area_high']) &  # Above VP Value Area
        (df['vwap_above'] == 1) &                    # Above VWAP
        (df['vp_volume_imbalance'] < -0.1)           # Sell pressure
    ).astype(float)

    # 2. VP-VWAP alignment
    vp_midpoint = (df['vp_value_area_high'] + df['vp_value_area_low']) / 2
    features['vp_vwap_alignment'] = (df['vwap'] - vp_midpoint) / vp_midpoint

    # 3. Support/Resistance from VP at VWAP
    features['vwap_near_vp_support'] = (
        (abs(df['vwap'] - df['vp_value_area_low']) / df['close'] < 0.005) &
        (df['close'] > df['vwap'])
    ).astype(float)

    features['vwap_near_vp_resistance'] = (
        (abs(df['vwap'] - df['vp_value_area_high']) / df['close'] < 0.005) &
        (df['close'] < df['vwap'])
    ).astype(float)

    # 4. Institutional activity zone (POC near VWAP)
    features['institutional_activity_zone'] = (
        abs(df['vp_poc'] - df['vwap']) / df['close'] < 0.005
    ).astype(float)

    # 5. Trend strength: VP + VWAP momentum alignment
    vp_trend = np.sign(features['vp_poc_momentum'])
    vwap_trend = np.sign(features['vwap_momentum'])
    features['vp_vwap_trend_alignment'] = (vp_trend == vwap_trend).astype(float)

    # 6. Volume Profile efficiency (tight VA = strong consensus)
    features['vp_efficiency'] = 1.0 / (features['vp_value_area_width_pct'] + 0.01)

    # Combine into DataFrame
    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def calculate_all_features_enhanced_v2(df, phase='phase1'):
    """
    Calculate ALL features with ratios and normalization

    Pipeline:
    1. Baseline features (LONG + SHORT) - 107 features
    2. Long-term features (200-period) - 23 features
    3. Advanced indicators (VP + VWAP) - 11 features
    4. Engineered ratio features - 24 features
    5. SHORT-specific features - 10 features

    Total: ~175 features

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data
    phase : str
        'phase1': VP + VWAP only (baseline)
        'phase2': Add Volume Flow indicators

    Returns:
    --------
    pd.DataFrame
        Enhanced features with ratios/normalization
    """

    print("="*80)
    print("ENHANCED FEATURE CALCULATION V2")
    print("="*80)
    print()

    initial_rows = len(df)

    # Step 1: Baseline features (107)
    print("Step 1/5: Calculating baseline features (LONG + SHORT)...")
    df = calculate_all_features(df)
    print(f"  ✅ Baseline: 107 features")

    # Step 2: Long-term features (23)
    print("\nStep 2/5: Calculating long-term features (200-period)...")
    df = calculate_long_term_features(df)
    print(f"  ✅ Long-term: 23 features")

    # Step 3: Advanced indicators (11 or 24)
    print(f"\nStep 3/5: Calculating advanced indicators ({phase})...")
    df = calculate_all_advanced_indicators(df, phase=phase)

    if phase == 'phase1':
        print(f"  ✅ Advanced: 11 features (VP 7 + VWAP 4)")
    elif phase == 'phase2':
        print(f"  ✅ Advanced: 24 features (VP 7 + VWAP 4 + Volume Flow 13)")

    # Step 4: Engineered ratio features (24)
    print("\nStep 4/5: Calculating engineered ratio features...")
    df = calculate_advanced_ratio_features(df)
    print(f"  ✅ Ratios: 24 features")

    # Step 5: SHORT-specific features (10)
    print("\nStep 5/5: Calculating SHORT-specific features...")
    df = calculate_short_specific_features(df)
    print(f"  ✅ SHORT-specific: 10 features")

    # Clean NaN values
    print("\nCleaning NaN values...")
    df = df.dropna().reset_index(drop=True)
    final_rows = len(df)

    # Summary
    print()
    print("="*80)
    print("FEATURE CALCULATION COMPLETE")
    print("="*80)

    if phase == 'phase1':
        total_features = 107 + 23 + 11 + 24 + 10
    elif phase == 'phase2':
        total_features = 107 + 23 + 24 + 24 + 10

    print(f"\n✅ Total features: {total_features}")
    print(f"   - Baseline: 107")
    print(f"   - Long-term: 23")
    if phase == 'phase1':
        print(f"   - Advanced: 11 (VP + VWAP)")
    elif phase == 'phase2':
        print(f"   - Advanced: 24 (VP + VWAP + Volume Flow)")
    print(f"   - Engineered ratios: 24")
    print(f"   - SHORT-specific: 10")

    print(f"\n✅ Data coverage: {final_rows:,} / {initial_rows:,} rows")
    print(f"   Lost {initial_rows - final_rows:,} rows due to lookback periods")
    print()

    return df


def get_all_feature_names(phase='phase1'):
    """
    Get list of all feature column names

    Returns names in order:
    1. Baseline features
    2. Long-term features
    3. Advanced indicators
    4. Engineered ratios
    """
    from scripts.indicators.advanced_indicators import get_all_advanced_features

    # This would need to import from each module
    # For now, return empty list (implement if needed)
    print("⚠️  Feature name extraction not implemented yet")
    print("   Use df.columns to get all feature names")
    return []


# ========================================================================
# TEST EXECUTION
# ========================================================================

if __name__ == "__main__":
    """
    Test enhanced feature calculation
    """
    print("="*80)
    print("ENHANCED FEATURES V2 TEST")
    print("="*80)
    print()

    # Load sample data
    data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} candles")
    print()

    # Calculate all features
    df_enhanced = calculate_all_features_enhanced_v2(df, phase='phase1')

    # Show sample
    print("="*80)
    print("SAMPLE FEATURES (last 3 rows)")
    print("="*80)
    print()

    # Show engineered ratio features
    ratio_features = [
        'vp_value_area_width_pct',
        'vp_poc_momentum',
        'vwap_momentum',
        'vwap_vs_ma20',
        'vp_vwap_bullish_confluence',
        'vp_vwap_bearish_confluence'
    ]

    print("Engineered Ratio Features:")
    print(df_enhanced[ratio_features].tail(3))
    print()

    # Statistics
    print("="*80)
    print("FEATURE STATISTICS")
    print("="*80)
    print()

    for feature in ratio_features:
        mean_val = df_enhanced[feature].mean()
        std_val = df_enhanced[feature].std()
        min_val = df_enhanced[feature].min()
        max_val = df_enhanced[feature].max()
        print(f"{feature:35s} mean={mean_val:8.4f}, std={std_val:8.4f}, min={min_val:8.4f}, max={max_val:8.4f}")

    print()
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)


# ===================================
# PRODUCTION VERSION - DO NOT MODIFY
# Last synced: 2025-11-03 from calculate_all_features_enhanced_v2.py
# Feature count: 85 (LONG entry), 79 (SHORT entry)
# ===================================
