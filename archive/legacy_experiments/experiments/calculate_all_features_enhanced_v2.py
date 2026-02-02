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

    Total: ~165 features

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
    print("Step 1/4: Calculating baseline features (LONG + SHORT)...")
    df = calculate_all_features(df)
    print(f"  ✅ Baseline: 107 features")

    # Step 2: Long-term features (23)
    print("\nStep 2/4: Calculating long-term features (200-period)...")
    df = calculate_long_term_features(df)
    print(f"  ✅ Long-term: 23 features")

    # Step 3: Advanced indicators (11 or 24)
    print(f"\nStep 3/4: Calculating advanced indicators ({phase})...")
    df = calculate_all_advanced_indicators(df, phase=phase)

    if phase == 'phase1':
        print(f"  ✅ Advanced: 11 features (VP 7 + VWAP 4)")
    elif phase == 'phase2':
        print(f"  ✅ Advanced: 24 features (VP 7 + VWAP 4 + Volume Flow 13)")

    # Step 4: Engineered ratio features (24)
    print("\nStep 4/4: Calculating engineered ratio features...")
    df = calculate_advanced_ratio_features(df)
    print(f"  ✅ Ratios: 24 features")

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
        total_features = 107 + 23 + 11 + 24
    elif phase == 'phase2':
        total_features = 107 + 23 + 24 + 24

    print(f"\n✅ Total features: {total_features}")
    print(f"   - Baseline: 107")
    print(f"   - Long-term: 23")
    if phase == 'phase1':
        print(f"   - Advanced: 11 (VP + VWAP)")
    elif phase == 'phase2':
        print(f"   - Advanced: 24 (VP + VWAP + Volume Flow)")
    print(f"   - Engineered ratios: 24")

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
