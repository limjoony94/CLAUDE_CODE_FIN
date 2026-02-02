"""
Advanced Trading Indicators Module
===================================

Modern, powerful indicators beyond traditional RSI/MACD/BB:
- Volume Profile (institutional activity levels)
- VWAP (institutional benchmark)
- Volume Flow Indicators (OBV, A/D, CMF, MFI)

Phase 1: Volume Profile + VWAP (11 features)
Phase 2: Volume Flow (13 features)
Phase 3: Ichimoku Cloud (10 features)
Phase 4: Channels & Force Index (13 features)

Created: 2025-10-23
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Tuple


# ========================================================================
# PHASE 1: VOLUME PROFILE (7 features)
# ========================================================================

def calculate_volume_profile(
    df: pd.DataFrame,
    lookback: int = 100,
    bins: int = 20,
    value_area_pct: float = 0.70
) -> pd.DataFrame:
    """
    Volume Profile - 가격대별 거래량 분포 분석

    기관 투자자들이 가장 활발히 거래한 가격대를 식별하여
    주요 지지/저항 레벨과 공정가치 영역을 파악합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV 데이터 (high, low, close, volume 필수)
    lookback : int
        분석 기간 (캔들 수, 기본값: 100)
    bins : int
        가격대 구간 수 (기본값: 20)
    value_area_pct : float
        Value Area 범위 (기본값: 0.70 = 70%)

    Returns:
    --------
    pd.DataFrame
        7개 Volume Profile 지표가 추가된 DataFrame

    Features Added:
    ---------------
    1. vp_poc (Point of Control): 최대 거래량 가격대
    2. vp_value_area_high: 거래량 70% 구간 상단
    3. vp_value_area_low: 거래량 70% 구간 하단
    4. vp_distance_to_poc_pct: POC까지 거리 (%)
    5. vp_in_value_area: Value Area 내부 여부 (1/0)
    6. vp_percentile: 현재가의 VP 상 위치 (0~1)
    7. vp_volume_imbalance: POC 위/아래 거래량 불균형

    Trading Signals:
    ----------------
    - Price near POC: 공정가치 영역 (양방향 가능)
    - Price above Value Area: 과매수 영역 (SHORT 기회)
    - Price below Value Area: 과매도 영역 (LONG 기회)
    - Volume Imbalance > 0.2: 매수세 우위
    - Volume Imbalance < -0.2: 매도세 우위
    """

    # Initialize output columns
    df['vp_poc'] = np.nan
    df['vp_value_area_high'] = np.nan
    df['vp_value_area_low'] = np.nan
    df['vp_distance_to_poc_pct'] = np.nan
    df['vp_in_value_area'] = 0.0
    df['vp_percentile'] = np.nan
    df['vp_volume_imbalance'] = np.nan

    # Calculate Volume Profile for each candle (rolling window)
    for i in range(lookback, len(df)):
        # Get recent window
        window_start = max(0, i - lookback)
        recent_df = df.iloc[window_start:i].copy()

        if len(recent_df) < 10:  # Skip if insufficient data
            continue

        # Price range
        price_min = recent_df['low'].min()
        price_max = recent_df['high'].max()
        price_range = price_max - price_min

        if price_range == 0:  # Skip if no price movement
            continue

        # Create price bins
        bin_size = price_range / bins
        volume_profile = np.zeros(bins)

        # Aggregate volume by price level
        # Each candle's volume is distributed across the bins it touched
        for _, row in recent_df.iterrows():
            candle_low = row['low']
            candle_high = row['high']
            candle_volume = row['volume']

            # Find bins this candle touches
            low_bin = int((candle_low - price_min) / bin_size)
            high_bin = int((candle_high - price_min) / bin_size)

            # Clamp to valid range
            low_bin = max(0, min(bins - 1, low_bin))
            high_bin = max(0, min(bins - 1, high_bin))

            # Distribute volume across touched bins
            touched_bins = high_bin - low_bin + 1
            for b in range(low_bin, high_bin + 1):
                volume_profile[b] += candle_volume / touched_bins

        # 1. POC (Point of Control) - Highest volume price level
        poc_bin = np.argmax(volume_profile)
        poc_price = price_min + (poc_bin + 0.5) * bin_size
        df.loc[df.index[i], 'vp_poc'] = poc_price

        # 2-3. Value Area (70% of total volume)
        total_volume = volume_profile.sum()
        target_volume = total_volume * value_area_pct

        # Start from POC and expand outward
        value_area_bins = {poc_bin}
        accumulated_volume = volume_profile[poc_bin]

        # Expand to bins with highest volume until reaching 70%
        remaining_bins = set(range(bins)) - value_area_bins
        while accumulated_volume < target_volume and remaining_bins:
            # Find adjacent bin with highest volume
            candidates = []
            for bin_idx in list(remaining_bins):
                # Check if adjacent to value area
                is_adjacent = any(abs(bin_idx - va_bin) == 1 for va_bin in value_area_bins)
                if is_adjacent:
                    candidates.append((volume_profile[bin_idx], bin_idx))

            if not candidates:
                # If no adjacent bins, add any remaining bin
                candidates = [(volume_profile[bin_idx], bin_idx) for bin_idx in remaining_bins]

            if not candidates:
                break

            # Add bin with highest volume
            _, best_bin = max(candidates)
            value_area_bins.add(best_bin)
            accumulated_volume += volume_profile[best_bin]
            remaining_bins.remove(best_bin)

        # Value Area High/Low
        value_area_high_bin = max(value_area_bins)
        value_area_low_bin = min(value_area_bins)
        value_area_high = price_min + (value_area_high_bin + 1) * bin_size
        value_area_low = price_min + value_area_low_bin * bin_size

        df.loc[df.index[i], 'vp_value_area_high'] = value_area_high
        df.loc[df.index[i], 'vp_value_area_low'] = value_area_low

        # 4. Distance to POC
        current_price = df.loc[df.index[i], 'close']
        distance_to_poc_pct = (current_price - poc_price) / current_price
        df.loc[df.index[i], 'vp_distance_to_poc_pct'] = distance_to_poc_pct

        # 5. In Value Area
        in_value_area = 1.0 if value_area_low <= current_price <= value_area_high else 0.0
        df.loc[df.index[i], 'vp_in_value_area'] = in_value_area

        # 6. Percentile (0-1)
        if price_range > 0:
            percentile = (current_price - price_min) / price_range
        else:
            percentile = 0.5
        df.loc[df.index[i], 'vp_percentile'] = percentile

        # 7. Volume Imbalance (POC 위/아래 거래량 비율)
        volume_above_poc = volume_profile[poc_bin:].sum()
        volume_below_poc = volume_profile[:poc_bin].sum()
        total_vol = volume_above_poc + volume_below_poc

        if total_vol > 0:
            volume_imbalance = (volume_above_poc - volume_below_poc) / total_vol
        else:
            volume_imbalance = 0.0
        df.loc[df.index[i], 'vp_volume_imbalance'] = volume_imbalance

    return df


# ========================================================================
# PHASE 1: VWAP (4 features)
# ========================================================================

def calculate_vwap(
    df: pd.DataFrame,
    period_candles: int = 288  # 5분봉 기준 1일 (24시간)
) -> pd.DataFrame:
    """
    VWAP (Volume-Weighted Average Price) - 거래량 가중 평균가

    기관 투자자들이 벤치마크로 사용하는 가격 지표.
    VWAP 위: 매수세 우위, VWAP 아래: 매도세 우위

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV 데이터 (high, low, close, volume 필수)
    period_candles : int
        VWAP 계산 기간 (캔들 수)
        기본값: 288 (5분봉 기준 1일)

    Returns:
    --------
    pd.DataFrame
        4개 VWAP 지표가 추가된 DataFrame

    Features Added:
    ---------------
    1. vwap: VWAP 가격
    2. vwap_distance_pct: VWAP까지 거리 (%)
    3. vwap_above: VWAP 위/아래 (1/0)
    4. vwap_band_position: VWAP 밴드 내 위치 (0~1)

    Trading Signals:
    ----------------
    - Price crosses above VWAP: 매수 신호
    - Price crosses below VWAP: 매도 신호
    - Price near upper band (>0.8): 과매수
    - Price near lower band (<0.2): 과매도
    """

    # Typical Price = (High + Low + Close) / 3
    typical_price = (df['high'] + df['low'] + df['close']) / 3

    # VWAP = Σ(Typical Price × Volume) / Σ(Volume)
    price_volume = typical_price * df['volume']
    vwap = price_volume.rolling(window=period_candles).sum() / df['volume'].rolling(window=period_candles).sum()

    df['vwap'] = vwap

    # Distance from VWAP (%)
    df['vwap_distance_pct'] = (df['close'] - df['vwap']) / df['close']

    # Above/Below VWAP (binary)
    df['vwap_above'] = (df['close'] > df['vwap']).astype(float)

    # VWAP Bands (±2 std)
    vwap_deviation = (df['close'] - df['vwap']).rolling(window=period_candles).std()
    vwap_upper = df['vwap'] + vwap_deviation * 2
    vwap_lower = df['vwap'] - vwap_deviation * 2

    # Band Position (0-1 scale)
    band_range = vwap_upper - vwap_lower
    df['vwap_band_position'] = np.where(
        band_range > 0,
        (df['close'] - vwap_lower) / band_range,
        0.5
    )
    # Clamp to [0, 1]
    df['vwap_band_position'] = df['vwap_band_position'].clip(0, 1)

    return df


# ========================================================================
# PHASE 2: VOLUME FLOW INDICATORS (13 features)
# ========================================================================

def calculate_volume_flow_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume Flow Indicators - 거래량 기반 자금 흐름 분석

    거래량과 가격 움직임의 관계를 분석하여
    매수/매도 압력과 자금 유입/유출을 측정합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV 데이터 (high, low, close, volume 필수)

    Returns:
    --------
    pd.DataFrame
        13개 Volume Flow 지표가 추가된 DataFrame

    Features Added:
    ---------------
    OBV (On-Balance Volume) - 3 features:
    1. obv: 누적 거래량 (상승시 +, 하락시 -)
    2. obv_ema: OBV의 지수이동평균
    3. obv_divergence: OBV-Price 다이버전스

    A/D Line (Accumulation/Distribution) - 3 features:
    4. ad_line: 누적 분배선
    5. ad_line_ema: A/D Line의 지수이동평균
    6. ad_oscillator: A/D Line - EMA

    CMF (Chaikin Money Flow) - 3 features:
    7. cmf_14: 14-period CMF
    8. cmf_21: 21-period CMF
    9. cmf_divergence: CMF-Price 다이버전스

    MFI (Money Flow Index) - 4 features:
    10. mfi: 14-period MFI (RSI의 거래량 버전)
    11. mfi_overbought: MFI > 80 (과매수)
    12. mfi_oversold: MFI < 20 (과매도)
    13. mfi_divergence: MFI-Price 다이버전스

    Trading Signals:
    ----------------
    - OBV rising + Price rising: 강한 상승 추세
    - OBV falling + Price rising: 약한 상승 (다이버전스, 하락 가능)
    - CMF > 0.15: 강한 매수 압력
    - CMF < -0.15: 강한 매도 압력
    - MFI > 80: 과매수 (조정 가능)
    - MFI < 20: 과매도 (반등 가능)
    """

    # ====================================================================
    # OBV (On-Balance Volume)
    # ====================================================================
    # OBV = Σ(volume if close > close_prev else -volume)
    close_change = df['close'].diff()
    obv = (df['volume'] * np.sign(close_change)).fillna(0).cumsum()

    df['obv'] = obv
    df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()

    # OBV Divergence (price up but OBV down = bearish)
    price_change_5 = df['close'].diff(5)
    obv_change_5 = df['obv'].diff(5)
    df['obv_divergence'] = (
        ((price_change_5 > 0) & (obv_change_5 < 0)).astype(float) -  # Bearish divergence
        ((price_change_5 < 0) & (obv_change_5 > 0)).astype(float)    # Bullish divergence
    )

    # ====================================================================
    # A/D Line (Accumulation/Distribution)
    # ====================================================================
    # MFM (Money Flow Multiplier) = ((C - L) - (H - C)) / (H - L)
    # MFV (Money Flow Volume) = MFM × Volume
    # A/D = Σ(MFV)

    high_low_range = df['high'] - df['low']
    mfm = np.where(
        high_low_range > 0,
        ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low_range,
        0
    )
    mfv = mfm * df['volume']
    ad_line = mfv.cumsum()

    df['ad_line'] = ad_line
    df['ad_line_ema'] = df['ad_line'].ewm(span=20, adjust=False).mean()
    df['ad_oscillator'] = df['ad_line'] - df['ad_line_ema']

    # ====================================================================
    # CMF (Chaikin Money Flow)
    # ====================================================================
    # CMF = Σ(MFV, period) / Σ(Volume, period)

    def calculate_cmf(period):
        mfv_sum = mfv.rolling(window=period).sum()
        volume_sum = df['volume'].rolling(window=period).sum()
        return np.where(volume_sum > 0, mfv_sum / volume_sum, 0)

    df['cmf_14'] = calculate_cmf(14)
    df['cmf_21'] = calculate_cmf(21)

    # CMF Divergence
    cmf_change_5 = df['cmf_14'].diff(5)
    df['cmf_divergence'] = (
        ((price_change_5 > 0) & (cmf_change_5 < 0)).astype(float) -  # Bearish
        ((price_change_5 < 0) & (cmf_change_5 > 0)).astype(float)    # Bullish
    )

    # ====================================================================
    # MFI (Money Flow Index) - RSI with Volume
    # ====================================================================
    # MFI = 100 - (100 / (1 + Money Flow Ratio))
    # Money Flow Ratio = (14-period Positive Money Flow) / (14-period Negative Money Flow)

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    # Positive/Negative money flow
    price_change = typical_price.diff()
    positive_flow = np.where(price_change > 0, money_flow, 0)
    negative_flow = np.where(price_change < 0, money_flow, 0)

    # 14-period sums
    positive_mf_sum = pd.Series(positive_flow).rolling(window=14).sum()
    negative_mf_sum = pd.Series(negative_flow).rolling(window=14).sum()

    # MFI
    mf_ratio = np.where(negative_mf_sum > 0, positive_mf_sum / negative_mf_sum, 100)
    mfi = 100 - (100 / (1 + mf_ratio))

    df['mfi'] = mfi
    df['mfi_overbought'] = (df['mfi'] > 80).astype(float)
    df['mfi_oversold'] = (df['mfi'] < 20).astype(float)

    # MFI Divergence
    mfi_change_5 = df['mfi'].diff(5)
    df['mfi_divergence'] = (
        ((price_change_5 > 0) & (mfi_change_5 < 0)).astype(float) -  # Bearish
        ((price_change_5 < 0) & (mfi_change_5 > 0)).astype(float)    # Bullish
    )

    return df


# ========================================================================
# ALL-IN-ONE: Calculate All Advanced Indicators
# ========================================================================

def calculate_all_advanced_indicators(
    df: pd.DataFrame,
    phase: str = 'phase1'
) -> pd.DataFrame:
    """
    Calculate all advanced indicators based on selected phase

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV 데이터
    phase : str
        'phase1': Volume Profile + VWAP (11 features)
        'phase2': Phase 1 + Volume Flow (24 features)
        'all': All available indicators

    Returns:
    --------
    pd.DataFrame
        Advanced indicators added to DataFrame
    """

    df_result = df.copy()

    # Phase 1: Volume Profile + VWAP (11 features)
    if phase in ['phase1', 'phase2', 'all']:
        print("  [1/3] Calculating Volume Profile (7 features)...")
        df_result = calculate_volume_profile(df_result)

        print("  [2/3] Calculating VWAP (4 features)...")
        df_result = calculate_vwap(df_result)

    # Phase 2: Volume Flow Indicators (13 features)
    if phase in ['phase2', 'all']:
        print("  [3/3] Calculating Volume Flow Indicators (13 features)...")
        df_result = calculate_volume_flow_indicators(df_result)

    # Clean NaN values
    initial_rows = len(df_result)
    df_result = df_result.dropna().reset_index(drop=True)
    final_rows = len(df_result)

    print(f"\n✅ Advanced indicators calculated:")
    if phase == 'phase1':
        print(f"   - Volume Profile: 7 features")
        print(f"   - VWAP: 4 features")
        print(f"   - Total: 11 features")
    elif phase == 'phase2':
        print(f"   - Volume Profile: 7 features")
        print(f"   - VWAP: 4 features")
        print(f"   - Volume Flow: 13 features")
        print(f"   - Total: 24 features")

    print(f"   - Rows: {initial_rows:,} → {final_rows:,} (lost {initial_rows - final_rows:,} due to lookback)")

    return df_result


# ========================================================================
# FEATURE LIST GETTERS
# ========================================================================

def get_volume_profile_features() -> list:
    """Return list of Volume Profile feature names"""
    return [
        'vp_poc',
        'vp_value_area_high',
        'vp_value_area_low',
        'vp_distance_to_poc_pct',
        'vp_in_value_area',
        'vp_percentile',
        'vp_volume_imbalance'
    ]


def get_vwap_features() -> list:
    """Return list of VWAP feature names"""
    return [
        'vwap',
        'vwap_distance_pct',
        'vwap_above',
        'vwap_band_position'
    ]


def get_volume_flow_features() -> list:
    """Return list of Volume Flow feature names"""
    return [
        'obv',
        'obv_ema',
        'obv_divergence',
        'ad_line',
        'ad_line_ema',
        'ad_oscillator',
        'cmf_14',
        'cmf_21',
        'cmf_divergence',
        'mfi',
        'mfi_overbought',
        'mfi_oversold',
        'mfi_divergence'
    ]


def get_all_advanced_features(phase: str = 'phase1') -> list:
    """
    Get list of all advanced feature names for specified phase

    Parameters:
    -----------
    phase : str
        'phase1': Volume Profile + VWAP (11 features)
        'phase2': Phase 1 + Volume Flow (24 features)

    Returns:
    --------
    list : Feature names
    """
    features = []

    if phase in ['phase1', 'phase2']:
        features.extend(get_volume_profile_features())
        features.extend(get_vwap_features())

    if phase == 'phase2':
        features.extend(get_volume_flow_features())

    return features


# ========================================================================
# TEST EXECUTION
# ========================================================================

if __name__ == "__main__":
    """
    Test advanced indicators calculation
    """
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("="*80)
    print("ADVANCED INDICATORS TEST")
    print("="*80)
    print()

    # Load sample data
    data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} candles")
    print()

    # Test Phase 1 (Volume Profile + VWAP)
    print("="*80)
    print("PHASE 1: Volume Profile + VWAP")
    print("="*80)
    print()

    df_phase1 = calculate_all_advanced_indicators(df, phase='phase1')

    print()
    print("Sample values (last 5 candles):")
    print("-"*80)

    phase1_features = get_all_advanced_features('phase1')
    print(df_phase1[phase1_features].tail())

    print()
    print("Feature statistics:")
    print("-"*80)
    for feature in phase1_features:
        mean_val = df_phase1[feature].mean()
        std_val = df_phase1[feature].std()
        print(f"  {feature:30s} mean={mean_val:10.4f}, std={std_val:10.4f}")

    print()
    print("="*80)
    print("PHASE 1 TEST COMPLETE")
    print("="*80)
