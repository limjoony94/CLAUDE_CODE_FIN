"""
Advanced Technical Features for Trading

전문 트레이더가 사용하는 고급 기술적 분석 추가:
1. Support/Resistance Detection (지지선/저항선)
2. Trend Line Analysis (추세선)
3. Divergence Detection (다이버전스)
4. Chart Pattern Recognition (패턴 인식)
5. Volume Profile Analysis (거래량 프로파일)

비판적 사고: "단일 캔들이 아닌, 여러 캔들의 패턴을 봐야 진짜 신호를 찾을 수 있다"
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import ta


class AdvancedTechnicalFeatures:
    """
    Advanced Technical Analysis Features

    전문 트레이더 수준의 기술적 분석 구현
    """

    def __init__(self, lookback_sr=50, lookback_trend=20):
        """
        Args:
            lookback_sr: Support/Resistance 탐지를 위한 lookback period
            lookback_trend: Trend line 계산을 위한 lookback period
        """
        self.lookback_sr = lookback_sr
        self.lookback_trend = lookback_trend

    def calculate_all_features(self, df):
        """
        모든 고급 기술적 분석 features 계산

        Args:
            df: OHLCV + indicators DataFrame

        Returns:
            df with additional advanced features
        """
        # 1. Support/Resistance
        df = self.detect_support_resistance(df)

        # 2. Trend Lines
        df = self.calculate_trend_lines(df)

        # 3. Divergence
        df = self.detect_divergences(df)

        # 4. Chart Patterns
        df = self.detect_chart_patterns(df)

        # 5. Volume Profile
        df = self.calculate_volume_profile(df)

        # 6. Price Action
        df = self.calculate_price_action_features(df)

        # 7. SHORT-specific Features (2025-10-15: Feature Bias Fix)
        df = self.calculate_short_specific_features(df)

        return df

    def detect_support_resistance(self, df):
        """
        Support/Resistance 탐지

        원리:
        - 과거 N 캔들에서 local minima (지지선) / maxima (저항선) 찾기
        - 현재 가격과의 거리 계산
        - 가까운 지지/저항 수준 features로 추가
        """
        # Find local extrema (5 candle window)
        local_min_idx = argrelextrema(df['low'].values, np.less_equal, order=5)[0]
        local_max_idx = argrelextrema(df['high'].values, np.greater_equal, order=5)[0]

        # Initialize columns
        df['nearest_support'] = np.nan
        df['nearest_resistance'] = np.nan
        df['distance_to_support_pct'] = np.nan
        df['distance_to_resistance_pct'] = np.nan
        df['num_support_touches'] = 0
        df['num_resistance_touches'] = 0

        # Calculate for each candle
        for i in range(len(df)):
            if i < self.lookback_sr:
                continue

            current_price = df['close'].iloc[i]

            # Get recent local minima (supports) within lookback
            recent_supports = df['low'].iloc[local_min_idx[
                (local_min_idx >= i - self.lookback_sr) & (local_min_idx < i)
            ]].values

            # Get recent local maxima (resistances) within lookback
            recent_resistances = df['high'].iloc[local_max_idx[
                (local_max_idx >= i - self.lookback_sr) & (local_max_idx < i)
            ]].values

            # Find nearest support (below current price)
            supports_below = recent_supports[recent_supports < current_price]
            if len(supports_below) > 0:
                nearest_support = supports_below.max()
                df.loc[df.index[i], 'nearest_support'] = nearest_support
                df.loc[df.index[i], 'distance_to_support_pct'] = \
                    ((current_price - nearest_support) / current_price) * 100

                # Count touches (within 0.5%)
                tolerance = nearest_support * 0.005
                df.loc[df.index[i], 'num_support_touches'] = \
                    np.sum((recent_supports >= nearest_support - tolerance) &
                           (recent_supports <= nearest_support + tolerance))

            # Find nearest resistance (above current price)
            resistances_above = recent_resistances[recent_resistances > current_price]
            if len(resistances_above) > 0:
                nearest_resistance = resistances_above.min()
                df.loc[df.index[i], 'nearest_resistance'] = nearest_resistance
                df.loc[df.index[i], 'distance_to_resistance_pct'] = \
                    ((nearest_resistance - current_price) / current_price) * 100

                # Count touches (within 0.5%)
                tolerance = nearest_resistance * 0.005
                df.loc[df.index[i], 'num_resistance_touches'] = \
                    np.sum((recent_resistances >= nearest_resistance - tolerance) &
                           (recent_resistances <= nearest_resistance + tolerance))

        return df

    def calculate_trend_lines(self, df):
        """
        Trend Line 계산

        원리:
        - 최근 N개 고점/저점을 linear regression으로 연결
        - 추세선 기울기 계산 (양수 = 상승, 음수 = 하락)
        - 현재 가격이 추세선 위/아래 있는지 확인
        """
        df['upper_trendline_slope'] = np.nan
        df['lower_trendline_slope'] = np.nan
        df['price_vs_upper_trendline_pct'] = np.nan
        df['price_vs_lower_trendline_pct'] = np.nan

        for i in range(self.lookback_trend, len(df)):
            # Recent data
            recent = df.iloc[i - self.lookback_trend:i]

            # Upper trend line (connect highs)
            highs = recent['high'].values
            x = np.arange(len(highs))

            # Linear regression for highs
            if len(x) > 1:
                upper_slope, upper_intercept = np.polyfit(x, highs, 1)
                upper_trendline_value = upper_slope * len(x) + upper_intercept

                df.loc[df.index[i], 'upper_trendline_slope'] = upper_slope
                df.loc[df.index[i], 'price_vs_upper_trendline_pct'] = \
                    ((df['close'].iloc[i] - upper_trendline_value) / df['close'].iloc[i]) * 100

            # Lower trend line (connect lows)
            lows = recent['low'].values

            # Linear regression for lows
            if len(x) > 1:
                lower_slope, lower_intercept = np.polyfit(x, lows, 1)
                lower_trendline_value = lower_slope * len(x) + lower_intercept

                df.loc[df.index[i], 'lower_trendline_slope'] = lower_slope
                df.loc[df.index[i], 'price_vs_lower_trendline_pct'] = \
                    ((df['close'].iloc[i] - lower_trendline_value) / df['close'].iloc[i]) * 100

        return df

    def detect_divergences(self, df):
        """
        Divergence 탐지 (가격 vs 지표)

        원리:
        - Bullish Divergence: 가격 lower low, RSI/MACD higher low
        - Bearish Divergence: 가격 higher high, RSI/MACD lower high

        Features:
        - rsi_bullish_divergence: 1 if detected, 0 otherwise
        - rsi_bearish_divergence: 1 if detected, 0 otherwise
        - macd_bullish_divergence: 1 if detected, 0 otherwise
        - macd_bearish_divergence: 1 if detected, 0 otherwise
        """
        # Ensure RSI and MACD exist
        if 'rsi' not in df.columns:
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        if 'macd' not in df.columns:
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()

        df['rsi_bullish_divergence'] = 0
        df['rsi_bearish_divergence'] = 0
        df['macd_bullish_divergence'] = 0
        df['macd_bearish_divergence'] = 0

        lookback_div = 10  # Look back 10 candles for divergence

        for i in range(lookback_div, len(df)):
            recent_price = df['close'].iloc[i - lookback_div:i]
            recent_rsi = df['rsi'].iloc[i - lookback_div:i]
            recent_macd = df['macd'].iloc[i - lookback_div:i]

            # Skip if not enough valid data
            if recent_rsi.isna().all() or recent_macd.isna().all():
                continue

            # Find local extrema in recent window
            price_min_idx = recent_price.idxmin()
            price_max_idx = recent_price.idxmax()

            # Skip if RSI/MACD has NaN values
            try:
                rsi_min_idx = recent_rsi.idxmin()
                rsi_max_idx = recent_rsi.idxmax()

                # Check for valid index (not NaN)
                if pd.isna(rsi_min_idx) or pd.isna(rsi_max_idx):
                    continue

            except (ValueError, KeyError):
                continue

            try:
                macd_min_idx = recent_macd.idxmin()
                macd_max_idx = recent_macd.idxmax()

                # Check for valid index (not NaN)
                if pd.isna(macd_min_idx) or pd.isna(macd_max_idx):
                    continue

            except (ValueError, KeyError):
                continue

            # RSI Bullish Divergence: Price lower low, RSI higher low
            try:
                if price_min_idx != rsi_min_idx:
                    if recent_price[price_min_idx] < recent_price.iloc[0] and \
                       recent_rsi[rsi_min_idx] > recent_rsi.iloc[0]:
                        df.loc[df.index[i], 'rsi_bullish_divergence'] = 1
            except (KeyError, IndexError):
                pass

            # RSI Bearish Divergence: Price higher high, RSI lower high
            try:
                if price_max_idx != rsi_max_idx:
                    if recent_price[price_max_idx] > recent_price.iloc[0] and \
                       recent_rsi[rsi_max_idx] < recent_rsi.iloc[0]:
                        df.loc[df.index[i], 'rsi_bearish_divergence'] = 1
            except (KeyError, IndexError):
                pass

            # MACD Bullish Divergence
            try:
                if price_min_idx != macd_min_idx:
                    if recent_price[price_min_idx] < recent_price.iloc[0] and \
                       recent_macd[macd_min_idx] > recent_macd.iloc[0]:
                        df.loc[df.index[i], 'macd_bullish_divergence'] = 1
            except (KeyError, IndexError):
                pass

            # MACD Bearish Divergence
            try:
                if price_max_idx != macd_max_idx:
                    if recent_price[price_max_idx] > recent_price.iloc[0] and \
                       recent_macd[macd_max_idx] < recent_macd.iloc[0]:
                        df.loc[df.index[i], 'macd_bearish_divergence'] = 1
            except (KeyError, IndexError):
                pass

        return df

    def detect_chart_patterns(self, df):
        """
        Chart Pattern 탐지

        간단한 패턴:
        1. Double Top: 두 번의 고점이 비슷한 가격
        2. Double Bottom: 두 번의 저점이 비슷한 가격
        3. Higher Highs & Higher Lows (상승 추세)
        4. Lower Highs & Lower Lows (하락 추세)
        """
        df['double_top'] = 0
        df['double_bottom'] = 0
        df['higher_highs_lows'] = 0
        df['lower_highs_lows'] = 0

        lookback_pattern = 20

        for i in range(lookback_pattern, len(df)):
            recent = df.iloc[i - lookback_pattern:i]

            # Find peaks and troughs
            peaks_idx = argrelextrema(recent['high'].values, np.greater, order=3)[0]
            troughs_idx = argrelextrema(recent['low'].values, np.less, order=3)[0]

            # Double Top: 2 peaks within 1% of each other
            if len(peaks_idx) >= 2:
                last_two_peaks = recent['high'].iloc[peaks_idx[-2:]].values
                if abs(last_two_peaks[1] - last_two_peaks[0]) / last_two_peaks[0] < 0.01:
                    df.loc[df.index[i], 'double_top'] = 1

            # Double Bottom: 2 troughs within 1% of each other
            if len(troughs_idx) >= 2:
                last_two_troughs = recent['low'].iloc[troughs_idx[-2:]].values
                if abs(last_two_troughs[1] - last_two_troughs[0]) / last_two_troughs[0] < 0.01:
                    df.loc[df.index[i], 'double_bottom'] = 1

            # Higher Highs & Higher Lows
            if len(peaks_idx) >= 2 and len(troughs_idx) >= 2:
                if recent['high'].iloc[peaks_idx[-1]] > recent['high'].iloc[peaks_idx[-2]] and \
                   recent['low'].iloc[troughs_idx[-1]] > recent['low'].iloc[troughs_idx[-2]]:
                    df.loc[df.index[i], 'higher_highs_lows'] = 1

            # Lower Highs & Lower Lows
            if len(peaks_idx) >= 2 and len(troughs_idx) >= 2:
                if recent['high'].iloc[peaks_idx[-1]] < recent['high'].iloc[peaks_idx[-2]] and \
                   recent['low'].iloc[troughs_idx[-1]] < recent['low'].iloc[troughs_idx[-2]]:
                    df.loc[df.index[i], 'lower_highs_lows'] = 1

        return df

    def calculate_volume_profile(self, df):
        """
        Volume Profile 분석

        원리:
        - 특정 가격대에서 거래된 volume 합계
        - High volume node (HVN): 많이 거래된 가격대 = 지지/저항
        - Low volume node (LVN): 적게 거래된 가격대 = 빠른 가격 이동
        """
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_price_correlation'] = df['volume'].rolling(20).corr(df['close'])

        # Price-volume trend (accumulation/distribution)
        df['price_volume_trend'] = 0

        for i in range(1, len(df)):
            price_change = df['close'].iloc[i] - df['close'].iloc[i-1]
            volume_change = df['volume'].iloc[i] - df['volume'].iloc[i-1]

            # Accumulation: price up + volume up
            if price_change > 0 and volume_change > 0:
                df.loc[df.index[i], 'price_volume_trend'] = 1
            # Distribution: price down + volume up
            elif price_change < 0 and volume_change > 0:
                df.loc[df.index[i], 'price_volume_trend'] = -1
            else:
                df.loc[df.index[i], 'price_volume_trend'] = 0

        return df

    def calculate_price_action_features(self, df):
        """
        Price Action Features

        캔들 패턴 기반:
        1. Bullish/Bearish Engulfing
        2. Hammer/Shooting Star
        3. Doji
        4. Long Upper/Lower Shadow
        """
        # Candle body and shadow sizes
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']

        # Relative sizes
        df['body_to_range_ratio'] = df['body_size'] / df['total_range']
        df['upper_shadow_ratio'] = df['upper_shadow'] / df['total_range']
        df['lower_shadow_ratio'] = df['lower_shadow'] / df['total_range']

        # Pattern detection
        df['bullish_engulfing'] = 0
        df['bearish_engulfing'] = 0
        df['hammer'] = 0
        df['shooting_star'] = 0
        df['doji'] = 0

        for i in range(1, len(df)):
            # Bullish Engulfing
            if df['close'].iloc[i-1] < df['open'].iloc[i-1] and \
               df['close'].iloc[i] > df['open'].iloc[i] and \
               df['open'].iloc[i] < df['close'].iloc[i-1] and \
               df['close'].iloc[i] > df['open'].iloc[i-1]:
                df.loc[df.index[i], 'bullish_engulfing'] = 1

            # Bearish Engulfing
            if df['close'].iloc[i-1] > df['open'].iloc[i-1] and \
               df['close'].iloc[i] < df['open'].iloc[i] and \
               df['open'].iloc[i] > df['close'].iloc[i-1] and \
               df['close'].iloc[i] < df['open'].iloc[i-1]:
                df.loc[df.index[i], 'bearish_engulfing'] = 1

            # Hammer: small body at top, long lower shadow
            if df['body_to_range_ratio'].iloc[i] < 0.3 and \
               df['lower_shadow_ratio'].iloc[i] > 0.6:
                df.loc[df.index[i], 'hammer'] = 1

            # Shooting Star: small body at bottom, long upper shadow
            if df['body_to_range_ratio'].iloc[i] < 0.3 and \
               df['upper_shadow_ratio'].iloc[i] > 0.6:
                df.loc[df.index[i], 'shooting_star'] = 1

            # Doji: very small body
            if df['body_to_range_ratio'].iloc[i] < 0.1:
                df.loc[df.index[i], 'doji'] = 1

        return df

    def calculate_short_specific_features(self, df):
        """
        SHORT-specific Features (2025-10-15: Feature Bias Fix)

        목적: LONG-biased features (8개) vs SHORT-biased features (3개) 불균형 해결

        SHORT 진입 조건 감지를 위한 하락 패턴 전용 지표:
        1. distance_from_recent_high_pct: 최근 고점 대비 거리 (과매수 지역)
        2. bearish_candle_count: 연속 하락 캔들 수
        3. red_candle_volume_ratio: 하락 캔들 거래량 비율 (매도 압력)
        4. selling_pressure_ratio: 위 꼬리 비율 (저항 반발)
        5. price_momentum_near_resistance: 저항선 근처 상승 모멘텀 (exhaustion)
        6. rsi_from_recent_peak: RSI 고점 대비 현재 위치
        7. consecutive_up_candles: 연속 상승 캔들 (과열 감지)
        """
        # 1. Distance from Recent High (과매수 감지)
        lookback_high = 20
        df['distance_from_recent_high_pct'] = np.nan

        for i in range(lookback_high, len(df)):
            recent_high = df['high'].iloc[i - lookback_high:i].max()
            current_price = df['close'].iloc[i]
            df.loc[df.index[i], 'distance_from_recent_high_pct'] = \
                ((current_price - recent_high) / recent_high) * 100

        # 2. Bearish Candle Count (연속 하락 캔들)
        df['bearish_candle_count'] = 0

        for i in range(1, len(df)):
            if df['close'].iloc[i] < df['open'].iloc[i]:  # Red candle
                if df.loc[df.index[i-1], 'bearish_candle_count'] > 0:
                    df.loc[df.index[i], 'bearish_candle_count'] = \
                        df.loc[df.index[i-1], 'bearish_candle_count'] + 1
                else:
                    df.loc[df.index[i], 'bearish_candle_count'] = 1
            else:
                df.loc[df.index[i], 'bearish_candle_count'] = 0

        # 3. Red Candle Volume Ratio (매도 압력)
        lookback_vol = 10
        df['red_candle_volume_ratio'] = 0.0

        for i in range(lookback_vol, len(df)):
            recent = df.iloc[i - lookback_vol:i]
            red_candles = recent[recent['close'] < recent['open']]

            if len(recent) > 0:
                red_volume = red_candles['volume'].sum()
                total_volume = recent['volume'].sum()
                df.loc[df.index[i], 'red_candle_volume_ratio'] = \
                    red_volume / total_volume if total_volume > 0 else 0

        # 4. Selling Pressure Ratio (위 꼬리 = 저항 반발)
        # Already have 'upper_shadow_ratio' from price_action_features
        # Add a version focused on strong rejection
        df['strong_selling_pressure'] = 0

        for i in range(len(df)):
            # Strong upper shadow (>50%) + small body (<30%) = rejection
            if df['upper_shadow_ratio'].iloc[i] > 0.5 and \
               df['body_to_range_ratio'].iloc[i] < 0.3:
                df.loc[df.index[i], 'strong_selling_pressure'] = 1

        # 5. Price Momentum Near Resistance (exhaustion 감지)
        df['price_momentum_near_resistance'] = 0.0

        for i in range(5, len(df)):
            # Only calculate if near resistance (within 1%)
            if 'distance_to_resistance_pct' in df.columns and \
               not pd.isna(df['distance_to_resistance_pct'].iloc[i]) and \
               df['distance_to_resistance_pct'].iloc[i] < 1.0:

                # Calculate recent momentum (5-candle price change)
                momentum = (df['close'].iloc[i] - df['close'].iloc[i-5]) / df['close'].iloc[i-5]
                df.loc[df.index[i], 'price_momentum_near_resistance'] = momentum

        # 6. RSI from Recent Peak (RSI 과열 감지)
        if 'rsi' not in df.columns:
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        lookback_rsi = 20
        df['rsi_from_recent_peak'] = np.nan

        for i in range(lookback_rsi, len(df)):
            recent_rsi = df['rsi'].iloc[i - lookback_rsi:i]
            if not recent_rsi.isna().all():
                rsi_peak = recent_rsi.max()
                current_rsi = df['rsi'].iloc[i]
                if not pd.isna(current_rsi):
                    df.loc[df.index[i], 'rsi_from_recent_peak'] = current_rsi - rsi_peak

        # 7. Consecutive Up Candles (과열 감지)
        df['consecutive_up_candles'] = 0

        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['open'].iloc[i]:  # Green candle
                if df.loc[df.index[i-1], 'consecutive_up_candles'] > 0:
                    df.loc[df.index[i], 'consecutive_up_candles'] = \
                        df.loc[df.index[i-1], 'consecutive_up_candles'] + 1
                else:
                    df.loc[df.index[i], 'consecutive_up_candles'] = 1
            else:
                df.loc[df.index[i], 'consecutive_up_candles'] = 0

        return df

    def get_feature_names(self):
        """
        Return list of all advanced feature names
        """
        features = [
            # Support/Resistance
            'distance_to_support_pct',
            'distance_to_resistance_pct',
            'num_support_touches',
            'num_resistance_touches',

            # Trend Lines
            'upper_trendline_slope',
            'lower_trendline_slope',
            'price_vs_upper_trendline_pct',
            'price_vs_lower_trendline_pct',

            # Divergence
            'rsi_bullish_divergence',
            'rsi_bearish_divergence',
            'macd_bullish_divergence',
            'macd_bearish_divergence',

            # Chart Patterns
            'double_top',
            'double_bottom',
            'higher_highs_lows',
            'lower_highs_lows',

            # Volume Profile
            'volume_ma_ratio',
            'volume_price_correlation',
            'price_volume_trend',

            # Price Action
            'body_to_range_ratio',
            'upper_shadow_ratio',
            'lower_shadow_ratio',
            'bullish_engulfing',
            'bearish_engulfing',
            'hammer',
            'shooting_star',
            'doji',

            # SHORT-specific Features (2025-10-15: Feature Bias Fix)
            'distance_from_recent_high_pct',
            'bearish_candle_count',
            'red_candle_volume_ratio',
            'strong_selling_pressure',
            'price_momentum_near_resistance',
            'rsi_from_recent_peak',
            'consecutive_up_candles'
        ]

        return features


def main():
    """Test advanced features"""
    # Load sample data
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"

    df = pd.read_csv(data_file)
    df = df.tail(300)  # Last 300 candles

    print("="*80)
    print("Advanced Technical Features Test")
    print("="*80)

    # Calculate advanced features
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)

    print(f"\nData shape: {df.shape}")
    print(f"Features added: {len(adv_features.get_feature_names())}")

    # Show sample
    print("\n" + "="*80)
    print("Sample Advanced Features (last 5 rows)")
    print("="*80)

    feature_cols = adv_features.get_feature_names()
    print(df[feature_cols].tail().to_string())

    # Show some interesting signals
    print("\n" + "="*80)
    print("Interesting Signals Detected")
    print("="*80)

    # Divergences
    bullish_div = df[df['rsi_bullish_divergence'] == 1]
    bearish_div = df[df['rsi_bearish_divergence'] == 1]

    print(f"\nRSI Bullish Divergences: {len(bullish_div)}")
    print(f"RSI Bearish Divergences: {len(bearish_div)}")

    # Patterns
    double_tops = df[df['double_top'] == 1]
    double_bottoms = df[df['double_bottom'] == 1]

    print(f"\nDouble Tops: {len(double_tops)}")
    print(f"Double Bottoms: {len(double_bottoms)}")

    # Candlestick patterns
    bullish_engulfing = df[df['bullish_engulfing'] == 1]
    hammer = df[df['hammer'] == 1]

    print(f"\nBullish Engulfing: {len(bullish_engulfing)}")
    print(f"Hammer: {len(hammer)}")

    print("\n✅ Advanced features calculated successfully!")
    print(f"Total features: {len(feature_cols)}")


if __name__ == "__main__":
    main()
