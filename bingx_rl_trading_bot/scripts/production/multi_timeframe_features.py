"""
Multi-Timeframe Features for Entry Models

목적:
- 현행 라벨링 (15min/0.3%) 유지하면서 성능 향상
- 다중 시간대 정보로 더 나은 15min 예측
- 단기 신호 (15min) + 중기 맥락 (1h, 4h) + 장기 추세 (1d)

전략:
- 학습 가능한 작업 유지 (15min/0.3%)
- 다중 시간대 피처로 정보 추가
- 단기/장기 예측 갭을 bridge

예상 효과:
- F1: 15.8% → 25-35% (+58-121%, realistic after leakage fix)
- Feature 추가: 33 → 67 (+34 multi-timeframe, 2 removed for leakage)
"""

import pandas as pd
import numpy as np
import ta


class MultiTimeframeFeatures:
    """
    Multi-Timeframe Technical Features

    5분봉 데이터에서 여러 시간대의 피처 계산:
    - 15min (3 candles)
    - 1h (12 candles)
    - 4h (48 candles)
    - 1d (288 candles)
    """

    def __init__(self):
        """Initialize timeframe windows"""
        self.windows = {
            '15min': 3,    # 3 candles = 15 minutes
            '1h': 12,      # 12 candles = 1 hour
            '4h': 48,      # 48 candles = 4 hours
            '1d': 288      # 288 candles = 1 day
        }

    def calculate_all_features(self, df):
        """
        모든 다중 시간대 피처 계산

        Args:
            df: OHLCV DataFrame

        Returns:
            df with additional multi-timeframe features
        """
        df = df.copy()

        # 1. Multi-timeframe RSI
        df = self.calculate_multi_rsi(df)

        # 2. Multi-timeframe MACD
        df = self.calculate_multi_macd(df)

        # 3. Multi-timeframe EMAs
        df = self.calculate_multi_ema(df)

        # 4. Multi-timeframe Bollinger position
        df = self.calculate_multi_bollinger_position(df)

        # 5. ATR and normalized volatility
        df = self.calculate_atr_features(df)

        # 6. Volatility regime
        df = self.calculate_volatility_regime(df)

        # 7. Trend strength multi-timeframe
        df = self.calculate_multi_trend_strength(df)

        # 8. Price momentum multi-timeframe
        df = self.calculate_multi_momentum(df)

        return df

    def calculate_multi_rsi(self, df):
        """
        다중 시간대 RSI

        원리:
        - 짧은 window: 빠른 신호, 노이즈 많음
        - 긴 window: 느린 신호, 신뢰도 높음
        - 여러 시간대 조합: 단기 + 중기 + 장기 추세 파악
        """
        # 기존 RSI (14 = 70분)
        if 'rsi' not in df.columns:
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)

        # Multi-timeframe RSI
        df['rsi_15min'] = ta.momentum.rsi(df['close'], window=3)   # 15 min
        df['rsi_1h'] = ta.momentum.rsi(df['close'], window=12)     # 1 hour
        df['rsi_4h'] = ta.momentum.rsi(df['close'], window=48)     # 4 hours
        df['rsi_1d'] = ta.momentum.rsi(df['close'], window=288)    # 1 day

        # RSI divergence across timeframes
        df['rsi_divergence_15min_1h'] = df['rsi_15min'] - df['rsi_1h']
        df['rsi_divergence_1h_4h'] = df['rsi_1h'] - df['rsi_4h']

        return df

    def calculate_multi_macd(self, df):
        """
        다중 시간대 MACD

        원리:
        - Standard MACD: 12-26-9 (5분봉 기준)
        - 1h MACD: 48-104 비율 유지
        - 4h MACD: 192-416 비율 유지
        """
        # 기존 MACD (5분봉 기준)
        if 'macd' not in df.columns:
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

        # 1h MACD (fast=48, slow=104, signal=36)
        macd_1h = ta.trend.MACD(df['close'], window_fast=48, window_slow=104, window_sign=36)
        df['macd_1h'] = macd_1h.macd()
        df['macd_1h_diff'] = macd_1h.macd_diff()

        # 4h MACD (fast=192, slow=416, signal=144)
        macd_4h = ta.trend.MACD(df['close'], window_fast=192, window_slow=416, window_sign=144)
        df['macd_4h'] = macd_4h.macd()
        df['macd_4h_diff'] = macd_4h.macd_diff()

        return df

    def calculate_multi_ema(self, df):
        """
        다중 시간대 EMA

        원리:
        - EMA는 트렌드 방향과 강도 표시
        - 가격이 여러 EMA 위에 있으면 강한 상승 추세
        - 가격이 여러 EMA 아래 있으면 강한 하락 추세
        """
        # Multi-timeframe EMAs
        df['ema_15min'] = ta.trend.ema_indicator(df['close'], window=3)    # 15 min
        df['ema_1h'] = ta.trend.ema_indicator(df['close'], window=12)      # 1 hour
        df['ema_4h'] = ta.trend.ema_indicator(df['close'], window=48)      # 4 hours
        df['ema_1d'] = ta.trend.ema_indicator(df['close'], window=288)     # 1 day

        # Price position vs EMAs
        df['price_vs_ema_1h'] = (df['close'] - df['ema_1h']) / df['ema_1h']
        df['price_vs_ema_4h'] = (df['close'] - df['ema_4h']) / df['ema_4h']
        df['price_vs_ema_1d'] = (df['close'] - df['ema_1d']) / df['ema_1d']

        # EMA alignment (trend confirmation)
        df['ema_alignment'] = 0
        df.loc[(df['ema_15min'] > df['ema_1h']) &
               (df['ema_1h'] > df['ema_4h']) &
               (df['ema_4h'] > df['ema_1d']), 'ema_alignment'] = 1  # Strong uptrend
        df.loc[(df['ema_15min'] < df['ema_1h']) &
               (df['ema_1h'] < df['ema_4h']) &
               (df['ema_4h'] < df['ema_1d']), 'ema_alignment'] = -1  # Strong downtrend

        return df

    def calculate_multi_bollinger_position(self, df):
        """
        다중 시간대 Bollinger Bands position

        원리:
        - 가격이 Bollinger Bands의 어디에 위치하는지
        - 여러 시간대에서 동시에 상단/하단 → 강한 신호
        """
        # 1h Bollinger
        bb_1h = ta.volatility.BollingerBands(df['close'], window=12, window_dev=2)
        df['bb_position_1h'] = (df['close'] - bb_1h.bollinger_lband()) / \
                               (bb_1h.bollinger_hband() - bb_1h.bollinger_lband())

        # 4h Bollinger
        bb_4h = ta.volatility.BollingerBands(df['close'], window=48, window_dev=2)
        df['bb_position_4h'] = (df['close'] - bb_4h.bollinger_lband()) / \
                               (bb_4h.bollinger_hband() - bb_4h.bollinger_lband())

        return df

    def calculate_atr_features(self, df):
        """
        ATR (Average True Range) and normalized volatility

        원리:
        - ATR: 가격 변동성 측정 (절대값)
        - Normalized: 가격 대비 상대적 변동성 (%)
        - 높은 변동성 = 큰 가격 이동 가능 = 기회 또는 리스크
        """
        # ATR 1h
        atr_1h = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=12)
        df['atr_1h'] = atr_1h.average_true_range()
        df['atr_1h_normalized'] = df['atr_1h'] / df['close']

        # ATR 4h
        atr_4h = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=48)
        df['atr_4h'] = atr_4h.average_true_range()
        df['atr_4h_normalized'] = df['atr_4h'] / df['close']

        # ATR percentile (high/med/low volatility)
        # REMOVED 2025-10-15: Feature leakage through global percentile statistics
        # df['atr_percentile_1h'] = df['atr_1h'].rolling(window=288).apply(
        #     lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        # )

        return df

    def calculate_volatility_regime(self, df):
        """
        변동성 regime 분류

        원리:
        - Low volatility: 조용한 시장, 작은 움직임
        - High volatility: 활발한 시장, 큰 움직임
        - Regime에 따라 전략 조정 필요
        """
        # 1h realized volatility
        df['realized_vol_1h'] = df['close'].pct_change().rolling(window=12).std()

        # 4h realized volatility
        df['realized_vol_4h'] = df['close'].pct_change().rolling(window=48).std()

        # Volatility regime (based on 1d percentile)
        # REMOVED 2025-10-15: Feature leakage through global percentile statistics
        # vol_percentile = df['realized_vol_4h'].rolling(window=288).apply(
        #     lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        # )
        #
        # df['volatility_regime'] = 0  # Medium
        # df.loc[vol_percentile < 0.33, 'volatility_regime'] = -1  # Low
        # df.loc[vol_percentile > 0.67, 'volatility_regime'] = 1   # High

        return df

    def calculate_multi_trend_strength(self, df):
        """
        다중 시간대 추세 강도

        원리:
        - ADX (Average Directional Index): 추세 강도 측정
        - 높은 ADX = 강한 추세 (상승/하락 무관)
        - 낮은 ADX = 약한 추세 (횡보)
        """
        # 1h trend strength (ADX)
        adx_1h = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=12)
        df['adx_1h'] = adx_1h.adx()
        df['adx_pos_1h'] = adx_1h.adx_pos()
        df['adx_neg_1h'] = adx_1h.adx_neg()

        # 4h trend strength (ADX)
        adx_4h = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=48)
        df['adx_4h'] = adx_4h.adx()

        # Trend direction (from +DI and -DI)
        df['trend_direction_1h'] = 0
        df.loc[df['adx_pos_1h'] > df['adx_neg_1h'], 'trend_direction_1h'] = 1   # Uptrend
        df.loc[df['adx_pos_1h'] < df['adx_neg_1h'], 'trend_direction_1h'] = -1  # Downtrend

        return df

    def calculate_multi_momentum(self, df):
        """
        다중 시간대 모멘텀

        원리:
        - ROC (Rate of Change): 가격 변화율
        - 여러 시간대의 모멘텀 비교로 가속/감속 판단
        """
        # Multi-timeframe momentum (ROC)
        df['momentum_15min'] = df['close'].pct_change(3)    # 15 min
        df['momentum_1h'] = df['close'].pct_change(12)      # 1 hour
        df['momentum_4h'] = df['close'].pct_change(48)      # 4 hours
        df['momentum_1d'] = df['close'].pct_change(288)     # 1 day

        # Momentum acceleration (15min vs 1h)
        df['momentum_accel_15min_1h'] = df['momentum_15min'] - df['momentum_1h']

        return df

    def get_feature_names(self):
        """
        Return list of all multi-timeframe feature names
        """
        features = [
            # Multi-timeframe RSI (6)
            'rsi_15min', 'rsi_1h', 'rsi_4h', 'rsi_1d',
            'rsi_divergence_15min_1h', 'rsi_divergence_1h_4h',

            # Multi-timeframe MACD (4)
            'macd_1h', 'macd_1h_diff',
            'macd_4h', 'macd_4h_diff',

            # Multi-timeframe EMAs (8)
            'ema_15min', 'ema_1h', 'ema_4h', 'ema_1d',
            'price_vs_ema_1h', 'price_vs_ema_4h', 'price_vs_ema_1d',
            'ema_alignment',

            # Bollinger position (2)
            'bb_position_1h', 'bb_position_4h',

            # ATR features (2, was 3 - removed atr_percentile_1h for leakage)
            'atr_1h_normalized', 'atr_4h_normalized',

            # Volatility regime (2, was 3 - removed volatility_regime for leakage)
            'realized_vol_1h', 'realized_vol_4h',

            # Trend strength (5)
            'adx_1h', 'adx_pos_1h', 'adx_neg_1h', 'adx_4h', 'trend_direction_1h',

            # Multi-momentum (5)
            'momentum_15min', 'momentum_1h', 'momentum_4h', 'momentum_1d',
            'momentum_accel_15min_1h'
        ]

        return features


def main():
    """Test multi-timeframe features"""
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"

    df = pd.read_csv(data_file)
    df = df.tail(2000)  # Last 2000 candles (~1 week)

    print("=" * 80)
    print("Multi-Timeframe Features Test")
    print("=" * 80)

    # Calculate multi-timeframe features
    mtf_features = MultiTimeframeFeatures()
    df = mtf_features.calculate_all_features(df)

    print(f"\nData shape: {df.shape}")
    print(f"Features added: {len(mtf_features.get_feature_names())}")

    # Show sample
    print("\n" + "=" * 80)
    print("Sample Multi-Timeframe Features (last 5 rows)")
    print("=" * 80)

    feature_cols = mtf_features.get_feature_names()
    print(df[feature_cols].tail().to_string())

    # Show statistics
    print("\n" + "=" * 80)
    print("Feature Statistics")
    print("=" * 80)

    print("\nRSI across timeframes:")
    print(df[['rsi_15min', 'rsi_1h', 'rsi_4h', 'rsi_1d']].describe())

    print("\nRealized volatility (1h and 4h):")
    print(df[['realized_vol_1h', 'realized_vol_4h']].describe())

    print("\nTrend direction distribution (1h):")
    print(df['trend_direction_1h'].value_counts())

    print("\nEMA alignment distribution:")
    print(df['ema_alignment'].value_counts())

    print("\n✅ Multi-timeframe features calculated successfully!")
    print(f"Total features: {len(feature_cols)}")
    print("\n비판적 분석:")
    print("  현행: 33 features (단일 시간대)")
    print(f"  개선: +{len(feature_cols)} multi-timeframe features (2 removed for leakage)")
    print(f"  합계: {33 + len(feature_cols)} features")
    print("\n예상 효과:")
    print("  - 단기 신호 (15min) + 중기 맥락 (1h, 4h) + 장기 추세 (1d)")
    print("  - F1 개선: 15.8% → 25-35% (realistic after leakage fix)")
    print("  - 학습 가능한 작업 유지하면서 정보 추가 (no leakage)")


if __name__ == "__main__":
    main()
