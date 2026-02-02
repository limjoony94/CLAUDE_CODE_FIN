"""
SHORT-Specific Technical Features

하락 신호 전문 감지 features:
1. Bearish Divergence (가격↑ RSI/MACD↓)
2. Distribution Pressure (매도 압력)
3. Resistance Rejection (저항 거부)
4. Bearish Candlestick Patterns
5. Overbought Reversal Signals
6. Selling Volume Analysis
7. Momentum Exhaustion

목표: SHORT win rate 60%+ 달성을 위한 전문 하락 신호 features
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import ta


class ShortSpecificFeatures:
    """
    SHORT 전용 Technical Features

    기존 features는 "Will price increase?" 예측용
    이 features는 "Will price decrease?" 예측 전문
    """

    def __init__(self, lookback=20):
        """
        Args:
            lookback: 신호 탐지를 위한 lookback period
        """
        self.lookback = lookback

    def calculate_all_features(self, df):
        """
        모든 SHORT 전용 features 계산

        Args:
            df: OHLCV + indicators DataFrame

        Returns:
            df with SHORT-specific features
        """
        # Ensure basic indicators exist
        df = self._ensure_basic_indicators(df)

        # 1. Bearish Divergence (강력한 하락 신호)
        df = self.detect_bearish_divergence(df)

        # 2. Distribution Pressure (매도 압력)
        df = self.calculate_distribution_pressure(df)

        # 3. Resistance Rejection (저항 거부)
        df = self.detect_resistance_rejection(df)

        # 4. Bearish Candlestick Patterns
        df = self.detect_bearish_patterns(df)

        # 5. Overbought Reversal Signals
        df = self.detect_overbought_reversal(df)

        # 6. Selling Volume Analysis
        df = self.calculate_selling_volume(df)

        # 7. Momentum Exhaustion (상승 모멘텀 소진)
        df = self.detect_momentum_exhaustion(df)

        # 8. Price Action Weakness
        df = self.detect_price_weakness(df)

        return df

    def _ensure_basic_indicators(self, df):
        """기본 지표 확인 및 계산"""
        if 'rsi' not in df.columns:
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        if 'macd' not in df.columns:
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()

        if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()

        if 'ema_21' not in df.columns:
            df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()

        if 'ema_50' not in df.columns:
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()

        if 'atr' not in df.columns:
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

        # Stochastic RSI (overbought 감지용)
        if 'stoch_rsi' not in df.columns:
            stoch = ta.momentum.StochRSIIndicator(df['close'])
            df['stoch_rsi'] = stoch.stochrsi()
            df['stoch_rsi_k'] = stoch.stochrsi_k()
            df['stoch_rsi_d'] = stoch.stochrsi_d()

        return df

    def detect_bearish_divergence(self, df):
        """
        Bearish Divergence 탐지

        원리: 가격은 higher high, RSI/MACD는 lower high
        = 상승 모멘텀 약화, 하락 가능성 증가
        """
        df['bearish_div_rsi'] = 0
        df['bearish_div_macd'] = 0
        df['bearish_div_strength'] = 0.0

        for i in range(self.lookback, len(df)):
            recent_price = df['close'].iloc[i - self.lookback:i]
            recent_rsi = df['rsi'].iloc[i - self.lookback:i]
            recent_macd = df['macd'].iloc[i - self.lookback:i]

            # Skip if insufficient data
            if recent_rsi.isna().all() or recent_macd.isna().all():
                continue

            # Find peaks in recent window
            try:
                price_peaks = argrelextrema(recent_price.values, np.greater, order=2)[0]
                rsi_peaks = argrelextrema(recent_rsi.values, np.greater, order=2)[0]
                macd_peaks = argrelextrema(recent_macd.values, np.greater, order=2)[0]

                # Need at least 2 peaks for divergence
                if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                    # Get last 2 peaks
                    price_peak1, price_peak2 = recent_price.iloc[price_peaks[-2]], recent_price.iloc[price_peaks[-1]]
                    rsi_peak1, rsi_peak2 = recent_rsi.iloc[rsi_peaks[-2]], recent_rsi.iloc[rsi_peaks[-1]]

                    # Bearish divergence: price higher high, RSI lower high
                    if price_peak2 > price_peak1 and rsi_peak2 < rsi_peak1:
                        df.loc[df.index[i], 'bearish_div_rsi'] = 1

                        # Calculate divergence strength (price increase vs RSI decrease)
                        price_change = (price_peak2 - price_peak1) / price_peak1
                        rsi_change = (rsi_peak1 - rsi_peak2) / 100  # RSI normalized to 0-1
                        df.loc[df.index[i], 'bearish_div_strength'] = price_change + rsi_change

                # MACD divergence
                if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
                    price_peak1, price_peak2 = recent_price.iloc[price_peaks[-2]], recent_price.iloc[price_peaks[-1]]
                    macd_peak1, macd_peak2 = recent_macd.iloc[macd_peaks[-2]], recent_macd.iloc[macd_peaks[-1]]

                    if price_peak2 > price_peak1 and macd_peak2 < macd_peak1:
                        df.loc[df.index[i], 'bearish_div_macd'] = 1

            except (ValueError, IndexError, KeyError):
                pass

        return df

    def calculate_distribution_pressure(self, df):
        """
        Distribution Pressure (매도 압력) 계산

        원리:
        - 가격 상승 시 volume 증가 = Accumulation (매수)
        - 가격 하락 시 volume 증가 = Distribution (매도)
        """
        df['distribution_pressure'] = 0.0
        df['sell_volume_ratio'] = 0.0
        df['high_volume_down_candles'] = 0

        # Calculate for each candle
        for i in range(1, len(df)):
            price_change = df['close'].iloc[i] - df['close'].iloc[i-1]
            volume = df['volume'].iloc[i]
            volume_avg = df['volume'].iloc[max(0, i-20):i].mean()

            # Distribution: price down + high volume
            if price_change < 0 and volume > volume_avg * 1.5:
                df.loc[df.index[i], 'distribution_pressure'] = abs(price_change / df['close'].iloc[i]) * (volume / volume_avg)
                df.loc[df.index[i], 'high_volume_down_candles'] = 1

        # Calculate sell volume ratio (rolling 10 candles)
        for i in range(10, len(df)):
            recent = df.iloc[i-10:i]

            # Volume on down candles
            sell_volume = recent[recent['close'] < recent['open']]['volume'].sum()
            total_volume = recent['volume'].sum()

            if total_volume > 0:
                df.loc[df.index[i], 'sell_volume_ratio'] = sell_volume / total_volume

        return df

    def detect_resistance_rejection(self, df):
        """
        Resistance Rejection (저항 거부) 탐지

        원리:
        - 가격이 저항선에 도달했다가 거부됨 = 하락 신호
        - BB Upper 돌파 후 거부
        - Recent high 근처 rejection
        """
        df['bb_upper_rejection'] = 0
        df['recent_high_rejection'] = 0
        df['resistance_rejection_strength'] = 0.0

        for i in range(self.lookback, len(df)):
            current_close = df['close'].iloc[i]
            current_high = df['high'].iloc[i]
            bb_upper = df['bb_upper'].iloc[i]

            # BB Upper Rejection
            # High touched or crossed BB upper, but close below
            if current_high >= bb_upper * 0.998 and current_close < bb_upper * 0.995:
                df.loc[df.index[i], 'bb_upper_rejection'] = 1

                # Rejection strength (how far it fell from upper)
                rejection_strength = (bb_upper - current_close) / current_close
                df.loc[df.index[i], 'resistance_rejection_strength'] = rejection_strength

            # Recent High Rejection
            # Price reached near recent high (within 0.5%) but closed lower
            recent_highs = df['high'].iloc[i - self.lookback:i]
            max_recent_high = recent_highs.max()

            if current_high >= max_recent_high * 0.995 and current_close < max_recent_high * 0.99:
                df.loc[df.index[i], 'recent_high_rejection'] = 1

        return df

    def detect_bearish_patterns(self, df):
        """
        Bearish Candlestick Patterns

        패턴:
        1. Shooting Star: 작은 body, 긴 upper shadow
        2. Evening Star: 3-candle reversal pattern
        3. Bearish Engulfing: 하락 캔들이 상승 캔들 완전 삼킴
        4. Dark Cloud Cover: 상승 후 하락 시작
        """
        # Calculate candle properties
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']

        # Avoid division by zero
        df['upper_shadow_ratio'] = np.where(
            df['total_range'] > 0,
            df['upper_shadow'] / df['total_range'],
            0
        )
        df['body_to_range_ratio'] = np.where(
            df['total_range'] > 0,
            df['body_size'] / df['total_range'],
            0
        )

        # Initialize pattern columns
        df['shooting_star'] = 0
        df['evening_star'] = 0
        df['bearish_engulfing'] = 0
        df['dark_cloud_cover'] = 0

        for i in range(2, len(df)):
            # Shooting Star: small body at bottom, long upper shadow
            if (df['body_to_range_ratio'].iloc[i] < 0.3 and
                df['upper_shadow_ratio'].iloc[i] > 0.6 and
                df['close'].iloc[i] < df['open'].iloc[i]):  # Red candle
                df.loc[df.index[i], 'shooting_star'] = 1

            # Bearish Engulfing
            prev_bullish = df['close'].iloc[i-1] > df['open'].iloc[i-1]
            curr_bearish = df['close'].iloc[i] < df['open'].iloc[i]

            if (prev_bullish and curr_bearish and
                df['open'].iloc[i] > df['close'].iloc[i-1] and
                df['close'].iloc[i] < df['open'].iloc[i-1]):
                df.loc[df.index[i], 'bearish_engulfing'] = 1

            # Evening Star (3-candle pattern)
            if i >= 2:
                candle1_bullish = df['close'].iloc[i-2] > df['open'].iloc[i-2]
                candle2_small = df['body_to_range_ratio'].iloc[i-1] < 0.3
                candle3_bearish = df['close'].iloc[i] < df['open'].iloc[i]

                if (candle1_bullish and candle2_small and candle3_bearish and
                    df['close'].iloc[i] < df['close'].iloc[i-2]):
                    df.loc[df.index[i], 'evening_star'] = 1

            # Dark Cloud Cover
            prev_bullish = df['close'].iloc[i-1] > df['open'].iloc[i-1]
            curr_bearish = df['close'].iloc[i] < df['open'].iloc[i]

            if (prev_bullish and curr_bearish and
                df['open'].iloc[i] > df['high'].iloc[i-1] and  # Gap up
                df['close'].iloc[i] < (df['open'].iloc[i-1] + df['close'].iloc[i-1]) / 2):  # Close below 50%
                df.loc[df.index[i], 'dark_cloud_cover'] = 1

        return df

    def detect_overbought_reversal(self, df):
        """
        Overbought Reversal 신호

        원리:
        - RSI > 70 (overbought)
        - Stochastic RSI > 0.8 (overbought)
        - Price above BB upper
        - 이후 reversal 신호 (crossover 등)
        """
        df['rsi_overbought'] = 0
        df['stoch_rsi_overbought'] = 0
        df['rsi_bearish_cross'] = 0
        df['stoch_bearish_cross'] = 0
        df['overbought_reversal_signal'] = 0

        for i in range(1, len(df)):
            # RSI overbought (> 70)
            if df['rsi'].iloc[i] > 70:
                df.loc[df.index[i], 'rsi_overbought'] = 1

                # Bearish cross (RSI starts declining from overbought)
                if i > 0 and df['rsi'].iloc[i] < df['rsi'].iloc[i-1]:
                    df.loc[df.index[i], 'rsi_bearish_cross'] = 1

            # Stochastic RSI overbought
            if df['stoch_rsi'].iloc[i] > 0.8:
                df.loc[df.index[i], 'stoch_rsi_overbought'] = 1

                # Bearish cross (K crosses below D from overbought)
                if (i > 0 and
                    df['stoch_rsi_k'].iloc[i-1] > df['stoch_rsi_d'].iloc[i-1] and
                    df['stoch_rsi_k'].iloc[i] < df['stoch_rsi_d'].iloc[i]):
                    df.loc[df.index[i], 'stoch_bearish_cross'] = 1

            # Combined overbought reversal signal
            overbought_count = (df['rsi_overbought'].iloc[i] +
                              df['stoch_rsi_overbought'].iloc[i])

            reversal_count = (df['rsi_bearish_cross'].iloc[i] +
                            df['stoch_bearish_cross'].iloc[i])

            if overbought_count >= 1 and reversal_count >= 1:
                df.loc[df.index[i], 'overbought_reversal_signal'] = 1

        return df

    def calculate_selling_volume(self, df):
        """
        Selling Volume Analysis

        원리:
        - 하락 캔들의 volume이 상승 캔들보다 크면 매도 압력
        - CMF (Chaikin Money Flow) negative = selling
        """
        # CMF (Chaikin Money Flow)
        cmf = ta.volume.ChaikinMoneyFlowIndicator(
            df['high'], df['low'], df['close'], df['volume'], window=20
        )
        df['cmf'] = cmf.chaikin_money_flow()
        df['cmf_negative'] = (df['cmf'] < 0).astype(int)

        # Selling volume dominance
        df['selling_volume_dominance'] = 0.0

        for i in range(10, len(df)):
            recent = df.iloc[i-10:i]

            # Volume on down candles vs up candles
            down_volume = recent[recent['close'] < recent['open']]['volume'].sum()
            up_volume = recent[recent['close'] >= recent['open']]['volume'].sum()

            if up_volume + down_volume > 0:
                dominance = (down_volume - up_volume) / (down_volume + up_volume)
                df.loc[df.index[i], 'selling_volume_dominance'] = max(0, dominance)  # Only positive values

        return df

    def detect_momentum_exhaustion(self, df):
        """
        Momentum Exhaustion (상승 모멘텀 소진)

        원리:
        - 가격은 오르는데 momentum 감소
        - MACD histogram 감소
        - ADX 감소 (추세 약화)
        """
        # MACD histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_histogram_decreasing'] = 0

        # ADX (Average Directional Index) - trend strength
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['adx_decreasing'] = 0

        df['momentum_exhaustion'] = 0

        for i in range(5, len(df)):
            # Price increasing but MACD histogram decreasing
            price_increasing = df['close'].iloc[i] > df['close'].iloc[i-5]
            macd_hist_decreasing = df['macd_histogram'].iloc[i] < df['macd_histogram'].iloc[i-5]

            if price_increasing and macd_hist_decreasing:
                df.loc[df.index[i], 'macd_histogram_decreasing'] = 1

            # ADX decreasing (trend weakening)
            if df['adx'].iloc[i] < df['adx'].iloc[i-5]:
                df.loc[df.index[i], 'adx_decreasing'] = 1

            # Combined momentum exhaustion
            if (df['macd_histogram_decreasing'].iloc[i] == 1 or
                df['adx_decreasing'].iloc[i] == 1):
                df.loc[df.index[i], 'momentum_exhaustion'] = 1

        return df

    def detect_price_weakness(self, df):
        """
        Price Action Weakness

        원리:
        - Lower highs (고점 낮아짐)
        - Weak bounces (반등 약함)
        - Consistent lower closes
        """
        df['lower_highs'] = 0
        df['weak_bounce'] = 0
        df['consistent_lower_closes'] = 0

        for i in range(10, len(df)):
            recent_highs = df['high'].iloc[i-10:i]

            # Lower highs (최근 고점이 이전 고점보다 낮음)
            if len(recent_highs) >= 3:
                peaks = argrelextrema(recent_highs.values, np.greater, order=2)[0]
                if len(peaks) >= 2:
                    if recent_highs.iloc[peaks[-1]] < recent_highs.iloc[peaks[-2]]:
                        df.loc[df.index[i], 'lower_highs'] = 1

            # Weak bounce (반등이 약함)
            recent_lows = df['low'].iloc[i-5:i]
            recent_closes = df['close'].iloc[i-5:i]

            if len(recent_lows) > 0:
                lowest = recent_lows.min()
                current_close = df['close'].iloc[i]
                bounce_pct = (current_close - lowest) / lowest

                # Bounce less than 0.5% = weak
                if bounce_pct < 0.005:
                    df.loc[df.index[i], 'weak_bounce'] = 1

            # Consistent lower closes (5 out of last 7 closes are lower)
            recent_closes = df['close'].iloc[i-7:i]
            lower_closes = sum(recent_closes.iloc[j] < recent_closes.iloc[j-1] for j in range(1, len(recent_closes)))

            if lower_closes >= 5:
                df.loc[df.index[i], 'consistent_lower_closes'] = 1

        return df

    def get_feature_names(self):
        """
        Return list of all SHORT-specific feature names
        """
        features = [
            # Bearish Divergence
            'bearish_div_rsi',
            'bearish_div_macd',
            'bearish_div_strength',

            # Distribution Pressure
            'distribution_pressure',
            'sell_volume_ratio',
            'high_volume_down_candles',

            # Resistance Rejection
            'bb_upper_rejection',
            'recent_high_rejection',
            'resistance_rejection_strength',

            # Bearish Patterns
            'shooting_star',
            'evening_star',
            'bearish_engulfing',
            'dark_cloud_cover',
            'upper_shadow_ratio',

            # Overbought Reversal
            'rsi_overbought',
            'stoch_rsi_overbought',
            'rsi_bearish_cross',
            'stoch_bearish_cross',
            'overbought_reversal_signal',

            # Selling Volume
            'cmf',
            'cmf_negative',
            'selling_volume_dominance',

            # Momentum Exhaustion
            'macd_histogram',
            'macd_histogram_decreasing',
            'adx',
            'adx_decreasing',
            'momentum_exhaustion',

            # Price Weakness
            'lower_highs',
            'weak_bounce',
            'consistent_lower_closes'
        ]

        return features


def main():
    """Test SHORT-specific features"""
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"

    df = pd.read_csv(data_file)
    df = df.tail(500)  # Last 500 candles

    print("="*80)
    print("SHORT-Specific Features Test")
    print("="*80)

    # Calculate SHORT-specific features
    short_features = ShortSpecificFeatures(lookback=20)
    df = short_features.calculate_all_features(df)

    print(f"\nData shape: {df.shape}")
    print(f"Features added: {len(short_features.get_feature_names())}")

    # Show sample
    print("\n" + "="*80)
    print("Sample SHORT Features (last 5 rows)")
    print("="*80)

    feature_cols = short_features.get_feature_names()
    available_cols = [col for col in feature_cols if col in df.columns]
    print(df[available_cols].tail().to_string())

    # Show signal counts
    print("\n" + "="*80)
    print("SHORT Signals Detected (last 500 candles)")
    print("="*80)

    # Bearish divergence
    bearish_div_rsi = df[df['bearish_div_rsi'] == 1]
    bearish_div_macd = df[df['bearish_div_macd'] == 1]
    print(f"\nBearish Divergence RSI: {len(bearish_div_rsi)}")
    print(f"Bearish Divergence MACD: {len(bearish_div_macd)}")

    # Distribution pressure
    high_dist = df[df['distribution_pressure'] > 0.01]
    print(f"\nHigh Distribution Pressure: {len(high_dist)}")

    # Resistance rejection
    bb_reject = df[df['bb_upper_rejection'] == 1]
    recent_high_reject = df[df['recent_high_rejection'] == 1]
    print(f"\nBB Upper Rejection: {len(bb_reject)}")
    print(f"Recent High Rejection: {len(recent_high_reject)}")

    # Bearish patterns
    shooting_stars = df[df['shooting_star'] == 1]
    bearish_engulfing = df[df['bearish_engulfing'] == 1]
    print(f"\nShooting Star: {len(shooting_stars)}")
    print(f"Bearish Engulfing: {len(bearish_engulfing)}")

    # Overbought reversal
    overbought_reversal = df[df['overbought_reversal_signal'] == 1]
    print(f"\nOverbought Reversal Signal: {len(overbought_reversal)}")

    # Momentum exhaustion
    momentum_exhaust = df[df['momentum_exhaustion'] == 1]
    print(f"\nMomentum Exhaustion: {len(momentum_exhaust)}")

    print("\n✅ SHORT-specific features calculated successfully!")
    print(f"Total features: {len(available_cols)}")


if __name__ == "__main__":
    main()
