"""
Sell Signal Features - 매도 전용 Feature Engineering

Problem: 기존 features는 매수(buy) 신호에 최적화됨
         - Momentum strength → 매수
         - Trend following → 매수

Solution: 매도(sell) 신호 전용 features 설계
         - Momentum weakening → 매도
         - Divergences → 매도
         - Overbought → 매도
         - Volume exhaustion → 매도

Use Cases:
1. SHORT Entry (sell to open)
2. LONG Exit (sell to close)
"""

import pandas as pd
import numpy as np
import ta


class SellSignalFeatures:
    """
    매도 타이밍 예측을 위한 전용 Features

    Philosophy: "매수와 매도는 정반대가 아니라 다른 현상"
    - 매수: Momentum 시작, Trend 전환, Breakout
    - 매도: Momentum 약화, Divergence, Exhaustion
    """

    def __init__(self, divergence_lookback=14, exhaustion_lookback=20):
        """
        Args:
            divergence_lookback: Divergence 탐지 기간
            exhaustion_lookback: Exhaustion 판단 기간
        """
        self.divergence_lookback = divergence_lookback
        self.exhaustion_lookback = exhaustion_lookback

    def calculate_all_features(self, df):
        """모든 매도 신호 features 계산"""
        df = df.copy()

        # 1. Momentum Weakening
        df = self._calculate_momentum_weakening(df)

        # 2. Bearish Divergences
        df = self._calculate_divergences(df)

        # 3. Overbought Conditions
        df = self._calculate_overbought(df)

        # 4. Volume Exhaustion
        df = self._calculate_volume_exhaustion(df)

        # 5. Distribution Patterns
        df = self._calculate_distribution(df)

        # 6. Resistance Rejection
        df = self._calculate_resistance_rejection(df)

        # 7. Trend Exhaustion
        df = self._calculate_trend_exhaustion(df)

        # 8. Price Action Reversals
        df = self._calculate_reversal_patterns(df)

        return df

    def _calculate_momentum_weakening(self, df):
        """
        Momentum Weakening Features

        개념: Momentum이 약해지는 것을 포착 (절대값이 아니라 변화율)
        """
        # RSI weakening (RSI가 하락 중)
        df['rsi_weakening'] = df['rsi'].diff()  # Negative = weakening
        df['rsi_weakening_5'] = df['rsi'].diff(5)

        # RSI second derivative (가속도 감소)
        df['rsi_deceleration'] = df['rsi_weakening'].diff()

        # MACD weakening
        df['macd_weakening'] = df['macd'].diff()
        df['macd_histogram_weakening'] = df['macd_diff'].diff()

        # Stochastic weakening
        if 'stochrsi' in df.columns:
            df['stoch_weakening'] = df['stochrsi'].diff()

        # Price momentum weakening
        df['price_momentum_1'] = df['close'].pct_change(1)
        df['price_momentum_5'] = df['close'].pct_change(5)
        df['momentum_weakening'] = df['price_momentum_1'] - df['price_momentum_5']

        return df

    def _calculate_divergences(self, df):
        """
        Bearish Divergence Detection

        개념: 가격은 올라가지만 지표는 내려가는 경우 → 매도 신호
        """
        lookback = self.divergence_lookback

        # Price direction (higher highs)
        price_change = df['close'].diff(lookback)
        price_higher = price_change > 0

        # RSI Bearish Divergence
        rsi_change = df['rsi'].diff(lookback)
        rsi_lower = rsi_change < 0
        df['rsi_bearish_div'] = (price_higher & rsi_lower).astype(int)

        # MACD Bearish Divergence
        macd_change = df['macd'].diff(lookback)
        macd_lower = macd_change < 0
        df['macd_bearish_div'] = (price_higher & macd_lower).astype(int)

        # Stochastic Bearish Divergence
        if 'stochrsi' in df.columns:
            stoch_change = df['stochrsi'].diff(lookback)
            stoch_lower = stoch_change < 0
            df['stoch_bearish_div'] = (price_higher & stoch_lower).astype(int)
        else:
            df['stoch_bearish_div'] = 0

        # Divergence strength (magnitude)
        df['bearish_div_strength'] = (
            abs(price_change / df['close']) -
            abs(rsi_change / 100)
        ).fillna(0)

        return df

    def _calculate_overbought(self, df):
        """
        Overbought Extreme Conditions

        개념: 과매수 상태는 매도 신호 (매수 아님!)
        """
        # RSI overbought levels
        df['rsi_overbought_70'] = (df['rsi'] > 70).astype(int)
        df['rsi_overbought_80'] = (df['rsi'] > 80).astype(int)
        df['rsi_extreme_overbought'] = (df['rsi'] > 85).astype(int)

        # Stochastic overbought
        if 'stochrsi' in df.columns:
            df['stoch_overbought_80'] = (df['stochrsi'] > 0.8).astype(int)
            df['stoch_overbought_90'] = (df['stochrsi'] > 0.9).astype(int)
        else:
            df['stoch_overbought_80'] = 0
            df['stoch_overbought_90'] = 0

        # CCI extreme
        if 'cci' in df.columns:
            df['cci_overbought'] = (df['cci'] > 100).astype(int)
        else:
            df['cci_overbought'] = 0

        # Combined overbought score
        df['overbought_score'] = (
            df['rsi_overbought_70'] +
            df['stoch_overbought_80'] +
            df['cci_overbought']
        )

        # Overbought duration (how long in overbought zone)
        df['overbought_duration'] = 0
        in_overbought = df['rsi'] > 70
        duration = 0
        for i in range(len(df)):
            if in_overbought.iloc[i]:
                duration += 1
            else:
                duration = 0
            df.loc[df.index[i], 'overbought_duration'] = duration

        return df

    def _calculate_volume_exhaustion(self, df):
        """
        Volume Exhaustion Features

        개념: 거래량 감소 = 추세 약화 = 매도 신호
        """
        # Volume declining
        df['volume_declining'] = (df['volume'].diff() < 0).astype(int)
        df['volume_declining_3'] = (df['volume'].rolling(3).mean().diff() < 0).astype(int)

        # Volume ratio to average
        df['volume_vs_ma20'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_vs_ma50'] = df['volume'] / df['volume'].rolling(50).mean()

        # Volume decreasing while price increasing (distribution!)
        price_increasing = df['close'].diff() > 0
        volume_decreasing = df['volume'].diff() < 0
        df['volume_price_divergence'] = (price_increasing & volume_decreasing).astype(int)

        # Volume trend (linear regression slope)
        def volume_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            y = series.values
            if np.std(y) == 0:
                return 0
            slope = np.polyfit(x, y, 1)[0]
            return slope

        df['volume_trend_10'] = df['volume'].rolling(10).apply(volume_slope, raw=False)
        df['volume_trend_20'] = df['volume'].rolling(20).apply(volume_slope, raw=False)

        return df

    def _calculate_distribution(self, df):
        """
        Distribution Pattern Detection

        개념: 고점에서 횡보 = 매도 세력 분산 = 매도 신호
        """
        # Price range tightening at high levels
        high_20 = df['high'].rolling(20).max()
        current_high_pct = (df['high'] / high_20 - 1) * 100

        # Near 20-day high but range compressing
        near_high = current_high_pct > -2  # Within 2% of 20-day high

        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        atr_declining = atr.diff() < 0

        df['distribution_signal'] = (near_high & atr_declining).astype(int)

        # Volume spike at resistance (distribution)
        resistance_touch = current_high_pct > -1
        volume_spike = df['volume'] > df['volume'].rolling(20).mean() * 1.5
        df['volume_at_resistance'] = (resistance_touch & volume_spike).astype(int)

        return df

    def _calculate_resistance_rejection(self, df):
        """
        Resistance Rejection Features

        개념: 저항선 근처에서 반락 = 매도 신호
        """
        # Distance from recent high
        high_20 = df['high'].rolling(20).max()
        df['distance_from_high_pct'] = (df['close'] / high_20 - 1) * 100

        # Failed breakout (touched high but rejected)
        touched_high = df['high'] >= high_20 * 0.995  # Within 0.5%
        closed_below = df['close'] < high_20 * 0.99  # Closed 1% below
        df['failed_breakout'] = (touched_high & closed_below).astype(int)

        # Upper shadow at resistance (rejection candle)
        body_size = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        upper_shadow_ratio = upper_shadow / (body_size + 0.0001)  # Avoid div by 0

        at_resistance = df['distance_from_high_pct'] > -2
        long_upper_shadow = upper_shadow_ratio > 2
        df['rejection_candle'] = (at_resistance & long_upper_shadow).astype(int)

        return df

    def _calculate_trend_exhaustion(self, df):
        """
        Trend Exhaustion Features

        개념: 연속 상승 후 피로 = 매도 신호
        """
        # Consecutive up candles
        up_candle = (df['close'] > df['open']).astype(int)

        df['consecutive_up'] = 0
        count = 0
        for i in range(len(df)):
            if up_candle.iloc[i]:
                count += 1
            else:
                count = 0
            df.loc[df.index[i], 'consecutive_up'] = count

        # Extended move without pullback
        df['days_since_pullback'] = 0
        pullback_threshold = -0.01  # 1% decline
        days = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i] / df['close'].iloc[i-1] - 1 < pullback_threshold:
                days = 0
            else:
                days += 1
            df.loc[df.index[i], 'days_since_pullback'] = days

        # Distance from moving average (extended)
        df['distance_from_ema20'] = (df['close'] / df['ema_10'] - 1) * 100 if 'ema_10' in df.columns else 0
        df['distance_from_ema50'] = (df['close'] / df['sma_20'] - 1) * 100 if 'sma_20' in df.columns else 0

        # Parabolic move (accelerating gains = unsustainable)
        returns_5 = df['close'].pct_change(5)
        returns_10 = df['close'].pct_change(10)
        df['parabolic_move'] = (returns_5 > returns_10 * 1.5).astype(int)

        return df

    def _calculate_reversal_patterns(self, df):
        """
        Price Action Reversal Patterns

        개념: Candlestick reversal patterns = 매도 신호
        """
        # Shooting star (already in advanced features, but recalculate)
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']

        df['shooting_star_signal'] = (
            (upper_shadow > 2 * body) &
            (df['close'] < df['open']) &
            (lower_shadow < body * 0.3)
        ).astype(int)

        # Bearish engulfing
        prev_body = body.shift(1)
        prev_close = df['close'].shift(1)
        prev_open = df['open'].shift(1)

        current_bearish = df['close'] < df['open']
        prev_bullish = prev_close > prev_open
        engulfs = (df['open'] > prev_close) & (df['close'] < prev_open)

        df['bearish_engulfing'] = (current_bearish & prev_bullish & engulfs).astype(int)

        # Evening star (3-candle pattern)
        # Simplified version
        df['evening_star'] = 0  # Placeholder for complex pattern

        # Doji at top (indecision at high)
        small_body = body < (df['high'] - df['low']) * 0.1
        at_high = df['close'] > df['close'].rolling(20).mean() * 1.02
        df['doji_at_top'] = (small_body & at_high).astype(int)

        return df

    def get_feature_list(self):
        """Return list of all sell signal feature names"""
        return [
            # Momentum Weakening
            'rsi_weakening', 'rsi_weakening_5', 'rsi_deceleration',
            'macd_weakening', 'macd_histogram_weakening', 'stoch_weakening',
            'price_momentum_1', 'price_momentum_5', 'momentum_weakening',

            # Divergences
            'rsi_bearish_div', 'macd_bearish_div', 'stoch_bearish_div',
            'bearish_div_strength',

            # Overbought
            'rsi_overbought_70', 'rsi_overbought_80', 'rsi_extreme_overbought',
            'stoch_overbought_80', 'stoch_overbought_90', 'cci_overbought',
            'overbought_score', 'overbought_duration',

            # Volume Exhaustion
            'volume_declining', 'volume_declining_3',
            'volume_vs_ma20', 'volume_vs_ma50',
            'volume_price_divergence', 'volume_trend_10', 'volume_trend_20',

            # Distribution
            'distribution_signal', 'volume_at_resistance',

            # Resistance Rejection
            'distance_from_high_pct', 'failed_breakout', 'rejection_candle',

            # Trend Exhaustion
            'consecutive_up', 'days_since_pullback',
            'distance_from_ema20', 'distance_from_ema50', 'parabolic_move',

            # Reversal Patterns
            'shooting_star_signal', 'bearish_engulfing', 'evening_star', 'doji_at_top'
        ]


def test_sell_features():
    """Test sell signal features on sample data"""
    print("="*80)
    print("Testing Sell Signal Features")
    print("="*80)

    # Create sample data
    dates = pd.date_range('2025-01-01', periods=100, freq='5min')
    np.random.seed(42)

    # Simulate price data with an uptrend then reversal
    price = 100
    prices = []
    for i in range(100):
        if i < 70:
            # Uptrend with momentum
            price += np.random.normal(0.5, 1)
        else:
            # Reversal with weakening
            price += np.random.normal(-0.3, 1)
        prices.append(price)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.5)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.5)) for p in prices],
        'close': prices,
        'volume': [1000 + np.random.normal(0, 200) for _ in prices]
    })

    # Add basic indicators
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_diff'] = macd.macd_diff()
    df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['stochrsi'] = ta.momentum.stochrsi(df['close'], window=14)
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)

    # Calculate sell features
    sell_features = SellSignalFeatures()
    df = sell_features.calculate_all_features(df)

    # Show reversal period (candles 70-80)
    print("\nReversal Period Analysis (Candles 70-80):")
    print("="*80)

    reversal_df = df.iloc[70:81]

    key_features = [
        'close', 'rsi', 'rsi_weakening',
        'rsi_bearish_div', 'overbought_score',
        'volume_price_divergence', 'consecutive_up'
    ]

    print(reversal_df[key_features].to_string())

    # Count sell signals
    print("\n" + "="*80)
    print("Sell Signal Summary:")
    print("="*80)

    total_candles = len(df)

    signal_counts = {
        'RSI Overbought (>70)': df['rsi_overbought_70'].sum(),
        'Bearish Divergence': (df['rsi_bearish_div'] | df['macd_bearish_div']).sum(),
        'Volume Exhaustion': df['volume_price_divergence'].sum(),
        'Failed Breakout': df['failed_breakout'].sum(),
        'Shooting Star': df['shooting_star_signal'].sum(),
        'Trend Exhaustion (>5 consecutive up)': (df['consecutive_up'] > 5).sum()
    }

    for signal, count in signal_counts.items():
        pct = count / total_candles * 100
        print(f"  {signal:40s}: {count:3d} ({pct:5.1f}%)")

    print("\n✅ Sell features calculated successfully!")
    print(f"✅ Total features: {len(sell_features.get_feature_list())}")

    return df, sell_features


if __name__ == "__main__":
    df, sell_features = test_sell_features()
