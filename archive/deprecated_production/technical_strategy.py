"""
Technical Strategy Module

간단하고 효과적인 기술적 지표 기반 전략:
- EMA Cross (fast vs slow)
- RSI filter (overbought/oversold)
- ADX for trend strength
- Volatility regime detection

목적: XGBoost False signals 필터링
"""

import pandas as pd
import numpy as np
import ta


class TechnicalStrategy:
    """
    Simple but effective technical strategy

    Signal Types:
    - 'LONG': Strong bullish signal
    - 'HOLD': No clear signal or bearish
    - 'AVOID': Overbought or adverse conditions
    """

    def __init__(self,
                 ema_fast_period=9,
                 ema_slow_period=21,
                 rsi_period=14,
                 rsi_oversold=35,
                 rsi_overbought=70,
                 adx_period=14,
                 adx_trend_threshold=25,
                 min_volatility=0.0008):
        """
        Parameters:
            ema_fast_period: Fast EMA period (default 9 = 45 min)
            ema_slow_period: Slow EMA period (default 21 = 105 min)
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold level (buy signal)
            rsi_overbought: RSI overbought level (avoid signal)
            adx_period: ADX calculation period
            adx_trend_threshold: Minimum ADX for strong trend
            min_volatility: Minimum volatility threshold
        """
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.min_volatility = min_volatility

    def calculate_indicators(self, df):
        """Calculate all required technical indicators"""
        df = df.copy()

        # EMAs
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=self.ema_fast_period)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=self.ema_slow_period)

        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)

        # ADX (trend strength)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=self.adx_period)

        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(20).std()

        # MACD for additional confirmation
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        return df

    def get_signal(self, df, idx):
        """
        Get trading signal at specific index

        Returns:
            signal (str): 'LONG', 'HOLD', or 'AVOID'
            strength (float): Signal strength 0-1
            reason (str): Human-readable reason
        """
        if idx >= len(df):
            return 'HOLD', 0.0, 'Index out of bounds'

        row = df.iloc[idx]

        # Check for NaN values
        required_fields = ['ema_fast', 'ema_slow', 'rsi', 'adx', 'volatility']
        if any(pd.isna(row[field]) for field in required_fields):
            return 'HOLD', 0.0, 'Missing indicator values'

        ema_fast = row['ema_fast']
        ema_slow = row['ema_slow']
        rsi = row['rsi']
        adx = row['adx']
        volatility = row['volatility']
        macd_diff = row.get('macd_diff', 0)

        # Volatility check (market must be active)
        if volatility < self.min_volatility:
            return 'HOLD', 0.0, f'Low volatility ({volatility:.4f})'

        # Overbought check (avoid buying at top)
        if rsi > self.rsi_overbought:
            return 'AVOID', 0.0, f'Overbought RSI ({rsi:.1f})'

        # Calculate signal components
        signal_strength = 0.0
        reasons = []

        # 1. EMA Cross (strongest signal)
        if ema_fast > ema_slow:
            ema_diff_pct = ((ema_fast - ema_slow) / ema_slow) * 100
            signal_strength += min(ema_diff_pct * 50, 0.4)  # Max 0.4
            reasons.append(f'EMA+ ({ema_diff_pct:.2f}%)')
        else:
            return 'HOLD', 0.0, 'EMA bearish'

        # 2. RSI (oversold = opportunity)
        if rsi < self.rsi_oversold:
            signal_strength += 0.3
            reasons.append(f'RSI oversold ({rsi:.1f})')
        elif rsi < 50:
            signal_strength += 0.15
            reasons.append(f'RSI neutral ({rsi:.1f})')

        # 3. ADX (trend strength confirmation)
        if adx > self.adx_trend_threshold:
            signal_strength += 0.2
            reasons.append(f'Strong trend ADX={adx:.1f}')
        else:
            signal_strength += 0.1
            reasons.append(f'Weak trend ADX={adx:.1f}')

        # 4. MACD confirmation (optional boost)
        if not pd.isna(macd_diff) and macd_diff > 0:
            signal_strength += 0.1
            reasons.append('MACD+')

        # Classify signal
        if signal_strength >= 0.6:
            return 'LONG', signal_strength, ' | '.join(reasons)
        elif signal_strength >= 0.4:
            return 'LONG', signal_strength, ' | '.join(reasons)
        else:
            return 'HOLD', signal_strength, ' | '.join(reasons)

    def get_signals_batch(self, df):
        """
        Get signals for entire dataframe

        Returns:
            DataFrame with signal, strength, reason columns
        """
        df = self.calculate_indicators(df)

        signals = []
        strengths = []
        reasons = []

        for i in range(len(df)):
            signal, strength, reason = self.get_signal(df, i)
            signals.append(signal)
            strengths.append(strength)
            reasons.append(reason)

        result = df.copy()
        result['tech_signal'] = signals
        result['tech_strength'] = strengths
        result['tech_reason'] = reasons

        return result


if __name__ == '__main__':
    """Test the strategy"""
    from pathlib import Path

    # Load data
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"

    df = pd.read_csv(data_file)
    print(f"✅ Data loaded: {len(df)} candles")

    # Initialize strategy
    strategy = TechnicalStrategy()

    # Calculate indicators
    df = strategy.calculate_indicators(df)
    print(f"✅ Indicators calculated")

    # Get signals for recent data
    test_df = df.iloc[-1000:].reset_index(drop=True)
    results = strategy.get_signals_batch(test_df)

    # Analyze signals
    signal_counts = results['tech_signal'].value_counts()
    print(f"\n📊 Signal Distribution (last 1000 candles):")
    for signal, count in signal_counts.items():
        pct = (count / len(results)) * 100
        print(f"  {signal}: {count} ({pct:.1f}%)")

    # Show signal strength distribution
    long_signals = results[results['tech_signal'] == 'LONG']
    if len(long_signals) > 0:
        print(f"\n💪 LONG Signal Strength:")
        print(f"  Mean: {long_signals['tech_strength'].mean():.3f}")
        print(f"  Min: {long_signals['tech_strength'].min():.3f}")
        print(f"  Max: {long_signals['tech_strength'].max():.3f}")

        # Sample signals
        print(f"\n📋 Sample LONG Signals:")
        sample = long_signals.head(5)
        for idx, row in sample.iterrows():
            print(f"  [{idx}] Strength={row['tech_strength']:.3f}: {row['tech_reason']}")

    print(f"\n✅ Technical Strategy test complete!")
