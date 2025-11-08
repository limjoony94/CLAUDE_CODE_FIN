"""
Market Regime Filter for SHORT Trading

목적: SHORT 신호를 market regime에 따라 필터링

원리:
- SHORT는 특정 시장 상황에서만 효과적
- Uptrend에서 SHORT = 위험
- Downtrend/Range에서 SHORT = 기회

필터 조건:
1. Not in strong uptrend
2. Near resistance or in downtrend
3. Momentum not too strong
"""

import pandas as pd
import numpy as np
import ta


class MarketRegimeFilter:
    """
    Market Regime 필터

    SHORT 신호를 시장 상황에 맞게 필터링
    """

    def __init__(self):
        """Initialize filter"""
        pass

    def analyze_regime(self, df, current_idx):
        """
        Analyze current market regime

        Assumes indicators already calculated in get_regime_features()

        Returns:
            dict with regime information
        """
        # Current values (indicators already calculated)
        ema_21 = df['ema_21'].iloc[current_idx]
        ema_50 = df['ema_50'].iloc[current_idx]
        adx = df['adx'].iloc[current_idx]
        di_plus = df['di_plus'].iloc[current_idx]
        di_minus = df['di_minus'].iloc[current_idx]
        current_close = df['close'].iloc[current_idx]

        regime = {
            'ema_21': ema_21,
            'ema_50': ema_50,
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus,
            'current_close': current_close
        }

        # Trend classification
        if ema_21 > ema_50 * 1.02:
            regime['trend'] = 'strong_uptrend'
        elif ema_21 > ema_50:
            regime['trend'] = 'uptrend'
        elif ema_21 < ema_50 * 0.98:
            regime['trend'] = 'strong_downtrend'
        elif ema_21 < ema_50:
            regime['trend'] = 'downtrend'
        else:
            regime['trend'] = 'sideways'

        # Momentum strength
        if adx > 30:
            regime['momentum'] = 'strong'
        elif adx > 20:
            regime['momentum'] = 'moderate'
        else:
            regime['momentum'] = 'weak'

        # Directional bias
        if di_plus > di_minus * 1.5:
            regime['direction'] = 'bullish'
        elif di_minus > di_plus * 1.5:
            regime['direction'] = 'bearish'
        else:
            regime['direction'] = 'neutral'

        return regime

    def allow_short(self, df, current_idx):
        """
        Check if SHORT trading is allowed in current regime

        Args:
            df: DataFrame with price and indicators
            current_idx: Current candle index

        Returns:
            bool: True if SHORT allowed, False otherwise
            str: Reason for decision
        """
        regime = self.analyze_regime(df, current_idx)

        # Rule 1: No SHORT in strong uptrend
        if regime['trend'] == 'strong_uptrend':
            return False, "Strong uptrend - SHORT too risky"

        # Rule 2: No SHORT with strong bullish momentum
        if regime['momentum'] == 'strong' and regime['direction'] == 'bullish':
            return False, "Strong bullish momentum - wait for reversal"

        # Rule 3: No SHORT if price far above EMA 50 without resistance
        price_above_ema50_pct = (regime['current_close'] - regime['ema_50']) / regime['ema_50']

        if price_above_ema50_pct > 0.05:  # More than 5% above EMA 50
            # Check if near resistance (would be okay)
            if 'bb_upper' in df.columns:
                bb_upper = df['bb_upper'].iloc[current_idx]
                near_resistance = regime['current_close'] >= bb_upper * 0.99
                if not near_resistance:
                    return False, "Price too far above EMA 50 without resistance"

        # Rule 4: Prefer downtrend or sideways
        if regime['trend'] in ['downtrend', 'strong_downtrend', 'sideways']:
            return True, f"Good regime: {regime['trend']}"

        # Rule 5: Allow in uptrend only if near resistance
        if regime['trend'] == 'uptrend':
            if 'bb_upper' in df.columns:
                bb_upper = df['bb_upper'].iloc[current_idx]
                near_resistance = regime['current_close'] >= bb_upper * 0.99

                if near_resistance:
                    return True, "Uptrend but near resistance - reversal possible"
                else:
                    return False, "Uptrend without resistance - wait"

        # Default: allow
        return True, "Regime acceptable for SHORT"

    def get_regime_features(self, df):
        """
        Calculate regime features for entire DataFrame

        Returns:
            df with regime features added
        """
        # Ensure ALL indicators exist FIRST
        if 'ema_21' not in df.columns:
            df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        if 'ema_50' not in df.columns:
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()

        # Always calculate ADX indicators (even if adx exists, di_plus might not)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['di_plus'] = adx.adx_pos()
        df['di_minus'] = adx.adx_neg()

        # Initialize regime columns
        df['regime_trend'] = 'unknown'
        df['regime_momentum'] = 'unknown'
        df['regime_direction'] = 'unknown'
        df['short_allowed'] = 0
        df['short_filter_reason'] = ''

        # Calculate for each candle (after indicators are ready)
        for i in range(50, len(df)):  # Start after enough data
            regime = self.analyze_regime(df, i)
            allowed, reason = self.allow_short(df, i)

            df.loc[df.index[i], 'regime_trend'] = regime['trend']
            df.loc[df.index[i], 'regime_momentum'] = regime['momentum']
            df.loc[df.index[i], 'regime_direction'] = regime['direction']
            df.loc[df.index[i], 'short_allowed'] = 1 if allowed else 0
            df.loc[df.index[i], 'short_filter_reason'] = reason

        return df


def main():
    """Test market regime filter"""
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"

    df = pd.read_csv(data_file)
    df = df.tail(1000)  # Last 1000 candles

    print("="*80)
    print("Market Regime Filter Test")
    print("="*80)

    # Apply filter
    filter = MarketRegimeFilter()
    df = filter.get_regime_features(df)

    # Statistics
    print(f"\nData shape: {df.shape}")

    # Regime distribution
    print(f"\nRegime Distribution:")
    print(df['regime_trend'].value_counts())

    # SHORT allowed percentage
    short_allowed_pct = df['short_allowed'].sum() / len(df) * 100
    print(f"\nSHORT Allowed: {df['short_allowed'].sum()} / {len(df)} ({short_allowed_pct:.1f}%)")

    # Reasons
    print(f"\nFilter Reasons:")
    print(df['short_filter_reason'].value_counts())

    # Sample
    print(f"\n" + "="*80)
    print("Sample (last 10 candles)")
    print("="*80)
    cols = ['close', 'regime_trend', 'regime_momentum', 'regime_direction',
            'short_allowed', 'short_filter_reason']
    print(df[cols].tail(10).to_string())

    print("\n✅ Market regime filter working!")


if __name__ == "__main__":
    main()
