"""
Regime-Specific Hybrid Strategy

핵심 통찰:
- Bull 시장: Aggressive threshold (기회 놓치지 않기)
- Bear 시장: Conservative threshold (리스크 관리)
- Sideways: Moderate threshold (균형)

기존 문제:
- 모든 시장 상태에 동일 threshold → Bull에서 -4% ~ -5% 실패
- Conservative는 Bear/Sideways에서 좋지만 Bull에서 참패

해결:
- Real-time regime detection
- Regime별 최적 threshold 적용
"""

import pandas as pd
import numpy as np
import ta


class RegimeDetector:
    """
    실시간 시장 상태 감지

    방법:
    1. 단기 vs 중기 trend (EMA cross)
    2. Trend strength (ADX)
    3. Volatility level
    """

    def __init__(self,
                 fast_period=20,  # 100 minutes
                 slow_period=50,  # 250 minutes (4+ hours)
                 adx_period=14,
                 bull_threshold=0.015,  # 1.5% above slow EMA
                 bear_threshold=-0.010):  # 1.0% below slow EMA
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.adx_period = adx_period
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold

    def calculate_indicators(self, df):
        """Calculate regime detection indicators"""
        df = df.copy()

        # Trend EMAs
        df['ema_fast_regime'] = ta.trend.ema_indicator(df['close'], window=self.fast_period)
        df['ema_slow_regime'] = ta.trend.ema_indicator(df['close'], window=self.slow_period)

        # Price position vs slow EMA
        df['price_position'] = (df['close'] - df['ema_slow_regime']) / df['ema_slow_regime']

        # ADX for trend strength
        df['adx_regime'] = ta.trend.adx(df['high'], df['low'], df['close'], window=self.adx_period)

        # Volatility
        df['volatility_regime'] = df['close'].pct_change().rolling(20).std()

        return df

    def detect_regime(self, df, idx):
        """
        Detect market regime at specific index

        Returns:
            regime (str): 'Bull', 'Bear', or 'Sideways'
            confidence (float): 0-1
        """
        if idx >= len(df):
            return 'Sideways', 0.0

        row = df.iloc[idx]

        # Check for NaN
        if pd.isna(row['price_position']) or pd.isna(row['adx_regime']):
            return 'Sideways', 0.0

        price_position = row['price_position']
        ema_fast = row['ema_fast_regime']
        ema_slow = row['ema_slow_regime']
        adx = row['adx_regime']

        # Strong uptrend detection
        if price_position > self.bull_threshold and ema_fast > ema_slow:
            if adx > 25:  # Strong trend
                return 'Bull', 0.9
            else:
                return 'Bull', 0.6

        # Strong downtrend detection
        elif price_position < self.bear_threshold and ema_fast < ema_slow:
            if adx > 25:
                return 'Bear', 0.9
            else:
                return 'Bear', 0.6

        # Sideways (no clear trend)
        else:
            if adx < 20:
                return 'Sideways', 0.8
            else:
                return 'Sideways', 0.5


class RegimeSpecificHybridStrategy:
    """
    Regime-Specific Hybrid Strategy

    Different thresholds for different market regimes:
    - Bull: Aggressive (don't miss opportunities)
    - Bear: Conservative (manage risk)
    - Sideways: Moderate (balanced)
    """

    def __init__(self, xgboost_model, feature_columns, technical_strategy, regime_detector):
        self.xgboost = xgboost_model
        self.feature_columns = feature_columns
        self.technical = technical_strategy
        self.regime_detector = regime_detector

        # Regime-specific thresholds
        self.thresholds = {
            'Bull': {
                'xgb_strong': 0.4,  # Aggressive!
                'xgb_moderate': 0.3,
                'tech_strength': 0.55
            },
            'Bear': {
                'xgb_strong': 0.65,  # Conservative
                'xgb_moderate': 0.55,
                'tech_strength': 0.7
            },
            'Sideways': {
                'xgb_strong': 0.5,  # Moderate
                'xgb_moderate': 0.4,
                'tech_strength': 0.6
            }
        }

    def should_enter(self, df, idx):
        """
        Determine if we should enter based on current regime

        Returns:
            should_enter (bool)
            confidence (str): 'strong' or 'moderate'
            xgb_prob (float)
            tech_signal (str)
            tech_strength (float)
            regime (str)
        """
        if idx >= len(df):
            return False, None, 0.0, 'HOLD', 0.0, 'Unknown'

        # 1. Detect current regime
        regime, regime_confidence = self.regime_detector.detect_regime(df, idx)

        # 2. Get regime-specific thresholds
        thresholds = self.thresholds[regime]

        # 3. Get XGBoost probability
        features = df[self.feature_columns].iloc[idx:idx+1].values
        if np.isnan(features).any():
            return False, None, 0.0, 'HOLD', 0.0, regime

        xgb_prob = self.xgboost.predict_proba(features)[0][1]

        # 4. Get Technical signal
        tech_signal, tech_strength, tech_reason = self.technical.get_signal(df, idx)

        # 5. Apply regime-specific decision logic
        # Strong entry: Both highly confident
        if xgb_prob > thresholds['xgb_strong'] and tech_signal == 'LONG':
            return True, 'strong', xgb_prob, tech_signal, tech_strength, regime

        # Moderate entry: XGBoost moderate + Technical confirms
        if (xgb_prob > thresholds['xgb_moderate'] and
            tech_signal == 'LONG' and
            tech_strength >= thresholds['tech_strength']):
            return True, 'moderate', xgb_prob, tech_signal, tech_strength, regime

        return False, None, xgb_prob, tech_signal, tech_strength, regime


# Test the regime detector
if __name__ == '__main__':
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

    # Load data
    data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)

    # Initialize regime detector
    detector = RegimeDetector()
    df = detector.calculate_indicators(df)
    df = df.dropna()

    print("=" * 80)
    print("Regime Detector Test")
    print("=" * 80)

    # Test on recent data
    test_df = df.iloc[-1000:].reset_index(drop=True)

    regimes = []
    for i in range(len(test_df)):
        regime, confidence = detector.detect_regime(test_df, i)
        regimes.append((regime, confidence))

    # Count regimes
    regime_counts = {}
    for regime, _ in regimes:
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    print(f"\nRegime Distribution (last 1000 candles):")
    for regime, count in regime_counts.items():
        pct = (count / len(regimes)) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")

    # Sample detections
    print(f"\nSample Regime Detections:")
    indices = [0, 100, 200, 300, 400]
    for idx in indices:
        regime, confidence = detector.detect_regime(test_df, idx)
        price = test_df['close'].iloc[idx]
        print(f"  [{idx}] Price={price:.2f}, Regime={regime}, Confidence={confidence:.2f}")

    print(f"\n✅ Regime Detector test complete!")
