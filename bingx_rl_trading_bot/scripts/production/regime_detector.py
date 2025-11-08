"""
Market Regime Detection Module

비판적 사고:
"모든 시장 상황에 하나의 전략을 쓰는 것은 비효율적이다"

Bull market: Trend following이 최적
Bear market: Mean reversion이 최적
Sideways: Breakout이 최적

→ Regime을 감지하고 적절한 전략을 선택하자
"""

import pandas as pd
import numpy as np
from typing import Literal

RegimeType = Literal["Bull", "Bear", "Sideways"]

class RegimeDetector:
    """
    Market Regime Detection

    Uses multiple indicators to classify current market state:
    1. Price trend (EMA-based)
    2. Momentum (RSI)
    3. Volatility (ATR)
    4. Recent returns
    """

    def __init__(
        self,
        lookback_window: int = 240,  # 20 hours (240 × 5min)
        bull_threshold: float = 0.03,  # +3% for bull
        bear_threshold: float = -0.02,  # -2% for bear
        trend_weight: float = 0.5,
        momentum_weight: float = 0.3,
        volatility_weight: float = 0.2
    ):
        self.lookback = lookback_window
        self.bull_thresh = bull_threshold
        self.bear_thresh = bear_threshold
        self.trend_w = trend_weight
        self.momentum_w = momentum_weight
        self.volatility_w = volatility_weight

    def detect_regime(
        self,
        df: pd.DataFrame,
        idx: int
    ) -> RegimeType:
        """
        Detect market regime at given index

        Returns:
            "Bull": Strong uptrend
            "Bear": Strong downtrend
            "Sideways": Range-bound or weak trend
        """
        if idx < self.lookback:
            return "Sideways"  # Not enough history

        window = df.iloc[max(0, idx - self.lookback):idx + 1]

        # 1. Price Trend (50% weight)
        trend_score = self._calculate_trend_score(window)

        # 2. Momentum (30% weight)
        momentum_score = self._calculate_momentum_score(window)

        # 3. Volatility (20% weight)
        volatility_score = self._calculate_volatility_score(window)

        # Combined score
        combined_score = (
            self.trend_w * trend_score +
            self.momentum_w * momentum_score +
            self.volatility_w * volatility_score
        )

        # Classify regime
        if combined_score > 0.6:
            return "Bull"
        elif combined_score < 0.4:
            return "Bear"
        else:
            return "Sideways"

    def _calculate_trend_score(self, window: pd.DataFrame) -> float:
        """
        Calculate trend strength score [0, 1]

        Uses:
        - Price change over window
        - EMA slopes
        - Higher highs / Lower lows pattern
        """
        start_price = window['close'].iloc[0]
        end_price = window['close'].iloc[-1]
        price_change_pct = (end_price - start_price) / start_price

        # Normalize to [0, 1]
        # -5% → 0.0 (strong bear)
        #  0% → 0.5 (neutral)
        # +5% → 1.0 (strong bull)
        normalized = (price_change_pct + 0.05) / 0.10
        return np.clip(normalized, 0.0, 1.0)

    def _calculate_momentum_score(self, window: pd.DataFrame) -> float:
        """
        Calculate momentum score [0, 1]

        Uses RSI as proxy for momentum
        """
        if 'rsi' not in window.columns:
            return 0.5  # Neutral if RSI unavailable

        current_rsi = window['rsi'].iloc[-1]

        if pd.isna(current_rsi):
            return 0.5

        # RSI interpretation:
        # 70+ = overbought (strong bull momentum) → 1.0
        # 50 = neutral → 0.5
        # 30- = oversold (strong bear momentum) → 0.0

        if current_rsi >= 50:
            # 50-100 → 0.5-1.0
            normalized = 0.5 + (current_rsi - 50) / 100
        else:
            # 0-50 → 0.0-0.5
            normalized = current_rsi / 100

        return np.clip(normalized, 0.0, 1.0)

    def _calculate_volatility_score(self, window: pd.DataFrame) -> float:
        """
        Calculate volatility score [0, 1]

        High volatility → Likely trending (bull or bear)
        Low volatility → Likely sideways

        Returns:
        - High volatility: 0.0 or 1.0 (trending, direction from price)
        - Low volatility: 0.5 (sideways)
        """
        # Calculate realized volatility
        returns = window['close'].pct_change()
        volatility = returns.std()

        # Normalize volatility (typical BTC 5-min std: 0.001-0.003)
        # Low vol < 0.0015
        # High vol > 0.0025

        if volatility < 0.0015:
            # Low volatility → Sideways signal
            return 0.5
        elif volatility > 0.0025:
            # High volatility → Trending
            # Use price direction to determine bull vs bear
            start_price = window['close'].iloc[0]
            end_price = window['close'].iloc[-1]

            if end_price > start_price:
                return 1.0  # High vol + up = Bull
            else:
                return 0.0  # High vol + down = Bear
        else:
            # Medium volatility → Neutral
            return 0.5

    def classify_dataset_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime column to entire dataset

        Returns:
            DataFrame with 'regime' column added
        """
        regimes = []

        for i in range(len(df)):
            regime = self.detect_regime(df, i)
            regimes.append(regime)

        df = df.copy()
        df['regime'] = regimes

        return df

    def get_regime_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get regime distribution statistics
        """
        if 'regime' not in df.columns:
            df = self.classify_dataset_regimes(df)

        total = len(df)
        bull_count = (df['regime'] == 'Bull').sum()
        bear_count = (df['regime'] == 'Bear').sum()
        sideways_count = (df['regime'] == 'Sideways').sum()

        return {
            'total': total,
            'bull_count': bull_count,
            'bull_pct': (bull_count / total) * 100,
            'bear_count': bear_count,
            'bear_pct': (bear_count / total) * 100,
            'sideways_count': sideways_count,
            'sideways_pct': (sideways_count / total) * 100
        }


def simple_regime_detection(
    window: pd.DataFrame,
    bull_threshold: float = 0.03,
    bear_threshold: float = -0.02
) -> RegimeType:
    """
    Simple regime detection based on price change only

    Used for quick classification without full RegimeDetector
    """
    start_price = window['close'].iloc[0]
    end_price = window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1)

    if return_pct > bull_threshold:
        return "Bull"
    elif return_pct < bear_threshold:
        return "Bear"
    else:
        return "Sideways"


if __name__ == "__main__":
    # Test regime detection
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features

    DATA_DIR = PROJECT_ROOT / "data" / "historical"

    print("=" * 80)
    print("Testing Regime Detection")
    print("=" * 80)

    # Load data
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"✅ Data loaded: {len(df)} candles")

    # Calculate features (for RSI)
    df = calculate_features(df)
    df = df.ffill().dropna()
    print(f"✅ Features calculated: {len(df)} rows")

    # Detect regimes
    print("\nDetecting market regimes...")
    detector = RegimeDetector(lookback_window=240)
    df = detector.classify_dataset_regimes(df)

    # Statistics
    stats = detector.get_regime_statistics(df)

    print(f"\n{'=' * 80}")
    print("Regime Distribution")
    print(f"{'=' * 80}")
    print(f"Total candles: {stats['total']}")
    print(f"\nBull: {stats['bull_count']:,} ({stats['bull_pct']:.1f}%)")
    print(f"Bear: {stats['bear_count']:,} ({stats['bear_pct']:.1f}%)")
    print(f"Sideways: {stats['sideways_count']:,} ({stats['sideways_pct']:.1f}%)")

    print(f"\n✅ Regime detection complete!")
