"""
Dynamic Position Sizing for 10x Leverage - Scaled to Match 4x
===============================================================

목표: 10배 레버리지로 4배와 동일한 자본 노출 생성

4배 @ 20-95% = 0.8배-3.8배 노출
10배 @ 8-38% = 0.8배-3.8배 노출 (동일!)

장점:
1. 동일한 노출 → 동일한 수익/손실 패턴
2. 더 낮은 마진 사용 → 더 많은 여유 자금
3. 동일한 Position Sizer 로직 (검증됨)

단점:
1. 청산 위험 높음 (-10% vs -25%)
"""

import numpy as np
import pandas as pd


class DynamicPositionSizer10xScaled:
    """
    10배 레버리지로 4배 노출 재현

    4배 원본과 동일한 로직, 범위만 1/2.5 스케일링
    - 4배 @ 20-95% → 10배 @ 8-38%
    - 동일한 4-factor model
    - 동일한 signal response
    """

    def __init__(
        self,
        base_position_pct=0.20,  # 20% base (4배의 50% / 2.5)
        max_position_pct=0.38,   # 38% maximum (4배의 95% / 2.5)
        min_position_pct=0.08,   # 8% minimum (4배의 20% / 2.5)
        signal_weight=0.4,       # 4배와 동일
        volatility_weight=0.3,   # 4배와 동일
        regime_weight=0.2,       # 4배와 동일
        streak_weight=0.1        # 4배와 동일
    ):
        """
        Initialize 10x Scaled Position Sizer

        4배 DynamicPositionSizer와 동일한 로직
        범위만 1/2.5 스케일링 (10/4 = 2.5)

        Args:
            base_position_pct: 기본 포지션 (20%, 4배의 50% / 2.5)
            max_position_pct: 최대 포지션 (38%, 4배의 95% / 2.5)
            min_position_pct: 최소 포지션 (8%, 4배의 20% / 2.5)
            signal_weight: Signal 가중치 (4배와 동일)
            volatility_weight: Volatility 가중치 (4배와 동일)
            regime_weight: Regime 가중치 (4배와 동일)
            streak_weight: Streak 가중치 (4배와 동일)
        """
        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct

        self.signal_weight = signal_weight
        self.volatility_weight = volatility_weight
        self.regime_weight = regime_weight
        self.streak_weight = streak_weight

        # Normalize weights
        total_weight = signal_weight + volatility_weight + regime_weight + streak_weight
        self.signal_weight /= total_weight
        self.volatility_weight /= total_weight
        self.regime_weight /= total_weight
        self.streak_weight /= total_weight

    def calculate_position_size(
        self,
        capital: float,
        signal_strength: float,  # XGBoost probability (0-1)
        current_volatility: float,  # ATR or volatility measure
        avg_volatility: float,  # Historical average volatility
        market_regime: str,  # "Bull", "Bear", "Sideways"
        recent_trades: list,  # Recent trade results (for streak)
        leverage: float = 1.0
    ) -> dict:
        """
        Calculate dynamic position size (4배 로직과 동일)

        Returns:
            dict with:
                - position_pct: Final position percentage
                - position_dollars: Dollar amount
                - leverage_exposure: Total leveraged exposure
                - factors: Breakdown of each factor
        """
        # 1. Signal Factor (4배와 동일 로직)
        signal_factor = self._calculate_signal_factor(signal_strength)

        # 2. Volatility Factor (4배와 동일 로직)
        volatility_factor = self._calculate_volatility_factor(
            current_volatility, avg_volatility
        )

        # 3. Regime Factor (4배와 동일 로직)
        regime_factor = self._calculate_regime_factor(market_regime)

        # 4. Streak Factor (4배와 동일 로직)
        streak_factor = self._calculate_streak_factor(recent_trades)

        # Weighted average (4배와 동일 로직)
        weighted_factor = (
            signal_factor * self.signal_weight +
            volatility_factor * self.volatility_weight +
            regime_factor * self.regime_weight +
            streak_factor * self.streak_weight
        )

        # Final position (scaled to 8-38% range)
        position_pct = self.base_position_pct * weighted_factor
        position_pct = np.clip(position_pct, self.min_position_pct, self.max_position_pct)

        position_dollars = capital * position_pct
        leverage_exposure = position_dollars * leverage

        return {
            'position_pct': position_pct,
            'position_dollars': position_dollars,
            'leverage_exposure': leverage_exposure,
            'factors': {
                'signal': signal_factor,
                'volatility': volatility_factor,
                'regime': regime_factor,
                'streak': streak_factor,
                'weighted': weighted_factor
            }
        }

    def _calculate_signal_factor(self, signal_strength: float) -> float:
        """
        Calculate signal strength factor (4배와 동일 로직)

        Args:
            signal_strength: XGBoost probability (0-1)

        Returns:
            factor: 0.5 (weak) to 2.0 (strong)
        """
        if signal_strength >= 0.80:
            return 2.0  # Very strong signal
        elif signal_strength >= 0.70:
            return 1.5  # Strong signal
        elif signal_strength >= 0.60:
            return 1.0  # Normal signal
        else:
            return 0.5  # Weak signal

    def _calculate_volatility_factor(
        self, current_volatility: float, avg_volatility: float
    ) -> float:
        """
        Calculate volatility adjustment factor (4배와 동일 로직)

        Args:
            current_volatility: Current ATR or volatility
            avg_volatility: Historical average volatility

        Returns:
            factor: 0.6 (high vol) to 1.2 (low vol)
        """
        if avg_volatility == 0:
            return 1.0

        vol_ratio = current_volatility / avg_volatility

        if vol_ratio > 1.5:
            return 0.6  # Very high volatility, reduce position
        elif vol_ratio > 1.2:
            return 0.8  # High volatility
        elif vol_ratio < 0.7:
            return 1.2  # Low volatility, increase position
        else:
            return 1.0  # Normal volatility

    def _calculate_regime_factor(self, market_regime: str) -> float:
        """
        Calculate market regime factor (4배와 동일 로직)

        Args:
            market_regime: "Bull", "Bear", "Sideways"

        Returns:
            factor: 0.8 (sideways) to 1.2 (trending)
        """
        if market_regime == "Bull":
            return 1.2  # Trending up, increase position
        elif market_regime == "Bear":
            return 1.2  # Trending down, increase position (SHORT 가능)
        else:  # Sideways
            return 0.8  # Range-bound, reduce position

    def _calculate_streak_factor(self, recent_trades: list) -> float:
        """
        Calculate win/loss streak factor (4배와 동일 로직)

        Args:
            recent_trades: List of recent trade results (dict with 'pnl_usd_net')

        Returns:
            factor: 0.6 (losing streak) to 1.3 (winning streak)
        """
        if len(recent_trades) < 3:
            return 1.0

        # Last 5 trades
        last_5 = recent_trades[-5:]

        wins = sum(1 for trade in last_5 if trade.get('pnl_usd_net', 0) > 0)
        losses = sum(1 for trade in last_5 if trade.get('pnl_usd_net', 0) < 0)

        # Winning streak
        if wins >= 4:
            return 1.3  # Hot streak, slightly increase
        elif wins >= 3:
            return 1.1

        # Losing streak
        elif losses >= 4:
            return 0.6  # Cold streak, reduce significantly
        elif losses >= 3:
            return 0.8

        # Mixed results
        else:
            return 1.0

    def get_position_size_simple(self, signal_strength: float, leverage: float = 1.0) -> float:
        """
        Simplified position size (signal only, for quick estimates)
        4배와 동일 로직, 범위만 스케일링

        Args:
            signal_strength: XGBoost probability (0-1)
            leverage: Leverage multiplier

        Returns:
            position_pct: Position percentage
        """
        signal_factor = self._calculate_signal_factor(signal_strength)
        position_pct = self.base_position_pct * signal_factor
        position_pct = np.clip(position_pct, self.min_position_pct, self.max_position_pct)
        return position_pct


# Quick test
if __name__ == "__main__":
    sizer = DynamicPositionSizer10xScaled()

    print("="*80)
    print("10x SCALED POSITION SIZER TEST")
    print("="*80)
    print("\n목표: 10배 @ 8-38%로 4배 @ 20-95%와 동일한 노출 생성\n")

    test_cases = [
        ("Very Strong", 0.90, 500, 600, "Bull", [100, 50, 75, 80]),
        ("Strong", 0.75, 600, 600, "Sideways", [100, -50, 75]),
        ("Normal", 0.65, 700, 600, "Bear", [100, -50]),
        ("Weak", 0.55, 800, 600, "Sideways", [-50, -30, -40]),
    ]

    capital = 10000
    leverage = 10

    print(f"Capital: ${capital:,.2f}")
    print(f"Leverage: {leverage}x\n")

    print(f"{'Signal':<15} {'Prob':<8} {'Position':<10} {'Margin $':<12} {'Exposure $':<12} {'Actual':<10}")
    print("-" * 80)

    for signal_type, prob, curr_vol, avg_vol, regime, trades in test_cases:
        result = sizer.calculate_position_size(
            capital=capital,
            signal_strength=prob,
            current_volatility=curr_vol,
            avg_volatility=avg_vol,
            market_regime=regime,
            recent_trades=trades,
            leverage=leverage
        )

        position_pct = result['position_pct']
        position_dollars = result['position_dollars']
        leverage_exposure = result['leverage_exposure']
        actual_exposure = leverage_exposure / capital

        print(f"{signal_type:<15} {prob:<8.2f} {position_pct*100:>6.1f}%   "
              f"${position_dollars:>9,.2f}  ${leverage_exposure:>9,.2f}  {actual_exposure:>6.2f}x")

    print("\n" + "="*80)
    print("4배 @ 20-95% 범위와 비교:")
    print("  4배 최소: 20% × 4 = 0.80x 노출 → 10배: 8% × 10 = 0.80x ✅")
    print("  4배 최대: 95% × 4 = 3.80x 노출 → 10배: 38% × 10 = 3.80x ✅")
    print("  4배 베이스: 50% × 4 = 2.00x 노출 → 10배: 20% × 10 = 2.00x ✅")
    print("\n결론: 동일한 노출, 더 많은 마진 여유 (평균 80%)")
    print("="*80)
