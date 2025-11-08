"""
Dynamic Position Sizing Module

전문 트레이더 방식:
1. Signal Strength: XGBoost probability에 따라 조절
2. Volatility Adjustment: ATR 기반 변동성 고려
3. Market Regime: Bull/Bear/Sideways 차별화
4. Win/Loss Streak: 연속 손실 시 축소

비판적 사고:
"모든 거래에 95% 투입은 너무 조잡합니다.
신호가 강할 때 크게, 약할 때 작게 투입해야 합니다."
"""

import numpy as np
import pandas as pd


class DynamicPositionSizer:
    """
    Dynamic Position Sizing with Multiple Factors

    Features:
    1. Signal-based sizing (XGBoost probability)
    2. Volatility-adjusted sizing (ATR)
    3. Regime-based sizing (Bull/Bear/Sideways)
    4. Streak-based sizing (win/loss history)
    """

    def __init__(
        self,
        base_position_pct=0.50,  # 50% base (conservative)
        max_position_pct=0.95,   # 95% maximum
        min_position_pct=0.20,   # 20% minimum
        signal_weight=0.4,       # 40% weight on signal strength
        volatility_weight=0.3,   # 30% weight on volatility
        regime_weight=0.2,       # 20% weight on market regime
        streak_weight=0.1        # 10% weight on win/loss streak
    ):
        """
        Initialize Dynamic Position Sizer

        Args:
            base_position_pct: Base position size (default 50%)
            max_position_pct: Maximum allowed (default 95%)
            min_position_pct: Minimum allowed (default 20%)
            signal_weight: Weight for signal strength factor
            volatility_weight: Weight for volatility factor
            regime_weight: Weight for regime factor
            streak_weight: Weight for streak factor
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
        Calculate dynamic position size based on multiple factors

        Returns:
            dict with:
                - position_size_pct: Final position size percentage
                - position_value: Dollar value of position
                - leveraged_value: With leverage applied
                - quantity: Number of units to buy
                - factors: Breakdown of each factor's contribution
        """

        # 1. Signal Strength Factor (0-1)
        signal_factor = self._calculate_signal_factor(signal_strength)

        # 2. Volatility Factor (0-1)
        volatility_factor = self._calculate_volatility_factor(
            current_volatility, avg_volatility
        )

        # 3. Market Regime Factor (0-1)
        regime_factor = self._calculate_regime_factor(market_regime)

        # 4. Win/Loss Streak Factor (0-1)
        streak_factor = self._calculate_streak_factor(recent_trades)

        # Weighted combination
        combined_factor = (
            self.signal_weight * signal_factor +
            self.volatility_weight * volatility_factor +
            self.regime_weight * regime_factor +
            self.streak_weight * streak_factor
        )

        # Scale to position size
        position_size_pct = self.base_position_pct * (0.5 + combined_factor)

        # Clamp to min/max
        position_size_pct = np.clip(
            position_size_pct,
            self.min_position_pct,
            self.max_position_pct
        )

        # Calculate values
        position_value = capital * position_size_pct
        leveraged_value = position_value * leverage

        return {
            "position_size_pct": position_size_pct,
            "position_value": position_value,
            "leveraged_value": leveraged_value,
            "factors": {
                "signal": signal_factor,
                "volatility": volatility_factor,
                "regime": regime_factor,
                "streak": streak_factor,
                "combined": combined_factor
            }
        }

    def _calculate_signal_factor(self, signal_strength: float) -> float:
        """
        Signal Strength Factor: Higher XGBoost probability = Larger position

        Logic:
            - prob 0.50 → factor 0.0 (no confidence)
            - prob 0.70 → factor 0.5 (medium)
            - prob 0.90 → factor 1.0 (high confidence)
        """
        # Normalize from [0.5, 1.0] to [0.0, 1.0]
        normalized = (signal_strength - 0.5) / 0.5

        # Apply exponential scaling (reward very strong signals)
        factor = normalized ** 1.5

        return np.clip(factor, 0.0, 1.0)

    def _calculate_volatility_factor(
        self,
        current_volatility: float,
        avg_volatility: float
    ) -> float:
        """
        Volatility Factor: Lower volatility = Larger position (safer)

        Logic:
            - current_vol = 0.5 × avg_vol → factor 1.0 (calm market, larger position)
            - current_vol = avg_vol → factor 0.5 (normal)
            - current_vol = 2.0 × avg_vol → factor 0.0 (volatile, smaller position)
        """
        if avg_volatility == 0:
            return 0.5  # Default to medium

        volatility_ratio = current_volatility / avg_volatility

        # Inverse relationship: lower volatility = higher factor
        if volatility_ratio <= 0.5:
            factor = 1.0  # Very calm
        elif volatility_ratio <= 1.0:
            factor = 1.0 - (volatility_ratio - 0.5) / 0.5 * 0.5  # 1.0 → 0.5
        elif volatility_ratio <= 2.0:
            factor = 0.5 - (volatility_ratio - 1.0) / 1.0 * 0.5  # 0.5 → 0.0
        else:
            factor = 0.0  # Very volatile

        return np.clip(factor, 0.0, 1.0)

    def _calculate_regime_factor(self, market_regime: str) -> float:
        """
        Market Regime Factor: Adjust for market conditions

        Logic:
            - Bull: 1.0 (confident, larger positions)
            - Sideways: 0.6 (medium caution)
            - Bear: 0.3 (defensive, smaller positions)
        """
        regime_factors = {
            "Bull": 1.0,
            "Sideways": 0.6,
            "Bear": 0.3,
            "Unknown": 0.5
        }

        return regime_factors.get(market_regime, 0.5)

    def _calculate_streak_factor(self, recent_trades: list) -> float:
        """
        Win/Loss Streak Factor: Reduce after losses, cautious after wins

        Logic:
            - 3+ consecutive wins → 0.8 (reduce to avoid overconfidence)
            - 0-2 wins/losses mixed → 1.0 (normal)
            - 1 loss → 0.9
            - 2 consecutive losses → 0.6 (reduce risk)
            - 3+ consecutive losses → 0.3 (very defensive)
        """
        if not recent_trades or len(recent_trades) == 0:
            return 1.0  # No history, neutral

        # Look at last 5 trades
        recent = recent_trades[-5:]

        # Count consecutive wins/losses from most recent
        consecutive_wins = 0
        consecutive_losses = 0

        for trade in reversed(recent):
            pnl = trade.get('pnl_usd_net', 0)
            if pnl > 0:
                if consecutive_losses > 0:
                    break
                consecutive_wins += 1
            else:
                if consecutive_wins > 0:
                    break
                consecutive_losses += 1

        # Apply factor based on streak
        if consecutive_wins >= 3:
            return 0.8  # Reduce after hot streak (avoid overconfidence)
        elif consecutive_losses >= 3:
            return 0.3  # Very defensive after losses
        elif consecutive_losses == 2:
            return 0.6  # Cautious after 2 losses
        elif consecutive_losses == 1:
            return 0.9  # Slightly cautious
        else:
            return 1.0  # Normal

    def get_position_size_simple(
        self,
        capital: float,
        signal_strength: float,
        leverage: float = 1.0
    ) -> dict:
        """
        Simplified position sizing (signal-only)

        For quick calculation when other factors not available
        """
        signal_factor = self._calculate_signal_factor(signal_strength)
        position_size_pct = self.base_position_pct * (0.5 + signal_factor)
        position_size_pct = np.clip(
            position_size_pct,
            self.min_position_pct,
            self.max_position_pct
        )

        position_value = capital * position_size_pct
        leveraged_value = position_value * leverage

        return {
            "position_size_pct": position_size_pct,
            "position_value": position_value,
            "leveraged_value": leveraged_value,
            "factors": {
                "signal": signal_factor,
                "combined": signal_factor
            }
        }


# ============================================================================
# Example Usage and Testing
# ============================================================================

def example_usage():
    """Example of how to use DynamicPositionSizer"""

    sizer = DynamicPositionSizer(
        base_position_pct=0.50,  # 50% base
        max_position_pct=0.95,
        min_position_pct=0.20
    )

    capital = 10000.0

    print("=" * 80)
    print("Dynamic Position Sizing Examples")
    print("=" * 80)

    # Example 1: Strong signal, low volatility, bull market, no losses
    print("\n1. IDEAL CONDITIONS (강한 신호, 낮은 변동성, 강세장)")
    result = sizer.calculate_position_size(
        capital=capital,
        signal_strength=0.90,  # Very strong signal
        current_volatility=0.5,  # Low volatility
        avg_volatility=1.0,
        market_regime="Bull",
        recent_trades=[],
        leverage=2.0
    )
    print(f"   Signal: 0.90 (strong)")
    print(f"   Position Size: {result['position_size_pct']*100:.1f}%")
    print(f"   Position Value: ${result['position_value']:,.2f}")
    print(f"   Leveraged (2x): ${result['leveraged_value']:,.2f}")
    print(f"   Factors: {result['factors']}")

    # Example 2: Weak signal, high volatility, bear market
    print("\n2. POOR CONDITIONS (약한 신호, 높은 변동성, 약세장)")
    result = sizer.calculate_position_size(
        capital=capital,
        signal_strength=0.70,  # Weak signal
        current_volatility=2.0,  # High volatility
        avg_volatility=1.0,
        market_regime="Bear",
        recent_trades=[],
        leverage=2.0
    )
    print(f"   Signal: 0.70 (weak)")
    print(f"   Position Size: {result['position_size_pct']*100:.1f}%")
    print(f"   Position Value: ${result['position_value']:,.2f}")
    print(f"   Leveraged (2x): ${result['leveraged_value']:,.2f}")
    print(f"   Factors: {result['factors']}")

    # Example 3: After 2 consecutive losses
    print("\n3. AFTER 2 CONSECUTIVE LOSSES (리스크 감소)")
    result = sizer.calculate_position_size(
        capital=capital,
        signal_strength=0.85,  # Good signal
        current_volatility=1.0,  # Normal volatility
        avg_volatility=1.0,
        market_regime="Sideways",
        recent_trades=[
            {"pnl_usd_net": -50},
            {"pnl_usd_net": -30}
        ],
        leverage=2.0
    )
    print(f"   Signal: 0.85 (good)")
    print(f"   Position Size: {result['position_size_pct']*100:.1f}%")
    print(f"   Position Value: ${result['position_value']:,.2f}")
    print(f"   Leveraged (2x): ${result['leveraged_value']:,.2f}")
    print(f"   Factors: {result['factors']}")

    # Example 4: Fixed 95% comparison (old way)
    print("\n4. OLD WAY (고정 95% - 비교용)")
    old_position_pct = 0.95
    old_position_value = capital * old_position_pct
    old_leveraged_value = old_position_value * 2.0
    print(f"   Position Size: {old_position_pct*100:.1f}% (always!)")
    print(f"   Position Value: ${old_position_value:,.2f}")
    print(f"   Leveraged (2x): ${old_leveraged_value:,.2f}")
    print(f"   Problem: ❌ No adjustment for signal, volatility, or losses")

    print("\n" + "=" * 80)
    print("✅ Dynamic sizing adjusts to market conditions")
    print("✅ Strong signals → Larger positions")
    print("✅ High volatility → Smaller positions")
    print("✅ Bear markets → Defensive sizing")
    print("✅ After losses → Risk reduction")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
