"""
Dynamic Position Sizing for 10x Leverage - Conservative Profile
================================================================

10배 레버리지는 강력하지만 위험합니다. 보수적 포지션 사이징으로 리스크 관리.

Key Changes from 4x version:
1. Position Range: 10-30% (vs 20-95%)
2. Base Position: 20% (vs 50%)
3. Conservative Scaling: 더 보수적인 신호 해석
4. Full Factor Usage: 변동성, 연속 손실 고려

Risk Management:
- 10% position × 10x leverage = 자본의 1배 노출 (최소)
- 30% position × 10x leverage = 자본의 3배 노출 (최대)
- Stop Loss (-3%)로 최대 손실 제한
"""

import numpy as np
import pandas as pd


class DynamicPositionSizer10x:
    """
    10배 레버리지 전용 Dynamic Position Sizer

    Conservative approach:
    - Lower base position (20% vs 50%)
    - Tighter range (10-30% vs 20-95%)
    - Full factor consideration (signal + volatility + streak)
    - Adaptive to market conditions
    """

    def __init__(
        self,
        base_position_pct=0.20,  # 20% base (보수적)
        max_position_pct=0.30,   # 30% maximum (안전)
        min_position_pct=0.10,   # 10% minimum (방어적)
        signal_weight=0.5,       # 50% weight on signal strength
        volatility_weight=0.3,   # 30% weight on volatility
        streak_weight=0.2        # 20% weight on win/loss streak
    ):
        """
        10배 레버리지용 Position Sizer 초기화

        Args:
            base_position_pct: 기본 포지션 크기 (20%)
            max_position_pct: 최대 허용 (30%)
            min_position_pct: 최소 허용 (10%)
            signal_weight: 신호 강도 가중치
            volatility_weight: 변동성 가중치
            streak_weight: 연속 손익 가중치
        """
        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct

        self.signal_weight = signal_weight
        self.volatility_weight = volatility_weight
        self.streak_weight = streak_weight

        # Normalize weights
        total_weight = signal_weight + volatility_weight + streak_weight
        self.signal_weight /= total_weight
        self.volatility_weight /= total_weight
        self.streak_weight /= total_weight

    def calculate_position_size(
        self,
        capital: float,
        signal_strength: float,
        current_volatility: float,
        avg_volatility: float,
        recent_trades: list,
        leverage: float = 10.0
    ) -> dict:
        """
        다중 요소 기반 포지션 크기 계산

        Returns:
            dict with position details and factor breakdown
        """

        # 1. Signal Strength Factor (보수적 스케일링)
        signal_factor = self._calculate_signal_factor(signal_strength)

        # 2. Volatility Factor (변동성 높으면 축소)
        volatility_factor = self._calculate_volatility_factor(
            current_volatility, avg_volatility
        )

        # 3. Streak Factor (연속 손실 시 축소)
        streak_factor = self._calculate_streak_factor(recent_trades)

        # Weighted combination
        combined_factor = (
            self.signal_weight * signal_factor +
            self.volatility_weight * volatility_factor +
            self.streak_weight * streak_factor
        )

        # Scale to position size (보수적: base × factor)
        position_size_pct = self.base_position_pct + (
            (self.max_position_pct - self.base_position_pct) * combined_factor
        )

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
            "leverage": leverage,
            "factors": {
                "signal": signal_factor,
                "volatility": volatility_factor,
                "streak": streak_factor,
                "combined": combined_factor
            }
        }

    def _calculate_signal_factor(self, signal_strength: float) -> float:
        """
        보수적 신호 해석 (10배 레버리지용)

        Logic:
            - prob 0.65 (threshold) → factor 0.0 (최소)
            - prob 0.75 → factor 0.5 (중간)
            - prob 0.90+ → factor 1.0 (최대)

        더 높은 임계값과 보수적 스케일링
        """
        # Normalize from [0.65, 1.0] to [0.0, 1.0]
        if signal_strength < 0.65:
            return 0.0

        normalized = (signal_strength - 0.65) / 0.35

        # Linear scaling (no exponential boost for safety)
        factor = normalized

        return np.clip(factor, 0.0, 1.0)

    def _calculate_volatility_factor(
        self,
        current_volatility: float,
        avg_volatility: float
    ) -> float:
        """
        변동성 기반 조정 (높으면 축소)

        Logic:
            - current < 0.5 × avg → 1.0 (매우 안정)
            - current = avg → 0.5 (보통)
            - current > 1.5 × avg → 0.0 (매우 불안정)
        """
        if avg_volatility == 0 or avg_volatility is None:
            return 0.5  # Default to medium

        volatility_ratio = current_volatility / avg_volatility

        # Inverse relationship
        if volatility_ratio <= 0.5:
            factor = 1.0
        elif volatility_ratio <= 1.0:
            factor = 1.0 - (volatility_ratio - 0.5)
        elif volatility_ratio <= 1.5:
            factor = 0.5 - (volatility_ratio - 1.0)
        else:
            factor = 0.0

        return np.clip(factor, 0.0, 1.0)

    def _calculate_streak_factor(self, recent_trades: list) -> float:
        """
        연속 손익 기반 조정 (10배 레버리지는 더 보수적)

        Logic:
            - No trades → 1.0 (neutral)
            - 1 loss → 0.7 (cautious)
            - 2 losses → 0.4 (defensive)
            - 3+ losses → 0.2 (very defensive)
            - 3+ wins → 0.8 (avoid overconfidence)
        """
        if not recent_trades or len(recent_trades) == 0:
            return 1.0

        # Look at last 5 trades
        recent = recent_trades[-5:]

        consecutive_wins = 0
        consecutive_losses = 0

        for trade in reversed(recent):
            pnl = trade.get('pnl_net', 0)
            if pnl > 0:
                if consecutive_losses > 0:
                    break
                consecutive_wins += 1
            else:
                if consecutive_wins > 0:
                    break
                consecutive_losses += 1

        # More conservative factors for 10x leverage
        if consecutive_wins >= 3:
            return 0.8  # Avoid overconfidence
        elif consecutive_losses >= 3:
            return 0.2  # Very defensive (10배는 더 조심)
        elif consecutive_losses == 2:
            return 0.4  # Defensive
        elif consecutive_losses == 1:
            return 0.7  # Cautious
        else:
            return 1.0  # Normal

    def get_position_size_simple(
        self,
        capital: float,
        signal_strength: float,
        leverage: float = 10.0
    ) -> dict:
        """
        간단 버전 (신호만 고려)

        ⚠️ 비추천: 10배 레버리지는 full calculation 권장
        """
        signal_factor = self._calculate_signal_factor(signal_strength)

        # Base + range × factor
        position_size_pct = self.base_position_pct + (
            (self.max_position_pct - self.base_position_pct) * signal_factor
        )

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
            "leverage": leverage,
            "factors": {
                "signal": signal_factor,
                "combined": signal_factor
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """10배 레버리지 포지션 사이징 예제"""

    sizer = DynamicPositionSizer10x()
    capital = 10000.0

    print("="*80)
    print("10배 레버리지 Dynamic Position Sizing")
    print("="*80)

    # Example 1: 강한 신호, 낮은 변동성, 손실 없음
    print("\n1️⃣ 이상적 조건 (강신호 + 안정 + 손실없음)")
    result = sizer.calculate_position_size(
        capital=capital,
        signal_strength=0.90,
        current_volatility=0.5,
        avg_volatility=1.0,
        recent_trades=[],
        leverage=10.0
    )
    print(f"   신호: 0.90 (매우 강함)")
    print(f"   Position: {result['position_size_pct']*100:.1f}% (${result['position_value']:,.0f})")
    print(f"   10배 레버리지: ${result['leveraged_value']:,.0f}")
    print(f"   자본 노출: {result['leveraged_value']/capital:.1f}x")
    print(f"   Factors: signal={result['factors']['signal']:.2f}, "
          f"vol={result['factors']['volatility']:.2f}, "
          f"streak={result['factors']['streak']:.2f}")

    # Example 2: 약한 신호, 높은 변동성
    print("\n2️⃣ 나쁜 조건 (약신호 + 불안정)")
    result = sizer.calculate_position_size(
        capital=capital,
        signal_strength=0.70,
        current_volatility=2.0,
        avg_volatility=1.0,
        recent_trades=[],
        leverage=10.0
    )
    print(f"   신호: 0.70 (약함)")
    print(f"   Position: {result['position_size_pct']*100:.1f}% (${result['position_value']:,.0f})")
    print(f"   10배 레버리지: ${result['leveraged_value']:,.0f}")
    print(f"   자본 노출: {result['leveraged_value']/capital:.1f}x")

    # Example 3: 2연속 손실 후
    print("\n3️⃣ 2연속 손실 후 (방어적)")
    result = sizer.calculate_position_size(
        capital=capital,
        signal_strength=0.85,
        current_volatility=1.0,
        avg_volatility=1.0,
        recent_trades=[
            {"pnl_net": -100},
            {"pnl_net": -50}
        ],
        leverage=10.0
    )
    print(f"   신호: 0.85 (강함)")
    print(f"   Position: {result['position_size_pct']*100:.1f}% (${result['position_value']:,.0f})")
    print(f"   10배 레버리지: ${result['leveraged_value']:,.0f}")
    print(f"   자본 노출: {result['leveraged_value']/capital:.1f}x")
    print(f"   ⚠️ 연속 손실로 포지션 축소!")

    # Comparison with old 50% base
    print("\n4️⃣ 기존 50% Base 방식 (비교)")
    old_position_pct = 0.50
    old_leveraged = capital * old_position_pct * 10.0
    print(f"   Position: {old_position_pct*100:.1f}% (고정)")
    print(f"   10배 레버리지: ${old_leveraged:,.0f}")
    print(f"   자본 노출: {old_leveraged/capital:.1f}x")
    print(f"   ❌ 너무 공격적! 리스크 과다")

    print("\n" + "="*80)
    print("✅ 10배 레버리지 안전 운영 원칙")
    print("   • Position Range: 10-30% (vs 기존 20-95%)")
    print("   • 자본 노출: 1-3배 (vs 기존 2-9.5배)")
    print("   • 변동성 높으면 축소")
    print("   • 연속 손실 시 방어적")
    print("   • Stop Loss -3%로 최대 손실 제한")
    print("="*80)


if __name__ == "__main__":
    example_usage()
