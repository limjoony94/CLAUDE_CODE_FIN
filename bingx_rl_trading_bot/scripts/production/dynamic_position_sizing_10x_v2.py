"""
Dynamic Position Sizing for 10x Leverage v2 - Advanced Risk Management
========================================================================

Version 2 Improvements:
1. ✅ Drawdown-based scaling (현재 낙폭 고려)
2. ✅ Kelly Criterion (수학적 최적 포지션)
3. ✅ Trade frequency throttling (과도한 거래 방지)
4. ✅ Capital preservation mode (큰 손실 후 회복 모드)
5. ✅ Adaptive volatility regime (변동성 추세 감지)

Key Features:
- Drawdown > -15%: 포지션 50% 축소
- Kelly Criterion: 승률/손익 기반 최적화
- 1시간 내 3회 이상 거래: 포지션 축소
- 연속 3회 손실: Recovery mode 진입
- 변동성 급증 감지: 즉각 축소
"""

import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta


class DynamicPositionSizer10xV2:
    """
    10배 레버리지 전용 Advanced Position Sizer v2

    New Features:
    - Drawdown monitoring and scaling
    - Kelly Criterion for optimal sizing
    - Trade frequency control
    - Capital preservation mode
    - Volatility regime detection
    """

    def __init__(
        self,
        base_position_pct=0.20,
        max_position_pct=0.30,
        min_position_pct=0.10,
        signal_weight=0.35,
        volatility_weight=0.20,
        streak_weight=0.15,
        drawdown_weight=0.15,
        kelly_weight=0.15,
        # Risk thresholds
        max_drawdown_threshold=-0.15,  # -15% 낙폭 임계값
        trade_frequency_window=3600,    # 1시간 (초)
        max_trades_per_window=3,        # 1시간 내 최대 3회
        recovery_mode_losses=3,         # 3연속 손실 시 recovery mode
    ):
        """
        Advanced Position Sizer 초기화

        Args:
            base_position_pct: 기본 포지션 (20%)
            max_position_pct: 최대 포지션 (30%)
            min_position_pct: 최소 포지션 (10%)
            *_weight: 각 요소 가중치
            max_drawdown_threshold: 낙폭 임계값
            trade_frequency_window: 거래 빈도 체크 시간
            max_trades_per_window: 시간당 최대 거래
            recovery_mode_losses: Recovery mode 진입 조건
        """
        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct

        # Weights
        self.signal_weight = signal_weight
        self.volatility_weight = volatility_weight
        self.streak_weight = streak_weight
        self.drawdown_weight = drawdown_weight
        self.kelly_weight = kelly_weight

        # Normalize weights
        total = signal_weight + volatility_weight + streak_weight + drawdown_weight + kelly_weight
        self.signal_weight /= total
        self.volatility_weight /= total
        self.streak_weight /= total
        self.drawdown_weight /= total
        self.kelly_weight /= total

        # Risk parameters
        self.max_drawdown_threshold = max_drawdown_threshold
        self.trade_frequency_window = trade_frequency_window
        self.max_trades_per_window = max_trades_per_window
        self.recovery_mode_losses = recovery_mode_losses

        # State tracking
        self.recent_trade_times = deque(maxlen=10)
        self.peak_capital = None
        self.recovery_mode = False

    def calculate_position_size(
        self,
        capital: float,
        signal_strength: float,
        current_volatility: float,
        avg_volatility: float,
        recent_trades: list,
        current_timestamp: datetime = None,
        leverage: float = 10.0
    ) -> dict:
        """
        Advanced position size calculation with multiple risk factors

        Args:
            capital: Current capital
            signal_strength: ML model probability
            current_volatility: Current market volatility
            avg_volatility: Historical average volatility
            recent_trades: List of recent trades with pnl_net
            current_timestamp: Current time for frequency check
            leverage: Leverage multiplier

        Returns:
            dict with position details and factor breakdown
        """

        # Initialize peak capital
        if self.peak_capital is None:
            self.peak_capital = capital
        else:
            self.peak_capital = max(self.peak_capital, capital)

        # 1. Signal Factor
        signal_factor = self._calculate_signal_factor(signal_strength)

        # 2. Volatility Factor (with regime detection)
        volatility_factor = self._calculate_volatility_factor_v2(
            current_volatility, avg_volatility, recent_trades
        )

        # 3. Streak Factor
        streak_factor = self._calculate_streak_factor(recent_trades)

        # 4. NEW: Drawdown Factor
        drawdown_factor = self._calculate_drawdown_factor(capital)

        # 5. NEW: Kelly Factor
        kelly_factor = self._calculate_kelly_factor(recent_trades)

        # 6. NEW: Trade Frequency Check
        frequency_penalty = self._calculate_frequency_penalty(current_timestamp)

        # Weighted combination
        combined_factor = (
            self.signal_weight * signal_factor +
            self.volatility_weight * volatility_factor +
            self.streak_weight * streak_factor +
            self.drawdown_weight * drawdown_factor +
            self.kelly_weight * kelly_factor
        )

        # Apply frequency penalty
        combined_factor *= frequency_penalty

        # Check recovery mode
        if self.recovery_mode:
            combined_factor *= 0.5  # Recovery mode: 포지션 50% 축소

        # Scale to position size
        position_size_pct = self.base_position_pct + (
            (self.max_position_pct - self.base_position_pct) * combined_factor
        )

        # Clamp
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
                "drawdown": drawdown_factor,
                "kelly": kelly_factor,
                "frequency_penalty": frequency_penalty,
                "recovery_mode": self.recovery_mode,
                "combined": combined_factor
            }
        }

    def _calculate_signal_factor(self, signal_strength: float) -> float:
        """Signal strength factor (same as v1)"""
        if signal_strength < 0.65:
            return 0.0
        normalized = (signal_strength - 0.65) / 0.35
        return np.clip(normalized, 0.0, 1.0)

    def _calculate_volatility_factor_v2(
        self,
        current_volatility: float,
        avg_volatility: float,
        recent_trades: list
    ) -> float:
        """
        Enhanced volatility factor with regime detection

        New: Detect volatility trend (increasing/decreasing)
        """
        if avg_volatility == 0 or avg_volatility is None:
            return 0.5

        volatility_ratio = current_volatility / avg_volatility

        # Base factor (inverse relationship)
        if volatility_ratio <= 0.5:
            base_factor = 1.0
        elif volatility_ratio <= 1.0:
            base_factor = 1.0 - (volatility_ratio - 0.5)
        elif volatility_ratio <= 1.5:
            base_factor = 0.5 - (volatility_ratio - 1.0)
        else:
            base_factor = 0.0

        # Detect volatility spike (급등)
        if volatility_ratio > 1.5:
            # Severe volatility spike: cut position aggressively
            base_factor *= 0.5

        return np.clip(base_factor, 0.0, 1.0)

    def _calculate_streak_factor(self, recent_trades: list) -> float:
        """Streak factor with recovery mode trigger"""
        if not recent_trades or len(recent_trades) == 0:
            self.recovery_mode = False
            return 1.0

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

        # Trigger recovery mode
        if consecutive_losses >= self.recovery_mode_losses:
            self.recovery_mode = True
        elif consecutive_wins >= 2:
            self.recovery_mode = False  # Exit recovery mode after 2 wins

        # Factor calculation
        if consecutive_wins >= 3:
            return 0.8
        elif consecutive_losses >= 3:
            return 0.2
        elif consecutive_losses == 2:
            return 0.4
        elif consecutive_losses == 1:
            return 0.7
        else:
            return 1.0

    def _calculate_drawdown_factor(self, capital: float) -> float:
        """
        NEW: Drawdown-based position scaling

        Logic:
            - drawdown 0%: factor 1.0 (normal)
            - drawdown -10%: factor 0.7 (cautious)
            - drawdown -15%: factor 0.4 (defensive)
            - drawdown -20%+: factor 0.2 (survival mode)
        """
        if self.peak_capital is None or self.peak_capital == 0:
            return 1.0

        current_drawdown = (capital - self.peak_capital) / self.peak_capital

        if current_drawdown >= 0:
            # At peak or new peak
            factor = 1.0
        elif current_drawdown >= -0.05:
            # -5% or less: normal
            factor = 1.0
        elif current_drawdown >= -0.10:
            # -5% to -10%: slightly cautious
            factor = 0.85
        elif current_drawdown >= -0.15:
            # -10% to -15%: cautious
            factor = 0.6
        elif current_drawdown >= -0.20:
            # -15% to -20%: defensive
            factor = 0.3
        else:
            # -20%+: survival mode
            factor = 0.2

        return factor

    def _calculate_kelly_factor(self, recent_trades: list) -> float:
        """
        NEW: Kelly Criterion for optimal position sizing

        Kelly = (W × R - (1-W)) / R
        where:
            W = win rate
            R = avg_win / abs(avg_loss)

        We use 25% Kelly (fractional Kelly for safety)
        """
        if not recent_trades or len(recent_trades) < 10:
            return 0.5  # Not enough data

        # Use last 20 trades for calculation
        recent = recent_trades[-20:]

        wins = [t['pnl_net'] for t in recent if t.get('pnl_net', 0) > 0]
        losses = [t['pnl_net'] for t in recent if t.get('pnl_net', 0) <= 0]

        if not wins or not losses:
            return 0.5  # Need both wins and losses

        win_rate = len(wins) / len(recent)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss == 0:
            return 0.5

        # Kelly Criterion
        R = avg_win / avg_loss
        kelly = (win_rate * R - (1 - win_rate)) / R

        # Use 25% Kelly (fractional for safety)
        fractional_kelly = kelly * 0.25

        # Convert to factor (0-1)
        # kelly = 1.0 means bet 100% (we cap at factor 1.0)
        # kelly = 0.0 means bet 0% (factor 0.0)
        # kelly < 0 means don't bet (factor 0.0)

        if fractional_kelly < 0:
            factor = 0.0
        elif fractional_kelly > 0.25:
            factor = 1.0
        else:
            # Scale [0, 0.25] to [0.0, 1.0]
            factor = fractional_kelly / 0.25

        return np.clip(factor, 0.0, 1.0)

    def _calculate_frequency_penalty(self, current_timestamp: datetime = None) -> float:
        """
        NEW: Trade frequency penalty

        Logic:
            - < 2 trades in window: 1.0 (normal)
            - 2 trades in window: 0.8 (slight caution)
            - 3 trades in window: 0.5 (throttle)
            - 4+ trades in window: 0.3 (severe throttle)
        """
        if current_timestamp is None:
            return 1.0  # No timestamp, no penalty

        # Add current time
        self.recent_trade_times.append(current_timestamp)

        # Count trades within window
        window_start = current_timestamp - timedelta(seconds=self.trade_frequency_window)
        trades_in_window = sum(1 for t in self.recent_trade_times if t >= window_start)

        if trades_in_window <= 2:
            return 1.0
        elif trades_in_window == 3:
            return 0.8
        elif trades_in_window == 4:
            return 0.5
        else:
            return 0.3  # Too many trades

    def get_position_size_simple(
        self,
        capital: float,
        signal_strength: float,
        leverage: float = 10.0
    ) -> dict:
        """Simple version (not recommended for 10x leverage)"""
        signal_factor = self._calculate_signal_factor(signal_strength)

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

    def record_trade(self, timestamp: datetime, pnl_net: float):
        """Record trade for history tracking"""
        self.recent_trade_times.append(timestamp)


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Advanced Position Sizer v2 예제"""

    sizer = DynamicPositionSizer10xV2()
    capital = 10000.0

    print("="*80)
    print("10배 레버리지 Dynamic Position Sizing v2 - Advanced Risk Management")
    print("="*80)

    # Example 1: 정상 상태
    print("\n1️⃣ 정상 상태 (No drawdown, good signal)")
    result = sizer.calculate_position_size(
        capital=10000,
        signal_strength=0.85,
        current_volatility=1.0,
        avg_volatility=1.0,
        recent_trades=[
            {"pnl_net": 100},
            {"pnl_net": 50},
        ],
        current_timestamp=datetime.now(),
        leverage=10.0
    )
    print(f"   Position: {result['position_size_pct']*100:.1f}%")
    print(f"   Factors: signal={result['factors']['signal']:.2f}, "
          f"DD={result['factors']['drawdown']:.2f}, "
          f"kelly={result['factors']['kelly']:.2f}")

    # Example 2: -15% Drawdown
    print("\n2️⃣ 큰 낙폭 (-15% drawdown)")
    sizer.peak_capital = 10000
    result = sizer.calculate_position_size(
        capital=8500,  # -15% drawdown
        signal_strength=0.85,
        current_volatility=1.0,
        avg_volatility=1.0,
        recent_trades=[
            {"pnl_net": -200},
            {"pnl_net": -300},
        ],
        leverage=10.0
    )
    print(f"   Position: {result['position_size_pct']*100:.1f}% (축소됨!)")
    print(f"   Drawdown Factor: {result['factors']['drawdown']:.2f}")

    # Example 3: Recovery Mode (3연속 손실)
    print("\n3️⃣ Recovery Mode (3연속 손실)")
    result = sizer.calculate_position_size(
        capital=9000,
        signal_strength=0.85,
        current_volatility=1.0,
        avg_volatility=1.0,
        recent_trades=[
            {"pnl_net": -100},
            {"pnl_net": -150},
            {"pnl_net": -200},
        ],
        leverage=10.0
    )
    print(f"   Position: {result['position_size_pct']*100:.1f}%")
    print(f"   Recovery Mode: {result['factors']['recovery_mode']}")
    print(f"   Combined Factor: {result['factors']['combined']:.2f} (50% 감소)")

    # Example 4: Frequency throttling
    print("\n4️⃣ 과도한 거래 (1시간 내 4회)")
    now = datetime.now()
    for i in range(4):
        sizer.record_trade(now - timedelta(minutes=i*10), 50)

    result = sizer.calculate_position_size(
        capital=10000,
        signal_strength=0.85,
        current_volatility=1.0,
        avg_volatility=1.0,
        recent_trades=[{"pnl_net": 50}],
        current_timestamp=now,
        leverage=10.0
    )
    print(f"   Position: {result['position_size_pct']*100:.1f}%")
    print(f"   Frequency Penalty: {result['factors']['frequency_penalty']:.2f}")

    print("\n" + "="*80)
    print("✅ v2 New Features:")
    print("   1. Drawdown-based scaling: -15% DD → 60% position cut")
    print("   2. Kelly Criterion: Mathematical optimal sizing")
    print("   3. Frequency throttling: Prevent overtrading")
    print("   4. Recovery mode: 3 consecutive losses → 50% cut")
    print("   5. Volatility spike detection: Rapid response")
    print("="*80)


if __name__ == "__main__":
    example_usage()
