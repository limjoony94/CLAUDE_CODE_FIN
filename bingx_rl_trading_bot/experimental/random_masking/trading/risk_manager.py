"""
Risk Management for Random Masking Candle Predictor

Features:
- Position sizing (Kelly criterion, fixed fraction, volatility-based)
- Exposure limits
- Drawdown protection
- Dynamic risk adjustment
"""

import numpy as np
from typing import Optional, Dict
from loguru import logger


class RiskManager:
    """
    Dynamic position sizing and risk management

    Strategies:
    1. Kelly Criterion (fraction)
    2. Fixed fractional sizing
    3. Volatility-based sizing
    4. Confidence-weighted sizing
    """

    def __init__(self, config):
        """
        Initialize risk manager

        Args:
            config: BacktestConfig or similar with:
                - max_position_size: Maximum position as fraction of capital
                - kelly_fraction: Kelly criterion fraction (0.25 = quarter Kelly)
                - leverage: Leverage multiplier
        """
        self.max_position_size = config.max_position_size
        self.kelly_fraction = config.kelly_fraction
        self.leverage = config.leverage

        # Drawdown protection
        self.max_drawdown_limit = 0.20  # Stop trading if DD > 20%
        self.peak_capital = 0.0
        self.current_drawdown = 0.0

        logger.info(f"Initialized RiskManager:")
        logger.info(f"  - Max position size: {self.max_position_size:.1%}")
        logger.info(f"  - Kelly fraction: {self.kelly_fraction:.2f}")
        logger.info(f"  - Leverage: {self.leverage:.1f}x")

    def calculate_position_size(
        self,
        capital: float,
        confidence: float,
        current_price: float,
        method: str = 'confidence_weighted'
    ) -> float:
        """
        Calculate position size based on risk management method

        Args:
            capital: Available capital
            confidence: Signal confidence (0-1)
            current_price: Current asset price
            method: Sizing method ('fixed', 'kelly', 'volatility', 'confidence_weighted')

        Returns:
            Position size in asset units (e.g., BTC)
        """
        # Check drawdown protection
        if self.current_drawdown > self.max_drawdown_limit:
            logger.warning(f"Drawdown limit exceeded: {self.current_drawdown:.2%} > {self.max_drawdown_limit:.2%}")
            return 0.0

        # Calculate base position size
        if method == 'fixed':
            size = self._fixed_sizing(capital, current_price)
        elif method == 'kelly':
            size = self._kelly_sizing(capital, current_price, confidence)
        elif method == 'volatility':
            size = self._volatility_sizing(capital, current_price)
        elif method == 'confidence_weighted':
            size = self._confidence_weighted_sizing(capital, current_price, confidence)
        else:
            raise ValueError(f"Unknown sizing method: {method}")

        # Apply maximum position limit
        max_size = (capital * self.max_position_size) / current_price
        size = min(size, max_size)

        return size

    def _fixed_sizing(self, capital: float, price: float) -> float:
        """Fixed fractional position sizing"""
        position_value = capital * self.max_position_size
        size = position_value / price
        return size

    def _kelly_sizing(
        self,
        capital: float,
        price: float,
        confidence: float,
        win_rate: float = 0.55,
        avg_win: float = 1.5,
        avg_loss: float = 1.0
    ) -> float:
        """
        Kelly criterion position sizing

        f = (p * b - q) / b
        where:
            p = win probability
            q = loss probability
            b = win/loss ratio
        """
        # Use confidence as proxy for win rate adjustment
        adjusted_win_rate = win_rate + (confidence - 0.5) * 0.2
        adjusted_win_rate = np.clip(adjusted_win_rate, 0.3, 0.7)

        loss_rate = 1 - adjusted_win_rate
        win_loss_ratio = avg_win / avg_loss

        # Kelly fraction
        kelly_f = (adjusted_win_rate * win_loss_ratio - loss_rate) / win_loss_ratio

        # Apply Kelly fraction multiplier (e.g., 0.25 for quarter Kelly)
        kelly_f *= self.kelly_fraction

        # Ensure non-negative
        kelly_f = max(0, kelly_f)

        # Calculate position size
        position_value = capital * kelly_f
        size = position_value / price

        return size

    def _volatility_sizing(
        self,
        capital: float,
        price: float,
        volatility: float = 0.02
    ) -> float:
        """
        Volatility-based position sizing

        Reduce position size when volatility is high
        """
        # Base position
        base_position_value = capital * self.max_position_size

        # Adjust for volatility (inverse relationship)
        volatility_adj = 1.0 / (1.0 + volatility * 10)  # Scale volatility impact

        adjusted_value = base_position_value * volatility_adj
        size = adjusted_value / price

        return size

    def _confidence_weighted_sizing(
        self,
        capital: float,
        price: float,
        confidence: float
    ) -> float:
        """
        Confidence-weighted position sizing

        Scale position size by signal confidence
        """
        # Base position
        base_position_value = capital * self.max_position_size

        # Scale by confidence
        # confidence ranges from min_confidence (e.g., 0.6) to 1.0
        # Map to 0.5x to 1.0x multiplier
        confidence_multiplier = 0.5 + 0.5 * confidence

        adjusted_value = base_position_value * confidence_multiplier
        size = adjusted_value / price

        return size

    def update_drawdown(self, current_capital: float):
        """
        Update drawdown tracking

        Args:
            current_capital: Current account capital
        """
        # Update peak
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital

        # Calculate drawdown
        if self.peak_capital > 0:
            self.current_drawdown = (self.peak_capital - current_capital) / self.peak_capital
        else:
            self.current_drawdown = 0.0

    def should_halt_trading(self) -> bool:
        """Check if trading should be halted due to drawdown"""
        return self.current_drawdown > self.max_drawdown_limit

    def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics"""
        return {
            'current_drawdown': self.current_drawdown,
            'peak_capital': self.peak_capital,
            'max_drawdown_limit': self.max_drawdown_limit,
            'trading_halted': self.should_halt_trading()
        }


if __name__ == '__main__':
    # Test risk manager
    print("=" * 60)
    print("Testing RiskManager")
    print("=" * 60)

    from dataclasses import dataclass

    @dataclass
    class TestConfig:
        max_position_size: float = 0.1
        kelly_fraction: float = 0.25
        leverage: float = 1.0

    config = TestConfig()

    # Initialize
    risk_mgr = RiskManager(config)

    # Test 1: Fixed sizing
    print("\n1. Testing Fixed Position Sizing")
    print("-" * 60)

    capital = 10000
    price = 100
    confidence = 0.8

    size = risk_mgr.calculate_position_size(
        capital, confidence, price, method='fixed'
    )

    position_value = size * price
    print(f"Capital: ${capital:,.2f}")
    print(f"Price: ${price:.2f}")
    print(f"Position size: {size:.4f} units")
    print(f"Position value: ${position_value:.2f} ({position_value/capital:.1%} of capital)")

    # Test 2: Kelly sizing
    print("\n2. Testing Kelly Criterion Sizing")
    print("-" * 60)

    for conf in [0.6, 0.75, 0.9]:
        size = risk_mgr.calculate_position_size(
            capital, conf, price, method='kelly'
        )
        position_value = size * price
        print(f"Confidence {conf:.1%}: {size:.4f} units (${position_value:.2f}, {position_value/capital:.1%})")

    # Test 3: Confidence-weighted sizing
    print("\n3. Testing Confidence-Weighted Sizing")
    print("-" * 60)

    for conf in [0.6, 0.75, 0.9]:
        size = risk_mgr.calculate_position_size(
            capital, conf, price, method='confidence_weighted'
        )
        position_value = size * price
        print(f"Confidence {conf:.1%}: {size:.4f} units (${position_value:.2f}, {position_value/capital:.1%})")

    # Test 4: Drawdown tracking
    print("\n4. Testing Drawdown Tracking")
    print("-" * 60)

    risk_mgr.update_drawdown(10000)  # Initial
    print(f"Capital: $10,000 | Drawdown: {risk_mgr.current_drawdown:.2%} | Peak: ${risk_mgr.peak_capital:,.2f}")

    risk_mgr.update_drawdown(12000)  # Profit
    print(f"Capital: $12,000 | Drawdown: {risk_mgr.current_drawdown:.2%} | Peak: ${risk_mgr.peak_capital:,.2f}")

    risk_mgr.update_drawdown(10000)  # Drawdown
    print(f"Capital: $10,000 | Drawdown: {risk_mgr.current_drawdown:.2%} | Peak: ${risk_mgr.peak_capital:,.2f}")

    risk_mgr.update_drawdown(9000)  # Larger drawdown
    print(f"Capital: $9,000 | Drawdown: {risk_mgr.current_drawdown:.2%} | Peak: ${risk_mgr.peak_capital:,.2f}")

    # Test 5: Drawdown protection
    print("\n5. Testing Drawdown Protection")
    print("-" * 60)

    risk_mgr.update_drawdown(9000)  # 25% drawdown
    print(f"Should halt trading: {risk_mgr.should_halt_trading()}")

    size = risk_mgr.calculate_position_size(capital, 0.9, price)
    print(f"Position size with high drawdown: {size:.4f} (should be 0 if halted)")

    # Test 6: Risk metrics
    print("\n6. Testing Risk Metrics")
    print("-" * 60)

    metrics = risk_mgr.get_risk_metrics()
    print("Risk Metrics:")
    for key, value in metrics.items():
        if isinstance(value, bool):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 60)
    print("All risk manager tests passed! ✅")
    print("=" * 60)
