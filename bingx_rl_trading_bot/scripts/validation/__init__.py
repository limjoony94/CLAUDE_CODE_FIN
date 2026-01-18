"""
Strategy Validation Module

Implements the Standard Research Protocol for trading strategy validation.

Usage:
    from scripts.validation import StrategyValidator, ValidationThresholds

    validator = StrategyValidator(
        signal_func=my_signal_function,
        tp_pct=2.5,
        sl_pct=2.0,
        leverage=3,
        strategy_name="My Strategy"
    )

    report = validator.validate(df)
    print(report)
"""

from .strategy_validator import (
    StrategyValidator,
    ValidationThresholds,
    ValidationReport,
    Type1Result,
    Type2Result,
    WalkForwardResult,
    MonteCarloResult,
    validate_look_ahead_bias,
)

__all__ = [
    'StrategyValidator',
    'ValidationThresholds',
    'ValidationReport',
    'Type1Result',
    'Type2Result',
    'WalkForwardResult',
    'MonteCarloResult',
    'validate_look_ahead_bias',
]
