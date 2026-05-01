"""Friction model: BingX taker/maker fees + slippage diagnostic.

Design ref: docs/02-design/features/whale_inference_abm.design.md Section 6.

v1 scope (B3 patch decision): cash-margin spot-like dynamics. NO funding rate, NO
liquidations, NO leverage. v1 substrate hypotheses cannot depend on those mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass

from abm.constants import MAKER_FEE, TAKER_FEE
from abm.types import Role


@dataclass
class Friction:
    """Per-trade fee calculator + slippage diagnostic."""

    taker_fee: float = TAKER_FEE  # 0.0005 = 0.05% (BingX rate)
    maker_fee: float = MAKER_FEE  # 0.0002 = 0.02%

    def __post_init__(self) -> None:
        if self.taker_fee < 0:
            raise ValueError(f"taker_fee must be >= 0, got {self.taker_fee}")
        if self.maker_fee < 0:
            raise ValueError(f"maker_fee must be >= 0, got {self.maker_fee}")

    def fee(self, role: Role, notional: float) -> float:
        """Fee for one side of a trade. notional = price × size."""
        if notional < 0:
            raise ValueError(f"notional must be >= 0, got {notional}")
        if role == Role.TAKER:
            return self.taker_fee * notional
        if role == Role.MAKER:
            return self.maker_fee * notional
        raise ValueError(f"Unknown role: {role}")

    def slippage_observed(
        self,
        submitted_size: float,
        executed_avg_price: float,
        expected_price: float,
    ) -> float:
        """Diagnostic: relative slippage between expected and executed price.

        Positive = price moved against trader (worse fill).
        NOT deducted from PnL (already captured in book-walk price).
        """
        if expected_price <= 0:
            raise ValueError(f"expected_price must be > 0, got {expected_price}")
        if submitted_size <= 0:
            return 0.0
        return (executed_avg_price - expected_price) / expected_price
