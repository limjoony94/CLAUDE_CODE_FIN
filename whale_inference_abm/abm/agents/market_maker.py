"""Market-maker agent. Design Section 4.3.

Decision: quote BID at mid - target_spread/2 and ASK at mid + target_spread/2 each cycle.
Caller (simulation) must cancel previous quotes before re-quoting.
target_spread = base_spread + inventory_skew × current_inventory
Decision frequency: every 10 sim-seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from abm.agents.base import Agent
from abm.constants import LOT_STEP, MAX_ORDER_SIZE, MIN_ORDER_SIZE, NS_PER_SECOND
from abm.types import OrderbookSnapshot, OrderIntent, OrderType, Side


MM_DECISION_PERIOD_NS = 10 * NS_PER_SECOND  # 10s


@dataclass
class MarketMakerAgent(Agent):
    family: str = field(default="market_maker", init=False)
    base_spread: float = 0.001  # 10 bps
    inventory_skew: float = 0.0001  # per unit of inventory
    base_size_fraction: float = 0.10  # of current_wealth — calibrated post-Day-8-10 smoke
    """0.10 (was 0.01 in design pre-calibration). MM at 0.01 was sized below random agents,
    book got swept faster than MM could requote. Advisor calibration: 0.10 = MM provides
    real liquidity (size 0.002 BTC at 50000 wealth), 5x random size."""

    current_inventory: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        super().__post_init__()

    def decide(
        self, snapshot: OrderbookSnapshot, context: dict[str, Any]
    ) -> list[OrderIntent]:
        mid = snapshot.mid_price
        if mid is None or mid <= 0:
            return []

        target_spread = self.base_spread + self.inventory_skew * self.current_inventory
        target_spread = max(target_spread, self.base_spread / 4)  # don't invert

        bid_price = mid * (1.0 - target_spread / 2.0)
        ask_price = mid * (1.0 + target_spread / 2.0)

        size_btc = self.base_size_fraction * self.current_wealth / mid
        size_btc = _clip_round(size_btc)
        if size_btc < MIN_ORDER_SIZE:
            return []

        return [
            OrderIntent(
                order_type=OrderType.LIMIT,
                side=Side.BUY,
                size=size_btc,
                price=round(bid_price, 2),
            ),
            OrderIntent(
                order_type=OrderType.LIMIT,
                side=Side.SELL,
                size=size_btc,
                price=round(ask_price, 2),
            ),
        ]

    def next_decision_delay_ns(self) -> int:
        return MM_DECISION_PERIOD_NS

    def update_inventory(self, signed_size: float) -> None:
        """Wealth/simulation calls this on fill (positive = bought, negative = sold)."""
        self.current_inventory += signed_size


def _clip_round(size: float) -> float:
    size = max(min(size, MAX_ORDER_SIZE), 0.0)
    return round(size / LOT_STEP) * LOT_STEP
