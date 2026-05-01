"""Random agent. Design Section 4.4.

Decision: side ~ Uniform({buy, sell}); type ~ Uniform({MARKET, LIMIT}); LIMIT price = mid * (1 + uniform(-0.01, 0.01)).
Decision frequency: Poisson arrivals with lambda = 1/120s (avg 1 trade per 2 minutes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from abm.agents.base import Agent
from abm.constants import LOT_STEP, MAX_ORDER_SIZE, MIN_ORDER_SIZE, NS_PER_SECOND
from abm.types import OrderbookSnapshot, OrderIntent, OrderType, Side


RANDOM_RATE_LAMBDA = 1.0 / 120.0  # 1 per 2 minutes
RANDOM_LIMIT_OFFSET_PCT = 0.01  # ±1% from mid


@dataclass
class RandomAgent(Agent):
    family: str = field(default="random", init=False)
    wealth_fraction: float = 0.02

    def decide(
        self, snapshot: OrderbookSnapshot, context: dict[str, Any]
    ) -> list[OrderIntent]:
        mid = snapshot.mid_price
        if mid is None or mid <= 0:
            return []

        side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
        is_market = self.rng.random() < 0.5

        size = self.wealth_fraction * self.current_wealth / mid
        size = _clip_round(size)
        if size < MIN_ORDER_SIZE:
            return []

        if is_market:
            return [OrderIntent(order_type=OrderType.MARKET, side=side, size=size)]

        offset = self.rng.uniform(-RANDOM_LIMIT_OFFSET_PCT, RANDOM_LIMIT_OFFSET_PCT)
        price = round(mid * (1.0 + offset), 2)
        return [
            OrderIntent(order_type=OrderType.LIMIT, side=side, size=size, price=price)
        ]

    def next_decision_delay_ns(self) -> int:
        """Poisson inter-arrival time. ns-quantized."""
        delay_seconds = self.rng.exponential(scale=1.0 / RANDOM_RATE_LAMBDA)
        return max(1, int(delay_seconds * NS_PER_SECOND))


def _clip_round(size: float) -> float:
    size = max(min(size, MAX_ORDER_SIZE), 0.0)
    return round(size / LOT_STEP) * LOT_STEP
