"""Momentum agent. Design Section 4.1.

Decision: signal = sign(mid[t] - mid[t-N]); MARKET on signal direction; size = wealth_fraction × wealth / mid.
Decision frequency: every bar.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from abm.agents.base import Agent
from abm.constants import BAR_DURATION_NS, LOT_STEP, MAX_ORDER_SIZE, MIN_ORDER_SIZE
from abm.types import OrderbookSnapshot, OrderIntent, OrderType, Side


@dataclass
class MomentumAgent(Agent):
    family: str = field(default="momentum", init=False)
    N: int = 5  # lookback bars
    wealth_fraction: float = 0.05
    confirmation_threshold: float = 0.0

    price_history: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.N < 1:
            raise ValueError(f"N must be >= 1, got {self.N}")
        self.price_history = deque(maxlen=self.N + 1)

    def decide(
        self, snapshot: OrderbookSnapshot, context: dict[str, Any]
    ) -> list[OrderIntent]:
        mid = snapshot.mid_price
        if mid is None or mid <= 0:
            return []
        self.price_history.append(mid)
        if len(self.price_history) <= self.N:
            return []  # warmup

        old_mid = self.price_history[0]
        delta = (mid - old_mid) / old_mid
        if abs(delta) <= self.confirmation_threshold:
            return []

        side = Side.BUY if delta > 0 else Side.SELL
        size = self.wealth_fraction * self.current_wealth / mid
        size = _clip_round(size)
        if size < MIN_ORDER_SIZE:
            return []

        return [OrderIntent(order_type=OrderType.MARKET, side=side, size=size)]

    def next_decision_delay_ns(self) -> int:
        return BAR_DURATION_NS


def _clip_round(size: float) -> float:
    size = max(min(size, MAX_ORDER_SIZE), 0.0)
    return round(size / LOT_STEP) * LOT_STEP
