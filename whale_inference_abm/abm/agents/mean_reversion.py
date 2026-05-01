"""Mean-reversion agent. Design Section 4.2.

Decision: deviation = (mid - MA[t,N]) / MA[t,N]; MARKET opposite to deviation if |dev| > threshold.
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
class MeanReversionAgent(Agent):
    family: str = field(default="mean_reversion", init=False)
    N: int = 20  # MA window
    threshold: float = 0.001  # 0.1% deviation — calibrated post-G0 diagnostic
    """0.001 (was 0.005 in design pre-calibration). G0 acceptance diagnostic showed:
    - At 0.5% threshold: 0 MR trades in 1k-bar smoke (12,295 decisions made, all Hold)
    - At 0.3% threshold: 0 MR trades
    - At 0.2% threshold: 0 MR trades
    - At 0.1% threshold: 62 MR trades — G1 IRL training has signal to recover
    Mid range in 1k-bar smoke is only 0.655% so MA-mid deviation rarely exceeds 0.5%.
    Loosening to 0.1% makes MR an active participant without losing strategy character
    (still a contrarian fade, just on smaller deviations)."""
    wealth_fraction: float = 0.05

    price_history: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.N < 2:
            raise ValueError(f"N must be >= 2, got {self.N}")
        self.price_history = deque(maxlen=self.N)

    def decide(
        self, snapshot: OrderbookSnapshot, context: dict[str, Any]
    ) -> list[OrderIntent]:
        mid = snapshot.mid_price
        if mid is None or mid <= 0:
            return []

        # Need N PRIOR prices (excluding current). MA[t] = mean(mid[t-N : t]).
        # Append current AFTER computing deviation so it joins history for next call.
        if len(self.price_history) < self.N:
            self.price_history.append(mid)
            return []

        ma = sum(self.price_history) / self.N
        if ma <= 0:
            self.price_history.append(mid)
            return []
        deviation = (mid - ma) / ma
        self.price_history.append(mid)

        if abs(deviation) <= self.threshold:
            return []

        # Reversion: trade OPPOSITE to deviation
        side = Side.SELL if deviation > 0 else Side.BUY
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
