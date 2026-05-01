"""Piggyback agent. Design Section 4.5 (with B2 patch cold-start).

Decision: identify top wealth-growth performer in last `lookback` bars; copy their last action with `delay` lag.
Excludes other piggyback agents (anti-self-reference).
Cold-start (B2): for t < lookback × BAR_DURATION_NS, no trades / no quotes.
Decision frequency: every bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from abm.agents.base import Agent
from abm.constants import BAR_DURATION_NS, LOT_STEP, MAX_ORDER_SIZE, MIN_ORDER_SIZE, NS_PER_SECOND
from abm.types import OrderbookSnapshot, OrderIntent, OrderType, Side


PIGGYBACK_DELAY_NS = 60 * NS_PER_SECOND  # 1-bar lag


@dataclass
class PiggybackAgent(Agent):
    family: str = field(default="piggyback", init=False)
    lookback_bars: int = 1000
    wealth_fraction: float = 0.03

    def decide(
        self, snapshot: OrderbookSnapshot, context: dict[str, Any]
    ) -> list[OrderIntent]:
        # B2 cold-start
        cold_start_ns = self.lookback_bars * BAR_DURATION_NS
        if snapshot.timestamp_ns < cold_start_ns:
            return []

        # Context owed by simulation (provides recent leaderboard + last actions)
        leaderboard: Optional[list[tuple[str, float]]] = context.get(
            "wealth_growth_leaderboard"
        )
        last_actions: Optional[dict[str, dict[str, Any]]] = context.get(
            "last_actions_by_agent"
        )
        if not leaderboard or not last_actions:
            return []

        # Filter: exclude other piggyback agents + self (anti-self-reference)
        excluded_families = context.get("piggyback_excluded_ids", set())
        candidates = [
            (aid, growth)
            for aid, growth in leaderboard
            if aid != self.agent_id and aid not in excluded_families
        ]
        if not candidates:
            return []

        # Top performer
        top_id, _growth = max(candidates, key=lambda x: x[1])
        action = last_actions.get(top_id)
        if not action:
            return []

        # Lag respected: only follow if action older than delay (1-bar lag)
        action_ts = action.get("timestamp_ns", 0)
        if snapshot.timestamp_ns - action_ts < PIGGYBACK_DELAY_NS:
            return []

        side_str = action.get("side")
        if side_str not in ("buy", "sell"):
            return []
        side = Side.BUY if side_str == "buy" else Side.SELL

        mid = snapshot.mid_price
        if mid is None or mid <= 0:
            return []

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
