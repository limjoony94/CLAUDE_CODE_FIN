"""Agent abstract base class.

Design ref: docs/02-design/features/whale_inference_abm.design.md Section 4.

Agent contract:
- decide(snapshot, context) -> list[OrderIntent]: produce zero or more orders
- next_decision_delay_ns() -> int: when to schedule next decision
- family: str class attribute identifying strategy class

Determinism: agent uses self.rng (sub-seeded by registry from master_seed + hash(agent_id)).
NEVER call random.* or np.random.* directly outside self.rng.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from abm.constants import BAR_DURATION_NS, INITIAL_WEALTH
from abm.types import OrderbookSnapshot, OrderIntent


@dataclass
class Agent(ABC):
    """Base agent. Subclasses implement family + decide() + next_decision_delay_ns().

    Mutable state allowed (current_wealth, internal counters), but ALL non-init mutations
    must be deterministic functions of observed events + self.rng draws.
    """

    agent_id: str
    initial_wealth: float
    rng: np.random.Generator
    decision_offset_ns: int  # drawn from rng at registry add time, [0, BAR_DURATION_NS)

    current_wealth: float = field(init=False)
    """Mutated EXCLUSIVELY by WealthTracker (Day 8-10 module). Agent subclass code MUST NOT
    write to this field — only read. Single-owner invariant required for determinism."""

    order_counter: int = field(init=False, default=0)
    """Mutated EXCLUSIVELY by self.next_order_id(). Used to generate deterministic per-agent
    order IDs."""

    def __post_init__(self) -> None:
        self.current_wealth = self.initial_wealth
        if not (0 <= self.decision_offset_ns < BAR_DURATION_NS):
            raise ValueError(
                f"decision_offset_ns must be in [0, {BAR_DURATION_NS}), got {self.decision_offset_ns}"
            )

    # ----- Required overrides -----

    family: str = "base"  # override in subclass

    @abstractmethod
    def decide(
        self, snapshot: OrderbookSnapshot, context: dict[str, Any]
    ) -> list[OrderIntent]:
        """Produce zero or more order intents for current decision tick."""

    @abstractmethod
    def next_decision_delay_ns(self) -> int:
        """Delay (ns) until this agent's next scheduled decision."""

    # ----- Helpers -----

    def is_bankrupt(self, threshold: float) -> bool:
        return self.current_wealth <= threshold

    def next_order_id(self) -> str:
        """Issue a deterministic order_id unique within this agent."""
        self.order_counter += 1
        return f"{self.agent_id}_o_{self.order_counter:06d}"
