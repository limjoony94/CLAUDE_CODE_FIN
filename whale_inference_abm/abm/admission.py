"""AdmissionScheduler: open-system Poisson admissions + frozen-window enforcement.

Design ref: docs/02-design/features/whale_inference_abm.design.md Section 7.

Open phase (t < T_open_ns): new agents join via Poisson process, lambda = ADMISSION_RATE_LAMBDA.
Frozen phase (T_open_ns <= t < T_open_ns + T_extract_ns): admissions DISABLED.
                Inverse machinery (G3) operates on this window.

RNG contract (per advisor v0.4 checkpoint):
- All admission-time RNG draws use scheduler.rng (master), NOT per-agent RNG.
- Per-agent RNG only exists AFTER agent is constructed (registry.derived_seed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from abm.agents.market_maker import MarketMakerAgent
from abm.agents.mean_reversion import MeanReversionAgent
from abm.agents.momentum import MomentumAgent
from abm.agents.piggyback import PiggybackAgent
from abm.agents.random_agent import RandomAgent
from abm.constants import (
    ADMISSION_INITIAL_WEALTH,
    ADMISSION_RATE_LAMBDA,
    BAR_DURATION_NS,
    DEFAULT_T_EXTRACT_BARS,
    DEFAULT_T_OPEN_BARS,
    NS_PER_SECOND,
)

if TYPE_CHECKING:
    from abm.agents.base import Agent
    from abm.registry import AgentRegistry


# Family roster used for uniform admission draw
_ADMISSION_FAMILIES = ["momentum", "mean_reversion", "market_maker", "random", "piggyback"]


@dataclass
class AdmissionScheduler:
    """Generates new-agent admission events during the open phase.

    Stateful: tracks `_admission_counter` for unique agent_id generation.
    Caller (Simulation) is responsible for invoking only when `now() < T_open_ns`.
    """

    T_open_bars: int = DEFAULT_T_OPEN_BARS
    T_extract_bars: int = DEFAULT_T_EXTRACT_BARS
    rate_lambda: float = ADMISSION_RATE_LAMBDA  # admissions per second
    initial_wealth: float = ADMISSION_INITIAL_WEALTH

    def __post_init__(self) -> None:
        if self.T_open_bars <= 0:
            raise ValueError(f"T_open_bars must be > 0, got {self.T_open_bars}")
        if self.T_extract_bars <= 0:
            raise ValueError(f"T_extract_bars must be > 0, got {self.T_extract_bars}")
        if self.rate_lambda <= 0:
            raise ValueError(f"rate_lambda must be > 0, got {self.rate_lambda}")
        self._admission_counter: int = 0

    @property
    def T_open_ns(self) -> int:
        return self.T_open_bars * BAR_DURATION_NS

    @property
    def T_extract_ns(self) -> int:
        return self.T_extract_bars * BAR_DURATION_NS

    @property
    def terminal_time_ns(self) -> int:
        return self.T_open_ns + self.T_extract_ns

    def is_open_phase(self, now_ns: int) -> bool:
        return now_ns < self.T_open_ns

    def next_admission_delay_ns(self, rng: np.random.Generator) -> int:
        """Poisson inter-arrival: exponential(1/lambda) seconds → ns."""
        delay_seconds = rng.exponential(scale=1.0 / self.rate_lambda)
        return max(1, int(delay_seconds * NS_PER_SECOND))

    def create_new_agent(
        self,
        rng: np.random.Generator,
        registry: "AgentRegistry",
        now_ns: int,
    ) -> "Agent":
        """Construct a new agent with uniformly-drawn family + registry-derived sub-seed.

        Caller (Simulation) MUST then invoke registry.add_agent(agent) and push first
        AGENT_DECISION event for it.
        """
        family = _ADMISSION_FAMILIES[rng.integers(0, len(_ADMISSION_FAMILIES))]
        self._admission_counter += 1
        agent_id = f"adm_{family}_{self._admission_counter:06d}"

        sub_rng = registry.make_rng(agent_id)
        decision_offset = registry.make_decision_offset(agent_id)

        agent: Agent
        if family == "momentum":
            # Family-default N parameter; no jitter on N for v1
            agent = MomentumAgent(
                agent_id=agent_id,
                initial_wealth=self.initial_wealth,
                rng=sub_rng,
                decision_offset_ns=decision_offset,
            )
        elif family == "mean_reversion":
            agent = MeanReversionAgent(
                agent_id=agent_id,
                initial_wealth=self.initial_wealth,
                rng=sub_rng,
                decision_offset_ns=decision_offset,
            )
        elif family == "market_maker":
            agent = MarketMakerAgent(
                agent_id=agent_id,
                initial_wealth=self.initial_wealth,
                rng=sub_rng,
                decision_offset_ns=decision_offset,
            )
        elif family == "random":
            agent = RandomAgent(
                agent_id=agent_id,
                initial_wealth=self.initial_wealth,
                rng=sub_rng,
                decision_offset_ns=decision_offset,
            )
        elif family == "piggyback":
            agent = PiggybackAgent(
                agent_id=agent_id,
                initial_wealth=self.initial_wealth,
                rng=sub_rng,
                decision_offset_ns=decision_offset,
            )
        else:
            raise ValueError(f"Unknown family from admission draw: {family!r}")

        return agent
