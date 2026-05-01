"""Agent registry — derives sub-seeds, manages lifecycle, tracks alive agents.

Design ref: docs/02-design/features/whale_inference_abm.design.md Sections 4.7, 9.1.

Determinism: each agent's RNG is seeded from (master_seed, agent_id_hash). Same
master_seed + same agent_id => same sub-seed => same RNG draws.
"""

from __future__ import annotations

import hashlib
from typing import Iterator

import numpy as np

from abm.agents.base import Agent
from abm.constants import BAR_DURATION_NS


class AgentRegistry:
    def __init__(self, master_seed: int) -> None:
        self._master_seed: int = master_seed
        self._agents: dict[str, Agent] = {}
        self._removed_ids: set[str] = set()  # bankrupt agents preserved for tape lookup

    @property
    def master_seed(self) -> int:
        return self._master_seed

    def derived_seed(self, agent_id: str) -> int:
        """Deterministic sub-seed = hash(master_seed || agent_id) mod 2^32."""
        h = hashlib.sha256(f"{self._master_seed}:{agent_id}".encode("utf-8")).digest()
        return int.from_bytes(h[:4], "big")

    def make_rng(self, agent_id: str) -> np.random.Generator:
        return np.random.default_rng(self.derived_seed(agent_id))

    def make_decision_offset(self, agent_id: str) -> int:
        """Per-agent jitter offset (advisor B1 patch). Drawn from sub-RNG once at add time."""
        rng = self.make_rng(agent_id + "_offset")  # separate sub-stream
        return int(rng.integers(0, BAR_DURATION_NS))

    def add_agent(self, agent: Agent) -> None:
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent {agent.agent_id!r} already in registry")
        if agent.agent_id in self._removed_ids:
            raise ValueError(
                f"Agent {agent.agent_id!r} previously removed; cannot re-add (would break tape lookup)"
            )
        self._agents[agent.agent_id] = agent

    def remove_agent(self, agent_id: str) -> None:
        if agent_id not in self._agents:
            raise KeyError(f"Agent {agent_id!r} not in registry")
        del self._agents[agent_id]
        self._removed_ids.add(agent_id)

    def get(self, agent_id: str) -> Agent:
        return self._agents[agent_id]

    def has(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def alive_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def alive_ids(self) -> list[str]:
        return list(self._agents.keys())

    def __len__(self) -> int:
        return len(self._agents)

    def __iter__(self) -> Iterator[Agent]:
        return iter(self._agents.values())
