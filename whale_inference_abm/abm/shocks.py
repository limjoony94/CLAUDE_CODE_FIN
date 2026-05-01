"""External wealth shock scheduler — v2 ABM architecture.

Per advisor 2026-05-01: tests whether v1 wealth-weighted-sizing mechanism, when
given exogenous wealth perturbations, can convert random injections into emergent
persistent whales (vs. just transient noise).

Design:
- Periodic shocks every `shock_interval_bars` (default 1000)
- Selection: uniform random over alive agents (NOT biased to rich)
  Rationale: biased-to-rich is tautological; uniform tests mechanism amplification
- Magnitude: multiplicative factor (default 2.0 = double wealth)
  Asymmetric (no negative shock counterpart) — pure injection
- RNG: external (passed in from scheduler.rng), preserves single-RNG determinism
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abm.constants import BAR_DURATION_NS


@dataclass
class ShockScheduler:
    """Periodic external wealth shocks. Caller schedules ShockEvent via scheduler."""

    shock_interval_bars: int = 1000
    shock_magnitude: float = 2.0  # multiplicative; 2.0 = double wealth
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.shock_interval_bars <= 0:
            raise ValueError(f"shock_interval_bars must be > 0, got {self.shock_interval_bars}")
        if self.shock_magnitude <= 0:
            raise ValueError(f"shock_magnitude must be > 0, got {self.shock_magnitude}")
        self._shock_counter: int = 0

    @property
    def shock_interval_ns(self) -> int:
        return self.shock_interval_bars * BAR_DURATION_NS

    def select_target_agent(
        self, rng: np.random.Generator, alive_agent_ids: list[str]
    ) -> str | None:
        """Uniform random selection from alive agents."""
        if not alive_agent_ids:
            return None
        sorted_ids = sorted(alive_agent_ids)  # deterministic order before random pick
        idx = int(rng.integers(0, len(sorted_ids)))
        return sorted_ids[idx]

    def apply_shock(self, current_wealth: float) -> float:
        """Compute new wealth after applying multiplicative shock."""
        return current_wealth * self.shock_magnitude

    def next_shock_index(self) -> int:
        self._shock_counter += 1
        return self._shock_counter
