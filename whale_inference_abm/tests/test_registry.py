"""AgentRegistry tests."""

from __future__ import annotations

import pytest

from abm.agents.momentum import MomentumAgent
from abm.constants import BAR_DURATION_NS
from abm.registry import AgentRegistry


def _make_momentum(reg: AgentRegistry, agent_id: str) -> MomentumAgent:
    return MomentumAgent(
        agent_id=agent_id,
        initial_wealth=1000.0,
        rng=reg.make_rng(agent_id),
        decision_offset_ns=reg.make_decision_offset(agent_id),
    )


def test_derived_seed_deterministic_same_master_same_id() -> None:
    r1 = AgentRegistry(master_seed=42)
    r2 = AgentRegistry(master_seed=42)
    assert r1.derived_seed("agent_x") == r2.derived_seed("agent_x")


def test_derived_seed_different_master_different() -> None:
    r1 = AgentRegistry(master_seed=42)
    r2 = AgentRegistry(master_seed=43)
    assert r1.derived_seed("agent_x") != r2.derived_seed("agent_x")


def test_derived_seed_different_id_different() -> None:
    r = AgentRegistry(master_seed=42)
    assert r.derived_seed("agent_a") != r.derived_seed("agent_b")


def test_decision_offset_in_range() -> None:
    r = AgentRegistry(master_seed=42)
    for i in range(50):
        offset = r.make_decision_offset(f"agent_{i}")
        assert 0 <= offset < BAR_DURATION_NS


def test_decision_offset_deterministic() -> None:
    r1 = AgentRegistry(master_seed=42)
    r2 = AgentRegistry(master_seed=42)
    assert r1.make_decision_offset("agent_x") == r2.make_decision_offset("agent_x")


def test_decision_offsets_distributed_across_bar() -> None:
    """B1 patch: offsets should distribute (not all alphabetically clustered)."""
    r = AgentRegistry(master_seed=42)
    offsets = [r.make_decision_offset(f"agent_{i:03d}") for i in range(20)]
    # span >= half of bar duration is reasonable for 20 random samples
    span = max(offsets) - min(offsets)
    assert span > BAR_DURATION_NS / 2


def test_add_and_get_agent() -> None:
    r = AgentRegistry(master_seed=42)
    a = _make_momentum(r, "mo_n5_1")
    r.add_agent(a)
    assert r.has("mo_n5_1")
    assert r.get("mo_n5_1") is a
    assert len(r) == 1


def test_duplicate_add_raises() -> None:
    r = AgentRegistry(master_seed=42)
    r.add_agent(_make_momentum(r, "mo1"))
    with pytest.raises(ValueError, match="already in registry"):
        r.add_agent(_make_momentum(r, "mo1"))


def test_remove_agent() -> None:
    r = AgentRegistry(master_seed=42)
    r.add_agent(_make_momentum(r, "mo1"))
    r.remove_agent("mo1")
    assert not r.has("mo1")
    assert len(r) == 0


def test_remove_then_readd_raises() -> None:
    """Tape lookup integrity: removed agent_id must not be reused."""
    r = AgentRegistry(master_seed=42)
    r.add_agent(_make_momentum(r, "mo1"))
    r.remove_agent("mo1")
    with pytest.raises(ValueError, match="previously removed"):
        r.add_agent(_make_momentum(r, "mo1"))


def test_remove_unknown_raises() -> None:
    r = AgentRegistry(master_seed=42)
    with pytest.raises(KeyError):
        r.remove_agent("nonexistent")


def test_alive_agents_excludes_removed() -> None:
    r = AgentRegistry(master_seed=42)
    r.add_agent(_make_momentum(r, "a"))
    r.add_agent(_make_momentum(r, "b"))
    r.add_agent(_make_momentum(r, "c"))
    r.remove_agent("b")
    alive_ids = sorted(r.alive_ids())
    assert alive_ids == ["a", "c"]
