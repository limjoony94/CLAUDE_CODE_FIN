"""AdmissionScheduler tests."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from abm.admission import AdmissionScheduler
from abm.constants import (
    ADMISSION_INITIAL_WEALTH,
    BAR_DURATION_NS,
    DEFAULT_T_EXTRACT_BARS,
    DEFAULT_T_OPEN_BARS,
    NS_PER_SECOND,
)
from abm.registry import AgentRegistry


# ----- Construction + invariants -----

def test_default_construction() -> None:
    a = AdmissionScheduler()
    assert a.T_open_bars == DEFAULT_T_OPEN_BARS
    assert a.T_extract_bars == DEFAULT_T_EXTRACT_BARS


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="T_open_bars must be > 0"):
        AdmissionScheduler(T_open_bars=0)
    with pytest.raises(ValueError, match="T_extract_bars must be > 0"):
        AdmissionScheduler(T_extract_bars=0)
    with pytest.raises(ValueError, match="rate_lambda must be > 0"):
        AdmissionScheduler(rate_lambda=0)


def test_T_open_ns_computation() -> None:
    a = AdmissionScheduler(T_open_bars=10)
    assert a.T_open_ns == 10 * BAR_DURATION_NS


def test_terminal_time_ns_sums_open_extract() -> None:
    a = AdmissionScheduler(T_open_bars=7000, T_extract_bars=3000)
    assert a.terminal_time_ns == 10_000 * BAR_DURATION_NS


# ----- Frozen-window check -----

def test_is_open_phase_pre_T_open() -> None:
    a = AdmissionScheduler(T_open_bars=100)
    assert a.is_open_phase(now_ns=50 * BAR_DURATION_NS) is True


def test_is_open_phase_at_T_open_is_false() -> None:
    """Boundary: at exactly T_open_ns, frozen phase begins (closed-open interval)."""
    a = AdmissionScheduler(T_open_bars=100)
    assert a.is_open_phase(now_ns=100 * BAR_DURATION_NS) is False


def test_is_open_phase_post_T_open() -> None:
    a = AdmissionScheduler(T_open_bars=100)
    assert a.is_open_phase(now_ns=150 * BAR_DURATION_NS) is False


# ----- Poisson delay -----

def test_next_admission_delay_positive() -> None:
    a = AdmissionScheduler(rate_lambda=1.0 / 60.0)  # 1 per minute
    rng = np.random.default_rng(42)
    delays = [a.next_admission_delay_ns(rng) for _ in range(100)]
    assert all(d >= 1 for d in delays)


def test_next_admission_delay_mean_near_inverse_lambda() -> None:
    """Sanity: mean delay should be near 1/lambda seconds (Poisson exponential)."""
    a = AdmissionScheduler(rate_lambda=1.0 / 60.0)  # avg 60s
    rng = np.random.default_rng(42)
    delays_sec = [a.next_admission_delay_ns(rng) / NS_PER_SECOND for _ in range(1000)]
    mean_sec = sum(delays_sec) / len(delays_sec)
    # Loose bounds for 1000 samples; theoretical mean = 60s, std/sqrt(N) ~ 1.9
    assert 50 < mean_sec < 70


def test_delay_deterministic_for_same_rng_seed() -> None:
    a = AdmissionScheduler()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    delays1 = [a.next_admission_delay_ns(rng1) for _ in range(20)]
    delays2 = [a.next_admission_delay_ns(rng2) for _ in range(20)]
    assert delays1 == delays2


# ----- create_new_agent -----

def test_create_new_agent_returns_valid_agent() -> None:
    a = AdmissionScheduler()
    reg = AgentRegistry(master_seed=42)
    rng = np.random.default_rng(42)
    agent = a.create_new_agent(rng, reg, now_ns=0)
    assert agent.agent_id.startswith("adm_")
    assert agent.initial_wealth == ADMISSION_INITIAL_WEALTH
    assert 0 <= agent.decision_offset_ns < BAR_DURATION_NS
    assert agent.family in (
        "momentum",
        "mean_reversion",
        "market_maker",
        "random",
        "piggyback",
    )


def test_create_unique_agent_ids() -> None:
    a = AdmissionScheduler()
    reg = AgentRegistry(master_seed=42)
    rng = np.random.default_rng(42)
    ids = {a.create_new_agent(rng, reg, now_ns=0).agent_id for _ in range(50)}
    assert len(ids) == 50  # all unique


def test_family_distribution_uniform() -> None:
    """Over many admissions, family distribution should be roughly uniform 1/5 each."""
    a = AdmissionScheduler()
    reg = AgentRegistry(master_seed=42)
    rng = np.random.default_rng(42)
    families = [a.create_new_agent(rng, reg, now_ns=0).family for _ in range(500)]
    counts = Counter(families)
    # Expected: 100 each of 5 families. Multinomial std ~ sqrt(500*0.2*0.8) ~ 8.9
    # 70-130 is within ~3.5 std for 500 samples
    for fam in ["momentum", "mean_reversion", "market_maker", "random", "piggyback"]:
        assert 70 <= counts[fam] <= 130, f"{fam}: {counts[fam]}"


def test_create_new_agent_deterministic_for_same_rng_state() -> None:
    """Two AdmissionSchedulers + same registry seed + same RNG state → same agent stream."""
    a1 = AdmissionScheduler()
    a2 = AdmissionScheduler()
    reg1 = AgentRegistry(master_seed=42)
    reg2 = AgentRegistry(master_seed=42)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    ids1 = [a1.create_new_agent(rng1, reg1, 0).agent_id for _ in range(20)]
    ids2 = [a2.create_new_agent(rng2, reg2, 0).agent_id for _ in range(20)]
    assert ids1 == ids2
