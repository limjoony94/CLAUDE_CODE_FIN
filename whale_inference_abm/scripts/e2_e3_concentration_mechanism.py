"""E2 + E3: G2 diagnostic — concentration mechanism characterization.

Pre-registered in g2_diagnostic_prereg.md (option (b) per advisor 2026-05-01).

E2: Pareto-distributed initial wealth, no admissions, 5k bars.
    Tests whether wealth-weighted sizing AMPLIFIES given heterogeneity.
    Pareto(alpha=1.16, x_min=100) → ~80/20 inequality, no single agent dominant.

E3: Uniform initial wealth, no admissions, FIXED-SIZE trades (wealth-weighting disabled).
    Tests whether ANY concentration arises from skill differential alone.

Decision tree (advisor binding):
| E2 | E3 | Decision |
|----|----|----------|
| Amplifies (Gini >0.55) | Concentrates (>0.1) | (c) combined v2: Pareto + skill |
| Amplifies (>0.55) | Doesn't concentrate (<=0.1) | (c) seeded-Pareto v2 (cheaper) |
| Preserves (~0.5) | Concentrates (>0.1) | (c) skill-pivot, sizing optional |
| Preserves (~0.5) | Doesn't concentrate | (a) abandonment, v1 mechanisms insufficient |
| Shrinks (<0.45) | Doesn't concentrate | (a) + interesting destabilization finding |

Output: results/g2_concentration/e2_e3_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abm.admission import AdmissionScheduler
from abm.agents.market_maker import MarketMakerAgent
from abm.agents.mean_reversion import MeanReversionAgent
from abm.agents.momentum import MomentumAgent
from abm.agents.piggyback import PiggybackAgent
from abm.agents.random_agent import RandomAgent
from abm.constants import BAR_DURATION_NS
from abm.friction import Friction
from abm.metrics import gini, top_k_overlap, top_k_share
from abm.orderbook import Orderbook
from abm.registry import AgentRegistry
from abm.scheduler import Scheduler
from abm.simulation import NullLogger, Simulation
from abm.types import Order, OrderIntent, OrderType, Side
from abm.wealth import WealthTracker


def make_population(registry, wealth, agent_wealths: dict[str, float]):
    """Construct 15-agent population with given per-agent initial wealths."""
    momentum_specs = [(3, "n3"), (5, "n5"), (10, "n10")]
    meanrev_specs = [(10, "n10"), (20, "n20"), (30, "n30")]
    mm_specs = [(0.001, "a"), (0.0015, "b")]

    for n, suffix in momentum_specs:
        aid = f"momentum_{suffix}"
        w = agent_wealths[aid]
        registry.add_agent(MomentumAgent(
            agent_id=aid, initial_wealth=w,
            rng=registry.make_rng(aid),
            decision_offset_ns=registry.make_decision_offset(aid), N=n))
        wealth.initialize_agent(aid, initial_cash=w)
    for n in (10, 20, 30):
        aid = f"meanrev_n{n}"
        w = agent_wealths[aid]
        registry.add_agent(MeanReversionAgent(
            agent_id=aid, initial_wealth=w,
            rng=registry.make_rng(aid),
            decision_offset_ns=registry.make_decision_offset(aid), N=n))
        wealth.initialize_agent(aid, initial_cash=w)
    for spread, suffix in mm_specs:
        aid = f"mm_{suffix}"
        w = agent_wealths[aid]
        registry.add_agent(MarketMakerAgent(
            agent_id=aid, initial_wealth=w,
            rng=registry.make_rng(aid),
            decision_offset_ns=registry.make_decision_offset(aid),
            base_spread=spread))
        wealth.initialize_agent(aid, initial_cash=w)
    for i in range(5):
        aid = f"random_{i}"
        w = agent_wealths[aid]
        registry.add_agent(RandomAgent(
            agent_id=aid, initial_wealth=w,
            rng=registry.make_rng(aid),
            decision_offset_ns=registry.make_decision_offset(aid)))
        wealth.initialize_agent(aid, initial_cash=w)
    for i in range(2):
        aid = f"piggy_{i}"
        w = agent_wealths[aid]
        registry.add_agent(PiggybackAgent(
            agent_id=aid, initial_wealth=w,
            rng=registry.make_rng(aid),
            decision_offset_ns=registry.make_decision_offset(aid),
            lookback_bars=10))
        wealth.initialize_agent(aid, initial_cash=w)


AGENT_IDS = (
    [f"momentum_{s}" for _, s in [(3, "n3"), (5, "n5"), (10, "n10")]]
    + [f"meanrev_n{n}" for n in (10, 20, 30)]
    + [f"mm_{s}" for _, s in [(0.001, "a"), (0.0015, "b")]]
    + [f"random_{i}" for i in range(5)]
    + [f"piggy_{i}" for i in range(2)]
)
assert len(AGENT_IDS) == 15


def fixed_size_decorator(agent, fixed_size: float) -> None:
    """Wrap agent.decide() so all returned intents have size=fixed_size."""
    orig_decide = agent.decide

    def new_decide(snapshot, context):
        intents = orig_decide(snapshot, context)
        return [
            OrderIntent(
                order_type=i.order_type, side=i.side, size=fixed_size, price=i.price
            )
            for i in intents
        ]

    agent.decide = new_decide  # type: ignore[method-assign]


def build_sim(seed: int, terminal_bars: int, agent_wealths: dict[str, float],
              fixed_trade_size: float | None = None):
    initial_price = 50000.0
    terminal_ns = terminal_bars * BAR_DURATION_NS
    scheduler = Scheduler(seed=seed, terminal_time_ns=terminal_ns)
    orderbook = Orderbook(strict=False)
    registry = AgentRegistry(master_seed=seed)
    friction = Friction()
    wealth = WealthTracker()
    # No admissions for E2/E3
    admission = AdmissionScheduler(
        T_open_bars=1, T_extract_bars=terminal_bars, rate_lambda=1e-12
    )

    make_population(registry, wealth, agent_wealths)

    if fixed_trade_size is not None:
        for a in registry.alive_agents():
            fixed_size_decorator(a, fixed_trade_size)

    seed_bid = Order(order_id="seed_bid", agent_id="mm_a", order_type=OrderType.LIMIT,
                     side=Side.BUY, size=0.01, price=initial_price - 5.0,
                     sequence_no=scheduler.next_sequence_no())
    seed_ask = Order(order_id="seed_ask", agent_id="mm_a", order_type=OrderType.LIMIT,
                     side=Side.SELL, size=0.01, price=initial_price + 5.0,
                     sequence_no=scheduler.next_sequence_no())
    orderbook.submit(seed_bid, 0)
    orderbook.submit(seed_ask, 0)

    sim = Simulation(scheduler=scheduler, orderbook=orderbook, registry=registry,
                     friction=friction, wealth_tracker=wealth,
                     admission_scheduler=admission, logger=NullLogger(),
                     piggyback_lookback_bars=10)
    sim.seed_initial_events()
    return sim


def evaluate_at_end(sim, terminal_bars: int):
    history = sim.wealth_tracker._history
    if not history:
        return {"gini_at_end": 0.0, "n_alive_at_end": 0, "wealth_dist": {}}
    snap_at_end = history[-1][1]
    wealths = list(snap_at_end.values())
    return {
        "gini_at_end": round(gini(wealths), 4),
        "top_5pct_share": round(top_k_share(wealths, 0.05), 4),
        "n_alive_at_end": len(snap_at_end),
        "wealth_dist": {a: round(w, 2) for a, w in snap_at_end.items()},
    }


def run_E2(seed: int = 42, terminal_bars: int = 5000) -> dict:
    """Pareto initial wealth, no admissions, wealth-weighted sizing on (default)."""
    rng = np.random.default_rng(seed)
    # Pareto x10 scale (advisor calibration intent: x_min=1000 to keep all agents
    # above MIN_ORDER_SIZE * mid trading floor). Original x_min=100 sub-MIN_ORDER_SIZE
    # locked random agents out of trading and froze the simulation.
    pareto_wealths = (rng.pareto(1.16, size=15) * 100 + 100) * 10
    pareto_wealths = np.clip(pareto_wealths, 1000, 100000)
    initial_dict = {aid: float(pareto_wealths[i]) for i, aid in enumerate(AGENT_IDS)}
    initial_gini = gini(list(initial_dict.values()))

    t0 = time.time()
    sim = build_sim(seed=seed, terminal_bars=terminal_bars, agent_wealths=initial_dict)
    sim.run()
    elapsed = time.time() - t0

    eval_end = evaluate_at_end(sim, terminal_bars)
    return {
        "experiment": "E2_Pareto_initial",
        "seed": seed,
        "terminal_bars": terminal_bars,
        "elapsed_sec": round(elapsed, 1),
        "initial_wealths": {a: round(w, 2) for a, w in initial_dict.items()},
        "initial_gini": round(initial_gini, 4),
        "final_gini": eval_end["gini_at_end"],
        "amplification": round(eval_end["gini_at_end"] - initial_gini, 4),
        "top_5pct_share": eval_end["top_5pct_share"],
        "n_alive_at_end": eval_end["n_alive_at_end"],
    }


def run_E3(seed: int = 42, terminal_bars: int = 5000) -> dict:
    """Uniform initial wealth, no admissions, FIXED 0.001 BTC trade size."""
    initial_dict = {aid: 1000.0 for aid in AGENT_IDS}
    initial_gini = gini(list(initial_dict.values()))

    t0 = time.time()
    sim = build_sim(seed=seed, terminal_bars=terminal_bars,
                    agent_wealths=initial_dict, fixed_trade_size=0.001)
    sim.run()
    elapsed = time.time() - t0

    eval_end = evaluate_at_end(sim, terminal_bars)
    return {
        "experiment": "E3_fixed_size_no_wealth_weighting",
        "seed": seed,
        "terminal_bars": terminal_bars,
        "elapsed_sec": round(elapsed, 1),
        "initial_gini": round(initial_gini, 4),
        "final_gini": eval_end["gini_at_end"],
        "concentration_from_skill_alone": round(eval_end["gini_at_end"] - initial_gini, 4),
        "top_5pct_share": eval_end["top_5pct_share"],
        "n_alive_at_end": eval_end["n_alive_at_end"],
    }


def run_E2_extended(seed: int = 42, terminal_bars: int = 20000) -> dict:
    """E2 extended: same Pareto x10 setup but 20k bars with checkpoints at 5k/10k/15k/20k.

    Per advisor (c) sub-variant decision (2026-05-01): tests whether the +0.051/5k bars
    amplification observed in E2 scales linearly (→ crosses 0.55 at ~10k more bars,
    seeded-Pareto v2 viable) or plateaus (→ confirms (a) abandonment).
    """
    rng = np.random.default_rng(seed)
    pareto_wealths = (rng.pareto(1.16, size=15) * 100 + 100) * 10
    pareto_wealths = np.clip(pareto_wealths, 1000, 100000)
    initial_dict = {aid: float(pareto_wealths[i]) for i, aid in enumerate(AGENT_IDS)}
    initial_gini = gini(list(initial_dict.values()))

    t0 = time.time()
    sim = build_sim(seed=seed, terminal_bars=terminal_bars, agent_wealths=initial_dict)
    sim.run()
    elapsed = time.time() - t0

    history = sim.wealth_tracker._history  # list[(ts_ns, dict[aid, wealth])]
    checkpoint_bars = [5000, 10000, 15000, 20000]
    checkpoints: list[dict] = []
    for cb in checkpoint_bars:
        target_ns = cb * BAR_DURATION_NS
        snap_at = None
        for ts, snap in history:
            if ts <= target_ns:
                snap_at = snap
            else:
                break
        if snap_at is None:
            snap_at = history[-1][1] if history else {}
        wealths = list(snap_at.values())
        checkpoints.append({
            "bar": cb,
            "n_alive": len(snap_at),
            "gini": round(gini(wealths), 4),
            "top_5pct_share": round(top_k_share(wealths, 0.05), 4),
        })

    eval_end = evaluate_at_end(sim, terminal_bars)
    return {
        "experiment": "E2_extended_20k_bars",
        "seed": seed,
        "terminal_bars": terminal_bars,
        "elapsed_sec": round(elapsed, 1),
        "initial_gini": round(initial_gini, 4),
        "final_gini": eval_end["gini_at_end"],
        "amplification_total": round(eval_end["gini_at_end"] - initial_gini, 4),
        "checkpoints": checkpoints,
        "trade_count": sim.orderbook._trade_counter,
    }


def main_extended() -> None:
    """E2-extended only: 20k bars + checkpoints. Apply trajectory decision tree."""
    print("Running E2-extended (Pareto x10 initial, 20000 bars, checkpoints)...")
    result = run_E2_extended(terminal_bars=20000)
    print(json.dumps(result, indent=2))
    print()

    # Apply trajectory decision tree
    cps = result["checkpoints"]
    ginis = [c["gini"] for c in cps]
    print("Trajectory analysis:")
    print(f"  Initial Gini:         {result['initial_gini']:.4f}")
    for cp in cps:
        print(f"  Bar {cp['bar']:>5d}: Gini = {cp['gini']:.4f} (n_alive={cp['n_alive']})")
    print()

    # Classification
    crossed_055 = any(g > 0.55 for g in ginis)
    final_gini = ginis[-1]
    growth_5k_to_20k = ginis[-1] - ginis[0]

    print("Verdict per advisor trajectory decision tree:")
    if crossed_055 and growth_5k_to_20k > 0.05:
        verdict = "(c) seeded-Pareto v2 VIABLE"
        details = "Gini crosses 0.55 with monotonic growth → G2 criterion update + design v0.8 patch + T-G3 unblocks"
    elif final_gini < 0.51 and growth_5k_to_20k < 0.02:
        verdict = "(a) abandonment CONFIRMED"
        details = "Gini plateaus → mechanism produces one-time burst then stalls"
    elif 0.50 <= final_gini <= 0.55:
        verdict = "BORDERLINE — surface to user"
        details = f"Final Gini {final_gini:.4f} in (0.50, 0.55) range. User-level decision needed."
    else:
        verdict = "WEIRD — does not fit advisor decision tree"
        details = f"final={final_gini}, growth_5k_to_20k={growth_5k_to_20k:.4f}. Call advisor."

    print(f"VERDICT: {verdict}")
    print(f"  {details}")

    out_path = Path(__file__).resolve().parent.parent / "results" / "g2_concentration" / "e2_extended_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "result": result,
            "trajectory_analysis": {
                "checkpoints": cps,
                "crossed_0_55": crossed_055,
                "growth_5k_to_20k": round(growth_5k_to_20k, 4),
                "verdict": verdict,
                "details": details,
            },
        }, f, indent=2)
    print(f"\nResults: {out_path}")


def main() -> None:
    print("Running E2 (Pareto initial, wealth-weighted sizing on)...")
    e2 = run_E2()
    print(json.dumps(e2, indent=2))
    print()

    print("Running E3 (uniform initial, fixed-size trades, no wealth-weighting)...")
    e3 = run_E3()
    print(json.dumps(e3, indent=2))
    print()

    # Apply advisor decision tree
    e2_amp = e2["amplification"]
    e2_final = e2["final_gini"]
    e3_concentration = e3["concentration_from_skill_alone"]

    e2_amplifies = e2_final > 0.55
    e2_preserves = 0.45 <= e2_final <= 0.55
    e2_shrinks = e2_final < 0.45
    e3_concentrates = e3_concentration > 0.1

    print("=" * 70)
    print("ADVISOR DECISION TREE APPLIED:")
    print("=" * 70)
    print(f"E2 final Gini: {e2_final} (initial {e2['initial_gini']}, amp {e2_amp:+})")
    print(f"  Verdict: {'AMPLIFIES' if e2_amplifies else 'PRESERVES' if e2_preserves else 'SHRINKS'}")
    print(f"E3 final Gini: {e3['final_gini']} (concentration from skill: {e3_concentration:+})")
    print(f"  Verdict: {'CONCENTRATES' if e3_concentrates else 'NO CONCENTRATION'}")
    print()

    if e2_amplifies and e3_concentrates:
        verdict = "(c) combined v2: Pareto + skill-driven concentration"
    elif e2_amplifies and not e3_concentrates:
        verdict = "(c) seeded-Pareto v2 (cheaper than full skill pivot)"
    elif e2_preserves and e3_concentrates:
        verdict = "(c) skill-pivot, wealth-weighted sizing optional"
    elif e2_preserves and not e3_concentrates:
        verdict = "(a) abandonment — v1 mechanisms structurally insufficient"
    elif e2_shrinks and not e3_concentrates:
        verdict = "(a) abandonment + interesting destabilization research finding"
    else:
        verdict = "WEIRD — does not fit decision table. Call advisor."

    print(f"VERDICT: {verdict}")
    print()

    # Persist
    out_path = Path(__file__).resolve().parent.parent / "results" / "g2_concentration" / "e2_e3_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "e2": e2,
            "e3": e3,
            "verdict": verdict,
            "decision_tree_inputs": {
                "e2_final_gini": e2_final,
                "e2_amplifies": e2_amplifies,
                "e2_preserves": e2_preserves,
                "e2_shrinks": e2_shrinks,
                "e3_concentration": e3_concentration,
                "e3_concentrates": e3_concentrates,
            },
        }, f, indent=2)
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
