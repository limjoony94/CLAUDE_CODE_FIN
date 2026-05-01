"""E4: G2 diagnostic — frozen-window-aware Gini (advisor amendment).

Pre-registered in results/g2_concentration/g2_diagnostic_prereg.md.

Tests whether wealth concentration emerges among AGENTS PRESENT AT T_open boundary
during the frozen-admission window (T_open to T_open + T_extract). This matches the
G3 substrate-extraction window where admissions are by-design disabled.

Setup:
  - T_open = 5000 bars (admissions on, normal rate 1/600)
  - T_extract = 5000 bars (frozen, no new admissions)
  - At T=5000 (T_open boundary): record agent_ids alive
  - At T=10000: compute Gini both for ALL agents AND for incumbents-at-T_open only

Output: results/g2_concentration/e4_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abm.constants import BAR_DURATION_NS
from abm.metrics import evaluate_concentration, gini, top_k_overlap, top_k_share
from scripts.e1_admission_rate_sweep import build_smoke_with_rate


def main() -> None:
    # Match the G2 baseline setup: rate_lambda=1/600 (current), 10k bars total
    # T_open=5000 in the AdmissionScheduler (frozen for last 5000 bars)
    print("Running 10k bar with T_open=5000 (frozen-admission for last 5k bars)...")
    t0 = time.time()
    # Note: build_smoke_with_rate sets T_open_bars = terminal_bars * 2, so 10k bars
    # has T_open=20000 (always open). We need T_open=5000 explicitly.
    # Need a custom build for this — copy the build_smoke_with_rate logic with override.

    from abm.admission import AdmissionScheduler
    from abm.agents.market_maker import MarketMakerAgent
    from abm.agents.mean_reversion import MeanReversionAgent
    from abm.agents.momentum import MomentumAgent
    from abm.agents.piggyback import PiggybackAgent
    from abm.agents.random_agent import RandomAgent
    from abm.friction import Friction
    from abm.orderbook import Orderbook
    from abm.registry import AgentRegistry
    from abm.scheduler import Scheduler
    from abm.simulation import NullLogger, Simulation
    from abm.types import Order, OrderType, Side
    from abm.wealth import WealthTracker

    seed = 42
    terminal_bars = 10000
    T_open_bars = 5000
    initial_price = 50000.0
    terminal_ns = terminal_bars * BAR_DURATION_NS
    T_open_ns = T_open_bars * BAR_DURATION_NS

    scheduler = Scheduler(seed=seed, terminal_time_ns=terminal_ns)
    orderbook = Orderbook(strict=False)
    registry = AgentRegistry(master_seed=seed)
    friction = Friction()
    wealth = WealthTracker()
    admission = AdmissionScheduler(
        T_open_bars=T_open_bars,
        T_extract_bars=terminal_bars - T_open_bars,
        rate_lambda=1.0 / 600.0,
    )

    # 15 incumbents
    for n, suffix in [(3, "n3"), (5, "n5"), (10, "n10")]:
        aid = f"momentum_{suffix}"
        registry.add_agent(MomentumAgent(
            agent_id=aid, initial_wealth=1000.0,
            rng=registry.make_rng(aid),
            decision_offset_ns=registry.make_decision_offset(aid), N=n))
        wealth.initialize_agent(aid, initial_cash=1000.0)
    for n in (10, 20, 30):
        aid = f"meanrev_n{n}"
        registry.add_agent(MeanReversionAgent(
            agent_id=aid, initial_wealth=1000.0,
            rng=registry.make_rng(aid),
            decision_offset_ns=registry.make_decision_offset(aid), N=n))
        wealth.initialize_agent(aid, initial_cash=1000.0)
    for spread, suffix in [(0.001, "a"), (0.0015, "b")]:
        aid = f"mm_{suffix}"
        registry.add_agent(MarketMakerAgent(
            agent_id=aid, initial_wealth=1000.0,
            rng=registry.make_rng(aid),
            decision_offset_ns=registry.make_decision_offset(aid),
            base_spread=spread))
        wealth.initialize_agent(aid, initial_cash=1000.0)
    for i in range(5):
        aid = f"random_{i}"
        registry.add_agent(RandomAgent(
            agent_id=aid, initial_wealth=1000.0,
            rng=registry.make_rng(aid),
            decision_offset_ns=registry.make_decision_offset(aid)))
        wealth.initialize_agent(aid, initial_cash=1000.0)
    for i in range(2):
        aid = f"piggy_{i}"
        registry.add_agent(PiggybackAgent(
            agent_id=aid, initial_wealth=1000.0,
            rng=registry.make_rng(aid),
            decision_offset_ns=registry.make_decision_offset(aid),
            lookback_bars=10))
        wealth.initialize_agent(aid, initial_cash=1000.0)

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
    sim.run()
    elapsed = time.time() - t0

    # Find snapshot at T_open boundary and at T_end
    history = wealth.snapshot_method_alias_for_diagnostic = wealth._history
    snap_at_Topen = None
    snap_at_end = None
    actual_Topen_idx = None
    for i, (ts, snap) in enumerate(history):
        if ts <= T_open_ns:
            snap_at_Topen = snap
            actual_Topen_idx = i
        if ts <= terminal_ns:
            snap_at_end = snap
        if ts > terminal_ns:
            break

    incumbents_at_Topen = set(snap_at_Topen.keys()) if snap_at_Topen else set()
    snap_at_end_all = snap_at_end or {}
    snap_at_end_incumbents_only = {a: w for a, w in snap_at_end_all.items() if a in incumbents_at_Topen}

    # Metrics
    gini_all_at_end = gini(list(snap_at_end_all.values()))
    gini_incumbents_at_end = gini(list(snap_at_end_incumbents_only.values()))
    top_5pct_share_all = top_k_share(list(snap_at_end_all.values()), 0.05)
    top_5pct_share_incumbents = top_k_share(list(snap_at_end_incumbents_only.values()), 0.05)
    overlap_Topen_to_end_incumbents = top_k_overlap(snap_at_Topen, snap_at_end_incumbents_only, 0.05)

    result = {
        "T_open_bars": T_open_bars,
        "T_extract_bars": terminal_bars - T_open_bars,
        "elapsed_sec": round(elapsed, 1),
        "n_agents_at_Topen": len(incumbents_at_Topen),
        "n_agents_at_end_all": len(snap_at_end_all),
        "n_incumbents_alive_at_end": len(snap_at_end_incumbents_only),
        "gini_at_end_all_agents": round(gini_all_at_end, 4),
        "gini_at_end_incumbents_only": round(gini_incumbents_at_end, 4),
        "top_5pct_share_at_end_all": round(top_5pct_share_all, 4),
        "top_5pct_share_at_end_incumbents_only": round(top_5pct_share_incumbents, 4),
        "top_5pct_overlap_Topen_to_end_among_incumbents": round(overlap_Topen_to_end_incumbents, 4),
    }
    print(json.dumps(result, indent=2))

    out_path = Path(__file__).resolve().parent.parent / "results" / "g2_concentration" / "e4_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nE4 complete. Results: {out_path}")


if __name__ == "__main__":
    main()
