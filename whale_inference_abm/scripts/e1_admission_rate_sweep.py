"""E1: G2 diagnostic — admission rate sweep.

Pre-registered in results/g2_concentration/g2_diagnostic_prereg.md.
Runs 10k-bar smoke at 4 admission rates: [1/600 (current), 1/3600, 1/36000, 0].
Reports Gini at T=10k for each.

Output: results/g2_concentration/e1_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abm.admission import AdmissionScheduler
from abm.agents.market_maker import MarketMakerAgent
from abm.agents.mean_reversion import MeanReversionAgent
from abm.agents.momentum import MomentumAgent
from abm.agents.piggyback import PiggybackAgent
from abm.agents.random_agent import RandomAgent
from abm.constants import BAR_DURATION_NS
from abm.friction import Friction
from abm.metrics import evaluate_concentration
from abm.orderbook import Orderbook
from abm.registry import AgentRegistry
from abm.scheduler import Scheduler
from abm.simulation import NullLogger, Simulation
from abm.types import Order, OrderType, Side
from abm.wealth import WealthTracker


def build_smoke_with_rate(seed: int, terminal_bars: int, rate_lambda: float):
    initial_price = 50000.0
    terminal_ns = terminal_bars * BAR_DURATION_NS
    scheduler = Scheduler(seed=seed, terminal_time_ns=terminal_ns)
    orderbook = Orderbook(strict=False)
    registry = AgentRegistry(master_seed=seed)
    friction = Friction()
    wealth = WealthTracker()
    # Use rate_lambda parameter; if 0, no admissions (effectively rate_lambda very small)
    effective_rate = rate_lambda if rate_lambda > 0 else 1e-12
    admission = AdmissionScheduler(
        T_open_bars=terminal_bars * 2,
        T_extract_bars=terminal_bars,
        rate_lambda=effective_rate,
    )

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
    return sim


def main() -> None:
    rates = [
        ("1/600", 1.0 / 600.0),     # current G2 baseline
        ("1/3600", 1.0 / 3600.0),   # 6× lower
        ("1/36000", 1.0 / 36000.0), # 60× lower
        ("0", 0.0),                  # no admissions
    ]
    results = []
    for label, rate in rates:
        t0 = time.time()
        sim = build_smoke_with_rate(seed=42, terminal_bars=10000, rate_lambda=rate)
        sim.run()
        elapsed = time.time() - t0
        history = sim.wealth_tracker._history
        eval_result = evaluate_concentration(
            history, bar_indices_to_compare=(5000, 10000), k_pct=0.05
        )
        n_alive = len(sim.registry)
        result = {
            "rate_label": label,
            "rate_lambda": rate,
            "elapsed_sec": round(elapsed, 1),
            "n_alive_at_end": n_alive,
            "n_admissions": n_alive - 15,
            "gini_at_10k": round(eval_result["gini_at_late"], 4),
            "top_5pct_share_at_10k": round(eval_result["top_k_share_at_late"], 4),
            "top_5pct_overlap_5k_10k": round(eval_result["top_k_overlap_early_late"], 4),
        }
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    out_path = Path(__file__).resolve().parent.parent / "results" / "g2_concentration" / "e1_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nE1 sweep complete. Results: {out_path}")


if __name__ == "__main__":
    main()
