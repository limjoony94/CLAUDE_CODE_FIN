"""v2 ABM external shocks experiment.

Per advisor 2026-05-01 (c1 sub-variant): test whether v1 wealth-weighted-sizing
mechanism amplifies exogenous wealth perturbations into emergent persistent whales.

Setup:
- 15 canonical agents, uniform 1000 initial wealth
- No admissions (rate=1e-12 effectively zero)
- 10 shocks at every 1000 bars (×2 wealth, uniform random target)
- 10000 bar evaluation horizon
- Checkpoints at every 1000 bars (after each shock)

Decision tree (advisor binding):
| Trajectory | Top-5% identity | Verdict |
|-----------|-----------------|---------|
| Gini > 0.55 monotonic growth | Stable (overlap >= 0.5) | (c1) v2 viable, design v0.9, T-G3 unblocks |
| Gini > 0.55 | Churns each shock (overlap < 0.3) | (c2) ambiguous, surface to user |
| Gini < 0.51 plateau | (any) | (a) confirmed across v2 path 1 of 3 |

Output: results/g2_concentration/v2_shocks_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abm.constants import BAR_DURATION_NS
from abm.metrics import gini, top_k_overlap, top_k_share
from abm.shocks import ShockScheduler
from scripts.e2_e3_concentration_mechanism import AGENT_IDS, build_sim


def run_v2_shocks(seed: int = 42, terminal_bars: int = 10000,
                   shock_interval_bars: int = 1000, shock_magnitude: float = 2.0) -> dict:
    init = {aid: 1000.0 for aid in AGENT_IDS}
    shock = ShockScheduler(
        shock_interval_bars=shock_interval_bars,
        shock_magnitude=shock_magnitude,
    )

    t0 = time.time()
    sim = build_sim(
        seed=seed, terminal_bars=terminal_bars,
        agent_wealths=init, shock_scheduler=shock,
    )
    sim.run()
    elapsed = time.time() - t0

    history = sim.wealth_tracker._history
    # Checkpoint at every 1000 bars
    checkpoint_bars = list(range(1000, terminal_bars + 1, 1000))
    checkpoints = []
    snap_at_checkpoints = {}
    for cb in checkpoint_bars:
        target_ns = cb * BAR_DURATION_NS
        snap_at = None
        for ts, snap in history:
            if ts <= target_ns:
                snap_at = snap
            else:
                break
        if snap_at is None and history:
            snap_at = history[-1][1]
        snap_at_checkpoints[cb] = snap_at or {}
        wealths = list((snap_at or {}).values())
        # Top-5% identity (sorted desc, take top 5%)
        sorted_desc = sorted((snap_at or {}).items(), key=lambda x: x[1], reverse=True)
        k = max(1, int(len(sorted_desc) * 0.05))
        top_5pct_ids = {aid for aid, _ in sorted_desc[:k]}
        checkpoints.append({
            "bar": cb,
            "n_alive": len(snap_at or {}),
            "gini": round(gini(wealths), 4),
            "top_5pct_share": round(top_k_share(wealths, 0.05), 4),
            "top_5pct_ids": sorted(top_5pct_ids),
        })

    # Top-5% rank stability across consecutive checkpoints
    overlaps = []
    for i in range(1, len(checkpoints)):
        prev_ids = set(checkpoints[i - 1]["top_5pct_ids"])
        curr_ids = set(checkpoints[i]["top_5pct_ids"])
        if prev_ids and curr_ids:
            overlap = len(prev_ids & curr_ids) / max(len(prev_ids), len(curr_ids))
        else:
            overlap = 0.0
        overlaps.append(round(overlap, 4))

    avg_overlap = round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0
    final_gini = checkpoints[-1]["gini"] if checkpoints else 0.0

    # Are shocked agents in final top-5%?
    shocked_ids = {s["target_agent_id"] for s in sim._shock_log}
    final_top_5pct = set(checkpoints[-1]["top_5pct_ids"]) if checkpoints else set()
    shocked_in_top = shocked_ids & final_top_5pct

    result = {
        "experiment": "v2_external_shocks",
        "seed": seed,
        "terminal_bars": terminal_bars,
        "shock_interval_bars": shock_interval_bars,
        "shock_magnitude": shock_magnitude,
        "elapsed_sec": round(elapsed, 1),
        "trade_count": sim.orderbook._trade_counter,
        "shock_count": len(sim._shock_log),
        "checkpoints": checkpoints,
        "consecutive_overlaps": overlaps,
        "avg_consecutive_overlap": avg_overlap,
        "final_gini": final_gini,
        "shock_log": sim._shock_log,
        "shocked_agents": sorted(shocked_ids),
        "final_top_5pct_agents": sorted(final_top_5pct),
        "shocked_in_final_top5pct": sorted(shocked_in_top),
        "shock_to_top_ratio": round(len(shocked_in_top) / max(1, len(final_top_5pct)), 4),
    }

    return result


def apply_decision_tree(result: dict) -> tuple[str, str]:
    final_gini = result["final_gini"]
    avg_overlap = result["avg_consecutive_overlap"]
    cps = result["checkpoints"]
    growth = cps[-1]["gini"] - cps[0]["gini"] if len(cps) >= 2 else 0.0

    if final_gini > 0.55 and avg_overlap >= 0.5:
        return ("(c1) v2 external-shocks VIABLE",
                f"Gini {final_gini:.4f} > 0.55 + stable top-5% (avg overlap {avg_overlap:.4f}). "
                f"G2 criterion update + design v0.9 + T-G3 unblocks.")
    if final_gini > 0.55 and avg_overlap < 0.3:
        return ("(c2) AMBIGUOUS — top-5% churns",
                f"Gini {final_gini:.4f} > 0.55 BUT top-5% identity churns "
                f"(avg overlap {avg_overlap:.4f}). G3 substrate target moves with each shock. "
                f"User-level decision needed.")
    if final_gini < 0.51 and growth < 0.05:
        return ("(a) v2 path 1 of 3 also FAILS",
                f"Gini {final_gini:.4f} < 0.51 + minimal growth ({growth:+.4f}). "
                f"Shocks don't break plateau. Document and surface; consider skill or heavy-tail.")
    # Default: borderline / mixed
    return ("BORDERLINE — surface to user",
            f"Gini {final_gini:.4f}, avg overlap {avg_overlap:.4f}, growth {growth:+.4f}. "
            f"Doesn't fit clean (c1)/(c2)/(a) pattern.")


def main() -> None:
    print("Running v2 external-shocks (10k bars, 10 shocks at every 1000 bars, ×2 magnitude)...")
    result = run_v2_shocks(terminal_bars=10000, shock_interval_bars=1000, shock_magnitude=2.0)

    # Brief summary (not all checkpoints)
    print(f"\nElapsed: {result['elapsed_sec']:.1f}s")
    print(f"Trades: {result['trade_count']}, Shocks: {result['shock_count']}")
    print(f"\nGini trajectory at each checkpoint:")
    for cp in result["checkpoints"]:
        print(f"  Bar {cp['bar']:>5d}: Gini={cp['gini']:.4f}  top-5%={cp['top_5pct_share']:.4f}  n_alive={cp['n_alive']}")
    print(f"\nConsecutive top-5% overlaps: {result['consecutive_overlaps']}")
    print(f"Avg consecutive overlap: {result['avg_consecutive_overlap']:.4f}")
    print(f"\nFinal top-5% agents:    {result['final_top_5pct_agents']}")
    print(f"Shocked agents:          {result['shocked_agents']}")
    print(f"Shocked & in final top:  {result['shocked_in_final_top5pct']}  (ratio {result['shock_to_top_ratio']:.4f})")

    verdict, details = apply_decision_tree(result)
    print(f"\n{'=' * 70}")
    print(f"VERDICT: {verdict}")
    print(f"  {details}")
    print(f"{'=' * 70}")

    out_path = Path(__file__).resolve().parent.parent / "results" / "g2_concentration" / "v2_shocks_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"result": result, "verdict": verdict, "details": details}, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
