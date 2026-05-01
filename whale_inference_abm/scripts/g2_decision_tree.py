"""G2 diagnostic decision tree applier.

Reads e1_results.json (and e4_results.json if available) and outputs the verdict
per pre-registered decision tree in g2_diagnostic_prereg.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    g2_dir = Path(__file__).resolve().parent.parent / "results" / "g2_concentration"
    e1_path = g2_dir / "e1_results.json"
    e4_path = g2_dir / "e4_results.json"

    if not e1_path.exists():
        print(f"E1 results missing: {e1_path}")
        sys.exit(1)

    e1 = json.loads(e1_path.read_text())
    print("=" * 70)
    print("E1 admission rate sweep results:")
    print("=" * 70)
    print(f"{'rate':>10}{'admissions':>13}{'alive_end':>11}{'gini_10k':>11}{'top5_share':>13}{'top5_overlap':>14}")
    for r in e1:
        print(f"{r['rate_label']:>10}{r['n_admissions']:>13}{r['n_alive_at_end']:>11}"
              f"{r['gini_at_10k']:>11.4f}{r['top_5pct_share_at_10k']:>13.4f}"
              f"{r['top_5pct_overlap_5k_10k']:>14.4f}")
    print()

    # E1 monotonicity check
    sorted_by_rate = sorted(e1, key=lambda r: r["rate_lambda"], reverse=True)  # high rate first
    ginis = [r["gini_at_10k"] for r in sorted_by_rate]
    monotonic_increasing_as_rate_decreases = all(
        ginis[i] <= ginis[i + 1] for i in range(len(ginis) - 1)
    )
    rate_zero_gini = next((r["gini_at_10k"] for r in e1 if r["rate_lambda"] == 0), None)

    print(f"Monotonic Gini ↑ as rate ↓: {monotonic_increasing_as_rate_decreases}")
    print(f"Gini at rate=0 (no admissions): {rate_zero_gini}")
    print()

    # E1 decision branches
    e1_pass_05 = rate_zero_gini is not None and rate_zero_gini > 0.5
    e1_total_failure = rate_zero_gini is not None and rate_zero_gini < 0.3

    print("E1 verdict:")
    if e1_total_failure:
        print("  TOTAL FAILURE: Gini stays low even at rate=0")
        print("  → Wealth-weighted sizing does NOT concentrate. Architecture-level falsification.")
        print("  → Run E2/E3 (Pareto initial wealth, disable wealth-weighted sizing) for confirmation.")
        print("  → If those also fail: declare ABM v1 hypothesis falsified.")
    elif e1_pass_05:
        print("  DILUTION CONFIRMED: Gini >0.5 at rate=0 → admissions are the issue")
        print("  → E4 frozen-window test needed to determine G3 viability.")
    else:
        print(f"  PARTIAL: Gini at rate=0 = {rate_zero_gini:.3f} (between 0.3 and 0.5)")
        print("  → Concentration occurs but weak. E4 needed; possibly G2 criterion adjustment.")
    print()

    # E4 if available
    if not e4_path.exists():
        print("E4 results not yet available. Run scripts/e4_frozen_window_gini.py next.")
        return

    e4 = json.loads(e4_path.read_text())
    print("=" * 70)
    print("E4 frozen-window-aware Gini results:")
    print("=" * 70)
    print(json.dumps(e4, indent=2))
    print()

    e4_frozen_pass = e4["gini_at_end_incumbents_only"] > 0.5
    e4_frozen_overlap = e4["top_5pct_overlap_Topen_to_end_among_incumbents"] >= 0.5

    print("E4 verdict:")
    if e4_frozen_pass and e4_frozen_overlap:
        print("  FROZEN WINDOW PASSES: Gini > 0.5 + rank stability ≥ 0.5 among incumbents-at-T_open")
        print("  → G2 criterion redefined as frozen-window scope")
        print("  → Design v0.8 patch + G3 unblocked")
    elif e4_frozen_pass:
        print(f"  PARTIAL: Frozen Gini = {e4['gini_at_end_incumbents_only']:.3f} > 0.5")
        print(f"  but rank stability = {e4['top_5pct_overlap_Topen_to_end_among_incumbents']:.3f} < 0.5")
        print("  → Concentration emerges but whales unstable. G3 risk: substrate target moves.")
    else:
        print(f"  FROZEN WINDOW FAILS: Gini = {e4['gini_at_end_incumbents_only']:.3f} < 0.5 even among incumbents")
        print("  → Architecture revision: wealth-weighted sizing genuinely insufficient")
        print("  → Run E2/E3 to characterize where concentration mechanism breaks")


if __name__ == "__main__":
    main()
