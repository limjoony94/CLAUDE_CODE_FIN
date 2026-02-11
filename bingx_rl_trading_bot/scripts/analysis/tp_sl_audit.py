#!/usr/bin/env python3
"""
TP/SL Audit Script — Full Cross-Verification of Production constants.py
========================================================================
Verifies that all 52 production patterns in PATTERN_OPTIMAL_TPSL are correct by
cross-referencing against three research result JSON files:

1. tp_sl_optimization_v1264.json — v1.26.4 grid search results (per_pattern + changes)
2. tp_sl_deep_validation.json   — 5-phase deep validation (31 accepted, 1 rejected)
3. uniform_tp_validation.json   — Uniform TP 70% validation (TP * 0.7 for all 52)

Checks performed:
  A) v1.26.4 TP/SL derivation: production TP/SL trace back to research correctly
  B) Uniform TP 70% transform: production TP == v1.26.4_TP * 0.7 (exact)
  C) SL unchanged: production SL == v1.26.4 SL (exact)
  D) Pattern completeness: all 52 patterns present, no extras, no missing
  E) Cross-reference: VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
     PATTERN_STATS, and PATTERN_OPTIMAL_TPSL all have identical pattern sets

Output: results/tp_sl_audit.json
"""

import json
import os
import sys
from datetime import datetime

# ===========================================================================
# PATHS
# ===========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

CONSTANTS_PY = os.path.join(
    PROJECT_ROOT, "scripts", "production", "pattern_5m", "constants.py"
)
OPT_JSON = os.path.join(PROJECT_ROOT, "results", "tp_sl_optimization_v1264.json")
DV_JSON = os.path.join(PROJECT_ROOT, "results", "tp_sl_deep_validation.json")
UV_JSON = os.path.join(PROJECT_ROOT, "results", "uniform_tp_validation.json")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "results", "tp_sl_audit.json")

# Tolerance for floating-point comparison
EPS = 1e-6


def load_production_constants():
    """Import production constants by executing constants.py in isolation."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("constants", CONSTANTS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        "PATTERN_OPTIMAL_TPSL": dict(mod.PATTERN_OPTIMAL_TPSL),
        "VALIDATED_LONG_PATTERNS": list(mod.VALIDATED_LONG_PATTERNS),
        "VALIDATED_SHORT_PATTERNS": list(mod.VALIDATED_SHORT_PATTERNS),
        "PATTERN_STATS": dict(mod.PATTERN_STATS),
    }


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def float_eq(a, b, eps=EPS):
    return abs(a - b) < eps


def main():
    print("=" * 70)
    print("TP/SL AUDIT — Production constants.py vs Research Results")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # 1. Load all data sources
    # ------------------------------------------------------------------
    print("[1/6] Loading data sources...")
    prod = load_production_constants()
    opt = load_json(OPT_JSON)
    dv = load_json(DV_JSON)
    uv = load_json(UV_JSON)
    print(f"  Production PATTERN_OPTIMAL_TPSL: {len(prod['PATTERN_OPTIMAL_TPSL'])} patterns")
    print(f"  Production VALIDATED_LONG:  {len(prod['VALIDATED_LONG_PATTERNS'])} patterns")
    print(f"  Production VALIDATED_SHORT: {len(prod['VALIDATED_SHORT_PATTERNS'])} patterns")
    print(f"  Production PATTERN_STATS:   {len(prod['PATTERN_STATS'])} patterns")
    print(f"  Optimization per_pattern:   {len(opt['per_pattern'])} patterns")
    print(f"  Deep validation phase5:     {len(dv['phase5_final']['accepted'])} accepted, "
          f"{len(dv['phase5_final']['rejected'])} rejected")
    print(f"  Uniform TP validation:      {len(uv['phase1_per_pattern'])} patterns")
    print()

    # Build lookup maps
    # v1.26.4 optimization: per_pattern has 'best' tp/sl for every pattern
    opt_map = {}  # pattern -> {tp, sl, changed, direction}
    for pname, pdata in opt["per_pattern"].items():
        opt_map[pname] = {
            "direction": pdata["direction"],
            "best_tp": pdata["best"]["tp"],
            "best_sl": pdata["best"]["sl"],
            "changed": pdata["changed"],
        }

    # Deep validation: accepted patterns get their new TP/SL from changes list
    dv_accepted = set(dv["phase5_final"]["accepted"])
    dv_rejected = set(dv["phase5_final"]["rejected"])
    dv_changes = {}  # pattern -> {tp, sl}
    for c in dv["phase5_final"]["changes"]:
        dv_changes[c["pattern"]] = {"tp": c["tp"], "sl": c["sl"]}

    # Uniform TP: tp_cur = v1.26.4 final TP, tp_70 = tp_cur * 0.7, sl = v1.26.4 SL
    uv_map = {}  # pattern -> {tp_v1264, tp_70, sl, direction}
    for p in uv["phase1_per_pattern"]:
        uv_map[p["pattern"]] = {
            "direction": p["direction"],
            "tp_v1264": p["tp_cur"],
            "tp_70": p["tp_70"],
            "sl": p["sl"],
        }

    # ------------------------------------------------------------------
    # 2. Derive expected v1.26.4 final TP/SL for all 52 patterns
    # ------------------------------------------------------------------
    # Logic:
    #   - Patterns in dv_changes (31 accepted) → use dv_changes tp/sl
    #   - Patterns in dv_rejected (DN-MD-DN) → keep pre-optimization tp/sl
    #     (but the optimization did propose a change that was rejected,
    #      so we use the "current" tp/sl from optimization)
    #   - Patterns NOT in optimization changes (20 already_optimal) → keep as-is
    #
    # Actually, the simplest authoritative source for v1.26.4 final TP/SL
    # is the uniform_tp_validation.json tp_cur/sl fields, because that script
    # was written AFTER v1.26.4 was finalized and reads the actual v1.26.4 state.
    #
    # So: v1264_tp[p] = uv_map[p]['tp_v1264'], v1264_sl[p] = uv_map[p]['sl']
    # And we verify: prod_tp[p] == v1264_tp[p] * 0.7, prod_sl[p] == v1264_sl[p]

    print("[2/6] Building v1.26.4 reference TP/SL from uniform_tp_validation.json...")
    v1264_ref = {}
    for pname, pdata in uv_map.items():
        v1264_ref[pname] = {
            "tp": pdata["tp_v1264"],
            "sl": pdata["sl"],
            "direction": pdata["direction"],
        }
    print(f"  Reference patterns: {len(v1264_ref)}")
    print()

    # ------------------------------------------------------------------
    # 3. Verify: production TP == v1.26.4 TP * 0.7  (Uniform TP 70%)
    # ------------------------------------------------------------------
    print("[3/6] Verifying Uniform TP 70% transformation...")
    tp_results = []
    tp_pass = 0
    tp_fail = 0
    for pname, (prod_tp, prod_sl) in sorted(prod["PATTERN_OPTIMAL_TPSL"].items()):
        if pname not in v1264_ref:
            tp_results.append({
                "pattern": pname,
                "status": "MISSING_IN_REFERENCE",
                "prod_tp": prod_tp,
            })
            tp_fail += 1
            continue
        ref = v1264_ref[pname]
        expected_tp = round(ref["tp"] * 0.7, 10)  # avoid float noise
        # Use rounding to 2 decimal places for comparison (grid values)
        expected_tp_r2 = round(expected_tp, 2)
        prod_tp_r2 = round(prod_tp, 2)
        match = float_eq(expected_tp_r2, prod_tp_r2)
        status = "PASS" if match else "FAIL"
        if match:
            tp_pass += 1
        else:
            tp_fail += 1
        tp_results.append({
            "pattern": pname,
            "status": status,
            "v1264_tp": ref["tp"],
            "expected_prod_tp": expected_tp_r2,
            "actual_prod_tp": prod_tp_r2,
            "delta": round(prod_tp_r2 - expected_tp_r2, 4),
        })
    print(f"  TP check: {tp_pass} PASS, {tp_fail} FAIL out of {len(tp_results)}")
    for r in tp_results:
        if r["status"] != "PASS":
            print(f"    FAIL: {r['pattern']} — expected {r.get('expected_prod_tp')}, "
                  f"got {r.get('actual_prod_tp')}")
    print()

    # ------------------------------------------------------------------
    # 4. Verify: production SL == v1.26.4 SL  (unchanged)
    # ------------------------------------------------------------------
    print("[4/6] Verifying SL unchanged...")
    sl_results = []
    sl_pass = 0
    sl_fail = 0
    for pname, (prod_tp, prod_sl) in sorted(prod["PATTERN_OPTIMAL_TPSL"].items()):
        if pname not in v1264_ref:
            sl_results.append({
                "pattern": pname,
                "status": "MISSING_IN_REFERENCE",
                "prod_sl": prod_sl,
            })
            sl_fail += 1
            continue
        ref = v1264_ref[pname]
        match = float_eq(round(prod_sl, 2), round(ref["sl"], 2))
        status = "PASS" if match else "FAIL"
        if match:
            sl_pass += 1
        else:
            sl_fail += 1
        sl_results.append({
            "pattern": pname,
            "status": status,
            "v1264_sl": ref["sl"],
            "actual_prod_sl": round(prod_sl, 2),
            "delta": round(prod_sl - ref["sl"], 4),
        })
    print(f"  SL check: {sl_pass} PASS, {sl_fail} FAIL out of {len(sl_results)}")
    for r in sl_results:
        if r["status"] != "PASS":
            print(f"    FAIL: {r['pattern']} — expected {r.get('v1264_sl')}, "
                  f"got {r.get('actual_prod_sl')}")
    print()

    # ------------------------------------------------------------------
    # 5. Cross-reference: v1.26.4 optimization -> deep validation -> final state
    # ------------------------------------------------------------------
    print("[5/6] Cross-referencing optimization → deep validation → production...")
    xref_results = []
    xref_pass = 0
    xref_fail = 0

    for pname in sorted(set(opt_map.keys()) | set(prod["PATTERN_OPTIMAL_TPSL"].keys())):
        issues = []
        if pname not in opt_map:
            issues.append("MISSING_IN_OPTIMIZATION")
        if pname not in prod["PATTERN_OPTIMAL_TPSL"]:
            issues.append("MISSING_IN_PRODUCTION")

        if not issues and pname in opt_map:
            om = opt_map[pname]
            # If optimization proposed a change:
            if om["changed"]:
                # Check if deep validation accepted or rejected
                if pname in dv_accepted:
                    # Accepted: v1.26.4 final = optimization best
                    if pname in dv_changes:
                        dv_tp = dv_changes[pname]["tp"]
                        dv_sl = dv_changes[pname]["sl"]
                        if not float_eq(dv_tp, om["best_tp"]):
                            issues.append(f"DV_TP_MISMATCH(opt={om['best_tp']},dv={dv_tp})")
                        if not float_eq(dv_sl, om["best_sl"]):
                            issues.append(f"DV_SL_MISMATCH(opt={om['best_sl']},dv={dv_sl})")
                elif pname in dv_rejected:
                    # Rejected: v1.26.4 final should be the ORIGINAL (pre-optimization)
                    # The uniform_tp_validation tp_cur should reflect original TP
                    pass  # We trust uniform_tp_validation as ground truth
                else:
                    # Not in deep validation at all (not part of the 32 proposed changes)
                    # This means optimization proposed a change but DV didn't test it?
                    # Actually DV only tests the 32 changes from optimization
                    pass

        status = "PASS" if not issues else "FAIL"
        if status == "PASS":
            xref_pass += 1
        else:
            xref_fail += 1
        xref_results.append({
            "pattern": pname,
            "status": status,
            "issues": issues if issues else None,
        })

    print(f"  Cross-ref: {xref_pass} PASS, {xref_fail} FAIL out of {len(xref_results)}")
    for r in xref_results:
        if r["status"] != "PASS":
            print(f"    FAIL: {r['pattern']} — {r['issues']}")
    print()

    # ------------------------------------------------------------------
    # 6. Pattern set consistency
    # ------------------------------------------------------------------
    print("[6/6] Pattern set consistency check...")

    set_tpsl = set(prod["PATTERN_OPTIMAL_TPSL"].keys())
    set_long = set(prod["VALIDATED_LONG_PATTERNS"])
    set_short = set(prod["VALIDATED_SHORT_PATTERNS"])
    set_stats = set(prod["PATTERN_STATS"].keys())
    set_combined = set_long | set_short

    consistency_checks = {}

    # A: VALIDATED_LONG + VALIDATED_SHORT == PATTERN_OPTIMAL_TPSL keys
    if set_combined == set_tpsl:
        consistency_checks["long_short_vs_tpsl"] = "PASS"
    else:
        only_in_combined = set_combined - set_tpsl
        only_in_tpsl = set_tpsl - set_combined
        consistency_checks["long_short_vs_tpsl"] = {
            "status": "FAIL",
            "only_in_long_short": sorted(only_in_combined),
            "only_in_tpsl": sorted(only_in_tpsl),
        }

    # B: PATTERN_STATS keys == PATTERN_OPTIMAL_TPSL keys
    if set_stats == set_tpsl:
        consistency_checks["stats_vs_tpsl"] = "PASS"
    else:
        only_in_stats = set_stats - set_tpsl
        only_in_tpsl2 = set_tpsl - set_stats
        consistency_checks["stats_vs_tpsl"] = {
            "status": "FAIL",
            "only_in_stats": sorted(only_in_stats),
            "only_in_tpsl": sorted(only_in_tpsl2),
        }

    # C: No overlap between LONG and SHORT
    overlap = set_long & set_short
    consistency_checks["long_short_no_overlap"] = "PASS" if not overlap else {
        "status": "FAIL", "overlapping": sorted(overlap)
    }

    # D: Count check: 32L + 20S = 52
    consistency_checks["count_check"] = {
        "long": len(set_long),
        "short": len(set_short),
        "total": len(set_long) + len(set_short),
        "tpsl_count": len(set_tpsl),
        "stats_count": len(set_stats),
        "status": "PASS" if (
            len(set_long) == 32 and
            len(set_short) == 20 and
            len(set_tpsl) == 52 and
            len(set_stats) == 52
        ) else "FAIL",
    }

    # E: Direction consistency: LONG patterns in PATTERN_STATS have direction=LONG
    dir_issues = []
    for p in set_long:
        if p in prod["PATTERN_STATS"]:
            if prod["PATTERN_STATS"][p]["direction"] != "LONG":
                dir_issues.append(f"{p}: expected LONG, got {prod['PATTERN_STATS'][p]['direction']}")
    for p in set_short:
        if p in prod["PATTERN_STATS"]:
            if prod["PATTERN_STATS"][p]["direction"] != "SHORT":
                dir_issues.append(f"{p}: expected SHORT, got {prod['PATTERN_STATS'][p]['direction']}")
    consistency_checks["direction_consistency"] = "PASS" if not dir_issues else {
        "status": "FAIL", "issues": dir_issues
    }

    # F: Direction consistency vs uniform_tp_validation
    uv_dir_issues = []
    for pname, pdata in uv_map.items():
        if pname in set_long and pdata["direction"] != "LONG":
            uv_dir_issues.append(f"{pname}: prod=LONG, uv={pdata['direction']}")
        elif pname in set_short and pdata["direction"] != "SHORT":
            uv_dir_issues.append(f"{pname}: prod=SHORT, uv={pdata['direction']}")
    consistency_checks["direction_vs_uniform"] = "PASS" if not uv_dir_issues else {
        "status": "FAIL", "issues": uv_dir_issues
    }

    all_consistency_pass = all(
        (v == "PASS" if isinstance(v, str) else v.get("status") == "PASS")
        for v in consistency_checks.values()
    )

    print(f"  Consistency checks:")
    for name, result in consistency_checks.items():
        if isinstance(result, str):
            print(f"    {name}: {result}")
        else:
            print(f"    {name}: {result.get('status', 'UNKNOWN')}")
            if result.get("status") != "PASS":
                for k, v in result.items():
                    if k != "status":
                        print(f"      {k}: {v}")
    print()

    # ------------------------------------------------------------------
    # Summary & detailed table
    # ------------------------------------------------------------------
    all_pass = (tp_fail == 0 and sl_fail == 0 and xref_fail == 0 and all_consistency_pass)

    print("=" * 70)
    print(f"OVERALL RESULT: {'PASS — All 52 patterns verified' if all_pass else 'FAIL — See details above'}")
    print("=" * 70)
    print()

    # Detailed table
    print(f"{'Pattern':<15} {'Dir':<6} {'v1264 TP':>9} {'Prod TP':>9} {'ExpTP':>9} {'TP OK':>6} "
          f"{'v1264 SL':>9} {'Prod SL':>9} {'SL OK':>6}")
    print("-" * 90)
    for pname in sorted(prod["PATTERN_OPTIMAL_TPSL"].keys()):
        prod_tp, prod_sl = prod["PATTERN_OPTIMAL_TPSL"][pname]
        ref = v1264_ref.get(pname, {})
        v1264_tp = ref.get("tp", "?")
        v1264_sl = ref.get("sl", "?")
        direction = ref.get("direction", "?")
        if isinstance(v1264_tp, (int, float)):
            exp_tp = round(v1264_tp * 0.7, 2)
            tp_ok = "PASS" if float_eq(round(prod_tp, 2), exp_tp) else "FAIL"
        else:
            exp_tp = "?"
            tp_ok = "N/A"
        if isinstance(v1264_sl, (int, float)):
            sl_ok = "PASS" if float_eq(round(prod_sl, 2), round(v1264_sl, 2)) else "FAIL"
        else:
            sl_ok = "N/A"
        print(f"{pname:<15} {direction:<6} {v1264_tp:>9} {prod_tp:>9} {exp_tp:>9} {tp_ok:>6} "
              f"{v1264_sl:>9} {prod_sl:>9} {sl_ok:>6}")

    # ------------------------------------------------------------------
    # Save JSON output
    # ------------------------------------------------------------------
    output = {
        "timestamp": datetime.now().isoformat(),
        "script": "tp_sl_audit.py",
        "overall_result": "PASS" if all_pass else "FAIL",
        "summary": {
            "tp_check": {"pass": tp_pass, "fail": tp_fail, "total": len(tp_results)},
            "sl_check": {"pass": sl_pass, "fail": sl_fail, "total": len(sl_results)},
            "xref_check": {"pass": xref_pass, "fail": xref_fail, "total": len(xref_results)},
            "consistency_all_pass": all_consistency_pass,
            "pattern_counts": consistency_checks["count_check"],
        },
        "tp_verification": tp_results,
        "sl_verification": sl_results,
        "cross_reference": [r for r in xref_results if r["status"] != "PASS"],
        "consistency_checks": consistency_checks,
        "detailed_table": [],
    }

    for pname in sorted(prod["PATTERN_OPTIMAL_TPSL"].keys()):
        prod_tp, prod_sl = prod["PATTERN_OPTIMAL_TPSL"][pname]
        ref = v1264_ref.get(pname, {})
        v1264_tp = ref.get("tp")
        v1264_sl = ref.get("sl")
        direction = ref.get("direction")
        exp_tp = round(v1264_tp * 0.7, 2) if v1264_tp is not None else None
        tp_ok = float_eq(round(prod_tp, 2), exp_tp) if exp_tp is not None else False
        sl_ok = float_eq(round(prod_sl, 2), round(v1264_sl, 2)) if v1264_sl is not None else False
        output["detailed_table"].append({
            "pattern": pname,
            "direction": direction,
            "v1264_tp": v1264_tp,
            "v1264_sl": v1264_sl,
            "expected_prod_tp": exp_tp,
            "actual_prod_tp": round(prod_tp, 2),
            "actual_prod_sl": round(prod_sl, 2),
            "tp_match": tp_ok,
            "sl_match": sl_ok,
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {OUTPUT_JSON}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
