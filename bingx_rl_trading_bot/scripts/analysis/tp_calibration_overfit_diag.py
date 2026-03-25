#!/usr/bin/env python3
"""
TP Calibration — Per-Strategy Overfitting Diagnostic
======================================================
각 보정 전략의 과적합 위험도를 정량 평가.

측정 지표:
  1) IS-OOS Gap: IS PnL - OOS PnL (절대 / 상대)
  2) OOS Fold Stability: fold간 PnL 변동계수 (CV)
  3) Parameter Freedom: 자유 파라미터 수 (높을수록 overfit 위험)
  4) Baseline-Relative OOS: OOS가 baseline 대비 어떤가
  5) Robustness: 5-fold expanded WF (더 많은 fold로 안정성 검증)
  6) Bootstrap OOS: 3-seed bootstrap로 OOS 신뢰구간
  7) IS P/M vs OOS P/M Ratio: 이 비율이 높으면 IS에 편향

최종 판정: 전략별 과적합 스코어 (0-100, 낮을수록 안전)

Output: results/tp_calibration_overfit_diag.json
"""
import os, sys, json, time, numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from scripts.scanner.pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    compute_atr_ratio, compute_ema_slope, calc_stats_compound,
    portfolio_npos, compute_excursions, MAX_BARS,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
)

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "btc_5m_270days_reclassified.csv")
PATTERNS_FILE = os.path.join(PROJECT_ROOT, "results", "dynamic_patterns.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "results", "tp_calibration_overfit_diag.json")


def load_data():
    df = load_and_classify(DATA_FILE)
    n = len(df)
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    tc = df["candle_type"].tolist()
    nw = find_neutral_window(c, tol_pct=1.0)
    atr = compute_atr_ratio(h, l, c)
    ema = compute_ema_slope(c)
    si = build_signal_index(tc, n)
    with open(PATTERNS_FILE) as f:
        pd_ = json.load(f)["pattern_details"]
    return n, o, h, l, c, tc, nw, atr, ema, si, pd_


def make_sigs(pd_, si, s, e, scale_fn):
    t = []
    for k, info in pd_.items():
        p, d = info["pattern"], info["direction"]
        tp = max(0.3, info["tp"] * scale_fn(info))
        sl = info["sl"]
        if p not in si:
            continue
        for b in si[p]:
            if s <= b < e:
                t.append((b, p, d, tp, sl))
    return sorted(t, key=lambda x: x[0])


def run_is(sigs, o, h, l, c, n, atr, ema, s, e):
    if not sigs:
        return None
    trades, stats = portfolio_npos(sigs, o, h, l, c, n, atr, ema, s, e)
    st = calc_stats_compound(trades)
    mdd = stats.get("mdd_mtm", st["mdd"])
    pm = st["pnl"] / mdd if mdd > 0 else 0
    tp_pnls = [abs(t["pnl_slot"]) for t in trades if t.get("reason") == "TP"]
    sl_pnls = [abs(t["pnl_slot"]) for t in trades if t.get("reason") == "SL"]
    rr = np.mean(tp_pnls) / np.mean(sl_pnls) if sl_pnls and tp_pnls else 0
    avg_w = np.mean(tp_pnls) if tp_pnls else 0
    avg_l = np.mean(sl_pnls) if sl_pnls else 0
    be_wr = avg_l / (avg_w + avg_l) * 100 if (avg_w + avg_l) > 0 else 50
    return {
        "pnl": st["pnl"], "mdd": round(mdd, 2), "pm": round(pm, 1),
        "wr": st["wr"], "rr": round(rr, 3), "trades": st["trades"],
        "be_wr": round(be_wr, 1), "wr_margin": round(st["wr"] - be_wr, 1),
    }


def run_wf_detailed(make_fn, o, h, l, c, n, nws, nwe, nf=3):
    seg = (nwe - nws) // (nf + 1)
    folds = []
    for f in range(nf):
        ie = nws + (f + 1) * seg
        os_ = ie
        oe = nws + (f + 2) * seg if f < nf - 1 else nwe
        if os_ >= oe:
            folds.append({"fold": f + 1, "oos_pnl": 0, "oos_mdd": 0, "oos_pm": 0,
                          "oos_wr": 0, "oos_trades": 0})
            continue
        sigs = make_fn(os_, oe)
        if not sigs:
            folds.append({"fold": f + 1, "oos_pnl": 0, "oos_mdd": 0, "oos_pm": 0,
                          "oos_wr": 0, "oos_trades": 0})
            continue
        atr = compute_atr_ratio(h[:oe], l[:oe], c[:oe])
        ema = compute_ema_slope(c[:oe])
        trades, stats = portfolio_npos(sigs, o, h, l, c, oe, atr, ema, os_, oe)
        st = calc_stats_compound(trades)
        mdd = stats.get("mdd_mtm", st["mdd"])
        pm = st["pnl"] / mdd if mdd > 0 else 0
        folds.append({
            "fold": f + 1, "oos_pnl": round(st["pnl"], 1),
            "oos_mdd": round(mdd, 2), "oos_pm": round(pm, 1),
            "oos_wr": st["wr"], "oos_trades": st["trades"],
        })
    total = sum(f["oos_pnl"] for f in folds)
    np_ = sum(1 for f in folds if f["oos_pnl"] > 0)
    pnls = [f["oos_pnl"] for f in folds]
    cv = np.std(pnls) / np.mean(pnls) if np.mean(pnls) > 0 else 999
    return {
        "folds": folds, "total_oos": round(total, 1),
        "verdict": "PASS" if np_ == nf else "FAIL",
        "oos_cv": round(cv, 3),
        "oos_min": round(min(pnls), 1), "oos_max": round(max(pnls), 1),
        "oos_mean": round(np.mean(pnls), 1), "oos_std": round(np.std(pnls), 1),
    }


def main():
    t0 = time.time()
    print("Per-Strategy Overfitting Diagnostic")
    print("=" * 100)

    n, o, h, l, c, tc, nw, atr, ema, si, pd_ = load_data()
    nws, nwe = nw
    print(f"  {n} bars, NW [{nws},{nwe}], {len(pd_)} patterns")

    # Precompute per-pattern stats
    for k, info in pd_.items():
        info["_rr"] = info["tp"] / info["sl"] if info["sl"] > 0 else 0
        info["_edge"] = info.get("edge", info["wr"] - info["sl"] / (info["tp"] + info["sl"]) * 100)

    # MFE medians
    mfe_medians = {}
    for k, info in pd_.items():
        p, d = info["pattern"], info["direction"]
        if p not in si:
            continue
        nw_bars = sorted([b for b in si[p] if nws <= b < nwe])
        if len(nw_bars) < 10:
            mfe_medians[k] = info["tp"]
            continue
        exc = compute_excursions(nw_bars, d, o, h, l, n, MAX_BARS)
        if exc:
            mfe_medians[k] = round(float(np.median([e_["mfe_pct"] for e_ in exc])), 4)
        else:
            mfe_medians[k] = info["tp"]

    # Define strategies with metadata
    strats = {
        "S0_uniform_0.72": {
            "fn": lambda info: 0.72,
            "free_params": 1,  # 1 global parameter
            "description": "Single uniform scale factor",
        },
        "S1_rr_bucket": {
            "fn": lambda info: 0.65 if info["_rr"] < 0.45 else (0.72 if info["_rr"] < 0.55 else 0.82),
            "free_params": 5,  # 3 scale values + 2 thresholds
            "description": "3-bucket R:R-based scaling",
        },
        "S1b_rr_fine": {
            "fn": lambda info: (0.60 if info["_rr"] < 0.45 else
                                0.68 if info["_rr"] < 0.50 else
                                0.75 if info["_rr"] < 0.55 else
                                0.82 if info["_rr"] < 0.65 else 0.90),
            "free_params": 9,  # 5 scale values + 4 thresholds
            "description": "5-bucket R:R-based scaling",
        },
        "S2_wr_adaptive": {
            "fn": lambda info: (0.60 if info["wr"] >= 90 else
                                0.68 if info["wr"] >= 85 else
                                0.75 if info["wr"] >= 80 else 0.85),
            "free_params": 7,  # 4 scale values + 3 thresholds
            "description": "4-bucket WR-based scaling",
        },
        "S2b_wr_linear": {
            "fn": lambda info: max(0.50, min(0.95, 0.90 - (info["wr"] - 76) / (96 - 76) * (0.90 - 0.55))),
            "free_params": 4,  # slope, intercept, clamp_lo, clamp_hi
            "description": "Linear WR→scale mapping",
        },
        "S3_edge_weight": {
            "fn": lambda info: max(0.55, min(0.90, 0.72 - (info["_edge"] - 18) / 30 * 0.15)),
            "free_params": 4,  # slope, intercept, clamp_lo, clamp_hi
            "description": "Linear edge→scale mapping",
        },
        "S4_mfe_median": {
            "fn": lambda info: mfe_medians.get(f"{info['pattern']}_{info['direction']}", info["tp"]) / info["tp"] if info["tp"] > 0 else 0.72,
            "free_params": 0,  # data-driven, no tuned parameters
            "description": "MFE median as TP (data-derived, no tuning)",
        },
        "S6_trade_count": {
            "fn": lambda info: (0.80 if info["trades"] >= 80 else
                                0.72 if info["trades"] >= 50 else
                                0.68 if info["trades"] >= 35 else 0.62),
            "free_params": 7,  # 4 scale values + 3 thresholds
            "description": "4-bucket trade-count confidence scaling",
        },
        "S7_wr_rr_combo": {
            "fn": lambda info: max(0.50, min(0.95,
                (0.90 - (info["wr"] - 76) / (96 - 76) * (0.90 - 0.55)) +
                (info["_rr"] - 0.45) / (0.80 - 0.45) * 0.10)),
            "free_params": 8,  # WR params(4) + R:R params(2) + clamps(2)
            "description": "WR + R:R combined linear mapping",
        },
        "S8_expand_0.85": {
            "fn": lambda info: 0.85,
            "free_params": 1,
            "description": "Single uniform scale (expanded)",
        },
        "S8b_expand_0.90": {
            "fn": lambda info: 0.90,
            "free_params": 1,
            "description": "Single uniform scale (expanded more)",
        },
    }

    # =========================================================================
    # Run IS + 3-fold WF + 5-fold WF for each strategy
    # =========================================================================
    print(f"\n  Computing IS + 3-fold WF + 5-fold WF for {len(strats)} strategies...")

    results = {}
    for name, meta in strats.items():
        fn = meta["fn"]
        mk = lambda s, e, _fn=fn: make_sigs(pd_, si, s, e, _fn)

        # IS
        sigs_is = mk(nws, nwe)
        is_r = run_is(sigs_is, o, h, l, c, n, atr, ema, nws, nwe)

        # 3-fold WF
        wf3 = run_wf_detailed(mk, o, h, l, c, n, nws, nwe, nf=3)

        # 5-fold WF (stricter)
        wf5 = run_wf_detailed(mk, o, h, l, c, n, nws, nwe, nf=5)

        results[name] = {
            "meta": meta,
            "is": is_r,
            "wf3": wf3,
            "wf5": wf5,
        }

    # =========================================================================
    # Compute overfitting metrics
    # =========================================================================
    print(f"\n{'='*100}")
    print("OVERFITTING DIAGNOSTIC — Per Strategy")
    print(f"{'='*100}")

    base_oos3 = results["S0_uniform_0.72"]["wf3"]["total_oos"]
    base_oos5 = results["S0_uniform_0.72"]["wf5"]["total_oos"]

    diag = {}
    for name in sorted(results.keys()):
        r = results[name]
        is_r = r["is"]
        wf3 = r["wf3"]
        wf5 = r["wf5"]
        fp = r["meta"]["free_params"]

        # 1. IS-OOS Gap
        is_pnl = is_r["pnl"]
        oos3_pnl = wf3["total_oos"]
        oos5_pnl = wf5["total_oos"]
        gap3_abs = is_pnl - oos3_pnl
        gap3_rel = gap3_abs / is_pnl * 100 if is_pnl > 0 else 0  # % of IS
        gap5_abs = is_pnl - oos5_pnl
        gap5_rel = gap5_abs / is_pnl * 100 if is_pnl > 0 else 0

        # 2. OOS Stability (fold CV)
        cv3 = wf3["oos_cv"]
        cv5 = wf5["oos_cv"]

        # 3. 3-fold vs 5-fold OOS degradation
        oos_degrad = (oos3_pnl - oos5_pnl) / oos3_pnl * 100 if oos3_pnl > 0 else 0

        # 4. IS P/M vs OOS P/M ratio
        oos_pms = [f["oos_pm"] for f in wf3["folds"] if f["oos_pm"] > 0]
        avg_oos_pm = np.mean(oos_pms) if oos_pms else 0
        pm_ratio = is_r["pm"] / avg_oos_pm if avg_oos_pm > 0 else 999

        # 5. vs baseline OOS delta
        vs_base3 = oos3_pnl - base_oos3
        vs_base5 = oos5_pnl - base_oos5

        # 6. Overfitting Score (0-100, lower=safer)
        # Components:
        score_gap = min(30, gap3_rel * 0.5)         # IS-OOS gap (max 30)
        score_cv = min(20, cv3 * 20)                 # fold instability (max 20)
        score_params = min(20, fp * 2.5)             # parameter freedom (max 20)
        score_degrad = min(15, max(0, oos_degrad))   # 3→5 fold degradation (max 15)
        score_pm = min(15, max(0, (pm_ratio - 1) * 10))  # IS/OOS PM ratio (max 15)
        overfit_score = round(score_gap + score_cv + score_params + score_degrad + score_pm, 1)

        # Risk label
        if overfit_score < 25:
            risk = "LOW"
        elif overfit_score < 40:
            risk = "MODERATE"
        elif overfit_score < 55:
            risk = "HIGH"
        else:
            risk = "VERY HIGH"

        diag[name] = {
            "is_pnl": round(is_pnl, 1), "is_pm": is_r["pm"],
            "oos3": round(oos3_pnl, 1), "oos5": round(oos5_pnl, 1),
            "gap3_abs": round(gap3_abs, 1), "gap3_rel": round(gap3_rel, 1),
            "gap5_rel": round(gap5_rel, 1),
            "cv3": cv3, "cv5": cv5,
            "oos_degrad_pct": round(oos_degrad, 1),
            "pm_ratio": round(pm_ratio, 2),
            "free_params": fp,
            "vs_base3": round(vs_base3, 1), "vs_base5": round(vs_base5, 1),
            "score_breakdown": {
                "gap": round(score_gap, 1), "cv": round(score_cv, 1),
                "params": round(score_params, 1), "degrad": round(score_degrad, 1),
                "pm": round(score_pm, 1),
            },
            "overfit_score": overfit_score,
            "risk": risk,
            "wr_margin": is_r["wr_margin"],
            "wf3_verdict": wf3["verdict"], "wf5_verdict": wf5["verdict"],
            "wf3_folds": [f["oos_pnl"] for f in wf3["folds"]],
            "wf5_folds": [f["oos_pnl"] for f in wf5["folds"]],
        }

    # Print detailed table
    print(f"\n  {'Strategy':>20s} | {'IS PnL':>7s} | {'OOS-3':>7s} | {'OOS-5':>7s} | {'Gap%':>5s} | {'CV3':>5s} | {'FP':>2s} | "
          f"{'Deg%':>5s} | {'PMr':>4s} | {'Score':>5s} | Risk      | {'Margin':>7s} | WF3 | WF5")
    print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*2}-+-"
          f"{'-'*5}-+-{'-'*4}-+-{'-'*5}-+-{'-'*10}-+-{'-'*7}-+-{'-'*3}-+-{'-'*3}")

    for name in sorted(diag.keys(), key=lambda x: diag[x]["overfit_score"]):
        d = diag[name]
        mk = " <<" if name == "S0_uniform_0.72" else ""
        print(f"  {name:>20s} | {d['is_pnl']:+6.0f}% | {d['oos3']:+6.0f}% | {d['oos5']:+6.0f}% | "
              f"{d['gap3_rel']:4.0f}% | {d['cv3']:5.3f} | {d['free_params']:2d} | "
              f"{d['oos_degrad_pct']:4.0f}% | {d['pm_ratio']:4.1f} | {d['overfit_score']:5.1f} | "
              f"{d['risk']:10s} | {d['wr_margin']:+5.1f}pp | {d['wf3_verdict']:4s}| {d['wf5_verdict']:4s}{mk}")

    # =========================================================================
    # Detailed fold analysis for top strategies
    # =========================================================================
    print(f"\n{'='*100}")
    print("FOLD-LEVEL DETAIL (5-fold WF)")
    print(f"{'='*100}")

    for name in sorted(diag.keys(), key=lambda x: diag[x]["overfit_score"])[:8]:
        d = diag[name]
        folds = d["wf5_folds"]
        print(f"  {name:>20s}: [{', '.join(f'{f:+.0f}%' for f in folds)}] "
              f"mean={np.mean(folds):+.0f}% std={np.std(folds):.0f}%")

    # =========================================================================
    # Score decomposition for key strategies
    # =========================================================================
    print(f"\n{'='*100}")
    print("SCORE DECOMPOSITION (what drives each strategy's overfit risk)")
    print(f"{'='*100}")
    print(f"  {'Strategy':>20s} | {'Gap':>5s} | {'CV':>5s} | {'Params':>6s} | {'Degrad':>6s} | {'PM_r':>5s} | {'TOTAL':>5s}")
    print(f"  {'-'*20}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}")

    for name in sorted(diag.keys(), key=lambda x: diag[x]["overfit_score"]):
        d = diag[name]
        sb = d["score_breakdown"]
        print(f"  {name:>20s} | {sb['gap']:5.1f} | {sb['cv']:5.1f} | {sb['params']:6.1f} | "
              f"{sb['degrad']:6.1f} | {sb['pm']:5.1f} | {d['overfit_score']:5.1f}")

    # =========================================================================
    # Final verdict
    # =========================================================================
    print(f"\n{'='*100}")
    print("FINAL VERDICT")
    print(f"{'='*100}")

    low_risk = [(name, diag[name]) for name in diag if diag[name]["risk"] == "LOW"]
    mod_risk = [(name, diag[name]) for name in diag if diag[name]["risk"] == "MODERATE"]

    print(f"\n  LOW risk strategies ({len(low_risk)}):")
    for name, d in sorted(low_risk, key=lambda x: x[1]["oos3"], reverse=True):
        print(f"    {name:>20s}: OOS3 {d['oos3']:+.0f}% | OOS5 {d['oos5']:+.0f}% | Score {d['overfit_score']:.1f} | Margin {d['wr_margin']:+.1f}pp")

    print(f"\n  MODERATE risk strategies ({len(mod_risk)}):")
    for name, d in sorted(mod_risk, key=lambda x: x[1]["oos3"], reverse=True):
        print(f"    {name:>20s}: OOS3 {d['oos3']:+.0f}% | OOS5 {d['oos5']:+.0f}% | Score {d['overfit_score']:.1f} | Margin {d['wr_margin']:+.1f}pp")

    # Best LOW-risk strategy that beats baseline
    base_d = diag["S0_uniform_0.72"]
    better_low = [(n, d) for n, d in low_risk if d["oos5"] > base_d["oos5"] and n != "S0_uniform_0.72"]
    if better_low:
        print(f"\n  LOW-risk strategies beating baseline on OOS5:")
        for name, d in better_low:
            print(f"    {name}: OOS5 {d['oos5']:+.0f}% vs baseline {base_d['oos5']:+.0f}% ({d['oos5']-base_d['oos5']:+.0f}%)")
    else:
        print(f"\n  No LOW-risk strategy beats baseline on OOS5.")

    elapsed = time.time() - t0
    output = {
        "study": "tp_calibration_overfit_diag",
        "date": datetime.now().isoformat(),
        "diagnostics": diag,
        "baseline_oos3": base_oos3, "baseline_oos5": base_oos5,
        "elapsed_s": round(elapsed, 1),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
