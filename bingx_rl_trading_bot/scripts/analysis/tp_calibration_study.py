#!/usr/bin/env python3
"""
Per-Pattern TP Calibration Study
====================================
현행: uniform TP×0.72 (모든 패턴 동일 비율 축소)
연구: TP < SL 구조 내에서 per-pattern 최적화/보정 전략 비교

전략:
  S0) BASELINE: uniform TP×0.72 (현행)
  S1) RR-BUCKET: TP/SL ratio 구간별 차등 scale (저R:R는 축소, 고R:R는 확대)
  S2) WR-ADAPTIVE: WR 기반 TP scale (고WR → 더 축소, 저WR → 덜 축소)
  S3) EDGE-WEIGHTED: Edge 크기 비례 TP 배분
  S4) MFE-MEDIAN: MFE 중앙값을 직접 TP로 사용 (percentile grid 대신)
  S5) TP-SL-JOINT: TP뿐 아니라 SL도 동시 보정 (SL 축소→R:R 개선)
  S6) TRADE-COUNT: 거래수 기반 신뢰도 보정 (소표본 보수적)
  S7) PER-PATTERN-SWEEP: 각 패턴별 개별 TP scale 0.4~1.2 sweep (overfitting risk)

각 전략에 WF 3-fold + Discrimination test 수행.

Output: results/tp_calibration_study.json
"""
import os, sys, json, time, numpy as np
from datetime import datetime
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from scripts.scanner.pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    compute_atr_ratio, compute_ema_slope, calc_stats_compound,
    portfolio_npos, compute_excursions,
    FEE_PCT, LEVERAGE, SLIPPAGE_BUFFER, MAX_DAILY_LOSS_PCT, MAX_BARS,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, DEFAULT_REGIME_MULT,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI, TIMEOUT_BARS,
    DEFAULT_PREEMPTIVE_CASCADE_PCT, DEFAULT_PREEMPTIVE_TIGHTEN_PCT,
)

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "btc_5m_270days_reclassified.csv")
PATTERNS_FILE = os.path.join(PROJECT_ROOT, "results", "dynamic_patterns.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "results", "tp_calibration_study.json")


def load_data():
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    type_codes = df["candle_type"].tolist()
    nw = find_neutral_window(closes, tol_pct=1.0)
    atr = compute_atr_ratio(highs, lows, closes)
    ema = compute_ema_slope(closes)
    sig_idx = build_signal_index(type_codes, n_bars)
    with open(PATTERNS_FILE) as f:
        pd_ = json.load(f)["pattern_details"]
    return df, n_bars, opens, highs, lows, closes, type_codes, nw, atr, ema, sig_idx, pd_


def make_sigs_with_scaling(pd_, sig_idx, s, e, tp_scale_fn):
    """Build signal tuples with per-pattern TP scaling.

    tp_scale_fn(info) → float: per-pattern TP scale factor.
    """
    t = []
    for k, info in pd_.items():
        p, d = info["pattern"], info["direction"]
        raw_tp, sl = info["tp"], info["sl"]
        scale = tp_scale_fn(info)
        tp = raw_tp * scale
        tp = max(0.3, tp)  # floor
        if p not in sig_idx:
            continue
        for b in sig_idx[p]:
            if s <= b < e:
                t.append((b, p, d, tp, sl))
    return sorted(t, key=lambda x: x[0])


def run_bt(sigs, opens, highs, lows, closes, n_bars, atr, ema, start, end):
    if not sigs:
        return None
    trades, stats = portfolio_npos(sigs, opens, highs, lows, closes, n_bars,
                                    atr, ema, start, end)
    st = calc_stats_compound(trades)
    tp_c = sum(1 for t in trades if t.get("reason") == "TP")
    sl_c = sum(1 for t in trades if t.get("reason") == "SL")
    tp_pnls = [abs(t["pnl_slot"]) for t in trades if t.get("reason") == "TP"]
    sl_pnls = [abs(t["pnl_slot"]) for t in trades if t.get("reason") == "SL"]
    rr = np.mean(tp_pnls) / np.mean(sl_pnls) if sl_pnls and tp_pnls else 0
    mdd = stats.get("mdd_mtm", st["mdd"])
    pm = st["pnl"] / mdd if mdd > 0 else 0
    # BE WR = avg_loss / (avg_win + avg_loss)
    avg_w = np.mean(tp_pnls) if tp_pnls else 0
    avg_l = np.mean(sl_pnls) if sl_pnls else 0
    be_wr = avg_l / (avg_w + avg_l) * 100 if (avg_w + avg_l) > 0 else 50
    wr_margin = st["wr"] - be_wr
    return {
        "trades": st["trades"], "wr": st["wr"], "pnl": st["pnl"],
        "mdd": round(mdd, 2), "pm": round(pm, 1), "rr": round(rr, 3),
        "tp": tp_c, "sl": sl_c, "be_wr": round(be_wr, 1),
        "wr_margin": round(wr_margin, 1),
    }


def run_wf(make_fn, opens, highs, lows, closes, n_bars, nws, nwe, atr_full, ema_full, nf=3):
    seg = (nwe - nws) // (nf + 1)
    folds = []
    for f in range(nf):
        ie = nws + (f + 1) * seg
        os_ = ie
        oe = nws + (f + 2) * seg if f < nf - 1 else nwe
        if os_ >= oe:
            folds.append({"fold": f + 1, "oos_pnl": 0})
            continue
        sigs = make_fn(os_, oe)
        if not sigs:
            folds.append({"fold": f + 1, "oos_pnl": 0})
            continue
        atr = compute_atr_ratio(highs[:oe], lows[:oe], closes[:oe])
        ema = compute_ema_slope(closes[:oe])
        trades, stats = portfolio_npos(sigs, opens, highs, lows, closes, oe,
                                        atr, ema, os_, oe)
        st = calc_stats_compound(trades)
        folds.append({"fold": f + 1, "oos_pnl": st["pnl"]})
    total = sum(f["oos_pnl"] for f in folds)
    np_ = sum(1 for f in folds if f["oos_pnl"] > 0)
    return {"folds": folds, "total_oos": round(total, 1), "verdict": "PASS" if np_ == nf else "FAIL"}


def run_disc(make_fn, pd_, sig_idx, opens, highs, lows, closes, n_bars, nws, nwe, seeds=[42, 123, 7, 99, 256]):
    """Discrimination test: shuffle pattern directions."""
    real_wf = run_wf(make_fn, opens, highs, lows, closes, n_bars, nws, nwe, None, None)
    passes = 0
    for seed in seeds:
        rng = np.random.RandomState(seed)
        # Make shuffled version — randomize direction per pattern
        shuffled_dirs = {}
        for k in pd_:
            shuffled_dirs[k] = rng.choice(["LONG", "SHORT"])

        def make_shuf(s, e, _sd=shuffled_dirs):
            t = []
            for k, info in pd_.items():
                p = info["pattern"]
                d = _sd[k]
                raw_tp, sl = info["tp"], info["sl"]
                tp = raw_tp * 0.72  # baseline for disc test
                tp = max(0.3, tp)
                if p not in sig_idx:
                    continue
                for b in sig_idx[p]:
                    if s <= b < e:
                        t.append((b, p, d, tp, sl))
            return sorted(t, key=lambda x: x[0])

        swf = run_wf(make_shuf, opens, highs, lows, closes, n_bars, nws, nwe, None, None)
        if swf["verdict"] == "PASS":
            passes += 1
    return {
        "real_oos": real_wf["total_oos"], "real_verdict": real_wf["verdict"],
        "shuffled_passes": passes, "discrimination": "DISC" if passes <= 1 else "NON-DISC",
    }


def main():
    t0 = time.time()
    print("Per-Pattern TP Calibration Study")
    print("=" * 90)

    df, n_bars, opens, highs, lows, closes, type_codes, nw, atr, ema, sig_idx, pd_ = load_data()
    nws, nwe = nw
    print(f"  {n_bars} bars, NW [{nws},{nwe}], {len(pd_)} patterns")

    # Precompute per-pattern stats
    for k, info in pd_.items():
        rr = info["tp"] / info["sl"] if info["sl"] > 0 else 0
        info["_rr"] = rr
        info["_edge"] = info.get("edge", info["wr"] - info["sl"] / (info["tp"] + info["sl"]) * 100)

    # Compute MFE medians per pattern (for S4)
    print("  Computing MFE medians per pattern...")
    mfe_medians = {}
    for k, info in pd_.items():
        p, d = info["pattern"], info["direction"]
        if p not in sig_idx:
            continue
        nw_bars = sorted([b for b in sig_idx[p] if nws <= b < nwe])
        if len(nw_bars) < 10:
            mfe_medians[k] = info["tp"]  # fallback to scanner TP
            continue
        exc = compute_excursions(nw_bars, d, opens, highs, lows, n_bars, MAX_BARS)
        if exc:
            mfe_medians[k] = round(float(np.median([e["mfe_pct"] for e in exc])), 4)
        else:
            mfe_medians[k] = info["tp"]

    # =========================================================================
    # Define calibration strategies
    # =========================================================================

    strategies = {}

    # S0: Baseline — uniform 0.72
    strategies["S0_uniform_0.72"] = lambda info: 0.72

    # S1: RR-Bucket — differentiated by TP/SL ratio
    # Low R:R (<0.5): more aggressive scaling (TP very small relative to SL)
    # High R:R (>0.6): less scaling (TP already reasonable)
    def s1_rr_bucket(info):
        rr = info["_rr"]
        if rr < 0.45:
            return 0.65   # aggressive shrink for very low R:R
        elif rr < 0.55:
            return 0.72   # baseline
        else:
            return 0.82   # expand for higher R:R
    strategies["S1_rr_bucket"] = s1_rr_bucket

    # S1b: RR-Bucket v2 — finer grid
    def s1b_rr_bucket_v2(info):
        rr = info["_rr"]
        if rr < 0.45:
            return 0.60
        elif rr < 0.50:
            return 0.68
        elif rr < 0.55:
            return 0.75
        elif rr < 0.65:
            return 0.82
        else:
            return 0.90
    strategies["S1b_rr_fine"] = s1b_rr_bucket_v2

    # S2: WR-Adaptive — high WR can afford tighter TP (faster slot rotation)
    # Logic: WR 90%+ → scale 0.60 (tight), WR 80-90% → 0.72, WR 76-80% → 0.85
    def s2_wr_adaptive(info):
        wr = info["wr"]
        if wr >= 90:
            return 0.60
        elif wr >= 85:
            return 0.68
        elif wr >= 80:
            return 0.75
        else:
            return 0.85   # low WR → need larger TP to compensate
    strategies["S2_wr_adaptive"] = s2_wr_adaptive

    # S2b: WR-Linear — continuous mapping
    def s2b_wr_linear(info):
        wr = info["wr"]
        # WR 96% → 0.55, WR 76% → 0.90 (linear interpolation)
        scale = 0.90 - (wr - 76) / (96 - 76) * (0.90 - 0.55)
        return max(0.50, min(0.95, scale))
    strategies["S2b_wr_linear"] = s2b_wr_linear

    # S3: Edge-Weighted — higher edge patterns get proportionally more TP
    def s3_edge_weighted(info):
        edge = info["_edge"]
        # Edge 18pp → 0.72 (baseline), Edge 30pp → 0.62, Edge 10pp → 0.85
        # More edge = can afford tighter TP for rotation
        scale = 0.72 - (edge - 18) / 30 * 0.15
        return max(0.55, min(0.90, scale))
    strategies["S3_edge_weight"] = s3_edge_weighted

    # S4: MFE-Median — use actual MFE median as TP (not grid-searched percentile)
    def s4_mfe_median(info):
        k = f"{info['pattern']}_{info['direction']}"
        mfe_med = mfe_medians.get(k, info["tp"])
        if info["tp"] > 0:
            # Scale = MFE_median / scanner_TP (replaces grid TP with median MFE)
            return mfe_med / info["tp"]
        return 0.72
    strategies["S4_mfe_median"] = s4_mfe_median

    # S5: TP-SL Joint — boost TP and tighten SL simultaneously
    # Implementation: TP×0.80, SL×0.85 → improved R:R without changing TP alone
    # This requires custom signal generation (modifying SL too)

    # S6: Trade-Count — confidence-based scaling
    # Few trades → conservative (shrink more), many trades → trust scanner TP
    def s6_trade_count(info):
        n = info["trades"]
        if n >= 80:
            return 0.80   # high confidence, closer to scanner optimal
        elif n >= 50:
            return 0.72   # baseline
        elif n >= 35:
            return 0.68   # moderate shrink
        else:
            return 0.62   # conservative for small samples
    strategies["S6_trade_count"] = s6_trade_count

    # S7: Combined — WR + R:R joint calibration
    def s7_combined(info):
        wr = info["wr"]
        rr = info["_rr"]
        # Base from WR
        base = 0.90 - (wr - 76) / (96 - 76) * (0.90 - 0.55)
        # Adjust by R:R (high R:R → push scale up)
        rr_adj = (rr - 0.45) / (0.80 - 0.45) * 0.10  # ±0.10 adjustment
        scale = base + rr_adj
        return max(0.50, min(0.95, scale))
    strategies["S7_wr_rr_combo"] = s7_combined

    # S8: Inverted — what if we EXPAND TP instead of shrinking?
    strategies["S8_expand_0.85"] = lambda info: 0.85
    strategies["S8b_expand_0.90"] = lambda info: 0.90
    strategies["S8c_expand_1.00"] = lambda info: 1.00

    # =========================================================================
    # PHASE 1: IS comparison — all strategies
    # =========================================================================
    print(f"\n{'='*90}")
    print("PHASE 1: In-Sample Comparison (all strategies)")
    print(f"{'='*90}")

    hdr = f"  {'Strategy':>20s} | {'PnL':>8s} | {'MDD':>5s} | {'P/M':>6s} | {'WR':>5s} | {'R:R':>5s} | {'BE_WR':>5s} | {'Margin':>6s} | {'Trades':>6s} | {'TP':>4s} | {'SL':>4s}"
    print(hdr)
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*4}-+-{'-'*4}")

    is_results = {}
    for name, fn in sorted(strategies.items()):
        make_fn = lambda s, e, _fn=fn: make_sigs_with_scaling(pd_, sig_idx, s, e, _fn)
        sigs = make_fn(nws, nwe)
        r = run_bt(sigs, opens, highs, lows, closes, n_bars, atr, ema, nws, nwe)
        if r:
            is_results[name] = r
            mk = " <<BASE" if name == "S0_uniform_0.72" else ""
            print(f"  {name:>20s} | {r['pnl']:+7.1f}% | {r['mdd']:4.1f}% | {r['pm']:5.1f}x | "
                  f"{r['wr']:.1f}% | {r['rr']:.3f} | {r['be_wr']:.1f}% | {r['wr_margin']:+5.1f}pp | "
                  f"{r['trades']:>6d} | {r['tp']:>4d} | {r['sl']:>4d}{mk}")

    # =========================================================================
    # PHASE 2: S5 — Joint TP + SL calibration (separate because modifies SL)
    # =========================================================================
    print(f"\n{'='*90}")
    print("PHASE 2: Joint TP + SL Calibration")
    print(f"{'='*90}")

    def make_joint_sigs(pd_, sig_idx, s, e, tp_scale, sl_scale):
        t = []
        for k, info in pd_.items():
            p, d = info["pattern"], info["direction"]
            tp = max(0.3, info["tp"] * tp_scale)
            sl = max(0.5, info["sl"] * sl_scale)
            if p not in sig_idx:
                continue
            for b in sig_idx[p]:
                if s <= b < e:
                    t.append((b, p, d, tp, sl))
        return sorted(t, key=lambda x: x[0])

    print(f"  {'TP×':>5s} {'SL×':>5s} | {'PnL':>8s} | {'MDD':>5s} | {'P/M':>6s} | {'WR':>5s} | {'R:R':>5s} | {'BE_WR':>5s} | {'Margin':>6s} | {'Trades':>6s}")
    print(f"  {'-'*5} {'-'*5}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}")

    joint_results = {}
    for tp_s, sl_s in [
        (0.72, 1.00),  # baseline
        (0.80, 0.90),  # boost TP, tighten SL
        (0.80, 0.85),
        (0.80, 0.80),
        (0.75, 0.90),
        (0.75, 0.85),
        (0.72, 0.90),
        (0.72, 0.85),
        (0.85, 0.90),
        (0.85, 0.85),
        (0.90, 0.80),
        (0.90, 0.75),
        (0.65, 0.90),
        (0.65, 0.85),
    ]:
        label = f"TP{tp_s:.2f}_SL{sl_s:.2f}"
        sigs = make_joint_sigs(pd_, sig_idx, nws, nwe, tp_s, sl_s)
        r = run_bt(sigs, opens, highs, lows, closes, n_bars, atr, ema, nws, nwe)
        if r:
            joint_results[label] = {"is": r, "tp_scale": tp_s, "sl_scale": sl_s}
            mk = " <<BASE" if tp_s == 0.72 and sl_s == 1.00 else ""
            print(f"  {tp_s:5.2f} {sl_s:5.2f} | {r['pnl']:+7.1f}% | {r['mdd']:4.1f}% | {r['pm']:5.1f}x | "
                  f"{r['wr']:.1f}% | {r['rr']:.3f} | {r['be_wr']:.1f}% | {r['wr_margin']:+5.1f}pp | "
                  f"{r['trades']:>6d}{mk}")

    # =========================================================================
    # PHASE 3: WF validation — top strategies + joint
    # =========================================================================
    print(f"\n{'='*90}")
    print("PHASE 3: Walk-Forward Validation (top strategies)")
    print(f"{'='*90}")

    # Select top-5 IS P/M strategies + baseline + top joint combos
    sorted_is = sorted(is_results.items(), key=lambda x: x[1]["pm"], reverse=True)
    top_strats = [("S0_uniform_0.72", strategies["S0_uniform_0.72"])]  # always include baseline
    for name, _ in sorted_is[:6]:
        if name != "S0_uniform_0.72":
            top_strats.append((name, strategies[name]))

    sorted_joint = sorted(joint_results.items(), key=lambda x: x[1]["is"]["pm"], reverse=True)
    top_joints = sorted_joint[:4]

    print(f"  {'Strategy':>24s} | {'IS PnL':>8s} | {'IS P/M':>6s} | {'IS R:R':>5s} | {'IS Margin':>9s} | {'OOS':>8s} | WF")
    print(f"  {'-'*24}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*9}-+-{'-'*8}-+-{'-'*4}")

    wf_results = {}
    for name, fn in top_strats:
        make_fn = lambda s, e, _fn=fn: make_sigs_with_scaling(pd_, sig_idx, s, e, _fn)
        wf = run_wf(make_fn, opens, highs, lows, closes, n_bars, nws, nwe, atr, ema)
        r = is_results[name]
        wf_results[name] = {"is": r, "wf": wf}
        mk = " <<BASE" if name == "S0_uniform_0.72" else ""
        print(f"  {name:>24s} | {r['pnl']:+7.1f}% | {r['pm']:5.1f}x | {r['rr']:.3f} | "
              f"{r['wr_margin']:+7.1f}pp | {wf['total_oos']:+7.1f}% | {wf['verdict']}{mk}")

    for label, jinfo in top_joints:
        tp_s, sl_s = jinfo["tp_scale"], jinfo["sl_scale"]
        make_fn = lambda s, e, _t=tp_s, _s=sl_s: make_joint_sigs(pd_, sig_idx, s, e, _t, _s)
        wf = run_wf(make_fn, opens, highs, lows, closes, n_bars, nws, nwe, atr, ema)
        r = jinfo["is"]
        wf_results[label] = {"is": r, "wf": wf}
        print(f"  {label:>24s} | {r['pnl']:+7.1f}% | {r['pm']:5.1f}x | {r['rr']:.3f} | "
              f"{r['wr_margin']:+7.1f}pp | {wf['total_oos']:+7.1f}% | {wf['verdict']}")

    # =========================================================================
    # PHASE 4: Discrimination test — top 3 by OOS
    # =========================================================================
    print(f"\n{'='*90}")
    print("PHASE 4: Discrimination Test (top 3 OOS strategies)")
    print(f"{'='*90}")

    sorted_wf = sorted(wf_results.items(), key=lambda x: x[1]["wf"]["total_oos"], reverse=True)
    disc_candidates = [("S0_uniform_0.72", strategies.get("S0_uniform_0.72"))]
    for name, _ in sorted_wf[:3]:
        if name != "S0_uniform_0.72" and name in strategies:
            disc_candidates.append((name, strategies[name]))

    disc_results = {}
    for name, fn in disc_candidates:
        if fn is None:
            continue
        make_fn = lambda s, e, _fn=fn: make_sigs_with_scaling(pd_, sig_idx, s, e, _fn)
        dr = run_disc(make_fn, pd_, sig_idx, opens, highs, lows, closes, n_bars, nws, nwe)
        disc_results[name] = dr
        print(f"  {name:>24s}: OOS {dr['real_oos']:+.1f}% | Shuffled PASS: {dr['shuffled_passes']}/5 → {dr['discrimination']}")

    # =========================================================================
    # PHASE 5: Per-pattern TP scale distribution analysis
    # =========================================================================
    print(f"\n{'='*90}")
    print("PHASE 5: Scale Distribution of Top Strategies")
    print(f"{'='*90}")

    for name in ["S1_rr_bucket", "S2_wr_adaptive", "S2b_wr_linear", "S7_wr_rr_combo"]:
        fn = strategies.get(name)
        if fn is None:
            continue
        scales = [fn(info) for info in pd_.values()]
        print(f"  {name:>20s}: min={min(scales):.2f} max={max(scales):.2f} med={np.median(scales):.2f} "
              f"mean={np.mean(scales):.2f} std={np.std(scales):.3f}")
        # Histogram
        bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        counts, _ = np.histogram(scales, bins=bins)
        dist_str = " ".join(f"{b:.2f}:{c}" for b, c in zip(bins, counts) if c > 0)
        print(f"    distribution: {dist_str}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*90}")
    print("SUMMARY — Ranked by OOS PnL")
    print(f"{'='*90}")
    print(f"  {'#':>2s} {'Strategy':>24s} | {'IS PnL':>8s} | {'IS P/M':>6s} | {'R:R':>5s} | {'Margin':>7s} | {'OOS':>8s} | WF  | Disc")
    print(f"  {'-'*2} {'-'*24}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*4}-+-{'-'*8}")

    base_oos = wf_results.get("S0_uniform_0.72", {}).get("wf", {}).get("total_oos", 0)
    for i, (name, v) in enumerate(sorted_wf[:15], 1):
        r = v["is"]
        wf = v["wf"]
        delta = wf["total_oos"] - base_oos
        disc = disc_results.get(name, {}).get("discrimination", "-")
        mk = " <<BASE" if name == "S0_uniform_0.72" else f" ({delta:+.1f})"
        print(f"  {i:2d} {name:>24s} | {r['pnl']:+7.1f}% | {r['pm']:5.1f}x | {r['rr']:.3f} | "
              f"{r['wr_margin']:+5.1f}pp | {wf['total_oos']:+7.1f}% | {wf['verdict']} | {disc}{mk}")

    elapsed = time.time() - t0
    output = {
        "study": "tp_calibration",
        "date": datetime.now().isoformat(),
        "pattern_profile": {
            "count": len(pd_),
            "tp_range": [round(min(v["tp"] for v in pd_.values()), 3),
                         round(max(v["tp"] for v in pd_.values()), 3)],
            "sl_range": [round(min(v["sl"] for v in pd_.values()), 3),
                         round(max(v["sl"] for v in pd_.values()), 3)],
            "rr_range": [round(min(v["_rr"] for v in pd_.values()), 3),
                         round(max(v["_rr"] for v in pd_.values()), 3)],
        },
        "phase1_is": is_results,
        "phase2_joint": joint_results,
        "phase3_wf": wf_results,
        "phase4_disc": disc_results,
        "elapsed_s": round(elapsed, 1),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
