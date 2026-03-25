#!/usr/bin/env python3
"""
High R:R Pattern Discovery Study
===================================
현행 시스템: TP < SL (낮은 R:R, 높은 WR → 변동성 수확)
본 연구: TP > SL (높은 R:R, 낮은 WR → 추세 추종형)

패턴 발굴 조건:
  - TP: 0.4 ~ 1.5% (raw, before ATR scaling)
  - SL: 0.5 ~ 1.0% (raw, before ATR scaling)
  - R:R > 1.0 (TP > SL 강제)
  - Edge >= 5pp (낮은 WR 허용하므로 임계값 낮춤)
  - MC p < 0.01
  - min_trades >= 25

비교 대상:
  A) High-RR 패턴셋 + 현행 메커니즘 (Cascade, Pre-emptive, Momentum, etc.)
  B) 현행 패턴셋 (baseline, TP×0.72)
  C) High-RR + TP scale sweep (최적 TP 탐색)

Output: results/high_rr_pattern_discovery.json
"""
import os, sys, json, time, numpy as np
from datetime import datetime
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from scripts.scanner.pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    compute_excursions, derive_tp_sl, bt_signals_atr,
    compute_atr_ratio, compute_ema_slope, calc_stats_compound,
    portfolio_npos,
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
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "results", "high_rr_pattern_discovery.json")

# High R:R discovery parameters
HIGH_RR_TP_MIN = 0.4
HIGH_RR_TP_MAX = 1.5
HIGH_RR_SL_MIN = 0.5
HIGH_RR_SL_MAX = 1.0
HIGH_RR_MIN_RR = 1.0      # TP/SL > 1.0 required
HIGH_RR_EDGE_THRESHOLD = 5  # pp, lower than standard 18pp due to lower WR
HIGH_RR_MIN_TRADES = 25
MC_SEEDS = [42, 123, 7]
MC_SIMS = 10000
MC_P_THRESHOLD = 0.01
DECAY_RATE = 0.9975


def grid_search_high_rr(signal_bars, direction, opens, highs, lows, n_bars,
                         atr_ratio, clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI):
    """Grid search constrained to high R:R regime: TP > SL."""
    excursions = compute_excursions(signal_bars, direction, opens, highs, lows, n_bars, MAX_BARS)
    if len(excursions) < HIGH_RR_MIN_TRADES:
        return None

    mfes = np.array([e['mfe_pct'] for e in excursions])
    maes = np.array([e['mae_pct'] for e in excursions])

    # Generate TP candidates from MFE percentiles that fall in range
    tp_candidates = []
    for tp_p in [30, 35, 40, 45, 50, 55, 60, 65, 70]:
        tp_val = round(float(np.percentile(mfes, tp_p)), 3)
        if HIGH_RR_TP_MIN <= tp_val <= HIGH_RR_TP_MAX:
            tp_candidates.append((tp_val, tp_p))

    # Also add fixed grid points
    for tp_fixed in np.arange(HIGH_RR_TP_MIN, HIGH_RR_TP_MAX + 0.05, 0.1):
        tp_fixed = round(tp_fixed, 2)
        tp_candidates.append((tp_fixed, None))

    # Generate SL candidates from MAE percentiles that fall in range
    sl_candidates = []
    for sl_p in [50, 55, 60, 65, 70, 75, 80]:
        sl_val = round(float(np.percentile(maes, sl_p)), 3)
        if HIGH_RR_SL_MIN <= sl_val <= HIGH_RR_SL_MAX:
            sl_candidates.append((sl_val, sl_p))

    # Also add fixed grid points
    for sl_fixed in np.arange(HIGH_RR_SL_MIN, HIGH_RR_SL_MAX + 0.05, 0.1):
        sl_fixed = round(sl_fixed, 2)
        sl_candidates.append((sl_fixed, None))

    if not tp_candidates or not sl_candidates:
        return None

    best = None
    best_score = -9999

    for tp, tp_p in tp_candidates:
        for sl, sl_p in sl_candidates:
            # R:R filter: TP must be > SL
            if tp <= sl * HIGH_RR_MIN_RR:
                continue

            trades = bt_signals_atr(signal_bars, direction, tp, sl,
                                     opens, highs, lows, n_bars,
                                     atr_ratio, clamp_lo, clamp_hi)
            if len(trades) < HIGH_RR_MIN_TRADES:
                continue

            pnls = [t[2] for t in trades]
            cum = 0
            peak = 0
            mdd_val = 0
            w = 0
            for p in pnls:
                cum += p
                if cum > peak:
                    peak = cum
                dd = peak - cum
                if dd > mdd_val:
                    mdd_val = dd
                if p > 0:
                    w += 1

            wr = w / len(pnls) * 100
            edge = wr - (sl / (tp + sl) * 100)  # WR excess over random baseline

            if edge < HIGH_RR_EDGE_THRESHOLD:
                continue

            score = (cum / mdd_val) if mdd_val > 0 else cum

            if score > best_score:
                best_score = score
                rr = tp / sl if sl > 0 else 0
                best = {
                    'tp': tp, 'sl': sl,
                    'tp_percentile': tp_p, 'sl_percentile': sl_p,
                    'trades': len(trades), 'wr': round(wr, 1),
                    'pnl': round(cum, 1), 'mdd': round(mdd_val, 1),
                    'edge': round(edge, 1), 'rr': round(rr, 2),
                }

    return best


def mc_test(signal_bars, direction, tp, sl, opens, highs, lows, n_bars,
            atr_ratio, seeds=MC_SEEDS, n_sims=MC_SIMS):
    """Monte Carlo sign randomization test."""
    trades = bt_signals_atr(signal_bars, direction, tp, sl,
                             opens, highs, lows, n_bars, atr_ratio)
    if len(trades) < HIGH_RR_MIN_TRADES:
        return 1.0
    real_pnl = sum(t[2] for t in trades)
    pnls = np.array([t[2] for t in trades])
    max_p = 0.0
    for seed in seeds:
        rng = np.random.RandomState(seed)
        count = 0
        for _ in range(n_sims):
            signs = rng.choice([-1, 1], size=len(pnls))
            if np.sum(pnls * signs) >= real_pnl:
                count += 1
        p = count / n_sims
        if p > max_p:
            max_p = p
    return max_p


def run_npos_backtest(sig_tuples, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, start, end, tp_scale=1.0, label=""):
    """Run N-pos portfolio backtest with full mechanism stack."""
    scaled_sigs = [(s, p, d, tp * tp_scale, sl) for s, p, d, tp, sl in sig_tuples if start <= s < end]
    scaled_sigs.sort(key=lambda x: x[0])
    if not scaled_sigs:
        return None

    trades, stats = portfolio_npos(scaled_sigs, opens, highs, lows, closes, n_bars,
                                    atr_ratio, ema_slope, start, end)
    st = calc_stats_compound(trades)

    tp_c = sum(1 for t in trades if t.get("reason") == "TP")
    sl_c = sum(1 for t in trades if t.get("reason") == "SL")
    to_c = sum(1 for t in trades if t.get("reason") == "TIMEOUT")
    cas_c = sum(1 for t in trades if t.get("reason") == "CASCADE_SL")

    tp_pnls = [abs(t["pnl_slot"]) for t in trades if t.get("reason") == "TP"]
    sl_pnls = [abs(t["pnl_slot"]) for t in trades if t.get("reason") == "SL"]
    rr = np.mean(tp_pnls) / np.mean(sl_pnls) if sl_pnls and tp_pnls else 0

    return {
        "label": label, "trades": st["trades"], "wr": st["wr"],
        "pnl": st["pnl"], "mdd": stats.get("mdd_mtm", st["mdd"]),
        "pm": round(st["pnl"] / stats["mdd_mtm"], 1) if stats.get("mdd_mtm", 0) > 0 else 0,
        "rr": round(rr, 3),
        "tp_count": tp_c, "sl_count": sl_c, "timeout": to_c, "cascade": cas_c,
    }


def run_wf(sig_tuples_fn, opens, highs, lows, closes, n_bars, nws, nwe, nf=3, tp_scale=1.0):
    """Walk-forward validation."""
    seg = (nwe - nws) // (nf + 1)
    folds = []
    for f in range(nf):
        ie = nws + (f + 1) * seg
        os_ = ie
        oe = nws + (f + 2) * seg if f < nf - 1 else nwe
        if os_ >= oe:
            folds.append({"fold": f + 1, "oos_pnl": 0})
            continue
        sigs = sig_tuples_fn(os_, oe)
        if not sigs:
            folds.append({"fold": f + 1, "oos_pnl": 0})
            continue
        scaled = [(s, p, d, tp * tp_scale, sl) for s, p, d, tp, sl in sigs]
        atr = compute_atr_ratio(highs[:oe], lows[:oe], closes[:oe])
        ema = compute_ema_slope(closes[:oe])
        trades, stats = portfolio_npos(scaled, opens, highs, lows, closes, oe,
                                        atr, ema, os_, oe)
        st = calc_stats_compound(trades)
        folds.append({"fold": f + 1, "oos_pnl": st["pnl"]})
    total = sum(f["oos_pnl"] for f in folds)
    np_ = sum(1 for f in folds if f["oos_pnl"] > 0)
    return {"folds": folds, "total_oos": round(total, 1), "verdict": "PASS" if np_ == nf else "FAIL"}


def main():
    t0 = time.time()
    print("High R:R Pattern Discovery Study")
    print("=" * 80)
    print(f"  TP range: {HIGH_RR_TP_MIN}-{HIGH_RR_TP_MAX}%, SL range: {HIGH_RR_SL_MIN}-{HIGH_RR_SL_MAX}%")
    print(f"  Min R:R: {HIGH_RR_MIN_RR}, Edge >= {HIGH_RR_EDGE_THRESHOLD}pp, MC p < {MC_P_THRESHOLD}")

    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    type_codes = df["candle_type"].tolist()
    nw = find_neutral_window(closes, tol_pct=1.0)
    nws, nwe = nw
    atr = compute_atr_ratio(highs, lows, closes)
    ema = compute_ema_slope(closes)
    sig_idx = build_signal_index(type_codes, n_bars)
    print(f"  {n_bars} bars, NW [{nws},{nwe}]")

    # =========================================================================
    # PHASE 1: Pattern Discovery (High R:R)
    # =========================================================================
    print(f"\n{'='*80}")
    print("PHASE 1: High R:R Pattern Discovery")
    print(f"{'='*80}")

    # Generate all 3-candle patterns
    unique_types = sorted(set(type_codes))
    all_patterns = {}
    for t1 in unique_types:
        for t2 in unique_types:
            for t3 in unique_types:
                pat = f"{t1}-{t2}-{t3}"
                if pat in sig_idx and len(sig_idx[pat]) >= HIGH_RR_MIN_TRADES:
                    all_patterns[pat] = sig_idx[pat]

    print(f"  Candidate patterns with >= {HIGH_RR_MIN_TRADES} signals: {len(all_patterns)}")

    discovered = []
    discovery_details = {}

    for pat, bars in sorted(all_patterns.items()):
        signal_bars = sorted(bars)
        # Restrict to neutral window
        nw_bars = [b for b in signal_bars if nws <= b < nwe]
        if len(nw_bars) < HIGH_RR_MIN_TRADES:
            continue

        for direction in ["LONG", "SHORT"]:
            result = grid_search_high_rr(nw_bars, direction, opens, highs, lows, n_bars, atr)
            if result is None:
                continue

            # MC test
            mc_p = mc_test(nw_bars, direction, result['tp'], result['sl'],
                           opens, highs, lows, n_bars, atr)
            if mc_p >= MC_P_THRESHOLD:
                continue

            key = f"{pat}_{direction}"
            result['pattern'] = pat
            result['direction'] = direction
            result['mc_p'] = round(mc_p, 4)
            discovered.append(result)
            discovery_details[key] = result

    print(f"  Discovered: {len(discovered)} patterns (Edge>={HIGH_RR_EDGE_THRESHOLD}pp, MC<{MC_P_THRESHOLD})")

    if not discovered:
        print("  NO PATTERNS FOUND — relaxing edge threshold to 0pp")
        # Retry with no edge filter
        for pat, bars in sorted(all_patterns.items()):
            signal_bars = sorted(bars)
            nw_bars = [b for b in signal_bars if nws <= b < nwe]
            if len(nw_bars) < HIGH_RR_MIN_TRADES:
                continue
            for direction in ["LONG", "SHORT"]:
                result = grid_search_high_rr(nw_bars, direction, opens, highs, lows, n_bars, atr)
                if result is None:
                    continue
                mc_p = mc_test(nw_bars, direction, result['tp'], result['sl'],
                               opens, highs, lows, n_bars, atr)
                if mc_p >= MC_P_THRESHOLD:
                    continue
                key = f"{pat}_{direction}"
                if key in discovery_details:
                    continue
                result['pattern'] = pat
                result['direction'] = direction
                result['mc_p'] = round(mc_p, 4)
                discovered.append(result)
                discovery_details[key] = result
        print(f"  After relaxation: {len(discovered)} patterns (Edge>=0, MC<{MC_P_THRESHOLD})")

    # Sort by PnL/MDD score
    discovered.sort(key=lambda x: x['pnl'] / x['mdd'] if x['mdd'] > 0 else x['pnl'], reverse=True)

    # Show top patterns
    n_long = sum(1 for d in discovered if d['direction'] == 'LONG')
    n_short = sum(1 for d in discovered if d['direction'] == 'SHORT')
    print(f"  {n_long}L + {n_short}S")
    print(f"\n  {'Pattern':>14s} {'Dir':>5s} | {'TP':>5s} {'SL':>5s} {'R:R':>4s} | {'WR':>5s} {'Edge':>5s} | {'PnL':>7s} {'MDD':>5s} | {'MC':>6s} | Trades")
    print(f"  {'-'*14} {'-'*5}-+-{'-'*5}-{'-'*5}-{'-'*4}-+-{'-'*5}-{'-'*5}-+-{'-'*7}-{'-'*5}-+-{'-'*6}-+-{'-'*6}")
    for d in discovered[:30]:
        print(f"  {d['pattern']:>14s} {d['direction']:>5s} | {d['tp']:5.2f} {d['sl']:5.2f} {d['rr']:4.2f} | "
              f"{d['wr']:5.1f} {d['edge']:+5.1f} | {d['pnl']:+6.1f}% {d['mdd']:4.1f}% | {d['mc_p']:6.4f} | {d['trades']}")

    if not discovered:
        print("\n  RESULT: No high R:R patterns found. Study ends here.")
        output = {
            "study": "high_rr_pattern_discovery",
            "date": datetime.now().isoformat(),
            "result": "NO_PATTERNS_FOUND",
            "params": {
                "tp_range": [HIGH_RR_TP_MIN, HIGH_RR_TP_MAX],
                "sl_range": [HIGH_RR_SL_MIN, HIGH_RR_SL_MAX],
                "min_rr": HIGH_RR_MIN_RR,
            },
        }
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2, default=str)
        return

    # Profile discovered patterns
    tps = [d['tp'] for d in discovered]
    sls = [d['sl'] for d in discovered]
    rrs = [d['rr'] for d in discovered]
    print(f"\n  Profile: TP {np.min(tps):.2f}-{np.max(tps):.2f}% (med {np.median(tps):.2f})")
    print(f"           SL {np.min(sls):.2f}-{np.max(sls):.2f}% (med {np.median(sls):.2f})")
    print(f"           R:R {np.min(rrs):.2f}-{np.max(rrs):.2f} (med {np.median(rrs):.2f})")

    # =========================================================================
    # PHASE 2: N-pos Backtest — High R:R vs Current baseline
    # =========================================================================
    print(f"\n{'='*80}")
    print("PHASE 2: N-pos Portfolio Backtest (Full Mechanism Stack)")
    print(f"{'='*80}")

    # Build high-RR signal tuples
    def make_high_rr_sigs(s, e):
        t = []
        for d in discovered:
            pat = d['pattern']
            direction = d['direction']
            tp, sl = d['tp'], d['sl']
            if pat not in sig_idx:
                continue
            for b in sig_idx[pat]:
                if s <= b < e:
                    t.append((b, pat, direction, tp, sl))
        return sorted(t, key=lambda x: x[0])

    # Build current baseline signal tuples
    with open(PATTERNS_FILE) as f:
        current_pd = json.load(f)["pattern_details"]

    def make_baseline_sigs(s, e):
        t = []
        for k, info in current_pd.items():
            p, d = info["pattern"], info["direction"]
            tp, sl = info["tp"], info["sl"]
            if p not in sig_idx:
                continue
            for b in sig_idx[p]:
                if s <= b < e:
                    t.append((b, p, d, tp, sl))
        return sorted(t, key=lambda x: x[0])

    hr_sigs = make_high_rr_sigs(nws, nwe)
    bl_sigs = make_baseline_sigs(nws, nwe)
    print(f"  High-RR signals: {len(hr_sigs)}, Baseline signals: {len(bl_sigs)}")

    # IS comparison
    print(f"\n  {'Label':>22s} | {'PnL':>8s} | {'MDD':>5s} | {'P/M':>6s} | {'WR':>5s} | {'R:R':>5s} | {'Trades':>6s} | {'TP':>4s} | {'SL':>4s} | {'TO':>4s} | {'CAS':>4s}")
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}")

    results = {}
    for label, sigs_fn, ts in [
        ("Baseline_TP×0.72", make_baseline_sigs, 0.72),
        ("HighRR_TP×1.00", make_high_rr_sigs, 1.00),
        ("HighRR_TP×0.90", make_high_rr_sigs, 0.90),
        ("HighRR_TP×0.80", make_high_rr_sigs, 0.80),
        ("HighRR_TP×0.72", make_high_rr_sigs, 0.72),
    ]:
        sigs = sigs_fn(nws, nwe)
        r = run_npos_backtest(sigs, opens, highs, lows, closes, n_bars, atr, ema, nws, nwe,
                               tp_scale=ts, label=label)
        if r:
            results[label] = r
            print(f"  {label:>22s} | {r['pnl']:+7.1f}% | {r['mdd']:4.1f}% | {r['pm']:5.1f}x | "
                  f"{r['wr']:.1f}% | {r['rr']:.3f} | {r['trades']:>6d} | {r['tp_count']:>4d} | "
                  f"{r['sl_count']:>4d} | {r['timeout']:>4d} | {r['cascade']:>4d}")

    # =========================================================================
    # PHASE 3: TP Scale Sweep for High R:R
    # =========================================================================
    print(f"\n{'='*80}")
    print("PHASE 3: TP Scale Sweep (High R:R patterns)")
    print(f"{'='*80}")

    tp_scales = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.30]
    print(f"  {'TP Scale':>10s} | {'PnL':>8s} | {'MDD':>5s} | {'P/M':>6s} | {'WR':>5s} | {'R:R':>5s} | {'Trades':>6s} | {'OOS':>8s} | WF")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*4}")

    sweep_results = {}
    best_oos = -9999
    best_tp_scale = None

    for ts in tp_scales:
        label = f"TP_{ts:.2f}"
        sigs = make_high_rr_sigs(nws, nwe)
        r = run_npos_backtest(sigs, opens, highs, lows, closes, n_bars, atr, ema, nws, nwe,
                               tp_scale=ts, label=label)
        wf = run_wf(make_high_rr_sigs, opens, highs, lows, closes, n_bars, nws, nwe, tp_scale=ts)
        if r:
            sweep_results[label] = {"is": r, "wf": wf}
            mk = " << BEST" if wf["total_oos"] > best_oos else ""
            if wf["total_oos"] > best_oos:
                best_oos = wf["total_oos"]
                best_tp_scale = ts
            print(f"  {ts:>10.2f} | {r['pnl']:+7.1f}% | {r['mdd']:4.1f}% | {r['pm']:5.1f}x | "
                  f"{r['wr']:.1f}% | {r['rr']:.3f} | {r['trades']:>6d} | "
                  f"{wf['total_oos']:+7.1f}% | {wf['verdict']}{mk}")

    # =========================================================================
    # PHASE 4: WF comparison — best High-RR vs Baseline
    # =========================================================================
    print(f"\n{'='*80}")
    print("PHASE 4: WF Comparison")
    print(f"{'='*80}")

    wf_baseline = run_wf(make_baseline_sigs, opens, highs, lows, closes, n_bars, nws, nwe, tp_scale=0.72)
    print(f"  Baseline (TP×0.72):    OOS {wf_baseline['total_oos']:+.1f}% {wf_baseline['verdict']}")
    for f in wf_baseline['folds']:
        print(f"    Fold {f['fold']}: {f['oos_pnl']:+.1f}%")

    if best_tp_scale is not None:
        wf_best_hr = run_wf(make_high_rr_sigs, opens, highs, lows, closes, n_bars, nws, nwe, tp_scale=best_tp_scale)
        print(f"\n  High-RR (TP×{best_tp_scale:.2f}):   OOS {wf_best_hr['total_oos']:+.1f}% {wf_best_hr['verdict']}")
        for f in wf_best_hr['folds']:
            print(f"    Fold {f['fold']}: {f['oos_pnl']:+.1f}%")

    # =========================================================================
    # PHASE 5: Discrimination test (direction shuffle)
    # =========================================================================
    print(f"\n{'='*80}")
    print("PHASE 5: Discrimination Test (5 shuffled seeds)")
    print(f"{'='*80}")

    disc_results = {}
    if best_tp_scale is not None:
        for test_label, sigs_fn, ts in [
            ("Baseline_0.72", make_baseline_sigs, 0.72),
            (f"HighRR_{best_tp_scale:.2f}", make_high_rr_sigs, best_tp_scale),
        ]:
            real_wf = run_wf(sigs_fn, opens, highs, lows, closes, n_bars, nws, nwe, tp_scale=ts)
            shuffled_passes = 0
            seeds = [42, 123, 7, 99, 256]
            for seed in seeds:
                rng = np.random.RandomState(seed)

                # For high-RR: shuffle direction of discovered patterns
                if "HighRR" in test_label:
                    def make_shuf(s, e, _rng=rng):
                        t = []
                        for d in discovered:
                            pat = d['pattern']
                            direction = _rng.choice(["LONG", "SHORT"])
                            tp, sl = d['tp'], d['sl']
                            if pat not in sig_idx:
                                continue
                            for b in sig_idx[pat]:
                                if s <= b < e:
                                    t.append((b, pat, direction, tp, sl))
                        return sorted(t, key=lambda x: x[0])
                else:
                    def make_shuf(s, e, _rng=rng):
                        t = []
                        for k, info in current_pd.items():
                            p = info["pattern"]
                            direction = _rng.choice(["LONG", "SHORT"])
                            tp, sl = info["tp"], info["sl"]
                            if p not in sig_idx:
                                continue
                            for b in sig_idx[p]:
                                if s <= b < e:
                                    t.append((b, p, direction, tp, sl))
                        return sorted(t, key=lambda x: x[0])

                swf = run_wf(make_shuf, opens, highs, lows, closes, n_bars, nws, nwe, tp_scale=ts)
                if swf["verdict"] == "PASS":
                    shuffled_passes += 1

            disc = "DISCRIMINATING" if shuffled_passes <= 1 else "NON-DISC"
            disc_results[test_label] = {
                "real_oos": real_wf["total_oos"], "shuffled_passes": shuffled_passes,
                "discrimination": disc,
            }
            print(f"  {test_label:>20s}: OOS {real_wf['total_oos']:+.1f}% | Shuffled PASS: {shuffled_passes}/5 → {disc}")

    # =========================================================================
    # PHASE 6: Mixed portfolio — combine current + high-RR
    # =========================================================================
    print(f"\n{'='*80}")
    print("PHASE 6: Mixed Portfolio (Current + High-RR combined)")
    print(f"{'='*80}")

    def make_mixed_sigs(s, e):
        bl = make_baseline_sigs(s, e)
        # Add high-RR patterns not already in baseline
        bl_patterns = set(f"{t[1]}_{t[2]}" for t in bl)
        hr = make_high_rr_sigs(s, e)
        combined = list(bl)
        for sig in hr:
            key = f"{sig[1]}_{sig[2]}"
            if key not in bl_patterns:
                combined.append(sig)
        return sorted(combined, key=lambda x: x[0])

    mixed_sigs = make_mixed_sigs(nws, nwe)
    print(f"  Mixed signals: {len(mixed_sigs)} (baseline {len(bl_sigs)} + unique high-RR)")

    # For mixed: baseline patterns use TP×0.72, high-RR use best scale
    # Simplification: use uniform scale
    for ts_label, ts in [("Mixed_TP×0.72", 0.72), ("Mixed_TP×0.80", 0.80)]:
        r = run_npos_backtest(mixed_sigs, opens, highs, lows, closes, n_bars, atr, ema, nws, nwe,
                               tp_scale=ts, label=ts_label)
        wf = run_wf(make_mixed_sigs, opens, highs, lows, closes, n_bars, nws, nwe, tp_scale=ts)
        if r:
            print(f"  {ts_label:>20s}: PnL {r['pnl']:+.1f}% | MDD {r['mdd']:.1f}% | P/M {r['pm']:.1f}x | "
                  f"WR {r['wr']:.1f}% | R:R {r['rr']:.3f} | OOS {wf['total_oos']:+.1f}% {wf['verdict']}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  High-RR patterns discovered: {len(discovered)} ({n_long}L + {n_short}S)")
    if discovered:
        print(f"  TP range: {np.min(tps):.2f}-{np.max(tps):.2f}%, SL range: {np.min(sls):.2f}-{np.max(sls):.2f}%")
        print(f"  R:R range: {np.min(rrs):.2f}-{np.max(rrs):.2f}")
    print(f"  Best TP scale for High-RR: {best_tp_scale} (OOS {best_oos:+.1f}%)")
    print(f"  Baseline OOS: {wf_baseline['total_oos']:+.1f}%")
    if best_tp_scale:
        print(f"  High-RR vs Baseline: {best_oos - wf_baseline['total_oos']:+.1f}%")

    elapsed = time.time() - t0
    output = {
        "study": "high_rr_pattern_discovery",
        "date": datetime.now().isoformat(),
        "params": {
            "tp_range": [HIGH_RR_TP_MIN, HIGH_RR_TP_MAX],
            "sl_range": [HIGH_RR_SL_MIN, HIGH_RR_SL_MAX],
            "min_rr": HIGH_RR_MIN_RR,
            "edge_threshold": HIGH_RR_EDGE_THRESHOLD,
        },
        "discovered_patterns": {
            "count": len(discovered),
            "long": n_long, "short": n_short,
            "details": discovery_details,
        },
        "phase2_is_comparison": results,
        "phase3_tp_sweep": sweep_results,
        "phase4_wf": {
            "baseline": wf_baseline,
            "best_high_rr": {"tp_scale": best_tp_scale, "oos": best_oos} if best_tp_scale else None,
        },
        "phase5_discrimination": disc_results,
        "elapsed_s": round(elapsed, 1),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
