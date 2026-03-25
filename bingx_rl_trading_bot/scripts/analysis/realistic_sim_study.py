#!/usr/bin/env python3
"""
Realistic Simulation Study
=============================
Scanner Cascade의 동일 바 연쇄(75% SL) vs 1-bar 지연 현실 반영.

Configurations:
  A) Scanner 현행 (same-bar cascade, no decay, no vol_adapt, TIMEOUT=DROP)
  B) +1-bar cascade delay (cascade tightening은 다음 바부터 적용)
  C) +vol_adapt (SL 동적 확대/축소)
  D) +decay (TP 지수 감소)
  E) +TIMEOUT as PnL (DROP 대신 실제 PnL 기록)
  F) REALISTIC_FULL (B+C+D+E: 1-bar cascade + vol + decay + timeout_pnl)
  G) REALISTIC + no cascade (cascade 완전 비활성화)

각 설정별 SL 분포, cascade 비율, avg SL, R:R을 라이브와 비교.

Output: results/realistic_sim_study.json
"""
import os, sys, json, time, numpy as np
from datetime import datetime
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from scripts.scanner.pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    compute_atr_ratio, compute_ema_slope, calc_stats_compound,
    portfolio_npos,
    FEE_PCT, LEVERAGE, SLIPPAGE_BUFFER, MAX_DAILY_LOSS_PCT,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, DEFAULT_REGIME_MULT,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI, TIMEOUT_BARS,
    DEFAULT_PREEMPTIVE_CASCADE_PCT, DEFAULT_PREEMPTIVE_TIGHTEN_PCT,
)

DATA_FILE = "data/btc_5m_270days_reclassified.csv"
PATTERNS_FILE = "results/dynamic_patterns.json"
OUTPUT_FILE = "results/realistic_sim_study.json"
DECAY_RATE = 0.9975
CL, CH = DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI


def portfolio_realistic(sig_tuples, opens, highs, lows, closes, n_bars,
                         atr_ratio, ema_slope, start, end,
                         cascade_delay_bars=0, use_decay=False, use_vol_adapt=False,
                         timeout_drop=True, cascade_enabled=True):
    """
    Realistic N-pos portfolio sim.

    cascade_delay_bars: 0 = same-bar (scanner default), 1 = next-bar (realistic)
    """
    ns = DEFAULT_N_SLOTS
    sp = 100.0 / ns
    fee = FEE_PCT * LEVERAGE
    dc = DEFAULT_DIRECTION_CAP
    positions, trades = [], []
    eq, peak = 100.0, 100.0
    mdd = 0.0
    mp = {"LONG": -1, "SHORT": -1}
    pre_keep = 1.0 - DEFAULT_PREEMPTIVE_TIGHTEN_PCT / 100.0
    cascade_keep = 1.0 - DEFAULT_CASCADE_TIGHTEN_PCT / 100.0
    sigs = sorted([(s, p, d, tp, sl) for s, p, d, tp, sl in sig_tuples if start <= s < end],
                  key=lambda x: x[0])
    si = 0
    # Pending cascade: {bar_to_apply: [(direction, cascade_keep_ratio)]}
    pending_cascades = {}

    cascade_sl_count = 0
    non_cascade_sl_count = 0

    for bar in range(start, end):
        # Apply pending cascades from previous bars
        if cascade_delay_bars > 0 and bar in pending_cascades:
            for casc_dir, casc_kr in pending_cascades[bar]:
                for pos in positions:
                    if pos["direction"] != casc_dir:
                        continue
                    if pos.get("cascaded"):
                        continue
                    sg = pos["signal_bar"]
                    rv = 1.0
                    if atr_ratio is not None and sg < len(atr_ratio) and not np.isnan(atr_ratio[sg]):
                        rv = max(CL, min(CH, atr_ratio[sg]))
                    ps = pos["sl_pct"]
                    if ps > 0:
                        rv = min(rv, MAX_DAILY_LOSS_PCT / LEVERAGE / ps)
                    pos["eff_sl_override"] = ps * rv * casc_kr
                    pos["cascaded"] = True
            del pending_cascades[bar]

        # Pre-emptive cascade
        if DEFAULT_PREEMPTIVE_CASCADE_PCT > 0 and cascade_enabled and len(positions) >= 2:
            for pre_dir in ('LONG', 'SHORT'):
                dir_pos = [p for p in positions if p['direction'] == pre_dir and not p.get('pre_cascaded')]
                if len(dir_pos) < 2:
                    continue
                unreal = 0.0
                for p in dir_pos:
                    eb_ = p['entry_bar']
                    if eb_ >= n_bars or bar < eb_:
                        continue
                    ep = opens[eb_]
                    if ep <= 0:
                        continue
                    unr = ((closes[bar] / ep - 1) if pre_dir == 'LONG' else (1 - closes[bar] / ep)) * 100 * LEVERAGE
                    if unr < 0:
                        unreal += abs(unr) * (sp / 100) * p.get('size_mult', 1.0)
                if unreal > DEFAULT_PREEMPTIVE_CASCADE_PCT:
                    for p in dir_pos:
                        sg = p['signal_bar']
                        rv = 1.0
                        if atr_ratio is not None and sg < len(atr_ratio) and not np.isnan(atr_ratio[sg]):
                            rv = max(CL, min(CH, atr_ratio[sg]))
                        ps = p['sl_pct']
                        if ps > 0:
                            rv = min(rv, MAX_DAILY_LOSS_PCT / LEVERAGE / ps)
                        new_sl = ps * rv * pre_keep
                        cur = p.get('eff_sl_override')
                        if cur is None or new_sl < cur:
                            p['eff_sl_override'] = new_sl
                            p['pre_cascaded'] = True
                            p['cascaded'] = True

        # Check exits
        closed, bpnl = [], 0.0
        bar_sl_orig_dirs = set()
        for pos in positions:
            eb = pos["entry_bar"]
            if bar < eb:
                continue
            entry = opens[eb]
            if entry <= 0:
                continue
            tp_pct, sl_pct = pos["tp_pct"], pos["sl_pct"]
            d, sb = pos["direction"], pos["signal_bar"]

            entry_atr = 1.0
            if atr_ratio is not None and sb < len(atr_ratio) and not np.isnan(atr_ratio[sb]):
                entry_atr = max(CL, min(CH, atr_ratio[sb]))
            r = entry_atr
            if sl_pct > 0:
                r = min(r, MAX_DAILY_LOSS_PCT / LEVERAGE / sl_pct)

            bh = bar - eb
            if bh >= TIMEOUT_BARS:
                if timeout_drop:
                    closed.append(pos["slot"])
                    continue
                else:
                    xp = closes[bar] if bar < n_bars else opens[min(bar, n_bars - 1)]
                    pnl = ((xp / entry - 1) if d == "LONG" else (1 - xp / entry)) * 100 * LEVERAGE - fee
                    sm = pos.get("size_mult", 1.0)
                    trades.append({"entry_bar": eb, "exit_bar": bar, "pnl_slot": pnl,
                                   "reason": "TIMEOUT", "pattern": pos["pattern"],
                                   "direction": d, "size_mult": sm,
                                   "pnl_portfolio": pnl * (sp / 100) * sm})
                    closed.append(pos["slot"])
                    bpnl += pnl * (sp / 100) * sm
                    continue

            # Vol adapt
            vol_factor = 1.0
            if use_vol_adapt and atr_ratio is not None and bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
                cur_atr = max(CL, min(CH, atr_ratio[bar]))
                if entry_atr > 0:
                    vol_factor = max(0.5, min(2.0, cur_atr / entry_atr))

            # Decay
            decay = DECAY_RATE ** bh if use_decay else 1.0

            eff_tp = tp_pct * decay * vol_factor * r + SLIPPAGE_BUFFER
            eso = pos.get("eff_sl_override")
            if eso is not None:
                eff_sl = max(0.1, eso * vol_factor - SLIPPAGE_BUFFER)
            else:
                eff_sl = max(0.1, sl_pct * r * vol_factor - SLIPPAGE_BUFFER)

            if d == "LONG":
                tp_p = entry * (1 + eff_tp / 100)
                sl_p = entry * (1 - eff_sl / 100)
            else:
                tp_p = entry * (1 - eff_tp / 100)
                sl_p = entry * (1 + eff_sl / 100)

            hv, lv = highs[bar], lows[bar]
            ht = (hv >= tp_p if d == "LONG" else lv <= tp_p)
            hs = (lv <= sl_p if d == "LONG" else hv >= sl_p)

            if not ht and not hs:
                continue

            if ht and hs:
                if abs(tp_p - opens[bar]) <= abs(sl_p - opens[bar]):
                    xp, rs = tp_p, "TP"
                else:
                    xp, rs = sl_p, "SL"
            elif ht:
                xp, rs = tp_p, "TP"
            else:
                xp, rs = sl_p, "SL"

            pnl = ((xp / entry - 1) if d == "LONG" else (1 - xp / entry)) * 100 * LEVERAGE - fee
            sm = pos.get("size_mult", 1.0)
            trades.append({"entry_bar": eb, "exit_bar": bar, "pnl_slot": pnl,
                           "reason": rs, "pattern": pos["pattern"], "direction": d,
                           "size_mult": sm, "pnl_portfolio": pnl * (sp / 100) * sm,
                           "eff_sl": eff_sl, "cascaded": pos.get("cascaded", False)})
            closed.append(pos["slot"])
            bpnl += pnl * (sp / 100) * sm
            if rs == "SL":
                if pos.get("cascaded"):
                    cascade_sl_count += 1
                else:
                    non_cascade_sl_count += 1
                    bar_sl_orig_dirs.add(d)

        # Reactive cascade
        if cascade_enabled and DEFAULT_CASCADE_TIGHTEN_PCT > 0 and len(bar_sl_orig_dirs) > 0:
            if cascade_delay_bars == 0:
                # Same-bar: immediate (scanner default)
                for sd in bar_sl_orig_dirs:
                    for pos in positions:
                        if pos["slot"] in closed or pos["direction"] != sd:
                            continue
                        sg = pos["signal_bar"]
                        rv = 1.0
                        if atr_ratio is not None and sg < len(atr_ratio) and not np.isnan(atr_ratio[sg]):
                            rv = max(CL, min(CH, atr_ratio[sg]))
                        ps = pos["sl_pct"]
                        if ps > 0:
                            rv = min(rv, MAX_DAILY_LOSS_PCT / LEVERAGE / ps)
                        pos["eff_sl_override"] = ps * rv * cascade_keep
                        pos["cascaded"] = True
            else:
                # Delayed cascade: schedule for future bar
                apply_bar = bar + cascade_delay_bars
                if apply_bar not in pending_cascades:
                    pending_cascades[apply_bar] = []
                for sd in bar_sl_orig_dirs:
                    pending_cascades[apply_bar].append((sd, cascade_keep))

        positions = [p for p in positions if p["slot"] not in closed]
        eq += bpnl
        if eq > peak:
            peak = eq

        # Momentum
        if DEFAULT_MOMENTUM_LOOKBACK > 0 and bar >= DEFAULT_MOMENTUM_LOOKBACK:
            pc = (closes[bar] / closes[bar - DEFAULT_MOMENTUM_LOOKBACK] - 1) * 100 \
                if closes[bar - DEFAULT_MOMENTUM_LOOKBACK] > 0 else 0
            if pc > DEFAULT_MOMENTUM_THRESHOLD:
                mp["SHORT"] = bar + DEFAULT_MOMENTUM_COOLDOWN
            elif pc < -DEFAULT_MOMENTUM_THRESHOLD:
                mp["LONG"] = bar + DEFAULT_MOMENTUM_COOLDOWN

        # Entries
        while si < len(sigs) and sigs[si][0] == bar:
            sb, pat, d, tp, sl = sigs[si]
            si += 1
            if len(positions) >= ns:
                continue
            if sum(1 for p in positions if p["direction"] == d) >= dc:
                continue
            if any(p["pattern"] == pat for p in positions):
                continue
            eb_ = sb + 1
            if eb_ >= n_bars:
                continue
            if bar < mp.get(d, -1):
                continue
            sm = 1.0
            if DEFAULT_REGIME_MULT is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if (s > 0 and d == "SHORT") or (s <= 0 and d == "LONG"):
                    sm = DEFAULT_REGIME_MULT
            if DEFAULT_AGG_RISK_COUNTER > 0 or DEFAULT_AGG_RISK_WITH > 0:
                up = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                ctr = (up and d == "SHORT") or (not up and d == "LONG")
                cap = DEFAULT_AGG_RISK_COUNTER if ctr else DEFAULT_AGG_RISK_WITH
                exp = 0.0
                for p in positions:
                    if p["direction"] == d:
                        psl, psig = p["sl_pct"], p["signal_bar"]
                        pr = 1.0
                        if atr_ratio is not None and psig < len(atr_ratio) and not np.isnan(atr_ratio[psig]):
                            pr = max(CL, min(CH, atr_ratio[psig]))
                        if psl > 0:
                            pr = min(pr, MAX_DAILY_LOSS_PCT / LEVERAGE / psl)
                        exp += psl * pr * (1.0 / ns) * LEVERAGE * p.get("size_mult", 1.0)
                nr = 1.0
                if atr_ratio is not None and sb < len(atr_ratio) and not np.isnan(atr_ratio[sb]):
                    nr = max(CL, min(CH, atr_ratio[sb]))
                if sl > 0:
                    nr = min(nr, MAX_DAILY_LOSS_PCT / LEVERAGE / sl)
                if exp + sl * nr * (1.0 / ns) * LEVERAGE * sm > cap:
                    continue
            positions.append({
                "slot": f"{pat}_{sb}", "signal_bar": sb, "entry_bar": eb_,
                "direction": d, "pattern": pat, "tp_pct": tp, "sl_pct": sl, "size_mult": sm,
            })

        # MTM MDD
        if positions and bar < n_bars:
            mtm = eq
            for pos in positions:
                peb = pos["entry_bar"]
                if peb >= n_bars or bar < peb:
                    continue
                ep = opens[peb]
                if ep <= 0:
                    continue
                unr = ((closes[bar] / ep - 1) if pos["direction"] == "LONG" else (1 - closes[bar] / ep)) * 100 * LEVERAGE
                mtm += unr * (sp / 100) * pos.get("size_mult", 1.0)
            if mtm > peak:
                peak = mtm
            dd = (peak - mtm) / peak * 100 if peak > 0 else 0
            if dd > mdd:
                mdd = dd
        elif not positions:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            if dd > mdd:
                mdd = dd

    for pos in positions:
        peb = pos["entry_bar"]
        if peb >= n_bars:
            continue
        entry = opens[peb]
        if entry <= 0:
            continue
        xb = min(end - 1, n_bars - 1)
        xp = opens[xb]
        pnl = ((xp / entry - 1) if pos["direction"] == "LONG" else (1 - xp / entry)) * 100 * LEVERAGE - fee
        sm = pos.get("size_mult", 1.0)
        trades.append({"entry_bar": peb, "exit_bar": xb, "pnl_slot": pnl, "reason": "OOS_END",
                       "pattern": pos["pattern"], "direction": pos["direction"],
                       "size_mult": sm, "pnl_portfolio": pnl * (sp / 100) * sm})

    return trades, {"mdd_mtm": round(mdd, 2), "cascade_sl": cascade_sl_count, "non_cascade_sl": non_cascade_sl_count}


def analyze(trades, stats):
    st = calc_stats_compound(trades)
    mdd = stats.get("mdd_mtm", st["mdd"])
    pm = st["pnl"] / mdd if mdd > 0 else 0
    rc = Counter(t.get("reason") for t in trades)
    tp_c, sl_c, to_c = rc.get("TP", 0), rc.get("SL", 0), rc.get("TIMEOUT", 0)
    tp_pnls = [abs(t["pnl_slot"]) for t in trades if t.get("reason") == "TP"]
    sl_pnls = [abs(t["pnl_slot"]) for t in trades if t.get("reason") == "SL"]
    rr = np.mean(tp_pnls) / np.mean(sl_pnls) if sl_pnls and tp_pnls else 0
    tpsl_wr = tp_c / (tp_c + sl_c) * 100 if (tp_c + sl_c) > 0 else 0

    # SL breakdown
    casc_sl = [t for t in trades if t.get("reason") == "SL" and t.get("cascaded")]
    orig_sl = [t for t in trades if t.get("reason") == "SL" and not t.get("cascaded")]

    return {
        "pnl": st["pnl"], "mdd": round(mdd, 2), "pm": round(pm, 1),
        "wr": st["wr"], "trades": st["trades"], "rr": round(rr, 3),
        "tpsl_wr": round(tpsl_wr, 1),
        "tp": tp_c, "sl": sl_c, "to": to_c,
        "avg_sl": round(np.mean(sl_pnls), 2) if sl_pnls else 0,
        "avg_tp": round(np.mean(tp_pnls), 2) if tp_pnls else 0,
        "casc_sl": len(casc_sl), "orig_sl": len(orig_sl),
        "casc_sl_avg": round(np.mean([abs(t["pnl_slot"]) for t in casc_sl]), 2) if casc_sl else 0,
        "orig_sl_avg": round(np.mean([abs(t["pnl_slot"]) for t in orig_sl]), 2) if orig_sl else 0,
        "sl_p50": round(np.median(sl_pnls), 2) if sl_pnls else 0,
        "sl_p90": round(np.percentile(sl_pnls, 90), 2) if sl_pnls else 0,
    }


def run_wf(make_fn, o, h, l, c, n, nws, nwe, nf=3, **kw):
    seg = (nwe - nws) // (nf + 1)
    folds = []
    for f in range(nf):
        ie = nws + (f + 1) * seg
        os_ = ie
        oe = nws + (f + 2) * seg if f < nf - 1 else nwe
        if os_ >= oe:
            folds.append(0)
            continue
        sigs = make_fn(os_, oe)
        if not sigs:
            folds.append(0)
            continue
        a = compute_atr_ratio(h[:oe], l[:oe], c[:oe])
        e = compute_ema_slope(c[:oe])
        tr, st = portfolio_realistic(sigs, o, h, l, c, oe, a, e, os_, oe, **kw)
        s = calc_stats_compound(tr)
        folds.append(round(s["pnl"], 1))
    total = sum(folds)
    np_ = sum(1 for f in folds if f > 0)
    return {"total": round(total, 1), "verdict": "PASS" if np_ == nf else "FAIL", "folds": folds}


def main():
    t0 = time.time()
    print("Realistic Simulation Study")
    print("=" * 110)

    df = load_and_classify(DATA_FILE)
    n = len(df)
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    tc = df["candle_type"].tolist()
    nws, nwe = find_neutral_window(c, tol_pct=1.0)
    atr = compute_atr_ratio(h, l, c)
    ema = compute_ema_slope(c)
    si = build_signal_index(tc, n)

    with open(PATTERNS_FILE) as f:
        pd_ = json.load(f)["pattern_details"]

    def make_sigs(s, e):
        t = []
        for pk, pv in pd_.items():
            p, d = pv["pattern"], pv["direction"]
            tp = pv["exc_stats"]["mfe_median"]
            sl = pv["sl"]
            if p not in si:
                continue
            for b in si[p]:
                if s <= b < e:
                    t.append((b, p, d, tp, sl))
        return sorted(t, key=lambda x: x[0])

    all_sigs = make_sigs(nws, nwe)
    print(f"  {n} bars, NW [{nws},{nwe}], {len(all_sigs)} signals\n")

    configs = [
        ("A_scanner_default",   {"cascade_delay_bars": 0, "use_decay": False, "use_vol_adapt": False, "timeout_drop": True}),
        ("B_cascade_1bar",      {"cascade_delay_bars": 1, "use_decay": False, "use_vol_adapt": False, "timeout_drop": True}),
        ("C_cascade_3bar",      {"cascade_delay_bars": 3, "use_decay": False, "use_vol_adapt": False, "timeout_drop": True}),
        ("D_+vol_adapt",        {"cascade_delay_bars": 1, "use_decay": False, "use_vol_adapt": True,  "timeout_drop": True}),
        ("E_+decay",            {"cascade_delay_bars": 1, "use_decay": True,  "use_vol_adapt": False, "timeout_drop": True}),
        ("F_+timeout_pnl",      {"cascade_delay_bars": 1, "use_decay": False, "use_vol_adapt": False, "timeout_drop": False}),
        ("G_REALISTIC_FULL",    {"cascade_delay_bars": 1, "use_decay": True,  "use_vol_adapt": True,  "timeout_drop": False}),
        ("H_realistic_3bar",    {"cascade_delay_bars": 3, "use_decay": True,  "use_vol_adapt": True,  "timeout_drop": False}),
        ("I_no_cascade",        {"cascade_delay_bars": 0, "use_decay": True,  "use_vol_adapt": True,  "timeout_drop": False, "cascade_enabled": False}),
    ]

    # IS results
    hdr = (f"  {'Config':>22s} | {'PnL':>8s} | {'MDD':>5s} | {'P/M':>6s} | {'WR':>5s} | {'TPSL%':>5s} | "
           f"{'R:R':>5s} | {'AvgTP':>6s} | {'AvgSL':>6s} | {'SL_p50':>6s} | {'SL_p90':>6s} | "
           f"{'CascSL':>6s} | {'OrigSL':>6s} | {'TO':>4s} | {'Tr':>5s}")
    print(hdr)
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-"
          f"{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*4}-+-{'-'*5}")

    results = {}
    for name, kw in configs:
        tr, st = portfolio_realistic(all_sigs, o, h, l, c, n, atr, ema, nws, nwe, **kw)
        r = analyze(tr, st)
        results[name] = {"is": r, "kw": kw}
        mk = ""
        if name == "A_scanner_default":
            mk = " <<SCAN"
        elif name == "G_REALISTIC_FULL":
            mk = " <<REAL"
        print(f"  {name:>22s} | {r['pnl']:+7.1f}% | {r['mdd']:4.1f}% | {r['pm']:5.1f}x | "
              f"{r['wr']:.1f}% | {r['tpsl_wr']:4.1f}% | {r['rr']:.3f} | "
              f"{r['avg_tp']:5.2f}% | {r['avg_sl']:5.2f}% | {r['sl_p50']:5.2f}% | {r['sl_p90']:5.2f}% | "
              f"{r['casc_sl']:>5d} | {r['orig_sl']:>5d} | {r['to']:>4d} | {r['trades']:>5d}{mk}")

    # WF for key configs
    print(f"\n  Walk-Forward (3-fold):")
    for name in ["A_scanner_default", "B_cascade_1bar", "G_REALISTIC_FULL", "H_realistic_3bar", "I_no_cascade"]:
        kw = results[name]["kw"]
        wf = run_wf(make_sigs, o, h, l, c, n, nws, nwe, nf=3, **kw)
        results[name]["wf3"] = wf
        fstr = " | ".join(f"F{i+1}:{f:+.0f}%" for i, f in enumerate(wf["folds"]))
        print(f"    {name:>22s}: {fstr} | Total: {wf['total']:+.1f}% {wf['verdict']}")

    # Live comparison
    print(f"\n{'='*110}")
    print("LIVE COMPARISON")
    print("=" * 110)

    with open("results/pattern_5m_metrics.json") as f:
        live_all = json.load(f).get("trade_history", [])
    live_tp = [t for t in live_all if t["exit_reason"] == "TP"]
    live_sl = [t for t in live_all if t["exit_reason"] == "SL"]
    live_cas = [t for t in live_all if t["exit_reason"] == "CASCADE_SL"]
    live_to = [t for t in live_all if t["exit_reason"] == "TIMEOUT"]

    la_tp = np.mean([abs(t["pnl_slot"]) for t in live_tp]) if live_tp else 0
    la_sl = np.mean([abs(t["pnl_slot"]) for t in live_sl]) if live_sl else 0
    la_cas = np.mean([abs(t["pnl_slot"]) for t in live_cas]) if live_cas else 0
    la_to = np.mean([t["pnl_slot"] for t in live_to]) if live_to else 0
    live_tpsl_wr = len(live_tp) / (len(live_tp) + len(live_sl) + len(live_cas)) * 100 \
        if (len(live_tp) + len(live_sl) + len(live_cas)) > 0 else 0
    live_all_sl = live_sl + live_cas
    la_all_sl = np.mean([abs(t["pnl_slot"]) for t in live_all_sl]) if live_all_sl else 0

    real = results["G_REALISTIC_FULL"]["is"]

    print(f"\n  {'Metric':>25s} | {'Scanner':>10s} | {'Realistic':>10s} | {'Live':>10s} | {'R-L gap':>10s}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    scan = results["A_scanner_default"]["is"]
    rows = [
        ("TP+SL WR", f"{scan['tpsl_wr']:.1f}%", f"{real['tpsl_wr']:.1f}%",
         f"{live_tpsl_wr:.1f}%", f"{real['tpsl_wr']-live_tpsl_wr:+.1f}%"),
        ("Avg TP win", f"{scan['avg_tp']:.2f}%", f"{real['avg_tp']:.2f}%",
         f"{la_tp:.2f}%", f"{real['avg_tp']-la_tp:+.2f}%"),
        ("Avg SL (all)", f"{scan['avg_sl']:.2f}%", f"{real['avg_sl']:.2f}%",
         f"{la_all_sl:.2f}%", f"{real['avg_sl']-la_all_sl:+.2f}%"),
        ("Avg orig SL", f"{scan['orig_sl_avg']:.2f}%", f"{real['orig_sl_avg']:.2f}%",
         f"{la_sl:.2f}%", f"{real['orig_sl_avg']-la_sl:+.2f}%"),
        ("Avg cascade SL", f"{scan['casc_sl_avg']:.2f}%", f"{real['casc_sl_avg']:.2f}%",
         f"{la_cas:.2f}%", f"{real['casc_sl_avg']-la_cas:+.2f}%"),
        ("Cascade SL ratio", f"{scan['casc_sl']/(scan['casc_sl']+scan['orig_sl'])*100:.0f}%" if (scan['casc_sl']+scan['orig_sl'])>0 else "0%",
         f"{real['casc_sl']/(real['casc_sl']+real['orig_sl'])*100:.0f}%" if (real['casc_sl']+real['orig_sl'])>0 else "0%",
         f"{len(live_cas)/(len(live_sl)+len(live_cas))*100:.0f}%" if (len(live_sl)+len(live_cas))>0 else "0%",
         ""),
        ("SL p50", f"{scan['sl_p50']:.2f}%", f"{real['sl_p50']:.2f}%",
         f"{np.median([abs(t['pnl_slot']) for t in live_all_sl]):.2f}%" if live_all_sl else "0",
         ""),
        ("R:R", f"{scan['rr']:.3f}", f"{real['rr']:.3f}",
         f"{la_tp/la_all_sl:.3f}" if la_all_sl > 0 else "0", ""),
        ("TIMEOUT avg", "DROP", f"{real.get('avg_to', 0):.2f}%" if 'avg_to' in real else "N/A",
         f"{la_to:+.2f}%", ""),
    ]
    for label, sc, re, lv, gap in rows:
        print(f"  {label:>25s} | {sc:>10s} | {re:>10s} | {lv:>10s} | {gap:>10s}")

    # Summary
    print(f"\n{'='*110}")
    print("SUMMARY")
    print("=" * 110)
    scan_pnl = results["A_scanner_default"]["is"]["pnl"]
    real_pnl = results["G_REALISTIC_FULL"]["is"]["pnl"]
    print(f"  Scanner IS PnL:    {scan_pnl:+.1f}% (overstated)")
    print(f"  Realistic IS PnL:  {real_pnl:+.1f}% (1-bar cascade delay + decay + vol_adapt + timeout_pnl)")
    print(f"  Gap: {scan_pnl - real_pnl:+.1f}% ({(scan_pnl-real_pnl)/scan_pnl*100:.0f}% overstatement)")

    scan_casc = results["A_scanner_default"]["is"]
    real_casc = results["G_REALISTIC_FULL"]["is"]
    print(f"\n  Cascade SL ratio: Scanner {scan_casc['casc_sl']}/{scan_casc['casc_sl']+scan_casc['orig_sl']} "
          f"({scan_casc['casc_sl']/(scan_casc['casc_sl']+scan_casc['orig_sl'])*100:.0f}%) → "
          f"Realistic {real_casc['casc_sl']}/{real_casc['casc_sl']+real_casc['orig_sl']} "
          f"({real_casc['casc_sl']/(real_casc['casc_sl']+real_casc['orig_sl'])*100:.0f}%)")

    elapsed = time.time() - t0
    output = {
        "study": "realistic_sim", "date": datetime.now().isoformat(),
        "results": {k: v["is"] for k, v in results.items()},
        "wf": {k: v.get("wf3") for k, v in results.items() if "wf3" in v},
        "elapsed_s": round(elapsed, 1),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}\nTime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
