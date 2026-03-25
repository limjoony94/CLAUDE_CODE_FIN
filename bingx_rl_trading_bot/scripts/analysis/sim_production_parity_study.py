#!/usr/bin/env python3
"""
Simulation-Production Parity Study
=====================================
라이브 SL 4배 괴리, TIMEOUT 처리, CASCADE 동작의 근본 원인 분석.

Scanner _check_exit_npos vs Production 차이:
  1. TP Decay: Scanner 없음 / Production 있음 (0.9975^bars)
  2. Vol Adaptation: Scanner 없음 / Production 있음 (cur_atr/entry_atr)
  3. TIMEOUT: Scanner DROP(pnl=0) / Production 실제 PnL 기록
  4. Emergency SL: Scanner 없음 / Production 13% daily loss limit

비교:
  A) Scanner 현행 (no decay, no vol_adapt, TIMEOUT=DROP)
  B) +TP Decay
  C) +Vol Adaptation
  D) +TIMEOUT as actual PnL (not DROP)
  E) Full production-faithful (B+C+D)
  F) Live 실적과 비교

Output: results/sim_production_parity_study.json
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
OUTPUT_FILE = "results/sim_production_parity_study.json"
DECAY_RATE = 0.9975


def check_exit_production(pos, bar, opens, highs, lows, closes, n_bars, atr_ratio, fee,
                           cl, ch, to, use_decay=True, use_vol_adapt=True, timeout_drop=True):
    """Production-faithful exit check with all mechanisms."""
    eb = pos["entry_bar"]
    if bar < eb:
        return None
    entry = opens[eb]
    if entry <= 0:
        return None
    tp_pct, sl_pct = pos["tp_pct"], pos["sl_pct"]
    d, sb = pos["direction"], pos["signal_bar"]

    # ATR scaling at entry
    entry_atr = 1.0
    if atr_ratio is not None and sb < len(atr_ratio) and not np.isnan(atr_ratio[sb]):
        entry_atr = max(cl, min(ch, atr_ratio[sb]))
    r = entry_atr
    if sl_pct > 0:
        r = min(r, MAX_DAILY_LOSS_PCT / LEVERAGE / sl_pct)

    bh = bar - eb

    # TIMEOUT
    if bh >= to:
        if timeout_drop:
            return {"entry_bar": eb, "exit_bar": bar, "pnl_slot": 0, "reason": "TIMEOUT", "drop": True}
        else:
            # Record actual PnL at timeout
            xp = closes[bar] if bar < n_bars else opens[min(bar, n_bars - 1)]
            pnl = ((xp / entry - 1) if d == "LONG" else (1 - xp / entry)) * 100 * LEVERAGE - fee
            return {"entry_bar": eb, "exit_bar": bar, "pnl_slot": pnl, "reason": "TIMEOUT", "drop": False}

    # Vol adaptation (current ATR / entry ATR)
    vol_factor = 1.0
    if use_vol_adapt and atr_ratio is not None and bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
        cur_atr = max(cl, min(ch, atr_ratio[bar]))
        if entry_atr > 0:
            vol_factor = max(0.5, min(2.0, cur_atr / entry_atr))

    # TP Decay
    decay = DECAY_RATE ** bh if use_decay else 1.0

    eff_tp = tp_pct * decay * vol_factor * r + SLIPPAGE_BUFFER

    # SL: cascade override or base
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

    h, l = highs[bar], lows[bar]
    ht = (h >= tp_p if d == "LONG" else l <= tp_p)
    hs = (l <= sl_p if d == "LONG" else h >= sl_p)

    if not ht and not hs:
        return None
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
    return {"entry_bar": eb, "exit_bar": bar, "pnl_slot": pnl, "reason": rs, "drop": False,
            "eff_sl": eff_sl, "eff_tp": eff_tp, "vol_factor": vol_factor, "decay": decay}


def portfolio_npos_production(sig_tuples, opens, highs, lows, closes, n_bars,
                               atr_ratio, ema_slope, start, end,
                               use_decay=True, use_vol_adapt=True, timeout_drop=True):
    """N-pos portfolio sim with production-faithful exit logic."""
    ns = DEFAULT_N_SLOTS
    sp = 100.0 / ns
    fee = FEE_PCT * LEVERAGE
    dc = DEFAULT_DIRECTION_CAP
    positions, trades = [], []
    eq, peak = 100.0, 100.0
    mdd = 0.0
    mp = {"LONG": -1, "SHORT": -1}
    pre_keep = 1.0 - DEFAULT_PREEMPTIVE_TIGHTEN_PCT / 100.0
    sigs = sorted([(s, p, d, tp, sl) for s, p, d, tp, sl in sig_tuples if start <= s < end],
                  key=lambda x: x[0])
    si = 0

    for bar in range(start, end):
        # Pre-emptive cascade
        if DEFAULT_PREEMPTIVE_CASCADE_PCT > 0 and len(positions) >= 2:
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
                            rv = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[sg]))
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
            res = check_exit_production(pos, bar, opens, highs, lows, closes, n_bars, atr_ratio, fee,
                                         DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI, TIMEOUT_BARS,
                                         use_decay=use_decay, use_vol_adapt=use_vol_adapt,
                                         timeout_drop=timeout_drop)
            if res:
                if res.get("drop"):
                    closed.append(pos["slot"])
                    continue
                sm = pos.get("size_mult", 1.0)
                res["pnl_portfolio"] = res["pnl_slot"] * (sp / 100) * sm
                res["pattern"] = pos["pattern"]
                res["direction"] = pos["direction"]
                res["size_mult"] = sm
                trades.append(res)
                closed.append(pos["slot"])
                bpnl += res["pnl_portfolio"]
                if res["reason"] == "SL":
                    if not pos.get('cascaded'):
                        bar_sl_orig_dirs.add(pos['direction'])

        # Reactive cascade
        if DEFAULT_CASCADE_TIGHTEN_PCT > 0 and len(bar_sl_orig_dirs) > 0:
            kr = 1.0 - DEFAULT_CASCADE_TIGHTEN_PCT / 100.0
            for sd in bar_sl_orig_dirs:
                for pos in positions:
                    if pos["slot"] in closed or pos["direction"] != sd:
                        continue
                    sg = pos["signal_bar"]
                    rv = 1.0
                    if atr_ratio is not None and sg < len(atr_ratio) and not np.isnan(atr_ratio[sg]):
                        rv = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[sg]))
                    ps = pos["sl_pct"]
                    if ps > 0:
                        rv = min(rv, MAX_DAILY_LOSS_PCT / LEVERAGE / ps)
                    pos["eff_sl_override"] = ps * rv * kr
                    pos["cascaded"] = True

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
                            pr = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[psig]))
                        if psl > 0:
                            pr = min(pr, MAX_DAILY_LOSS_PCT / LEVERAGE / psl)
                        exp += psl * pr * (1.0 / ns) * LEVERAGE * p.get("size_mult", 1.0)
                nr = 1.0
                if atr_ratio is not None and sb < len(atr_ratio) and not np.isnan(atr_ratio[sb]):
                    nr = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[sb]))
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

    # Close remaining
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
        trades.append({
            "entry_bar": peb, "exit_bar": xb, "pnl_slot": pnl, "reason": "OOS_END",
            "pattern": pos["pattern"], "direction": pos["direction"],
            "size_mult": sm, "pnl_portfolio": pnl * (sp / 100) * sm,
        })
    return trades, {"mdd_mtm": round(mdd, 2)}


def analyze(label, trades, stats):
    st = calc_stats_compound(trades)
    mdd = stats.get("mdd_mtm", st["mdd"])
    pm = st["pnl"] / mdd if mdd > 0 else 0
    reasons = Counter(t.get("reason", "?") for t in trades)
    tp_c = reasons.get("TP", 0)
    sl_c = reasons.get("SL", 0)
    to_c = reasons.get("TIMEOUT", 0)
    cas_c = reasons.get("CASCADE_SL", 0)
    tp_pnls = [abs(t["pnl_slot"]) for t in trades if t.get("reason") == "TP"]
    sl_pnls = [abs(t["pnl_slot"]) for t in trades if t.get("reason") == "SL"]
    to_pnls = [t["pnl_slot"] for t in trades if t.get("reason") == "TIMEOUT"]
    rr = np.mean(tp_pnls) / np.mean(sl_pnls) if sl_pnls and tp_pnls else 0
    avg_tp = np.mean(tp_pnls) if tp_pnls else 0
    avg_sl = np.mean(sl_pnls) if sl_pnls else 0
    avg_to = np.mean(to_pnls) if to_pnls else 0
    tpsl_wr = tp_c / (tp_c + sl_c) * 100 if (tp_c + sl_c) > 0 else 0
    return {
        "label": label, "pnl": st["pnl"], "mdd": round(mdd, 2), "pm": round(pm, 1),
        "wr": st["wr"], "trades": st["trades"], "rr": round(rr, 3),
        "tp": tp_c, "sl": sl_c, "to": to_c,
        "avg_tp": round(avg_tp, 2), "avg_sl": round(avg_sl, 2), "avg_to": round(avg_to, 2),
        "tpsl_wr": round(tpsl_wr, 1),
        "reasons": dict(reasons.most_common()),
    }


def main():
    t0 = time.time()
    print("Simulation-Production Parity Study")
    print("=" * 100)

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
            tp = pv["exc_stats"]["mfe_median"]  # MFE Median TP (v1.67)
            sl = pv["sl"]
            if p not in si:
                continue
            for b in si[p]:
                if s <= b < e:
                    t.append((b, p, d, tp, sl))
        return sorted(t, key=lambda x: x[0])

    all_sigs = make_sigs(nws, nwe)
    print(f"  {n} bars, NW [{nws},{nwe}], {len(pd_)} patterns, {len(all_sigs)} signals")

    # =========================================================================
    # Run all configurations
    # =========================================================================
    configs = [
        ("A_scanner_current",    {"use_decay": False, "use_vol_adapt": False, "timeout_drop": True}),
        ("B_+decay",             {"use_decay": True,  "use_vol_adapt": False, "timeout_drop": True}),
        ("C_+vol_adapt",         {"use_decay": False, "use_vol_adapt": True,  "timeout_drop": True}),
        ("D_+timeout_pnl",       {"use_decay": False, "use_vol_adapt": False, "timeout_drop": False}),
        ("E_decay+vol",          {"use_decay": True,  "use_vol_adapt": True,  "timeout_drop": True}),
        ("F_decay+timeout_pnl",  {"use_decay": True,  "use_vol_adapt": False, "timeout_drop": False}),
        ("G_vol+timeout_pnl",    {"use_decay": False, "use_vol_adapt": True,  "timeout_drop": False}),
        ("H_FULL_PRODUCTION",    {"use_decay": True,  "use_vol_adapt": True,  "timeout_drop": False}),
    ]

    # Also run scanner's own portfolio_npos for comparison
    print("\n  Running scanner baseline (portfolio_npos)...")
    scanner_trades, scanner_stats = portfolio_npos(all_sigs, o, h, l, c, n, atr, ema, nws, nwe)
    scanner_r = analyze("Z_scanner_npos", scanner_trades, scanner_stats)

    results = {"Z_scanner_npos": scanner_r}

    print(f"\n  Running {len(configs)} production-parity configurations...")
    for name, kw in configs:
        trades, stats = portfolio_npos_production(all_sigs, o, h, l, c, n, atr, ema, nws, nwe, **kw)
        r = analyze(name, trades, stats)
        results[name] = r

    # =========================================================================
    # Print comparison table
    # =========================================================================
    print(f"\n{'='*100}")
    print("RESULTS COMPARISON")
    print(f"{'='*100}")

    hdr = (f"  {'Config':>22s} | {'PnL':>8s} | {'MDD':>5s} | {'P/M':>6s} | {'WR':>5s} | {'TP/SL WR':>8s} | "
           f"{'R:R':>5s} | {'AvgTP':>6s} | {'AvgSL':>6s} | {'AvgTO':>6s} | {'Trades':>6s} | {'TP':>4s} | {'SL':>4s} | {'TO':>4s}")
    print(hdr)
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*8}-+-"
          f"{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}")

    for name in ["Z_scanner_npos"] + [c[0] for c in configs]:
        r = results[name]
        mk = ""
        if name == "Z_scanner_npos":
            mk = " <<SCANNER"
        elif name == "H_FULL_PRODUCTION":
            mk = " <<PROD"
        print(f"  {name:>22s} | {r['pnl']:+7.1f}% | {r['mdd']:4.1f}% | {r['pm']:5.1f}x | "
              f"{r['wr']:.1f}% | {r['tpsl_wr']:6.1f}% | {r['rr']:.3f} | "
              f"{r['avg_tp']:5.2f}% | {r['avg_sl']:5.2f}% | {r['avg_to']:+5.2f}% | "
              f"{r['trades']:>6d} | {r['tp']:>4d} | {r['sl']:>4d} | {r['to']:>4d}{mk}")

    # =========================================================================
    # Live comparison
    # =========================================================================
    print(f"\n{'='*100}")
    print("LIVE COMPARISON")
    print(f"{'='*100}")

    with open("results/pattern_5m_metrics.json") as f:
        live_trades = json.load(f).get("trade_history", [])
    live_tp = [t for t in live_trades if t["exit_reason"] == "TP"]
    live_sl = [t for t in live_trades if t["exit_reason"] == "SL"]
    live_to = [t for t in live_trades if t["exit_reason"] == "TIMEOUT"]
    live_avg_tp = np.mean([abs(t["pnl_slot"]) for t in live_tp]) if live_tp else 0
    live_avg_sl = np.mean([abs(t["pnl_slot"]) for t in live_sl]) if live_sl else 0
    live_avg_to = np.mean([t["pnl_slot"] for t in live_to]) if live_to else 0
    live_rr = live_avg_tp / live_avg_sl if live_avg_sl > 0 else 0
    live_tpsl_wr = len(live_tp) / (len(live_tp) + len(live_sl)) * 100 if (len(live_tp) + len(live_sl)) > 0 else 0

    prod = results["H_FULL_PRODUCTION"]

    print(f"\n  {'Metric':>20s} | {'Scanner':>10s} | {'Full Prod':>10s} | {'Live':>10s} | {'Prod-Live':>10s}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    metrics = [
        ("TP+SL WR", f"{scanner_r['tpsl_wr']:.1f}%", f"{prod['tpsl_wr']:.1f}%", f"{live_tpsl_wr:.1f}%", f"{prod['tpsl_wr']-live_tpsl_wr:+.1f}%"),
        ("Avg TP win", f"{scanner_r['avg_tp']:.2f}%", f"{prod['avg_tp']:.2f}%", f"{live_avg_tp:.2f}%", f"{prod['avg_tp']-live_avg_tp:+.2f}%"),
        ("Avg SL loss", f"{scanner_r['avg_sl']:.2f}%", f"{prod['avg_sl']:.2f}%", f"{live_avg_sl:.2f}%", f"{prod['avg_sl']-live_avg_sl:+.2f}%"),
        ("R:R", f"{scanner_r['rr']:.3f}", f"{prod['rr']:.3f}", f"{live_rr:.3f}", f"{prod['rr']-live_rr:+.3f}"),
        ("Avg TIMEOUT pnl", "DROP", f"{prod['avg_to']:+.2f}%", f"{live_avg_to:+.2f}%", f"{prod['avg_to']-live_avg_to:+.2f}%"),
    ]
    for label, sc, pr, lv, gap in metrics:
        print(f"  {label:>20s} | {sc:>10s} | {pr:>10s} | {lv:>10s} | {gap:>10s}")

    # =========================================================================
    # Impact decomposition
    # =========================================================================
    print(f"\n{'='*100}")
    print("IMPACT DECOMPOSITION: What each fix contributes")
    print(f"{'='*100}")

    base_pnl = results["A_scanner_current"]["pnl"]
    for name in [c[0] for c in configs]:
        r = results[name]
        delta = r["pnl"] - base_pnl
        print(f"  {name:>22s}: PnL {r['pnl']:+.1f}% (delta {delta:+.1f}%) | "
              f"AvgSL {r['avg_sl']:.2f}% | R:R {r['rr']:.3f} | TPSL_WR {r['tpsl_wr']:.1f}%")

    # =========================================================================
    # SL distribution comparison
    # =========================================================================
    print(f"\n{'='*100}")
    print("SL LOSS DISTRIBUTION: Scanner vs Full-Production vs Live")
    print(f"{'='*100}")

    for label, trade_list in [
        ("Scanner", scanner_trades),
        ("Full_Prod", []),  # will fill below
        ("Live", live_sl),
    ]:
        if label == "Full_Prod":
            _, prod_stats = portfolio_npos_production(all_sigs, o, h, l, c, n, atr, ema, nws, nwe,
                                                       use_decay=True, use_vol_adapt=True, timeout_drop=False)
            prod_trades_full, _ = portfolio_npos_production(all_sigs, o, h, l, c, n, atr, ema, nws, nwe,
                                                            use_decay=True, use_vol_adapt=True, timeout_drop=False)
            sl_list = [abs(t["pnl_slot"]) for t in prod_trades_full if t.get("reason") == "SL"]
        elif label == "Scanner":
            sl_list = [abs(t["pnl_slot"]) for t in trade_list if t.get("reason") == "SL"]
        else:
            sl_list = [abs(t["pnl_slot"]) for t in trade_list]

        if not sl_list:
            continue
        print(f"\n  {label} ({len(sl_list)} SL trades):")
        for p in [25, 50, 75, 90, 95]:
            print(f"    P{p}: {np.percentile(sl_list, p):.2f}%")
        print(f"    Mean: {np.mean(sl_list):.2f}%, Max: {np.max(sl_list):.2f}%")

    elapsed = time.time() - t0
    output = {
        "study": "sim_production_parity",
        "date": datetime.now().isoformat(),
        "results": results,
        "live_stats": {
            "avg_tp": round(live_avg_tp, 2), "avg_sl": round(live_avg_sl, 2),
            "rr": round(live_rr, 3), "tpsl_wr": round(live_tpsl_wr, 1),
        },
        "elapsed_s": round(elapsed, 1),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
