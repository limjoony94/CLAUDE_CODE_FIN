#!/usr/bin/env python3
"""
OPP_SIGNAL Partial Close Study
================================
현행: 2회 반대 신호 + 수익 >0.1% → 100% 청산
연구: 2회 반대 신호 + 수익 >0.1% → 50% 부분 청산 (잔여는 원래 TP/SL 유지)

연구 항목:
  1) 100% vs 50% 부분청산 비교 (현행 TP scale 0.72 기준)
  2) 50% 부분청산 시 최적 TP scale factor sweep (0.50 ~ 1.00)
  3) WF 3-fold 검증
  4) Discrimination test (방향 셔플)

Output: results/opp_partial_close_study.json
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
    FEE_PCT, LEVERAGE, SLIPPAGE_BUFFER, MAX_DAILY_LOSS_PCT,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, DEFAULT_REGIME_MULT,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI, TIMEOUT_BARS,
    DEFAULT_PREEMPTIVE_CASCADE_PCT, DEFAULT_PREEMPTIVE_TIGHTEN_PCT,
)

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "btc_5m_270days_reclassified.csv")
PATTERNS_FILE = os.path.join(PROJECT_ROOT, "results", "dynamic_patterns.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "results", "opp_partial_close_study.json")
DECAY_RATE = 0.9975


def check_exit_base(pos, bar, opens, highs, lows, closes, n_bars, atr_ratio, fee, cl, ch, to):
    """Standard exit with decay + vol adaptation."""
    eb = pos["entry_bar"]
    if bar < eb:
        return None
    entry = opens[eb]
    if entry <= 0:
        return None
    tp_pct, sl_pct = pos["tp_pct"], pos["sl_pct"]
    d, sb = pos["direction"], pos["signal_bar"]
    entry_atr = 1.0
    if atr_ratio is not None and sb < len(atr_ratio) and not np.isnan(atr_ratio[sb]):
        entry_atr = max(cl, min(ch, atr_ratio[sb]))
    r = entry_atr
    if sl_pct > 0:
        r = min(r, MAX_DAILY_LOSS_PCT / LEVERAGE / sl_pct)
    bh = bar - eb
    if bh >= to:
        return {"entry_bar": eb, "exit_bar": bar, "pnl_slot": 0, "reason": "TIMEOUT", "drop": True}
    vol_factor = 1.0
    if atr_ratio is not None and bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
        cur_atr = max(cl, min(ch, atr_ratio[bar]))
        if entry_atr > 0:
            vol_factor = max(0.5, min(2.0, cur_atr / entry_atr))
    decay = DECAY_RATE ** bh
    eff_tp = tp_pct * decay * vol_factor * r + SLIPPAGE_BUFFER
    eso = pos.get("eff_sl_override")
    if eso is not None:
        eff_sl = max(0.1, eso * vol_factor - SLIPPAGE_BUFFER)
    else:
        eff_sl = max(0.1, sl_pct * r * vol_factor - SLIPPAGE_BUFFER)
    if d == "LONG":
        tp_p, sl_p = entry * (1 + eff_tp / 100), entry * (1 - eff_sl / 100)
    else:
        tp_p, sl_p = entry * (1 - eff_tp / 100), entry * (1 + eff_sl / 100)
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
    return {"entry_bar": eb, "exit_bar": bar, "pnl_slot": pnl, "reason": rs, "drop": False}


def portfolio_npos_partial_opp(sig_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, start, end, all_signals_by_bar,
                                tp_scale=0.72, opp_consecutive=2, opp_profit_pct=0.1,
                                opp_close_pct=1.0):
    """
    N-pos sim with partial opposite-signal exit.

    opp_close_pct: fraction to close (1.0 = full, 0.5 = half).
    When partial close: size_mult is reduced, position stays open with remaining TP/SL.
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
    sigs = sorted([(s, p, d, tp * tp_scale, sl) for s, p, d, tp, sl in sig_tuples if start <= s < end],
                  key=lambda x: x[0])
    si = 0
    opp_exit_count = 0
    opp_pnl_total = 0.0
    consec_opposite = {"LONG": 0, "SHORT": 0}

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
                        orig_sl = ps * rv
                        new_sl = orig_sl * pre_keep
                        cur = p.get('eff_sl_override')
                        if cur is None or new_sl < cur:
                            p['eff_sl_override'] = new_sl
                            p['pre_cascaded'] = True
                            p['cascaded'] = True

        # Check TP/SL exits
        closed, bpnl = [], 0.0
        bar_sl_orig_dirs = set()
        for pos in positions:
            res = check_exit_base(pos, bar, opens, highs, lows, closes, n_bars, atr_ratio, fee,
                                  DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI, TIMEOUT_BARS)
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

        # Track opposite signals
        bar_signals = all_signals_by_bar.get(bar, [])
        bar_long_sigs = sum(1 for _, _, d, _, _ in bar_signals if d == "LONG")
        bar_short_sigs = sum(1 for _, _, d, _, _ in bar_signals if d == "SHORT")

        if bar_short_sigs > 0:
            consec_opposite["LONG"] += 1
        else:
            consec_opposite["LONG"] = 0
        if bar_long_sigs > 0:
            consec_opposite["SHORT"] += 1
        else:
            consec_opposite["SHORT"] = 0

        # Opposite-signal exit (partial or full)
        for direction in ["LONG", "SHORT"]:
            if consec_opposite[direction] < opp_consecutive:
                continue

            for pos in list(positions):
                if pos["slot"] in closed or pos["direction"] != direction:
                    continue
                if pos.get("_opp_already_partial"):
                    continue  # Already partially closed — don't re-trigger
                eb_ = pos["entry_bar"]
                if eb_ >= n_bars or bar < eb_:
                    continue
                entry = opens[eb_]
                if entry <= 0:
                    continue

                if direction == "LONG":
                    unr_pct = (closes[bar] / entry - 1) * 100 * LEVERAGE
                else:
                    unr_pct = (1 - closes[bar] / entry) * 100 * LEVERAGE

                if unr_pct < opp_profit_pct:
                    continue

                sm = pos.get("size_mult", 1.0)
                pnl = unr_pct - fee

                if opp_close_pct >= 1.0:
                    # Full close
                    trades.append({
                        "entry_bar": eb_, "exit_bar": bar, "pnl_slot": pnl,
                        "reason": "OPP_SIGNAL", "pattern": pos["pattern"],
                        "direction": direction, "size_mult": sm,
                        "pnl_portfolio": pnl * (sp / 100) * sm,
                    })
                    closed.append(pos["slot"])
                    eq += pnl * (sp / 100) * sm
                else:
                    # Partial close — record the closed portion
                    close_sm = sm * opp_close_pct
                    remain_sm = sm * (1.0 - opp_close_pct)
                    trades.append({
                        "entry_bar": eb_, "exit_bar": bar, "pnl_slot": pnl,
                        "reason": "OPP_PARTIAL", "pattern": pos["pattern"],
                        "direction": direction, "size_mult": close_sm,
                        "pnl_portfolio": pnl * (sp / 100) * close_sm,
                    })
                    eq += pnl * (sp / 100) * close_sm
                    # Reduce remaining position size
                    pos["size_mult"] = remain_sm
                    pos["_opp_already_partial"] = True

                opp_exit_count += 1
                opp_pnl_total += pnl * (sp / 100) * (sm if opp_close_pct >= 1.0 else sm * opp_close_pct)
                consec_opposite[direction] = 0
                break  # One per direction per bar

        positions = [p for p in positions if p["slot"] not in closed]

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
    return trades, {"mdd_mtm": round(mdd, 2), "opp_exits": opp_exit_count, "opp_pnl": round(opp_pnl_total, 2)}


def run_variant(label, sigs, all_by_bar, opens, highs, lows, closes, n_bars, atr, ema, s, e, **kw):
    trades, ns = portfolio_npos_partial_opp(sigs, opens, highs, lows, closes, n_bars, atr, ema, s, e,
                                             all_by_bar, **kw)
    st = calc_stats_compound(trades)
    tp_c = sum(1 for t in trades if t.get("reason") == "TP")
    sl_c = sum(1 for t in trades if t.get("reason") == "SL")
    opp_c = sum(1 for t in trades if t.get("reason") in ("OPP_SIGNAL", "OPP_PARTIAL"))
    pm = st["pnl"] / ns["mdd_mtm"] if ns["mdd_mtm"] > 0 else 0
    tp_pnls = [abs(t["pnl_slot"]) for t in trades if t.get("reason") == "TP"]
    sl_pnls = [abs(t["pnl_slot"]) for t in trades if t.get("reason") == "SL"]
    rr = np.mean(tp_pnls) / np.mean(sl_pnls) if sl_pnls and tp_pnls else 0
    return {
        "label": label, "trades": st["trades"], "wr": st["wr"], "pnl": st["pnl"],
        "mdd": ns["mdd_mtm"], "pm": round(pm, 1), "tp": tp_c, "sl": sl_c,
        "opp": opp_c, "rr": round(rr, 3), "opp_pnl": ns["opp_pnl"],
    }


def run_wf(label, make_fn, all_by_bar_fn, opens, highs, lows, closes, n_bars, nws, nwe, nf=3, **kw):
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
        abb = all_by_bar_fn(os_, oe)
        if not sigs:
            folds.append({"fold": f + 1, "oos_pnl": 0})
            continue
        atr = compute_atr_ratio(highs[:oe], lows[:oe], closes[:oe])
        ema = compute_ema_slope(closes[:oe])
        trades, ns = portfolio_npos_partial_opp(sigs, opens, highs, lows, closes, oe, atr, ema, os_, oe, abb, **kw)
        st = calc_stats_compound(trades)
        folds.append({"fold": f + 1, "oos_pnl": st["pnl"]})
    total = sum(f["oos_pnl"] for f in folds)
    np_ = sum(1 for f in folds if f["oos_pnl"] > 0)
    return {"folds": folds, "total_oos": round(total, 1), "verdict": "PASS" if np_ == nf else "FAIL"}


def main():
    t0 = time.time()
    print("OPP_SIGNAL Partial Close Study")
    print("=" * 80)

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

    with open(PATTERNS_FILE) as f:
        pd_ = json.load(f)["pattern_details"]

    def make_sigs(s, e):
        t = []
        for k, info in pd_.items():
            p, d = info["pattern"], info["direction"]
            tp, sl = info["tp"], info["sl"]
            if p not in sig_idx:
                continue
            for b in sig_idx[p]:
                if s <= b < e:
                    t.append((b, p, d, tp, sl))
        return sorted(t, key=lambda x: x[0])

    def make_all_by_bar(s, e):
        by_bar = defaultdict(list)
        for k, info in pd_.items():
            p, d = info["pattern"], info["direction"]
            tp, sl = info["tp"], info["sl"]
            if p not in sig_idx:
                continue
            for b in sig_idx[p]:
                if s <= b < e:
                    by_bar[b].append((b, p, d, tp, sl))
        return dict(by_bar)

    all_sigs = make_sigs(nws, nwe)
    all_by_bar = make_all_by_bar(nws, nwe)
    print(f"  {n_bars} bars, NW [{nws},{nwe}], {len(all_sigs)} signals")

    # =========================================================================
    # PHASE 1: 100% vs 50% partial close comparison (TP scale = 0.72)
    # =========================================================================
    print(f"\n{'='*80}")
    print("PHASE 1: 100% close vs 50% partial close (tp_scale=0.72)")
    print(f"{'='*80}")

    phase1_variants = [
        ("BASELINE_no_opp", {"tp_scale": 0.72, "opp_consecutive": 999, "opp_profit_pct": 999}),
        ("100%_close",      {"tp_scale": 0.72, "opp_consecutive": 2, "opp_profit_pct": 0.1, "opp_close_pct": 1.0}),
        ("50%_close",       {"tp_scale": 0.72, "opp_consecutive": 2, "opp_profit_pct": 0.1, "opp_close_pct": 0.5}),
        ("75%_close",       {"tp_scale": 0.72, "opp_consecutive": 2, "opp_profit_pct": 0.1, "opp_close_pct": 0.75}),
        ("25%_close",       {"tp_scale": 0.72, "opp_consecutive": 2, "opp_profit_pct": 0.1, "opp_close_pct": 0.25}),
    ]

    hdr = f"  {'Variant':>18s} | {'PnL':>8s} | {'MDD':>5s} | {'P/M':>6s} | {'WR':>5s} | {'R:R':>5s} | {'Trades':>6s} | {'OPP':>4s} | {'OPP PnL':>8s} | {'OOS':>8s} | WF"
    print(hdr)
    print(f"  {'-'*18}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*4}")

    phase1_results = {}
    for name, kw in phase1_variants:
        r = run_variant(name, all_sigs, all_by_bar, opens, highs, lows, closes, n_bars, atr, ema, nws, nwe, **kw)
        wf = run_wf(name, make_sigs, make_all_by_bar, opens, highs, lows, closes, n_bars, nws, nwe, **kw)
        phase1_results[name] = {"is": r, "wf": wf, "params": kw}
        mk = " <<" if name == "BASELINE_no_opp" else ""
        print(f"  {name:>18s} | {r['pnl']:+7.1f}% | {r['mdd']:4.1f}% | {r['pm']:5.1f}x | "
              f"{r['wr']:.1f}% | {r['rr']:.3f} | {r['trades']:>6d} | {r['opp']:>4d} | {r['opp_pnl']:+7.2f}% | "
              f"{wf['total_oos']:+7.1f}% | {wf['verdict']}{mk}")

    # =========================================================================
    # PHASE 2: TP scale sweep for 50% partial close
    # =========================================================================
    print(f"\n{'='*80}")
    print("PHASE 2: TP scale sweep with 50% partial close")
    print(f"{'='*80}")

    tp_scales = [0.50, 0.55, 0.60, 0.65, 0.68, 0.72, 0.76, 0.80, 0.85, 0.90, 0.95, 1.00]

    print(f"  {'TP Scale':>10s} | {'PnL':>8s} | {'MDD':>5s} | {'P/M':>6s} | {'WR':>5s} | {'R:R':>5s} | {'Trades':>6s} | {'OPP':>4s} | {'OOS':>8s} | WF")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*4}-+-{'-'*8}-+-{'-'*4}")

    phase2_results = {}
    best_oos = -9999
    best_tp = None
    for ts in tp_scales:
        kw = {"tp_scale": ts, "opp_consecutive": 2, "opp_profit_pct": 0.1, "opp_close_pct": 0.5}
        label = f"TP_{ts:.2f}"
        r = run_variant(label, all_sigs, all_by_bar, opens, highs, lows, closes, n_bars, atr, ema, nws, nwe, **kw)
        wf = run_wf(label, make_sigs, make_all_by_bar, opens, highs, lows, closes, n_bars, nws, nwe, **kw)
        phase2_results[label] = {"is": r, "wf": wf, "params": kw}
        mk = " << BEST" if wf["total_oos"] > best_oos else ""
        if wf["total_oos"] > best_oos:
            best_oos = wf["total_oos"]
            best_tp = ts
        print(f"  {ts:>10.2f} | {r['pnl']:+7.1f}% | {r['mdd']:4.1f}% | {r['pm']:5.1f}x | "
              f"{r['wr']:.1f}% | {r['rr']:.3f} | {r['trades']:>6d} | {r['opp']:>4d} | "
              f"{wf['total_oos']:+7.1f}% | {wf['verdict']}{mk}")

    # =========================================================================
    # PHASE 3: TP scale sweep for 100% close (comparison)
    # =========================================================================
    print(f"\n{'='*80}")
    print("PHASE 3: TP scale sweep with 100% close (for comparison)")
    print(f"{'='*80}")

    print(f"  {'TP Scale':>10s} | {'PnL':>8s} | {'MDD':>5s} | {'P/M':>6s} | {'WR':>5s} | {'R:R':>5s} | {'Trades':>6s} | {'OPP':>4s} | {'OOS':>8s} | WF")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*4}-+-{'-'*8}-+-{'-'*4}")

    phase3_results = {}
    best_oos_full = -9999
    best_tp_full = None
    for ts in tp_scales:
        kw = {"tp_scale": ts, "opp_consecutive": 2, "opp_profit_pct": 0.1, "opp_close_pct": 1.0}
        label = f"FULL_TP_{ts:.2f}"
        r = run_variant(label, all_sigs, all_by_bar, opens, highs, lows, closes, n_bars, atr, ema, nws, nwe, **kw)
        wf = run_wf(label, make_sigs, make_all_by_bar, opens, highs, lows, closes, n_bars, nws, nwe, **kw)
        phase3_results[label] = {"is": r, "wf": wf, "params": kw}
        mk = " << BEST" if wf["total_oos"] > best_oos_full else ""
        if wf["total_oos"] > best_oos_full:
            best_oos_full = wf["total_oos"]
            best_tp_full = ts
        print(f"  {ts:>10.2f} | {r['pnl']:+7.1f}% | {r['mdd']:4.1f}% | {r['pm']:5.1f}x | "
              f"{r['wr']:.1f}% | {r['rr']:.3f} | {r['trades']:>6d} | {r['opp']:>4d} | "
              f"{wf['total_oos']:+7.1f}% | {wf['verdict']}{mk}")

    # =========================================================================
    # PHASE 4: Discrimination test (direction shuffle, best variants only)
    # =========================================================================
    print(f"\n{'='*80}")
    print("PHASE 4: Discrimination test (5 shuffled seeds)")
    print(f"{'='*80}")

    disc_results = {}
    for test_label, test_kw in [
        ("50%_close_0.72", {"tp_scale": 0.72, "opp_consecutive": 2, "opp_profit_pct": 0.1, "opp_close_pct": 0.5}),
        ("100%_close_0.72", {"tp_scale": 0.72, "opp_consecutive": 2, "opp_profit_pct": 0.1, "opp_close_pct": 1.0}),
        (f"50%_best_{best_tp:.2f}", {"tp_scale": best_tp, "opp_consecutive": 2, "opp_profit_pct": 0.1, "opp_close_pct": 0.5}),
    ]:
        real_wf = run_wf(test_label, make_sigs, make_all_by_bar, opens, highs, lows, closes, n_bars, nws, nwe, **test_kw)
        shuffled_passes = 0
        seeds = [42, 123, 7, 99, 256]
        for seed in seeds:
            rng = np.random.RandomState(seed)

            def make_sigs_shuf(s, e, _rng=rng):
                t = []
                for k, info in pd_.items():
                    p = info["pattern"]
                    d = _rng.choice(["LONG", "SHORT"])
                    tp, sl = info["tp"], info["sl"]
                    if p not in sig_idx:
                        continue
                    for b in sig_idx[p]:
                        if s <= b < e:
                            t.append((b, p, d, tp, sl))
                return sorted(t, key=lambda x: x[0])

            def make_abb_shuf(s, e, _rng=rng):
                by_bar = defaultdict(list)
                for k, info in pd_.items():
                    p = info["pattern"]
                    d = _rng.choice(["LONG", "SHORT"])
                    tp, sl = info["tp"], info["sl"]
                    if p not in sig_idx:
                        continue
                    for b in sig_idx[p]:
                        if s <= b < e:
                            by_bar[b].append((b, p, d, tp, sl))
                return dict(by_bar)

            swf = run_wf(f"shuf_{seed}", make_sigs_shuf, make_abb_shuf, opens, highs, lows, closes, n_bars, nws, nwe, **test_kw)
            if swf["verdict"] == "PASS":
                shuffled_passes += 1

        disc = "DISCRIMINATING" if shuffled_passes <= 1 else "NON-DISC"
        disc_results[test_label] = {
            "real_oos": real_wf["total_oos"], "real_verdict": real_wf["verdict"],
            "shuffled_passes": shuffled_passes, "discrimination": disc,
        }
        print(f"  {test_label:>22s}: OOS {real_wf['total_oos']:+.1f}% {real_wf['verdict']} | "
              f"Shuffled PASS: {shuffled_passes}/5 → {disc}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  50% partial close best TP scale: {best_tp:.2f} (OOS {best_oos:+.1f}%)")
    print(f"  100% full close best TP scale:   {best_tp_full:.2f} (OOS {best_oos_full:+.1f}%)")

    p1_base = phase1_results["BASELINE_no_opp"]["wf"]["total_oos"]
    p1_full = phase1_results["100%_close"]["wf"]["total_oos"]
    p1_half = phase1_results["50%_close"]["wf"]["total_oos"]
    print(f"\n  At tp_scale=0.72:")
    print(f"    No OPP:    OOS {p1_base:+.1f}%")
    print(f"    100% close: OOS {p1_full:+.1f}% (vs no-opp: {p1_full-p1_base:+.1f}%)")
    print(f"    50% close:  OOS {p1_half:+.1f}% (vs no-opp: {p1_half-p1_base:+.1f}%)")
    print(f"    50% vs 100%: {p1_half-p1_full:+.1f}%")

    elapsed = time.time() - t0
    output = {
        "study": "opp_partial_close",
        "date": datetime.now().isoformat(),
        "phase1_close_pct_comparison": phase1_results,
        "phase2_tp_sweep_50pct": phase2_results,
        "phase3_tp_sweep_100pct": phase3_results,
        "phase4_discrimination": disc_results,
        "summary": {
            "best_tp_50pct": best_tp,
            "best_oos_50pct": best_oos,
            "best_tp_100pct": best_tp_full,
            "best_oos_100pct": best_oos_full,
            "baseline_oos_no_opp": p1_base,
        },
        "elapsed_s": round(elapsed, 1),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
