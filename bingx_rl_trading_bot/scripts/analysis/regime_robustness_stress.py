#!/usr/bin/env python3
"""
Regime Robustness Stress Test — Is the system regime-dependent?

Method: Split 303d data into 30d rolling windows (15d step).
Classify each: BULL(>5%), BEAR(<-5%), SIDEWAYS.
Run 4 variants per window: REFERENCE, RANDOM, INVERTED, NO_MECHANISM.
Compare per-regime performance.

Output: results/regime_robustness_stress.json
"""
import os, sys, json, time, numpy as np, pandas as pd
from datetime import datetime
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)
from scripts.scanner.pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    compute_atr_ratio, compute_ema_slope, calc_stats_compound,
    FEE_PCT, LEVERAGE, MAX_BARS, SLIPPAGE_BUFFER, MAX_DAILY_LOSS_PCT,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, DEFAULT_REGIME_MULT,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI, TIMEOUT_BARS,
)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "btc_5m_270days_reclassified.csv")
PATTERNS_FILE = os.path.join(PROJECT_ROOT, "results", "dynamic_patterns.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "results", "regime_robustness_stress.json")
TSF = 0.72; DR = 0.9975
WINDOW_BARS = 30 * 288; STEP_BARS = 15 * 288
BULL_THR = 5.0; BEAR_THR = -5.0

def check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee, cl, ch, to):
    eb = pos["entry_bar"]
    if bar < eb: return None
    entry = opens[eb]
    if entry <= 0: return None
    tp, sl, d, sb = pos["tp_pct"], pos["sl_pct"], pos["direction"], pos["signal_bar"]
    r = 1.0
    if atr_ratio is not None and sb < len(atr_ratio) and not np.isnan(atr_ratio[sb]):
        r = max(cl, min(ch, atr_ratio[sb]))
    if sl > 0: r = min(r, MAX_DAILY_LOSS_PCT / LEVERAGE / sl)
    bh = bar - eb
    if bh >= to: return {"entry_bar": eb, "exit_bar": bar, "pnl_slot": 0, "reason": "TIMEOUT", "drop": True}
    dtp = tp * (DR ** bh); etp = dtp * r + SLIPPAGE_BUFFER
    eso = pos.get("eff_sl_override")
    esl = max(0.1, eso - SLIPPAGE_BUFFER) if eso is not None else max(0.1, sl*r - SLIPPAGE_BUFFER)
    if d == "LONG": tp_p, sl_p = entry*(1+etp/100), entry*(1-esl/100)
    else: tp_p, sl_p = entry*(1-etp/100), entry*(1+esl/100)
    h, l = highs[bar], lows[bar]
    ht = (h >= tp_p if d == "LONG" else l <= tp_p)
    hs = (l <= sl_p if d == "LONG" else h >= sl_p)
    if not ht and not hs: return None
    if ht and hs:
        if abs(tp_p - opens[bar]) <= abs(sl_p - opens[bar]): xp, rs = tp_p, "TP"
        else: xp, rs = sl_p, "SL"
    elif ht: xp, rs = tp_p, "TP"
    else: xp, rs = sl_p, "SL"
    pnl = ((xp/entry-1) if d=="LONG" else (1-xp/entry)) * 100 * LEVERAGE - fee
    return {"entry_bar": eb, "exit_bar": bar, "pnl_slot": pnl, "reason": rs, "drop": False}

def portfolio_npos(sig_tuples, opens, highs, lows, closes, n_bars, atr_ratio, ema_slope, start, end,
                   use_cascade=True, use_momentum=True, use_aggrisk=True):
    ns = DEFAULT_N_SLOTS; sp = 100.0/ns; fee = FEE_PCT*LEVERAGE
    positions, trades = [], []; eq, peak = 100.0, 100.0; mdd = 0.0
    mp = {"LONG": -1, "SHORT": -1}
    sigs = sorted([(s,p,d,tp,sl) for s,p,d,tp,sl in sig_tuples if start<=s<end], key=lambda x:x[0])
    si = 0
    for bar in range(start, end):
        closed, bpnl, bsl = [], 0.0, 0
        for pos in positions:
            res = check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee,
                             DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI, TIMEOUT_BARS)
            if res:
                if res.get("drop"): closed.append(pos["slot"]); continue
                sm = pos.get("size_mult",1.0)
                res["pnl_portfolio"] = res["pnl_slot"]*(sp/100)*sm
                res["pattern"],res["direction"],res["size_mult"] = pos["pattern"],pos["direction"],sm
                trades.append(res); closed.append(pos["slot"])
                bpnl += res["pnl_portfolio"]
                if res["reason"]=="SL": bsl+=1
        if use_cascade and DEFAULT_CASCADE_TIGHTEN_PCT>0 and bsl>0:
            kr = 1.0-DEFAULT_CASCADE_TIGHTEN_PCT/100.0
            sd = {t["direction"] for t in trades[len(trades)-len(closed):] if t.get("reason")=="SL"}
            for d in sd:
                for pos in positions:
                    if pos["slot"] in closed or pos["direction"]!=d: continue
                    sg = pos["signal_bar"]; rv=1.0
                    if atr_ratio is not None and sg<len(atr_ratio) and not np.isnan(atr_ratio[sg]):
                        rv = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, atr_ratio[sg]))
                    ps = pos["sl_pct"]
                    if ps>0: rv=min(rv, MAX_DAILY_LOSS_PCT/LEVERAGE/ps)
                    cur = pos.get("eff_sl_override") or (ps*rv)
                    pos["eff_sl_override"] = cur*kr
        positions = [p for p in positions if p["slot"] not in closed]
        eq += bpnl
        if eq > peak: peak = eq
        if use_momentum and DEFAULT_MOMENTUM_LOOKBACK>0 and bar>=DEFAULT_MOMENTUM_LOOKBACK:
            pc = (closes[bar]/closes[bar-DEFAULT_MOMENTUM_LOOKBACK]-1)*100 if closes[bar-DEFAULT_MOMENTUM_LOOKBACK]>0 else 0
            if pc>DEFAULT_MOMENTUM_THRESHOLD: mp["SHORT"]=bar+DEFAULT_MOMENTUM_COOLDOWN
            elif pc<-DEFAULT_MOMENTUM_THRESHOLD: mp["LONG"]=bar+DEFAULT_MOMENTUM_COOLDOWN
        while si<len(sigs) and sigs[si][0]==bar:
            sb,pat,d,tp,sl = sigs[si]; si+=1
            if len(positions)>=ns: continue
            if sum(1 for p in positions if p["direction"]==d)>=DEFAULT_DIRECTION_CAP: continue
            if any(p["pattern"]==pat for p in positions): continue
            eb=sb+1
            if eb>=n_bars: continue
            if use_momentum and bar<mp.get(d,-1): continue
            sm=1.0
            if DEFAULT_REGIME_MULT is not None and bar<len(ema_slope):
                s=ema_slope[bar]
                if (s>0 and d=="SHORT") or (s<=0 and d=="LONG"): sm=DEFAULT_REGIME_MULT
            if use_aggrisk and (DEFAULT_AGG_RISK_COUNTER>0 or DEFAULT_AGG_RISK_WITH>0):
                up = ema_slope[bar]>0 if bar<len(ema_slope) else False
                ctr = (up and d=="SHORT") or (not up and d=="LONG")
                cap = DEFAULT_AGG_RISK_COUNTER if ctr else DEFAULT_AGG_RISK_WITH
                exp=0.0
                for p in positions:
                    if p["direction"]==d:
                        psl,psig=p["sl_pct"],p["signal_bar"]; pr=1.0
                        if atr_ratio is not None and psig<len(atr_ratio) and not np.isnan(atr_ratio[psig]):
                            pr=max(DEFAULT_ATR_CLAMP_LO,min(DEFAULT_ATR_CLAMP_HI,atr_ratio[psig]))
                        if psl>0: pr=min(pr,MAX_DAILY_LOSS_PCT/LEVERAGE/psl)
                        exp+=psl*pr*(1.0/ns)*LEVERAGE*p.get("size_mult",1.0)
                nr=1.0
                if atr_ratio is not None and sb<len(atr_ratio) and not np.isnan(atr_ratio[sb]):
                    nr=max(DEFAULT_ATR_CLAMP_LO,min(DEFAULT_ATR_CLAMP_HI,atr_ratio[sb]))
                if sl>0: nr=min(nr,MAX_DAILY_LOSS_PCT/LEVERAGE/sl)
                if exp+sl*nr*(1.0/ns)*LEVERAGE*sm>cap: continue
            positions.append({"slot":f"{pat}_{sb}","signal_bar":sb,"entry_bar":eb,
                              "direction":d,"pattern":pat,"tp_pct":tp,"sl_pct":sl,"size_mult":sm})
        # MTM MDD
        if positions and bar<n_bars:
            mtm=eq
            for pos in positions:
                peb=pos["entry_bar"]
                if peb>=n_bars or bar<peb: continue
                ep=opens[peb]
                if ep<=0: continue
                unr=((closes[bar]/ep-1) if pos["direction"]=="LONG" else (1-closes[bar]/ep))*100*LEVERAGE
                mtm+=unr*(sp/100)*pos.get("size_mult",1.0)
            if mtm>peak: peak=mtm
            dd=(peak-mtm)/peak*100 if peak>0 else 0
            if dd>mdd: mdd=dd
        elif not positions:
            if eq>peak: peak=eq
            dd=(peak-eq)/peak*100 if peak>0 else 0
            if dd>mdd: mdd=dd
    # Close remaining
    for pos in positions:
        peb=pos["entry_bar"]
        if peb>=n_bars: continue
        entry=opens[peb]
        if entry<=0: continue
        xb=min(end-1,n_bars-1); xp=opens[xb]
        pnl=((xp/entry-1) if pos["direction"]=="LONG" else (1-xp/entry))*100*LEVERAGE-fee
        sm=pos.get("size_mult",1.0)
        trades.append({"entry_bar":peb,"exit_bar":xb,"pnl_slot":pnl,"reason":"OOS_END",
                       "pattern":pos["pattern"],"direction":pos["direction"],
                       "size_mult":sm,"pnl_portfolio":pnl*(sp/100)*sm})
    return trades, {"mdd_mtm":round(mdd,2)}

def run_window(sigs, opens, highs, lows, closes, n_bars, atr, ema, s, e, **kw):
    if not sigs: return {"trades":0,"wr":0,"pnl":0}
    trades, ns = portfolio_npos(sigs, opens, highs, lows, closes, n_bars, atr, ema, s, e, **kw)
    st = calc_stats_compound(trades)
    return {"trades":st["trades"],"wr":st["wr"],"pnl":st["pnl"]}

def gen_random(n, s, e, seed, tp, sl, n_pat=50):
    rng = np.random.RandomState(seed)
    span = max(1, e - s - 10)
    bars = sorted(rng.randint(s+5, s+5+span, size=max(1,n)))
    dirs = rng.choice(["LONG","SHORT"], size=max(1,n))
    return [(int(b), f"R{i%n_pat}", d, tp, sl) for i,(b,d) in enumerate(zip(bars,dirs))]

def main():
    t0 = time.time()
    print("Regime Robustness Stress Test")
    print("="*80)
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens,highs,lows,closes = df["open"].values,df["high"].values,df["low"].values,df["close"].values
    type_codes = df["candle_type"].tolist()
    atr = compute_atr_ratio(highs,lows,closes)
    ema = compute_ema_slope(closes)
    sig_idx = build_signal_index(type_codes, n_bars)
    with open(PATTERNS_FILE) as f: pd_ = json.load(f)["pattern_details"]
    med_tp = float(np.median([i["tp"]*TSF for i in pd_.values()]))
    med_sl = float(np.median([i["sl"] for i in pd_.values()]))
    print(f"  {n_bars} bars ({n_bars/288:.0f}d), median TP={med_tp:.3f}% SL={med_sl:.3f}%")

    def ref_sigs(s, e):
        t=[]
        for k,info in pd_.items():
            p,d=info["pattern"],info["direction"]; tp,sl=info["tp"]*TSF,info["sl"]
            if p not in sig_idx: continue
            for b in sig_idx[p]:
                if s<=b<e: t.append((b,p,d,tp,sl))
        return sorted(t,key=lambda x:x[0])
    def inv_sigs(s, e):
        return [(b,f"I_{p}","SHORT" if d=="LONG" else "LONG",tp,sl) for b,p,d,tp,sl in ref_sigs(s,e)]

    # Generate windows
    windows = []
    for start in range(0, n_bars - WINDOW_BARS + 1, STEP_BARS):
        end = min(start + WINDOW_BARS, n_bars)
        ps, pe = closes[start], closes[end-1]
        if ps <= 0: continue
        chg = (pe/ps-1)*100
        regime = "BULL" if chg>BULL_THR else ("BEAR" if chg<BEAR_THR else "SIDEWAYS")
        windows.append({"start":start,"end":end,"day_s":start//288,"day_e":end//288,"chg":round(chg,1),"regime":regime})

    rc = {}
    for w in windows: rc[w["regime"]] = rc.get(w["regime"],0)+1
    print(f"  {len(windows)} windows (30d/15d step), regimes: {rc}")

    print(f"\n  {'Window':>10s} | {'Regime':>8s} | {'Chg':>6s} | {'REF':>8s} | {'RND':>8s} | {'INV':>8s} | {'NoM':>8s} | {'Gap':>6s}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")

    results = []
    for w in windows:
        s, e = w["start"], w["end"]
        rs = ref_sigs(s, e)
        ref_r = run_window(rs, opens, highs, lows, closes, n_bars, atr, ema, s, e)
        rnd_pnls = [run_window(gen_random(max(10,len(rs)),s,e,sd,med_tp,med_sl),
                               opens,highs,lows,closes,n_bars,atr,ema,s,e)["pnl"] for sd in [42,123,7]]
        rnd_avg = np.mean(rnd_pnls)
        inv_r = run_window(inv_sigs(s,e), opens, highs, lows, closes, n_bars, atr, ema, s, e)
        nom_r = run_window(rs, opens, highs, lows, closes, n_bars, atr, ema, s, e,
                           use_cascade=False, use_momentum=False, use_aggrisk=False)
        gap = ref_r["pnl"] - rnd_avg
        wr = {"window":f"d{w['day_s']}-{w['day_e']}","regime":w["regime"],"chg":w["chg"],
              "ref":round(ref_r["pnl"],1),"rnd":round(rnd_avg,1),
              "inv":round(inv_r["pnl"],1),"nom":round(nom_r["pnl"],1),"gap":round(gap,1)}
        results.append(wr)
        print(f"  {wr['window']:>10s} | {w['regime']:>8s} | {w['chg']:+5.1f}% | "
              f"{ref_r['pnl']:+7.1f}% | {rnd_avg:+7.1f}% | {inv_r['pnl']:+7.1f}% | {nom_r['pnl']:+7.1f}% | {gap:+5.1f}%")

    # Aggregate
    print("\n" + "="*80)
    print("REGIME AGGREGATION")
    print("="*80)
    regime_stats = {}
    for regime in ["SIDEWAYS","BULL","BEAR"]:
        rws = [w for w in results if w["regime"]==regime]
        if not rws: continue
        st = {
            "n": len(rws),
            "ref_avg": round(np.mean([w["ref"] for w in rws]),1),
            "rnd_avg": round(np.mean([w["rnd"] for w in rws]),1),
            "inv_avg": round(np.mean([w["inv"] for w in rws]),1),
            "nom_avg": round(np.mean([w["nom"] for w in rws]),1),
            "gap_avg": round(np.mean([w["gap"] for w in rws]),1),
        }
        regime_stats[regime] = st
        ratio = st["rnd_avg"]/st["ref_avg"]*100 if st["ref_avg"]!=0 else 0
        print(f"\n  [{regime}] ({st['n']} windows)")
        print(f"    REF={st['ref_avg']:+.1f}% | RND={st['rnd_avg']:+.1f}% | INV={st['inv_avg']:+.1f}% | NoM={st['nom_avg']:+.1f}%")
        print(f"    REF-RND gap={st['gap_avg']:+.1f}% | RND/REF={ratio:.0f}%")
        if st["inv_avg"] > 0: print(f"    => Direction irrelevant in {regime}")
        else: print(f"    => Direction matters in {regime}")

    # Non-neutral tail
    nw = find_neutral_window(closes, tol_pct=1.0)
    tail_result = None
    if nw and nw[1] < n_bars - 288:
        ts, te = nw[1], n_bars
        tc = (closes[te-1]/closes[ts]-1)*100 if closes[ts]>0 else 0
        print(f"\n  NON-NEUTRAL TAIL (day {ts//288}+, {(te-ts)/288:.0f}d, chg={tc:+.1f}%)")
        rs = ref_sigs(ts,te)
        ref_r = run_window(rs,opens,highs,lows,closes,n_bars,atr,ema,ts,te)
        rnd_avg = np.mean([run_window(gen_random(max(10,len(rs)),ts,te,sd,med_tp,med_sl),
                                      opens,highs,lows,closes,n_bars,atr,ema,ts,te)["pnl"] for sd in [42,123,7]])
        inv_r = run_window(inv_sigs(ts,te),opens,highs,lows,closes,n_bars,atr,ema,ts,te)
        print(f"    REF={ref_r['pnl']:+.1f}% | RND={rnd_avg:+.1f}% | INV={inv_r['pnl']:+.1f}%")
        tail_result = {"ref":round(ref_r["pnl"],1),"rnd":round(rnd_avg,1),"inv":round(inv_r["pnl"],1),"chg":round(tc,1)}

    # Diagnosis
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    all_gaps = [w["gap"] for w in results]
    avg_gap = np.mean(all_gaps)
    inv_pos = sum(1 for w in results if w["inv"]>0)
    n_w = len(results)
    print(f"\n  Avg REF-RND gap: {avg_gap:+.1f}%")
    print(f"  INVERTED profitable: {inv_pos}/{n_w} ({inv_pos/n_w*100:.0f}%)")

    sw = [w for w in results if w["regime"]=="SIDEWAYS"]
    tr = [w for w in results if w["regime"]!="SIDEWAYS"]
    if sw and tr:
        sw_pnl = np.mean([w["ref"] for w in sw])
        tr_pnl = np.mean([w["ref"] for w in tr])
        print(f"\n  SIDEWAYS avg REF PnL: {sw_pnl:+.1f}%")
        print(f"  TREND avg REF PnL:    {tr_pnl:+.1f}%")
        if tr_pnl < 0:
            print(f"  => REGIME-DEPENDENT: System LOSES in trends")
            print(f"  => ACTION: Implement regime detection, reduce/halt in trends")
        elif tr_pnl < sw_pnl * 0.3:
            print(f"  => REGIME-BIASED: 3x+ better in sideways")
            print(f"  => ACTION: Reduce sizing in trends")
        else:
            print(f"  => REGIME-ROBUST: Works across conditions")

    elapsed = time.time() - t0
    output = {"study":"regime_robustness_stress","date":datetime.now().isoformat(),
              "windows":results,"regime_stats":regime_stats,"tail":tail_result,
              "diagnostics":{"avg_gap":round(avg_gap,1),"inv_positive_ratio":round(inv_pos/n_w,2)},
              "elapsed_s":round(elapsed,1)}
    with open(OUTPUT_FILE,"w") as f: json.dump(output,f,indent=2,default=str)
    print(f"\nSaved: {OUTPUT_FILE}\nTime: {elapsed:.0f}s")

if __name__=="__main__": main()
