#!/usr/bin/env python3
"""
Cascade SL Tighten % Sweep (Realistic Sim with v1.67.1 fix)
=============================================================
현행 95% 축소는 BT 동일 바에서만 유효.
Realistic sim (1-bar delay + vol_adapt + decay + timeout_pnl + original SL fix)로 최적 비율 탐색.

Output: results/cascade_tighten_sweep.json
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
    FEE_PCT, LEVERAGE, SLIPPAGE_BUFFER, MAX_DAILY_LOSS_PCT,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, DEFAULT_REGIME_MULT,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI, TIMEOUT_BARS,
    DEFAULT_PREEMPTIVE_CASCADE_PCT,
)

DATA_FILE = "data/btc_5m_270days_reclassified.csv"
PATTERNS_FILE = "results/dynamic_patterns.json"
OUTPUT_FILE = "results/cascade_tighten_sweep.json"
DECAY_RATE = 0.9975
CL, CH = DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI


def portfolio_sim(sig_tuples, opens, highs, lows, closes, n_bars,
                   atr_ratio, ema_slope, start, end,
                   cascade_tighten_pct=95, pre_tighten_pct=95,
                   cascade_delay=1):
    ns = DEFAULT_N_SLOTS; sp = 100.0/ns; fee = FEE_PCT*LEVERAGE; dc = DEFAULT_DIRECTION_CAP
    positions, trades = [], []; eq, peak = 100.0, 100.0; mdd = 0.0
    mp = {"LONG": -1, "SHORT": -1}
    c_keep = 1.0 - cascade_tighten_pct/100.0
    p_keep = 1.0 - pre_tighten_pct/100.0
    sigs = sorted([(s,p,d,tp,sl) for s,p,d,tp,sl in sig_tuples if start<=s<end], key=lambda x:x[0])
    si = 0; pending = {}

    for bar in range(start, end):
        if bar in pending:
            for cd, ck in pending[bar]:
                for pos in positions:
                    if pos["direction"]!=cd or pos.get("cascaded"): continue
                    sg=pos["signal_bar"]; rv=1.0
                    if atr_ratio is not None and sg<len(atr_ratio) and not np.isnan(atr_ratio[sg]):
                        rv=max(CL,min(CH,atr_ratio[sg]))
                    ps=pos["sl_pct"]
                    if ps>0: rv=min(rv,MAX_DAILY_LOSS_PCT/LEVERAGE/ps)
                    pos["eff_sl_override"]=ps*rv*ck; pos["cascaded"]=True
            del pending[bar]

        if DEFAULT_PREEMPTIVE_CASCADE_PCT>0 and len(positions)>=2:
            for pd_ in ('LONG','SHORT'):
                dp=[p for p in positions if p['direction']==pd_ and not p.get('pre_cascaded')]
                if len(dp)<2: continue
                ul=0.0
                for p in dp:
                    eb_=p['entry_bar']
                    if eb_>=n_bars or bar<eb_: continue
                    ep=opens[eb_]
                    if ep<=0: continue
                    unr=((closes[bar]/ep-1) if pd_=='LONG' else (1-closes[bar]/ep))*100*LEVERAGE
                    if unr<0: ul+=abs(unr)*(sp/100)*p.get('size_mult',1.0)
                if ul>DEFAULT_PREEMPTIVE_CASCADE_PCT:
                    for p in dp:
                        sg=p['signal_bar']; rv=1.0
                        if atr_ratio is not None and sg<len(atr_ratio) and not np.isnan(atr_ratio[sg]):
                            rv=max(CL,min(CH,atr_ratio[sg]))
                        ps=p['sl_pct']
                        if ps>0: rv=min(rv,MAX_DAILY_LOSS_PCT/LEVERAGE/ps)
                        ns_=ps*rv*p_keep; cur=p.get('eff_sl_override')
                        if cur is None or ns_<cur:
                            p['eff_sl_override']=ns_; p['pre_cascaded']=True; p['cascaded']=True

        closed=[]; bpnl=0.0; sl_dirs=set()
        for pos in positions:
            eb=pos["entry_bar"]
            if bar<eb: continue
            entry=opens[eb]
            if entry<=0: continue
            tp_pct,sl_pct=pos["tp_pct"],pos["sl_pct"]
            d,sb=pos["direction"],pos["signal_bar"]
            ea=1.0
            if atr_ratio is not None and sb<len(atr_ratio) and not np.isnan(atr_ratio[sb]):
                ea=max(CL,min(CH,atr_ratio[sb]))
            r=ea
            if sl_pct>0: r=min(r,MAX_DAILY_LOSS_PCT/LEVERAGE/sl_pct)
            bh=bar-eb
            if bh>=TIMEOUT_BARS:
                xp=closes[bar] if bar<n_bars else opens[min(bar,n_bars-1)]
                pnl=((xp/entry-1) if d=="LONG" else (1-xp/entry))*100*LEVERAGE-fee
                sm=pos.get("size_mult",1.0)
                trades.append({"entry_bar":eb,"exit_bar":bar,"pnl_slot":pnl,"reason":"TIMEOUT",
                               "pattern":pos["pattern"],"direction":d,"size_mult":sm,
                               "pnl_portfolio":pnl*(sp/100)*sm})
                closed.append(pos["slot"]); bpnl+=pnl*(sp/100)*sm; continue
            vf=1.0
            if atr_ratio is not None and bar<len(atr_ratio) and not np.isnan(atr_ratio[bar]):
                ca=max(CL,min(CH,atr_ratio[bar]))
                if ea>0: vf=max(0.5,min(2.0,ca/ea))
            decay=DECAY_RATE**bh
            eff_tp=tp_pct*decay*vf*r+SLIPPAGE_BUFFER
            eso=pos.get("eff_sl_override")
            if eso is not None: eff_sl=max(0.1,eso*vf-SLIPPAGE_BUFFER)
            else: eff_sl=max(0.1,sl_pct*r*vf-SLIPPAGE_BUFFER)
            if d=="LONG": tp_p,sl_p=entry*(1+eff_tp/100),entry*(1-eff_sl/100)
            else: tp_p,sl_p=entry*(1-eff_tp/100),entry*(1+eff_sl/100)
            hv,lv=highs[bar],lows[bar]
            ht=(hv>=tp_p if d=="LONG" else lv<=tp_p)
            hs=(lv<=sl_p if d=="LONG" else hv>=sl_p)
            if not ht and not hs: continue
            if ht and hs:
                if abs(tp_p-opens[bar])<=abs(sl_p-opens[bar]): xp,rs=tp_p,"TP"
                else: xp,rs=sl_p,"SL"
            elif ht: xp,rs=tp_p,"TP"
            else: xp,rs=sl_p,"SL"
            pnl=((xp/entry-1) if d=="LONG" else (1-xp/entry))*100*LEVERAGE-fee
            sm=pos.get("size_mult",1.0)
            trades.append({"entry_bar":eb,"exit_bar":bar,"pnl_slot":pnl,"reason":rs,
                           "pattern":pos["pattern"],"direction":d,"size_mult":sm,
                           "pnl_portfolio":pnl*(sp/100)*sm,"cascaded":pos.get("cascaded",False)})
            closed.append(pos["slot"]); bpnl+=pnl*(sp/100)*sm
            if rs=="SL" and not pos.get("cascaded"): sl_dirs.add(d)

        if cascade_tighten_pct>0 and sl_dirs:
            ab=bar+cascade_delay
            if ab not in pending: pending[ab]=[]
            for sd in sl_dirs: pending[ab].append((sd,c_keep))

        positions=[p for p in positions if p["slot"] not in closed]
        eq+=bpnl
        if eq>peak: peak=eq
        if DEFAULT_MOMENTUM_LOOKBACK>0 and bar>=DEFAULT_MOMENTUM_LOOKBACK:
            pc=(closes[bar]/closes[bar-DEFAULT_MOMENTUM_LOOKBACK]-1)*100 if closes[bar-DEFAULT_MOMENTUM_LOOKBACK]>0 else 0
            if pc>DEFAULT_MOMENTUM_THRESHOLD: mp["SHORT"]=bar+DEFAULT_MOMENTUM_COOLDOWN
            elif pc<-DEFAULT_MOMENTUM_THRESHOLD: mp["LONG"]=bar+DEFAULT_MOMENTUM_COOLDOWN
        while si<len(sigs) and sigs[si][0]==bar:
            sb,pat,d,tp,sl=sigs[si]; si+=1
            if len(positions)>=ns: continue
            if sum(1 for p in positions if p["direction"]==d)>=dc: continue
            if any(p["pattern"]==pat for p in positions): continue
            eb_=sb+1
            if eb_>=n_bars: continue
            if bar<mp.get(d,-1): continue
            sm=1.0
            if DEFAULT_REGIME_MULT is not None and bar<len(ema_slope):
                s=ema_slope[bar]
                if (s>0 and d=="SHORT") or (s<=0 and d=="LONG"): sm=DEFAULT_REGIME_MULT
            if DEFAULT_AGG_RISK_COUNTER>0 or DEFAULT_AGG_RISK_WITH>0:
                up=ema_slope[bar]>0 if bar<len(ema_slope) else False
                ctr=(up and d=="SHORT") or (not up and d=="LONG")
                cap=DEFAULT_AGG_RISK_COUNTER if ctr else DEFAULT_AGG_RISK_WITH
                exp=0.0
                for p in positions:
                    if p["direction"]==d:
                        psl,psig=p["sl_pct"],p["signal_bar"]; pr=1.0
                        if atr_ratio is not None and psig<len(atr_ratio) and not np.isnan(atr_ratio[psig]):
                            pr=max(CL,min(CH,atr_ratio[psig]))
                        if psl>0: pr=min(pr,MAX_DAILY_LOSS_PCT/LEVERAGE/psl)
                        exp+=psl*pr*(1.0/ns)*LEVERAGE*p.get("size_mult",1.0)
                nr=1.0
                if atr_ratio is not None and sb<len(atr_ratio) and not np.isnan(atr_ratio[sb]):
                    nr=max(CL,min(CH,atr_ratio[sb]))
                if sl>0: nr=min(nr,MAX_DAILY_LOSS_PCT/LEVERAGE/sl)
                if exp+sl*nr*(1.0/ns)*LEVERAGE*sm>cap: continue
            positions.append({"slot":f"{pat}_{sb}","signal_bar":sb,"entry_bar":eb_,
                              "direction":d,"pattern":pat,"tp_pct":tp,"sl_pct":sl,"size_mult":sm})
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


def main():
    t0 = time.time()
    print("Cascade SL Tighten % Sweep (Realistic, v1.67.1 fix)")
    print("=" * 105)
    df = load_and_classify(DATA_FILE)
    n = len(df)
    o,h,l,c = df["open"].values,df["high"].values,df["low"].values,df["close"].values
    tc = df["candle_type"].tolist()
    nws,nwe = find_neutral_window(c, tol_pct=1.0)
    atr = compute_atr_ratio(h,l,c)
    ema = compute_ema_slope(c)
    si = build_signal_index(tc, n)
    with open(PATTERNS_FILE) as f: pd_ = json.load(f)["pattern_details"]

    def make_sigs(s,e):
        t=[]
        for pk,pv in pd_.items():
            p,d=pv["pattern"],pv["direction"]
            tp=pv["exc_stats"]["mfe_median"]; sl=pv["sl"]
            if p not in si: continue
            for b in si[p]:
                if s<=b<e: t.append((b,p,d,tp,sl))
        return sorted(t, key=lambda x:x[0])

    all_sigs = make_sigs(nws, nwe)
    print(f"  {n} bars, NW [{nws},{nwe}], {len(all_sigs)} signals\n")

    tighten_pcts = [0, 30, 40, 50, 60, 70, 80, 85, 90, 95]

    print(f"  {'Tighten%':>10s} | {'PnL':>8s} | {'MDD':>5s} | {'P/M':>6s} | {'WR':>5s} | {'R:R':>5s} | {'AvgSL':>6s} | "
          f"{'CascSL':>6s} | {'OrigSL':>6s} | {'Tr':>5s} | {'OOS':>8s} | WF")
    sep = f"  {'-'*10}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*8}-+-{'-'*4}"
    print(sep)

    results = {}
    for tp_pct in tighten_pcts:
        kw = {"cascade_tighten_pct": tp_pct, "pre_tighten_pct": tp_pct, "cascade_delay": 1}
        tr, st = portfolio_sim(all_sigs, o,h,l,c, n, atr, ema, nws, nwe, **kw)
        s = calc_stats_compound(tr)
        mdd = st["mdd_mtm"]; pm = s["pnl"]/mdd if mdd>0 else 0
        sl_tr=[t for t in tr if t.get("reason")=="SL"]
        tp_tr=[t for t in tr if t.get("reason")=="TP"]
        casc=[t for t in sl_tr if t.get("cascaded")]
        orig=[t for t in sl_tr if not t.get("cascaded")]
        avg_sl=np.mean([abs(t["pnl_slot"]) for t in sl_tr]) if sl_tr else 0
        tp_p=[abs(t["pnl_slot"]) for t in tp_tr]; sl_p=[abs(t["pnl_slot"]) for t in sl_tr]
        rr=np.mean(tp_p)/np.mean(sl_p) if sl_p and tp_p else 0

        # WF
        seg=(nwe-nws)//4; folds=[]
        for f in range(3):
            ie=nws+(f+1)*seg; os_=ie; oe=nws+(f+2)*seg if f<2 else nwe
            sg=make_sigs(os_,oe)
            if not sg: folds.append(0); continue
            a2=compute_atr_ratio(h[:oe],l[:oe],c[:oe]); e2=compute_ema_slope(c[:oe])
            t2,_=portfolio_sim(sg,o,h,l,c,oe,a2,e2,os_,oe,**kw)
            folds.append(round(calc_stats_compound(t2)["pnl"],1))
        oos=sum(folds); verd="PASS" if all(f>0 for f in folds) else "FAIL"

        label = f"{tp_pct}%" if tp_pct > 0 else "OFF"
        results[label] = {"pnl":s["pnl"],"mdd":mdd,"pm":round(pm,1),"wr":s["wr"],"rr":round(rr,3),
                          "avg_sl":round(avg_sl,2),"casc":len(casc),"orig":len(orig),
                          "trades":s["trades"],"oos":round(oos,1),"verdict":verd}
        mk = " <<CUR" if tp_pct==95 else ""
        print(f"  {label:>10s} | {s['pnl']:+7.1f}% | {mdd:4.1f}% | {pm:5.1f}x | "
              f"{s['wr']:.1f}% | {rr:.3f} | {avg_sl:5.2f}% | {len(casc):>5d} | {len(orig):>5d} | "
              f"{s['trades']:>5d} | {oos:+7.1f}% | {verd}{mk}")

    best = max(results.items(), key=lambda x: x[1]["oos"])
    print(f"\n  Best OOS: {best[0]} ({best[1]['oos']:+.1f}%, P/M {best[1]['pm']:.1f}x)")

    elapsed = time.time() - t0
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"study":"cascade_tighten_sweep","date":datetime.now().isoformat(),
                    "results":results,"elapsed_s":round(elapsed,1)}, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}\nTime: {elapsed:.0f}s")

if __name__ == "__main__": main()
