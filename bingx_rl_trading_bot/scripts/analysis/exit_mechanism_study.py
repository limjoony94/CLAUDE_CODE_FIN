"""
Exit Mechanism Study - v1.53.0 Production-Aligned Deep Research
================================================================
  F1: Breakeven SL | F2: Trailing Stop | F3: TP Expansion
  F4: Time Decay TP | F5: Cascade SL variants

Standard Research Protocol: Entry next-bar open, Intrabar exit,
  Compound 1/N sizing, Fee 0.30%, Slippage 0.02%, WF 3-fold expanding
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.production.pattern_5m.indicators import classify_candle

LEVERAGE = 3
FEE_PCT = 0.0010 * LEVERAGE
SLIPPAGE_PCT = 0.0002
N_SLOTS = 9
DIRECTION_CAP = 7
TIMEOUT_BARS = 288
AGG_RISK_COUNTER = 8.0
AGG_RISK_WITH = 15.0
MOMENTUM_LOOKBACK = 3
MOMENTUM_THRESHOLD = 1.5
MOMENTUM_COOLDOWN = 12
CASCADE_TIGHTEN_DEFAULT = 85
ATR_PERIOD = 14
ATR_WINDOW = 576
ATR_CLAMP_LO = 0.5
ATR_CLAMP_HI = 1.5
EMA_PERIOD = 20
EMA_LOOKBACK = 5

DATA_FILE = PROJECT_ROOT / "data" / "btc_5m_270days_reclassified.csv"
PATTERNS_FILE = PROJECT_ROOT / "results" / "dynamic_patterns.json"
OUTPUT_FILE = PROJECT_ROOT / "results" / "exit_mechanism_study.json"

def load_data():
    df = pd.read_csv(DATA_FILE)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_patterns():
    with open(PATTERNS_FILE, "r") as f:
        data = json.load(f)
    return data["patterns"], data.get("patterns_tpsl", {}), data.get("pattern_details", {})


def classify_candles(df):
    if "type_code" in df.columns:
        df["candle_type"] = df["type_code"]
        return df
    df["avg_body"] = (abs(df["close"] - df["open"])).rolling(20).mean()
    types = []
    for i in range(len(df)):
        row = df.iloc[i]
        ab = row.get("avg_body", 1.0)
        if pd.isna(ab):
            ab = 1.0
        types.append(classify_candle(row, ab).value)
    df["candle_type"] = types
    return df


def compute_atr(df):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    atr = pd.Series(tr).rolling(ATR_PERIOD, min_periods=1).mean().values
    med = pd.Series(atr).rolling(ATR_WINDOW, min_periods=1).median().values
    df["atr_ratio"] = np.clip(np.where(med > 0, atr / med, 1.0), ATR_CLAMP_LO, ATR_CLAMP_HI)
    return df


def compute_ema_slope(closes):
    ema = pd.Series(closes).ewm(span=EMA_PERIOD, adjust=False).mean().values
    slope = np.zeros(len(closes))
    for i in range(EMA_LOOKBACK, len(closes)): slope[i] = ema[i] - ema[i - EMA_LOOKBACK]
    return slope


def build_signal_index(df, patterns, tpsl):
    ct = df["candle_type"].values; ar = df["atr_ratio"].values
    signals = []
    for i in range(2, len(df) - 1):
        pat3 = f"{ct[i-2]}-{ct[i-1]}-{ct[i]}"
        for d in ["long", "short"]:
            if pat3 in patterns.get(d, []):
                ts = tpsl.get(pat3, [1.5, 2.5])
                btp, bsl = ts[0] / 100.0, ts[1] / 100.0
                signals.append({"bar": i, "entry_bar": i + 1, "pattern": pat3,
                    "direction": d, "entry_price": df["open"].iloc[i + 1],
                    "base_tp_pct": btp, "base_sl_pct": bsl,
                    "eff_tp_pct": btp * ar[i], "eff_sl_pct": bsl * ar[i],
                    "atr_ratio": ar[i]})
    return signals


# ========== Global Arrays (for speed) ==========
_G_HIGHS = None
_G_LOWS = None
_G_OPENS = None
_G_CLOSES = None

def _init_arrays(df):
    global _G_HIGHS, _G_LOWS, _G_OPENS, _G_CLOSES
    _G_HIGHS = df["high"].values
    _G_LOWS = df["low"].values
    _G_OPENS = df["open"].values
    _G_CLOSES = df["close"].values

# ========== Exit Functions ==========

def _exit_loop(entry, direction, entry_bar, df, tp_fn, sl_fn):
    st = {"best": entry}
    highs, lows, opens = _G_HIGHS, _G_LOWS, _G_OPENS
    end = min(entry_bar + TIMEOUT_BARS, len(highs))
    for j in range(entry_bar, end):
        h, l, o = highs[j], lows[j], opens[j]
        tp = tp_fn(j, st, h, l); sl = sl_fn(j, st, h, l)
        if direction == "long":
            if h > st["best"]: st["best"] = h
            tp_hit, sl_hit = h >= tp, l <= sl
        else:
            if l < st["best"]: st["best"] = l
            tp_hit, sl_hit = l <= tp, h >= sl
        td, sd = abs(tp - o), abs(sl - o)
        if tp_hit and sl_hit: return (j, tp, "TP") if td <= sd else (j, sl, "SL")
        if tp_hit: return j, tp, "TP"
        if sl_hit: return j, sl, "SL"
    if entry_bar + TIMEOUT_BARS < len(highs):
        return entry_bar + TIMEOUT_BARS, opens[entry_bar + TIMEOUT_BARS], "TIMEOUT"
    return None


def check_exit_baseline(pos, bar, df, **kw):
    e = pos["entry_price"]; d = pos["direction"]
    tp, sl = pos["eff_tp_pct"], pos["eff_sl_pct"]
    if d == "long": tpp, slp = e * (1 + tp), e * (1 - sl)
    else: tpp, slp = e * (1 - tp), e * (1 + sl)
    return _exit_loop(e, d, pos["entry_bar"], df, lambda j,s,h,l: tpp, lambda j,s,h,l: slp)


def check_exit_breakeven(pos, bar, df, profit_trigger_pct=0.005, **kw):
    e = pos["entry_price"]; d = pos["direction"]
    tp_pct, sl_pct = pos["eff_tp_pct"], pos["eff_sl_pct"]
    if d == "long":
        tpp = e * (1 + tp_pct); orig_sl = e * (1 - sl_pct); trig = e * (1 + profit_trigger_pct)
    else:
        tpp = e * (1 - tp_pct); orig_sl = e * (1 + sl_pct); trig = e * (1 - profit_trigger_pct)
    cur_sl = [orig_sl]
    def sl_fn(j, st, h, l):
        if not st.get("be"):
            if (d == "long" and h >= trig) or (d == "short" and l <= trig):
                st["be"] = True; cur_sl[0] = e
        return cur_sl[0]
    return _exit_loop(e, d, pos["entry_bar"], df, lambda j,s,h,l: tpp, sl_fn)


def check_exit_trailing(pos, bar, df, trail_pct=0.008, activation_pct=0.005, **kw):
    e = pos["entry_price"]; d = pos["direction"]
    tp_pct, sl_pct = pos["eff_tp_pct"], pos["eff_sl_pct"]
    if d == "long":
        tpp = e * (1 + tp_pct); orig_sl = e * (1 - sl_pct); act_p = e * (1 + activation_pct)
    else:
        tpp = e * (1 - tp_pct); orig_sl = e * (1 + sl_pct); act_p = e * (1 - activation_pct)
    cur_sl = [orig_sl]
    def sl_fn(j, st, h, l):
        b = st["best"]
        if d == "long":
            if b >= act_p: cur_sl[0] = max(cur_sl[0], b * (1 - trail_pct))
        else:
            if b <= act_p: cur_sl[0] = min(cur_sl[0], b * (1 + trail_pct))
        return cur_sl[0]
    return _exit_loop(e, d, pos["entry_bar"], df, lambda j,s,h,l: tpp, sl_fn)


def check_exit_tp_expansion(pos, bar, df, expansion_mult=1.5, momentum_bars=6, **kw):
    e = pos["entry_price"]; d = pos["direction"]; eb = pos["entry_bar"]
    tp_pct, sl_pct = pos["eff_tp_pct"], pos["eff_sl_pct"]
    if d == "long":
        cur_tp = [e * (1 + tp_pct)]; cur_sl = [e * (1 - sl_pct)]
    else:
        cur_tp = [e * (1 - tp_pct)]; cur_sl = [e * (1 + sl_pct)]
    def tp_fn(j, st, h, l):
        if not st.get("exp") and j >= eb + momentum_bars:
            if d == "long":
                prog = (h - e) / (cur_tp[0] - e) if cur_tp[0] != e else 0
                mok = _G_CLOSES[j] > _G_CLOSES[j - momentum_bars]
            else:
                prog = (e - l) / (e - cur_tp[0]) if cur_tp[0] != e else 0
                mok = _G_CLOSES[j] < _G_CLOSES[j - momentum_bars]
            if prog >= 0.70 and mok:
                st["exp"] = True
                if d == "long":
                    cur_tp[0] = e * (1 + tp_pct * expansion_mult)
                    cur_sl[0] = e * (1 - sl_pct * 0.5)
                else:
                    cur_tp[0] = e * (1 - tp_pct * expansion_mult)
                    cur_sl[0] = e * (1 + sl_pct * 0.5)
        return cur_tp[0]
    return _exit_loop(e, d, eb, df, tp_fn, lambda j,s,h,l: cur_sl[0])


def check_exit_time_decay(pos, bar, df, decay_start_bar=144, decay_end_pct=0.5, **kw):
    e = pos["entry_price"]; d = pos["direction"]; eb = pos["entry_bar"]
    tp_pct, sl_pct = pos["eff_tp_pct"], pos["eff_sl_pct"]
    if d == "long": slp = e * (1 - sl_pct)
    else: slp = e * (1 + sl_pct)
    rem = max(TIMEOUT_BARS - decay_start_bar, 1)
    def tp_fn(j, st, h, l):
        held = j - eb
        if held < decay_start_bar: ctp = tp_pct
        else: ctp = tp_pct * (1.0 - min((held - decay_start_bar) / rem, 1.0) * (1.0 - decay_end_pct))
        return e * (1 + ctp) if d == "long" else e * (1 - ctp)
    return _exit_loop(e, d, eb, df, tp_fn, lambda j,s,h,l: slp)


EXIT_FN_MAP = {
    "baseline": check_exit_baseline, "breakeven": check_exit_breakeven,
    "trailing": check_exit_trailing, "tp_expansion": check_exit_tp_expansion,
    "time_decay": check_exit_time_decay,
}


# ========== N-pos Portfolio ==========

def portfolio_npos(signals, df, exit_fn, exit_kw=None, cascade_pct=CASCADE_TIGHTEN_DEFAULT):
    """N-pos portfolio sim with cached exit computation."""
    if exit_kw is None: exit_kw = {}
    closes = df["close"].values
    ema_sl = compute_ema_slope(closes)
    equity, peak = 100.0, 100.0
    slots = {}; nx = 0
    trades = []
    mblk = {"long": 0, "short": 0}
    sbyb = {}
    for sig in signals: sbyb.setdefault(sig["entry_bar"], []).append(sig)
    # Cache: slot_id -> (exit_bar, exit_price, reason)
    exit_cache = {}

    def _compute_exit(sid, pos):
        res = exit_fn(pos, pos["entry_bar"], df, **exit_kw)
        exit_cache[sid] = res

    for bar in range(len(df)):
        # Check exits
        closed = []
        for sid, pos in list(slots.items()):
            res = exit_cache.get(sid)
            if res and res[0] <= bar:
                exb, exp, rsn = res
                ep = pos["entry_price"]; dr = pos["direction"]
                if dr == "long": pnl = ((exp - ep) / ep) * LEVERAGE
                else: pnl = ((ep - exp) / ep) * LEVERAGE
                pnl -= FEE_PCT + SLIPPAGE_PCT * LEVERAGE
                equity += pos["slot_eq"] * pnl
                trades.append({"entry_bar": pos["entry_bar"], "exit_bar": exb,
                    "direction": dr, "pattern": pos["pattern"],
                    "pnl_pct": pnl * 100, "reason": rsn})
                closed.append((sid, dr, rsn))
        for sid, dr, rsn in closed:
            if "SL" in rsn and cascade_pct > 0:
                f = (100 - cascade_pct) / 100.0
                for os_id, op in slots.items():
                    if os_id != sid and op["direction"] == dr:
                        op["eff_sl_pct"] *= f
                        # Re-compute exit for affected positions
                        _compute_exit(os_id, op)
            if sid in slots:
                del slots[sid]
                exit_cache.pop(sid, None)
        if equity > peak: peak = equity
        if bar in sbyb and equity > 0:
            for sig in sbyb[bar]:
                if len(slots) >= N_SLOTS: break
                dr = sig["direction"]
                if sum(1 for s in slots.values() if s["direction"] == dr) >= DIRECTION_CAP: continue
                if mblk[dr] > bar: continue
                if bar >= MOMENTUM_LOOKBACK:
                    pc = abs(closes[bar] - closes[bar - MOMENTUM_LOOKBACK]) / closes[bar - MOMENTUM_LOOKBACK] * 100
                    if pc >= MOMENTUM_THRESHOLD:
                        up = closes[bar] > closes[bar - MOMENTUM_LOOKBACK]
                        if (dr == "short" and up) or (dr == "long" and not up):
                            mblk[dr] = bar + MOMENTUM_COOLDOWN; continue
                rup = ema_sl[bar] > 0
                ce = sum(s["eff_sl_pct"] / N_SLOTS * LEVERAGE * 100 for s in slots.values()
                         if (s["direction"] == "long" and not rup) or (s["direction"] == "short" and rup))
                we = sum(s["eff_sl_pct"] / N_SLOTS * LEVERAGE * 100 for s in slots.values()
                         if not ((s["direction"] == "long" and not rup) or (s["direction"] == "short" and rup)))
                ne = sig["eff_sl_pct"] / N_SLOTS * LEVERAGE * 100
                icc = (dr == "long" and not rup) or (dr == "short" and rup)
                if icc and ce + ne > AGG_RISK_COUNTER: continue
                if not icc and we + ne > AGG_RISK_WITH: continue
                if any(s["pattern"] == sig["pattern"] and s["direction"] == dr for s in slots.values()): continue
                dd = (peak - equity) / peak if peak > 0 else 0
                sm = 1.0 if dd < 0.05 else (0.25 if dd >= 0.2 else 1.0 - (dd - 0.05) / 0.15 * 0.75)
                slots[nx] = {"entry_bar": sig["entry_bar"], "entry_price": sig["entry_price"],
                    "direction": dr, "pattern": sig["pattern"],
                    "eff_tp_pct": sig["eff_tp_pct"], "eff_sl_pct": sig["eff_sl_pct"],
                    "slot_eq": (equity / N_SLOTS) * sm, "atr_ratio": sig["atr_ratio"]}
                _compute_exit(nx, slots[nx])
                nx += 1
    return trades, equity


def calc_stats(trades, feq):
    if not trades: return {"trades": 0, "wr": 0, "pnl": 0, "mdd": 0, "pnl_mdd": 0}
    w = sum(1 for t in trades if t["pnl_pct"] > 0)
    p = feq - 100.0
    eq = [100.0]
    for t in sorted(trades, key=lambda x: x["exit_bar"]):
        eq.append(eq[-1] + eq[-1] / N_SLOTS * t["pnl_pct"] / 100)
    pk, md = 100.0, 0.0
    for v in eq:
        if v > pk: pk = v
        dd = (pk - v) / pk * 100
        if dd > md: md = dd
    return {"trades": len(trades), "wins": w, "wr": round(w / len(trades) * 100, 1),
            "pnl": round(p, 2), "mdd": round(md, 2),
            "pnl_mdd": round(p / md if md > 0 else p, 1)}


HYPOTHESES = {
    "H0_baseline": {"exit_fn": "baseline", "cascade_pct": CASCADE_TIGHTEN_DEFAULT, "kwargs": {}},
    "F1a_BE_0.5pct": {"exit_fn": "breakeven", "cascade_pct": CASCADE_TIGHTEN_DEFAULT, "kwargs": {"profit_trigger_pct": 0.005}},
    "F1b_BE_1.0pct": {"exit_fn": "breakeven", "cascade_pct": CASCADE_TIGHTEN_DEFAULT, "kwargs": {"profit_trigger_pct": 0.010}},
    "F1c_BE_0.3pct": {"exit_fn": "breakeven", "cascade_pct": CASCADE_TIGHTEN_DEFAULT, "kwargs": {"profit_trigger_pct": 0.003}},
    "F2a_Trail_0.8_0.5": {"exit_fn": "trailing", "cascade_pct": CASCADE_TIGHTEN_DEFAULT, "kwargs": {"trail_pct": 0.008, "activation_pct": 0.005}},
    "F2b_Trail_1.2_0.8": {"exit_fn": "trailing", "cascade_pct": CASCADE_TIGHTEN_DEFAULT, "kwargs": {"trail_pct": 0.012, "activation_pct": 0.008}},
    "F2c_Trail_0.5_0.3": {"exit_fn": "trailing", "cascade_pct": CASCADE_TIGHTEN_DEFAULT, "kwargs": {"trail_pct": 0.005, "activation_pct": 0.003}},
    "F3a_TPExp_1.5x": {"exit_fn": "tp_expansion", "cascade_pct": CASCADE_TIGHTEN_DEFAULT, "kwargs": {"expansion_mult": 1.5, "momentum_bars": 6}},
    "F3b_TPExp_2.0x": {"exit_fn": "tp_expansion", "cascade_pct": CASCADE_TIGHTEN_DEFAULT, "kwargs": {"expansion_mult": 2.0, "momentum_bars": 6}},
    "F3c_TPExp_1.3x_12": {"exit_fn": "tp_expansion", "cascade_pct": CASCADE_TIGHTEN_DEFAULT, "kwargs": {"expansion_mult": 1.3, "momentum_bars": 12}},
    "F4a_Decay_144_50": {"exit_fn": "time_decay", "cascade_pct": CASCADE_TIGHTEN_DEFAULT, "kwargs": {"decay_start_bar": 144, "decay_end_pct": 0.5}},
    "F4b_Decay_72_60": {"exit_fn": "time_decay", "cascade_pct": CASCADE_TIGHTEN_DEFAULT, "kwargs": {"decay_start_bar": 72, "decay_end_pct": 0.6}},
    "F4c_Decay_192_40": {"exit_fn": "time_decay", "cascade_pct": CASCADE_TIGHTEN_DEFAULT, "kwargs": {"decay_start_bar": 192, "decay_end_pct": 0.4}},
    "F5a_Cascade_90": {"exit_fn": "baseline", "cascade_pct": 90, "kwargs": {}},
    "F5b_Cascade_70": {"exit_fn": "baseline", "cascade_pct": 70, "kwargs": {}},
    "F5c_Cascade_OFF": {"exit_fn": "baseline", "cascade_pct": 0, "kwargs": {}},
}


def run_wf(signals, df, hc, n_folds=3):
    nb = len(df); fs = nb // (n_folds + 1)
    results = []
    for fold in range(n_folds):
        is_e = (fold + 1) * fs
        oos_e = (fold + 2) * fs if fold < n_folds - 1 else nb
        if oos_e <= is_e: continue
        osig = [s for s in signals if is_e <= s["entry_bar"] < oos_e]
        ef = EXIT_FN_MAP[hc["exit_fn"]]
        tr, feq = portfolio_npos(osig, df, ef, hc["kwargs"], hc["cascade_pct"])
        st = calc_stats(tr, feq); st["fold"] = fold + 1; st["oos_bars"] = oos_e - is_e
        results.append(st)
    return results


def main():
    print("=" * 70)
    print("Exit Mechanism Study - v1.53.0 Production-Aligned")
    print("=" * 70)
    t0 = time.time()

    print("\n[1/5] Loading data...")
    df = load_data()
    print(f"  Loaded {len(df)} bars")

    print("[2/5] Classifying candles...")
    df = classify_candles(df)
    df = compute_atr(df)
    _init_arrays(df)

    print("[3/5] Loading patterns...")
    patterns, tpsl, details = load_patterns()
    nl = len(patterns.get("long", [])); ns = len(patterns.get("short", []))
    print(f"  {nl}L + {ns}S = {nl+ns} patterns")

    print("[4/5] Building signal index...")
    signals = build_signal_index(df, patterns, tpsl)
    print(f"  {len(signals)} raw signals")

    print("\n[5/5] Running simulations...")
    print("-" * 70)

    all_res = {}
    for hn, hc in HYPOTHESES.items():
        ef = EXIT_FN_MAP[hc["exit_fn"]]
        tr, feq = portfolio_npos(signals, df, ef, hc["kwargs"], hc["cascade_pct"])
        is_st = calc_stats(tr, feq)
        wf = run_wf(signals, df, hc)
        wfp = all(r["pnl"] > 0 for r in wf) if wf else False
        wfmn = min((r["pnl"] for r in wf), default=0)
        wfav = float(np.mean([r["pnl"] for r in wf])) if wf else 0
        rc = {}
        for t in tr: rc[t["reason"]] = rc.get(t["reason"], 0) + 1
        all_res[hn] = {"is": is_st, "wf_pass": wfp, "wf_folds": wf,
            "wf_min_pnl": round(wfmn, 2), "wf_avg_pnl": round(wfav, 2),
            "exit_reasons": rc,
            "config": {"exit_fn": hc["exit_fn"], "cascade_pct": hc["cascade_pct"], "kwargs": hc["kwargs"]}}
        ws = "PASS" if wfp else "FAIL"
        print(f"  {hn:25s} | IS PnL {is_st['pnl']:+8.1f}% MDD {is_st['mdd']:5.1f}% "
              f"PnL/MDD {is_st['pnl_mdd']:6.1f} WR {is_st['wr']:5.1f}% "
              f"| WF {ws} min {wfmn:+.1f}% avg {wfav:+.1f}%")

    print("\n" + "=" * 70 + "\nRANKING by IS PnL/MDD:\n" + "-" * 70)
    rk = sorted(all_res.items(), key=lambda x: x[1]["is"]["pnl_mdd"], reverse=True)
    for i, (n, r) in enumerate(rk, 1):
        ws = "PASS" if r["wf_pass"] else "FAIL"
        mk = " <-- BASELINE" if n == "H0_baseline" else ""
        print(f"  #{i:2d} {n:25s} PnL/MDD {r['is']['pnl_mdd']:6.1f} PnL {r['is']['pnl']:+8.1f}% WF:{ws}{mk}")

    bl = all_res.get("H0_baseline", {}).get("is", {}).get("pnl_mdd", 0)
    wn = [n for n, r in all_res.items() if n != "H0_baseline" and r["wf_pass"] and r["is"]["pnl_mdd"] > bl]

    print("\n" + "=" * 70)
    if wn:
        print(f"WINNERS (beat baseline + WF 3/3 PASS): {len(wn)}")
        for w in wn: print(f"  {w}: PnL/MDD {all_res[w]['is']['pnl_mdd']:.1f} (lift +{all_res[w]['is']['pnl_mdd'] - bl:.1f})")
    else:
        print("NO WINNERS - baseline is optimal among tested configs")

    print("\n" + "=" * 70)
    if not wn:
        vd = "KEEP_BASELINE"
        print("VERDICT: KEEP BASELINE - no exit mechanism improvement found")
    else:
        bst = max(wn, key=lambda w: all_res[w]["is"]["pnl_mdd"])
        vd = f"CANDIDATE: {bst}"
        print(f"VERDICT: {bst} is a candidate for further validation")
        print(f"  PnL/MDD: {all_res[bst]['is']['pnl_mdd']:.1f} vs baseline {bl:.1f}")

    el = time.time() - t0
    print(f"\nElapsed: {el:.1f}s")

    out = {"study": "exit_mechanism_study", "version": "v1.53.0",
        "generated_at": datetime.now().isoformat(),
        "data_bars": len(df), "patterns": f"{nl}L+{ns}S",
        "signals": len(signals), "elapsed_sec": round(el, 1),
        "verdict": vd, "baseline": all_res.get("H0_baseline", {}),
        "results": all_res, "winners": wn,
        "ranking": [n for n, _ in rk]}
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
