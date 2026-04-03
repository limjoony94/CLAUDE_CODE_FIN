#!/usr/bin/env python3
"""
partial_close_proper.py — PROPER Partial Close Simulation

Previous study (realistic_cascade_npos.py scenario D) had a flaw:
"saved" positions (non-closed half) were DROPPED entirely from results
instead of continuing to trade with original SLs.

This script implements a FULL portfolio simulation with proper partial close:
- When directional unrealized loss exceeds threshold
- Close worst 50% of direction positions at market (closes[bar])
- KEEP remaining 50% with ORIGINAL SLs — they continue normally
- No SL tightening at all

Uses scanner's actual signal generation but custom portfolio loop.

Standard: compound, LEVERAGE=3, fee=0.10%, timeout=576, ATR clamp [1.0,1.5]
"""
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "scanner"))
sys.path.insert(0, str(ROOT / "scripts" / "production"))

from pattern_scanner import build_signal_index, compute_atr_ratio, compute_ema_slope
from pattern_5m.indicators import classify_candle

LEVERAGE = 3
FEE = 0.05 / 100 * LEVERAGE * 2  # 0.05% * 2 sides * leverage
TIMEOUT = 576
ATR_CLAMP_LO = 1.0
ATR_CLAMP_HI = 1.5


def load():
    df = pd.read_csv(ROOT / "data" / "btc_5m_270days_reclassified.csv",
                      parse_dates=["timestamp"])
    o, h, l, c = [df[x].values.astype(float) for x in ["open", "high", "low", "close"]]
    n = len(df)
    body = np.abs(c - o)
    avg = pd.Series(body).rolling(20, min_periods=1).mean().values
    types = []
    for i in range(n):
        row = type("R", (), {"open": o[i], "high": h[i], "low": l[i], "close": c[i]})()
        types.append(classify_candle(row, avg[i]).value)

    si = build_signal_index(types, n)
    ar = compute_atr_ratio(h, l, c)

    with open(ROOT / "results" / "dynamic_patterns.json") as f:
        dp = json.load(f)
    pd_ = dp.get("pattern_details", {})

    # Build signals indexed by bar
    sigs_by_bar = {}
    for pk, info in pd_.items():
        pat = info["pattern"]
        if pat not in si:
            continue
        for bar in si[pat]:
            eb = bar + 1
            if eb >= n:
                continue
            ratio = ar[bar] if bar < len(ar) else 1.0
            ratio = max(ATR_CLAMP_LO, min(ATR_CLAMP_HI, ratio))
            entry = o[eb]
            tp_pct = info["tp"] / 100.0 * ratio
            sl_pct = info["sl"] / 100.0 * ratio
            d = info["direction"]
            if d == "LONG":
                tp = entry * (1 + tp_pct)
                sl = entry * (1 - sl_pct)
            else:
                tp = entry * (1 - tp_pct)
                sl = entry * (1 + sl_pct)
            if eb not in sigs_by_bar:
                sigs_by_bar[eb] = []
            sigs_by_bar[eb].append(dict(direction=d, entry=entry, tp=tp, sl=sl,
                                        orig_sl=sl, pattern=pk))

    return sigs_by_bar, o, h, l, c, n


def simulate(sigs_by_bar, o, h, l, c, n,
             n_slots=9, direction_cap=6,
             cascade_mode="none",  # "none", "tighten_98", "partial_close"
             cascade_threshold=1.0,
             close_pct=50,  # % of direction positions to close
             start_bar=0, end_bar=None):
    """Full portfolio simulation with proper partial close."""
    if end_bar is None:
        end_bar = n

    positions = []  # list of active positions
    equity = 100.0
    peak = 100.0
    max_dd = 0.0
    trades = []  # all exit records

    for bar in range(start_bar, end_bar):
        cp = c[bar]
        hi = h[bar]
        lo = l[bar]

        # 1. Check exits (TP/SL/Timeout)
        surviving = []
        for pos in positions:
            ent = pos["entry"]
            d = pos["direction"]
            tp = pos["tp"]
            sl = pos["sl"]
            bars_held = bar - pos["bar"]

            hit_tp = (d == "LONG" and hi >= tp) or (d == "SHORT" and lo <= tp)
            hit_sl = (d == "LONG" and lo <= sl) or (d == "SHORT" and hi >= sl)
            timed_out = bars_held >= TIMEOUT

            exit_price = None
            reason = None

            if hit_tp and hit_sl:
                if abs(o[bar] - tp) <= abs(o[bar] - sl):
                    exit_price, reason = tp, "TP"
                else:
                    exit_price, reason = sl, "SL"
            elif hit_tp:
                exit_price, reason = tp, "TP"
            elif hit_sl:
                exit_price, reason = sl, "SL"
            elif timed_out:
                exit_price, reason = cp, "TIMEOUT"

            if exit_price is not None:
                if d == "LONG":
                    raw = (exit_price / ent - 1) * LEVERAGE
                else:
                    raw = (1 - exit_price / ent) * LEVERAGE
                raw -= FEE
                pf_pnl = raw / n_slots
                equity *= (1 + pf_pnl)
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100
                max_dd = max(max_dd, dd)
                trades.append(dict(pnl_slot=raw * 100, pnl_pf=pf_pnl * 100,
                                   reason=reason, bar=bar))
            else:
                surviving.append(pos)
        positions = surviving

        # 2. Cascade / Partial close
        if cascade_mode != "none" and len(positions) >= 2:
            for direction in ["LONG", "SHORT"]:
                dir_pos = [p for p in positions if p["direction"] == direction]
                if len(dir_pos) < 2:
                    continue

                # Compute directional unrealized loss
                total_loss = 0.0
                pos_losses = []
                for p in dir_pos:
                    ent = p["entry"]
                    if direction == "LONG":
                        unr = (cp / ent - 1) * 100 * LEVERAGE
                    else:
                        unr = (1 - cp / ent) * 100 * LEVERAGE
                    if unr < 0:
                        total_loss += abs(unr) / n_slots
                    pos_losses.append((p, unr))

                if total_loss < cascade_threshold:
                    continue

                # Filter to non-cascaded
                actionable = [(p, unr) for p, unr in pos_losses
                              if not p.get("cascaded")]
                if not actionable:
                    continue

                if cascade_mode == "partial_close":
                    # Sort by unrealized PnL (worst first)
                    actionable.sort(key=lambda x: x[1])
                    n_close = max(1, len(actionable) * close_pct // 100)

                    for p, unr in actionable[:n_close]:
                        # Close at current market price
                        ent = p["entry"]
                        if direction == "LONG":
                            raw = (cp / ent - 1) * LEVERAGE
                        else:
                            raw = (1 - cp / ent) * LEVERAGE
                        raw -= FEE
                        pf_pnl = raw / n_slots
                        equity *= (1 + pf_pnl)
                        peak = max(peak, equity)
                        dd = (peak - equity) / peak * 100
                        max_dd = max(max_dd, dd)
                        trades.append(dict(pnl_slot=raw * 100, pnl_pf=pf_pnl * 100,
                                           reason="PARTIAL_CLOSE", bar=bar))
                        positions.remove(p)

                    # Mark remaining as cascaded (don't trigger again)
                    for p, unr in actionable[n_close:]:
                        p["cascaded"] = True

                elif cascade_mode == "tighten_98":
                    # Current production behavior: 98% tighten with fallback
                    for p, unr in actionable:
                        orig_sl = p.get("orig_sl", p["sl"])
                        orig_dist = abs(p["entry"] - orig_sl)
                        new_dist = max(orig_dist * 0.02, 30.0)
                        ent = p["entry"]
                        if direction == "LONG":
                            new_sl = ent - new_dist
                            if new_sl >= cp:
                                new_sl = cp * 0.996
                            if new_sl <= p["sl"]:
                                continue
                        else:
                            new_sl = ent + new_dist
                            if new_sl <= cp:
                                new_sl = cp * 1.004
                            if new_sl >= p["sl"]:
                                continue
                        p["sl"] = new_sl
                        p["cascaded"] = True

        # 3. New entries
        if bar in sigs_by_bar:
            for sig in sigs_by_bar[bar]:
                if len(positions) >= n_slots:
                    break
                dc = sum(1 for p in positions if p["direction"] == sig["direction"])
                if dc >= direction_cap:
                    continue
                positions.append(dict(
                    bar=bar, direction=sig["direction"],
                    entry=sig["entry"], tp=sig["tp"],
                    sl=sig["sl"], orig_sl=sig["orig_sl"],
                    cascaded=False, pattern=sig["pattern"]))

    # Stats
    if not trades:
        return dict(pnl=0, mdd=0, trades=0, wr=0, rr=0, pm=0)

    wins = sum(1 for t in trades if t["pnl_slot"] > 0)
    wr = 100 * wins / len(trades)
    total_pnl = equity - 100
    pm = total_pnl / max_dd if max_dd > 0 else 999

    wp = [t["pnl_slot"] for t in trades if t["pnl_slot"] > 0]
    lp = [t["pnl_slot"] for t in trades if t["pnl_slot"] < 0]
    rr = abs(np.mean(wp) / np.mean(lp)) if wp and lp else 999

    # Reason breakdown
    reasons = {}
    for t in trades:
        r = t["reason"]
        if r not in reasons:
            reasons[r] = {"n": 0, "pnl": 0}
        reasons[r]["n"] += 1
        reasons[r]["pnl"] += t["pnl_pf"]

    return dict(pnl=round(total_pnl, 1), mdd=round(max_dd, 2),
                trades=len(trades), wr=round(wr, 1), rr=round(rr, 3),
                pm=round(pm, 1), reasons=reasons)


def main():
    print("Loading...")
    sigs, o, h, l, c, n = load()
    total_sigs = sum(len(v) for v in sigs.values())
    print("  {} signals, {} bars".format(total_sigs, n))

    configs = [
        ("A_NO_CASCADE_N9", dict(n_slots=9, direction_cap=6, cascade_mode="none")),
        ("B_TIGHTEN98_N9", dict(n_slots=9, direction_cap=6, cascade_mode="tighten_98", cascade_threshold=1.0)),
        ("C_PARTIAL50_N9", dict(n_slots=9, direction_cap=6, cascade_mode="partial_close", cascade_threshold=1.0, close_pct=50)),
        ("D_PARTIAL50_N5", dict(n_slots=5, direction_cap=3, cascade_mode="partial_close", cascade_threshold=1.0, close_pct=50)),
        ("E_PARTIAL33_N9", dict(n_slots=9, direction_cap=6, cascade_mode="partial_close", cascade_threshold=1.0, close_pct=33)),
        ("F_PARTIAL67_N9", dict(n_slots=9, direction_cap=6, cascade_mode="partial_close", cascade_threshold=1.0, close_pct=67)),
        ("G_PARTIAL100_N9", dict(n_slots=9, direction_cap=6, cascade_mode="partial_close", cascade_threshold=1.0, close_pct=100)),
        ("H_PARTIAL50_N9_T15", dict(n_slots=9, direction_cap=6, cascade_mode="partial_close", cascade_threshold=1.5, close_pct=50)),
        ("I_PARTIAL50_N9_T05", dict(n_slots=9, direction_cap=6, cascade_mode="partial_close", cascade_threshold=0.5, close_pct=50)),
        ("J_NO_CASCADE_N5", dict(n_slots=5, direction_cap=3, cascade_mode="none")),
    ]

    fmt = "{:<24} {:>8} {:>7} {:>7} {:>6} {:>6} {:>6}"
    print("\n" + fmt.format("Config", "PnL%", "MDD%", "P/M", "Trades", "WR%", "R:R"))
    print("=" * 72)

    results = {}
    for name, kw in configs:
        t0 = time.time()
        r = simulate(sigs, o, h, l, c, n, **kw)
        dt = time.time() - t0
        print("{:<24} {:>8.1f} {:>7.2f} {:>7.1f} {:>6d} {:>6.1f} {:>6.3f} ({:.1f}s)".format(
            name, r["pnl"], r["mdd"], r["pm"], r["trades"], r["wr"], r["rr"], dt))

        # Print reason breakdown
        for reason, data in sorted(r.get("reasons", {}).items()):
            print("  {:>18}: {:>5d} trades, pf PnL {:>+8.1f}%".format(
                reason, data["n"], data["pnl"]))

        results[name] = {"IS": {k: v for k, v in r.items() if k != "reasons"},
                          "IS_reasons": {k: {"n": v["n"], "pnl": round(v["pnl"], 1)}
                                         for k, v in r.get("reasons", {}).items()}}

    # WF for key configs
    print("\n" + "=" * 80)
    print("WF 5-fold expanding window")
    print("=" * 80)

    wf_configs = ["A_NO_CASCADE_N9", "B_TIGHTEN98_N9", "C_PARTIAL50_N9",
                  "D_PARTIAL50_N5", "G_PARTIAL100_N9", "H_PARTIAL50_N9_T15"]

    for name, kw in configs:
        if name not in wf_configs:
            continue
        oos = []
        for fi in range(5):
            s = int(n * (fi + 1) / 6)
            e = int(n * (fi + 2) / 6)
            r = simulate(sigs, o, h, l, c, n, start_bar=s, end_bar=e, **kw)
            oos.append(r["pnl"])
        tot = sum(oos)
        ps = sum(1 for p in oos if p > 0)
        fs = " ".join("{:+6.0f}".format(p) for p in oos)
        print("{:<24} OOS {:>+7.1f}% [{}] {}/5".format(name, tot, fs, ps))
        results[name]["WF"] = dict(total=round(tot, 1),
                                    folds=[round(p, 1) for p in oos],
                                    passes=ps)

    with open(ROOT / "results" / "partial_close_proper.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/partial_close_proper.json")


if __name__ == "__main__":
    main()
