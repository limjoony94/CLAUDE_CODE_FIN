#!/usr/bin/env python3
"""
cascade_cluster_fix_v2.py — Optimized Cascade Cluster Fix Study

CRITICAL: 98% tighten always triggers 0.4% fallback, concentrating ALL SLs
at identical price → cluster loss. Tests 10 alternatives.

Standard: compound, 0.1% fee, next-bar entry, WF 5-fold expanding.
"""
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "production"))
from pattern_5m.indicators import classify_candle

LEVERAGE = 3
FEE_PCT = 0.05
MAX_POSITIONS = 9
TIMEOUT_BARS = 576
ATR_PERIOD = 14
ATR_WINDOW = 576
ATR_CLAMP_LO = 1.0
ATR_CLAMP_HI = 1.5


def load_data():
    df = pd.read_csv(ROOT / "data" / "btc_5m_270days_reclassified.csv",
                      parse_dates=["timestamp"])
    opens = df["open"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n = len(df)

    body = np.abs(closes - opens)
    avg_body_20 = pd.Series(body).rolling(20, min_periods=1).mean().values

    types = []
    for i in range(n):
        row = type("R", (), {
            "open": opens[i], "high": highs[i],
            "low": lows[i], "close": closes[i]
        })()
        ct = classify_candle(row, avg_body_20[i])
        types.append(ct.value if hasattr(ct, "value") else str(ct))

    with open(ROOT / "results" / "dynamic_patterns.json") as f:
        dp = json.load(f)
    pat_details = dp.get("pattern_details", {})

    # ATR
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - prev_close),
                               np.abs(lows - prev_close)))
    atr = pd.Series(tr).rolling(ATR_PERIOD, min_periods=1).mean().values
    atr_ma = pd.Series(atr).rolling(ATR_WINDOW, min_periods=1).mean().values

    # Build signal index
    signals_by_bar = {}
    total_sigs = 0
    for i in range(3, n - 1):
        pat_base = "{}-{}-{}".format(types[i-2], types[i-1], types[i])
        for direction in ["LONG", "SHORT"]:
            pat_key = "{}_{}".format(pat_base, direction)
            if pat_key not in pat_details:
                continue
            pd_info = pat_details[pat_key]
            base_tp = pd_info.get("tp", 1.0) / 100.0
            base_sl = pd_info.get("sl", 2.0) / 100.0

            ratio = atr[i] / atr_ma[i] if atr_ma[i] > 0 else 1.0
            ratio = max(ATR_CLAMP_LO, min(ATR_CLAMP_HI, ratio))

            entry_bar = i + 1
            entry_price = opens[entry_bar]
            tp_pct = base_tp * ratio
            sl_pct = base_sl * ratio

            if direction == "LONG":
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
            else:
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)

            if entry_bar not in signals_by_bar:
                signals_by_bar[entry_bar] = []
            signals_by_bar[entry_bar].append({
                "direction": direction,
                "entry": entry_price,
                "tp": tp_price,
                "sl": sl_price,
            })
            total_sigs += 1

    return signals_by_bar, opens, highs, lows, closes, n, total_sigs


def simulate(signals_by_bar, opens, highs, lows, closes, n,
             direction_cap=6, cascade_mode="none", cascade_threshold=1.0,
             partial_close_pct=50, mild_keep=0.5,
             start_bar=0, end_bar=None):
    if end_bar is None:
        end_bar = n
    n_slots = MAX_POSITIONS

    positions = []
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    trades = []

    for bar in range(start_bar, end_bar):
        cp = closes[bar]
        h = highs[bar]
        l = lows[bar]

        # 1. Exits
        surviving = []
        for pos in positions:
            entry = pos["entry"]
            d = pos["direction"]
            tp = pos["tp"]
            sl = pos["sl"]
            bars_held = bar - pos["bar"]

            hit_tp = (d == "LONG" and h >= tp) or (d == "SHORT" and l <= tp)
            hit_sl = (d == "LONG" and l <= sl) or (d == "SHORT" and h >= sl)
            timed_out = bars_held >= TIMEOUT_BARS

            exit_price = None
            reason = None

            if hit_tp and hit_sl:
                tp_dist = abs(opens[bar] - tp)
                sl_dist = abs(opens[bar] - sl)
                if tp_dist <= sl_dist:
                    exit_price, reason = tp, "TP"
                else:
                    exit_price, reason = sl, "SL"
            elif hit_tp:
                exit_price, reason = tp, "TP"
            elif hit_sl:
                exit_price, reason = sl, "SL"
            elif timed_out:
                exit_price, reason = cp, "TIMEOUT"

            if exit_price:
                if d == "LONG":
                    raw = (exit_price / entry - 1) * LEVERAGE
                else:
                    raw = (1 - exit_price / entry) * LEVERAGE
                raw -= FEE_PCT / 100 * LEVERAGE * 2
                equity *= (1 + raw / n_slots)
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100
                max_dd = max(max_dd, dd)
                trades.append(raw * 100)
            else:
                surviving.append(pos)
        positions = surviving

        # 2. Cascade
        if cascade_mode != "none" and len(positions) >= 2:
            for direction in ["LONG", "SHORT"]:
                dir_pos = [p for p in positions if p["direction"] == direction]
                if len(dir_pos) < 2:
                    continue
                total_loss = 0.0
                for p in dir_pos:
                    if direction == "LONG":
                        unr = (cp / p["entry"] - 1) * 100 * LEVERAGE
                    else:
                        unr = (1 - cp / p["entry"]) * 100 * LEVERAGE
                    if unr < 0:
                        total_loss += abs(unr) / n_slots
                if total_loss < cascade_threshold:
                    continue

                actionable = [p for p in dir_pos if not p.get("cascaded")]
                if not actionable:
                    continue

                if cascade_mode == "partial_close":
                    # Sort by loss (worst first) and close top N
                    actionable.sort(
                        key=lambda p: abs((cp/p["entry"]-1) if direction == "LONG"
                                          else (1-cp/p["entry"])),
                        reverse=True)
                    n_close = max(1, len(dir_pos) * partial_close_pct // 100)
                    for p in actionable[:n_close]:
                        if direction == "LONG":
                            raw = (cp / p["entry"] - 1) * LEVERAGE
                        else:
                            raw = (1 - cp / p["entry"]) * LEVERAGE
                        raw -= FEE_PCT / 100 * LEVERAGE * 2
                        equity *= (1 + raw / n_slots)
                        peak = max(peak, equity)
                        dd = (peak - equity) / peak * 100
                        max_dd = max(max_dd, dd)
                        trades.append(raw * 100)
                        positions.remove(p)

                elif cascade_mode == "baseline":
                    # 98% tighten with 0.4% fallback (current production)
                    for p in actionable:
                        orig_sl = p.get("orig_sl", p["sl"])
                        orig_dist = abs(p["entry"] - orig_sl)
                        new_dist = max(orig_dist * 0.02, 30.0)
                        if direction == "LONG":
                            new_sl = p["entry"] - new_dist
                            if new_sl >= cp:
                                new_sl = cp * 0.996
                            if new_sl <= p["sl"]:
                                continue
                        else:
                            new_sl = p["entry"] + new_dist
                            if new_sl <= cp:
                                new_sl = cp * 1.004
                            if new_sl >= p["sl"]:
                                continue
                        p["sl"] = new_sl
                        p["cascaded"] = True

                elif cascade_mode == "mild":
                    for p in actionable:
                        orig_sl = p.get("orig_sl", p["sl"])
                        orig_dist = abs(p["entry"] - orig_sl)
                        new_dist = orig_dist * mild_keep
                        if direction == "LONG":
                            new_sl = p["entry"] - new_dist
                            if new_sl >= cp or new_sl <= p["sl"]:
                                continue
                        else:
                            new_sl = p["entry"] + new_dist
                            if new_sl <= cp or new_sl >= p["sl"]:
                                continue
                        p["sl"] = new_sl
                        p["cascaded"] = True

        # 3. New entries
        if bar in signals_by_bar:
            for sig in signals_by_bar[bar]:
                if len(positions) >= n_slots:
                    break
                dir_count = sum(1 for p in positions
                               if p["direction"] == sig["direction"])
                if dir_count >= direction_cap:
                    continue
                positions.append({
                    "bar": bar,
                    "direction": sig["direction"],
                    "entry": sig["entry"],
                    "tp": sig["tp"],
                    "sl": sig["sl"],
                    "orig_sl": sig["sl"],
                    "cascaded": False,
                })

    if not trades:
        return {"pnl": 0, "mdd": 0, "trades": 0, "wr": 0, "pm": 0, "rr": 0}

    wins = sum(1 for t in trades if t > 0)
    wr = wins / len(trades) * 100
    total_pnl = (equity - 1) * 100
    pm = total_pnl / max_dd if max_dd > 0 else 999

    win_pnls = [t for t in trades if t > 0]
    loss_pnls = [t for t in trades if t < 0]
    avg_w = np.mean(win_pnls) if win_pnls else 0
    avg_l = np.mean(loss_pnls) if loss_pnls else 0
    rr = abs(avg_w / avg_l) if avg_l != 0 else 999

    return {
        "pnl": round(total_pnl, 1),
        "mdd": round(max_dd, 2),
        "trades": len(trades),
        "wr": round(wr, 1),
        "pm": round(pm, 1),
        "rr": round(rr, 3),
    }


def main():
    print("Loading data...")
    sbb, opens, highs, lows, closes, n, total_sigs = load_data()
    print("  {} signals, {} bars".format(total_sigs, n))

    configs = {
        "A_BASELINE_98pct": dict(direction_cap=6, cascade_mode="baseline", cascade_threshold=1.0),
        "B_NO_CASCADE_CAP6": dict(direction_cap=6, cascade_mode="none"),
        "C_PARTIAL_CLOSE_50": dict(direction_cap=6, cascade_mode="partial_close", cascade_threshold=1.0, partial_close_pct=50),
        "D_CAP3_BASELINE": dict(direction_cap=3, cascade_mode="baseline", cascade_threshold=1.0),
        "E_CAP3_NONE": dict(direction_cap=3, cascade_mode="none"),
        "F_CAP4_NONE": dict(direction_cap=4, cascade_mode="none"),
        "G_MILD_KEEP50": dict(direction_cap=6, cascade_mode="mild", cascade_threshold=1.0, mild_keep=0.5),
        "H_MILD_KEEP30": dict(direction_cap=6, cascade_mode="mild", cascade_threshold=1.0, mild_keep=0.3),
        "I_CAP3_PARTIAL": dict(direction_cap=3, cascade_mode="partial_close", cascade_threshold=1.5, partial_close_pct=50),
        "J_CAP4_PARTIAL": dict(direction_cap=4, cascade_mode="partial_close", cascade_threshold=1.0, partial_close_pct=50),
    }

    # IS run
    header = "{:<24} {:>8} {:>7} {:>7} {:>7} {:>6} {:>6}".format(
        "Config", "PnL%", "MDD%", "P/M", "Trades", "WR%", "R:R")
    print("\n" + "=" * 70)
    print("IN-SAMPLE (full data)")
    print("=" * 70)
    print(header)
    print("-" * 70)

    results = {}
    for name, kw in configs.items():
        t0 = time.time()
        r = simulate(sbb, opens, highs, lows, closes, n, **kw)
        dt = time.time() - t0
        line = "{:<24} {:>8.1f} {:>7.2f} {:>7.1f} {:>7d} {:>6.1f} {:>6.3f}".format(
            name, r["pnl"], r["mdd"], r["pm"], r["trades"], r["wr"], r["rr"])
        print("{} ({:.1f}s)".format(line, dt))
        results[name] = {"IS": r}

    # WF
    print("\n" + "=" * 70)
    print("WALK-FORWARD OOS (5-fold expanding window)")
    print("=" * 70)
    print("{:<24} {:>8} {:>40} {:>5}".format("Config", "OOS Tot", "Folds", "Pass"))
    print("-" * 80)

    for name, kw in configs.items():
        oos_pnls = []
        for fi in range(5):
            is_end = int(n * (fi + 1) / 6)
            oos_end = int(n * (fi + 2) / 6)
            r = simulate(sbb, opens, highs, lows, closes, n,
                        start_bar=is_end, end_bar=oos_end, **kw)
            oos_pnls.append(r["pnl"])
        total = sum(oos_pnls)
        passes = sum(1 for p in oos_pnls if p > 0)
        folds_str = " ".join("{:+6.0f}".format(p) for p in oos_pnls)
        print("{:<24} {:>+8.1f} [{}] {}/5".format(name, total, folds_str, passes))
        results[name]["WF"] = {
            "oos_total": round(total, 1),
            "folds": [round(p, 1) for p in oos_pnls],
            "pass": "{}/5".format(passes),
        }

    # Save
    out_path = ROOT / "results" / "cascade_cluster_fix.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to {}".format(out_path))

    # Summary
    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)
    a = results.get("A_BASELINE_98pct", {}).get("IS", {})
    b = results.get("B_NO_CASCADE_CAP6", {}).get("IS", {})
    e = results.get("E_CAP3_NONE", {}).get("IS", {})
    c = results.get("C_PARTIAL_CLOSE_50", {}).get("IS", {})
    print("  BASELINE (98% cascade): PnL {}, R:R {}".format(a.get("pnl"), a.get("rr")))
    print("  NO CASCADE (cap6):      PnL {}, R:R {}".format(b.get("pnl"), b.get("rr")))
    print("  CAP3 NO CASCADE:        PnL {}, R:R {}".format(e.get("pnl"), e.get("rr")))
    print("  PARTIAL CLOSE 50%:      PnL {}, R:R {}".format(c.get("pnl"), c.get("rr")))


if __name__ == "__main__":
    main()
