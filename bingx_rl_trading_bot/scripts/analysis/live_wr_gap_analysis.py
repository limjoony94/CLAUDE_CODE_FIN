"""
Live WR Gap Analysis v2 — Why is live WR 52.7% when WF OOS expects 88.8%?

Separates pattern trades from recovery trades, identifies correlated loss bursts,
and computes adjusted WR excluding contamination.
"""

import json
import math
import os
from datetime import datetime, timedelta
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
METRICS_PATH = os.path.join(PROJECT_ROOT, "results", "pattern_5m_metrics.json")


def load_trades():
    with open(METRICS_PATH) as f:
        data = json.load(f)
    return data.get("trade_history", []), data


def wr(trade_list):
    if not trade_list:
        return 0, 0, 0.0
    wins = sum(1 for t in trade_list if t["pnl_slot"] >= 0)
    return wins, len(trade_list), wins / len(trade_list) * 100


def analyze():
    trades, metrics = load_trades()

    print("=" * 70)
    print("LIVE WR GAP ANALYSIS v2")
    print("=" * 70)
    print(f"Total trades (metrics): {metrics['total_trades']}")
    print(f"Winning (metrics):      {metrics['winning_trades']}")
    print(f"WR (metrics):           {metrics['actual_win_rate']:.1f}%")
    print(f"Total PnL:              {metrics['total_pnl_pct']:.2f}%")
    print(f"Trade history records:  {len(trades)}")
    print()

    # --- 1. Separate pattern vs recovery ---
    pattern_trades = [t for t in trades if t["pattern"] != "N/A"]
    recovery_trades = [t for t in trades if t["pattern"] == "N/A"]

    pw, pn, pwr = wr(pattern_trades)
    rw, rn, rwr = wr(recovery_trades)

    print("-" * 70)
    print("1. PATTERN vs RECOVERY SEPARATION")
    print("-" * 70)
    print(f"Pattern trades:  {pn} ({pw}W / {pn - pw}L) = {pwr:.1f}% WR")
    print(f"Recovery (N/A):  {rn} ({rw}W / {rn - rw}L) = {rwr:.1f}% WR")
    print(f"Recovery PnL:    {sum(t['pnl_portfolio'] for t in recovery_trades):+.2f}%")
    print()

    # --- 2. Direction breakdown ---
    print("-" * 70)
    print("2. DIRECTION BREAKDOWN (pattern trades only)")
    print("-" * 70)

    for direction in ["LONG", "SHORT"]:
        dir_trades = [t for t in pattern_trades if t["direction"] == direction]
        dw, dn, dwr = wr(dir_trades)
        pnl = sum(t["pnl_portfolio"] for t in dir_trades)
        wins_list = [t["pnl_slot"] for t in dir_trades if t["pnl_slot"] >= 0]
        loss_list = [t["pnl_slot"] for t in dir_trades if t["pnl_slot"] < 0]
        avg_win = sum(wins_list) / len(wins_list) if wins_list else 0
        avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0
        print(f"{direction}: {dn} trades ({dw}W / {dn - dw}L) = {dwr:.1f}% WR | "
              f"Portfolio PnL: {pnl:+.2f}% | Avg Win: {avg_win:+.1f}% | Avg Loss: {avg_loss:+.1f}%")
    print()

    # --- 3. Correlated loss burst detection ---
    print("-" * 70)
    print("3. CORRELATED LOSS BURST DETECTION")
    print("-" * 70)
    print("(Losses within 5min of each other + same direction = correlated burst)")
    print()

    losses = [t for t in pattern_trades if t["pnl_slot"] < 0]
    losses.sort(key=lambda t: t["timestamp"])

    bursts = []
    current_burst = []

    for loss in losses:
        ts = datetime.fromisoformat(loss["timestamp"])
        if not current_burst:
            current_burst = [loss]
        else:
            prev_ts = datetime.fromisoformat(current_burst[-1]["timestamp"])
            if ((ts - prev_ts) < timedelta(minutes=5)
                    and loss["direction"] == current_burst[0]["direction"]):
                current_burst.append(loss)
            else:
                if len(current_burst) >= 2:
                    bursts.append(current_burst)
                current_burst = [loss]
    if len(current_burst) >= 2:
        bursts.append(current_burst)

    correlated_loss_timestamps = set()
    total_correlated_pnl = 0

    for i, burst in enumerate(bursts):
        ts = burst[0]["timestamp"][:16]
        direction = burst[0]["direction"]
        exit_price = burst[0]["exit_price"]
        total_pnl = sum(t["pnl_portfolio"] for t in burst)
        total_correlated_pnl += total_pnl

        print(f"  Burst #{i + 1}: {ts} | {direction} x{len(burst)} | "
              f"Exit ~${exit_price:.0f} | Portfolio PnL: {total_pnl:+.2f}%")
        for t in burst:
            print(f"    - {t['pattern']}: ${t['entry_price']:.0f} -> "
                  f"${t['exit_price']:.0f} | slot {t['pnl_slot']:+.1f}% | {t['exit_reason']}")
            correlated_loss_timestamps.add(t["timestamp"])

    total_correlated_count = len(correlated_loss_timestamps)
    excess_losses = sum(len(b) - 1 for b in bursts)

    print(f"\n  Total burst losses:     {total_correlated_count} trades in {len(bursts)} bursts")
    print(f"  Excess losses:          {excess_losses} (each burst = 1 independent event)")
    print(f"  Correlated PnL impact:  {total_correlated_pnl:+.2f}%")
    print()

    # --- 4. Adjusted WR ---
    print("-" * 70)
    print("4. ADJUSTED WR ANALYSIS")
    print("-" * 70)

    # Method: Remove excess correlated losses (keep 1 per burst)
    adj_wins = pw
    adj_total = pn - excess_losses
    adj_wr = adj_wins / adj_total * 100 if adj_total > 0 else 0

    print(f"Pattern WR (raw):              {pwr:.1f}% ({pw}W / {pn - pw}L = {pn} trades)")
    print(f"Excess correlated losses:      -{excess_losses}")
    print(f"Pattern WR (burst-adjusted):   {adj_wr:.1f}% ({adj_wins}W / {adj_total - adj_wins}L = {adj_total} trades)")
    print()

    # --- 5. Per-date breakdown ---
    print("-" * 70)
    print("5. PER-DATE PERFORMANCE (pattern trades)")
    print("-" * 70)

    by_date = defaultdict(list)
    for t in pattern_trades:
        date = t["timestamp"][:10]
        by_date[date].append(t)

    for date in sorted(by_date.keys()):
        day_trades = by_date[date]
        w, n, wr_ = wr(day_trades)
        pnl = sum(t["pnl_portfolio"] for t in day_trades)
        directions = defaultdict(int)
        for t in day_trades:
            directions[t["direction"]] += 1
        dir_str = " ".join(f"{d[0]}{v}" for d, v in sorted(directions.items()))
        print(f"  {date}: {n:2d} trades ({w}W/{n - w}L) = {wr_:5.1f}% WR | "
              f"PnL: {pnl:+6.2f}% | {dir_str}")
    print()

    # --- 6. Exit reason breakdown ---
    print("-" * 70)
    print("6. EXIT REASON BREAKDOWN (pattern trades)")
    print("-" * 70)

    by_reason = defaultdict(list)
    for t in pattern_trades:
        by_reason[t["exit_reason"]].append(t)

    for reason in sorted(by_reason.keys()):
        rtrades = by_reason[reason]
        w, n, wr_ = wr(rtrades)
        pnl = sum(t["pnl_portfolio"] for t in rtrades)
        avg_pnl_slot = sum(t["pnl_slot"] for t in rtrades) / len(rtrades)
        print(f"  {reason:15s}: {n:2d} trades ({w}W/{n - w}L) | "
              f"Avg slot PnL: {avg_pnl_slot:+.1f}% | Portfolio: {pnl:+.2f}%")
    print()

    # --- 7. Statistical test ---
    print("-" * 70)
    print("7. STATISTICAL SIGNIFICANCE")
    print("-" * 70)

    expected_wr = 0.888
    observed_pattern = pwr / 100
    se = math.sqrt(expected_wr * (1 - expected_wr) / pn)
    z = (observed_pattern - expected_wr) / se

    print(f"Expected WR (WF OOS):   {expected_wr * 100:.1f}%")
    print(f"Observed WR (pattern):  {pwr:.1f}% (n={pn})")
    print(f"Z-score:                {z:.1f}")
    print(f"Interpretation:         {'*** SIGNIFICANT SHORTFALL ***' if z < -3 else 'Within noise'}")
    print()

    se_adj = math.sqrt(expected_wr * (1 - expected_wr) / adj_total)
    z_adj = (adj_wr / 100 - expected_wr) / se_adj
    print(f"Burst-adjusted WR:      {adj_wr:.1f}% (n={adj_total})")
    print(f"Z-score (adjusted):     {z_adj:.1f}")
    print(f"Interpretation:         {'*** SIGNIFICANT SHORTFALL ***' if z_adj < -3 else 'Within noise'}")
    print()

    # --- 8. R:R and breakeven analysis ---
    print("-" * 70)
    print("8. R:R AND BREAKEVEN ANALYSIS")
    print("-" * 70)

    all_wins = [t["pnl_slot"] for t in pattern_trades if t["pnl_slot"] >= 0]
    all_losses = [abs(t["pnl_slot"]) for t in pattern_trades if t["pnl_slot"] < 0]
    avg_w = sum(all_wins) / len(all_wins) if all_wins else 0
    avg_l = sum(all_losses) / len(all_losses) if all_losses else 0
    rr = avg_w / avg_l if avg_l > 0 else 0
    breakeven_wr = avg_l / (avg_w + avg_l) * 100 if (avg_w + avg_l) > 0 else 50

    print(f"Avg win (slot):   +{avg_w:.2f}%")
    print(f"Avg loss (slot):  -{avg_l:.2f}%")
    print(f"R:R ratio:        {rr:.3f}")
    print(f"Breakeven WR:     {breakeven_wr:.1f}%")
    print(f"Current WR:       {pwr:.1f}%  {'>>> PROFITABLE' if pwr > breakeven_wr else '>>> UNPROFITABLE'}")
    print()

    # --- 9. PnL contribution ---
    print("-" * 70)
    print("9. PNL CONTRIBUTION")
    print("-" * 70)

    pattern_pnl = sum(t["pnl_portfolio"] for t in pattern_trades)
    recovery_pnl = sum(t["pnl_portfolio"] for t in recovery_trades)

    print(f"Pattern PnL:   {pattern_pnl:+.2f}%")
    print(f"Recovery PnL:  {recovery_pnl:+.2f}%")
    print(f"Total PnL:     {pattern_pnl + recovery_pnl:+.2f}%")
    print()

    # --- 10. ATR compression check ---
    print("-" * 70)
    print("10. TP/SL DISTANCE ANALYSIS (proxy for ATR compression)")
    print("-" * 70)

    for direction in ["LONG", "SHORT"]:
        dir_trades = [t for t in pattern_trades if t["direction"] == direction]
        if not dir_trades:
            continue
        tp_dists = []
        sl_dists = []
        for t in dir_trades:
            entry = t["entry_price"]
            tp = t["tp_price"]
            sl = t["sl_price"]
            if direction == "LONG":
                tp_dists.append((tp - entry) / entry * 100)
                sl_dists.append((entry - sl) / entry * 100)
            else:
                tp_dists.append((entry - tp) / entry * 100)
                sl_dists.append((sl - entry) / entry * 100)

        avg_tp = sum(tp_dists) / len(tp_dists)
        avg_sl = sum(sl_dists) / len(sl_dists)
        rr_live = avg_tp / avg_sl if avg_sl > 0 else 0
        print(f"  {direction}: Avg TP dist {avg_tp:.2f}% | Avg SL dist {avg_sl:.2f}% | "
              f"TP/SL ratio {rr_live:.3f}")
        print(f"    Expected (backtest): TP 0.91-2.76% | SL 1.44-4.84%")
    print()

    # --- FINAL VERDICT ---
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print()
    print(f"1. Pattern trades WR: {pwr:.1f}% (expected 88.8%, gap = {pwr - 88.8:+.1f}pp)")
    print(f"2. Burst-adjusted WR: {adj_wr:.1f}% (still {adj_wr - 88.8:+.1f}pp from expected)")
    print(f"3. R:R = {rr:.3f}, Breakeven WR = {breakeven_wr:.1f}%")
    print(f"4. Current WR {'above' if pwr > breakeven_wr else 'BELOW'} breakeven")
    print()
    print("The gap between backtest WR (88.8%) and live WR is too large")
    print("to be explained by correlated losses alone. Possible root causes:")
    print("  A. ATR compression in low-vol regime changes effective TP/SL dynamics")
    print("  B. Pattern edge may have decayed since discovery date")
    print("  C. Backtest conditions (neutral window) differ from live market regime")
    print("  D. Small sample size (high variance) — need 200+ trades for reliable WR")


if __name__ == "__main__":
    analyze()
