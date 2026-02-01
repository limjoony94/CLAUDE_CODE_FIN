"""
Per-Pattern TP/SL Optimization Study
For all 13 production patterns, find optimal TP/SL vs uniform 1.0/1.0.
Uses Standard Research Protocol (next-bar open entry, intrabar TP/SL exit).
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Import canonical classify_candle from production
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.production.pattern_5m.indicators import classify_candle

print("=" * 70)
print("Per-Pattern TP/SL Optimization Study")
print("=" * 70)

# === Production patterns ===
LONG_PATTERNS = [
    "U-MU-H", "MD-ST-MD", "GS-U-BD", "MD-MD-ST",
    "BU-IH-DN", "MD-H-MD", "IH-MD-MD",
]
SHORT_PATTERNS = [
    "DN-D-BD", "BD-U-GS", "DN-GS-H",
    "U-DF-BU", "BD-GS-BD", "DN-IH-IH",
]
ALL_PATTERNS = [(p, "LONG") for p in LONG_PATTERNS] + [(p, "SHORT") for p in SHORT_PATTERNS]

# === Parameters ===
LEVERAGE = 3
FEE_PCT = 0.10  # 0.05% x 2
MAX_BARS = 500
SLIPPAGE = 0.02

# TP/SL grid
TP_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
SL_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]

# === Load data ===
df = pd.read_csv("data/btc_5m_270days.csv")
print(f"Loaded {len(df)} bars")

# === Candle classification (production-consistent) ===
body = df['close'].values - df['open'].values
body_abs = np.abs(body)
range_ = df['high'].values - df['low'].values
upper_wick = df['high'].values - np.maximum(df['open'].values, df['close'].values)
lower_wick = np.minimum(df['open'].values, df['close'].values) - df['low'].values
avg_body_20 = pd.Series(body_abs).rolling(20).mean().values  # NaN for bars 0-19

opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
n_bars = len(df)
types = [classify_candle(opens[i], highs[i], lows[i], closes[i],
                         avg_body_20[i] if not np.isnan(avg_body_20[i]) else 1.0)
         for i in range(n_bars)]

# Build patterns
patterns = [None, None]
for i in range(2, n_bars):
    patterns.append(f"{types[i-2]}-{types[i-1]}-{types[i]}")

# Next bar open for entry
next_open = np.empty(n_bars)
next_open[:] = np.nan
next_open[:-1] = opens[1:]

print("Classification done\n")

# === Backtest function ===
def backtest_pattern(pattern_indices, direction, tp_pct, sl_pct):
    """Backtest with compound equity."""
    equity = 100.0
    trades = 0
    wins = 0
    max_equity = 100.0
    max_dd = 0.0
    results = []

    for idx in pattern_indices:
        entry = next_open[idx]
        if np.isnan(entry):
            continue

        tp_pct_eff = tp_pct
        sl_pct_eff = sl_pct

        if direction == "LONG":
            tp_price = entry * (1 + tp_pct_eff / 100)
            sl_price = entry * (1 - sl_pct_eff / 100)
        else:
            tp_price = entry * (1 - tp_pct_eff / 100)
            sl_price = entry * (1 + sl_pct_eff / 100)

        outcome = None
        for j in range(1, MAX_BARS + 1):
            fi = idx + j
            if fi >= n_bars:
                break
            fh = highs[fi]
            fl = lows[fi]

            if direction == "LONG":
                tp_dist = tp_price - entry
                sl_dist = entry - sl_price
                tp_hit = fh >= tp_price
                sl_hit = fl <= sl_price
            else:
                tp_dist = entry - tp_price
                sl_dist = sl_price - entry
                tp_hit = fl <= tp_price
                sl_hit = fh >= sl_price

            if tp_hit and sl_hit:
                # Both hit — use distance-based resolution
                if tp_dist <= sl_dist:
                    outcome = "WIN"
                else:
                    outcome = "LOSS"
                break
            elif tp_hit:
                outcome = "WIN"
                break
            elif sl_hit:
                outcome = "LOSS"
                break

        if outcome is None:
            continue

        trades += 1
        if outcome == "WIN":
            wins += 1
            pnl_pct = tp_pct_eff * LEVERAGE - FEE_PCT - SLIPPAGE
        else:
            pnl_pct = -(sl_pct_eff * LEVERAGE + FEE_PCT + SLIPPAGE)

        equity *= (1 + pnl_pct / 100)
        max_equity = max(max_equity, equity)
        dd = (max_equity - equity) / max_equity * 100
        max_dd = max(max_dd, dd)
        results.append(outcome)

    if trades == 0:
        return None

    wr = wins / trades * 100
    pnl = equity - 100
    pf = 0
    gross_win = sum(1 for r in results if r == "WIN") * (tp_pct * LEVERAGE - FEE_PCT - SLIPPAGE)
    gross_loss = abs(sum(1 for r in results if r == "LOSS") * (sl_pct * LEVERAGE + FEE_PCT + SLIPPAGE))
    if gross_loss > 0:
        pf = gross_win / gross_loss

    return {
        "trades": trades,
        "wr": wr,
        "pnl": pnl,
        "mdd": max_dd,
        "pf": pf,
    }

def walk_forward(pattern_indices, direction, tp, sl, n_folds=5):
    """Walk-forward validation."""
    fold_size = len(pattern_indices) // n_folds
    if fold_size < 3:
        return 0
    profitable = 0
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(pattern_indices)
        fold_idx = pattern_indices[start:end]
        result = backtest_pattern(fold_idx, direction, tp, sl)
        if result and result["pnl"] > 0:
            profitable += 1
    return profitable

def monte_carlo(pattern_indices, direction, tp, sl, n_sims=5000):
    """Monte Carlo sign randomization."""
    result = backtest_pattern(pattern_indices, direction, tp, sl)
    if not result:
        return 1.0
    actual_pnl = result["pnl"]

    # Get individual trade outcomes
    outcomes = []
    for idx in pattern_indices:
        entry = next_open[idx]
        if np.isnan(entry):
            continue
        if direction == "LONG":
            tp_price = entry * (1 + tp / 100)
            sl_price = entry * (1 - sl / 100)
        else:
            tp_price = entry * (1 - tp / 100)
            sl_price = entry * (1 + sl / 100)

        for j in range(1, MAX_BARS + 1):
            fi = idx + j
            if fi >= n_bars:
                break
            fh, fl = highs[fi], lows[fi]
            if direction == "LONG":
                tp_hit = fh >= tp_price
                sl_hit = fl <= sl_price
            else:
                tp_hit = fl <= tp_price
                sl_hit = fh >= sl_price

            if tp_hit and sl_hit:
                tp_dist = abs(tp_price - entry)
                sl_dist = abs(sl_price - entry)
                outcomes.append(1 if tp_dist <= sl_dist else -1)
                break
            elif tp_hit:
                outcomes.append(1)
                break
            elif sl_hit:
                outcomes.append(-1)
                break

    if not outcomes:
        return 1.0

    outcomes = np.array(outcomes)
    win_pnl = tp * LEVERAGE - FEE_PCT - SLIPPAGE
    loss_pnl = -(sl * LEVERAGE + FEE_PCT + SLIPPAGE)

    count_better = 0
    for _ in range(n_sims):
        signs = np.random.choice([-1, 1], size=len(outcomes))
        shuffled = np.where(signs == 1, win_pnl, loss_pnl)
        sim_eq = 100.0
        for p in shuffled:
            sim_eq *= (1 + p / 100)
        if (sim_eq - 100) >= actual_pnl:
            count_better += 1

    return count_better / n_sims

# === Build pattern index lookup ===
pattern_indices_map = {}
for i in range(n_bars):
    if patterns[i] is not None:
        if patterns[i] not in pattern_indices_map:
            pattern_indices_map[patterns[i]] = []
        pattern_indices_map[patterns[i]].append(i)

# === Run optimization ===
print("=" * 70)
print("Optimizing TP/SL for each pattern...")
print("=" * 70)

results = []

for pattern, direction in ALL_PATTERNS:
    indices = pattern_indices_map.get(pattern, [])
    if len(indices) < 10:
        print(f"\n{pattern} ({direction}): Only {len(indices)} occurrences, skipping")
        continue

    # Baseline: uniform 1.0/1.0
    baseline = backtest_pattern(indices, direction, 1.0, 1.0)
    baseline_wf = walk_forward(indices, direction, 1.0, 1.0)

    # Grid search
    best_score = -999
    best_tp, best_sl = 1.0, 1.0
    best_result = None
    all_combos = []

    for tp in TP_GRID:
        for sl in SL_GRID:
            r = backtest_pattern(indices, direction, tp, sl)
            if r is None or r["trades"] < 10:
                continue

            # Score: balance WR, PnL, and low MDD
            score = r["wr"] * 0.3 + r["pnl"] * 0.4 + (100 - r["mdd"]) * 0.3
            all_combos.append((tp, sl, r, score))

            if score > best_score:
                best_score = score
                best_tp, best_sl = tp, sl
                best_result = r

    if best_result is None:
        continue

    best_wf = walk_forward(indices, direction, best_tp, best_sl)
    best_mc = monte_carlo(indices, direction, best_tp, best_sl)

    baseline_mc = monte_carlo(indices, direction, 1.0, 1.0) if baseline else 1.0

    print(f"\n{'='*60}")
    print(f"  {pattern} ({direction}) — {len(indices)} occurrences")
    print(f"{'='*60}")
    print(f"  {'':20s} {'Uniform 1.0/1.0':>18s}  {'Optimal':>18s}")
    print(f"  {'TP/SL':20s} {'1.0/1.0':>18s}  {f'{best_tp}/{best_sl}':>18s}")
    if baseline:
        print(f"  {'Trades':20s} {baseline['trades']:>18d}  {best_result['trades']:>18d}")
        print(f"  {'WR':20s} {baseline['wr']:>17.1f}%  {best_result['wr']:>17.1f}%")
        print(f"  {'PnL':20s} {baseline['pnl']:>17.1f}%  {best_result['pnl']:>17.1f}%")
        print(f"  {'MDD':20s} {baseline['mdd']:>17.1f}%  {best_result['mdd']:>17.1f}%")
        print(f"  {'PF':20s} {baseline['pf']:>18.2f}  {best_result['pf']:>18.2f}")
        print(f"  {'WF':20s} {baseline_wf:>17d}/5  {best_wf:>17d}/5")
        print(f"  {'MC p-value':20s} {baseline_mc:>18.4f}  {best_mc:>18.4f}")
    else:
        print(f"  Baseline: N/A")
        print(f"  Optimal: {best_result['trades']}t, WR {best_result['wr']:.1f}%, PnL {best_result['pnl']:.1f}%")

    # Top 5 combos
    all_combos.sort(key=lambda x: x[3], reverse=True)
    print(f"\n  Top 5 TP/SL combinations:")
    for rank, (tp, sl, r, sc) in enumerate(all_combos[:5], 1):
        print(f"    [{rank}] TP={tp:.1f} SL={sl:.1f} → {r['trades']}t, WR {r['wr']:.1f}%, PnL {r['pnl']:.1f}%, MDD {r['mdd']:.1f}%")

    improvement = ""
    if baseline and best_result["pnl"] > baseline["pnl"] * 1.1:
        improvement = "⬆️ IMPROVEMENT"
    elif baseline and best_result["pnl"] <= baseline["pnl"]:
        improvement = "➡️ UNIFORM OK"
    else:
        improvement = "↗️ MARGINAL"
    print(f"\n  Verdict: {improvement}")

    results.append({
        "pattern": pattern,
        "direction": direction,
        "count": len(indices),
        "uniform_tp": 1.0, "uniform_sl": 1.0,
        "uniform_trades": baseline["trades"] if baseline else 0,
        "uniform_wr": baseline["wr"] if baseline else 0,
        "uniform_pnl": round(baseline["pnl"], 2) if baseline else 0,
        "uniform_mdd": round(baseline["mdd"], 2) if baseline else 0,
        "uniform_wf": baseline_wf,
        "uniform_mc": round(baseline_mc, 4) if baseline else 1,
        "opt_tp": best_tp, "opt_sl": best_sl,
        "opt_trades": best_result["trades"],
        "opt_wr": round(best_result["wr"], 1),
        "opt_pnl": round(best_result["pnl"], 2),
        "opt_mdd": round(best_result["mdd"], 2),
        "opt_pf": round(best_result["pf"], 2),
        "opt_wf": best_wf,
        "opt_mc": round(best_mc, 4),
    })

# === Summary ===
print("\n" + "=" * 70)
print("SUMMARY: Uniform vs Optimal TP/SL")
print("=" * 70)
print(f"\n{'Pattern':<12s} {'Dir':>5s} {'Cnt':>4s} | {'Uni PnL':>8s} {'Uni WR':>7s} {'Uni WF':>6s} | {'Opt TP/SL':>9s} {'Opt PnL':>8s} {'Opt WR':>7s} {'Opt WF':>6s} {'MC':>7s} | {'Verdict'}")
print("-" * 110)
for r in results:
    verdict = "⬆️ OPT" if r["opt_pnl"] > r["uniform_pnl"] * 1.1 and r["opt_wf"] >= 4 and r["opt_mc"] < 0.05 else "➡️ UNI"
    print(f"{r['pattern']:<12s} {r['direction']:>5s} {r['count']:>4d} | "
          f"{r['uniform_pnl']:>7.1f}% {r['uniform_wr']:>6.1f}% {r['uniform_wf']:>4d}/5 | "
          f"{r['opt_tp']:.1f}/{r['opt_sl']:.1f} {r['opt_pnl']:>7.1f}% {r['opt_wr']:>6.1f}% {r['opt_wf']:>4d}/5 {r['opt_mc']:>7.4f} | "
          f"{verdict}")

# Save
results_df = pd.DataFrame(results)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = f"results/per_pattern_tpsl_optimization_{ts}.csv"
results_df.to_csv(out, index=False)
print(f"\nResults saved to: {out}")
