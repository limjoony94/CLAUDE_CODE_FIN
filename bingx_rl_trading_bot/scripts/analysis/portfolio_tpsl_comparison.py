"""
Portfolio Backtest: Per-Pattern Optimal TP/SL vs Uniform 1.0/1.0
Compound equity, Standard Research Protocol.
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("Portfolio Backtest: Per-Pattern vs Uniform TP/SL")
print("=" * 70)

# === Pattern configs ===
# Per-pattern optimal (from optimization study, WF>=4 & MC<0.05 only)
OPTIMAL_CONFIG = {
    # LONG
    "U-MU-H":   ("LONG",  1.5, 1.5),
    "MD-ST-MD": ("LONG",  2.0, 2.0),
    "GS-U-BD":  ("LONG",  2.0, 0.8),  # MC=0.0372, borderline
    "MD-MD-ST": ("LONG",  1.5, 2.0),
    "BU-IH-DN": ("LONG",  1.5, 2.0),
    "MD-H-MD":  ("LONG",  1.0, 1.0),  # uniform best
    "IH-MD-MD": ("LONG",  1.5, 2.0),
    # SHORT
    "DN-D-BD":  ("SHORT", 1.0, 1.0),  # MC=0.24, keep uniform
    "BD-U-GS":  ("SHORT", 1.5, 2.0),
    "DN-GS-H":  ("SHORT", 2.0, 1.5),
    "U-DF-BU":  ("SHORT", 1.0, 1.5),
    "BD-GS-BD": ("SHORT", 1.0, 1.0),  # uniform best
    "DN-IH-IH": ("SHORT", 1.0, 1.5),
}

UNIFORM_CONFIG = {k: (v[0], 1.0, 1.0) for k, v in OPTIMAL_CONFIG.items()}

# Conservative optimal: only patterns where MC < 0.01
CONSERVATIVE_CONFIG = {
    "U-MU-H":   ("LONG",  1.5, 1.5),   # MC=0.0000
    "MD-ST-MD": ("LONG",  2.0, 2.0),   # MC=0.0078
    "GS-U-BD":  ("LONG",  1.0, 1.0),   # MC=0.0372 → keep uniform
    "MD-MD-ST": ("LONG",  1.5, 2.0),   # MC=0.0002
    "BU-IH-DN": ("LONG",  1.5, 2.0),   # MC=0.0022
    "MD-H-MD":  ("LONG",  1.0, 1.0),   # MC=0.0014
    "IH-MD-MD": ("LONG",  1.5, 2.0),   # MC=0.0020
    "DN-D-BD":  ("SHORT", 1.0, 1.0),   # MC=0.2390 → keep uniform
    "BD-U-GS":  ("SHORT", 1.5, 2.0),   # MC=0.0042
    "DN-GS-H":  ("SHORT", 2.0, 1.5),   # MC=0.0176 → keep uniform
    "U-DF-BU":  ("SHORT", 1.0, 1.5),   # MC=0.0010
    "BD-GS-BD": ("SHORT", 1.0, 1.0),   # MC=0.0120 → keep uniform
    "DN-IH-IH": ("SHORT", 1.0, 1.5),   # MC=0.0000
}

LEVERAGE = 3
FEE_PCT = 0.10
SLIPPAGE = 0.02
MAX_BARS = 500

# === Load & classify ===
df = pd.read_csv("data/btc_5m_270days.csv")
print(f"Loaded {len(df)} bars")

opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
n_bars = len(df)

body = closes - opens
body_abs = np.abs(body)
range_ = highs - lows
upper_wick = highs - np.maximum(opens, closes)
lower_wick = np.minimum(opens, closes) - lows
avg_body_20 = pd.Series(body_abs).rolling(20).mean().values

def classify_candle(o, h, l, c, avg_b20):
    b = c - o
    b_abs = abs(b)
    r = h - l
    if r < 1e-10:
        return "D"
    uw = h - max(o, c)
    lw = min(o, c) - l
    body_ratio = b_abs / r
    upper_ratio = uw / r
    lower_ratio = lw / r
    wick_ratio = (uw + lw) / r
    norm_body = b_abs / avg_b20 if avg_b20 > 0 else 1.0
    if body_ratio < 0.10:
        if lower_ratio > 0.70: return "DF"
        elif upper_ratio > 0.70: return "GS"
        return "D"
    if wick_ratio < 0.15:
        return "MU" if b > 0 else "MD"
    if b_abs > 0:
        if lw > 2.0 * b_abs and uw < 0.3 * b_abs: return "H"
        if uw > 2.0 * b_abs and lw < 0.3 * b_abs: return "IH"
    if norm_body < 0.5 and uw > 0.5 * b_abs and lw > 0.5 * b_abs: return "ST"
    if norm_body > 1.5: return "BU" if b > 0 else "BD"
    return "U" if b > 0 else "DN"

types = [classify_candle(opens[i], highs[i], lows[i], closes[i],
                         avg_body_20[i] if not np.isnan(avg_body_20[i]) else 1.0)
         for i in range(n_bars)]

patterns = [None, None]
for i in range(2, n_bars):
    patterns.append(f"{types[i-2]}-{types[i-1]}-{types[i]}")

next_open = np.empty(n_bars)
next_open[:] = np.nan
next_open[:-1] = opens[1:]

print("Classification done\n")

# === Portfolio backtest ===
def portfolio_backtest(config, label=""):
    """Run portfolio backtest with compound equity."""
    # Collect all trade events (bar_index, pattern, direction, tp, sl)
    events = []
    for pat, (direction, tp, sl) in config.items():
        for i in range(2, n_bars):
            if patterns[i] == pat and not np.isnan(next_open[i]):
                events.append((i, pat, direction, tp, sl))

    events.sort(key=lambda x: x[0])

    equity = 100.0
    trades = 0
    wins = 0
    max_equity = 100.0
    max_dd = 0.0
    trade_log = []

    # Track periods
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp']).values
    else:
        timestamps = pd.to_datetime(df['date']).values if 'date' in df.columns else None

    period_pnl = {"H1": 0, "H2": 0, "H3": 0}
    period_trades = {"H1": 0, "H2": 0, "H3": 0}
    third = n_bars // 3

    for bar_idx, pat, direction, tp, sl in events:
        entry = next_open[bar_idx]

        if direction == "LONG":
            tp_price = entry * (1 + tp / 100)
            sl_price = entry * (1 - sl / 100)
        else:
            tp_price = entry * (1 - tp / 100)
            sl_price = entry * (1 + sl / 100)

        outcome = None
        for j in range(1, MAX_BARS + 1):
            fi = bar_idx + j
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
                outcome = "WIN" if tp_dist <= sl_dist else "LOSS"
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
            pnl_pct = tp * LEVERAGE - FEE_PCT - SLIPPAGE
        else:
            pnl_pct = -(sl * LEVERAGE + FEE_PCT + SLIPPAGE)

        pre_eq = equity
        equity *= (1 + pnl_pct / 100)
        max_equity = max(max_equity, equity)
        dd = (max_equity - equity) / max_equity * 100
        max_dd = max(max_dd, dd)

        # Period tracking
        if bar_idx < third:
            period_pnl["H1"] += pnl_pct
            period_trades["H1"] += 1
        elif bar_idx < 2 * third:
            period_pnl["H2"] += pnl_pct
            period_trades["H2"] += 1
        else:
            period_pnl["H3"] += pnl_pct
            period_trades["H3"] += 1

        trade_log.append({
            "bar": bar_idx, "pattern": pat, "direction": direction,
            "tp": tp, "sl": sl, "outcome": outcome,
            "pnl_pct": round(pnl_pct, 2), "equity": round(equity, 2),
        })

    if trades == 0:
        return None

    wr = wins / trades * 100
    total_pnl = equity - 100

    gross_win = sum(t["pnl_pct"] for t in trade_log if t["outcome"] == "WIN")
    gross_loss = abs(sum(t["pnl_pct"] for t in trade_log if t["outcome"] == "LOSS"))
    pf = gross_win / gross_loss if gross_loss > 0 else 999

    periods_profitable = sum(1 for v in period_pnl.values() if v > 0)

    return {
        "label": label,
        "trades": trades,
        "wins": wins,
        "wr": round(wr, 1),
        "pnl": round(total_pnl, 1),
        "mdd": round(max_dd, 1),
        "pf": round(pf, 2),
        "pnl_mdd": round(total_pnl / max_dd, 1) if max_dd > 0 else 999,
        "periods": periods_profitable,
        "period_pnl": {k: round(v, 1) for k, v in period_pnl.items()},
        "period_trades": period_trades,
        "trade_log": trade_log,
    }

def walk_forward_portfolio(config, n_folds=5):
    """Walk-forward for portfolio."""
    fold_size = n_bars // n_folds
    profitable = 0

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n_bars

        events = []
        for pat, (direction, tp, sl) in config.items():
            for i in range(max(2, start), end):
                if patterns[i] == pat and not np.isnan(next_open[i]):
                    events.append((i, pat, direction, tp, sl))
        events.sort(key=lambda x: x[0])

        eq = 100.0
        for bar_idx, pat, direction, tp, sl in events:
            entry = next_open[bar_idx]
            if direction == "LONG":
                tp_price = entry * (1 + tp / 100)
                sl_price = entry * (1 - sl / 100)
            else:
                tp_price = entry * (1 - tp / 100)
                sl_price = entry * (1 + sl / 100)

            outcome = None
            for j in range(1, MAX_BARS + 1):
                fi = bar_idx + j
                if fi >= n_bars: break
                fh, fl = highs[fi], lows[fi]
                if direction == "LONG":
                    tp_hit = fh >= tp_price
                    sl_hit = fl <= sl_price
                else:
                    tp_hit = fl <= tp_price
                    sl_hit = fh >= sl_price

                if tp_hit and sl_hit:
                    outcome = "WIN" if abs(tp_price - entry) <= abs(sl_price - entry) else "LOSS"
                    break
                elif tp_hit: outcome = "WIN"; break
                elif sl_hit: outcome = "LOSS"; break

            if outcome is None: continue
            if outcome == "WIN":
                eq *= (1 + (tp * LEVERAGE - FEE_PCT - SLIPPAGE) / 100)
            else:
                eq *= (1 + (-(sl * LEVERAGE + FEE_PCT + SLIPPAGE)) / 100)

        if eq > 100:
            profitable += 1

    return profitable

def monte_carlo_portfolio(config, n_sims=10000):
    """Monte Carlo for portfolio."""
    events = []
    for pat, (direction, tp, sl) in config.items():
        for i in range(2, n_bars):
            if patterns[i] == pat and not np.isnan(next_open[i]):
                events.append((i, pat, direction, tp, sl))
    events.sort(key=lambda x: x[0])

    # Get outcomes
    outcomes = []
    for bar_idx, pat, direction, tp, sl in events:
        entry = next_open[bar_idx]
        if direction == "LONG":
            tp_price = entry * (1 + tp / 100)
            sl_price = entry * (1 - sl / 100)
        else:
            tp_price = entry * (1 - tp / 100)
            sl_price = entry * (1 + sl / 100)

        for j in range(1, MAX_BARS + 1):
            fi = bar_idx + j
            if fi >= n_bars: break
            fh, fl = highs[fi], lows[fi]
            if direction == "LONG":
                tp_hit = fh >= tp_price; sl_hit = fl <= sl_price
            else:
                tp_hit = fl <= tp_price; sl_hit = fh >= sl_price

            if tp_hit and sl_hit:
                outcomes.append((tp, sl, 1 if abs(tp_price-entry) <= abs(sl_price-entry) else -1))
                break
            elif tp_hit: outcomes.append((tp, sl, 1)); break
            elif sl_hit: outcomes.append((tp, sl, -1)); break

    if not outcomes:
        return 1.0

    # Actual PnL
    eq = 100.0
    for tp, sl, w in outcomes:
        if w == 1: eq *= (1 + (tp * LEVERAGE - FEE_PCT - SLIPPAGE) / 100)
        else: eq *= (1 + (-(sl * LEVERAGE + FEE_PCT + SLIPPAGE)) / 100)
    actual_pnl = eq - 100

    # Simulations
    count_better = 0
    for _ in range(n_sims):
        eq = 100.0
        for tp, sl, w in outcomes:
            sign = np.random.choice([-1, 1])
            if sign == 1: eq *= (1 + (tp * LEVERAGE - FEE_PCT - SLIPPAGE) / 100)
            else: eq *= (1 + (-(sl * LEVERAGE + FEE_PCT + SLIPPAGE)) / 100)
        if (eq - 100) >= actual_pnl:
            count_better += 1

    return count_better / n_sims

# === Run all three configs ===
configs = [
    ("A: Uniform 1.0/1.0", UNIFORM_CONFIG),
    ("B: Per-Pattern Optimal", OPTIMAL_CONFIG),
    ("C: Conservative Optimal", CONSERVATIVE_CONFIG),
]

results = []
for label, config in configs:
    print(f"\nRunning {label}...")
    r = portfolio_backtest(config, label)
    wf = walk_forward_portfolio(config)
    mc = monte_carlo_portfolio(config)
    r["wf"] = wf
    r["mc"] = round(mc, 4)
    results.append(r)
    print(f"  {r['trades']}t, WR {r['wr']}%, PnL {r['pnl']}%, MDD {r['mdd']}%, PF {r['pf']}, WF {wf}/5, MC {mc:.4f}")

# === Display comparison ===
print("\n" + "=" * 70)
print("PORTFOLIO COMPARISON")
print("=" * 70)

header = f"{'Metric':<25s}"
for r in results:
    header += f" {r['label']:>22s}"
print(header)
print("-" * 95)

metrics = [
    ("Trades", "trades", "d"),
    ("Win Rate", "wr", ".1f"),
    ("Total PnL", "pnl", ".1f"),
    ("Max Drawdown", "mdd", ".1f"),
    ("PnL/MDD Ratio", "pnl_mdd", ".1f"),
    ("Profit Factor", "pf", ".2f"),
    ("Walk-Forward", "wf", "d"),
    ("MC p-value", "mc", ".4f"),
    ("Period Profit", "periods", "d"),
]

for name, key, fmt in metrics:
    row = f"{name:<25s}"
    for r in results:
        val = r[key]
        if key in ("pnl", "mdd", "wr"):
            row += f" {val:>21{fmt}}%"
        elif key == "wf":
            row += f" {val:>20d}/5"
        elif key == "periods":
            row += f" {val:>20d}/3"
        else:
            row += f" {val:>22{fmt}}"
    print(row)

# Period detail
print(f"\n{'Period Detail':<25s}")
for period in ["H1", "H2", "H3"]:
    row = f"  {period}:<23s"
    for r in results:
        pnl = r["period_pnl"][period]
        tc = r["period_trades"][period]
        row += f" {pnl:>+8.1f}% ({tc:>3d}t)        "
    # Simpler formatting
    parts = []
    for r in results:
        pnl = r["period_pnl"][period]
        tc = r["period_trades"][period]
        parts.append(f"{pnl:>+8.1f}% ({tc:>3d}t)")
    print(f"  {period:<23s} {'   '.join(parts)}")

# TP/SL config detail
print(f"\n{'='*70}")
print("TP/SL Configuration Detail")
print("="*70)
print(f"{'Pattern':<12s} {'Dir':>5s} | {'Uniform':>9s} | {'Optimal':>9s} | {'Conservative':>12s}")
print("-" * 60)
for pat in OPTIMAL_CONFIG:
    d = OPTIMAL_CONFIG[pat][0]
    u = f"{UNIFORM_CONFIG[pat][1]}/{UNIFORM_CONFIG[pat][2]}"
    o = f"{OPTIMAL_CONFIG[pat][1]}/{OPTIMAL_CONFIG[pat][2]}"
    c = f"{CONSERVATIVE_CONFIG[pat][1]}/{CONSERVATIVE_CONFIG[pat][2]}"
    changed = "***" if o != u else ""
    print(f"{pat:<12s} {d:>5s} | {u:>9s} | {o:>9s} | {c:>12s} {changed}")

# Save
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = f"results/portfolio_tpsl_comparison_{ts}.csv"
summary = []
for r in results:
    summary.append({k: v for k, v in r.items() if k != "trade_log"})
pd.DataFrame(summary).to_csv(out, index=False)
print(f"\nResults saved to: {out}")
