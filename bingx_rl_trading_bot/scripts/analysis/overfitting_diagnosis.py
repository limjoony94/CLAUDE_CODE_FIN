"""
Overfitting Diagnosis for Pattern 5m Strategy
Tests whether 270-day pattern selection is overfit or has genuine edge.

4 Tests:
  1. True Train/Test Split (180d train, 90d test)
  2. Random Pattern Baseline (1000 random portfolios)
  3. Bonferroni Multiple Testing Correction
  4. Time-Decay Analysis (6 periods)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import random

print("=" * 70)
print("OVERFITTING DIAGNOSIS - Pattern 5m Strategy")
print("=" * 70)

# === Production patterns (v1.21.0 Conservative) ===
PRODUCTION_PATTERNS = {
    "U-MU-H":   ("LONG",  1.5, 1.5),
    "MD-ST-MD": ("LONG",  2.0, 2.0),
    "GS-U-BD":  ("LONG",  1.0, 1.0),
    "MD-MD-ST": ("LONG",  1.5, 2.0),
    "BU-IH-DN": ("LONG",  1.5, 2.0),
    "MD-H-MD":  ("LONG",  1.0, 1.0),
    "IH-MD-MD": ("LONG",  1.5, 2.0),
    "DN-D-BD":  ("SHORT", 1.0, 1.0),
    "BD-U-GS":  ("SHORT", 1.5, 2.0),
    "DN-GS-H":  ("SHORT", 1.0, 1.0),
    "U-DF-BU":  ("SHORT", 1.0, 1.5),
    "BD-GS-BD": ("SHORT", 1.0, 1.0),
    "DN-IH-IH": ("SHORT", 1.0, 1.5),
}

# === Parameters ===
LEVERAGE = 3
FEE_PCT = 0.10
MAX_BARS = 500
MIN_TRADES_DISCOVERY = 15
MIN_WR_DISCOVERY = 60.0
MIN_WF_DISCOVERY = 4

# Discovery TP/SL grid (reduced for speed)
TP_GRID = [0.5, 0.8, 1.0, 1.5, 2.0]
SL_GRID = [0.5, 0.8, 1.0, 1.5, 2.0]

# === Load and classify ===
df = pd.read_csv("data/btc_5m_270days.csv")
print(f"Loaded {len(df)} bars ({len(df)/288:.0f} days)")

body = df['close'].values - df['open'].values
body_abs = np.abs(body)
range_ = df['high'].values - df['low'].values
upper_wick = df['high'].values - np.maximum(df['open'].values, df['close'].values)
lower_wick = np.minimum(df['open'].values, df['close'].values) - df['low'].values
avg_body_20 = pd.Series(body_abs).rolling(20).mean().values

opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
n_bars = len(df)


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
        if lower_ratio > 0.70:
            return "DF"
        elif upper_ratio > 0.70:
            return "GS"
        return "D"
    if wick_ratio < 0.15:
        return "MU" if b > 0 else "MD"
    if b_abs > 0:
        if lw > 2.0 * b_abs and uw < 0.3 * b_abs:
            return "H"
        if uw > 2.0 * b_abs and lw < 0.3 * b_abs:
            return "IH"
    if norm_body < 0.5 and uw > 0.5 * b_abs and lw > 0.5 * b_abs:
        return "ST"
    if norm_body > 1.5:
        return "BU" if b > 0 else "BD"
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


# === Core backtest function ===
def backtest_indices(indices, direction, tp_pct, sl_pct):
    """Returns (trades, wins, pnl_pct_list, equity_curve)."""
    equity = 100.0
    results = []
    eq_curve = [100.0]

    for idx in indices:
        entry = next_open[idx]
        if np.isnan(entry):
            continue

        if direction == "LONG":
            tp_price = entry * (1 + tp_pct / 100)
            sl_price = entry * (1 - sl_pct / 100)
        else:
            tp_price = entry * (1 - tp_pct / 100)
            sl_price = entry * (1 + sl_pct / 100)

        outcome = None
        for j in range(1, MAX_BARS + 1):
            fi = idx + j
            if fi >= n_bars:
                break
            fh, fl = highs[fi], lows[fi]

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
                outcome = "WIN" if tp_dist <= sl_dist else "LOSS"
                break
            elif tp_hit:
                outcome = "WIN"
                break
            elif sl_hit:
                outcome = "LOSS"
                break

        if outcome:
            if outcome == "WIN":
                pnl = tp_pct * LEVERAGE - FEE_PCT
            else:
                pnl = -(sl_pct * LEVERAGE + FEE_PCT)
            equity *= (1 + pnl / 100)
            results.append((outcome, pnl))
            eq_curve.append(equity)

    trades = len(results)
    wins = sum(1 for r in results if r[0] == "WIN")
    pnl_list = [r[1] for r in results]
    return trades, wins, pnl_list, eq_curve


def portfolio_backtest(pattern_config, start_idx=0, end_idx=None):
    """Portfolio backtest with compound equity for given patterns."""
    if end_idx is None:
        end_idx = n_bars

    # Collect all trades with timestamps
    all_trades = []
    for pat, (direction, tp, sl) in pattern_config.items():
        indices = [i for i in range(max(2, start_idx), end_idx)
                   if patterns[i] == pat]
        for idx in indices:
            entry = next_open[idx]
            if np.isnan(entry):
                continue

            if direction == "LONG":
                tp_price = entry * (1 + tp / 100)
                sl_price = entry * (1 - sl / 100)
            else:
                tp_price = entry * (1 - tp / 100)
                sl_price = entry * (1 + sl / 100)

            outcome = None
            exit_bar = idx
            for j in range(1, MAX_BARS + 1):
                fi = idx + j
                if fi >= end_idx:
                    break
                fh, fl = highs[fi], lows[fi]

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
                    outcome = "WIN" if tp_dist <= sl_dist else "LOSS"
                    exit_bar = fi
                    break
                elif tp_hit:
                    outcome = "WIN"
                    exit_bar = fi
                    break
                elif sl_hit:
                    outcome = "LOSS"
                    exit_bar = fi
                    break

            if outcome:
                pnl = (tp * LEVERAGE - FEE_PCT) if outcome == "WIN" else -(sl * LEVERAGE + FEE_PCT)
                all_trades.append((idx, outcome, pnl))

    # Sort by signal bar index (time order)
    all_trades.sort(key=lambda x: x[0])

    trades = len(all_trades)
    if trades == 0:
        return 0, 0, 0.0, 0.0, 0.0

    wins = sum(1 for t in all_trades if t[1] == "WIN")
    wr = wins / trades * 100

    # Compound equity
    equity = 100.0
    max_eq = 100.0
    max_dd = 0.0
    for _, _, pnl in all_trades:
        equity *= (1 + pnl / 100)
        if equity > max_eq:
            max_eq = equity
        dd = (max_eq - equity) / max_eq * 100
        if dd > max_dd:
            max_dd = dd

    total_pnl = equity - 100.0
    gross_win = sum(t[2] for t in all_trades if t[1] == "WIN")
    gross_loss = abs(sum(t[2] for t in all_trades if t[1] == "LOSS"))
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0

    return trades, wr, total_pnl, max_dd, pf


# =========================================================
# TEST 1: TRUE TRAIN/TEST SPLIT
# =========================================================
print("=" * 70)
print("[TEST 1] TRUE TRAIN/TEST SPLIT (180d train / 90d test)")
print("=" * 70)

# Split at 2/3 point
split_idx = int(n_bars * 2 / 3)
print(f"Split at bar {split_idx} ({split_idx/288:.0f} days)")
print(f"Train: bars 0-{split_idx-1}, Test: bars {split_idx}-{n_bars-1}")

# --- Discover patterns on TRAIN only ---
print("\n--- Pattern Discovery on TRAIN data only ---")
train_pattern_counts = {}
for i in range(2, split_idx):
    p = patterns[i]
    if p:
        train_pattern_counts[p] = train_pattern_counts.get(p, 0) + 1

discovered = {}
for direction in ["LONG", "SHORT"]:
    for pat, count in train_pattern_counts.items():
        if count < MIN_TRADES_DISCOVERY:
            continue
        indices = [i for i in range(2, split_idx) if patterns[i] == pat]

        best_score = -999
        best_tp, best_sl, best_wr, best_trades = 1.0, 1.0, 0, 0

        for tp in TP_GRID:
            for sl in SL_GRID:
                trades, wins, _, _ = backtest_indices(indices, direction, tp, sl)
                if trades < MIN_TRADES_DISCOVERY:
                    continue
                wr = wins / trades * 100
                if wr < MIN_WR_DISCOVERY:
                    continue
                pnl_per = wins * (tp * LEVERAGE - FEE_PCT) + (trades - wins) * (-(sl * LEVERAGE + FEE_PCT))
                edge = pnl_per / trades
                score = wr * 0.5 + edge * 30
                if score > best_score:
                    best_score = score
                    best_tp, best_sl = tp, sl
                    best_wr = wr
                    best_trades = trades

        if best_trades >= MIN_TRADES_DISCOVERY and best_wr >= MIN_WR_DISCOVERY:
            # Walk-forward on train data (5-fold)
            fold_size = len(indices) // 5
            if fold_size >= 3:
                wf_pass = 0
                for f in range(5):
                    s = f * fold_size
                    e = s + fold_size if f < 4 else len(indices)
                    fold_idx = indices[s:e]
                    ft, fw, _, _ = backtest_indices(fold_idx, direction, best_tp, best_sl)
                    if ft > 0:
                        fold_pnl = fw * (best_tp * LEVERAGE - FEE_PCT) + (ft - fw) * (-(best_sl * LEVERAGE + FEE_PCT))
                        if fold_pnl > 0:
                            wf_pass += 1
                if wf_pass >= MIN_WF_DISCOVERY:
                    key = (pat, direction)
                    if key not in discovered or best_score > discovered[key][0]:
                        discovered[key] = (best_score, best_tp, best_sl, best_trades, best_wr, wf_pass)

# Rank and pick top 13
ranked = sorted(discovered.items(), key=lambda x: x[1][0], reverse=True)[:13]
train_selected = {}
for (pat, direction), (score, tp, sl, trades, wr, wf) in ranked:
    train_selected[pat] = (direction, tp, sl)

print(f"Discovered {len(train_selected)} patterns on train data")

# --- Test on HOLDOUT ---
print("\n--- Evaluating on TEST (holdout) data ---")
train_t, train_wr, train_pnl, train_mdd, train_pf = portfolio_backtest(train_selected, 0, split_idx)
test_t, test_wr, test_pnl, test_mdd, test_pf = portfolio_backtest(train_selected, split_idx, n_bars)

print(f"\n  Train (180d): Trades={train_t}, WR={train_wr:.1f}%, PnL={train_pnl:+.1f}%, MDD={train_mdd:.1f}%, PF={train_pf:.2f}")
print(f"  Test  (90d):  Trades={test_t}, WR={test_wr:.1f}%, PnL={test_pnl:+.1f}%, MDD={test_mdd:.1f}%, PF={test_pf:.2f}")

if test_t > 0 and train_wr > 0:
    wr_degradation = (train_wr - test_wr) / train_wr * 100
    print(f"  WR Degradation: {wr_degradation:+.1f}%")
    if wr_degradation < 10:
        print("  → PASS: Minimal out-of-sample degradation")
    elif wr_degradation < 20:
        print("  → WARN: Moderate degradation, some overfitting risk")
    else:
        print("  → FAIL: Significant degradation, likely overfitted")
else:
    print("  → WARN: Insufficient test trades for reliable comparison")

# Also test PRODUCTION patterns on holdout
print("\n--- Production patterns (full 270d optimized) on TEST holdout ---")
prod_test_t, prod_test_wr, prod_test_pnl, prod_test_mdd, prod_test_pf = portfolio_backtest(
    PRODUCTION_PATTERNS, split_idx, n_bars)
prod_train_t, prod_train_wr, prod_train_pnl, prod_train_mdd, prod_train_pf = portfolio_backtest(
    PRODUCTION_PATTERNS, 0, split_idx)
print(f"  Prod Train (180d): Trades={prod_train_t}, WR={prod_train_wr:.1f}%, PF={prod_train_pf:.2f}")
print(f"  Prod Test  (90d):  Trades={prod_test_t}, WR={prod_test_wr:.1f}%, PF={prod_test_pf:.2f}")
if prod_test_t > 0 and prod_train_wr > 0:
    prod_deg = (prod_train_wr - prod_test_wr) / prod_train_wr * 100
    print(f"  Prod WR Degradation: {prod_deg:+.1f}%")


# =========================================================
# TEST 2: RANDOM PATTERN BASELINE
# =========================================================
print("\n" + "=" * 70)
print("[TEST 2] RANDOM PATTERN BASELINE (1000 simulations)")
print("=" * 70)

# Get all unique patterns with enough trades
all_unique = list(set(p for p in patterns if p is not None))
pattern_trade_counts = {}
for p in all_unique:
    cnt = sum(1 for i in range(2, n_bars) if patterns[i] == p)
    if cnt >= 10:
        pattern_trade_counts[p] = cnt

eligible_patterns = list(pattern_trade_counts.keys())
print(f"Eligible patterns (>=10 trades): {len(eligible_patterns)}")

# Real strategy PnL
real_t, real_wr, real_pnl, real_mdd, real_pf = portfolio_backtest(PRODUCTION_PATTERNS)
print(f"Real strategy: Trades={real_t}, WR={real_wr:.1f}%, PnL={real_pnl:+.1f}%, PF={real_pf:.2f}")

N_SIMS = 1000
random_pnls = []
random_wrs = []
print(f"\nRunning {N_SIMS} random portfolio simulations...")

for sim in range(N_SIMS):
    if (sim + 1) % 200 == 0:
        print(f"  Sim {sim+1}/{N_SIMS}...")

    # Pick 13 random patterns with random directions
    chosen = random.sample(eligible_patterns, min(13, len(eligible_patterns)))
    rand_config = {}
    for p in chosen:
        d = random.choice(["LONG", "SHORT"])
        rand_config[p] = (d, 1.0, 1.0)  # uniform TP/SL

    t, wr, pnl, mdd, pf = portfolio_backtest(rand_config)
    random_pnls.append(pnl)
    random_wrs.append(wr if t > 0 else 50.0)

random_pnls = np.array(random_pnls)
random_wrs = np.array(random_wrs)

pnl_percentile = np.mean(random_pnls < real_pnl) * 100
wr_percentile = np.mean(random_wrs < real_wr) * 100

print(f"\nRandom PnL distribution: median={np.median(random_pnls):+.1f}%, "
      f"mean={np.mean(random_pnls):+.1f}%, std={np.std(random_pnls):.1f}%")
print(f"Random WR distribution: median={np.median(random_wrs):.1f}%, "
      f"mean={np.mean(random_wrs):.1f}%")
print(f"\nReal strategy PnL percentile: {pnl_percentile:.1f}th")
print(f"Real strategy WR percentile: {wr_percentile:.1f}th")
p_value = 1 - pnl_percentile / 100
print(f"p-value (PnL): {p_value:.4f}")

if p_value < 0.01:
    print("→ STRONG PASS: Strategy significantly outperforms random (p<0.01)")
elif p_value < 0.05:
    print("→ PASS: Strategy outperforms random (p<0.05)")
elif p_value < 0.10:
    print("→ WARN: Marginal significance (p<0.10)")
else:
    print("→ FAIL: Cannot distinguish from random selection")


# =========================================================
# TEST 3: BONFERRONI MULTIPLE TESTING CORRECTION
# =========================================================
print("\n" + "=" * 70)
print("[TEST 3] BONFERRONI MULTIPLE TESTING CORRECTION")
print("=" * 70)

# Number of hypotheses tested
n_patterns_tested = 1728  # 12^3
n_tpsl_combos = len(TP_GRID) * len(SL_GRID)
n_directions = 2
n_hypotheses = n_patterns_tested * n_tpsl_combos * n_directions
print(f"Total hypotheses: {n_patterns_tested} patterns × {n_tpsl_combos} TP/SL × {n_directions} dir = {n_hypotheses}")

# Per-pattern MC p-values (from v1.21.0 optimization)
pattern_mc = {
    "U-MU-H":   0.0000,
    "MD-ST-MD": 0.0078,
    "GS-U-BD":  0.0372,
    "MD-MD-ST": 0.0002,
    "BU-IH-DN": 0.0022,
    "MD-H-MD":  0.0014,
    "IH-MD-MD": 0.0020,
    "DN-D-BD":  0.2390,
    "BD-U-GS":  0.0042,
    "DN-GS-H":  0.0176,
    "U-DF-BU":  0.0010,
    "BD-GS-BD": 0.0120,
    "DN-IH-IH": 0.0000,
}

alpha = 0.05
bonferroni_alpha = alpha / n_hypotheses
print(f"Bonferroni-corrected alpha: {alpha} / {n_hypotheses} = {bonferroni_alpha:.2e}")

# Also use less conservative Holm-Bonferroni
sorted_pvals = sorted(pattern_mc.items(), key=lambda x: x[1])
surviving_bonferroni = []
surviving_holm = []

print(f"\n{'Pattern':<12} {'Raw MC':>10} {'Bonf adj':>12} {'Bonf':>8} {'Holm':>8}")
print("-" * 55)

for rank, (pat, pval) in enumerate(sorted_pvals):
    bonf_adj = min(pval * n_hypotheses, 1.0)
    holm_alpha = alpha / (len(pattern_mc) - rank)
    bonf_pass = "✓" if pval < bonferroni_alpha else "✗"
    holm_pass = "✓" if pval < holm_alpha else "✗"
    print(f"{pat:<12} {pval:>10.4f} {bonf_adj:>12.4f} {bonf_pass:>8} {holm_pass:>8}")

    if pval < bonferroni_alpha:
        surviving_bonferroni.append(pat)
    if pval < holm_alpha:
        surviving_holm.append(pat)

print(f"\nSurviving strict Bonferroni: {len(surviving_bonferroni)}/13")
if surviving_bonferroni:
    print(f"  Patterns: {', '.join(surviving_bonferroni)}")

print(f"Surviving Holm-Bonferroni (less conservative): {len(surviving_holm)}/13")
if surviving_holm:
    print(f"  Patterns: {', '.join(surviving_holm)}")

# FDR (Benjamini-Hochberg) as additional reference
print(f"\n--- Benjamini-Hochberg FDR (alpha=0.05) ---")
bh_surviving = []
m = len(sorted_pvals)
for rank, (pat, pval) in enumerate(sorted_pvals):
    bh_threshold = alpha * (rank + 1) / m
    if pval <= bh_threshold:
        bh_surviving.append(pat)
    else:
        break  # BH procedure stops at first failure

print(f"Surviving BH-FDR: {len(bh_surviving)}/13")
if bh_surviving:
    print(f"  Patterns: {', '.join(bh_surviving)}")

# Note about MC=0.0000
print("\nNote: MC=0.0000 means <1/10000 in sign-randomization test.")
print("  Even with Bonferroni, these represent very strong statistical significance.")
print("  However, the correction assumes patterns were independently discovered,")
print("  which may be overly conservative.")


# =========================================================
# TEST 4: TIME-DECAY ANALYSIS
# =========================================================
print("\n" + "=" * 70)
print("[TEST 4] TIME-DECAY ANALYSIS (6 periods)")
print("=" * 70)

n_periods = 6
period_size = n_bars // n_periods
print(f"Period size: ~{period_size} bars (~{period_size/288:.0f} days)")

period_results = []
for p in range(n_periods):
    start = p * period_size
    end = start + period_size if p < n_periods - 1 else n_bars

    t, wr, pnl, mdd, pf = portfolio_backtest(PRODUCTION_PATTERNS, start, end)
    period_results.append({
        'period': p + 1,
        'trades': t,
        'wr': wr,
        'pnl': pnl,
        'pf': pf,
    })
    profitable = "✓" if pnl > 0 else "✗"
    print(f"  Period {p+1}: Trades={t:>3}, WR={wr:>5.1f}%, PnL={pnl:>+8.1f}%, PF={pf:>5.2f} {profitable}")

wrs = [r['wr'] for r in period_results if r['trades'] > 0]
if len(wrs) >= 3:
    # Simple linear regression for trend
    x = np.arange(len(wrs))
    slope = np.polyfit(x, wrs, 1)[0]
    print(f"\n  WR Trend: [{' → '.join(f'{w:.0f}%' for w in wrs)}]")
    print(f"  WR Slope: {slope:+.2f}%/period")

    profitable_periods = sum(1 for r in period_results if r['pnl'] > 0)
    print(f"  Profitable periods: {profitable_periods}/{n_periods}")

    if slope > -1.0 and profitable_periods >= 4:
        print("  → PASS: No significant time-decay, consistent across periods")
    elif slope > -2.0 and profitable_periods >= 3:
        print("  → WARN: Mild decay or inconsistency, monitor closely")
    else:
        print("  → FAIL: Significant time-decay detected, likely overfitted")


# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 70)
print("OVERFITTING DIAGNOSIS SUMMARY")
print("=" * 70)

print(f"""
[Test 1] True Train/Test Split
  Train: {train_t} trades, WR={train_wr:.1f}%, PF={train_pf:.2f}
  Test:  {test_t} trades, WR={test_wr:.1f}%, PF={test_pf:.2f}
  Production on Test: {prod_test_t} trades, WR={prod_test_wr:.1f}%, PF={prod_test_pf:.2f}

[Test 2] Random Baseline
  Strategy PnL: {pnl_percentile:.1f}th percentile (p={p_value:.4f})

[Test 3] Multiple Testing Correction
  Bonferroni surviving: {len(surviving_bonferroni)}/13
  BH-FDR surviving: {len(bh_surviving)}/13

[Test 4] Time-Decay
  Profitable periods: {profitable_periods}/{n_periods}
  WR slope: {slope:+.2f}%/period
""")

print("RECOMMENDATION:")
if test_wr > 55 and p_value < 0.05 and len(bh_surviving) >= 5 and profitable_periods >= 4:
    print("  ✅ Strategy shows genuine edge with acceptable overfitting risk")
    print(f"  Consider reducing to {len(bh_surviving)} BH-surviving patterns for extra safety")
elif test_wr > 50 and p_value < 0.10:
    print("  ⚠️ Strategy shows some edge but with notable overfitting risk")
    print("  Recommend: reduce pattern count, use uniform TP/SL, extend validation data")
else:
    print("  ❌ High overfitting risk detected")
    print("  Recommend: fundamental strategy redesign or much larger dataset")
