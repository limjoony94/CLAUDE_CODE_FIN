"""
Reverse Edge Research
=====================
비선정 패턴 및 반대 방향 edge 분석.

Phase 1: 52 validated 패턴의 반대 방향 edge 검증
  - BD-BD-U (LONG) → BD-BD-U (SHORT) 테스트
  - 반대 방향 WR vs distance WR 비교
  - MC test for significance

Phase 2: 전체 패턴 유니버스 스캔 (1728 × 2 = 3456)
  - 거래 >= 10건인 패턴 필터
  - Edge 유의성 (MC p < 0.01) 검증
  - 기존 52개와 비교

Phase 3: Anti-pattern 분석
  - Distance WR 대비 유의하게 낮은 WR (역edge)
  - 역edge → 반대 방향 진입 시 수익 가능 여부

Standard Research Protocol: next-bar open entry, distance-based exit,
  0.10% fee, 3x leverage, 10k MC sims
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
    PATTERN_OPTIMAL_TPSL,
)

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000
TP_SCALE = 0.70
MIN_TRADES = 10

np.random.seed(42)

print("=" * 70)
print("Reverse Edge Research")
print("=" * 70)

# ============================================================
# Data loading & classification
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

print(f"Data: {n_bars} bars")

print("Classifying candles...")
body_abs = np.abs(closes - opens)
avg_body_20 = pd.Series(body_abs).rolling(20).mean().values
types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20[i] if not pd.isna(avg_body_20[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)
print("Done.\n")

# Build current portfolio
PORTFOLIO = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("LONG", round(tp * TP_SCALE, 2), sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("SHORT", round(tp * TP_SCALE, 2), sl)


# ============================================================
# Shared helpers
# ============================================================
def simulate_trade(direction, tp, sl, signal_bar):
    """Returns (pnl,) or None."""
    eb = signal_bar + 1
    if eb >= n_bars:
        return None
    entry = opens[eb]
    if entry <= 0:
        return None
    if direction == 'LONG':
        tpp = entry * (1 + tp / 100)
        slp = entry * (1 - sl / 100)
    else:
        tpp = entry * (1 - tp / 100)
        slp = entry * (1 + sl / 100)
    for j in range(eb + 1, min(eb + MAX_BARS, n_bars)):
        ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
        hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
        if ht and hs:
            pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
            return pnl
        elif ht:
            return tp * LEVERAGE - FEE_PCT
        elif hs:
            return -sl * LEVERAGE - FEE_PCT
    return None


def mc_test(pnl_list, n_sims=MC_SIMS):
    if len(pnl_list) < 8:
        return 1.0
    pa = np.array(pnl_list)
    real = np.sum(pa)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pa)))
    return float(np.mean(np.sum(pa * signs, axis=1) >= real))


def distance_wr(tp, sl):
    """Random walk expected WR = SL / (TP + SL)."""
    return sl / (tp + sl) * 100 if (tp + sl) > 0 else 50.0


def scan_pattern(pat_name, direction, tp, sl):
    """Scan all bars for pattern, return trade PnLs."""
    pnls = []
    for i in range(2, n_bars):
        p = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if p != pat_name:
            continue
        result = simulate_trade(direction, tp, sl, i)
        if result is not None:
            pnls.append(result)
    return pnls


results = {}

# ============================================================
# Phase 1: Reverse edge of 52 validated patterns
# ============================================================
print("=" * 70)
print("Phase 1: Reverse Edge of 52 Validated Patterns")
print("=" * 70)
print()

phase1 = []

for pat, (direction, tp, sl) in PORTFOLIO.items():
    # Original direction
    orig_pnls = scan_pattern(pat, direction, tp, sl)
    orig_wr = sum(1 for p in orig_pnls if p > 0) / len(orig_pnls) * 100 if orig_pnls else 0
    orig_mc = mc_test(orig_pnls)

    # Reverse direction with SAME TP/SL
    rev_dir = "SHORT" if direction == "LONG" else "LONG"
    rev_pnls = scan_pattern(pat, rev_dir, tp, sl)
    rev_wr = sum(1 for p in rev_pnls if p > 0) / len(rev_pnls) * 100 if rev_pnls else 0
    rev_mc = mc_test(rev_pnls)
    rev_total = sum(rev_pnls) if rev_pnls else 0

    d_wr = distance_wr(tp, sl)

    phase1.append({
        'pattern': pat,
        'orig_dir': direction,
        'tp': tp, 'sl': sl,
        'orig_trades': len(orig_pnls),
        'orig_wr': round(orig_wr, 1),
        'orig_mc': round(orig_mc, 4),
        'rev_trades': len(rev_pnls),
        'rev_wr': round(rev_wr, 1),
        'rev_pnl': round(rev_total, 2),
        'rev_mc': round(rev_mc, 4),
        'distance_wr': round(d_wr, 1),
        'orig_edge': round(orig_wr - d_wr, 1),
        'rev_edge': round(rev_wr - d_wr, 1),
    })

# Sort by reverse edge (descending to find any positive reverse edges)
phase1.sort(key=lambda x: x['rev_edge'], reverse=True)

# Summary stats
rev_positive_edge = [p for p in phase1 if p['rev_edge'] > 0]
rev_negative_edge = [p for p in phase1 if p['rev_edge'] < 0]
rev_mc_sig = [p for p in phase1 if p['rev_mc'] < 0.01 and p['rev_pnl'] > 0]
anti_patterns = [p for p in phase1 if p['rev_edge'] < -10]

print(f"  52 patterns reverse direction analysis:")
print(f"  Reverse edge > 0 (profit in reverse): {len(rev_positive_edge)}")
print(f"  Reverse edge < 0 (loss in reverse):   {len(rev_negative_edge)}")
print(f"  Reverse MC < 0.01 (significant):      {len(rev_mc_sig)}")
print(f"  Anti-patterns (rev_edge < -10pp):      {len(anti_patterns)}")
print()

# Show top reverse edges
print("  Patterns with HIGHEST reverse edge (reverse might also work?):")
print(f"  {'Pattern':<14} {'Dir':<6} {'TP/SL':>8} {'DistWR':>7} {'OrigWR':>7} {'RevWR':>7} {'RevEdge':>8} {'RevMC':>8} {'RevPnL':>8}")
print("  " + "-" * 88)
for p in phase1[:10]:
    print(f"  {p['pattern']:<14} {p['orig_dir']:<6} {p['tp']:.1f}/{p['sl']:.1f} "
          f"{p['distance_wr']:>6.1f}% {p['orig_wr']:>6.1f}% {p['rev_wr']:>6.1f}% "
          f"{p['rev_edge']:>+7.1f}pp {p['rev_mc']:>7.4f} {p['rev_pnl']:>+7.1f}%")
print()

# Show anti-patterns (strong negative reverse edge)
print("  Anti-patterns (reverse direction clearly LOSES, confirms original edge):")
print(f"  {'Pattern':<14} {'Dir':<6} {'TP/SL':>8} {'DistWR':>7} {'OrigWR':>7} {'RevWR':>7} {'RevEdge':>8} {'RevMC':>8}")
print("  " + "-" * 80)
for p in sorted(phase1, key=lambda x: x['rev_edge'])[:10]:
    print(f"  {p['pattern']:<14} {p['orig_dir']:<6} {p['tp']:.1f}/{p['sl']:.1f} "
          f"{p['distance_wr']:>6.1f}% {p['orig_wr']:>6.1f}% {p['rev_wr']:>6.1f}% "
          f"{p['rev_edge']:>+7.1f}pp {p['rev_mc']:>7.4f}")

results['phase1'] = phase1

# ============================================================
# Phase 2: Full Universe Scan (1728 patterns × 2 directions)
# ============================================================
print()
print("=" * 70)
print("Phase 2: Full Universe Scan (1728 × 2 = 3456 combinations)")
print("=" * 70)
print()

# Get unique candle types from data
unique_types = sorted(set(types))
print(f"  Unique candle types in data: {len(unique_types)}")
print(f"  Types: {', '.join(unique_types)}")

# Count all 3-candle pattern occurrences
pattern_counts = defaultdict(int)
for i in range(2, n_bars):
    p = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    pattern_counts[p] += 1

all_patterns = sorted(pattern_counts.keys())
print(f"  Total unique 3-candle patterns observed: {len(all_patterns)}")
print(f"  Patterns with >= {MIN_TRADES} occurrences: {sum(1 for c in pattern_counts.values() if c >= MIN_TRADES)}")
print()

# Use a representative TP/SL for non-portfolio patterns
# Median of current portfolio TP/SL
all_tp = [v[1] for v in PORTFOLIO.values()]
all_sl = [v[2] for v in PORTFOLIO.values()]
median_tp = round(np.median(all_tp), 2)
median_sl = round(np.median(all_sl), 2)
print(f"  Representative TP/SL for non-portfolio scan: TP={median_tp}%, SL={median_sl}%")
print()

# Scan all patterns × both directions
phase2 = []
n_scanned = 0
n_sufficient = 0

for pat in all_patterns:
    if pattern_counts[pat] < MIN_TRADES:
        continue

    in_portfolio = pat in PORTFOLIO
    if in_portfolio:
        port_dir, port_tp, port_sl = PORTFOLIO[pat]
    else:
        port_dir, port_tp, port_sl = None, None, None

    for direction in ['LONG', 'SHORT']:
        # Use portfolio TP/SL if available, otherwise median
        if in_portfolio:
            tp, sl = port_tp, port_sl
        else:
            tp, sl = median_tp, median_sl

        pnls = scan_pattern(pat, direction, tp, sl)
        if len(pnls) < MIN_TRADES:
            continue

        n_sufficient += 1
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        mc = mc_test(pnls)
        total_pnl = sum(pnls)
        d_wr = distance_wr(tp, sl)
        edge = wr - d_wr

        is_validated = in_portfolio and (
            (direction == port_dir)
        )
        is_reverse = in_portfolio and (direction != port_dir)

        phase2.append({
            'pattern': pat,
            'direction': direction,
            'tp': tp, 'sl': sl,
            'trades': len(pnls),
            'wr': round(wr, 1),
            'pnl': round(total_pnl, 2),
            'mc': round(mc, 4),
            'distance_wr': round(d_wr, 1),
            'edge': round(edge, 1),
            'status': 'validated' if is_validated else ('reverse' if is_reverse else 'new'),
        })

    n_scanned += 1

print(f"  Patterns scanned: {n_scanned}")
print(f"  Direction combos with >= {MIN_TRADES} trades: {n_sufficient}")

# Classify results
validated = [p for p in phase2 if p['status'] == 'validated']
reverse = [p for p in phase2 if p['status'] == 'reverse']
new_pats = [p for p in phase2 if p['status'] == 'new']

# New patterns with significant edge
new_sig = [p for p in new_pats if p['mc'] < 0.01 and p['pnl'] > 0]
new_sig.sort(key=lambda x: x['pnl'], reverse=True)

print(f"\n  Validated (in portfolio, same dir): {len(validated)}")
print(f"  Reverse (in portfolio, opposite dir): {len(reverse)}")
print(f"  New (not in portfolio): {len(new_pats)}")
print(f"  New with MC < 0.01 & PnL > 0: {len(new_sig)}")
print()

if new_sig:
    print("  NEW patterns with significant edge (MC < 0.01):")
    print(f"  {'Pattern':<14} {'Dir':<6} {'TP/SL':>8} {'Trades':>7} {'WR':>6} {'Edge':>8} {'MC':>8} {'PnL':>8}")
    print("  " + "-" * 75)
    for p in new_sig[:20]:
        print(f"  {p['pattern']:<14} {p['direction']:<6} {p['tp']:.1f}/{p['sl']:.1f} "
              f"{p['trades']:>6} {p['wr']:>5.1f}% {p['edge']:>+7.1f}pp {p['mc']:>7.4f} {p['pnl']:>+7.1f}%")
    print()
else:
    print("  No new patterns with significant edge found.\n")

# Edge distribution
all_edges = [p['edge'] for p in phase2]
print(f"  Edge distribution across all combos:")
print(f"    Mean: {np.mean(all_edges):+.1f}pp")
print(f"    Std:  {np.std(all_edges):.1f}pp")
print(f"    Min:  {np.min(all_edges):+.1f}pp")
print(f"    Max:  {np.max(all_edges):+.1f}pp")
print(f"    Edge > +5pp: {sum(1 for e in all_edges if e > 5)}")
print(f"    Edge > +10pp: {sum(1 for e in all_edges if e > 10)}")
print(f"    Edge < -5pp: {sum(1 for e in all_edges if e < -5)}")
print(f"    Edge < -10pp: {sum(1 for e in all_edges if e < -10)}")

results['phase2_summary'] = {
    'patterns_scanned': n_scanned,
    'combos_sufficient': n_sufficient,
    'validated': len(validated),
    'reverse': len(reverse),
    'new': len(new_pats),
    'new_significant': len(new_sig),
    'edge_mean': round(float(np.mean(all_edges)), 2),
    'edge_std': round(float(np.std(all_edges)), 2),
    'median_tp': median_tp,
    'median_sl': median_sl,
}
results['phase2_new_significant'] = new_sig
results['phase2_all'] = phase2

# ============================================================
# Phase 3: Anti-Pattern Analysis
# ============================================================
print()
print("=" * 70)
print("Phase 3: Anti-Pattern Analysis")
print("=" * 70)
print()

# Anti-patterns: reverse direction of validated patterns with significant negative edge
print("  3a. Reverse of validated patterns — confirming original edge")
print()

# Among the 52 validated, check reverse direction
rev_with_sig_loss = [p for p in phase1 if p['rev_edge'] < -5 and p['rev_trades'] >= MIN_TRADES]
rev_with_sig_loss.sort(key=lambda x: x['rev_edge'])

print(f"  Reverse with edge < -5pp AND trades >= {MIN_TRADES}: {len(rev_with_sig_loss)}")
if rev_with_sig_loss:
    print(f"  → These patterns have CONFIRMED directional bias")
    print(f"  {'Pattern':<14} {'OrigDir':<7} {'OrigEdge':>9} {'RevEdge':>9} {'RevTrades':>10}")
    print("  " + "-" * 55)
    for p in rev_with_sig_loss:
        print(f"  {p['pattern']:<14} {p['orig_dir']:<7} {p['orig_edge']:>+8.1f}pp {p['rev_edge']:>+8.1f}pp {p['rev_trades']:>9}")
print()

# Anti-patterns in the broader universe: patterns where BOTH directions lose
print("  3b. Patterns where BOTH directions have negative edge")
print()

# Group by pattern
pat_dirs = defaultdict(dict)
for p in phase2:
    pat_dirs[p['pattern']][p['direction']] = p

both_negative = []
one_positive = []
for pat, dirs in pat_dirs.items():
    if len(dirs) < 2:
        continue
    l_edge = dirs.get('LONG', {}).get('edge', 0)
    s_edge = dirs.get('SHORT', {}).get('edge', 0)
    if l_edge < 0 and s_edge < 0:
        both_negative.append({
            'pattern': pat,
            'long_edge': l_edge,
            'short_edge': s_edge,
            'long_wr': dirs['LONG']['wr'],
            'short_wr': dirs['SHORT']['wr'],
            'long_trades': dirs['LONG']['trades'],
            'short_trades': dirs['SHORT']['trades'],
            'total_abs_edge': abs(l_edge) + abs(s_edge),
        })
    if (l_edge > 5 and s_edge < -5) or (s_edge > 5 and l_edge < -5):
        winner = 'LONG' if l_edge > s_edge else 'SHORT'
        one_positive.append({
            'pattern': pat,
            'long_edge': l_edge,
            'short_edge': s_edge,
            'winner': winner,
            'winner_edge': max(l_edge, s_edge),
            'loser_edge': min(l_edge, s_edge),
            'in_portfolio': pat in PORTFOLIO,
        })

both_negative.sort(key=lambda x: x['total_abs_edge'], reverse=True)
one_positive.sort(key=lambda x: x['winner_edge'], reverse=True)

print(f"  Both directions negative edge: {len(both_negative)} patterns")
print(f"  → These are TRUE noise patterns (neither direction has predictive power)")
if both_negative:
    print(f"  {'Pattern':<14} {'L_Edge':>8} {'S_Edge':>8} {'L_WR':>6} {'S_WR':>6}")
    print("  " + "-" * 50)
    for p in both_negative[:10]:
        print(f"  {p['pattern']:<14} {p['long_edge']:>+7.1f}pp {p['short_edge']:>+7.1f}pp "
              f"{p['long_wr']:>5.1f}% {p['short_wr']:>5.1f}%")
print()

print(f"  3c. Polarized patterns (one dir edge>+5pp, other <-5pp): {len(one_positive)}")
print(f"  → Strong directional bias confirmed by BOTH sides")
if one_positive:
    print(f"  {'Pattern':<14} {'Winner':<7} {'WinEdge':>9} {'LoseEdge':>9} {'InPortfolio':>12}")
    print("  " + "-" * 60)
    for p in one_positive[:15]:
        portfolio_mark = "YES" if p['in_portfolio'] else "NO"
        print(f"  {p['pattern']:<14} {p['winner']:<7} {p['winner_edge']:>+8.1f}pp {p['loser_edge']:>+8.1f}pp "
              f"{portfolio_mark:>11}")

not_in_portfolio = [p for p in one_positive if not p['in_portfolio']]
print(f"\n  Polarized but NOT in portfolio: {len(not_in_portfolio)}")
if not_in_portfolio:
    print("  → Candidates for portfolio addition (need WF+MC validation):")
    for p in not_in_portfolio[:10]:
        print(f"    {p['pattern']} → {p['winner']} (edge {p['winner_edge']:+.1f}pp)")

results['phase3'] = {
    'rev_confirmed_bias': len(rev_with_sig_loss),
    'both_negative': len(both_negative),
    'polarized': len(one_positive),
    'polarized_not_in_portfolio': len(not_in_portfolio),
}
results['phase3_both_negative'] = both_negative[:20]
results['phase3_polarized'] = one_positive

# ============================================================
# Phase 4: Summary — Are non-selected patterns "trend continuation"?
# ============================================================
print()
print("=" * 70)
print("Phase 4: Summary — Non-Selected Pattern Behavior")
print("=" * 70)
print()

# Calculate: among non-portfolio patterns, what's the average edge?
non_portfolio = [p for p in phase2 if p['status'] == 'new']
if non_portfolio:
    avg_edge_new = np.mean([p['edge'] for p in non_portfolio])
    avg_wr_new = np.mean([p['wr'] for p in non_portfolio])
    pos_edge = sum(1 for p in non_portfolio if p['edge'] > 0)
    neg_edge = sum(1 for p in non_portfolio if p['edge'] < 0)
    avg_pnl_new = np.mean([p['pnl'] for p in non_portfolio])

    print(f"  Non-portfolio pattern-direction combos: {len(non_portfolio)}")
    print(f"  Average edge:  {avg_edge_new:+.2f}pp")
    print(f"  Average WR:    {avg_wr_new:.1f}%")
    print(f"  Average PnL:   {avg_pnl_new:+.2f}%")
    print(f"  Positive edge: {pos_edge} ({pos_edge/len(non_portfolio)*100:.1f}%)")
    print(f"  Negative edge: {neg_edge} ({neg_edge/len(non_portfolio)*100:.1f}%)")
    print()

    # Is there a "continuation" bias in LONG vs SHORT?
    new_long = [p for p in non_portfolio if p['direction'] == 'LONG']
    new_short = [p for p in non_portfolio if p['direction'] == 'SHORT']
    avg_edge_l = np.mean([p['edge'] for p in new_long]) if new_long else 0
    avg_edge_s = np.mean([p['edge'] for p in new_short]) if new_short else 0
    print(f"  Non-portfolio LONG  avg edge: {avg_edge_l:+.2f}pp ({len(new_long)} combos)")
    print(f"  Non-portfolio SHORT avg edge: {avg_edge_s:+.2f}pp ({len(new_short)} combos)")
    print()

    # Edge by pattern structure
    # Check if bullish candle patterns (U, BU, MU, H) have LONG edge
    bullish_candles = {'U', 'BU', 'MU', 'H', 'IH'}
    bearish_candles = {'DN', 'BD', 'MD'}

    def pattern_bias(pat_name):
        """Count bullish vs bearish candles in pattern."""
        candles = pat_name.split('-')
        bull = sum(1 for c in candles if c in bullish_candles)
        bear = sum(1 for c in candles if c in bearish_candles)
        if bull > bear:
            return 'bullish'
        elif bear > bull:
            return 'bearish'
        return 'neutral'

    bias_groups = defaultdict(list)
    for p in non_portfolio:
        bias = pattern_bias(p['pattern'])
        bias_groups[(bias, p['direction'])].append(p['edge'])

    print("  Edge by pattern structure × direction:")
    print(f"  {'Structure':<10} {'Direction':<7} {'Count':>6} {'AvgEdge':>9}")
    print("  " + "-" * 40)
    for (bias, direction), edges in sorted(bias_groups.items()):
        print(f"  {bias:<10} {direction:<7} {len(edges):>6} {np.mean(edges):>+8.2f}pp")
    print()

    # Test "continuation hypothesis": bullish patterns → LONG edge? bearish → SHORT?
    bull_long = bias_groups.get(('bullish', 'LONG'), [])
    bull_short = bias_groups.get(('bullish', 'SHORT'), [])
    bear_long = bias_groups.get(('bearish', 'LONG'), [])
    bear_short = bias_groups.get(('bearish', 'SHORT'), [])

    print("  Continuation hypothesis test:")
    print(f"  Bullish pattern → LONG (continuation):  avg edge = {np.mean(bull_long):+.2f}pp" if bull_long else "  N/A")
    print(f"  Bullish pattern → SHORT (reversal):     avg edge = {np.mean(bull_short):+.2f}pp" if bull_short else "  N/A")
    print(f"  Bearish pattern → SHORT (continuation):  avg edge = {np.mean(bear_short):+.2f}pp" if bear_short else "  N/A")
    print(f"  Bearish pattern → LONG (reversal):      avg edge = {np.mean(bear_long):+.2f}pp" if bear_long else "  N/A")

    # If continuation edges are near 0, no continuation bias
    cont_edges = bull_long + bear_short
    rev_edges = bull_short + bear_long
    avg_cont = np.mean(cont_edges) if cont_edges else 0
    avg_rev = np.mean(rev_edges) if rev_edges else 0
    print(f"\n  Overall continuation edge: {avg_cont:+.2f}pp")
    print(f"  Overall reversal edge:     {avg_rev:+.2f}pp")
    print(f"  → {'Continuation bias EXISTS' if avg_cont > 2 else 'Reversal bias EXISTS' if avg_rev > 2 else 'NO systematic direction bias (noise)'}")

    results['phase4'] = {
        'non_portfolio_count': len(non_portfolio),
        'avg_edge': round(float(avg_edge_new), 2),
        'avg_wr': round(float(avg_wr_new), 1),
        'avg_pnl': round(float(avg_pnl_new), 2),
        'positive_edge_pct': round(pos_edge / len(non_portfolio) * 100, 1),
        'non_portfolio_long_avg_edge': round(float(avg_edge_l), 2),
        'non_portfolio_short_avg_edge': round(float(avg_edge_s), 2),
        'continuation_avg_edge': round(float(avg_cont), 2),
        'reversal_avg_edge': round(float(avg_rev), 2),
    }

# ============================================================
# Save
# ============================================================
out_path = Path(__file__).resolve().parent / '../../results/reverse_edge_research.json'
results['timestamp'] = datetime.now().isoformat()
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n\u2705 Saved: {out_path}")

# ============================================================
# Final Summary
# ============================================================
print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print()
print(f"  Phase 1: Reverse edge of 52 validated patterns")
print(f"    Reverse with positive edge: {len(rev_positive_edge)}/52")
print(f"    Reverse with MC < 0.01:     {len(rev_mc_sig)}/52")
print(f"    Anti-patterns (rev < -10pp): {len(anti_patterns)}/52")
print()
print(f"  Phase 2: Full universe scan")
print(f"    Total combos tested:  {n_sufficient}")
print(f"    New with MC < 0.01:   {len(new_sig)}")
print(f"    Validated confirmed:  {len(validated)}")
print()
print(f"  Phase 3: Anti-pattern analysis")
print(f"    Both-direction negative: {len(both_negative)} (pure noise)")
print(f"    Polarized patterns:     {len(one_positive)} (strong directional)")
print(f"    Polarized NOT in portfolio: {len(not_in_portfolio)} (candidates)")
print()
if non_portfolio:
    print(f"  Phase 4: Non-selected pattern behavior")
    print(f"    Average edge:           {avg_edge_new:+.2f}pp")
    print(f"    Continuation avg edge:  {avg_cont:+.2f}pp")
    print(f"    Reversal avg edge:      {avg_rev:+.2f}pp")
print("=" * 70)
