"""
Low-WR Individual Pattern Deep Validation (v1.27.2)
=====================================================
U-H-BU 제거 후 나머지 25개 low-WR 패턴의 개별 심층 검증.

Per-pattern analysis:
1. Period-by-Period Performance (P1/P2/P3 각 90일)
2. Monthly Stability (9개월 중 수익 월 수)
3. MFE/MAE Analysis (Maximum Favorable/Adverse Excursion)
4. Recent vs Early Performance (후반 vs 전반)
5. Streak Analysis (연속손실 패턴)
6. MC Robustness (10k sims at current TP/SL)
7. WF Individual (5-fold temporal)
8. Edge Stability (rolling edge across time)
9. Execution Quality (SL proximity — 실전 SL 히트 확률)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    PATTERN_OPTIMAL_TPSL, VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
    PATTERN_STATS,
)

# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000

print("=" * 80)
print("Low-WR Individual Pattern Deep Validation (v1.27.2)")
print("=" * 80)

# ============================================================
# Load and classify
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
print(f"\nLoaded: {len(df)} bars")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

df['timestamp'] = pd.to_datetime(df['timestamp'])
dates = df['timestamp'].values
timestamps = df['timestamp'].values

print("Classifying candles...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values
types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)

pattern_indices = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    pattern_indices[pat].append(i)
for k in pattern_indices:
    pattern_indices[k] = np.array(pattern_indices[k])

# Build current map (post U-H-BU removal)
current_map = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    current_map[pat] = ('LONG', tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    current_map[pat] = ('SHORT', tp, sl)

# Period definitions
period_bounds = {
    'P1': (np.datetime64('2025-05-01'), np.datetime64('2025-08-01')),
    'P2': (np.datetime64('2025-08-01'), np.datetime64('2025-11-01')),
    'P3': (np.datetime64('2025-11-01'), np.datetime64('2026-02-01')),
}

# Monthly definitions
months = []
for year in [2025, 2026]:
    for month in range(1, 13):
        m_start = np.datetime64(f'{year}-{month:02d}-01')
        if month == 12:
            m_end = np.datetime64(f'{year+1}-01-01')
        else:
            m_end = np.datetime64(f'{year}-{month+1:02d}-01')
        if m_start >= np.datetime64('2025-05-01') and m_start < np.datetime64('2026-02-01'):
            months.append((f'{year}-{month:02d}', m_start, m_end))

print(f"Periods: {list(period_bounds.keys())}")
print(f"Months: {[m[0] for m in months]}")


# ============================================================
# Engine
# ============================================================
def bt_detailed(indices, direction, tp_pct, sl_pct):
    """Backtest with detailed trade info (entry, exit, duration, MFE, MAE)."""
    trades = []
    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        entry_bar = idx + 1

        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)

        mfe = 0  # max favorable excursion (%)
        mae = 0  # max adverse excursion (%)
        pnl = None
        exit_bar = None

        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            # Track MFE/MAE
            if direction == 'LONG':
                fav = (highs[j] - entry) / entry * 100
                adv = (entry - lows[j]) / entry * 100
            else:
                fav = (entry - lows[j]) / entry * 100
                adv = (highs[j] - entry) / entry * 100

            if fav > mfe: mfe = fav
            if adv > mae: mae = adv

            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)

            if ht and hs:
                pnl = (tp_pct if abs(tpp - entry) <= abs(slp - entry) else -sl_pct) * LEVERAGE - FEE_PCT
                exit_bar = j
                break
            elif ht:
                pnl = tp_pct * LEVERAGE - FEE_PCT
                exit_bar = j
                break
            elif hs:
                pnl = -sl_pct * LEVERAGE - FEE_PCT
                exit_bar = j
                break

        if pnl is not None:
            duration_bars = exit_bar - entry_bar
            trades.append({
                'signal_bar': idx,
                'entry_bar': entry_bar,
                'exit_bar': exit_bar,
                'entry_price': entry,
                'pnl': pnl,
                'win': pnl > 0,
                'duration_bars': duration_bars,
                'duration_hours': duration_bars * 5 / 60,
                'mfe': round(mfe, 3),
                'mae': round(mae, 3),
            })
    return trades


def mc_test(pnl_list, n_sims=MC_SIMS):
    if len(pnl_list) < 5:
        return 1.0
    pa = np.array(pnl_list)
    real = np.sum(pa)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pa)))
    return float(np.mean(np.sum(pa * signs, axis=1) >= real))


# ============================================================
# Identify target patterns
# ============================================================
all_patterns_wr = []
for pat, stats in PATTERN_STATS.items():
    if pat not in current_map:  # skip removed patterns
        continue
    all_patterns_wr.append((pat, stats['wr'], stats['direction']))
all_patterns_wr.sort(key=lambda x: x[1])

n_target = len(all_patterns_wr) // 2  # bottom 50%
target_list = all_patterns_wr[:n_target]

print(f"\nTarget: {n_target} bottom-WR patterns for individual validation")
print(f"WR range: {target_list[0][1]:.1f}% ~ {target_list[-1][1]:.1f}%")


# ============================================================
# Per-pattern deep analysis
# ============================================================
t0 = time.time()
results = {}

for pat, wr_stat, _ in target_list:
    direction, tp, sl = current_map[pat]
    indices = pattern_indices.get(pat, np.array([]))
    rr = tp / sl if sl > 0 else 999

    # Detailed backtest
    trades = bt_detailed(indices, direction, tp, sl)
    pnls = [t['pnl'] for t in trades]
    n_trades = len(trades)

    if n_trades == 0:
        results[pat] = {'error': 'no trades'}
        continue

    total_pnl = sum(pnls)
    wr = sum(1 for t in trades if t['win']) / n_trades * 100
    wins = [t['pnl'] for t in trades if t['win']]
    losses = [abs(t['pnl']) for t in trades if not t['win']]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # --- 1. Period-by-Period ---
    period_results = {}
    for pname, (p_start, p_end) in period_bounds.items():
        p_mask = (dates[indices] >= p_start) & (dates[indices] < p_end)
        p_idx = indices[p_mask]
        p_trades = bt_detailed(p_idx, direction, tp, sl)
        p_pnls = [t['pnl'] for t in p_trades]
        p_n = len(p_trades)
        if p_n > 0:
            p_wr = sum(1 for p in p_pnls if p > 0) / p_n * 100
            p_pnl = sum(p_pnls)
        else:
            p_wr = 0
            p_pnl = 0
        period_results[pname] = {
            'trades': p_n, 'wr': round(p_wr, 1), 'pnl': round(p_pnl, 1)
        }

    # --- 2. Monthly Stability ---
    monthly_results = {}
    profitable_months = 0
    for m_name, m_start, m_end in months:
        m_mask = (dates[indices] >= m_start) & (dates[indices] < m_end)
        m_idx = indices[m_mask]
        m_trades = bt_detailed(m_idx, direction, tp, sl)
        m_pnls = [t['pnl'] for t in m_trades]
        m_n = len(m_trades)
        m_pnl = sum(m_pnls) if m_pnls else 0
        if m_pnl > 0:
            profitable_months += 1
        monthly_results[m_name] = {
            'trades': m_n, 'pnl': round(m_pnl, 1)
        }

    # --- 3. MFE/MAE ---
    mfes = [t['mfe'] for t in trades]
    maes = [t['mae'] for t in trades]
    win_mfes = [t['mfe'] for t in trades if t['win']]
    loss_maes = [t['mae'] for t in trades if not t['win']]

    # Efficiency = PnL / MFE (how much of max move was captured)
    win_trades_with_mfe = [t for t in trades if t['win'] and t['mfe'] > 0]
    if win_trades_with_mfe:
        avg_efficiency = np.mean([t['pnl'] / (t['mfe'] * LEVERAGE) for t in win_trades_with_mfe])
    else:
        avg_efficiency = 0

    # SL proximity: how close losing trades get to SL before stopping out
    # MAE/SL ratio for losing trades
    loss_trades = [t for t in trades if not t['win']]
    if loss_trades:
        mae_sl_ratios = [t['mae'] / sl * 100 for t in loss_trades]
        avg_mae_sl = np.mean(mae_sl_ratios)
    else:
        avg_mae_sl = 0

    # --- 4. Recent vs Early ---
    half = n_trades // 2
    early_pnls = pnls[:half]
    recent_pnls = pnls[half:]
    early_wr = sum(1 for p in early_pnls if p > 0) / len(early_pnls) * 100 if early_pnls else 0
    recent_wr = sum(1 for p in recent_pnls if p > 0) / len(recent_pnls) * 100 if recent_pnls else 0
    early_pnl = sum(early_pnls)
    recent_pnl = sum(recent_pnls)

    # --- 5. Streak Analysis ---
    max_streak = 0
    current_streak = 0
    max_streak_pnl = 0
    current_streak_pnl = 0
    streaks = []
    for p in pnls:
        if p < 0:
            current_streak += 1
            current_streak_pnl += p
        else:
            if current_streak > 0:
                streaks.append((current_streak, current_streak_pnl))
                if current_streak > max_streak:
                    max_streak = current_streak
                    max_streak_pnl = current_streak_pnl
            current_streak = 0
            current_streak_pnl = 0
    if current_streak > 0:
        streaks.append((current_streak, current_streak_pnl))
        if current_streak > max_streak:
            max_streak = current_streak
            max_streak_pnl = current_streak_pnl

    # --- 6. MC Test ---
    mc_p = mc_test(pnls) if n_trades >= 5 else 1.0

    # --- 7. WF Individual ---
    si = np.sort(indices)
    fs = len(si) // 5
    wf_folds = []
    if fs >= 3:
        for f in range(5):
            s = f * fs
            e = s + fs if f < 4 else len(si)
            fold_trades = bt_detailed(si[s:e], direction, tp, sl)
            fold_pnl = sum(t['pnl'] for t in fold_trades)
            fold_n = len(fold_trades)
            fold_wr = sum(1 for t in fold_trades if t['win']) / fold_n * 100 if fold_n > 0 else 0
            wf_folds.append({
                'trades': fold_n, 'pnl': round(fold_pnl, 1),
                'wr': round(fold_wr, 1), 'profitable': fold_pnl > 0
            })
    wf_pass = sum(1 for f in wf_folds if f['profitable'])

    # --- 8. Edge Stability (rolling) ---
    # Split into 3 equal parts and check edge in each
    rw_wr = sl / (tp + sl) * 100
    thirds = np.array_split(np.sort(indices), 3)
    edge_parts = []
    for part in thirds:
        if len(part) < 3:
            edge_parts.append(None)
            continue
        part_trades = bt_detailed(part, direction, tp, sl)
        part_pnls = [t['pnl'] for t in part_trades]
        if part_pnls:
            part_wr = sum(1 for p in part_pnls if p > 0) / len(part_pnls) * 100
            part_edge = part_wr - rw_wr
        else:
            part_wr = 0
            part_edge = -rw_wr
        edge_parts.append(round(part_edge, 1))

    # Edge consistency: all parts positive edge?
    edge_consistent = all(e is not None and e > 0 for e in edge_parts)

    # --- 9. Duration Analysis ---
    durations = [t['duration_bars'] for t in trades]
    avg_duration = np.mean(durations)
    median_duration = np.median(durations)
    max_duration = max(durations)
    quick_sl_hits = sum(1 for t in trades if not t['win'] and t['duration_bars'] <= 3)
    # Percentage of losses that are "instant" SL hits (<=3 bars = 15min)
    instant_sl_pct = quick_sl_hits / len(loss_trades) * 100 if loss_trades else 0

    # --- COMPOSITE HEALTH SCORE ---
    health = 0
    health_max = 8
    if mc_p < 0.01: health += 1
    if wf_pass >= 4: health += 1
    if profitable_months >= 5: health += 1  # majority of 9 months
    if edge_consistent: health += 1
    if total_pnl > 0: health += 1
    if recent_pnl > 0: health += 1  # not degrading
    if max_streak <= 3: health += 1
    if wr_stat - rw_wr >= 5: health += 1  # genuine edge

    # Risk assessment
    if health >= 7:
        risk_level = "LOW"
    elif health >= 5:
        risk_level = "MODERATE"
    elif health >= 3:
        risk_level = "ELEVATED"
    else:
        risk_level = "HIGH"

    results[pat] = {
        'direction': direction, 'tp': tp, 'sl': sl, 'rr': round(rr, 2),
        'trades': n_trades, 'wr': round(wr, 1), 'total_pnl': round(total_pnl, 1),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'rw_wr': round(rw_wr, 1), 'excess_wr': round(wr - rw_wr, 1),
        'mc_p': round(mc_p, 4),
        'periods': period_results,
        'monthly': monthly_results,
        'profitable_months': profitable_months,
        'total_months': len(months),
        'mfe_mae': {
            'avg_mfe': round(np.mean(mfes), 3),
            'avg_mae': round(np.mean(maes), 3),
            'win_avg_mfe': round(np.mean(win_mfes), 3) if win_mfes else 0,
            'loss_avg_mae': round(np.mean(loss_maes), 3) if loss_maes else 0,
            'avg_efficiency': round(avg_efficiency, 2),
            'avg_mae_sl_ratio': round(avg_mae_sl, 1),
        },
        'recency': {
            'early_wr': round(early_wr, 1), 'early_pnl': round(early_pnl, 1),
            'recent_wr': round(recent_wr, 1), 'recent_pnl': round(recent_pnl, 1),
            'trend': 'IMPROVING' if recent_wr > early_wr else ('DEGRADING' if recent_wr < early_wr - 5 else 'STABLE'),
        },
        'streaks': {
            'max_consecutive_loss': max_streak,
            'max_streak_pnl': round(max_streak_pnl, 1),
        },
        'wf': {
            'folds': wf_folds,
            'pass': wf_pass,
        },
        'edge_stability': {
            'parts': edge_parts,
            'consistent': edge_consistent,
        },
        'duration': {
            'avg_bars': round(avg_duration, 1),
            'median_bars': round(median_duration, 1),
            'max_bars': max_duration,
            'avg_hours': round(avg_duration * 5 / 60, 1),
            'instant_sl_pct': round(instant_sl_pct, 1),
        },
        'health_score': health,
        'health_max': health_max,
        'risk_level': risk_level,
    }

elapsed = time.time() - t0
print(f"\nAnalysis complete in {elapsed:.1f}s")


# ============================================================
# Display Results
# ============================================================
print(f"\n{'='*80}")
print("INDIVIDUAL PATTERN VALIDATION RESULTS")
print(f"{'='*80}")

# Summary table
print(f"\n  {'Pattern':<12} {'Dir':>5} {'WR':>5} {'Edge':>6} {'MC':>7} {'WF':>4} {'Mon':>4} "
      f"{'EdgeCon':>7} {'MaxStr':>6} {'Recent':>7} {'Health':>7} {'Risk':>9}")
print("  " + "-" * 100)

for pat, wr_stat, _ in target_list:
    if pat not in results or 'error' in results[pat]:
        continue
    r = results[pat]
    edge_con = "Y" if r['edge_stability']['consistent'] else "N"
    recent = r['recency']['trend'][:3]
    print(f"  {pat:<12} {r['direction']:>5} {r['wr']:>4.1f}% {r['excess_wr']:>+5.1f}p "
          f"{r['mc_p']:>7.4f} {r['wf']['pass']:>2}/5 {r['profitable_months']:>2}/9 "
          f"{edge_con:>7} "
          f"{r['streaks']['max_consecutive_loss']:>6} "
          f"{recent:>7} {r['health_score']:>4}/{r['health_max']} {r['risk_level']:>9}")


# ============================================================
# Detailed per-pattern reports for concerning patterns
# ============================================================
print(f"\n{'='*80}")
print("DETAILED REPORTS — Tier 1 & 2 (WR < 75%)")
print(f"{'='*80}")

tier12_cutoff = 75.0
tier12 = [(pat, wr, d) for pat, wr, d in target_list if wr < tier12_cutoff]

for pat, wr_stat, _ in tier12:
    if pat not in results or 'error' in results[pat]:
        continue
    r = results[pat]
    print(f"\n  {'='*60}")
    print(f"  {pat} ({r['direction']}) — Health {r['health_score']}/{r['health_max']} [{r['risk_level']}]")
    print(f"  {'='*60}")
    print(f"    TP/SL: {r['tp']}/{r['sl']}  R:R={r['rr']}  Trades={r['trades']}  WR={r['wr']}%")
    print(f"    PnL: {r['total_pnl']:+.1f}%  Avg Win: {r['avg_win']:.2f}%  Avg Loss: {r['avg_loss']:.2f}%")
    print(f"    Edge: +{r['excess_wr']:.1f}pp (RW baseline: {r['rw_wr']:.1f}%)  MC: p={r['mc_p']:.4f}")

    print(f"\n    Periods:")
    for pname, pr in r['periods'].items():
        marker = " <<<" if pr['pnl'] < 0 else ""
        print(f"      {pname}: {pr['trades']} trades, WR {pr['wr']}%, PnL {pr['pnl']:+.1f}%{marker}")

    print(f"\n    Monthly ({r['profitable_months']}/{r['total_months']} profitable):")
    for m_name, mr in r['monthly'].items():
        if mr['trades'] > 0:
            marker = " <<<" if mr['pnl'] < 0 else ""
            print(f"      {m_name}: {mr['trades']}t, PnL {mr['pnl']:+.1f}%{marker}")

    print(f"\n    Recency: Early WR {r['recency']['early_wr']}% / Recent WR {r['recency']['recent_wr']}% "
          f"[{r['recency']['trend']}]")
    print(f"      Early PnL: {r['recency']['early_pnl']:+.1f}%  Recent PnL: {r['recency']['recent_pnl']:+.1f}%")

    print(f"\n    Edge Stability (3 equal parts): {r['edge_stability']['parts']} "
          f"[{'CONSISTENT' if r['edge_stability']['consistent'] else 'INCONSISTENT'}]")

    print(f"\n    WF: {r['wf']['pass']}/5 folds profitable")
    for i, f in enumerate(r['wf']['folds']):
        marker = " <<<" if not f['profitable'] else ""
        print(f"      F{i+1}: {f['trades']}t, WR {f['wr']}%, PnL {f['pnl']:+.1f}%{marker}")

    print(f"\n    Duration: avg {r['duration']['avg_bars']:.0f} bars ({r['duration']['avg_hours']:.1f}h), "
          f"max {r['duration']['max_bars']} bars")
    print(f"    Instant SL hits (<=15min): {r['duration']['instant_sl_pct']:.0f}% of losses")

    print(f"\n    MFE/MAE:")
    mm = r['mfe_mae']
    print(f"      Avg MFE: {mm['avg_mfe']:.3f}%  Win MFE: {mm['win_avg_mfe']:.3f}%")
    print(f"      Avg MAE: {mm['avg_mae']:.3f}%  Loss MAE: {mm['loss_avg_mae']:.3f}%")
    print(f"      Win efficiency: {mm['avg_efficiency']:.0f}%  Loss MAE/SL: {mm['avg_mae_sl_ratio']:.0f}%")

    print(f"\n    Streaks: max {r['streaks']['max_consecutive_loss']} consecutive losses "
          f"(PnL {r['streaks']['max_streak_pnl']:+.1f}%)")


# ============================================================
# Risk Rating Summary
# ============================================================
print(f"\n{'='*80}")
print("RISK RATING SUMMARY")
print(f"{'='*80}")

risk_counts = {'LOW': [], 'MODERATE': [], 'ELEVATED': [], 'HIGH': []}
for pat, wr_stat, _ in target_list:
    if pat in results and 'error' not in results[pat]:
        risk_counts[results[pat]['risk_level']].append(pat)

for level in ['HIGH', 'ELEVATED', 'MODERATE', 'LOW']:
    pats = risk_counts[level]
    if pats:
        print(f"\n  {level} RISK ({len(pats)}):")
        for pat in pats:
            r = results[pat]
            flags = []
            if r['mc_p'] >= 0.01: flags.append(f"MC={r['mc_p']:.4f}")
            if r['wf']['pass'] < 4: flags.append(f"WF={r['wf']['pass']}/5")
            if not r['edge_stability']['consistent']: flags.append("EdgeIncon")
            if r['recency']['trend'] == 'DEGRADING': flags.append("Degrading")
            if r['profitable_months'] < 5: flags.append(f"Mon={r['profitable_months']}/9")
            if r['streaks']['max_consecutive_loss'] > 3: flags.append(f"Str={r['streaks']['max_consecutive_loss']}")
            flags_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"    {pat:<14} Health {r['health_score']}/{r['health_max']}  "
                  f"WR {r['wr']:.1f}%  PnL {r['total_pnl']:+.1f}%{flags_str}")

# Patterns that might warrant further action
print(f"\n  --- Patterns Requiring Attention ---")
attention = []
for pat, wr_stat, _ in target_list:
    if pat not in results or 'error' in results[pat]:
        continue
    r = results[pat]
    issues = []
    if r['health_score'] <= 4:
        issues.append(f"Health {r['health_score']}/{r['health_max']}")
    if r['mc_p'] >= 0.01:
        issues.append(f"MC fail (p={r['mc_p']:.4f})")
    if r['recency']['trend'] == 'DEGRADING' and r['recency']['recent_pnl'] < 0:
        issues.append(f"Degrading (recent PnL {r['recency']['recent_pnl']:+.1f}%)")
    if not r['edge_stability']['consistent'] and min(e for e in r['edge_stability']['parts'] if e is not None) < -5:
        issues.append(f"Edge unstable ({r['edge_stability']['parts']})")

    if issues:
        attention.append((pat, r, issues))

if attention:
    for pat, r, issues in attention:
        print(f"    {pat}: {'; '.join(issues)}")
else:
    print(f"    None — all patterns within acceptable parameters")


# ============================================================
# Save
# ============================================================
output = {
    'meta': {
        'script': 'low_wr_individual_validation.py',
        'version': 'v1.27.2',
        'timestamp': datetime.now().isoformat(),
        'target_patterns': len(target_list),
    },
    'per_pattern': results,
    'risk_summary': {level: pats for level, pats in risk_counts.items()},
    'attention_patterns': [
        {'pattern': pat, 'issues': issues}
        for pat, _, issues in attention
    ] if attention else [],
}

out_path = Path(__file__).resolve().parent / '../../results/low_wr_individual_validation.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("Done.")
