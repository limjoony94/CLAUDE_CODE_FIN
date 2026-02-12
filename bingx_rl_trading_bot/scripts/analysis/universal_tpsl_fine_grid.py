#!/usr/bin/env python3
"""
Universal TP/SL Fine Grid Search (0.1 step)
=============================================
v6 Phase 4에서 coarse grid (8단계)로 발견한 TP 2.0/SL 3.0을
0.1 단위 정밀 grid (28×28 = 784 조합)로 재검증.

평가 기준:
  - Pre-overlap (진정한 OOS): PnL, MDD, PnL/MDD, Safety Margin, Trades, WR
  - In-sample (참고): 과적합 확인용
  - Full 720d: 전체 기간

출력:
  - 콘솔: 양수 PnL 조합만 출력
  - JSON: results/universal_tpsl_fine_grid.json
  - Top 20 후보 상세 비교
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(SCRIPT_DIR.parent / 'production'))
from pattern_5m.indicators import classify_candle

FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500

# 51 current patterns (direction only — TP/SL will be universal)
PATTERNS_CURRENT = {
    "BD-BD-U": "LONG", "BD-MU-BD": "LONG", "BD-ST-U": "LONG", "BU-BU-BD": "LONG",
    "D-MU-U": "LONG", "DN-BD-BD": "LONG", "DN-DF-MU": "LONG", "DN-DF-ST": "LONG",
    "DN-DN-H": "LONG", "DN-MD-DN": "LONG", "GS-ST-ST": "LONG", "H-BU-BU": "LONG",
    "H-MU-MD": "LONG", "IH-MD-MD": "LONG", "IH-ST-MU": "LONG", "MD-BU-MD": "LONG",
    "MD-DN-MU": "LONG", "MD-H-MD": "LONG", "MD-MD-ST": "LONG", "MD-ST-BD": "LONG",
    "MD-ST-MD": "LONG", "MU-BD-ST": "LONG", "MU-DF-U": "LONG", "MU-H-MU": "LONG",
    "MU-IH-DN": "LONG", "MU-U-H": "LONG", "U-H-MU": "LONG", "U-MD-GS": "LONG",
    "U-MD-MD": "LONG", "U-MU-H": "LONG", "U-MU-IH": "LONG", "U-ST-DF": "LONG",
    "BD-BU-DN": "SHORT", "BD-D-D": "SHORT", "BD-U-H": "SHORT", "BU-MD-MD": "SHORT",
    "BU-ST-GS": "SHORT", "D-BD-ST": "SHORT", "D-DN-DN": "SHORT", "DN-BD-BU": "SHORT",
    "DN-D-BD": "SHORT", "DN-DF-DN": "SHORT", "H-U-BD": "SHORT", "IH-ST-ST": "SHORT",
    "MD-MD-MD": "SHORT", "ST-BD-BU": "SHORT", "ST-DN-BU": "SHORT", "ST-DN-U": "SHORT",
    "ST-MU-ST": "SHORT", "U-GS-DN": "SHORT", "U-ST-DN": "SHORT",
}


def load_and_classify(csv_path):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    o = df['open'].values; h = df['high'].values
    l = df['low'].values; c = df['close'].values
    n = len(df)
    body_abs = np.abs(c - o)
    avg_b20 = pd.Series(body_abs).rolling(20).mean().values
    types = []
    for i in range(n):
        row = pd.Series({'open': o[i], 'high': h[i], 'low': l[i], 'close': c[i]})
        ab = avg_b20[i] if not pd.isna(avg_b20[i]) else 1.0
        types.append(classify_candle(row, ab).value)
    return o, h, l, c, np.array(types), df['timestamp'].values, n


def collect_trades_1pos(types, opens, highs, lows, n_bars, pattern_dirs,
                        tp_pct, sl_pct, start_bar=0, end_bar=None):
    """Collect trades with 1-position-at-a-time constraint, universal TP/SL."""
    if end_bar is None:
        end_bar = n_bars
    raw = []
    for i in range(max(2, start_bar), end_bar):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in pattern_dirs:
            continue
        if i + 1 >= n_bars:
            continue
        entry = opens[i + 1]
        if entry <= 0:
            continue
        d = pattern_dirs[pat]
        eb = i + 1
        if d == 'LONG':
            tpp = entry * (1 + tp_pct / 100); slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100); slp = entry * (1 + sl_pct / 100)
        for j in range(i + 2, min(i + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if d == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if d == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnl = (tp_pct if abs(tpp - entry) <= abs(slp - entry) else -sl_pct) * LEVERAGE - FEE_PCT
                raw.append((eb, j, pnl)); break
            elif ht:
                raw.append((eb, j, tp_pct * LEVERAGE - FEE_PCT)); break
            elif hs:
                raw.append((eb, j, -sl_pct * LEVERAGE - FEE_PCT)); break
    # 1-position-at-a-time filter
    raw.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl in raw:
        if eb > last_exit:
            filtered.append((eb, xb, pnl))
            last_exit = xb
    return filtered


def calc_stats(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'pnl_mdd': 0, 'avg_win': 0, 'avg_loss': 0, 'safety': 0}
    pnls = [t[2] for t in trades]
    cum = 0; peak = 0; mdd = 0
    wins_list = []; losses_list = []
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
        if p > 0: wins_list.append(p)
        else: losses_list.append(p)
    avg_w = float(np.mean(wins_list)) if wins_list else 0
    avg_l = float(np.mean(losses_list)) if losses_list else 0
    be_wr = abs(avg_l) / (avg_w + abs(avg_l)) * 100 if (avg_w + abs(avg_l)) > 0 else 50
    wr = len(wins_list) / len(pnls) * 100 if pnls else 0
    wsum = sum(wins_list); lsum = sum(abs(x) for x in losses_list)
    return {
        'pnl': round(cum, 1), 'trades': len(pnls),
        'wr': round(wr, 1), 'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
        'pnl_mdd': round(cum / mdd, 1) if mdd > 0 else 999,
        'avg_win': round(avg_w, 3), 'avg_loss': round(avg_l, 3),
        'safety': round(wr - be_wr, 1),
    }


# =======================================================================
# MAIN
# =======================================================================
print("=" * 70)
print("Universal TP/SL Fine Grid Search (0.1 step)")
print("=" * 70)
t0 = time.time()

# Load data
print("\nLoading 720d Binance data...", flush=True)
bn_path = DATA_DIR / "btc_5m_720days_binance.csv"
opens, highs, lows, closes, types, timestamps, n_bars = load_and_classify(bn_path)

overlap_start_ts = np.datetime64('2025-05-05T15:00:00')
overlap_end_ts = np.datetime64('2026-01-30T14:55:00')
overlap_start_bar = int(np.searchsorted(timestamps, overlap_start_ts))
overlap_end_bar = int(np.searchsorted(timestamps, overlap_end_ts, side='right'))

print(f"Total: {n_bars} bars")
print(f"Pre-overlap: 0 ~ {overlap_start_bar} ({overlap_start_bar} bars, ~{overlap_start_bar/288:.0f}d)")
print(f"In-sample: {overlap_start_bar} ~ {overlap_end_bar} ({overlap_end_bar-overlap_start_bar} bars, ~{(overlap_end_bar-overlap_start_bar)/288:.0f}d)")
print(f"Patterns: {len(PATTERNS_CURRENT)}")

# Fine grid: 0.3 to 3.0 in 0.1 steps = 28 values
tp_values = [round(0.3 + i * 0.1, 1) for i in range(28)]  # 0.3, 0.4, ..., 3.0
sl_values = [round(0.3 + i * 0.1, 1) for i in range(28)]
total_combos = len(tp_values) * len(sl_values)
print(f"\nGrid: TP {tp_values[0]}~{tp_values[-1]} × SL {sl_values[0]}~{sl_values[-1]} "
      f"(0.1 step) = {total_combos} combinations")

# Run grid search
results = {'grid': [], 'top_pre_pnl_mdd': [], 'top_pre_safety': []}
grid_data = []
done = 0

print(f"\n{'TP':>5} {'SL':>5} │ {'Pre_PnL':>8} {'Pre_MDD':>8} {'P/M':>6} "
      f"{'Pre_WR':>7} {'Safety':>7} {'Trades':>7} │ {'IS_PnL':>8}", flush=True)
print("-" * 80)

for tp in tp_values:
    for sl in sl_values:
        pre_trades = collect_trades_1pos(types, opens, highs, lows, n_bars,
                                          PATTERNS_CURRENT, tp, sl,
                                          0, overlap_start_bar)
        is_trades = collect_trades_1pos(types, opens, highs, lows, n_bars,
                                         PATTERNS_CURRENT, tp, sl,
                                         overlap_start_bar, overlap_end_bar)
        full_trades = collect_trades_1pos(types, opens, highs, lows, n_bars,
                                           PATTERNS_CURRENT, tp, sl,
                                           0, overlap_end_bar)
        pre_s = calc_stats(pre_trades)
        is_s = calc_stats(is_trades)
        full_s = calc_stats(full_trades)

        row = {
            'tp': tp, 'sl': sl,
            'pre': pre_s, 'is': is_s, 'full': full_s,
        }
        grid_data.append(row)

        done += 1
        # Print only positive pre PnL rows
        if pre_s['pnl'] > 0:
            print(f"{tp:>5.1f} {sl:>5.1f} │ {pre_s['pnl']:>+7.1f}% {pre_s['mdd']:>7.1f}% "
                  f"{pre_s['pnl_mdd']:>5.1f}x {pre_s['wr']:>6.1f}% {pre_s['safety']:>+6.1f}pp "
                  f"{pre_s['trades']:>6} │ {is_s['pnl']:>+7.1f}%", flush=True)

    # Progress
    pct = done / total_combos * 100
    elapsed = time.time() - t0
    eta = elapsed / done * (total_combos - done) if done > 0 else 0
    print(f"  ... TP={tp:.1f} done ({pct:.0f}%, ETA {eta:.0f}s)", flush=True)

results['grid'] = grid_data

# =======================================================================
# Analysis: Top 20 by each metric
# =======================================================================
print("\n" + "=" * 70)
print("Top 20 by Pre-overlap PnL/MDD")
print("=" * 70)

positive = [r for r in grid_data if r['pre']['pnl'] > 0]
by_pnl_mdd = sorted(positive, key=lambda x: x['pre']['pnl_mdd'], reverse=True)[:20]
by_safety = sorted(positive, key=lambda x: x['pre']['safety'], reverse=True)[:20]
by_pnl = sorted(positive, key=lambda x: x['pre']['pnl'], reverse=True)[:20]

print(f"\n{'#':>3} {'TP':>5} {'SL':>5} │ {'PnL/MDD':>8} {'PnL':>8} {'MDD':>7} "
      f"{'WR':>6} {'Safety':>7} {'Trades':>7} │ {'IS_PnL':>8} {'Full_PnL':>9}")
print("-" * 95)
for i, r in enumerate(by_pnl_mdd, 1):
    p = r['pre']; s = r['is']; f = r['full']
    print(f"{i:>3} {r['tp']:>5.1f} {r['sl']:>5.1f} │ {p['pnl_mdd']:>7.1f}x {p['pnl']:>+7.1f}% "
          f"{p['mdd']:>6.1f}% {p['wr']:>5.1f}% {p['safety']:>+6.1f}pp {p['trades']:>6} │ "
          f"{s['pnl']:>+7.1f}% {f['pnl']:>+8.1f}%")

results['top_pre_pnl_mdd'] = by_pnl_mdd[:20]

print(f"\n{'='*70}")
print("Top 20 by Pre-overlap Safety Margin")
print("=" * 70)
print(f"\n{'#':>3} {'TP':>5} {'SL':>5} │ {'Safety':>7} {'PnL/MDD':>8} {'PnL':>8} "
      f"{'MDD':>7} {'WR':>6} {'Trades':>7} │ {'IS_PnL':>8}")
print("-" * 90)
for i, r in enumerate(by_safety, 1):
    p = r['pre']; s = r['is']
    print(f"{i:>3} {r['tp']:>5.1f} {r['sl']:>5.1f} │ {p['safety']:>+6.1f}pp {p['pnl_mdd']:>7.1f}x "
          f"{p['pnl']:>+7.1f}% {p['mdd']:>6.1f}% {p['wr']:>5.1f}% {p['trades']:>6} │ "
          f"{s['pnl']:>+7.1f}%")

results['top_pre_safety'] = by_safety[:20]

print(f"\n{'='*70}")
print("Top 20 by Pre-overlap Absolute PnL")
print("=" * 70)
print(f"\n{'#':>3} {'TP':>5} {'SL':>5} │ {'PnL':>8} {'PnL/MDD':>8} {'MDD':>7} "
      f"{'WR':>6} {'Safety':>7} {'Trades':>7} │ {'IS_PnL':>8}")
print("-" * 90)
for i, r in enumerate(by_pnl, 1):
    p = r['pre']; s = r['is']
    print(f"{i:>3} {r['tp']:>5.1f} {r['sl']:>5.1f} │ {p['pnl']:>+7.1f}% {p['pnl_mdd']:>7.1f}x "
          f"{p['mdd']:>6.1f}% {p['wr']:>5.1f}% {p['safety']:>+6.1f}pp {p['trades']:>6} │ "
          f"{s['pnl']:>+7.1f}%")

results['top_pre_pnl'] = by_pnl[:20]

# =======================================================================
# Heatmap data: pre_pnl_mdd for each (tp, sl)
# =======================================================================
heatmap = {}
for r in grid_data:
    key = f"{r['tp']:.1f}_{r['sl']:.1f}"
    heatmap[key] = {
        'pnl_mdd': r['pre']['pnl_mdd'],
        'safety': r['pre']['safety'],
        'pnl': r['pre']['pnl'],
        'trades': r['pre']['trades'],
    }
results['heatmap'] = heatmap

# =======================================================================
# Comparison: Current (2.0/3.0) vs neighbors
# =======================================================================
print(f"\n{'='*70}")
print("Neighborhood around TP 2.0/SL 3.0 (current setting)")
print("=" * 70)

print(f"\n{'TP':>5} {'SL':>5} │ {'PnL/MDD':>8} {'PnL':>8} {'MDD':>7} "
      f"{'WR':>6} {'Safety':>7} {'Trades':>7} │ {'IS_PnL':>8} {'Full_PnL':>9}")
print("-" * 95)
for r in grid_data:
    if 1.5 <= r['tp'] <= 2.5 and 2.5 <= r['sl'] <= 3.0:
        p = r['pre']; s = r['is']; f = r['full']
        flag = " ★" if r['tp'] == 2.0 and r['sl'] == 3.0 else ""
        print(f"{r['tp']:>5.1f} {r['sl']:>5.1f} │ {p['pnl_mdd']:>7.1f}x {p['pnl']:>+7.1f}% "
              f"{p['mdd']:>6.1f}% {p['wr']:>5.1f}% {p['safety']:>+6.1f}pp {p['trades']:>6} │ "
              f"{s['pnl']:>+7.1f}% {f['pnl']:>+8.1f}%{flag}")

# =======================================================================
# Summary
# =======================================================================
elapsed = time.time() - t0
best = by_pnl_mdd[0] if by_pnl_mdd else None
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Total combinations: {total_combos}")
print(f"Positive pre-PnL: {len(positive)}/{total_combos}")
if best:
    print(f"Best pre PnL/MDD: TP {best['tp']:.1f} / SL {best['sl']:.1f} "
          f"→ PnL/MDD {best['pre']['pnl_mdd']}x, PnL {best['pre']['pnl']:+.1f}%, "
          f"Safety {best['pre']['safety']:+.1f}pp, Trades {best['pre']['trades']}")
safest = by_safety[0] if by_safety else None
if safest:
    print(f"Best pre Safety:  TP {safest['tp']:.1f} / SL {safest['sl']:.1f} "
          f"→ Safety {safest['pre']['safety']:+.1f}pp, PnL/MDD {safest['pre']['pnl_mdd']}x, "
          f"PnL {safest['pre']['pnl']:+.1f}%, Trades {safest['pre']['trades']}")
# Current setting
cur = next((r for r in grid_data if r['tp'] == 2.0 and r['sl'] == 3.0), None)
if cur:
    print(f"Current (2.0/3.0): PnL/MDD {cur['pre']['pnl_mdd']}x, "
          f"PnL {cur['pre']['pnl']:+.1f}%, Safety {cur['pre']['safety']:+.1f}pp, "
          f"Trades {cur['pre']['trades']}")

results['summary'] = {
    'total_combos': total_combos,
    'positive_pre_pnl': len(positive),
    'best_pnl_mdd': {'tp': best['tp'], 'sl': best['sl'], **best['pre']} if best else None,
    'best_safety': {'tp': safest['tp'], 'sl': safest['sl'], **safest['pre']} if safest else None,
    'current_2_3': cur['pre'] if cur else None,
}
results['elapsed_seconds'] = round(elapsed, 1)

# Save
out_path = RESULTS_DIR / "universal_tpsl_fine_grid.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
print(f"Elapsed: {elapsed:.1f}s")
