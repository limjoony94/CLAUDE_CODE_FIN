#!/usr/bin/env python3
"""
Universal TP/SL Fine Grid — Compound Returns
=============================================
이전 fine grid (simple returns)에서 MDD > 100% 문제 발견.
실제 트레이딩은 compound이므로 compound 기준으로 재계산.

Compound PnL: equity *= (1 + trade_pnl_pct/100)
Compound MDD: max peak-to-trough of equity curve (% of peak)

0.1 단위 full grid (28×28 = 784 조합)
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

FEE_PCT = 0.10   # 0.10% round-trip fee
LEVERAGE = 3
MAX_BARS = 500
POSITION_SIZE = 0.95  # 95% of equity per trade

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
    """Collect trades with 1-position-at-a-time, universal TP/SL.
    Returns list of (entry_bar, exit_bar, pnl_pct) where pnl_pct is the
    percentage return on the POSITION (leveraged, after fees)."""
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
                bo = opens[j]; pnl = (tp_pct if abs(tpp - bo) <= abs(slp - bo) else -sl_pct) * LEVERAGE - FEE_PCT
                raw.append((eb, j, pnl)); break
            elif ht:
                raw.append((eb, j, tp_pct * LEVERAGE - FEE_PCT)); break
            elif hs:
                raw.append((eb, j, -sl_pct * LEVERAGE - FEE_PCT)); break
    raw.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl in raw:
        if eb > last_exit:
            filtered.append((eb, xb, pnl))
            last_exit = xb
    return filtered


def calc_stats_compound(trades):
    """Calculate stats using compound (geometric) returns.
    Each trade's PnL % is applied to current equity.
    Position size = POSITION_SIZE (95%) of equity."""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'pnl_mdd': 0, 'avg_win': 0, 'avg_loss': 0, 'safety': 0,
                'final_equity': 100, 'min_equity': 100}
    equity = 100.0
    peak_equity = 100.0
    mdd = 0.0
    min_equity = 100.0
    wins_pnl = []; losses_pnl = []

    for _, _, pnl_pct in trades:
        # pnl_pct is leveraged position return (e.g. +5.9% or -9.1%)
        # Apply to equity with position sizing
        trade_return = pnl_pct / 100 * POSITION_SIZE
        equity *= (1 + trade_return)

        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > mdd:
            mdd = dd
        if equity < min_equity:
            min_equity = equity

        if pnl_pct > 0:
            wins_pnl.append(pnl_pct)
        else:
            losses_pnl.append(pnl_pct)

    total_pnl = (equity / 100 - 1) * 100  # % return from starting equity
    n_trades = len(trades)
    wr = len(wins_pnl) / n_trades * 100 if n_trades > 0 else 0
    avg_w = float(np.mean(wins_pnl)) if wins_pnl else 0
    avg_l = float(np.mean(losses_pnl)) if losses_pnl else 0
    be_wr = abs(avg_l) / (avg_w + abs(avg_l)) * 100 if (avg_w + abs(avg_l)) > 0 else 50
    wsum = sum(wins_pnl); lsum = sum(abs(x) for x in losses_pnl)

    return {
        'pnl': round(total_pnl, 1),
        'trades': n_trades,
        'wr': round(wr, 1),
        'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
        'pnl_mdd': round(total_pnl / mdd, 1) if mdd > 0 else 999,
        'avg_win': round(avg_w, 3),
        'avg_loss': round(avg_l, 3),
        'safety': round(wr - be_wr, 1),
        'final_equity': round(equity, 1),
        'min_equity': round(min_equity, 1),
    }


# =======================================================================
# MAIN
# =======================================================================
print("=" * 70)
print("Universal TP/SL Fine Grid — COMPOUND Returns")
print(f"Position size: {POSITION_SIZE*100:.0f}% | Leverage: {LEVERAGE}x | Fee: {FEE_PCT}%")
print("=" * 70)
t0 = time.time()

print("\nLoading 720d Binance data...", flush=True)
bn_path = DATA_DIR / "btc_5m_720days_binance.csv"
opens, highs, lows, closes, types, timestamps, n_bars = load_and_classify(bn_path)

overlap_start_ts = np.datetime64('2025-05-05T15:00:00')
overlap_end_ts = np.datetime64('2026-01-30T14:55:00')
overlap_start_bar = int(np.searchsorted(timestamps, overlap_start_ts))
overlap_end_bar = int(np.searchsorted(timestamps, overlap_end_ts, side='right'))

print(f"Total: {n_bars} bars")
print(f"Pre-overlap (OOS): ~{overlap_start_bar/288:.0f}d | In-sample: ~{(overlap_end_bar-overlap_start_bar)/288:.0f}d")

tp_values = [round(0.3 + i * 0.1, 1) for i in range(28)]
sl_values = [round(0.3 + i * 0.1, 1) for i in range(28)]
total_combos = len(tp_values) * len(sl_values)
print(f"Grid: {len(tp_values)}×{len(sl_values)} = {total_combos} combinations\n")

grid_data = []
done = 0

print(f"{'TP':>5} {'SL':>5} │ {'Pre PnL':>8} {'Pre MDD':>8} {'P/M':>6} "
      f"{'WR':>6} {'Safety':>7} {'Trades':>6} {'MinEq':>7} │ "
      f"{'IS PnL':>8} {'Full PnL':>9}", flush=True)
print("-" * 100)

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

        pre_s = calc_stats_compound(pre_trades)
        is_s = calc_stats_compound(is_trades)
        full_s = calc_stats_compound(full_trades)

        row = {'tp': tp, 'sl': sl, 'pre': pre_s, 'is': is_s, 'full': full_s}
        grid_data.append(row)
        done += 1

        if pre_s['pnl'] > 0:
            print(f"{tp:>5.1f} {sl:>5.1f} │ {pre_s['pnl']:>+7.1f}% {pre_s['mdd']:>7.1f}% "
                  f"{pre_s['pnl_mdd']:>5.1f}x {pre_s['wr']:>5.1f}% {pre_s['safety']:>+6.1f}pp "
                  f"{pre_s['trades']:>5} {pre_s['min_equity']:>6.1f} │ "
                  f"{is_s['pnl']:>+7.1f}% {full_s['pnl']:>+8.1f}%", flush=True)

    pct = done / total_combos * 100
    elapsed = time.time() - t0
    eta = elapsed / done * (total_combos - done) if done > 0 else 0
    print(f"  ... TP={tp:.1f} done ({pct:.0f}%, ETA {eta:.0f}s)", flush=True)

# =======================================================================
# Rankings
# =======================================================================
positive = [r for r in grid_data if r['pre']['pnl'] > 0]

# Filter: MDD < 50% (realistic constraint)
safe = [r for r in positive if r['pre']['mdd'] < 50]

print(f"\n{'='*70}")
print(f"RANKINGS — Compound Returns (MDD < 50% filter: {len(safe)}/{len(positive)} pass)")
print(f"{'='*70}")

# By PnL/MDD
by_pnl_mdd = sorted(safe, key=lambda x: x['pre']['pnl_mdd'], reverse=True)[:20]
print(f"\nTop 20 by Pre PnL/MDD (compound, MDD<50%)")
print(f"{'#':>3} {'TP':>5} {'SL':>5} │ {'P/M':>6} {'PnL':>8} {'MDD':>7} "
      f"{'WR':>6} {'Safety':>7} {'Trades':>6} {'MinEq':>7} │ {'IS PnL':>8} {'Full':>9}")
print("-" * 100)
for i, r in enumerate(by_pnl_mdd, 1):
    p = r['pre']; s = r['is']; f = r['full']
    print(f"{i:>3} {r['tp']:>5.1f} {r['sl']:>5.1f} │ {p['pnl_mdd']:>5.1f}x {p['pnl']:>+7.1f}% "
          f"{p['mdd']:>6.1f}% {p['wr']:>5.1f}% {p['safety']:>+6.1f}pp {p['trades']:>5} "
          f"{p['min_equity']:>6.1f} │ {s['pnl']:>+7.1f}% {f['pnl']:>+8.1f}%")

# By Safety
by_safety = sorted(safe, key=lambda x: x['pre']['safety'], reverse=True)[:20]
print(f"\nTop 20 by Pre Safety (compound, MDD<50%)")
print(f"{'#':>3} {'TP':>5} {'SL':>5} │ {'Safety':>7} {'P/M':>6} {'PnL':>8} "
      f"{'MDD':>7} {'WR':>6} {'Trades':>6} │ {'IS PnL':>8}")
print("-" * 90)
for i, r in enumerate(by_safety, 1):
    p = r['pre']; s = r['is']
    print(f"{i:>3} {r['tp']:>5.1f} {r['sl']:>5.1f} │ {p['safety']:>+6.1f}pp {p['pnl_mdd']:>5.1f}x "
          f"{p['pnl']:>+7.1f}% {p['mdd']:>6.1f}% {p['wr']:>5.1f}% {p['trades']:>5} │ "
          f"{s['pnl']:>+7.1f}%")

# Also show without MDD filter for comparison
by_pnl_mdd_all = sorted(positive, key=lambda x: x['pre']['pnl_mdd'], reverse=True)[:20]
print(f"\nTop 20 by Pre PnL/MDD (compound, NO MDD filter)")
print(f"{'#':>3} {'TP':>5} {'SL':>5} │ {'P/M':>6} {'PnL':>8} {'MDD':>7} "
      f"{'WR':>6} {'Safety':>7} {'Trades':>6} {'MinEq':>7} │ {'IS PnL':>8} {'Full':>9}")
print("-" * 100)
for i, r in enumerate(by_pnl_mdd_all, 1):
    p = r['pre']; s = r['is']; f = r['full']
    mdd_flag = " !" if p['mdd'] >= 50 else ""
    print(f"{i:>3} {r['tp']:>5.1f} {r['sl']:>5.1f} │ {p['pnl_mdd']:>5.1f}x {p['pnl']:>+7.1f}% "
          f"{p['mdd']:>6.1f}% {p['wr']:>5.1f}% {p['safety']:>+6.1f}pp {p['trades']:>5} "
          f"{p['min_equity']:>6.1f} │ {s['pnl']:>+7.1f}% {f['pnl']:>+8.1f}%{mdd_flag}")

# =======================================================================
# Neighborhood around current
# =======================================================================
print(f"\n{'='*70}")
print("Neighborhood: TP 1.5~2.5, SL 2.5~3.0 (compound)")
print("=" * 70)
print(f"{'TP':>5} {'SL':>5} │ {'P/M':>6} {'PnL':>8} {'MDD':>7} "
      f"{'WR':>6} {'Safety':>7} {'Trades':>6} {'MinEq':>7} │ {'IS PnL':>8} {'Full':>9}")
print("-" * 100)
for r in grid_data:
    if 1.5 <= r['tp'] <= 2.5 and 2.5 <= r['sl'] <= 3.0:
        p = r['pre']; s = r['is']; f = r['full']
        flag = " ★" if r['tp'] == 2.0 and r['sl'] == 3.0 else ""
        if p['pnl'] > 0:
            print(f"{r['tp']:>5.1f} {r['sl']:>5.1f} │ {p['pnl_mdd']:>5.1f}x {p['pnl']:>+7.1f}% "
                  f"{p['mdd']:>6.1f}% {p['wr']:>5.1f}% {p['safety']:>+6.1f}pp {p['trades']:>5} "
                  f"{p['min_equity']:>6.1f} │ {s['pnl']:>+7.1f}% {f['pnl']:>+8.1f}%{flag}")

# =======================================================================
# Summary
# =======================================================================
elapsed = time.time() - t0
best_safe = by_pnl_mdd[0] if by_pnl_mdd else None
best_all = by_pnl_mdd_all[0] if by_pnl_mdd_all else None
cur = next((r for r in grid_data if r['tp'] == 2.0 and r['sl'] == 3.0), None)

print(f"\n{'='*70}")
print("SUMMARY (Compound Returns)")
print(f"{'='*70}")
print(f"Positive pre-PnL: {len(positive)}/{total_combos}")
print(f"MDD < 50% filter: {len(safe)}/{len(positive)}")
if best_safe:
    p = best_safe['pre']
    print(f"Best (MDD<50%): TP {best_safe['tp']:.1f}/SL {best_safe['sl']:.1f} "
          f"→ P/M {p['pnl_mdd']}x, PnL {p['pnl']:+.1f}%, MDD {p['mdd']:.1f}%, "
          f"Safety {p['safety']:+.1f}pp, Trades {p['trades']}, MinEq ${p['min_equity']:.0f}")
if best_all:
    p = best_all['pre']
    print(f"Best (no filter): TP {best_all['tp']:.1f}/SL {best_all['sl']:.1f} "
          f"→ P/M {p['pnl_mdd']}x, PnL {p['pnl']:+.1f}%, MDD {p['mdd']:.1f}%")
if cur:
    p = cur['pre']
    print(f"Current 2.0/3.0:  P/M {p['pnl_mdd']}x, PnL {p['pnl']:+.1f}%, "
          f"MDD {p['mdd']:.1f}%, Safety {p['safety']:+.1f}pp, Trades {p['trades']}, "
          f"MinEq ${p['min_equity']:.0f}")

results = {
    'grid': grid_data,
    'top_pnl_mdd_safe': [{'tp': r['tp'], 'sl': r['sl'], **r['pre']} for r in by_pnl_mdd[:20]],
    'top_pnl_mdd_all': [{'tp': r['tp'], 'sl': r['sl'], **r['pre']} for r in by_pnl_mdd_all[:20]],
    'top_safety': [{'tp': r['tp'], 'sl': r['sl'], **r['pre']} for r in by_safety[:20]],
    'summary': {
        'total_combos': total_combos,
        'positive_pre': len(positive),
        'mdd_under_50': len(safe),
        'best_safe': {'tp': best_safe['tp'], 'sl': best_safe['sl'], **best_safe['pre']} if best_safe else None,
        'current_2_3': cur['pre'] if cur else None,
    },
    'params': {
        'position_size': POSITION_SIZE,
        'leverage': LEVERAGE,
        'fee_pct': FEE_PCT,
    },
    'elapsed_seconds': round(elapsed, 1),
}
out_path = RESULTS_DIR / "universal_tpsl_fine_grid_compound.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
print(f"Elapsed: {elapsed:.1f}s")
