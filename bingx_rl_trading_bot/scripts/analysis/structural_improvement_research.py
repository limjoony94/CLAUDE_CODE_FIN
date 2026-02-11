"""
Structural Improvement Research
================================
패턴/TP-SL 수준이 아닌 구조적 개선 가능성 탐색.
기존 연구에서 패턴 선택/제거/추가/SL조정 모두 WF 실패.
→ 1-pos-at-a-time 구조 자체를 변경하는 연구.

Phase 1: Multi-Position Study
  - 2-pos, 3-pos 동시 허용 시 성과 변화
  - 같은 방향만 / 양방향 허용 비교
  - 리스크 조정: 포지션 수에 반비례 사이징
  - WF validation

Phase 2: Time-Based Exit (Max Holding Period)
  - 현재: TP/SL 도달까지 무한 보유 (MAX_BARS=500)
  - 연구: max holding 12~288 bars (1시간~24시간)
  - 보유 시간 초과 시 시장가 청산
  - 기회비용 분석: 긴 보유가 다음 신호 차단
  - WF validation

Phase 3: ATR-Adaptive TP/SL
  - 현재: 고정 TP/SL %
  - 연구: ATR 기반 동적 TP/SL
  - ATR 높을 때 TP/SL 확대, 낮을 때 축소
  - WF validation

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
N_PERIODS = 5

np.random.seed(42)

print("=" * 70)
print("Structural Improvement Research")
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

# ATR calculation (14-period)
print("Computing ATR(14)...")
tr = np.maximum(highs - lows,
        np.maximum(np.abs(highs - np.roll(closes, 1)),
                   np.abs(lows - np.roll(closes, 1))))
tr[0] = highs[0] - lows[0]
atr14 = pd.Series(tr).rolling(14).mean().values
# ATR as % of price
atr_pct = atr14 / closes * 100
# Median ATR for normalization
median_atr_pct = np.nanmedian(atr_pct)
print(f"Median ATR(14): {median_atr_pct:.4f}%")
print("Done.\n")

# Build portfolio
PORTFOLIO = {}
ALL_PATTERNS = []
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("LONG", round(tp * TP_SCALE, 2), sl)
    ALL_PATTERNS.append(pat)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("SHORT", round(tp * TP_SCALE, 2), sl)
    ALL_PATTERNS.append(pat)

print(f"Portfolio: {len(PORTFOLIO)} patterns")

# ============================================================
# Precompute: simulate trade with optional max_hold and atr_scale
# ============================================================
def simulate_trade(direction, tp_pct, sl_pct, signal_bar, max_hold=MAX_BARS):
    """Returns (entry_bar, exit_bar, pnl) or None."""
    eb = signal_bar + 1
    if eb >= n_bars:
        return None
    entry = opens[eb]
    if entry <= 0:
        return None
    if direction == 'LONG':
        tpp = entry * (1 + tp_pct / 100)
        slp = entry * (1 - sl_pct / 100)
    else:
        tpp = entry * (1 - tp_pct / 100)
        slp = entry * (1 + sl_pct / 100)

    for j in range(eb + 1, min(eb + max_hold, n_bars)):
        ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
        hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
        if ht and hs:
            pnl = (tp_pct if abs(tpp - entry) <= abs(slp - entry) else -sl_pct) * LEVERAGE - FEE_PCT
            return (eb, j, pnl)
        elif ht:
            return (eb, j, tp_pct * LEVERAGE - FEE_PCT)
        elif hs:
            return (eb, j, -sl_pct * LEVERAGE - FEE_PCT)

    # Time exit: close at market
    last_bar = min(eb + max_hold, n_bars - 1)
    if last_bar > eb:
        exit_price = closes[last_bar]
        if direction == 'LONG':
            pnl_raw = (exit_price - entry) / entry * 100
        else:
            pnl_raw = (entry - exit_price) / entry * 100
        pnl = pnl_raw * LEVERAGE - FEE_PCT
        return (eb, last_bar, pnl)
    return None


def collect_signals(bar_min=0, bar_max=None):
    """Get all signal events: (signal_bar, pattern, direction, tp, sl)."""
    if bar_max is None:
        bar_max = n_bars - 1
    signals = []
    for i in range(max(2, bar_min), min(bar_max + 1, n_bars)):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat in PORTFOLIO:
            direction, tp, sl = PORTFOLIO[pat]
            signals.append((i, pat, direction, tp, sl))
    return signals


def run_1pos(signals, max_hold=MAX_BARS, atr_mult=None):
    """1-position-at-a-time execution."""
    trades = []
    last_exit = -1
    for sig_bar, pat, direction, tp, sl in signals:
        if sig_bar <= last_exit:
            continue
        # Optional ATR scaling
        if atr_mult is not None:
            a = atr_pct[sig_bar]
            if np.isnan(a) or a <= 0:
                continue
            scale = a / median_atr_pct * atr_mult
            tp_adj = tp * scale
            sl_adj = sl * scale
        else:
            tp_adj = tp
            sl_adj = sl

        result = simulate_trade(direction, tp_adj, sl_adj, sig_bar, max_hold)
        if result:
            trades.append((*result, pat))
            last_exit = result[1]
    return trades


def run_npos(signals, max_positions=2, max_hold=MAX_BARS, same_dir_only=False, size_adjust=True):
    """N-position simultaneous execution."""
    trades = []
    open_positions = []  # list of (exit_bar, direction)

    for sig_bar, pat, direction, tp, sl in signals:
        # Clean expired positions
        open_positions = [(xb, d) for xb, d in open_positions if xb >= sig_bar]

        if len(open_positions) >= max_positions:
            continue

        if same_dir_only and open_positions:
            # All open must be same direction
            if any(d != direction for _, d in open_positions):
                continue

        result = simulate_trade(direction, tp, sl, sig_bar, max_hold)
        if result:
            eb, xb, pnl = result
            # Size adjustment: divide PnL by position count at entry time
            if size_adjust:
                concurrent = len(open_positions) + 1
                pnl_adj = pnl / concurrent
            else:
                pnl_adj = pnl
            trades.append((eb, xb, pnl_adj, pat))
            open_positions.append((xb, direction))

    return trades


def eval_full(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_mdd': 0}
    pnls = [t[2] for t in trades]
    cum = peak = max_dd = win_sum = loss_sum = 0
    wins = 0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
        if p > 0:
            wins += 1
            win_sum += p
        else:
            loss_sum += abs(p)
    return {
        'pnl': round(cum, 2),
        'trades': len(pnls),
        'wr': round(wins / len(pnls) * 100, 1) if pnls else 0,
        'mdd': round(max_dd, 2),
        'pf': round(win_sum / loss_sum, 2) if loss_sum > 0 else 999,
        'pnl_mdd': round(cum / max_dd, 1) if max_dd > 0 else 999,
    }


def mc_test(pnl_list, n_sims=MC_SIMS):
    if len(pnl_list) < 8:
        return 1.0
    pa = np.array(pnl_list)
    real = np.sum(pa)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pa)))
    return float(np.mean(np.sum(pa * signs, axis=1) >= real))


# WF periods
period_size = n_bars // N_PERIODS
period_bounds = [(i * period_size, (i + 1) * period_size - 1) for i in range(N_PERIODS)]
period_bounds[-1] = (period_bounds[-1][0], n_bars - 1)

results = {}

# ============================================================
# Phase 1: Multi-Position Study
# ============================================================
print("=" * 70)
print("Phase 1: Multi-Position Study")
print("=" * 70)
print()

# 1a. Full sample comparison
print("  1a. Full sample (270d):")
print()

all_signals = collect_signals()
print(f"  Total signals: {len(all_signals)}")

multi_configs = [
    ('1-pos (current)', 1, False, True),
    ('2-pos any-dir', 2, False, True),
    ('2-pos same-dir', 2, True, True),
    ('3-pos any-dir', 3, False, True),
    ('3-pos same-dir', 3, True, True),
    ('2-pos no-adj', 2, False, False),
    ('3-pos no-adj', 3, False, False),
]

multi_results = {}
for name, max_pos, same_dir, size_adj in multi_configs:
    if max_pos == 1:
        trades = run_1pos(all_signals)
    else:
        trades = run_npos(all_signals, max_positions=max_pos, same_dir_only=same_dir, size_adjust=size_adj)
    stats = eval_full(trades)
    mc = mc_test([t[2] for t in trades])
    multi_results[name] = {**stats, 'mc': round(mc, 4)}

print(f"  {'Config':<20} {'PnL':>9} {'Trades':>7} {'WR':>6} {'MDD':>7} {'PF':>6} {'PnL/MDD':>8} {'MC':>8}")
print("  " + "-" * 80)
for name, stats in multi_results.items():
    print(f"  {name:<20} {stats['pnl']:>+8.1f}% {stats['trades']:>6} {stats['wr']:>5.1f}% "
          f"{stats['mdd']:>6.1f}% {stats['pf']:>5.2f} {stats['pnl_mdd']:>7.1f}x {stats['mc']:>7.4f}")

# 1b. WF validation of multi-position
print(f"\n  1b. WF OOS validation:")
print()

wf_configs = [
    ('1-pos (current)', 1, False, True),
    ('2-pos any-dir', 2, False, True),
    ('2-pos same-dir', 2, True, True),
    ('3-pos any-dir', 3, False, True),
]

phase1_wf = {}
for cfg_name, max_pos, same_dir, size_adj in wf_configs:
    all_oos_trades = []
    fold_stats = []

    for fold_idx in range(1, N_PERIODS):
        oos_start = period_bounds[fold_idx][0]
        oos_end = period_bounds[fold_idx][1]
        sigs = collect_signals(bar_min=oos_start, bar_max=oos_end)

        if max_pos == 1:
            oos_trades = run_1pos(sigs)
        else:
            oos_trades = run_npos(sigs, max_positions=max_pos, same_dir_only=same_dir, size_adjust=size_adj)

        stats = eval_full(oos_trades)
        fold_stats.append(stats)
        all_oos_trades.extend(oos_trades)

    agg = eval_full(all_oos_trades)
    oos_mc = mc_test([t[2] for t in all_oos_trades])
    phase1_wf[cfg_name] = {'folds': fold_stats, 'aggregate': agg, 'oos_mc': round(oos_mc, 4)}

print(f"  {'Config':<20} {'OOS_PnL':>9} {'Trades':>7} {'WR':>6} {'MDD':>7} {'PF':>6} {'PnL/MDD':>8} {'MC':>8}")
print("  " + "-" * 80)
for name, data in phase1_wf.items():
    a = data['aggregate']
    print(f"  {name:<20} {a['pnl']:>+8.1f}% {a['trades']:>6} {a['wr']:>5.1f}% "
          f"{a['mdd']:>6.1f}% {a['pf']:>5.2f} {a['pnl_mdd']:>7.1f}x {data['oos_mc']:>7.4f}")

# Per-fold detail
print(f"\n  Per-fold comparison:")
for fold_idx in range(N_PERIODS - 1):
    print(f"\n  Fold {fold_idx + 1}:")
    for name, data in phase1_wf.items():
        f = data['folds'][fold_idx]
        print(f"    {name:<20} PnL={f['pnl']:>+7.1f}% T={f['trades']:>4} WR={f['wr']:>5.1f}% MDD={f['mdd']:>5.1f}%")

results['phase1_full'] = multi_results
results['phase1_wf'] = {k: {'aggregate': v['aggregate'], 'oos_mc': v['oos_mc']} for k, v in phase1_wf.items()}

# ============================================================
# Phase 2: Time-Based Exit (Max Holding Period)
# ============================================================
print()
print("=" * 70)
print("Phase 2: Time-Based Exit (Max Holding Period)")
print("=" * 70)
print()

# 2a. Holding time distribution of current trades
current_trades = run_1pos(all_signals)
hold_times = [t[1] - t[0] for t in current_trades]  # exit - entry bars
print(f"  2a. Current holding time distribution:")
print(f"  Mean:   {np.mean(hold_times):.1f} bars ({np.mean(hold_times)*5:.0f} min)")
print(f"  Median: {np.median(hold_times):.0f} bars ({np.median(hold_times)*5:.0f} min)")
print(f"  P90:    {np.percentile(hold_times, 90):.0f} bars ({np.percentile(hold_times, 90)*5:.0f} min)")
print(f"  P99:    {np.percentile(hold_times, 99):.0f} bars ({np.percentile(hold_times, 99)*5:.0f} min)")
print(f"  Max:    {np.max(hold_times):.0f} bars ({np.max(hold_times)*5:.0f} min)")
print()

# PnL by holding time bucket
buckets = [(0, 12), (12, 36), (36, 72), (72, 144), (144, 288), (288, 9999)]
print(f"  PnL by holding time bucket:")
print(f"  {'Bucket':>16} {'Count':>6} {'AvgPnL':>8} {'WR':>6} {'TotalPnL':>10}")
print("  " + "-" * 55)
for lo, hi in buckets:
    bucket_trades = [t for t, h in zip(current_trades, hold_times) if lo <= h < hi]
    if not bucket_trades:
        continue
    pnls = [t[2] for t in bucket_trades]
    avg = np.mean(pnls)
    wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
    label = f"{lo*5//60}h-{hi*5//60}h" if hi < 9999 else f"{lo*5//60}h+"
    print(f"  {label:>16} {len(bucket_trades):>6} {avg:>+7.2f}% {wr:>5.1f}% {sum(pnls):>+9.1f}%")

# 2b. Max holding period sweep
print(f"\n  2b. Max holding period sweep (270d):")
print()

MAX_HOLD_LEVELS = [12, 24, 36, 48, 72, 96, 144, 192, 288, 500]

hold_results = {}
for mh in MAX_HOLD_LEVELS:
    trades = run_1pos(all_signals, max_hold=mh)
    stats = eval_full(trades)
    hold_results[mh] = stats

print(f"  {'MaxHold':>8} {'Hours':>6} {'PnL':>9} {'Trades':>7} {'WR':>6} {'MDD':>7} {'PnL/MDD':>8}")
print("  " + "-" * 60)
for mh, stats in hold_results.items():
    hours = mh * 5 / 60
    print(f"  {mh:>7}b {hours:>5.1f}h {stats['pnl']:>+8.1f}% {stats['trades']:>6} "
          f"{stats['wr']:>5.1f}% {stats['mdd']:>6.1f}% {stats['pnl_mdd']:>7.1f}x")

# 2c. WF validation of best holding periods
print(f"\n  2c. WF OOS validation:")
print()

WF_HOLD_LEVELS = [36, 72, 144, 288, 500]
phase2_wf = {}

for mh in WF_HOLD_LEVELS:
    all_oos = []
    for fold_idx in range(1, N_PERIODS):
        oos_start = period_bounds[fold_idx][0]
        oos_end = period_bounds[fold_idx][1]
        sigs = collect_signals(bar_min=oos_start, bar_max=oos_end)
        trades = run_1pos(sigs, max_hold=mh)
        all_oos.extend(trades)

    agg = eval_full(all_oos)
    oos_mc = mc_test([t[2] for t in all_oos])
    hours = mh * 5 / 60
    phase2_wf[mh] = {'aggregate': agg, 'oos_mc': round(oos_mc, 4)}

print(f"  {'MaxHold':>8} {'Hours':>6} {'OOS_PnL':>9} {'Trades':>7} {'WR':>6} {'MDD':>7} {'PnL/MDD':>8} {'MC':>8}")
print("  " + "-" * 70)
for mh, data in phase2_wf.items():
    a = data['aggregate']
    hours = mh * 5 / 60
    print(f"  {mh:>7}b {hours:>5.1f}h {a['pnl']:>+8.1f}% {a['trades']:>6} "
          f"{a['wr']:>5.1f}% {a['mdd']:>6.1f}% {a['pnl_mdd']:>7.1f}x {data['oos_mc']:>7.4f}")

results['phase2_hold_distribution'] = {
    'mean_bars': round(np.mean(hold_times), 1),
    'median_bars': round(np.median(hold_times), 0),
    'p90_bars': round(np.percentile(hold_times, 90), 0),
    'p99_bars': round(np.percentile(hold_times, 99), 0),
}
results['phase2_full'] = {str(k): v for k, v in hold_results.items()}
results['phase2_wf'] = {str(k): {'aggregate': v['aggregate'], 'oos_mc': v['oos_mc']} for k, v in phase2_wf.items()}

# ============================================================
# Phase 3: ATR-Adaptive TP/SL
# ============================================================
print()
print("=" * 70)
print("Phase 3: ATR-Adaptive TP/SL")
print("=" * 70)
print()

# 3a. ATR distribution and current TP/SL calibration
print(f"  3a. ATR(14) as % of price distribution:")
valid_atr = atr_pct[~np.isnan(atr_pct)]
print(f"  Mean:   {np.mean(valid_atr):.4f}%")
print(f"  Median: {np.median(valid_atr):.4f}%")
print(f"  P10:    {np.percentile(valid_atr, 10):.4f}%")
print(f"  P90:    {np.percentile(valid_atr, 90):.4f}%")
print(f"  Ratio P90/P10: {np.percentile(valid_atr, 90) / np.percentile(valid_atr, 10):.2f}x")
print()

# 3b. ATR multiplier sweep
print(f"  3b. ATR multiplier sweep (scale = ATR/median_ATR × mult):")
print(f"  mult=1.0: TP/SL scales linearly with ATR")
print(f"  mult=0.5: TP/SL scales at half ATR rate")
print()

ATR_MULTS = [0.3, 0.5, 0.7, 1.0, 1.3]

atr_results = {}
for mult in ATR_MULTS:
    trades = run_1pos(all_signals, atr_mult=mult)
    stats = eval_full(trades)
    atr_results[mult] = stats

# Also baseline (no ATR scaling)
base_stats = eval_full(run_1pos(all_signals))
atr_results['fixed'] = base_stats

print(f"  {'ATR_Mult':>9} {'PnL':>9} {'Trades':>7} {'WR':>6} {'MDD':>7} {'PF':>6} {'PnL/MDD':>8}")
print("  " + "-" * 60)
print(f"  {'fixed':>9} {atr_results['fixed']['pnl']:>+8.1f}% {atr_results['fixed']['trades']:>6} "
      f"{atr_results['fixed']['wr']:>5.1f}% {atr_results['fixed']['mdd']:>6.1f}% "
      f"{atr_results['fixed']['pf']:>5.2f} {atr_results['fixed']['pnl_mdd']:>7.1f}x")
for mult in ATR_MULTS:
    stats = atr_results[mult]
    print(f"  {mult:>9.1f} {stats['pnl']:>+8.1f}% {stats['trades']:>6} "
          f"{stats['wr']:>5.1f}% {stats['mdd']:>6.1f}% {stats['pf']:>5.2f} {stats['pnl_mdd']:>7.1f}x")

# 3c. WF validation
print(f"\n  3c. WF OOS validation:")
print()

WF_ATR = [None, 0.5, 0.7, 1.0]  # None = fixed
phase3_wf = {}

for mult in WF_ATR:
    all_oos = []
    for fold_idx in range(1, N_PERIODS):
        oos_start = period_bounds[fold_idx][0]
        oos_end = period_bounds[fold_idx][1]
        sigs = collect_signals(bar_min=oos_start, bar_max=oos_end)
        trades = run_1pos(sigs, atr_mult=mult)
        all_oos.extend(trades)

    agg = eval_full(all_oos)
    oos_mc = mc_test([t[2] for t in all_oos])
    label = 'fixed' if mult is None else f'{mult:.1f}'
    phase3_wf[label] = {'aggregate': agg, 'oos_mc': round(oos_mc, 4)}

print(f"  {'ATR_Mult':>9} {'OOS_PnL':>9} {'Trades':>7} {'WR':>6} {'MDD':>7} {'PF':>6} {'PnL/MDD':>8} {'MC':>8}")
print("  " + "-" * 70)
for label, data in phase3_wf.items():
    a = data['aggregate']
    print(f"  {label:>9} {a['pnl']:>+8.1f}% {a['trades']:>6} "
          f"{a['wr']:>5.1f}% {a['mdd']:>6.1f}% {a['pf']:>5.2f} {a['pnl_mdd']:>7.1f}x {data['oos_mc']:>7.4f}")

results['phase3_atr_dist'] = {
    'mean': round(float(np.mean(valid_atr)), 4),
    'median': round(float(np.median(valid_atr)), 4),
    'p10': round(float(np.percentile(valid_atr, 10)), 4),
    'p90': round(float(np.percentile(valid_atr, 90)), 4),
}
results['phase3_full'] = {str(k): v for k, v in atr_results.items()}
results['phase3_wf'] = phase3_wf

# ============================================================
# Phase 4: Combined Best Strategies
# ============================================================
print()
print("=" * 70)
print("Phase 4: Combined Best Strategies WF Test")
print("=" * 70)
print()

# Identify if any structural change outperformed baseline in WF
# Test combinations of best performers
base_oos_pnl = phase1_wf['1-pos (current)']['aggregate']['pnl']

combos = [
    ('Baseline (1pos, 500bar, fixed)', 1, False, 500, None),
]

# Add best multi-pos if it beat baseline
for cfg_name, data in phase1_wf.items():
    if data['aggregate']['pnl'] > base_oos_pnl and cfg_name != '1-pos (current)':
        parts = cfg_name.split()
        mp = int(parts[0].split('-')[0])
        sd = 'same' in cfg_name
        combos.append((f'Best multi ({cfg_name})', mp, sd, 500, None))
        break

# Add best time exit if it beat baseline
for mh, data in phase2_wf.items():
    if data['aggregate']['pnl'] > base_oos_pnl and mh != 500:
        combos.append((f'Best time-exit ({mh}b)', 1, False, mh, None))
        break

# Add best ATR if it beat baseline
for label, data in phase3_wf.items():
    if data['aggregate']['pnl'] > base_oos_pnl and label != 'fixed':
        combos.append((f'Best ATR (mult={label})', 1, False, 500, float(label)))
        break

# Test combinations
phase4 = {}
for combo_name, max_pos, same_dir, max_hold, atr_m in combos:
    all_oos = []
    for fold_idx in range(1, N_PERIODS):
        oos_start = period_bounds[fold_idx][0]
        oos_end = period_bounds[fold_idx][1]
        sigs = collect_signals(bar_min=oos_start, bar_max=oos_end)
        if max_pos == 1:
            trades = run_1pos(sigs, max_hold=max_hold, atr_mult=atr_m)
        else:
            trades = run_npos(sigs, max_positions=max_pos, same_dir_only=same_dir, max_hold=max_hold, size_adjust=True)
        all_oos.extend(trades)

    agg = eval_full(all_oos)
    oos_mc = mc_test([t[2] for t in all_oos])
    phase4[combo_name] = {'aggregate': agg, 'oos_mc': round(oos_mc, 4)}

print(f"  {'Strategy':<35} {'OOS_PnL':>9} {'Trades':>7} {'WR':>6} {'MDD':>7} {'PnL/MDD':>8}")
print("  " + "-" * 80)
for name, data in phase4.items():
    a = data['aggregate']
    print(f"  {name:<35} {a['pnl']:>+8.1f}% {a['trades']:>6} {a['wr']:>5.1f}% "
          f"{a['mdd']:>6.1f}% {a['pnl_mdd']:>7.1f}x")

results['phase4'] = phase4

# ============================================================
# Save
# ============================================================
out_path = Path(__file__).resolve().parent / '../../results/structural_improvement_research.json'
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
print(f"  Baseline WF OOS: PnL={base_oos_pnl:+.1f}%")
print()
print(f"  Phase 1 (Multi-Pos): ", end="")
best_mp = max(phase1_wf.items(), key=lambda x: x[1]['aggregate']['pnl'])
print(f"Best={best_mp[0]} OOS {best_mp[1]['aggregate']['pnl']:+.1f}% (diff {best_mp[1]['aggregate']['pnl'] - base_oos_pnl:+.1f}%)")

print(f"  Phase 2 (Time Exit): ", end="")
best_te = max(phase2_wf.items(), key=lambda x: x[1]['aggregate']['pnl'])
print(f"Best={best_te[0]}bars OOS {best_te[1]['aggregate']['pnl']:+.1f}% (diff {best_te[1]['aggregate']['pnl'] - base_oos_pnl:+.1f}%)")

print(f"  Phase 3 (ATR-Adpt): ", end="")
best_atr = max(phase3_wf.items(), key=lambda x: x[1]['aggregate']['pnl'])
print(f"Best={best_atr[0]} OOS {best_atr[1]['aggregate']['pnl']:+.1f}% (diff {best_atr[1]['aggregate']['pnl'] - base_oos_pnl:+.1f}%)")
print("=" * 70)
