"""
Risk Management Research v1.26.4
=================================
v1.26.4 종합 리스크 관리 연구

배경: 라이브 봇에서 2연속 SL (-6.49% + -9.30% = -15.79%) 발생, 10% daily loss limit 트리거.
원인: 52개 패턴 중 26개(50%)가 SL >= 3.0% → 3x 레버리지로 1회 손실 = -9%.

Phase 1: Daily Loss Limit 최적화
Phase 2: SL 확대 영향 분석 (v1.26.2 vs v1.26.4)
Phase 3: 포트폴리오 리스크 관리 (연속손실, MC, 파산확률, Kelly, 회복, 리스크지표)
Phase 4: 종합 권고사항
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
    VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
    PATTERN_OPTIMAL_TPSL, BOT_VERSION,
)

# ============================================================
# Parameters (same as portfolio_pruning_v4.py)
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10          # 0.05% * 2 (round trip)
SLIPPAGE = 0.02
LEVERAGE = 3

print("=" * 70)
print(f"Risk Management Research (v{BOT_VERSION})")
print(f"  Patterns: {len(VALIDATED_LONG_PATTERNS)}L + {len(VALIDATED_SHORT_PATTERNS)}S = "
      f"{len(VALIDATED_LONG_PATTERNS) + len(VALIDATED_SHORT_PATTERNS)}")
print("=" * 70)

# ============================================================
# Build portfolio map
# ============================================================
PORTFOLIO = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("LONG", tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("SHORT", tp, sl)

# v1.26.2 original TP/SL (before v1.26.4 deep validation optimization)
# Source: CLAUDE.md v1.26.1 T5_Optimized table (v1.26.2 only removed patterns, didn't change TP/SL)
V1262_TPSL = {
    # LONG patterns
    "BD-BD-U":   (1.5, 1.5),
    "BD-MU-BD":  (1.0, 0.7),
    "BD-ST-U":   (1.5, 1.5),
    "BU-BU-BD":  (3.0, 2.5),
    "D-MU-U":    (1.5, 2.0),
    "DN-BD-BD":  (2.0, 1.0),
    "DN-DF-MU":  (1.5, 1.5),
    "DN-DF-ST":  (2.0, 1.5),
    "DN-DN-H":   (1.0, 1.0),
    "DN-MD-DN":  (1.5, 2.0),
    "GS-ST-ST":  (1.5, 2.0),
    "H-BU-BU":   (1.5, 1.0),
    "H-MU-MD":   (0.7, 0.5),
    "IH-MD-MD":  (1.0, 0.5),
    "IH-ST-MU":  (0.5, 0.3),
    "MD-BU-MD":  (2.0, 1.5),
    "MD-DN-MU":  (1.0, 1.0),
    "MD-H-MD":   (1.0, 0.5),
    "MD-MD-ST":  (1.5, 1.0),
    "MD-ST-BD":  (1.0, 0.5),
    "MD-ST-MD":  (2.0, 2.0),
    "MU-BD-ST":  (2.5, 3.0),
    "MU-DF-U":   (1.5, 0.7),
    "MU-H-MU":   (1.5, 0.7),
    "MU-IH-DN":  (2.0, 1.5),
    "MU-U-H":    (2.5, 3.0),
    "U-H-MU":    (1.5, 2.0),
    "U-MD-GS":   (0.5, 0.3),
    "U-MD-MD":   (1.5, 0.7),
    "U-MU-H":    (2.0, 0.5),
    "U-MU-IH":   (3.0, 0.3),
    "U-ST-DF":   (2.0, 2.5),
    # SHORT patterns
    "BD-BU-DN":  (3.0, 3.0),
    "BD-D-D":    (1.5, 1.5),
    "BD-U-H":    (2.5, 3.0),
    "BU-MD-MD":  (3.0, 2.0),
    "BU-ST-GS":  (0.5, 0.5),
    "D-BD-ST":   (2.5, 3.0),
    "D-DN-DN":   (2.5, 3.0),
    "DN-BD-BU":  (2.5, 3.0),
    "DN-D-BD":   (2.5, 0.3),
    "DN-DF-DN":  (2.0, 2.0),
    "H-U-BD":    (3.0, 2.0),
    "IH-ST-ST":  (2.0, 2.5),
    "MD-MD-MD":  (3.0, 3.0),
    "ST-BD-BU":  (3.0, 3.0),
    "ST-DN-BU":  (2.0, 2.5),
    "ST-DN-U":   (3.0, 3.0),
    "ST-MU-ST":  (2.0, 2.5),
    "U-GS-DN":   (3.0, 3.0),
    "U-H-BU":    (1.0, 0.3),
    "U-ST-DN":   (3.0, 3.0),
}

# ============================================================
# Load and classify data (same pipeline as portfolio_pruning_v4.py)
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
print(f"Loaded: {len(df)} bars")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

df['timestamp'] = pd.to_datetime(df['timestamp'])
timestamps = df['timestamp'].values

print("Classifying candles...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values

types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)

print("Classification done.")


# ============================================================
# Backtest engine (from portfolio_pruning_v4.py)
# ============================================================
def bt_fixed(indices, direction, tp_pct, sl_pct):
    pnls = []
    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)
        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnls.append((tp_pct if abs(tpp - entry) <= abs(slp - entry) else -sl_pct) * LEVERAGE - FEE_PCT)
                break
            elif ht:
                pnls.append(tp_pct * LEVERAGE - FEE_PCT)
                break
            elif hs:
                pnls.append(-sl_pct * LEVERAGE - FEE_PCT)
                break
    return pnls


# ============================================================
# BACKBONE: collect_trades_extended()
# 1-pos-at-a-time with metadata (pattern, direction, TP/SL, date)
# ============================================================
def collect_trades_extended(pattern_map):
    """Collect trades with full metadata. 1-position-at-a-time."""
    raw_trades = []
    for i in range(2, n_bars):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in pattern_map:
            continue
        direction, tp, sl = pattern_map[pat]
        if i + 1 >= n_bars:
            continue
        entry_price = opens[i + 1]
        if entry_price <= 0:
            continue
        entry_bar = i + 1
        if direction == 'LONG':
            tpp = entry_price * (1 + tp / 100)
            slp = entry_price * (1 - sl / 100)
        else:
            tpp = entry_price * (1 - tp / 100)
            slp = entry_price * (1 + sl / 100)
        for j in range(i + 2, min(i + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            hit = False
            if ht and hs:
                pnl = (tp if abs(tpp - entry_price) <= abs(slp - entry_price) else -sl) * LEVERAGE - FEE_PCT
                hit = True
            elif ht:
                pnl = tp * LEVERAGE - FEE_PCT
                hit = True
            elif hs:
                pnl = -sl * LEVERAGE - FEE_PCT
                hit = True
            if hit:
                raw_trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': j,
                    'pnl': pnl,
                    'pattern': pat,
                    'direction': direction,
                    'tp': tp,
                    'sl': sl,
                    'entry_price': entry_price,
                    'date': str(timestamps[entry_bar])[:10],
                    'is_win': pnl > 0,
                })
                break
    # Sort by entry_bar
    raw_trades.sort(key=lambda x: x['entry_bar'])
    # 1-position-at-a-time filter
    filtered = []
    last_exit = -1
    for t in raw_trades:
        if t['entry_bar'] > last_exit:
            filtered.append(t)
            last_exit = t['exit_bar']
    return filtered


def eval_trades_simple(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0, 'expectancy': 0, 'pnl_mdd': 0}
    pnl_list = [t['pnl'] if isinstance(t, dict) else t[2] for t in trades]
    cum_pnl = 0
    peak_pnl = 0
    max_dd = 0
    wins = 0
    win_pnls = []
    loss_pnls = []
    for p in pnl_list:
        cum_pnl += p
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        dd = peak_pnl - cum_pnl
        if dd > max_dd:
            max_dd = dd
        if p > 0:
            wins += 1
            win_pnls.append(p)
        else:
            loss_pnls.append(abs(p))
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    total_win = sum(win_pnls)
    total_loss = sum(loss_pnls)
    wr_pct = wins / len(pnl_list) * 100
    expectancy = (wr_pct / 100 * avg_win) - ((1 - wr_pct / 100) * avg_loss)
    return {
        'pnl': cum_pnl,
        'trades': len(pnl_list),
        'wr': wr_pct,
        'mdd': max_dd,
        'pf': total_win / total_loss if total_loss > 0 else 999,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'pnl_mdd': cum_pnl / max_dd if max_dd > 0 else 999,
    }


# ============================================================
# Collect trades (backbone for all phases)
# ============================================================
print("\nCollecting extended trades (1-pos-at-a-time)...")
t0 = time.time()
trades = collect_trades_extended(PORTFOLIO)
elapsed = time.time() - t0
stats = eval_trades_simple(trades)
print(f"  {stats['trades']} trades in {elapsed:.1f}s")
print(f"  PnL: {stats['pnl']:+.1f}%, WR: {stats['wr']:.1f}%, MDD: {stats['mdd']:.1f}%, PF: {stats['pf']:.2f}")


# ============================================================
# Phase 1: Daily Loss Limit Optimization
# ============================================================
print(f"\n{'='*70}")
print("Phase 1: Daily Loss Limit Optimization")
print(f"{'='*70}")


def group_trades_by_day(trade_list):
    """Group trades by date."""
    by_day = defaultdict(list)
    for t in trade_list:
        by_day[t['date']].append(t)
    return dict(sorted(by_day.items()))


def simulate_daily_loss_limit(trade_list, limit_pct):
    """Simulate with daily loss limit. Returns filtered trades and stats."""
    by_day = group_trades_by_day(trade_list)
    result_trades = []
    skipped = 0
    trigger_days = 0
    trigger_details = []

    for day, day_trades in by_day.items():
        day_pnl = 0
        triggered = False
        for t in day_trades:
            if triggered:
                skipped += 1
                continue
            result_trades.append(t)
            day_pnl += t['pnl']
            if limit_pct is not None and day_pnl <= -limit_pct:
                triggered = True
                trigger_days += 1
                trigger_details.append({
                    'date': day,
                    'pnl_at_trigger': round(day_pnl, 2),
                    'trades_taken': len([x for x in day_trades if x in result_trades]),
                    'trades_skipped': len(day_trades) - len([x for x in day_trades if x in result_trades]),
                })

    return result_trades, skipped, trigger_days, trigger_details


def analyze_post_trigger_recovery(trade_list, limit_pct):
    """Analyze what happens after a daily limit trigger day."""
    by_day = group_trades_by_day(trade_list)
    days = sorted(by_day.keys())

    trigger_days = []
    for day, day_trades in by_day.items():
        day_pnl = sum(t['pnl'] for t in day_trades)
        if limit_pct is not None and day_pnl <= -limit_pct:
            trigger_days.append(day)

    if not trigger_days:
        return {'trigger_days': 0, 'next_day_positive': 0, 'next_3_days_positive': 0}

    next_day_positive = 0
    next_3_days_positive = 0

    for trigger_day in trigger_days:
        idx = days.index(trigger_day) if trigger_day in days else -1
        if idx < 0:
            continue
        # Next day
        if idx + 1 < len(days):
            next_pnl = sum(t['pnl'] for t in by_day[days[idx + 1]])
            if next_pnl > 0:
                next_day_positive += 1
        # Next 3 days
        next_3_pnl = 0
        for offset in range(1, 4):
            if idx + offset < len(days):
                next_3_pnl += sum(t['pnl'] for t in by_day[days[idx + offset]])
        if next_3_pnl > 0:
            next_3_days_positive += 1

    return {
        'trigger_days': len(trigger_days),
        'next_day_positive': next_day_positive,
        'next_day_recovery_rate': round(next_day_positive / len(trigger_days) * 100, 1) if trigger_days else 0,
        'next_3_days_positive': next_3_days_positive,
        'next_3_days_recovery_rate': round(next_3_days_positive / len(trigger_days) * 100, 1) if trigger_days else 0,
    }


def run_daily_limit_sweep(trade_list):
    """Sweep daily loss limits: 5/7/10/12/15/20/25/None."""
    limits = [5, 7, 10, 12, 15, 20, 25, None]
    results = {}

    print(f"\n  {'Limit':>8} {'Trades':>7} {'Skip':>5} {'Trig':>5} {'PnL':>10} {'MDD':>8} {'P/M':>7} {'PF':>6} {'WR':>6} | {'Recov1d':>8} {'Recov3d':>8}")
    print("  " + "-" * 100)

    for limit in limits:
        label = f"{limit}%" if limit is not None else "None"
        filtered, skipped, trigger_days, trigger_details = simulate_daily_loss_limit(trade_list, limit)
        s = eval_trades_simple(filtered)
        recovery = analyze_post_trigger_recovery(trade_list, limit)

        print(f"  {label:>8} {s['trades']:>7} {skipped:>5} {trigger_days:>5} "
              f"{s['pnl']:>+9.1f}% {s['mdd']:>7.1f}% {s['pnl_mdd']:>6.2f}x {s['pf']:>5.2f} {s['wr']:>5.1f}% | "
              f"{recovery.get('next_day_recovery_rate', 0):>6.1f}% {recovery.get('next_3_days_recovery_rate', 0):>6.1f}%")

        results[label] = {
            'limit_pct': limit,
            'trades': s['trades'],
            'skipped': skipped,
            'trigger_days': trigger_days,
            'pnl': round(s['pnl'], 1),
            'mdd': round(s['mdd'], 1),
            'pnl_mdd': round(s['pnl_mdd'], 2),
            'pf': round(s['pf'], 2),
            'wr': round(s['wr'], 1),
            'avg_win': round(s['avg_win'], 2),
            'avg_loss': round(s['avg_loss'], 2),
            'recovery': recovery,
            'trigger_details': trigger_details,
        }

    return results


phase1 = run_daily_limit_sweep(trades)


# ============================================================
# Phase 2: SL Expansion Impact Analysis
# ============================================================
print(f"\n{'='*70}")
print("Phase 2: SL Expansion Impact Analysis (v1.26.2 vs v1.26.4)")
print(f"{'='*70}")


def compare_tpsl_versions():
    """Compare v1.26.2 vs v1.26.4 TP/SL differences."""
    changes = []
    for pat in sorted(PORTFOLIO.keys()):
        direction = PORTFOLIO[pat][0]
        tp_new, sl_new = PATTERN_OPTIMAL_TPSL[pat]
        tp_old, sl_old = V1262_TPSL.get(pat, (tp_new, sl_new))
        if tp_old != tp_new or sl_old != sl_new:
            changes.append({
                'pattern': pat,
                'direction': direction,
                'tp_old': tp_old, 'sl_old': sl_old,
                'tp_new': tp_new, 'sl_new': sl_new,
                'tp_change': tp_new - tp_old,
                'sl_change': sl_new - sl_old,
                'rr_old': round(tp_old / sl_old, 2) if sl_old > 0 else 999,
                'rr_new': round(tp_new / sl_new, 2) if sl_new > 0 else 999,
                'max_loss_old': round(sl_old * LEVERAGE, 2),
                'max_loss_new': round(sl_new * LEVERAGE, 2),
            })

    print(f"\n  TP/SL Changes: {len(changes)} patterns modified")
    print(f"\n  {'Pattern':<12} {'Dir':>5} {'TP old':>7} {'TP new':>7} {'SL old':>7} {'SL new':>7} {'MaxL old':>8} {'MaxL new':>8} {'SL chg':>7}")
    print("  " + "-" * 85)

    sl_increased = 0
    sl_decreased = 0
    for c in changes:
        flag = ""
        if c['sl_change'] > 0:
            flag = " [SL+]"
            sl_increased += 1
        elif c['sl_change'] < 0:
            sl_decreased += 1
        print(f"  {c['pattern']:<12} {c['direction']:>5} {c['tp_old']:>7.1f} {c['tp_new']:>7.1f} "
              f"{c['sl_old']:>7.1f} {c['sl_new']:>7.1f} {c['max_loss_old']:>7.1f}% {c['max_loss_new']:>7.1f}% "
              f"{c['sl_change']:>+6.1f}{flag}")

    print(f"\n  Summary: SL increased in {sl_increased}, decreased in {sl_decreased} patterns")
    return changes


def analyze_loss_distribution(trade_list):
    """Analyze distribution of losses (leveraged)."""
    losses = [t['pnl'] for t in trade_list if t['pnl'] < 0]
    if not losses:
        return {}

    losses_abs = [abs(l) for l in losses]
    result = {
        'total_losses': len(losses),
        'total_trades': len(trade_list),
        'loss_rate': round(len(losses) / len(trade_list) * 100, 1),
        'mean_loss': round(np.mean(losses_abs), 2),
        'median_loss': round(np.median(losses_abs), 2),
        'max_loss': round(max(losses_abs), 2),
        'min_loss': round(min(losses_abs), 2),
        'std_loss': round(np.std(losses_abs), 2),
        'p90_loss': round(np.percentile(losses_abs, 90), 2),
        'p95_loss': round(np.percentile(losses_abs, 95), 2),
        'p99_loss': round(np.percentile(losses_abs, 99), 2),
    }

    # Distribution by buckets
    buckets = {'0-2%': 0, '2-4%': 0, '4-6%': 0, '6-8%': 0, '8-10%': 0, '10%+': 0}
    for l in losses_abs:
        if l < 2:
            buckets['0-2%'] += 1
        elif l < 4:
            buckets['2-4%'] += 1
        elif l < 6:
            buckets['4-6%'] += 1
        elif l < 8:
            buckets['6-8%'] += 1
        elif l < 10:
            buckets['8-10%'] += 1
        else:
            buckets['10%+'] += 1
    result['buckets'] = buckets

    # Losses by pattern
    loss_by_pattern = defaultdict(list)
    for t in trade_list:
        if t['pnl'] < 0:
            loss_by_pattern[t['pattern']].append(abs(t['pnl']))
    top_loss_patterns = sorted(loss_by_pattern.items(), key=lambda x: -max(x[1]))[:10]
    result['top_loss_patterns'] = [(p, {'count': len(v), 'max': round(max(v), 2), 'avg': round(np.mean(v), 2)})
                                   for p, v in top_loss_patterns]

    print(f"\n  Loss Distribution ({len(losses)} losses out of {len(trade_list)} trades):")
    print(f"    Mean: {result['mean_loss']:.2f}%, Median: {result['median_loss']:.2f}%, Max: {result['max_loss']:.2f}%")
    print(f"    StdDev: {result['std_loss']:.2f}%, P90: {result['p90_loss']:.2f}%, P95: {result['p95_loss']:.2f}%, P99: {result['p99_loss']:.2f}%")
    print(f"\n    Buckets:")
    for bucket, count in buckets.items():
        bar = '#' * (count * 2)
        print(f"      {bucket:>6}: {count:>3} ({count/len(losses)*100:5.1f}%) {bar}")

    print(f"\n    Top 10 Patterns by Max Loss:")
    for pat, info in result['top_loss_patterns']:
        direction = PORTFOLIO.get(pat, ('?', 0, 0))[0]
        print(f"      {pat:<12} ({direction:>5}): {info['count']} losses, max={info['max']:.2f}%, avg={info['avg']:.2f}%")

    return result


def simulate_mdd_comparison():
    """Compare MDD between v1.26.2 and v1.26.4 TP/SL."""
    # Build v1.26.2 portfolio map
    v1262_portfolio = {}
    for pat in VALIDATED_LONG_PATTERNS:
        if pat in V1262_TPSL:
            tp, sl = V1262_TPSL[pat]
            v1262_portfolio[pat] = ("LONG", tp, sl)
    for pat in VALIDATED_SHORT_PATTERNS:
        if pat in V1262_TPSL:
            tp, sl = V1262_TPSL[pat]
            v1262_portfolio[pat] = ("SHORT", tp, sl)

    # Collect trades for both versions
    trades_v1262 = collect_trades_extended(v1262_portfolio)
    trades_v1264 = collect_trades_extended(PORTFOLIO)

    s_old = eval_trades_simple(trades_v1262)
    s_new = eval_trades_simple(trades_v1264)

    print(f"\n  Portfolio MDD Comparison:")
    print(f"    {'Metric':<15} {'v1.26.2':>12} {'v1.26.4':>12} {'Change':>12}")
    print("    " + "-" * 55)
    for key, label in [('trades', 'Trades'), ('wr', 'WR %'), ('pnl', 'PnL %'),
                        ('mdd', 'MDD %'), ('pnl_mdd', 'PnL/MDD'), ('pf', 'PF'),
                        ('avg_win', 'Avg Win %'), ('avg_loss', 'Avg Loss %')]:
        old_val = s_old[key]
        new_val = s_new[key]
        if isinstance(old_val, float):
            print(f"    {label:<15} {old_val:>12.2f} {new_val:>12.2f} {new_val - old_val:>+12.2f}")
        else:
            print(f"    {label:<15} {old_val:>12} {new_val:>12} {new_val - old_val:>+12}")

    # SL >= 3% stats
    high_sl_count_old = sum(1 for p in v1262_portfolio.values() if p[2] >= 3.0)
    high_sl_count_new = sum(1 for p in PORTFOLIO.values() if p[2] >= 3.0)
    print(f"\n    Patterns with SL >= 3.0%: v1.26.2={high_sl_count_old}, v1.26.4={high_sl_count_new}")
    print(f"    Max single loss (3x lev): v1.26.2={s_old['avg_loss']:.2f}%, v1.26.4={s_new['avg_loss']:.2f}%")

    return {
        'v1262': {k: round(v, 2) if isinstance(v, float) else v for k, v in s_old.items()},
        'v1264': {k: round(v, 2) if isinstance(v, float) else v for k, v in s_new.items()},
        'high_sl_patterns': {'v1262': high_sl_count_old, 'v1264': high_sl_count_new},
    }


tpsl_changes = compare_tpsl_versions()
loss_dist = analyze_loss_distribution(trades)
mdd_comparison = simulate_mdd_comparison()

phase2 = {
    'tpsl_changes': tpsl_changes,
    'loss_distribution': loss_dist,
    'mdd_comparison': mdd_comparison,
}


# ============================================================
# Phase 3: Portfolio Risk Management
# ============================================================
print(f"\n{'='*70}")
print("Phase 3: Portfolio Risk Management")
print(f"{'='*70}")


def analyze_consecutive_losses(trade_list):
    """Analyze consecutive loss sequences."""
    pnls = [t['pnl'] for t in trade_list]
    n = len(pnls)

    # Find consecutive loss streaks
    streaks = []
    current_streak = 0
    current_streak_pnl = 0
    current_streak_patterns = []
    for i, p in enumerate(pnls):
        if p < 0:
            current_streak += 1
            current_streak_pnl += p
            current_streak_patterns.append(trade_list[i]['pattern'])
        else:
            if current_streak > 0:
                streaks.append({
                    'length': current_streak,
                    'total_pnl': round(current_streak_pnl, 2),
                    'patterns': current_streak_patterns,
                })
            current_streak = 0
            current_streak_pnl = 0
            current_streak_patterns = []
    if current_streak > 0:
        streaks.append({
            'length': current_streak,
            'total_pnl': round(current_streak_pnl, 2),
            'patterns': current_streak_patterns,
        })

    max_streak = max((s['length'] for s in streaks), default=0)
    streak_counts = defaultdict(int)
    for s in streaks:
        streak_counts[s['length']] += 1

    # Theoretical probabilities
    loss_rate = sum(1 for p in pnls if p < 0) / n
    theoretical = {}
    for k in range(1, max_streak + 2):
        prob = loss_rate ** k
        expected_occurrences = (n - k + 1) * prob
        theoretical[k] = {
            'probability': round(prob, 6),
            'expected_occurrences': round(expected_occurrences, 2),
            'actual_occurrences': streak_counts.get(k, 0),
        }

    # Worst streaks
    worst_streaks = sorted(streaks, key=lambda s: s['total_pnl'])[:5]

    print(f"\n  Consecutive Loss Analysis:")
    print(f"    Total trades: {n}, Loss rate: {loss_rate:.1%}")
    print(f"    Max consecutive losses: {max_streak}")

    print(f"\n    {'Streak':>7} {'Actual':>7} {'Expected':>9} {'Prob':>10}")
    print("    " + "-" * 40)
    for k in range(1, max_streak + 2):
        t = theoretical[k]
        print(f"    {k:>7} {t['actual_occurrences']:>7} {t['expected_occurrences']:>9.2f} {t['probability']:>10.4%}")

    print(f"\n    Worst 5 Loss Streaks:")
    for i, s in enumerate(worst_streaks):
        print(f"      #{i+1}: {s['length']} losses, PnL={s['total_pnl']:+.2f}%, patterns={', '.join(s['patterns'])}")

    return {
        'total_trades': n,
        'loss_rate': round(loss_rate, 4),
        'max_consecutive_losses': max_streak,
        'streak_counts': dict(streak_counts),
        'theoretical': theoretical,
        'worst_streaks': worst_streaks[:5],
    }


def monte_carlo_portfolio_sim(trade_list, n_sims=10000):
    """Monte Carlo simulation: shuffle trade order, compute MDD distribution."""
    pnls = np.array([t['pnl'] for t in trade_list])
    n = len(pnls)
    mdds = np.zeros(n_sims)
    final_pnls = np.zeros(n_sims)

    for sim in range(n_sims):
        shuffled = np.random.permutation(pnls)
        cum = np.cumsum(shuffled)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        mdds[sim] = np.max(dd)
        final_pnls[sim] = cum[-1]

    result = {
        'n_sims': n_sims,
        'mdd_p50': round(float(np.percentile(mdds, 50)), 2),
        'mdd_p75': round(float(np.percentile(mdds, 75)), 2),
        'mdd_p90': round(float(np.percentile(mdds, 90)), 2),
        'mdd_p95': round(float(np.percentile(mdds, 95)), 2),
        'mdd_p99': round(float(np.percentile(mdds, 99)), 2),
        'mdd_max': round(float(np.max(mdds)), 2),
        'mdd_mean': round(float(np.mean(mdds)), 2),
        'actual_mdd': round(float(stats['mdd']), 2),
        'actual_mdd_percentile': round(float(np.mean(mdds <= stats['mdd']) * 100), 1),
        'pnl_p5': round(float(np.percentile(final_pnls, 5)), 2),
        'pnl_p50': round(float(np.percentile(final_pnls, 50)), 2),
    }

    print(f"\n  Monte Carlo MDD Distribution ({n_sims} simulations, {n} trades shuffled):")
    print(f"    P50: {result['mdd_p50']:.1f}%, P75: {result['mdd_p75']:.1f}%, "
          f"P90: {result['mdd_p90']:.1f}%, P95: {result['mdd_p95']:.1f}%, P99: {result['mdd_p99']:.1f}%")
    print(f"    Mean: {result['mdd_mean']:.1f}%, Max: {result['mdd_max']:.1f}%")
    print(f"    Actual MDD: {result['actual_mdd']:.1f}% (percentile: {result['actual_mdd_percentile']:.1f}%)")

    return result


def probability_of_ruin(trade_list, thresholds=None):
    """MC-based probability of hitting drawdown thresholds."""
    if thresholds is None:
        thresholds = [20, 30, 40, 50]
    pnls = np.array([t['pnl'] for t in trade_list])
    n = len(pnls)
    n_sims = 10000
    ruin_counts = {t: 0 for t in thresholds}

    for _ in range(n_sims):
        shuffled = np.random.permutation(pnls)
        cum = np.cumsum(shuffled)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = np.max(dd)
        for t in thresholds:
            if max_dd >= t:
                ruin_counts[t] += 1

    result = {}
    print(f"\n  Probability of Ruin (DD hitting threshold, {n_sims} MC sims):")
    for t in thresholds:
        prob = ruin_counts[t] / n_sims * 100
        result[f'{t}%'] = round(prob, 2)
        print(f"    DD >= {t:>3}%: {prob:>6.2f}%")

    return result


def kelly_criterion(trade_list):
    """Calculate Kelly criterion for optimal position sizing."""
    pnls = [t['pnl'] for t in trade_list]
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]

    if not wins or not losses:
        return {}

    wr = len(wins) / len(pnls)
    avg_win = np.mean(wins)
    avg_loss = np.mean(losses)
    rr = avg_win / avg_loss

    # Kelly formula: f* = (bp - q) / b  where b=R:R, p=WR, q=1-WR
    full_kelly = (rr * wr - (1 - wr)) / rr if rr > 0 else 0
    half_kelly = full_kelly / 2
    quarter_kelly = full_kelly / 4

    # Current position sizing (95% of balance with 3x leverage)
    # Effective risk per trade = position_size * SL * leverage
    avg_sl = np.mean([t['sl'] for t in trade_list if t['pnl'] < 0])
    current_risk_per_trade = 0.95 * avg_sl * LEVERAGE / 100  # fraction of balance

    result = {
        'win_rate': round(wr, 4),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'reward_risk': round(rr, 2),
        'full_kelly': round(full_kelly, 4),
        'half_kelly': round(half_kelly, 4),
        'quarter_kelly': round(quarter_kelly, 4),
        'current_risk_fraction': round(current_risk_per_trade, 4),
        'kelly_vs_current': round(full_kelly / current_risk_per_trade, 2) if current_risk_per_trade > 0 else 999,
    }

    print(f"\n  Kelly Criterion:")
    print(f"    WR: {wr:.1%}, Avg Win: {avg_win:.2f}%, Avg Loss: {avg_loss:.2f}%, R:R: {rr:.2f}")
    print(f"    Full Kelly:    {full_kelly:.2%} of bankroll per trade")
    print(f"    Half Kelly:    {half_kelly:.2%} (recommended)")
    print(f"    Quarter Kelly: {quarter_kelly:.2%} (conservative)")
    print(f"    Current risk:  {current_risk_per_trade:.2%} of bankroll per trade")
    print(f"    Kelly ratio:   {result['kelly_vs_current']:.2f}x (current/full_kelly)")

    return result


def recovery_analysis(trade_list, levels=None):
    """Analyze recovery trades needed after drawdown levels."""
    if levels is None:
        levels = [10, 15, 20, 25, 30]
    pnls = [t['pnl'] for t in trade_list]

    # Find all drawdown events and measure recovery
    cum = 0
    peak = 0
    drawdown_events = []
    in_dd = False
    dd_start_idx = 0

    for i, p in enumerate(pnls):
        cum += p
        if cum > peak:
            if in_dd:
                # Recovered
                drawdown_events.append({
                    'dd_depth': round(peak - min_in_dd, 2),
                    'trades_to_recover': i - dd_start_idx,
                    'start_idx': dd_start_idx,
                    'end_idx': i,
                })
                in_dd = False
            peak = cum
        else:
            if not in_dd:
                in_dd = True
                dd_start_idx = i
                min_in_dd = cum
            else:
                min_in_dd = min(min_in_dd, cum)

    # Theoretical recovery calculation
    avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
    avg_loss = np.mean([abs(p) for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
    wr = sum(1 for p in pnls if p > 0) / len(pnls)
    expectancy = wr * avg_win - (1 - wr) * avg_loss

    result = {}
    print(f"\n  Recovery Analysis (Expectancy: {expectancy:+.3f}%/trade):")
    print(f"    {'DD Level':>10} {'Trades Needed':>14} {'Days (~1.2t/d)':>14}")
    print("    " + "-" * 42)
    for level in levels:
        if expectancy > 0:
            trades_needed = level / expectancy
            days_needed = trades_needed / (len(pnls) / 270)
        else:
            trades_needed = float('inf')
            days_needed = float('inf')
        result[f'{level}%'] = {
            'trades_needed': round(trades_needed, 1) if trades_needed != float('inf') else 'inf',
            'days_needed': round(days_needed, 1) if days_needed != float('inf') else 'inf',
        }
        if trades_needed != float('inf'):
            print(f"    {level:>9}% {trades_needed:>14.1f} {days_needed:>14.1f}")
        else:
            print(f"    {level:>9}% {'inf':>14} {'inf':>14}")

    # Actual recovery events
    if drawdown_events:
        print(f"\n    Historical DD Recovery Events: {len(drawdown_events)}")
        big_events = [e for e in drawdown_events if e['dd_depth'] >= 5]
        if big_events:
            for e in sorted(big_events, key=lambda x: -x['dd_depth'])[:5]:
                print(f"      DD={e['dd_depth']:+.1f}% → recovered in {e['trades_to_recover']} trades")
    result['historical_events'] = len(drawdown_events)

    return result


def risk_adjusted_metrics(trade_list):
    """Calculate Sharpe, Sortino, Calmar ratios."""
    pnls = np.array([t['pnl'] for t in trade_list])
    n = len(pnls)

    # Per-trade metrics
    mean_return = np.mean(pnls)
    std_return = np.std(pnls)
    downside_returns = pnls[pnls < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0

    # Annualized (assuming ~1.2 trades/day, 365 days/year)
    trades_per_year = (n / 270) * 365
    annual_return = mean_return * trades_per_year
    annual_std = std_return * np.sqrt(trades_per_year)
    annual_downside_std = downside_std * np.sqrt(trades_per_year)

    # MDD
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    mdd = np.max(peak - cum)

    # Ratios (risk-free rate = 0 for simplicity)
    sharpe = annual_return / annual_std if annual_std > 0 else 0
    sortino = annual_return / annual_downside_std if annual_downside_std > 0 else 0
    calmar = annual_return / mdd if mdd > 0 else 0

    result = {
        'mean_return_per_trade': round(mean_return, 3),
        'std_per_trade': round(std_return, 3),
        'annual_return': round(annual_return, 1),
        'annual_std': round(annual_std, 1),
        'sharpe_ratio': round(sharpe, 2),
        'sortino_ratio': round(sortino, 2),
        'calmar_ratio': round(calmar, 2),
        'trades_per_year': round(trades_per_year, 1),
        'mdd': round(mdd, 1),
    }

    print(f"\n  Risk-Adjusted Metrics:")
    print(f"    Mean return/trade: {mean_return:+.3f}%, Std: {std_return:.3f}%")
    print(f"    Trades/year (est): {trades_per_year:.0f}")
    print(f"    Annual return (est): {annual_return:+.1f}%, Annual Std: {annual_std:.1f}%")
    print(f"    Sharpe Ratio:  {sharpe:.2f}")
    print(f"    Sortino Ratio: {sortino:.2f}")
    print(f"    Calmar Ratio:  {calmar:.2f}")

    return result


# Run Phase 3
consec_losses = analyze_consecutive_losses(trades)
mc_sim = monte_carlo_portfolio_sim(trades, n_sims=10000)
ruin_prob = probability_of_ruin(trades)
kelly = kelly_criterion(trades)
recovery = recovery_analysis(trades)
risk_metrics = risk_adjusted_metrics(trades)

phase3 = {
    'consecutive_losses': consec_losses,
    'monte_carlo_mdd': mc_sim,
    'probability_of_ruin': ruin_prob,
    'kelly_criterion': kelly,
    'recovery_analysis': recovery,
    'risk_adjusted_metrics': risk_metrics,
}


# ============================================================
# Phase 4: Comprehensive Recommendations
# ============================================================
print(f"\n{'='*70}")
print("Phase 4: Comprehensive Recommendations")
print(f"{'='*70}")


def generate_recommendations(p1, p2, p3):
    """Generate comprehensive risk management recommendations."""
    recs = {}

    # 1. Daily Loss Limit
    print("\n  [1] Daily Loss Limit Recommendation:")
    # Find optimal: best PnL/MDD ratio
    best_limit = None
    best_pm = 0
    for label, data in p1.items():
        if data.get('pnl_mdd', 0) > best_pm:
            best_pm = data['pnl_mdd']
            best_limit = label

    current_limit = p1.get('10%', {})
    recommended = p1.get(best_limit, {})
    print(f"    Current: 10% (PnL/MDD={current_limit.get('pnl_mdd', 0):.2f}x)")
    print(f"    Optimal: {best_limit} (PnL/MDD={recommended.get('pnl_mdd', 0):.2f}x)")

    # Also check: None vs 10% — how much does the limit cost?
    no_limit = p1.get('None', {})
    limit_cost = no_limit.get('pnl', 0) - current_limit.get('pnl', 0)
    print(f"    10% limit cost vs no limit: {limit_cost:+.1f}% PnL")
    print(f"    10% limit MDD improvement: {no_limit.get('mdd', 0) - current_limit.get('mdd', 0):+.1f}%")

    recs['daily_loss_limit'] = {
        'current': '10%',
        'recommended': best_limit,
        'rationale': f'Best PnL/MDD ratio ({best_pm:.2f}x)',
        'pnl_cost_of_current': round(limit_cost, 1),
    }

    # 2. Kelly-based Position Sizing
    print(f"\n  [2] Position Sizing Recommendation:")
    k = p3['kelly_criterion']
    if k:
        print(f"    Half Kelly suggests: {k.get('half_kelly', 0):.2%} risk per trade")
        print(f"    Current risk: {k.get('current_risk_fraction', 0):.2%} per trade")
        ratio = k.get('kelly_vs_current', 0)
        if ratio > 1.5:
            sizing_rec = "OVER-BETTING: Reduce position size or leverage"
        elif ratio < 0.5:
            sizing_rec = "UNDER-BETTING: Could increase position size"
        else:
            sizing_rec = "APPROPRIATE: Within Kelly range"
        print(f"    Assessment: {sizing_rec}")
        recs['position_sizing'] = {
            'half_kelly': round(k.get('half_kelly', 0), 4),
            'current_risk': round(k.get('current_risk_fraction', 0), 4),
            'assessment': sizing_rec,
        }

    # 3. Consecutive Loss Protection
    print(f"\n  [3] Consecutive Loss Protection:")
    max_consec = p3['consecutive_losses']['max_consecutive_losses']
    loss_rate = p3['consecutive_losses']['loss_rate']
    worst = p3['consecutive_losses']['worst_streaks']
    worst_pnl = worst[0]['total_pnl'] if worst else 0
    print(f"    Max observed: {max_consec} consecutive losses")
    print(f"    Worst streak impact: {worst_pnl:+.2f}%")
    # Recommended threshold: max_observed + 1 with pause
    threshold = max_consec + 1
    prob_threshold = loss_rate ** threshold
    print(f"    Recommended threshold: {threshold} consecutive losses → pause trading")
    print(f"    Probability of {threshold} consecutive: {prob_threshold:.4%}")
    recs['consecutive_loss_protection'] = {
        'max_observed': max_consec,
        'worst_streak_pnl': round(worst_pnl, 2),
        'recommended_threshold': threshold,
        'threshold_probability': round(prob_threshold, 6),
    }

    # 4. SL Cap
    print(f"\n  [4] SL Cap Recommendation:")
    high_sl = sum(1 for p in PORTFOLIO.values() if p[2] >= 3.0)
    total = len(PORTFOLIO)
    loss_dist = p2['loss_distribution']
    p95_loss = loss_dist.get('p95_loss', 0)
    max_loss = loss_dist.get('max_loss', 0)
    print(f"    Patterns with SL >= 3.0%: {high_sl}/{total} ({high_sl/total*100:.1f}%)")
    print(f"    Max single loss (leveraged): {max_loss:.2f}%")
    print(f"    P95 loss: {p95_loss:.2f}%")
    # Max acceptable single loss = 1/3 of max tolerable DD
    mc_p95_mdd = p3['monte_carlo_mdd'].get('mdd_p95', 0)
    max_single_target = mc_p95_mdd / 3
    max_sl_target = max_single_target / LEVERAGE
    print(f"    MC P95 MDD: {mc_p95_mdd:.1f}% → target max single loss: {max_single_target:.1f}% → SL cap: {max_sl_target:.1f}%")
    recs['sl_cap'] = {
        'high_sl_patterns': high_sl,
        'total_patterns': total,
        'max_loss_observed': round(max_loss, 2),
        'p95_loss': round(p95_loss, 2),
        'recommended_sl_cap': round(max_sl_target, 1),
        'rationale': f'1/3 of MC P95 MDD ({mc_p95_mdd:.1f}%)',
    }

    # 5. Max Risk Budget per Trade
    print(f"\n  [5] Max Risk Budget per Trade:")
    # Risk budget = maximum acceptable loss per trade as % of portfolio
    avg_loss = loss_dist.get('mean_loss', 0)
    # Based on recovery: if expectancy = E, to recover from L loss, need L/E trades
    expectancy = stats['expectancy']
    if expectancy > 0:
        recovery_trades_for_max = max_loss / expectancy
        recovery_trades_for_avg = avg_loss / expectancy
        print(f"    Avg loss: {avg_loss:.2f}% → {recovery_trades_for_avg:.1f} trades to recover")
        print(f"    Max loss: {max_loss:.2f}% → {recovery_trades_for_max:.1f} trades to recover")
        # Target: max loss recoverable in < 10 trades
        target_max_loss = expectancy * 10
        target_sl = target_max_loss / LEVERAGE
        print(f"    Target (recover in <10 trades): max loss {target_max_loss:.2f}% → SL {target_sl:.2f}%")
    recs['risk_budget'] = {
        'avg_loss': round(avg_loss, 2),
        'max_loss': round(max_loss, 2),
        'expectancy': round(expectancy, 3),
        'target_max_loss_10_trade_recovery': round(target_max_loss, 2) if expectancy > 0 else None,
    }

    # Summary
    print(f"\n  {'='*60}")
    print(f"  SUMMARY OF RECOMMENDATIONS")
    print(f"  {'='*60}")
    print(f"  1. Daily Loss Limit: {recs['daily_loss_limit']['recommended']}")
    if 'position_sizing' in recs:
        print(f"  2. Position Sizing: {recs['position_sizing']['assessment']}")
    print(f"  3. Consecutive Loss Threshold: {recs['consecutive_loss_protection']['recommended_threshold']} losses → pause")
    print(f"  4. SL Cap: {recs['sl_cap']['recommended_sl_cap']:.1f}%")
    if recs['risk_budget'].get('target_max_loss_10_trade_recovery') is not None:
        print(f"  5. Max Risk Budget: {recs['risk_budget']['target_max_loss_10_trade_recovery']:.2f}% per trade")

    return recs


recommendations = generate_recommendations(phase1, phase2, phase3)

# ============================================================
# Save Results
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'research': 'Risk Management Research v1.26.4',
    'version': BOT_VERSION,
    'portfolio_size': len(PORTFOLIO),
    'baseline_stats': {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()},
    'phase1_daily_loss_limit': phase1,
    'phase2_sl_impact': {
        'tpsl_changes_count': len(tpsl_changes),
        'loss_distribution': loss_dist,
        'mdd_comparison': mdd_comparison,
    },
    'phase3_portfolio_risk': phase3,
    'phase4_recommendations': recommendations,
}

out_path = Path(__file__).resolve().parent / '../../results/risk_management_research.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("\nDone.")
