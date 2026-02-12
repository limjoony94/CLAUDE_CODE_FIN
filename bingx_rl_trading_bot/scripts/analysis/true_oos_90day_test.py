#!/usr/bin/env python3
"""
True Out-of-Sample 90-Day Test
==============================
v1.27.3 Pattern-Rediscovery WF 연구 결과, 270일 데이터에서의 패턴 발굴은
pattern selection look-ahead bias에 의해 PnL의 90%가 편향됨.

이 스크립트는 270일 데이터 종료 이후(2026-01-31~)의 새 데이터를 사용하여
51개 패턴의 진짜 forward performance를 측정합니다.

== 사전 조건 ==
1. 새 5m BTC 데이터 파일: data/btc_5m_oos_90day.csv
   - 기간: 2026-01-31 ~ 2026-05-01 (또는 90일 이상)
   - 형식: timestamp,open,high,low,close,volume (reclassified 불필요)
   - 다운로드: BingX API에서 직접 다운로드

2. 데이터 생성 (필요 시):
   python scripts/utils/download_btc_5m.py --start 2026-01-31 --end 2026-05-01

== 실행 ==
python scripts/analysis/true_oos_90day_test.py

== 패턴 발굴 270일 데이터 vs OOS 90일 데이터 ==
패턴 발굴: 2025-05-05 ~ 2026-01-30 (270일) — data/btc_5m_270days_reclassified.csv
True OOS:  2026-01-31 ~ 2026-05-01 (90일)  — data/btc_5m_oos_90day.csv
→ 완전히 분리된 데이터. 패턴/TP/SL 모두 OOS 데이터에 노출 없음.

== Pass/Fail 기준 ==
| Metric          | PASS         | CAUTION       | FAIL         |
|-----------------|--------------|---------------|--------------|
| WR              | > 65%        | 60-65%        | < 60%        |
| PnL/trade       | > 0          | -0.2 ~ 0      | < -0.2       |
| MC p-value      | < 0.05       | 0.05-0.10     | > 0.10       |
| MDD             | < 30%        | 30-50%        | > 50%        |
| PF              | > 1.2        | 1.0-1.2       | < 1.0        |
| vs Genuine OOS  | within 20%   | 20-50% gap    | > 50% gap    |

기준 근거:
- WR 65%: 라이브 WR ~64.7%, Rediscovery WF genuine WR ~68.5%
- Break-even WR ~64.8% (R:R 0.646 기준)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ── project root ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── 51 patterns + TP/SL from constants.py ──────────────────
from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import (
    VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
    PATTERN_OPTIMAL_TPSL,
)

ALL_PATTERNS = {p: "LONG" for p in VALIDATED_LONG_PATTERNS}
ALL_PATTERNS.update({p: "SHORT" for p in VALIDATED_SHORT_PATTERNS})

# ── Candle classification (Ground Truth) ─────────────────────
def classify_candle(row: pd.Series, avg_body_20: float) -> str:
    o, h, l, c = row['open'], row['high'], row['low'], row['close']
    rng = h - l
    if rng == 0:
        return "D"
    body = c - o
    body_abs = abs(body)
    body_ratio = body_abs / rng
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    total_wick_ratio = (upper_wick + lower_wick) / rng if rng > 0 else 1.0
    norm_body = body_abs / avg_body_20 if avg_body_20 > 0 else 1.0

    # Priority order (Ground Truth v1.24.0)
    if total_wick_ratio < 0.15:
        return "MU" if body > 0 else "MD"
    if body_abs > 0 and lower_wick / body_abs > 2.0:
        return "H"
    if body_abs > 0 and upper_wick / body_abs > 2.0:
        return "IH"
    if body_ratio < 0.10:
        if rng > 0 and lower_wick / rng > 0.70:
            return "DF"
        if rng > 0 and upper_wick / rng > 0.70:
            return "GS"
        return "D"
    if norm_body < 0.5 and lower_wick >= 0.5 * body_abs and upper_wick >= 0.5 * body_abs:
        return "ST"
    if norm_body > 1.5:
        return "BU" if body > 0 else "BD"
    return "U" if body > 0 else "DN"


def prepare_data(filepath: str) -> pd.DataFrame:
    """Load and classify 5m data."""
    df = pd.read_csv(filepath)

    # Handle both reclassified and raw formats
    if 'type_code' in df.columns and 'pattern_3' in df.columns:
        print(f"  Using pre-classified data: {len(df)} bars")
        return df

    # Raw data — classify from scratch
    print(f"  Classifying {len(df)} bars from scratch...")
    df['body_abs'] = abs(df['close'] - df['open'])
    df['avg_body_20'] = df['body_abs'].rolling(20, min_periods=1).mean()

    codes = []
    for i, row in df.iterrows():
        avg_b = df['avg_body_20'].iloc[i] if i >= 20 else df['body_abs'].iloc[:max(i, 1)].mean()
        codes.append(classify_candle(row, avg_b))
    df['type_code'] = codes

    # Build 3-candle patterns
    df['pattern_3'] = (
        df['type_code'].shift(2) + '-' +
        df['type_code'].shift(1) + '-' +
        df['type_code']
    )
    df.loc[:1, 'pattern_3'] = None

    return df


# ── Backtest engine (Standard Research Protocol) ─────────────
FEE_PCT = 0.10  # 0.05% × 2
MAX_BARS = 500


def backtest_portfolio(df: pd.DataFrame, patterns: dict, tpsl: dict):
    """1-position-at-a-time portfolio backtest."""
    trades = []
    in_position = False
    entry_bar = None
    entry_price = None
    tp_pct = sl_pct = None
    direction = None
    pattern_name = None

    for i in range(2, len(df)):
        if in_position:
            bars_held = i - entry_bar
            if bars_held >= MAX_BARS:
                # Time-based exit at current close
                exit_price = df['close'].iloc[i]
                if direction == "LONG":
                    pnl = (exit_price - entry_price) / entry_price * 100 - FEE_PCT
                else:
                    pnl = (entry_price - exit_price) / entry_price * 100 - FEE_PCT
                trades.append({
                    'pattern': pattern_name, 'direction': direction,
                    'entry_bar': entry_bar, 'exit_bar': i,
                    'pnl': pnl, 'exit_type': 'TIMEOUT',
                    'bars_held': bars_held,
                })
                in_position = False
                continue

            # Intrabar TP/SL check (distance-based)
            h, l = df['high'].iloc[i], df['low'].iloc[i]
            if direction == "LONG":
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)
                tp_dist = abs(h - tp_price) if h >= tp_price else float('inf')
                sl_dist = abs(l - sl_price) if l <= sl_price else float('inf')
            else:
                tp_price = entry_price * (1 - tp_pct / 100)
                sl_price = entry_price * (1 + sl_pct / 100)
                tp_dist = abs(l - tp_price) if l <= tp_price else float('inf')
                sl_dist = abs(h - sl_price) if h >= sl_price else float('inf')

            tp_hit = tp_dist < float('inf')
            sl_hit = sl_dist < float('inf')

            if tp_hit and sl_hit:
                # Both hit — closer one wins
                if tp_dist <= sl_dist:
                    pnl = tp_pct - FEE_PCT
                    exit_type = 'TP'
                else:
                    pnl = -sl_pct - FEE_PCT
                    exit_type = 'SL'
            elif tp_hit:
                pnl = tp_pct - FEE_PCT
                exit_type = 'TP'
            elif sl_hit:
                pnl = -sl_pct - FEE_PCT
                exit_type = 'SL'
            else:
                continue  # still in position

            trades.append({
                'pattern': pattern_name, 'direction': direction,
                'entry_bar': entry_bar, 'exit_bar': i,
                'pnl': pnl, 'exit_type': exit_type,
                'bars_held': i - entry_bar,
            })
            in_position = False
        else:
            # Check for entry signal
            pat = df['pattern_3'].iloc[i]
            if pd.isna(pat) or pat not in patterns:
                continue
            # Enter next bar's open
            if i + 1 >= len(df):
                break
            entry_bar = i + 1
            entry_price = df['open'].iloc[i + 1]
            direction = patterns[pat]
            pattern_name = pat
            tpsl_vals = tpsl.get(pat, {'tp': 1.0, 'sl': 1.0})
            tp_pct = tpsl_vals['tp']
            sl_pct = tpsl_vals['sl']
            in_position = True

    return trades


def eval_trades(trades: list) -> dict:
    """Evaluate trade list."""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0, 'max_consec_loss': 0}

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # MDD
    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    mdd = float(np.max(dd)) if len(dd) > 0 else 0

    # Max consecutive losses
    max_cl = 0
    cl = 0
    for p in pnls:
        if p <= 0:
            cl += 1
            max_cl = max(max_cl, cl)
        else:
            cl = 0

    return {
        'pnl': round(sum(pnls), 2),
        'trades': len(trades),
        'wr': round(len(wins) / len(trades) * 100, 1) if trades else 0,
        'mdd': round(mdd, 2),
        'pf': round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 999,
        'avg_win': round(np.mean(wins), 2) if wins else 0,
        'avg_loss': round(np.mean(losses), 2) if losses else 0,
        'max_consec_loss': max_cl,
    }


def mc_test(trades: list, n_sims: int = 10000) -> dict:
    """Monte Carlo sign randomization test."""
    if len(trades) < 10:
        return {'p_value': 1.0, 'note': 'insufficient trades'}
    pnls = np.array([t['pnl'] for t in trades])
    actual_pnl = np.sum(pnls)
    abs_pnls = np.abs(pnls)

    rng = np.random.default_rng(42)
    count_ge = 0
    for _ in range(n_sims):
        signs = rng.choice([-1, 1], size=len(pnls))
        sim_pnl = np.sum(signs * abs_pnls)
        if sim_pnl >= actual_pnl:
            count_ge += 1

    # MDD distribution
    mdd_dist = []
    for _ in range(min(n_sims, 2000)):
        idx = rng.permutation(len(pnls))
        shuffled = pnls[idx]
        cum = np.cumsum(shuffled)
        rm = np.maximum.accumulate(cum)
        mdd_dist.append(float(np.max(rm - cum)))

    return {
        'p_value': round(count_ge / n_sims, 6),
        'mdd_p50': round(float(np.percentile(mdd_dist, 50)), 1),
        'mdd_p95': round(float(np.percentile(mdd_dist, 95)), 1),
        'mdd_p99': round(float(np.percentile(mdd_dist, 99)), 1),
    }


def per_pattern_analysis(trades: list) -> dict:
    """Per-pattern breakdown."""
    from collections import defaultdict
    by_pat = defaultdict(list)
    for t in trades:
        by_pat[t['pattern']].append(t)

    results = {}
    for pat, pat_trades in sorted(by_pat.items()):
        pnls = [t['pnl'] for t in pat_trades]
        wins = [p for p in pnls if p > 0]
        results[pat] = {
            'trades': len(pat_trades),
            'wr': round(len(wins) / len(pat_trades) * 100, 1),
            'pnl': round(sum(pnls), 2),
            'avg_pnl': round(np.mean(pnls), 2),
            'direction': pat_trades[0]['direction'],
        }
    return results


# ── Pass/Fail evaluation ─────────────────────────────────────
CRITERIA = {
    'wr': {'pass': 65.0, 'caution': 60.0},
    'pnl_per_trade': {'pass': 0.0, 'caution': -0.2},
    'mc_p': {'pass': 0.05, 'caution': 0.10},
    'mdd': {'pass': 30.0, 'caution': 50.0},
    'pf': {'pass': 1.2, 'caution': 1.0},
}

GENUINE_OOS_REF = {
    'pnl_per_trade': 80.5 / 89,  # +80.5% / 89 trades from D_Dynamic_Train
    'wr': 68.5,
}


def evaluate_criteria(stats: dict, mc: dict) -> dict:
    """Evaluate pass/fail criteria."""
    results = {}
    pnl_per_trade = stats['pnl'] / stats['trades'] if stats['trades'] > 0 else -999

    checks = {
        'wr': stats['wr'],
        'pnl_per_trade': pnl_per_trade,
        'mc_p': mc['p_value'],
        'mdd': stats['mdd'],
        'pf': stats['pf'],
    }

    for metric, value in checks.items():
        thresholds = CRITERIA[metric]
        if metric in ('mc_p', 'mdd'):
            # Lower is better
            if value <= thresholds['pass']:
                results[metric] = {'value': value, 'verdict': 'PASS'}
            elif value <= thresholds['caution']:
                results[metric] = {'value': value, 'verdict': 'CAUTION'}
            else:
                results[metric] = {'value': value, 'verdict': 'FAIL'}
        else:
            # Higher is better
            if value >= thresholds['pass']:
                results[metric] = {'value': value, 'verdict': 'PASS'}
            elif value >= thresholds['caution']:
                results[metric] = {'value': value, 'verdict': 'CAUTION'}
            else:
                results[metric] = {'value': value, 'verdict': 'FAIL'}

    # vs Genuine OOS reference
    if stats['trades'] > 0:
        ref_ppt = GENUINE_OOS_REF['pnl_per_trade']
        actual_ppt = pnl_per_trade
        gap = abs(actual_ppt - ref_ppt) / abs(ref_ppt) * 100 if ref_ppt != 0 else 999
        if gap <= 20:
            results['vs_genuine'] = {'value': gap, 'verdict': 'PASS'}
        elif gap <= 50:
            results['vs_genuine'] = {'value': gap, 'verdict': 'CAUTION'}
        else:
            results['vs_genuine'] = {'value': gap, 'verdict': 'FAIL'}

    pass_count = sum(1 for v in results.values() if v['verdict'] == 'PASS')
    fail_count = sum(1 for v in results.values() if v['verdict'] == 'FAIL')
    total = len(results)

    if fail_count >= 3:
        overall = 'FAIL — strategy edge not confirmed'
    elif fail_count >= 2:
        overall = 'MARGINAL — significant concerns'
    elif pass_count >= 4:
        overall = 'PASS — genuine forward edge confirmed'
    else:
        overall = 'CAUTION — mixed signals, extend collection period'

    return {
        'criteria': results,
        'summary': {
            'pass': pass_count, 'caution': total - pass_count - fail_count,
            'fail': fail_count, 'total': total,
            'overall': overall,
        }
    }


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("True Out-of-Sample 90-Day Test")
    print("  51 patterns from 270-day discovery → tested on NEW data only")
    print("=" * 80)

    # Check for OOS data
    oos_file = DATA_DIR / "btc_5m_oos_90day.csv"
    if not oos_file.exists():
        print(f"\n  ERROR: OOS data file not found: {oos_file}")
        print(f"  Expected: 5m BTC data from 2026-01-31 onwards")
        print(f"  To create: python scripts/utils/download_btc_5m.py --start 2026-01-31 --end 2026-05-01")
        print(f"\n  === Current Timeline ===")
        print(f"  Pattern discovery data: 2025-05-05 ~ 2026-01-30 (270 days)")
        print(f"  OOS test data needed:   2026-01-31 ~ 2026-05-01 (90 days)")
        print(f"  Today: {datetime.now().strftime('%Y-%m-%d')}")
        days_since = (datetime.now() - datetime(2026, 1, 30)).days
        days_remaining = max(0, 90 - days_since)
        print(f"  Days since data cutoff: {days_since}")
        print(f"  Days remaining:         {days_remaining}")
        if days_remaining > 0:
            target = datetime(2026, 1, 30) + pd.Timedelta(days=90)
            print(f"  Target collection date: {target.strftime('%Y-%m-%d')}")
        else:
            print(f"  Collection period complete — download data and re-run!")
        return

    # Load and prepare OOS data
    print(f"\nLoading OOS data: {oos_file}")
    df = prepare_data(str(oos_file))
    n_days = len(df) / 288  # 288 = 5min bars per day
    print(f"  Bars: {len(df)}, Days: {n_days:.1f}")

    if n_days < 30:
        print(f"  WARNING: Only {n_days:.1f} days — minimum 30 recommended, 90 ideal")

    # Build pattern dict with TP/SL
    tpsl = {}
    for pat in ALL_PATTERNS:
        if pat in PATTERN_OPTIMAL_TPSL:
            tpsl[pat] = {
                'tp': PATTERN_OPTIMAL_TPSL[pat]['tp'],
                'sl': PATTERN_OPTIMAL_TPSL[pat]['sl'],
            }
        else:
            tpsl[pat] = {'tp': 1.0, 'sl': 1.0}

    # Run backtest
    print(f"\n{'='*80}")
    print(f"PHASE 1: Portfolio Backtest on OOS Data")
    print(f"{'='*80}")
    trades = backtest_portfolio(df, ALL_PATTERNS, tpsl)
    stats = eval_trades(trades)

    print(f"  Trades: {stats['trades']}")
    print(f"  WR: {stats['wr']}%")
    print(f"  PnL: +{stats['pnl']}%")
    print(f"  MDD: {stats['mdd']}%")
    print(f"  PF: {stats['pf']}")
    print(f"  Avg Win: {stats['avg_win']}%")
    print(f"  Avg Loss: {stats['avg_loss']}%")
    print(f"  Max Consec Loss: {stats['max_consec_loss']}")
    ppt = stats['pnl'] / stats['trades'] if stats['trades'] > 0 else 0
    print(f"  PnL/trade: {ppt:.3f}%")

    # MC test
    print(f"\n{'='*80}")
    print(f"PHASE 2: Monte Carlo Significance Test")
    print(f"{'='*80}")
    mc = mc_test(trades)
    print(f"  MC p-value: {mc['p_value']}")
    print(f"  MDD P50/P95/P99: {mc.get('mdd_p50','N/A')}/{mc.get('mdd_p95','N/A')}/{mc.get('mdd_p99','N/A')}")

    # Per-pattern analysis
    print(f"\n{'='*80}")
    print(f"PHASE 3: Per-Pattern Breakdown")
    print(f"{'='*80}")
    pat_analysis = per_pattern_analysis(trades)
    print(f"\n  {'Pattern':<15} {'Dir':<6} {'Trades':>6} {'WR':>6} {'PnL':>8} {'Avg':>7}")
    print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*7}")
    for pat, data in sorted(pat_analysis.items(), key=lambda x: x[1]['pnl'], reverse=True):
        print(f"  {pat:<15} {data['direction']:<6} {data['trades']:>6} {data['wr']:>5.1f}% {data['pnl']:>+7.1f}% {data['avg_pnl']:>+6.2f}%")

    # Patterns with no trades
    active_pats = set(pat_analysis.keys())
    inactive_pats = set(ALL_PATTERNS.keys()) - active_pats
    if inactive_pats:
        print(f"\n  Patterns with 0 trades: {len(inactive_pats)}")
        for p in sorted(inactive_pats):
            print(f"    {p} ({ALL_PATTERNS[p]})")

    # Pass/Fail evaluation
    print(f"\n{'='*80}")
    print(f"PHASE 4: Pass/Fail Evaluation")
    print(f"{'='*80}")
    evaluation = evaluate_criteria(stats, mc)
    for metric, result in evaluation['criteria'].items():
        icon = {'PASS': 'PASS', 'CAUTION': 'WARN', 'FAIL': 'FAIL'}[result['verdict']]
        print(f"  [{icon:>4}] {metric:<16} = {result['value']:.4f}")

    summary = evaluation['summary']
    print(f"\n  PASS: {summary['pass']} | CAUTION: {summary['caution']} | FAIL: {summary['fail']}")
    print(f"\n  === VERDICT: {summary['overall']} ===")

    # Reference comparison
    print(f"\n{'='*80}")
    print(f"PHASE 5: Reference Comparison")
    print(f"{'='*80}")
    print(f"  {'Metric':<20} {'OOS 90d':>10} {'Genuine WF':>12} {'Backtest':>10} {'Live':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
    print(f"  {'WR':<20} {stats['wr']:>9.1f}% {'68.5':>11}% {'84.9':>9}% {'64.7':>9}%")
    print(f"  {'PnL/trade':<20} {ppt:>9.3f}% {'0.904':>11}% {'2.70':>9}% {'0.62':>9}%")
    print(f"  {'MDD':<20} {stats['mdd']:>9.1f}% {'31.7':>11}% {'16.2':>9}% {'N/A':>9}")
    print(f"  {'PF':<20} {stats['pf']:>9.2f} {'~1.5':>11} {'3.82':>9} {'N/A':>9}")

    # Save results
    results = {
        'meta': {
            'script': 'true_oos_90day_test.py',
            'version': 'v1.27.3',
            'timestamp': datetime.now().isoformat(),
            'oos_file': str(oos_file),
            'oos_days': round(n_days, 1),
            'patterns_tested': len(ALL_PATTERNS),
            'discovery_period': '2025-05-05 ~ 2026-01-30',
        },
        'portfolio': stats,
        'mc_test': mc,
        'per_pattern': pat_analysis,
        'inactive_patterns': sorted(list(inactive_pats)),
        'evaluation': evaluation,
        'reference': {
            'genuine_wf_oos': {'wr': 68.5, 'pnl_per_trade': 0.904, 'mdd': 31.7},
            'backtest_270d': {'wr': 84.9, 'pnl': 966.0, 'mdd': 16.2},
            'live_34trades': {'wr': 64.7, 'pnl': 20.94, 'trades': 34},
        },
    }

    out_path = RESULTS_DIR / "true_oos_90day_test.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Action recommendations based on verdict
    print(f"\n{'='*80}")
    print(f"RECOMMENDED ACTIONS")
    print(f"{'='*80}")
    if 'PASS' in summary['overall']:
        print("  1. 현재 51패턴 + TP/SL 유지")
        print("  2. EXPECTED_WIN_RATE를 OOS WR로 업데이트")
        print("  3. 데이터를 기존 270일에 90일 추가하여 360일 데이터셋으로 확장")
    elif 'MARGINAL' in summary['overall'] or 'CAUTION' in summary['overall']:
        print("  1. 추가 90일 데이터 수집 (총 180일 OOS)")
        print("  2. 레버리지 3x → 2x 축소 고려")
        print("  3. Robust-4 패턴만 사용 고려 (B option)")
    elif 'FAIL' in summary['overall']:
        print("  1. 봇 중지 또는 레버리지 1x로 축소")
        print("  2. Robust-4 패턴만으로 재시작 고려")
        print("  3. 전략 피봇 검토 (다른 시그널/타임프레임)")


if __name__ == '__main__':
    main()
