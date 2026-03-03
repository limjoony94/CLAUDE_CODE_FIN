#!/usr/bin/env python3
"""
Mechanism Optimization Study — 최적 메커니즘 조합 탐색

Phase 1 결과 기반 후속 연구:
  - G5 AggRisk, M4 AdaptiveLev, M2 Regime는 ablation에서 "제거 시 개선"
  - M2+M4 redundancy -46.94 (극심)
  - P3 CascadeSL + G5 AggRisk synergy +19.40

연구 질문:
  Q1. M2 vs M4 — 어느 쪽을 유지? (redundancy 해소)
  Q2. 비용 큰 3개 (M2/M4/G5) 제거 조합 → WF에서도 개선?
  Q3. Top 6만 (양수 ablation) vs 전체 10개?
  Q4. 최적 부분집합은?

12 configurations × WF 3-fold 전체 검증
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.analysis.mechanism_portfolio_study import (
    load_data, load_patterns, build_signal_index,
    compute_atr_ratio, compute_ema_slope, find_neutral_window,
    portfolio_npos, calc_stats, run_wf,
    FULL_STACK, MINIMAL, MECHANISM_NAMES,
    FEE_PCT, LEVERAGE, SLIPPAGE_BUFFER, N_SLOTS,
)

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'mechanism_optimization_study.json')


# ============================================================
# Configuration builder
# ============================================================
def _build_config(overrides: dict) -> dict:
    """Start from FULL_STACK, apply overrides."""
    cfg = deepcopy(FULL_STACK)
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


# 12 configurations to test
EXPERIMENTS = {
    # Baseline
    'A0_full_stack': {},  # no overrides

    # Q1: M2 vs M4 redundancy resolution
    'Q1a_no_regime': {'regime_sizing': False, 'regime_mult': 1.0},
    'Q1b_no_adaptive_lev': {'adaptive_leverage': False},
    'Q1c_no_regime_no_lev': {'regime_sizing': False, 'regime_mult': 1.0, 'adaptive_leverage': False},

    # Q2: Costly trio removal
    'Q2a_no_agg_risk': {'agg_risk_cap': False},
    'Q2b_no_agg_no_regime': {'agg_risk_cap': False, 'regime_sizing': False, 'regime_mult': 1.0},
    'Q2c_no_costly_trio': {'agg_risk_cap': False, 'regime_sizing': False, 'regime_mult': 1.0, 'adaptive_leverage': False},

    # Q3: Top 6 only (positive ablation mechanisms only)
    'Q3a_top6_only': {
        'agg_risk_cap': False,
        'mdd_sizing': False,
        'regime_sizing': False, 'regime_mult': 1.0,
        'adaptive_leverage': False,
    },

    # Q3b: Top 6 + MDD safety net
    'Q3b_top6_plus_mdd': {
        'agg_risk_cap': False,
        'regime_sizing': False, 'regime_mult': 1.0,
        'adaptive_leverage': False,
    },

    # Q3c: Top 6 + AggRisk (for MDD defense)
    'Q3c_top6_plus_agg': {
        'mdd_sizing': False,
        'regime_sizing': False, 'regime_mult': 1.0,
        'adaptive_leverage': False,
    },

    # Q4: Best synergy combo (P3+G5 synergy +19.40, keep both but remove M2+M4 redundancy)
    'Q4a_synergy_focus': {
        'regime_sizing': False, 'regime_mult': 1.0,
        'adaptive_leverage': False,
        # Keep: P3, G5, M5, G1, G2, P1, P2, M1
    },

    # Q4b: Only P3+M5 (top 2 ablation) + core guards
    'Q4b_essential': {
        'agg_risk_cap': False,
        'mdd_sizing': False,
        'regime_sizing': False, 'regime_mult': 1.0,
        'adaptive_leverage': False,
        # Keep: P3, M5, G1, G2, P1, P2
    },
}


def describe_config(name, overrides):
    """List disabled mechanisms for display."""
    disabled = []
    for k, v in overrides.items():
        if k == 'regime_mult':
            continue  # accompanies regime_sizing
        mech = MECHANISM_NAMES.get(k, k)
        if isinstance(v, bool) and not v:
            disabled.append(mech)
        elif k == 'dir_cap' and v == 999:
            disabled.append(mech)
    if not disabled:
        return "all ON"
    return f"OFF: {', '.join(disabled)}"


def main():
    t0 = time.time()
    print("=" * 72)
    print("Mechanism Optimization Study")
    print("12 Configurations × WF 3-fold Validation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # Load data
    print("\n[Phase 0] Loading data...")
    df, types = load_data()
    patterns = load_patterns()
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n_bars = len(df)

    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)

    nw = find_neutral_window(closes)
    if nw is None:
        print("ERROR: No neutral window found")
        return
    neutral_start, neutral_end = nw
    print(f"  Neutral window: bar {neutral_start} → {neutral_end} ({(neutral_end - neutral_start) // 288}d)")

    type_codes = df['candle_type'].values
    sig_idx = build_signal_index(types[neutral_start:neutral_end + 1],
                                 neutral_end - neutral_start + 1)
    signal_tuples = []
    for pat_name, info in patterns.items():
        direction = info['direction']
        tp_pct = info['tp_pct']
        sl_pct = info['sl_pct']
        if pat_name in sig_idx:
            for offset in sig_idx[pat_name]:
                abs_bar = neutral_start + offset
                signal_tuples.append((abs_bar, pat_name, direction, tp_pct, sl_pct))
    signal_tuples.sort(key=lambda x: x[0])
    print(f"  Patterns: {len(patterns)} | Signals: {len(signal_tuples)}")

    # ====================================================================
    # Phase 1: Run all configs — IS (full neutral window)
    # ====================================================================
    print(f"\n[Phase 1] In-Sample evaluation (12 configs)...")
    print(f"  {'Config':<26} | {'PnL/MDD':>8} | {'PnL':>8} | {'MDD':>7} | {'WR':>6} | {'Trades':>6} | Disabled")
    print("  " + "-" * 95)

    is_results = {}
    for name, overrides in EXPERIMENTS.items():
        cfg = _build_config(overrides)
        trades, equity, max_dd, extra = portfolio_npos(
            signal_tuples, opens, highs, lows, closes, type_codes,
            n_bars, atr_ratio, ema_slope, neutral_start, neutral_end, cfg,
        )
        stats = calc_stats(trades, equity, max_dd)
        stats.update(extra)
        is_results[name] = stats
        desc = describe_config(name, overrides)
        print(f"  {name:<26} | {stats['pnl_mdd']:>8.2f} | {stats['pnl']:>+7.1f}% | {stats['mdd']:>6.2f}% | {stats['wr']:>5.1f} | {stats['trades']:>6} | {desc}")

    # ====================================================================
    # Phase 2: WF Validation (3-fold) for ALL configs
    # ====================================================================
    print(f"\n[Phase 2] Walk-Forward Validation (3-fold Expanding Window)...")
    print(f"  {'Config':<26} | {'Result':>6} | {'F1':>8} | {'F2':>8} | {'F3':>8} | {'Total':>8} | {'Min':>8}")
    print("  " + "-" * 85)

    wf_results = {}
    for name, overrides in EXPERIMENTS.items():
        cfg = _build_config(overrides)
        folds = run_wf(
            signal_tuples, opens, highs, lows, closes, type_codes,
            n_bars, atr_ratio, ema_slope, neutral_start, neutral_end, cfg,
        )
        fold_pnls = [f['pnl'] for f in folds]
        total = sum(fold_pnls)
        min_fold = min(fold_pnls) if fold_pnls else -999
        passed = all(p > 0 for p in fold_pnls)
        result = "PASS" if passed else "FAIL"
        wf_results[name] = {
            'passed': passed,
            'folds': fold_pnls,
            'total': round(total, 1),
            'min_fold': round(min_fold, 1),
        }
        mark = "✅" if passed else "❌"
        f_str = " | ".join(f"{p:>+7.1f}%" for p in fold_pnls)
        print(f"  {name:<26} | {mark} {result} | {f_str} | {total:>+7.1f}% | {min_fold:>+7.1f}%")

    # ====================================================================
    # Phase 3: Composite Ranking
    # ====================================================================
    print(f"\n[Phase 3] Composite Ranking (WF PASS only)...")
    print(f"  {'Rank':>4} | {'Config':<26} | {'IS PnL/MDD':>10} | {'IS PnL':>8} | {'IS MDD':>7} | {'OOS Total':>10} | {'OOS Min':>8}")
    print("  " + "-" * 95)

    # Filter WF PASS, sort by IS PnL/MDD
    candidates = []
    for name in EXPERIMENTS:
        if wf_results[name]['passed']:
            candidates.append({
                'name': name,
                'is_pnl_mdd': is_results[name]['pnl_mdd'],
                'is_pnl': is_results[name]['pnl'],
                'is_mdd': is_results[name]['mdd'],
                'oos_total': wf_results[name]['total'],
                'oos_min': wf_results[name]['min_fold'],
            })

    # Sort by IS PnL/MDD descending
    candidates.sort(key=lambda x: x['is_pnl_mdd'], reverse=True)

    for rank, c in enumerate(candidates, 1):
        marker = " ★" if rank == 1 else ""
        print(f"  {rank:>4} | {c['name']:<26} | {c['is_pnl_mdd']:>10.2f} | {c['is_pnl']:>+7.1f}% | {c['is_mdd']:>6.2f}% | {c['oos_total']:>+9.1f}% | {c['oos_min']:>+7.1f}%{marker}")

    failed = [name for name in EXPERIMENTS if not wf_results[name]['passed']]
    if failed:
        print(f"\n  WF FAIL: {', '.join(failed)}")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    if candidates:
        best = candidates[0]
        base = is_results['A0_full_stack']
        print(f"\n  Current (full_stack): PnL/MDD {base['pnl_mdd']:.2f} | PnL {base['pnl']:+.1f}% | MDD {base['mdd']:.2f}%")
        print(f"  Best WF-PASS:        PnL/MDD {best['is_pnl_mdd']:.2f} | PnL {best['is_pnl']:+.1f}% | MDD {best['is_mdd']:.2f}%")
        print(f"  Config: {best['name']}")
        print(f"  OOS: {best['oos_total']:+.1f}% total, min fold {best['oos_min']:+.1f}%")

        improvement = best['is_pnl_mdd'] - base['pnl_mdd']
        print(f"\n  Potential improvement: PnL/MDD {improvement:+.2f} ({improvement / base['pnl_mdd'] * 100 if base['pnl_mdd'] > 0 else 0:+.1f}%)")

    # Questions answered
    print(f"\n  Q1 (M2 vs M4): ", end="")
    q1a = is_results.get('Q1a_no_regime', {}).get('pnl_mdd', 0)
    q1b = is_results.get('Q1b_no_adaptive_lev', {}).get('pnl_mdd', 0)
    if q1a > q1b:
        print(f"M4 유지 권장 (no_regime {q1a:.2f} > no_lev {q1b:.2f})")
    else:
        print(f"M2 유지 권장 (no_lev {q1b:.2f} > no_regime {q1a:.2f})")

    print(f"  Q2 (Costly trio): ", end="")
    q2c = is_results.get('Q2c_no_costly_trio', {}).get('pnl_mdd', 0)
    q2c_wf = wf_results.get('Q2c_no_costly_trio', {}).get('passed', False)
    print(f"PnL/MDD {q2c:.2f}, WF {'PASS' if q2c_wf else 'FAIL'}")

    print(f"  Q3 (Top 6 only): ", end="")
    q3a = is_results.get('Q3a_top6_only', {}).get('pnl_mdd', 0)
    q3a_wf = wf_results.get('Q3a_top6_only', {}).get('passed', False)
    print(f"PnL/MDD {q3a:.2f}, WF {'PASS' if q3a_wf else 'FAIL'}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("=" * 72)

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'is_results': is_results,
        'wf_results': wf_results,
        'ranking': [c for c in candidates],
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
