"""
Baton K_pre Sweep — E Phase 1 follow-up (2026-04-22)
=====================================================
Pre-activation baton의 trigger 거리를 K_pre로 스윕하여
27-trade 구간 + 333일 full에서 Sweet-spot 탐색.

기본 findings (baton_only_backtest_20260422):
- K=2.5 (current): 27 구간 -20.94% (LIVE 근접 ✓) / 333일 -190% (파산 ❌)
- 문제: pre-activation에서 too tight

가설: K_pre를 크게(looser) 하면 trigger가 멀어져 연속 소손실 회피.
목표: K_pre 스윕 [2.5, 3.5, 5.0, 7.0, 10.0, 15.0, 20.0]에서
      - Part 1: LIVE gap <= 10pp
      - Part 2: 333d additive PnL >= +150% (baseline 88%+)
"""

import sys, os, json, math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Import functions from baton_only_backtest
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from baton_only_backtest_20260422 import (
    fetch_candles_ccxt, load_candles_csv, run_bt, stats,
    CONFIG as BASE_CONFIG, baton_check_exit, baton_trail_trigger,
)

# Override baton_check_exit to use separate K_pre (pre-activation) vs K_post
def make_baton_check_exit_with_k_pre(k_pre_val, activation_pct):
    import baton_only_backtest_20260422 as bobt

    def custom_exit(direction, entry, best, high, low, sl_price, atr, bars_held, cfg):
        # SL / Emergency / Timeout priority (same as original)
        emergency_pct = cfg['emergency_sl_pct']
        max_hold = cfg['max_hold_bars']

        if direction == 'LONG':
            if low <= sl_price:
                return {'exit_price': sl_price, 'reason': 'SL'}
            worst_pnl = (low / entry - 1) * 100
            if worst_pnl <= -emergency_pct:
                return {'exit_price': entry * (1 - emergency_pct/100), 'reason': 'EMERGENCY'}
        else:
            if high >= sl_price:
                return {'exit_price': sl_price, 'reason': 'SL'}
            worst_pnl = (1 - high / entry) * 100
            if worst_pnl <= -emergency_pct:
                return {'exit_price': entry * (1 + emergency_pct/100), 'reason': 'EMERGENCY'}

        if bars_held >= max_hold:
            return {'exit_price': (high + low) / 2, 'reason': 'TIMEOUT'}

        # Baton trail with K_pre / K_post split
        if direction == 'LONG':
            best_pnl = (best / entry - 1) * 100
        else:
            best_pnl = (1 - best / entry) * 100

        pt = cfg.get('progressive_trail', {}) or {}
        if pt.get('enabled', False) and best_pnl >= pt.get('threshold_pct', 0.9):
            k_use = pt.get('trail_K_post', 0.5)
        elif best_pnl >= activation_pct:
            k_use = cfg['trail_K']  # post-activation (original K)
        else:
            k_use = k_pre_val  # pre-activation (swept)

        trigger = baton_trail_trigger(direction, entry, best, atr, k_use)
        if trigger is None:
            return None

        if direction == 'LONG':
            effective = max(trigger, sl_price)
            if low <= effective:
                return {'exit_price': effective, 'reason': 'BATON_TRAIL'}
        else:
            effective = min(trigger, sl_price)
            if high >= effective:
                return {'exit_price': effective, 'reason': 'BATON_TRAIL'}
        return None

    # Replace in module
    bobt.baton_check_exit = custom_exit


K_PRE_VALUES = [2.5, 3.5, 5.0, 7.0, 10.0, 15.0, 20.0]


def sweep_27(candles):
    print("\n=== Part 1: 27-trade 구간 (LIVE vs K_pre 스윕) ===")
    print(f"{'K_pre':>7} {'trades':>7} {'WR%':>6} {'PnL3x%':>9} {'endBal':>9} {'MDD%':>7} {'gap':>7}  reasons")
    print("-" * 95)
    results = {}
    for k in K_PRE_VALUES:
        make_baton_check_exit_with_k_pre(k, BASE_CONFIG['trail_activation_pct'])
        trades = run_bt(candles, datetime(2026, 4, 12), datetime(2026, 4, 22),
                        BASE_CONFIG, mode='baton')
        s = stats(trades)
        gap = round(s.get('sum_pnl_3x', 0) - (-16.09), 2)
        reasons = ' '.join(f"{k_}:{v}" for k_, v in sorted(s.get('reasons', {}).items()))
        print(f"{k:>7.1f} {s['trades']:>7} {s.get('wr_pct', 0):>5.1f} "
              f"{s.get('sum_pnl_3x', 0):>+8.2f}% ${s.get('end_bal', 0):>8.2f} "
              f"{s.get('mdd_pct', 0):>+6.2f}% {gap:>+6.2f}  {reasons}")
        results[f"{k}"] = {'stats': s, 'gap_vs_live': gap, 'trades': trades}
    return results


def sweep_333(candles):
    print("\n=== Part 2: 333일 Full BT (baseline preservation 검증) ===")
    print(f"{'K_pre':>7} {'trades':>6} {'WR%':>6} {'PnL1x add':>12} {'PnL3x cmp':>11} {'endBal3x':>12} {'MDD%':>7}  reasons")
    print("-" * 110)
    t_start = datetime.fromtimestamp(candles[0][0]/1000)
    t_end = datetime.fromtimestamp(candles[-1][0]/1000)
    results = {}
    for k in K_PRE_VALUES:
        make_baton_check_exit_with_k_pre(k, BASE_CONFIG['trail_activation_pct'])
        trades = run_bt(candles, t_start, t_end, BASE_CONFIG, mode='baton')
        s = stats(trades, start=100.0)
        add_1x = round(sum(t['pnl1x'] for t in trades), 2)
        reasons = ' '.join(f"{k_}:{v}" for k_, v in sorted(s.get('reasons', {}).items()))
        print(f"{k:>7.1f} {s['trades']:>6} {s.get('wr_pct', 0):>5.1f} "
              f"{add_1x:>+11.2f}% {s.get('sum_pnl_3x', 0):>+10.2f}% "
              f"${s.get('end_bal', 0):>11.2f} {s.get('mdd_pct', 0):>+6.2f}%  {reasons}")
        results[f"{k}"] = {'stats': s, 'add_1x': add_1x, 'reasons': s.get('reasons', {})}
    return results


def main():
    print("=" * 110)
    print("Baton K_pre Sweep — E Phase 1 follow-up (2026-04-22)")
    print("=" * 110)

    print("\nFetching 04-12 ~ 04-22 15m candles (27-trade 구간)...")
    c27 = fetch_candles_ccxt(datetime(2026, 4, 8), datetime(2026, 4, 22))
    print(f"Got {len(c27)} candles")
    r1 = sweep_27(c27)

    csv_path = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    if csv_path.exists():
        print(f"\nLoading full 333d data...")
        cf = load_candles_csv(str(csv_path))
        print(f"Got {len(cf)} 15m candles")
        r2 = sweep_333(cf)
    else:
        print(f"WARN: {csv_path} not found")
        r2 = {}

    out = {
        'date': datetime.now().isoformat(),
        'k_pre_values': K_PRE_VALUES,
        'part1': r1,
        'part2': r2,
    }
    path = ROOT / 'results' / f'baton_k_pre_sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
