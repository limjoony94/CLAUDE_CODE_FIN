#!/usr/bin/env python3
"""Stability Gate — 사용자 지정 6 hard criteria (2026-04-20).

모든 전략은 아래 6 조건 동시 통과해야 함:
  1. daily_pnl_1x >= 0.2  (1x 기준 일평균 0.2% 이상)
  2. mean_net_per_trade > 0  (수수료 차감 후 양수 = 수수료 상회)
  3. trades_per_day >= 2.0  (통계 유의)
  4. bootstrap_pos_pct >= 55  (3-day window 55%+ positive)
  5. bootstrap_p5 >= -3.5  (tail risk 유계)
  6. pnl_ex_top5pct > 0  (상위 5% 제거 후도 양수 — 과적합 방지)

선택 warning:
  sharpe >= 0.35, p_loss_2pp <= 12%
"""
import sys
import copy
import random
from pathlib import Path
from statistics import mean, stdev, median

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

import scripts.analysis.intrabar_trail_impact as ibt
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.analysis.c1_intrabar_parity import SLIPPAGE as SLIP_MED
from scripts.analysis.regime_filter_lowvol_study import run_bt_with_regime
from scripts.analysis.regime_filter_trend_study import precompute_trend_pass


DATA_DAYS = ibt.n15 / 96  # 15m bars / 96 per day
TAKER_RT_FEE = 0.10  # 이미 net에 차감되어 있음 (BT engine fee)


def run(combo, use_filter=False, thr=1.0, lb=192):
    ibt.strat = copy.deepcopy(ibt.strat); ibt.strat.update(combo)
    ibt.trail_K = combo['trail_K']; ibt.max_hold = combo['max_hold_bars']
    ibt.sig = C1BreakoutSignal(ibt.strat)
    passes = precompute_trend_pass(lb, thr) if use_filter else [True] * ibt.n15
    return run_bt_with_regime(mode='5m', regime_passes=passes, slippage=SLIP_MED)


def bootstrap_3day(trades, n_samples=1000, seed=42):
    START_MIN = 220
    WINDOW = 288
    START_MAX = ibt.n15 - WINDOW - 1
    rng = random.Random(seed)
    pnls = []
    for _ in range(n_samples):
        s = rng.randint(START_MIN, START_MAX)
        e = s + WINDOW
        pnls.append(sum(t['net'] for t in trades if s <= t['entry_bar'] < e))
    ps = sorted(pnls)
    return {
        'mean': mean(pnls), 'median': median(pnls), 'std': stdev(pnls),
        'pos_pct': sum(1 for p in pnls if p > 0) / n_samples * 100,
        'sharpe': mean(pnls) / stdev(pnls) if stdev(pnls) > 0 else 0,
        'p5': ps[int(0.05 * n_samples)],
        'p_loss_2pp': sum(1 for p in pnls if p < -2) / n_samples * 100,
    }


def stability_gate(name, trades):
    if not trades:
        return {'name': name, 'passed': 0, 'flags': {}, 'note': 'no trades'}

    # Basic stats
    total_pnl = sum(t['net'] for t in trades)
    daily_pnl = total_pnl / DATA_DAYS
    mean_per_trade = total_pnl / len(trades)
    trades_per_day = len(trades) / DATA_DAYS

    # Top 5% exclusion (outlier concentration test)
    n_top = max(1, int(len(trades) * 0.05))
    sorted_desc = sorted(trades, key=lambda t: t['net'], reverse=True)
    ex_top = sorted_desc[n_top:]
    pnl_ex_top = sum(t['net'] for t in ex_top)

    # Bootstrap
    bs = bootstrap_3day(trades)

    flags = {
        'f1_daily_pnl>=0.2':     daily_pnl >= 0.2,
        'f2_net_per_trade>0':    mean_per_trade > 0,
        'f3_trades_per_day>=2':  trades_per_day >= 2.0,
        'f4_boot_pos>=55':       bs['pos_pct'] >= 55,
        'f5_boot_p5>=-3.5':      bs['p5'] >= -3.5,
        'f6_pnl_ex_top5pct>0':   pnl_ex_top > 0,
    }
    warnings = {
        'w1_sharpe>=0.35':       bs['sharpe'] >= 0.35,
        'w2_p_loss_2pp<=12':     bs['p_loss_2pp'] <= 12,
    }
    passed_core = sum(1 for v in flags.values() if v)
    return {
        'name': name,
        'metrics': {
            'total_pnl': round(total_pnl, 2),
            'daily_pnl': round(daily_pnl, 3),
            'mean_per_trade': round(mean_per_trade, 3),
            'trades_per_day': round(trades_per_day, 2),
            'pnl_ex_top5pct': round(pnl_ex_top, 2),
            'n_top_excluded': n_top,
            'boot_mean': round(bs['mean'], 3),
            'boot_pos_pct': round(bs['pos_pct'], 1),
            'boot_sharpe': round(bs['sharpe'], 3),
            'boot_p5': round(bs['p5'], 2),
            'boot_p_loss_2pp': round(bs['p_loss_2pp'], 1),
        },
        'flags': flags,
        'warnings': warnings,
        'passed_core': passed_core,
        'passed_total': passed_core + sum(1 for v in warnings.values() if v),
    }


COMBOS = [
    ('baseline_b0.40',               {'max_sl_atr':3.3,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.40}, False, 1.0, 192),
    ('baseline_b0.60',               {'max_sl_atr':3.3,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.60}, False, 1.0, 192),
    ('cand_C_b0.40',                 {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.40}, False, 1.0, 192),
    ('cand_C_b0.60',                 {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.60}, False, 1.0, 192),
    ('cand_C_b0.60+trend1.0_192',    {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.60}, True,  1.0, 192),
    ('cand_C_b0.70+trend1.0_192',    {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.70}, True,  1.0, 192),
    ('cand_C_b0.60+trend1.2_192',    {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.60}, True,  1.2, 192),
    ('val_top_b0.60',                {'max_sl_atr':4.5,'trail_K':2.2,'max_hold_bars':144,'body_min_ratio':0.60}, False, 1.0, 192),
    ('cand_C_b0.50',                 {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.50}, False, 1.0, 192),
    ('cand_C_b0.60_hold96',          {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':96, 'body_min_ratio':0.60}, False, 1.0, 192),
    ('cand_C_b0.60_tk22',            {'max_sl_atr':4.0,'trail_K':2.2,'max_hold_bars':192,'body_min_ratio':0.60}, False, 1.0, 192),
    ('cand_C_b0.60_tk28',            {'max_sl_atr':4.0,'trail_K':2.8,'max_hold_bars':192,'body_min_ratio':0.60}, False, 1.0, 192),
]


def main():
    print('=' * 110)
    print('  Stability Gate Validation — 6 Core + 2 Warning (사용자 지정 2026-04-20)')
    print(f'  Data: {ibt.n15} bars = {DATA_DAYS:.1f} days')
    print('=' * 110)
    print(f'{"Combo":<30} {"core":>5} {"dp":>6} {"nt":>5} {"tpd":>5} {"ex5%":>8} {"pos":>6} {"p5":>7} {"shr":>6} {"flags 1-6"}')

    results = []
    for name, cfg, uf, thr, lb in COMBOS:
        trades = run(cfg, uf, thr, lb)
        r = stability_gate(name, trades)
        results.append(r)
        m = r['metrics']
        flag_str = ''.join(['✓' if v else '✗' for v in r['flags'].values()])
        print(f'{name:<30} {r["passed_core"]}/6 '
              f'{m["daily_pnl"]:>+5.2f} {m["mean_per_trade"]:>+5.2f} {m["trades_per_day"]:>5.2f} '
              f'{m["pnl_ex_top5pct"]:>+7.2f} {m["boot_pos_pct"]:>5.1f}% {m["boot_p5"]:>+6.2f} '
              f'{m["boot_sharpe"]:>+5.2f}   {flag_str}')

    print()
    results.sort(key=lambda x: (x['passed_core'], x['metrics']['boot_sharpe']), reverse=True)
    print('=== Ranking ===')
    for r in results[:5]:
        print(f'\n{r["name"]} — core {r["passed_core"]}/6')
        for k, v in r['flags'].items():
            mark = '✓' if v else '✗'
            print(f'  {mark} {k}')
        for k, v in r['warnings'].items():
            mark = '✓' if v else '⚠'
            print(f'  {mark} {k} (warning)')
        m = r['metrics']
        print(f'  Metrics: daily={m["daily_pnl"]:+.3f}, tpd={m["trades_per_day"]:.2f}, '
              f'ex_top5%={m["pnl_ex_top5pct"]:+.2f}, pos={m["boot_pos_pct"]:.1f}%, sharpe={m["boot_sharpe"]:.3f}')


if __name__ == '__main__':
    main()
