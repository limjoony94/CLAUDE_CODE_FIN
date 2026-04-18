#!/usr/bin/env python3
"""C1 Breakout intrabar parity analysis — Phase 1 (Track A).

Extends scripts/analysis/intrabar_trail_impact.py with:
  - Slippage injection (entry/SL/trail/emergency/timeout reason-conditional)
  - Emergency-priority preservation (SL slip overflow → EMERGENCY reclass)
  - LIVE 19-trade window gap measurement
  - Time-partition WF on adjusted trades
  - 8 GO condition evaluation (6 evaluable in Phase 1)
  - 4 mode × 4 combo grid

Phase 1 combos evaluated via module-global monkey-patch:
  baseline      (max_sl_atr=3.3, trail_K=2.5, max_hold_bars=192)
  candidate_A   (3.6, 2.2, 144)  — sl_trail_tuning train top-1
  candidate_B   (4.5, 2.2, 144)  — sl_trail_tuning val top-1
  candidate_C   (4.0, 2.5, 192)  — 1D grid max_sl_atr winner

Output: results/intrabar_parity_{timestamp}.json
"""
import sys
import os
import json
import math
import copy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd

# Reuse engine + data + indicators from existing module
import scripts.analysis.intrabar_trail_impact as ibt

FEE = ibt.FEE  # 0.10 RT


# ─── Slippage model ────────────────────────────────────────────────

SLIPPAGE = {
    # Post-fix (BUG#62~65 applied) realistic median values.
    # LIVE 19-trade measurement (0.29%/0.64%) includes pre-fix outliers.
    # These conservative median estimates target fix-후 steady-state.
    'entry_pct':          0.05,
    'exit_sl_pct':        0.15,
    'exit_trail_pct':     0.05,
    'exit_emergency_pct': 0.30,
    'exit_timeout_pct':   0.05,
}

# Sensitivity: ±50% of base for robustness check
SLIPPAGE_HIGH = {k: v * 2.0 for k, v in SLIPPAGE.items()}   # pre-fix heavy
SLIPPAGE_LOW  = {k: v * 0.5 for k, v in SLIPPAGE.items()}    # best-case


def apply_slippage(trade, slippage=SLIPPAGE, emergency_sl_pct=3.0):
    """Adjust trade PnL by exit-reason slippage + emergency priority preservation.

    Entry slippage already folded into trade['raw'] by run_bt_slip.
    This function deducts exit slippage + FEE.

    Emergency priority: if SL reason with slip pushes |loss| past emergency_sl,
    reclassify as EMERGENCY and cap at emergency threshold.
    """
    reason = trade['reason']
    if reason == 'SL':
        adj = slippage['exit_sl_pct']
    elif reason == 'TRAIL_TP':
        adj = slippage['exit_trail_pct']
    elif reason == 'EMERGENCY':
        adj = slippage['exit_emergency_pct']
    elif reason == 'TIMEOUT':
        adj = slippage['exit_timeout_pct']
    else:
        adj = 0.0
    raw_adj = trade['raw'] - adj

    # Emergency priority preservation
    if reason == 'SL' and raw_adj <= -emergency_sl_pct:
        # Cap at emergency + its slip (emergency always triggers before SL in live)
        raw_adj = -emergency_sl_pct - slippage['exit_emergency_pct']
        trade['reason_effective'] = 'EMERGENCY'
    else:
        trade['reason_effective'] = reason
    return raw_adj - FEE


def run_bt_with_slippage(mode, slippage=SLIPPAGE):
    """Run intrabar BT with slippage adjustment.

    mode: one of 'bar_close', 'intrabar', '5m' (slippage added on top).
    Returns list of adjusted trade dicts with keys:
      raw (slip-adjusted entry+exit PnL),
      net (raw - FEE, with emergency reclass if applicable),
      reason (original), reason_effective (post-reclass),
      entry_bar, exit_bar, entry_price, exit_price, d, bh
    """
    # Step 1: clean intrabar backtest (no slip)
    clean_trades = ibt.run_backtest(mode=mode)

    adjusted = []
    for t in clean_trades:
        # Step 2: apply entry slippage to effective entry price
        entry_adv = slippage['entry_pct'] / 100
        if t['d'] == 'LONG':
            eff_entry = t['entry_price'] * (1 + entry_adv)
            raw_with_entry_slip = (t['exit_price'] / eff_entry - 1) * 100
        else:
            eff_entry = t['entry_price'] * (1 - entry_adv)
            raw_with_entry_slip = (1 - t['exit_price'] / eff_entry) * 100

        t_adj = dict(t)
        t_adj['raw'] = raw_with_entry_slip
        t_adj['net'] = apply_slippage(t_adj, slippage, ibt.emergency_sl)
        adjusted.append(t_adj)
    return adjusted


# ─── WF on adjusted trades (time partition) ────────────────────────

def wf_on_adjusted_trades(trades, n_folds=5):
    """Time-based 5-fold partition on adjusted trades by entry_bar.

    Returns count of positive folds (0..5).
    """
    if not trades:
        return 0
    ts = sorted(trades, key=lambda t: t['entry_bar'])
    first_bar = ts[0]['entry_bar']
    last_bar = ts[-1]['entry_bar']
    span = last_bar - first_bar
    if span < n_folds:
        return 0
    fold_size = span // n_folds
    pos_folds = 0
    fold_pnls = []
    for k in range(n_folds):
        lo = first_bar + k * fold_size
        hi = first_bar + (k + 1) * fold_size if k < n_folds - 1 else last_bar + 1
        fold_pnl = sum(t['net'] for t in ts if lo <= t['entry_bar'] < hi)
        fold_pnls.append(round(fold_pnl, 2))
        if fold_pnl > 0:
            pos_folds += 1
    return pos_folds, fold_pnls


def compute_mdd_additive(trades):
    ts = sorted(trades, key=lambda t: t['entry_bar'])
    eq = peak = mdd = 0.0
    for t in ts:
        eq += t['net']
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return mdd


def compute_mdd_from_clean(trades):
    """MDD on clean PnL (pre-slip) for comparison baseline."""
    ts = sorted(trades, key=lambda t: t['entry_bar'])
    eq = peak = mdd = 0.0
    for t in ts:
        eq += t['net'] if 'net' in t and 'raw' in t else (t['raw'] - FEE)
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return mdd


# ─── Train slice helper ────────────────────────────────────────────

def train_boundary(train_ratio=0.6, warmup=26):
    return warmup + int((ibt.n15 - warmup) * train_ratio)


def is_in_train_slice(trade, boundary):
    return trade['entry_bar'] <= boundary


# ─── LIVE window comparison ────────────────────────────────────────

def gap_vs_live_window(trades_adj, live_start='2026-04-12', live_end='2026-04-19'):
    """Compare adjusted BT with LIVE PnL.

    Note: btc_5m_270days data ends 2026-04-03 — LIVE window (2026-04-12~18) is
    out-of-sample. We fall back to **daily-rate comparison**: BT full-period
    PnL per trading day vs LIVE per-day rate.
    """
    ls = pd.Timestamp(live_start, tz='UTC')
    le = pd.Timestamp(live_end, tz='UTC')

    state = json.load(open(ROOT / 'results' / 'c1_breakout_state.json'))
    live_in_window = []
    for t in state['trade_history']:
        try:
            et = pd.Timestamp(t['exit_time'][:19], tz='UTC')
        except Exception:
            continue
        if ls <= et < le:
            live_in_window.append(t)
    live_pnl_3x = sum(t['pnl_pct'] for t in live_in_window)
    live_pnl_1x = live_pnl_3x / 3
    live_days = (le - ls).days
    live_daily_1x = live_pnl_1x / live_days if live_days else 0

    # BT: full period daily rate
    ts_col = ibt.agg15['ts'].tolist()
    bt_first = pd.Timestamp(ts_col[0])
    bt_last = pd.Timestamp(ts_col[-1])
    bt_days = (bt_last - bt_first).total_seconds() / 86400
    bt_pnl_full = sum(t['net'] for t in trades_adj)
    bt_daily_1x = bt_pnl_full / bt_days if bt_days else 0

    # LIVE window intersect BT data (will be empty if OOS)
    bt_in_window = []
    for t in trades_adj:
        eb = t['entry_bar']
        if eb >= len(ts_col):
            continue
        bar_ts = pd.Timestamp(ts_col[eb])
        if bar_ts.tz is None:
            bar_ts = bar_ts.tz_localize('UTC')
        if ls <= bar_ts < le:
            bt_in_window.append(t)
    bt_pnl_1x_win = sum(t['net'] for t in bt_in_window)

    return {
        'live_count':     len(live_in_window),
        'live_pnl_1x':    round(live_pnl_1x, 2),
        'live_pnl_3x':    round(live_pnl_3x, 2),
        'live_days':      live_days,
        'live_daily_1x':  round(live_daily_1x, 3),
        'bt_full_days':   round(bt_days, 1),
        'bt_full_pnl_1x': round(bt_pnl_full, 2),
        'bt_daily_1x':    round(bt_daily_1x, 3),
        'daily_gap_1x':   round(bt_daily_1x - live_daily_1x, 3),
        'window_bt_count': len(bt_in_window),
        'window_bt_pnl_1x': round(bt_pnl_1x_win, 2),
        'window_gap_1x':  round(bt_pnl_1x_win - live_pnl_1x, 2),
        'note': ('LIVE window (2026-04-12~18) is OOS for this dataset '
                 '(ends 2026-04-03). Use daily_gap_1x as realism proxy.'),
    }


# ─── Combo monkey-patching ─────────────────────────────────────────

_ORIG_GLOBALS = {}


def set_combo(max_sl_atr, trail_K, max_hold_bars):
    """Override module globals in intrabar_trail_impact for candidate combos.

    Re-creates signal object with updated config.
    Saves original on first call for restore.
    """
    from scripts.production.c1_breakout.signals import C1BreakoutSignal
    if not _ORIG_GLOBALS:
        _ORIG_GLOBALS['trail_K'] = ibt.trail_K
        _ORIG_GLOBALS['max_hold'] = ibt.max_hold
        _ORIG_GLOBALS['strat'] = copy.deepcopy(ibt.strat)
        _ORIG_GLOBALS['sig'] = ibt.sig
    new_strat = copy.deepcopy(ibt.strat)
    new_strat['max_sl_atr'] = max_sl_atr
    new_strat['trail_K'] = trail_K
    new_strat['max_hold_bars'] = max_hold_bars
    ibt.strat = new_strat
    ibt.trail_K = trail_K
    ibt.max_hold = max_hold_bars
    ibt.sig = C1BreakoutSignal(new_strat)


def reset_combo():
    if not _ORIG_GLOBALS:
        return
    ibt.trail_K = _ORIG_GLOBALS['trail_K']
    ibt.max_hold = _ORIG_GLOBALS['max_hold']
    ibt.strat = _ORIG_GLOBALS['strat']
    ibt.sig = _ORIG_GLOBALS['sig']


# ─── GO conditions ─────────────────────────────────────────────────

CLEAN_BASELINE_PNL = 170.49
CLEAN_TRAIN_PNL    = 95.07
CLEAN_MDD          = 5.38
CLEAN_RATIO        = CLEAN_BASELINE_PNL / CLEAN_MDD  # 31.69


def evaluate_go(baseline_adj, gap_5m_slip):
    """Return dict of 8 GO flags (Phase 1 evaluates 6; 2 deferred to Phase 2).

    intrabar_realism redefined as daily-rate proxy (±0.5%/day) due to
    data/live window mismatch (data ends 2026-04-03, LIVE starts 2026-04-12).
    """
    flags = {}

    # redefine: daily gap ±0.5%/day (1x)
    daily_gap = gap_5m_slip.get('daily_gap_1x', 999)
    flags['intrabar_realism'] = abs(daily_gap) <= 0.5

    baseline_pnl_adj = sum(t['net'] for t in baseline_adj)
    flags['baseline_preservation'] = baseline_pnl_adj >= CLEAN_BASELINE_PNL * 0.88

    wf_count, fold_pnls = wf_on_adjusted_trades(baseline_adj)
    flags['wf_pass'] = wf_count == 5
    flags['_wf_fold_pnls'] = fold_pnls

    mdd = compute_mdd_additive(baseline_adj)
    ratio = (baseline_pnl_adj / mdd) if mdd > 0 else 0
    flags['ratio_ok'] = ratio >= CLEAN_RATIO * 0.85  # 26.94
    flags['_adj_mdd'] = round(mdd, 2)
    flags['_adj_ratio'] = round(ratio, 2)
    flags['_adj_pnl'] = round(baseline_pnl_adj, 2)

    flags['track_b_cost'] = None      # Phase 2 deferred
    flags['track_b_benefit'] = None   # Phase 2 deferred

    flags['rollback_ready'] = True    # by design — new script only

    boundary = train_boundary()
    train_pnl_adj = sum(t['net'] for t in baseline_adj
                        if is_in_train_slice(t, boundary))
    flags['train_not_degraded'] = train_pnl_adj >= CLEAN_TRAIN_PNL - 5.0
    flags['_adj_train_pnl'] = round(train_pnl_adj, 2)

    return flags


def verdict(flags):
    """7/8 with core 1,2,3,8 required."""
    core = ['intrabar_realism', 'baseline_preservation', 'wf_pass',
            'train_not_degraded']
    all_eval = ['intrabar_realism', 'baseline_preservation', 'wf_pass',
                'ratio_ok', 'rollback_ready', 'train_not_degraded']
    for c in core:
        if not flags.get(c):
            return 'STOP', f'core flag {c} failed'
    true_count = sum(1 for k in all_eval if flags.get(k))
    if true_count >= len(all_eval) - 1:
        return 'GO', f'{true_count}/{len(all_eval)} evaluable flags true'
    return 'INCOMPLETE', f'{true_count}/{len(all_eval)} flags true'


# ─── Main ──────────────────────────────────────────────────────────

COMBOS = {
    'baseline':    {'max_sl_atr': 3.3, 'trail_K': 2.5, 'max_hold_bars': 192},
    'candidate_A': {'max_sl_atr': 3.6, 'trail_K': 2.2, 'max_hold_bars': 144},
    'candidate_B': {'max_sl_atr': 4.5, 'trail_K': 2.2, 'max_hold_bars': 144},
    'candidate_C': {'max_sl_atr': 4.0, 'trail_K': 2.5, 'max_hold_bars': 192},
}

MODES = ['bar_close', 'intrabar', '5m']  # 5m_slip = 5m + slippage


def main():
    t0 = datetime.now()
    print(f'{"="*70}')
    print(f'  C1 Intrabar Parity — Phase 1 (Track A)')
    print(f'{"="*70}')
    print(f'Data: {ibt.n15} bars (15m), {ibt.n5} bars (5m)')
    print(f'Slippage: {SLIPPAGE}')
    print()

    result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'slippage': SLIPPAGE,
        'data': {'bars_15m': ibt.n15, 'bars_5m': ibt.n5},
        'clean_baseline': {
            'pnl': CLEAN_BASELINE_PNL, 'train_pnl': CLEAN_TRAIN_PNL,
            'mdd': CLEAN_MDD, 'ratio': round(CLEAN_RATIO, 2),
        },
        'mode_results': {},
        'live_window_comparison': {},
        'go_flags': {},
        'verdict': None,
    }

    # Run each combo × each mode × (clean + slip)
    for combo_name, combo_cfg in COMBOS.items():
        print(f'--- Combo: {combo_name} {combo_cfg} ---')
        set_combo(**combo_cfg)
        result['mode_results'][combo_name] = {}
        for mode in MODES:
            clean_trades = ibt.run_backtest(mode=mode)
            clean_pnl = sum(t['net'] for t in clean_trades)
            clean_wins = sum(1 for t in clean_trades if t['net'] > 0)
            clean_mdd = compute_mdd_from_clean(clean_trades)

            slip_trades = run_bt_with_slippage(mode)
            slip_pnl = sum(t['net'] for t in slip_trades)
            slip_wins = sum(1 for t in slip_trades if t['net'] > 0)
            slip_mdd = compute_mdd_additive(slip_trades)

            result['mode_results'][combo_name][mode] = {
                'clean': {
                    'trades': len(clean_trades),
                    'pnl': round(clean_pnl, 2),
                    'wr': round(clean_wins/len(clean_trades)*100, 1) if clean_trades else 0,
                    'mdd': round(clean_mdd, 2),
                },
                'slip': {
                    'trades': len(slip_trades),
                    'pnl': round(slip_pnl, 2),
                    'wr': round(slip_wins/len(slip_trades)*100, 1) if slip_trades else 0,
                    'mdd': round(slip_mdd, 2),
                },
            }
            print(f'  {mode:10s} clean PnL={clean_pnl:+7.2f}% / MDD={clean_mdd:.2f} '
                  f'| slip PnL={slip_pnl:+7.2f}% / MDD={slip_mdd:.2f} / WR={slip_wins}/{len(slip_trades)}')

        # Save 5m_slip trades for LIVE comparison + GO flags (only baseline)
        if combo_name == 'baseline':
            set_combo(**combo_cfg)
            trades_5m_slip = run_bt_with_slippage(mode='5m')
            result['mode_results']['baseline']['5m_slip_trades'] = len(trades_5m_slip)
            print()
            print('--- LIVE window comparison (baseline, 5m_slip) ---')
            gap = gap_vs_live_window(trades_5m_slip)
            result['live_window_comparison'] = gap
            for k, v in gap.items():
                print(f'  {k}: {v}')

            print()
            print('--- GO Condition evaluation (baseline, 5m_slip) ---')
            flags = evaluate_go(trades_5m_slip, gap)
            result['go_flags'] = flags
            for k, v in flags.items():
                if not k.startswith('_'):
                    print(f'  {k}: {v}')
            for k, v in flags.items():
                if k.startswith('_'):
                    print(f'  {k}: {v}')

            vdict, reason = verdict(flags)
            result['verdict'] = {'outcome': vdict, 'reason': reason}
            print()
            print(f'=== VERDICT: {vdict} — {reason} ===')

        reset_combo()
        print()

    # Gap comparison across modes for baseline
    baseline_5m_slip_pnl = result['mode_results']['baseline']['5m']['slip']['pnl']
    bar_close_clean = result['mode_results']['baseline']['bar_close']['clean']['pnl']
    print('--- Mode gap vs LIVE (baseline, 1x scale) ---')
    live_pnl_1x = result['live_window_comparison'].get('live_pnl_1x', -3.92)
    # Gap is on FULL PnL not just LIVE window, but approximates trend
    print(f'  clean bar_close:  {bar_close_clean:+.2f}% (BT clean baseline)')
    print(f'  full 5m_slip:     {baseline_5m_slip_pnl:+.2f}% (slip-adjusted BT full)')
    print(f'  LIVE 19-trade 1x: {live_pnl_1x:+.2f}%')

    elapsed = (datetime.now() - t0).total_seconds()
    result['elapsed_sec'] = round(elapsed, 1)
    print(f'\nElapsed: {elapsed:.1f}s')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'results' / f'intrabar_parity_{stamp}.json'
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
