"""
C1 Oracle Switching Analysis
============================
Measure the theoretical upper bound of a direction-switching filter.

At each breakout signal we simulate BOTH forward (original direction) and
reverse (flipped direction, mirrored SL) trades. The "Oracle" picks the
better outcome with perfect hindsight — the max across directions. This
provides the ceiling that any practical regime classifier could achieve
before it is built.

Standard Research Protocol:
  • Entry at next-bar open (opens[i+1])
  • Exit intrabar high/low via check_exit
  • Fee 0.10% RT (additive 1x)
  • Reuses BASE_CFG / check_exit / precompute from c1_refined_validation

Accuracy curve model (corrected from task spec):
  E[PnL | accuracy=p] = p · Oracle + (1-p) · AntiOracle
  At p=0.5 (random) → (Forward + Reverse)/2   (sanity)
  At p=1.0          → Oracle                  (perfect classifier)

Sanity guard: forward-only simulation MUST reproduce baseline (+170.5%, 1027
trades). If it diverges by more than 2 PnL-points we fail loudly.
"""
import sys, os, json, math, random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from scripts.analysis.c1_refined_validation import (
    BASE_CFG, entry_baseline, check_exit, precompute, summarize, FEE_RT_PCT
)


# ──────────────────────────────────────────────────────────────────────────
#  Single-trade simulation (matches run_bt inner loop semantics)
# ──────────────────────────────────────────────────────────────────────────
def _calc_pnl(direction, entry, exit_px):
    return (exit_px / entry - 1) * 100 if direction == 'LONG' else (1 - exit_px / entry) * 100


def _sim_single_trade(entry_bar_i, direction, sl_price, cfg, pc, end_i):
    """Simulate one trade from entry_bar_i+1 through exit.

    Returns (pnl_pct_after_fee, exit_bar_index, reason).
    Mirrors run_bt's in_pos loop behaviour exactly: same-bar check_exit on
    the entry bar, then walk forward updating best_px & bars_held.
    """
    ni = entry_bar_i + 1
    if ni > end_i:
        return 0.0, ni, 'NO_ENTRY'
    entry_px = pc['opens'][ni]
    best_px = pc['highs'][ni] if direction == 'LONG' else pc['lows'][ni]

    # Same-bar exit check at entry bar (bars_held = 0)
    atr_ni = pc['atr'][ni] if not math.isnan(pc['atr'][ni]) else pc['atr'][entry_bar_i]
    ex = check_exit(direction, entry_px, best_px,
                    pc['highs'][ni], pc['lows'][ni], pc['closes'][ni],
                    sl_price, atr_ni, 0, cfg)
    if ex:
        pnl = _calc_pnl(direction, entry_px, ex['exit_price']) - FEE_RT_PCT
        return pnl, ni, ex['reason']

    bars_held = 0
    for j in range(ni + 1, end_i + 1):
        bars_held += 1
        if direction == 'LONG':
            best_px = max(best_px, pc['highs'][j])
        else:
            best_px = min(best_px, pc['lows'][j])
        atr_j = pc['atr'][j] if not math.isnan(pc['atr'][j]) else pc['atr'][j - 1]
        ex = check_exit(direction, entry_px, best_px,
                        pc['highs'][j], pc['lows'][j], pc['closes'][j],
                        sl_price, atr_j, bars_held, cfg)
        if ex:
            pnl = _calc_pnl(direction, entry_px, ex['exit_price']) - FEE_RT_PCT
            return pnl, j, ex['reason']
    # No exit found — dataset end (TIMEOUT-like)
    last_close = pc['closes'][end_i]
    pnl = _calc_pnl(direction, entry_px, last_close) - FEE_RT_PCT
    return pnl, end_i, 'END'


# ──────────────────────────────────────────────────────────────────────────
#  Oracle scan
# ──────────────────────────────────────────────────────────────────────────
def _compute_reverse_sl(direction_rev, entry_px, atr_i, swl_i, swh_i, cfg):
    """Mirror the fractal/ATR SL logic for the flipped direction.

    Note: entry_baseline uses close[i] as 'e' for SL placement; here we use
    opens[i+1] as an approximation (consistent with the actual executed entry).
    Returns (sl_price, sl_pct) or (None, None) if SL bounds are violated.
    """
    if direction_rev == 'LONG':
        fs = swl_i if not math.isnan(swl_i) else entry_px - cfg['max_sl_atr'] * atr_i
        sl = max(fs, entry_px - cfg['max_sl_atr'] * atr_i)
    else:
        fs = swh_i if not math.isnan(swh_i) else entry_px + cfg['max_sl_atr'] * atr_i
        sl = min(fs, entry_px + cfg['max_sl_atr'] * atr_i)
    sp = abs(entry_px - sl) / entry_px * 100
    if sp < cfg['sl_min_pct'] or sp > cfg['sl_max_pct']:
        return None, None
    return sl, sp


def simulate_both_directions(df15, cfg, pc):
    """Iterate breakout signals; run forward + reverse at each."""
    comparisons = []
    warmup = 50
    end_i = len(df15) - 1
    cd_until = warmup

    for i in range(warmup, end_i):
        if i < cd_until:
            continue
        if math.isnan(pc['atr'][i]):
            continue
        sig = entry_baseline(
            pc['opens'][i], pc['highs'][i], pc['lows'][i], pc['closes'][i],
            pc['ch'][i], pc['cl'][i], pc['atr'][i],
            pc['swl'][i], pc['swh'][i], cfg,
        )
        if sig is None:
            continue

        # Forward simulation (actual baseline path)
        fwd_pnl, fwd_exit_bar, fwd_reason = _sim_single_trade(
            i, sig['direction'], sig['sl_price'], cfg, pc, end_i
        )

        # Reverse: flip direction, recompute mirrored SL using entry-bar atr + swings
        rev_dir = 'SHORT' if sig['direction'] == 'LONG' else 'LONG'
        ni = i + 1
        if ni > end_i:
            # Forward exists but reverse entry is past dataset
            comparisons.append({
                'entry_bar': i, 'timestamp': str(pc['timestamps'][ni - 1]),
                'fwd_dir': sig['direction'], 'fwd_pnl': fwd_pnl, 'fwd_reason': fwd_reason,
                'fwd_exit_bar': fwd_exit_bar,
                'rev_dir': rev_dir, 'rev_pnl': None, 'rev_reason': 'NO_ENTRY',
                'rev_valid': False,
            })
            cd_until = fwd_exit_bar + cfg['min_bars_between']
            continue

        entry_px_rev = pc['opens'][ni]
        rev_sl, rev_sp = _compute_reverse_sl(
            rev_dir, entry_px_rev, pc['atr'][i], pc['swl'][i], pc['swh'][i], cfg
        )
        if rev_sl is None:
            rev_pnl, rev_exit_bar, rev_reason, rev_valid = None, None, 'SL_OOB', False
        else:
            rev_pnl, rev_exit_bar, rev_reason = _sim_single_trade(
                i, rev_dir, rev_sl, cfg, pc, end_i
            )
            rev_valid = True

        comparisons.append({
            'entry_bar': i, 'timestamp': str(pc['timestamps'][ni]),
            'fwd_dir': sig['direction'], 'fwd_pnl': fwd_pnl, 'fwd_reason': fwd_reason,
            'fwd_exit_bar': fwd_exit_bar,
            'rev_dir': rev_dir, 'rev_pnl': rev_pnl, 'rev_reason': rev_reason,
            'rev_valid': rev_valid,
        })

        # Advance cooldown based on forward exit (consistent with run_bt timing)
        cd_until = fwd_exit_bar + cfg['min_bars_between']

    return comparisons


# ──────────────────────────────────────────────────────────────────────────
#  Aggregation + reporting
# ──────────────────────────────────────────────────────────────────────────
def _aggregate_trades(pnls):
    """Convert a list of pnl%/trade into trades-style summary."""
    fake = [{'pnl_pct': p} for p in pnls]
    n = len(fake)
    if n == 0:
        return {'trades': 0, 'WR': 0.0, 'PnL': 0.0, 'MDD': 0.0}
    wins = sum(1 for p in pnls if p > 0)
    pnl = sum(pnls)
    eq, peak, mdd = 0, 0, 0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return {'trades': n, 'WR': round(wins / n * 100, 2),
            'PnL': round(pnl, 2), 'MDD': round(mdd, 2)}


def main():
    csv = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    df = pd.read_csv(csv, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df15 = df.set_index('timestamp').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open']).reset_index()
    days = (df15['timestamp'].iloc[-1] - df15['timestamp'].iloc[0]).days
    print(f"Bars: {len(df15)} | {df15['timestamp'].iloc[0]} ~ {df15['timestamp'].iloc[-1]} ({days}d)")

    cfg = BASE_CFG
    pc = precompute(df15, cfg)
    # precompute returns python lists for opens/... and numpy arrays for indicators
    # ensure indicator access via index works consistently
    comps = simulate_both_directions(df15, cfg, pc)

    n_sig = len(comps)
    n_rev_valid = sum(1 for c in comps if c['rev_valid'])
    n_rev_invalid = n_sig - n_rev_valid

    # Full-signal forward PnLs (sanity check vs baseline +170.5%)
    fwd_all = [c['fwd_pnl'] for c in comps]
    fwd_sub = [c['fwd_pnl'] for c in comps if c['rev_valid']]
    rev_sub = [c['rev_pnl'] for c in comps if c['rev_valid']]

    # Oracle over ALL signals (fallback to forward when reverse invalid)
    oracle_all = []
    anti_all = []
    for c in comps:
        if c['rev_valid']:
            oracle_all.append(max(c['fwd_pnl'], c['rev_pnl']))
            anti_all.append(min(c['fwd_pnl'], c['rev_pnl']))
        else:
            oracle_all.append(c['fwd_pnl'])
            anti_all.append(c['fwd_pnl'])

    # Oracle over valid-reverse subset only (apples-to-apples)
    oracle_sub = [max(c['fwd_pnl'], c['rev_pnl']) for c in comps if c['rev_valid']]
    anti_sub = [min(c['fwd_pnl'], c['rev_pnl']) for c in comps if c['rev_valid']]

    s_fwd_all = _aggregate_trades(fwd_all)
    s_fwd_sub = _aggregate_trades(fwd_sub)
    s_rev_sub = _aggregate_trades(rev_sub)
    s_oracle_all = _aggregate_trades(oracle_all)
    s_oracle_sub = _aggregate_trades(oracle_sub)
    s_anti_all = _aggregate_trades(anti_all)
    s_anti_sub = _aggregate_trades(anti_sub)

    # Classification stats (valid-reverse subset)
    fwd_wins = sum(1 for c in comps if c['rev_valid'] and c['fwd_pnl'] > c['rev_pnl'])
    rev_wins = sum(1 for c in comps if c['rev_valid'] and c['rev_pnl'] > c['fwd_pnl'])
    ties = n_rev_valid - fwd_wins - rev_wins

    # ── Sanity check: forward-only must match baseline ──
    BASELINE_PNL = 170.5
    BASELINE_TRADES = 1027
    fwd_pnl_total = s_fwd_all['PnL']
    fwd_trades = s_fwd_all['trades']
    pnl_diff = abs(fwd_pnl_total - BASELINE_PNL)
    trades_diff = abs(fwd_trades - BASELINE_TRADES)
    sanity_ok = pnl_diff <= 2.0 and trades_diff <= 5

    print(f"\n=== Oracle Switching Analysis ({days} days) ===")
    print(f"Signals detected: {n_sig}")
    print(f"Valid reverse cases (SL bounds OK): {n_rev_valid} ({n_rev_valid/max(n_sig,1)*100:.1f}%)")
    print(f"Reverse SL-OOB (skipped): {n_rev_invalid} ({n_rev_invalid/max(n_sig,1)*100:.1f}%)")
    print()
    print(f"Sanity check vs baseline (+{BASELINE_PNL}%, {BASELINE_TRADES} trades):")
    print(f"  Forward-all PnL = {fwd_pnl_total:+.2f}% (diff {pnl_diff:+.2f}%)")
    print(f"  Forward-all trades = {fwd_trades} (diff {trades_diff:+d})")
    print(f"  Sanity: {'PASS' if sanity_ok else 'FAIL'}")
    if not sanity_ok:
        print("  !!! WARNING: forward loop diverges from run_bt — results suspect")

    print()
    print("--- Per-direction totals (ALL signals) ---")
    print(f"Forward only (all):   PnL={s_fwd_all['PnL']:+8.2f}%  WR={s_fwd_all['WR']:5.2f}%  "
          f"MDD={s_fwd_all['MDD']:5.2f}%  trades={s_fwd_all['trades']}")
    print(f"Oracle (all):         PnL={s_oracle_all['PnL']:+8.2f}%  WR={s_oracle_all['WR']:5.2f}%  "
          f"MDD={s_oracle_all['MDD']:5.2f}%  (invalid-rev -> fwd fallback)")
    print(f"Anti-oracle (all):    PnL={s_anti_all['PnL']:+8.2f}%  WR={s_anti_all['WR']:5.2f}%  "
          f"MDD={s_anti_all['MDD']:5.2f}%  (worst picker)")

    print()
    print("--- Per-direction totals (valid-reverse subset) ---")
    print(f"Forward only:   PnL={s_fwd_sub['PnL']:+8.2f}%  WR={s_fwd_sub['WR']:5.2f}%  "
          f"MDD={s_fwd_sub['MDD']:5.2f}%  trades={s_fwd_sub['trades']}")
    print(f"Reverse only:   PnL={s_rev_sub['PnL']:+8.2f}%  WR={s_rev_sub['WR']:5.2f}%  "
          f"MDD={s_rev_sub['MDD']:5.2f}%  trades={s_rev_sub['trades']}")
    print(f"Oracle (max):   PnL={s_oracle_sub['PnL']:+8.2f}%  WR={s_oracle_sub['WR']:5.2f}%  "
          f"MDD={s_oracle_sub['MDD']:5.2f}%  (ceiling — perfect classifier)")
    print(f"Anti-oracle:    PnL={s_anti_sub['PnL']:+8.2f}%  WR={s_anti_sub['WR']:5.2f}%  "
          f"MDD={s_anti_sub['MDD']:5.2f}%  (min picker — worst case)")

    print()
    print("--- Classification stats (valid-reverse subset) ---")
    tot_valid = max(n_rev_valid, 1)
    print(f"Forward wins:  {fwd_wins:4d} trades ({fwd_wins/tot_valid*100:5.2f}%)")
    print(f"Reverse wins:  {rev_wins:4d} trades ({rev_wins/tot_valid*100:5.2f}%)")
    print(f"Tie/zero diff: {ties:4d} trades ({ties/tot_valid*100:5.2f}%)")

    # ── Accuracy → PnL curve (analytic) ──
    # Over ALL signals: p · Oracle_all + (1-p) · AntiOracle_all
    # This aligns with "beat baseline +170.5%" comparisons.
    oracle_pnl_all = s_oracle_all['PnL']
    anti_pnl_all = s_anti_all['PnL']
    fwd_pnl_all_val = s_fwd_all['PnL']
    gap_all = oracle_pnl_all - anti_pnl_all

    print()
    print("--- Implementation gap (vs baseline forward) ---")
    print(f"Forward baseline : {fwd_pnl_all_val:+8.2f}%")
    print(f"Oracle ceiling   : {oracle_pnl_all:+8.2f}%  (gap {oracle_pnl_all - fwd_pnl_all_val:+.2f}%)")
    print(f"Anti-oracle floor: {anti_pnl_all:+8.2f}%  (gap {anti_pnl_all - fwd_pnl_all_val:+.2f}%)")
    print(f"Switching edge (Oracle − max(Fwd,Rev)) over subset: "
          f"{s_oracle_sub['PnL'] - max(s_fwd_sub['PnL'], s_rev_sub['PnL']):+.2f}%")

    print()
    print("--- Classification accuracy vs PnL curve ---")
    print("(analytic: E[PnL|p] = p · Oracle + (1-p) · AntiOracle, over ALL signals)")
    curve = {}
    for p_pct in [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        p = p_pct / 100.0
        exp_pnl = p * oracle_pnl_all + (1 - p) * anti_pnl_all
        flag = ''
        if p_pct == 50:
            flag = ' (random → (Fwd+Rev)/2)'
        if p_pct == 100:
            flag = ' (oracle)'
        print(f"  {p_pct:3d}% accuracy: PnL={exp_pnl:+8.2f}%{flag}")
        curve[p_pct] = round(exp_pnl, 2)

    # Breakeven accuracy to match / beat forward baseline
    # p* solving exp = target: p* = (target - anti) / (oracle - anti)
    def _acc_for_target(target):
        if gap_all <= 0:
            return None
        return (target - anti_pnl_all) / gap_all

    target_match = fwd_pnl_all_val
    target_plus_10pct = fwd_pnl_all_val * 1.10  # +10% over baseline
    target_plus_50abs = fwd_pnl_all_val + 50.0  # +50pp absolute
    acc_match = _acc_for_target(target_match)
    acc_10 = _acc_for_target(target_plus_10pct)
    acc_50 = _acc_for_target(target_plus_50abs)

    print()
    print("--- Accuracy thresholds ---")
    def _fmt_acc(a):
        if a is None:
            return 'N/A (gap<=0)'
        if a < 0: return f'<0% (always beat)'
        if a > 1: return f'>100% (impossible)'
        return f'{a*100:5.2f}%'
    print(f"Match Forward baseline ({target_match:+.1f}%): accuracy = {_fmt_acc(acc_match)}")
    print(f"Beat baseline by +10% ({target_plus_10pct:+.1f}%): accuracy = {_fmt_acc(acc_10)}")
    print(f"Beat baseline by +50pp ({target_plus_50abs:+.1f}%): accuracy = {_fmt_acc(acc_50)}")

    # ── Save JSON ──
    out = {
        'date_run': datetime.now().isoformat(),
        'script': 'c1_oracle_switching.py',
        'data_file': 'btc_5m_270days_reclassified.csv',
        'period_days': days,
        'bars_15m': len(df15),
        'cfg': cfg,
        'fee_rt_pct': FEE_RT_PCT,
        'signals_total': n_sig,
        'reverse_valid': n_rev_valid,
        'reverse_invalid_sl_oob': n_rev_invalid,
        'sanity_baseline_expected': {'PnL': BASELINE_PNL, 'trades': BASELINE_TRADES},
        'sanity_forward_observed': {'PnL': fwd_pnl_total, 'trades': fwd_trades,
                                    'PnL_diff': round(pnl_diff, 3),
                                    'trades_diff': trades_diff, 'pass': sanity_ok},
        'aggregates': {
            'forward_all': s_fwd_all,
            'forward_subset': s_fwd_sub,
            'reverse_subset': s_rev_sub,
            'oracle_all': s_oracle_all,
            'oracle_subset': s_oracle_sub,
            'antioracle_all': s_anti_all,
            'antioracle_subset': s_anti_sub,
        },
        'classification_stats_subset': {
            'forward_wins': fwd_wins, 'reverse_wins': rev_wins, 'ties': ties,
            'forward_win_pct': round(fwd_wins / tot_valid * 100, 2),
            'reverse_win_pct': round(rev_wins / tot_valid * 100, 2),
        },
        'accuracy_curve_pct_pnl': curve,
        'accuracy_thresholds': {
            'match_forward': (None if acc_match is None else round(acc_match * 100, 2)),
            'beat_plus_10pct_relative': (None if acc_10 is None else round(acc_10 * 100, 2)),
            'beat_plus_50pp_absolute': (None if acc_50 is None else round(acc_50 * 100, 2)),
        },
    }
    out_path = ROOT / 'results' / 'c1_oracle_switching.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
