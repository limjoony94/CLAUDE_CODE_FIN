"""
Refined Strategy Variants A/B/C/D — middle-ground experiments
==============================================================

Each variant attempts to preserve Refined's edge while mitigating specific risks.

  A: CD=2 restored (실전 친화, API rate limit 완화)
  B: Hard 2×ATR SL replaces emergency 3% (플래시 크래시 방어 강화)
  C: Channel soft filter (close within 0.5×ATR of channel boundary) + body
  D: Body 50% strict (fewer but stronger signals → fee 부담 경감)

Each evaluated on:
  - Full 332-day PnL / MDD / WR
  - WF 5-fold OOS (must pass all 5)
  - 3-way split (train/val/test all positive)
  - Fee stress 0.15%
"""
import sys, os, json, math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)
from scripts.analysis.c1_refined_validation import (
    BASE_CFG, entry_baseline, entry_refined,
    check_exit, run_bt, summarize, precompute,
    wf_5fold, three_way_split,
)
import scripts.analysis.c1_refined_validation as rv_mod


# ── Variant entry functions ──

def entry_refined_A(bo, bh, bl, bc, ch, cl, atr, swl, swh, cfg):
    """Same as REFINED (body + ATR SL) — CD handled in cfg."""
    return entry_refined(bo, bh, bl, bc, ch, cl, atr, swl, swh, cfg)


def entry_refined_B(bo, bh, bl, bc, ch, cl, atr, swl, swh, cfg):
    """Body filter + hard 2×ATR SL (tighter than emergency 3%)."""
    if math.isnan(atr) or atr <= 0:
        return None
    rng = bh - bl
    if rng <= 0:
        return None
    body = bc - bo
    if abs(body) / rng < cfg['body_min_ratio']:
        return None
    direction = 'LONG' if body > 0 else 'SHORT'
    e = bc
    sl_dist = 2.0 * atr  # HARD 2×ATR (was max_sl_atr=3.3)
    sl = e - sl_dist if direction == 'LONG' else e + sl_dist
    sp = abs(e - sl) / e * 100
    if sp < cfg['sl_min_pct'] or sp > cfg['sl_max_pct']:
        return None
    return {'direction': direction, 'sl_price': sl}


def entry_refined_C(bo, bh, bl, bc, ch, cl, atr, swl, swh, cfg):
    """Body filter + channel proximity (soft filter).

    Close must be within 0.5×ATR of channel boundary (not full breakout).
    Direction follows body + channel bias.
    """
    if math.isnan(ch) or math.isnan(cl) or math.isnan(atr) or atr <= 0 or ch <= cl:
        return None
    rng = bh - bl
    if rng <= 0:
        return None
    body = bc - bo
    if abs(body) / rng < cfg['body_min_ratio']:
        return None

    proximity = 0.5 * atr
    near_upper = bc >= (ch - proximity) and body > 0
    near_lower = bc <= (cl + proximity) and body < 0
    if not (near_upper or near_lower):
        return None

    direction = 'LONG' if near_upper else 'SHORT'
    e = bc
    sl_dist = cfg['max_sl_atr'] * atr
    sl = e - sl_dist if direction == 'LONG' else e + sl_dist
    sp = abs(e - sl) / e * 100
    if sp < cfg['sl_min_pct'] or sp > cfg['sl_max_pct']:
        return None
    return {'direction': direction, 'sl_price': sl}


def entry_refined_D(bo, bh, bl, bc, ch, cl, atr, swl, swh, cfg):
    """Body 50% (stricter threshold) + ATR cap SL."""
    if math.isnan(atr) or atr <= 0:
        return None
    rng = bh - bl
    if rng <= 0:
        return None
    body = bc - bo
    # STRICTER: 0.5 instead of cfg['body_min_ratio']
    if abs(body) / rng < 0.5:
        return None
    direction = 'LONG' if body > 0 else 'SHORT'
    e = bc
    sl_dist = cfg['max_sl_atr'] * atr
    sl = e - sl_dist if direction == 'LONG' else e + sl_dist
    sp = abs(e - sl) / e * 100
    if sp < cfg['sl_min_pct'] or sp > cfg['sl_max_pct']:
        return None
    return {'direction': direction, 'sl_price': sl}


def evaluate(name, entry_fn, cfg, df15):
    """Full evaluation: full-period + WF 5-fold + 3-way + fee stress."""
    pc = precompute(df15, cfg)
    # Full
    trades = run_bt(df15, cfg, entry_fn, 50, len(df15) - 1, **pc)
    full_s = summarize(trades)

    # WF 5-fold OOS
    wf = wf_5fold(df15, cfg, entry_fn, pc)
    wf_passed = sum(1 for f in wf if f['PnL'] > 0)
    wf_total = sum(f['PnL'] for f in wf)

    # 3-way
    tws = three_way_split(df15, cfg, entry_fn, pc)
    three_way_all_pos = all(s['PnL'] > 0 for s in tws.values())

    # Fee stress: 0.15% RT
    rv_mod.FEE_RT_PCT = 0.15
    try:
        trades_fee = run_bt(df15, cfg, entry_fn, 50, len(df15) - 1, **pc)
        fee15_s = summarize(trades_fee)
    finally:
        rv_mod.FEE_RT_PCT = 0.10

    return {
        'name': name,
        'full': full_s,
        'wf_passed': wf_passed,
        'wf_total_oos': round(wf_total, 2),
        'three_way': tws,
        'three_way_all_pos': three_way_all_pos,
        'fee_15pct_PnL': fee15_s['PnL'],
    }


def main():
    csv = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    df = pd.read_csv(csv, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df15 = df.set_index('timestamp').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open']).reset_index()
    print(f"Bars: {len(df15)} | {df15['timestamp'].iloc[0]} ~ {df15['timestamp'].iloc[-1]}")

    CD0 = {**BASE_CFG, 'min_bars_between': 0}
    CD2 = {**BASE_CFG, 'min_bars_between': 2}

    variants = [
        ('BASELINE',            entry_baseline,  CD2),
        ('REFINED_pure (CD=0)', entry_refined,   CD0),
        ('A: CD=2 restored',    entry_refined_A, CD2),
        ('B: Hard 2xATR SL',    entry_refined_B, CD0),
        ('C: Channel soft',     entry_refined_C, CD0),
        ('D: Body 50% strict',  entry_refined_D, CD0),
    ]

    print(f"\n{'='*110}")
    print(f"  VARIANT COMPARISON (332 days, additive 1x)")
    print(f"{'='*110}")
    print(f"{'Variant':25s} {'Trades':>7s} {'WR%':>6s} {'Full':>8s} "
          f"{'MDD':>6s} {'WF':>6s} {'WF_OOS':>8s} {'3way':>6s} "
          f"{'Test':>8s} {'fee15':>8s}")
    print('-'*110)

    results = []
    for name, fn, cfg in variants:
        r = evaluate(name, fn, cfg, df15)
        test_pnl = r['three_way']['test']['PnL']
        wf_pass = f"{r['wf_passed']}/5"
        tw_flag = 'Y' if r['three_way_all_pos'] else 'N'
        print(f"{name:25s} {r['full']['trades']:>7} {r['full']['WR']:>6} "
              f"{r['full']['PnL']:>+8.2f} {r['full']['MDD']:>6.2f} "
              f"{wf_pass:>6s} {r['wf_total_oos']:>+8.2f} "
              f"{tw_flag:>6s} {test_pnl:>+8.2f} {r['fee_15pct_PnL']:>+8.2f}")
        results.append(r)

    # Detailed 3-way breakdown
    print(f"\n{'='*110}")
    print("  3-way split detail")
    print(f"{'='*110}")
    for r in results:
        tws = r['three_way']
        print(f"{r['name']:25s}  Train={tws['train']['PnL']:>+8.2f}% "
              f"Val={tws['val']['PnL']:>+8.2f}%  Test={tws['test']['PnL']:>+8.2f}%  "
              f"[{'PASS' if r['three_way_all_pos'] else 'FAIL'}]")

    # Save
    out = {'date_run': datetime.now().isoformat(), 'variants': results}
    out_path = ROOT / 'results' / 'c1_refined_variants.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
