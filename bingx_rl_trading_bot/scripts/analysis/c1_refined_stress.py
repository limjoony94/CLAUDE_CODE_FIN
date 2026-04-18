"""
C1 Refined Strategy — Stress Tests
==================================
Critical-evaluation angles:
  1. Fee sensitivity (0.10 / 0.13 / 0.15 / 0.20% RT) — Refined has 1.57x trades → 1.57x fee burden
  2. Intra-fold rolling MDD — hidden peak drawdowns inside each WF fold
  3. Direction Monte Carlo — random direction p-value (is direction alpha?)
  4. Consecutive loss streaks — tail risk
  5. Slippage absorption — extra 0.03/0.05% on each trade
"""
import sys, os, json, math, random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)
from scripts.analysis.c1_refined_validation import (
    entry_baseline, entry_refined, check_exit, run_bt, summarize, precompute,
    BASE_CFG,
)


def run_with_fee(df15, cfg, entry_fn, fee_rt_pct):
    """Rerun backtest with different fee rate."""
    # Monkey-patch: override FEE_RT_PCT in the module's scope
    import scripts.analysis.c1_refined_validation as mod
    orig_fee = mod.FEE_RT_PCT
    mod.FEE_RT_PCT = fee_rt_pct
    try:
        pc = precompute(df15, cfg)
        trades = run_bt(df15, cfg, entry_fn, 50, len(df15) - 1, **pc)
    finally:
        mod.FEE_RT_PCT = orig_fee
    return trades


def rolling_mdd(trades, window=50):
    """Compute rolling MDD over a sliding window of trades."""
    if len(trades) < window:
        return 0
    max_rolling = 0
    for start in range(len(trades) - window):
        segment = trades[start:start + window]
        eq = 0
        peak = 0
        mdd = 0
        for t in segment:
            eq += t['pnl_pct']
            peak = max(peak, eq)
            mdd = max(mdd, peak - eq)
        max_rolling = max(max_rolling, mdd)
    return max_rolling


def max_consec_losses(trades):
    best = 0
    cur = 0
    for t in trades:
        if t['pnl_pct'] <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def direction_mc(df15, cfg, entry_fn, n_sims=200, seed_base=0):
    """Monte Carlo direction randomization: for each trade, flip direction w.p. 0.5.

    Compares actual PnL to random-direction distribution.
    Returns p-value = P(random PnL ≥ actual PnL).
    """
    pc = precompute(df15, cfg)
    actual_trades = run_bt(df15, cfg, entry_fn, 50, len(df15) - 1, **pc)
    actual_pnl = sum(t['pnl_pct'] for t in actual_trades)

    # Sign-randomize each trade's PnL
    random.seed(seed_base)
    sim_pnls = []
    for sim in range(n_sims):
        sim_pnl = 0
        for t in actual_trades:
            if random.random() < 0.5:
                sim_pnl -= (t['pnl_pct'] + 0.20)  # flip + adjust fee (approximate)
            else:
                sim_pnl += t['pnl_pct']
        sim_pnls.append(sim_pnl)

    # p-value: fraction of sims whose PnL >= actual
    beat = sum(1 for p in sim_pnls if p >= actual_pnl)
    p_val = beat / n_sims
    sim_mean = sum(sim_pnls) / n_sims
    sim_max = max(sim_pnls)
    return {'actual_pnl': round(actual_pnl, 2), 'sim_mean': round(sim_mean, 2),
            'sim_max': round(sim_max, 2), 'p_value': round(p_val, 4),
            'n_sims': n_sims}


def main():
    csv = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    df = pd.read_csv(csv, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df15 = df.set_index('timestamp').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open']).reset_index()

    strategies = [
        ('BASELINE', entry_baseline, BASE_CFG),
        ('REFINED', entry_refined, {**BASE_CFG, 'min_bars_between': 0}),
    ]

    # ── 1. Fee sensitivity ──
    print("=" * 80)
    print("1. FEE SENSITIVITY (full-period additive 1x PnL)")
    print("=" * 80)
    fee_levels = [0.10, 0.13, 0.15, 0.20, 0.25]
    print(f"{'Strategy':12s}  " + " ".join(f"{f'fee={f}%':>12s}" for f in fee_levels))
    for name, fn, cfg in strategies:
        row = [f"{name:12s}"]
        for fee in fee_levels:
            trades = run_with_fee(df15, cfg, fn, fee)
            s = summarize(trades)
            row.append(f"{s['PnL']:>+12.2f}")
        print("  ".join(row))

    # ── 2. Rolling MDD (window=50 trades) ──
    print(f"\n{'='*80}\n2. ROLLING MDD (50-trade window)\n{'='*80}")
    for name, fn, cfg in strategies:
        pc = precompute(df15, cfg)
        trades = run_bt(df15, cfg, fn, 50, len(df15) - 1, **pc)
        s = summarize(trades)
        rmdd = rolling_mdd(trades, window=50)
        cl = max_consec_losses(trades)
        print(f"{name:12s}  full_MDD={s['MDD']:>5.2f}%  rolling50_MDD={rmdd:>5.2f}%  max_consec_losses={cl}")

    # ── 3. Direction MC ──
    print(f"\n{'='*80}\n3. DIRECTION MONTE CARLO (200 sims)\n{'='*80}")
    for name, fn, cfg in strategies:
        r = direction_mc(df15, cfg, fn, n_sims=200, seed_base=42)
        print(f"{name:12s}  actual_PnL={r['actual_pnl']:>+8.2f}  "
              f"MC_mean={r['sim_mean']:>+8.2f}  MC_max={r['sim_max']:>+8.2f}  "
              f"p_value={r['p_value']:.4f}")

    # ── 4. Slippage absorption ──
    print(f"\n{'='*80}\n4. SLIPPAGE (extra slippage pps per trade)\n{'='*80}")
    # fee 0.10% base + extra slippage: 0.00, 0.02, 0.04, 0.06 per trade (raw pct, not pp)
    for slip in [0.00, 0.02, 0.04, 0.06, 0.10]:
        print(f"\n  Slippage={slip}% extra:")
        for name, fn, cfg in strategies:
            trades = run_with_fee(df15, cfg, fn, 0.10 + slip)
            s = summarize(trades)
            print(f"    {name:12s}  PnL={s['PnL']:>+8.2f}%  MDD={s['MDD']:>5.2f}%  "
                  f"trades={s['trades']}")

    # ── 5. Save full detail ──
    result = {'date_run': datetime.now().isoformat()}
    out_path = ROOT / 'results' / 'c1_refined_stress.json'
    # Collect detailed numbers
    detail = {}
    for name, fn, cfg in strategies:
        d = {}
        for fee in fee_levels:
            trades = run_with_fee(df15, cfg, fn, fee)
            d[f'fee_{fee}'] = summarize(trades)
        # Rolling MDD
        pc = precompute(df15, cfg)
        trades = run_bt(df15, cfg, fn, 50, len(df15) - 1, **pc)
        d['rolling50_mdd'] = round(rolling_mdd(trades, 50), 2)
        d['max_consec_losses'] = max_consec_losses(trades)
        d['direction_mc'] = direction_mc(df15, cfg, fn, n_sims=200, seed_base=42)
        detail[name] = d
    result['strategies'] = detail
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
