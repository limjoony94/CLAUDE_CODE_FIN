"""
Entry Sequencing Drift — Unmatched-LIVE Floor Investigation (2026-04-24)
=========================================================================

Motivation
----------
post_fix_legacy_gap_20260424.md discovered that of 16 LIVE trades in the
post-BUG-fix period, 5 were "unmatched" — LIVE entered bars where BT had no
corresponding signal — accounting for -7.97pp / 42% of the total -18.86pp gap.
F v2 (cycle_exit) can close matched-trade gaps but not this unmatched floor.

Hypothesis: unmatched trades arise from entry-sequencing drift. When a LIVE
exit fires at a different bar than BT (due to intrabar TRAILING tick vs BT
bar-close evaluation), the min_bars_between cooldown ends at a different bar
for LIVE vs BT, shifting the next eligible entry bar. Over multiple trades,
sequences diverge.

Research Questions
------------------
H1: Does increasing `min_bars_between` reduce drift / the unmatched floor?
H2: How sensitive is the entry sequence to small exit-timing perturbations?
H3: Define and measure a drift_fraction metric per config.
H4: Is there any LIVE-vs-BT entry timing divergence in the bot itself? (quick)

Methodology
-----------
- Data: data/btc_5m_270days_reclassified.csv resampled to 15m (same as ablation)
- Two BT runs per config:
  * Baseline: production config (identical to E8 in c1_ablation_study.json)
  * Perturbed: trail_K = 2.45 (realistic small economic perturbation; mimics
    LIVE intrabar exit drift; chosen over synthetic "delay by 1 bar" per
    advisor feedback)
- Metric: drift_fraction — for each entry in baseline run, does perturbed run
  have an entry within ±2 bars, same direction? Count unmatched baseline
  entries + unmatched perturbed entries (sym-diff). Express as fraction of
  union.
- Tolerance ±2 bars (30min): chosen config-independent (matches
  post_fix_legacy_gap_20260424.py matcher tolerance). Not tied to N, so
  comparable across configs.
- Sweep: min_bars_between ∈ {2, 3, 4, 6, 8, 12}
- 3-way split for overfit-guard: train/val/test by time thirds
- Output per config: trades, WR, PnL_1x, daily_PnL_1x, MDD, RR,
  drift_fraction, 3-way PnL

Expected result (a priori, null-predicting)
-------------------------------------------
The worst matched-class-C trade in the legacy window had a 55-bar BT-LIVE exit
divergence (57 vs 2 bars). Any min_bars_between ≤ 12 is orders of magnitude
smaller. The cooldown-buffer mechanism only helps when the perturbation
< N bars. Prediction: drift_fraction flat across N. If confirmed, the 42%
floor is structural and min_bars_between cannot meaningfully reduce it.

Null-result reporting is a clean finding, not a failure.

Guardrails
----------
- Read-only on production code; imports only.
- Results in results/entry_sequencing_drift_20260424.json
- No production config modification.
"""
from __future__ import annotations

import sys, os, json, math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings,
)
from scripts.production.c1_breakout.signals import C1BreakoutSignal

# ---------------------------------------------------------------------------
# Config

BASE_CFG = dict(
    channel_period=15, body_min_ratio=0.4, atr_period=14,
    trail_K=2.5, max_sl_atr=3.3, emergency_sl_pct=3.0,
    max_hold_bars=192, sl_min_pct=0.15, sl_max_pct=3.0,
    min_bars_between=2, trail_activation_pct=0.05,
    fractal_lookback=10,
    # Progressive trail enabled — matches production at time of writing
    progressive_trail={'enabled': True, 'threshold_pct': 0.9, 'trail_K_post': 0.5},
)

FEE_RT_PCT = 0.10
LEVERAGE = 3

# Sweep parameter
COOLDOWN_VALUES = [2, 3, 4, 6, 8, 12]

# Perturbation: small realistic trail_K change (mimics intrabar tick diff)
PERTURBATION = {'trail_K': 2.45}

# Drift matching tolerance (bars) — config-independent
DRIFT_TOL_BARS = 2


# ---------------------------------------------------------------------------
def load_15m():
    csv = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    df = pd.read_csv(csv, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df = df.set_index('timestamp').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open']).reset_index()
    return df


# ---------------------------------------------------------------------------
def run_bt(df15, cfg, start_offset=0, exit_delay_every=None) -> list[dict]:
    """Production-identical BT. Entry at next-bar open, intrabar exit. Returns
    trades with entry_bar (integer bar index), direction, pnl, reason, bars_held.

    Optional perturbations for H2/H3:
    - start_offset: begin loop at bar (50 + start_offset). Simulates "LIVE started
      1 bar later than BT" without any strategy change. Purest sequencing test.
    - exit_delay_every: if set (int N), every Nth exit is held 1 extra bar
      (intrabar-tick-style: exit fires in LIVE on later bar than BT would).
    """
    opens = df15['open'].tolist()
    highs = df15['high'].tolist()
    lows = df15['low'].tolist()
    closes = df15['close'].tolist()
    n = len(closes)

    atr = compute_atr(highs, lows, closes, cfg['atr_period'])
    ch_h, ch_l = compute_channel(highs, lows, cfg['channel_period'])
    sw_l, sw_h = compute_fractal_swings(highs, lows, cfg['fractal_lookback'])

    sig = C1BreakoutSignal(cfg)

    trades = []
    in_pos = False
    cd_until = 0
    pdir = pep = pspr = pbp = None
    pbh = 0
    pentry_bar = 0
    exit_count = 0
    pending_delay = False  # True = exit triggered last bar, close this bar instead

    for i in range(50 + start_offset, n):
        if in_pos:
            pbh += 1
            if pdir == 'LONG':
                pbp = max(pbp, highs[i])
            else:
                pbp = min(pbp, lows[i])
            er = sig.check_exit(
                direction=pdir, entry_price=pep, best_price=pbp,
                current_high=highs[i], current_low=lows[i], current_close=closes[i],
                sl_price=pspr,
                atr_val=atr[i] if not math.isnan(atr[i]) else atr[i-1],
                bars_held=pbh)
            if er:
                # Exit-delay perturbation: every Nth exit, defer actual close
                # one bar (simulates LIVE intrabar-tick firing "late" relative
                # to BT bar-close evaluation). Skip if timeout/emergency (which
                # are unambiguous).
                if exit_delay_every is not None and er['reason'] == 'TRAIL_TP':
                    exit_count += 1
                    if exit_count % exit_delay_every == 0 and i + 1 < n:
                        # Hold one more bar: re-evaluate at i+1 using close price
                        continue  # skip exit this bar, falls through to i+1 iteration
                xp = er['exit_price']
                pnl = (xp/pep - 1)*100 if pdir == 'LONG' else (1 - xp/pep)*100
                pnl_net = pnl - FEE_RT_PCT
                trades.append({
                    'entry_bar': pentry_bar,
                    'direction': pdir,
                    'entry_price': round(pep, 2),
                    'exit_price': round(xp, 2),
                    'pnl_pct': round(pnl_net, 4),
                    'reason': er['reason'],
                    'bars_held': pbh,
                })
                in_pos = False
                cd_until = i + cfg['min_bars_between']
                pdir = None

        if not in_pos and i >= cd_until and i < n - 1:
            if math.isnan(atr[i]) or math.isnan(ch_h[i]):
                continue
            es = sig.check_entry(
                bar_open=opens[i], bar_high=highs[i], bar_low=lows[i], bar_close=closes[i],
                channel_high=ch_h[i], channel_low=ch_l[i], atr_val=atr[i],
                last_swing_low=sw_l[i], last_swing_high=sw_h[i])
            if es is None:
                continue
            ni = i + 1
            if ni >= n:
                continue
            pdir = es['direction']
            pep = opens[ni]
            pspr = es['sl_price']
            pbh = 0
            in_pos = True
            pentry_bar = ni
            pbp = highs[ni] if pdir == 'LONG' else lows[ni]
            # Same-bar exit check
            er = sig.check_exit(
                direction=pdir, entry_price=pep, best_price=pbp,
                current_high=highs[ni], current_low=lows[ni], current_close=closes[ni],
                sl_price=pspr,
                atr_val=atr[ni] if not math.isnan(atr[ni]) else atr[i],
                bars_held=0)
            if er:
                xp = er['exit_price']
                pnl = (xp/pep - 1)*100 if pdir == 'LONG' else (1 - xp/pep)*100
                pnl_net = pnl - FEE_RT_PCT
                trades.append({
                    'entry_bar': pentry_bar,
                    'direction': pdir,
                    'entry_price': round(pep, 2),
                    'exit_price': round(xp, 2),
                    'pnl_pct': round(pnl_net, 4),
                    'reason': er['reason'],
                    'bars_held': 0,
                })
                in_pos = False
                cd_until = ni + cfg['min_bars_between']
                pdir = None

    return trades


# ---------------------------------------------------------------------------
def summarize(trades, n_bars, bar_minutes=15):
    if not trades:
        return {'trades': 0, 'WR': 0.0, 'PnL_1x': 0.0, 'daily_PnL_1x': 0.0,
                'MDD_1x': 0.0, 'RR': 0.0}
    n = len(trades)
    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    pnl = sum(t['pnl_pct'] for t in trades)
    avg_win = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0) / max(wins, 1)
    avg_loss = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0) / max(n - wins, 1)
    rr = abs(avg_win / avg_loss) if avg_loss else 0
    eq = peak = mdd = 0
    for t in trades:
        eq += t['pnl_pct']
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    days = n_bars * bar_minutes / (60 * 24)
    return {
        'trades': n,
        'WR': round(wins / n * 100, 1),
        'PnL_1x': round(pnl, 2),
        'daily_PnL_1x': round(pnl / days, 4),
        'MDD_1x': round(mdd, 2),
        'RR': round(rr, 2),
        'days': round(days, 1),
    }


# ---------------------------------------------------------------------------
def drift_fraction(trades_a, trades_b, tol_bars=DRIFT_TOL_BARS):
    """Compute drift fraction between two trade sequences.

    For each entry (bar, direction) in A, find a match in B within ±tol_bars
    of same direction. Unmatched entries count toward drift. Symmetric:
    unmatched B entries also count.

    Returns:
        drift_fraction: |sym_diff| / |union| (0 = identical, 1 = fully divergent)
        matched_count
        unmatched_a: A entries with no B match
        unmatched_b: B entries with no A match
    """
    used_b = set()
    unmatched_a = 0
    matched = 0

    # Sort A by bar for deterministic greedy matching
    sorted_a = sorted(enumerate(trades_a), key=lambda x: x[1]['entry_bar'])
    sorted_b = sorted(enumerate(trades_b), key=lambda x: x[1]['entry_bar'])

    for _, ta in sorted_a:
        best_j = -1
        best_gap = tol_bars + 1
        for j, tb in sorted_b:
            if j in used_b: continue
            if tb['direction'] != ta['direction']: continue
            gap = abs(tb['entry_bar'] - ta['entry_bar'])
            if gap <= tol_bars and gap < best_gap:
                best_gap = gap
                best_j = j
        if best_j >= 0:
            used_b.add(best_j)
            matched += 1
        else:
            unmatched_a += 1

    unmatched_b = sum(1 for j, _ in sorted_b if j not in used_b)
    union = matched + unmatched_a + unmatched_b
    drift = (unmatched_a + unmatched_b) / union if union else 0.0

    return {
        'drift_fraction': round(drift, 4),
        'matched': matched,
        'unmatched_a': unmatched_a,  # baseline-only
        'unmatched_b': unmatched_b,  # perturbed-only
        'union': union,
        'tol_bars': tol_bars,
    }


# ---------------------------------------------------------------------------
def three_way_split(trades, n_bars):
    """Split trades by entry_bar into train/val/test thirds; report PnL."""
    third = n_bars // 3
    train_end = third
    val_end = 2 * third
    buckets = {'train': [], 'val': [], 'test': []}
    for t in trades:
        if t['entry_bar'] < train_end:
            buckets['train'].append(t)
        elif t['entry_bar'] < val_end:
            buckets['val'].append(t)
        else:
            buckets['test'].append(t)
    return {
        k: {'trades': len(v),
            'PnL_1x': round(sum(x['pnl_pct'] for x in v), 2)}
        for k, v in buckets.items()
    }


# ---------------------------------------------------------------------------
def main():
    print("Loading BTC 15m data...")
    df15 = load_15m()
    n_bars = len(df15)
    print(f"  {n_bars} bars, period {df15['timestamp'].iloc[0]} ~ {df15['timestamp'].iloc[-1]}")

    results = {'sweep': [], 'meta': {
        'base_cfg': {k: v for k, v in BASE_CFG.items() if k != 'progressive_trail'},
        'progressive_trail': BASE_CFG['progressive_trail'],
        'perturbation': PERTURBATION,
        'drift_tol_bars': DRIFT_TOL_BARS,
        'fee_rt_pct': FEE_RT_PCT,
        'leverage_3x_reference': LEVERAGE,
        'bars': n_bars,
        'period_start': str(df15['timestamp'].iloc[0]),
        'period_end': str(df15['timestamp'].iloc[-1]),
    }}

    for cd in COOLDOWN_VALUES:
        cfg_base = {**BASE_CFG, 'min_bars_between': cd}
        cfg_pert = {**cfg_base, **PERTURBATION}

        print(f"\nmin_bars_between={cd}: running baseline (trail_K=2.5) + perturbed (trail_K=2.45) + exit_delay variant...")
        trades_base = run_bt(df15, cfg_base)
        trades_pert = run_bt(df15, cfg_pert)
        # H2: realistic intrabar-timing perturbation. Delay every 5th trail
        # exit by 1 bar (~20% of trail exits) to simulate LIVE intrabar tick
        # firing at different bar than BT bar-close evaluation.
        trades_exitdelay = run_bt(df15, cfg_base, exit_delay_every=5)

        sum_base = summarize(trades_base, n_bars)
        sum_pert = summarize(trades_pert, n_bars)
        sum_exitd = summarize(trades_exitdelay, n_bars)
        drift = drift_fraction(trades_base, trades_pert)
        drift_exitd = drift_fraction(trades_base, trades_exitdelay)
        tws_base = three_way_split(trades_base, n_bars)
        tws_pert = three_way_split(trades_pert, n_bars)

        # GO-criteria (task spec): daily_pnl ≥ current production, drift < 50% of current,
        #   3-way all positive
        rec = {
            'min_bars_between': cd,
            'baseline': sum_base,
            'perturbed_trail_K': sum_pert,
            'perturbed_exit_delay': sum_exitd,
            'drift_trail_K': drift,
            'drift_exit_delay': drift_exitd,
            'three_way_baseline': tws_base,
            'three_way_perturbed': tws_pert,
        }
        results['sweep'].append(rec)

        print(f"  baseline : trades={sum_base['trades']} WR={sum_base['WR']:.1f} "
              f"PnL={sum_base['PnL_1x']:+.2f} daily={sum_base['daily_PnL_1x']:+.4f} "
              f"MDD={sum_base['MDD_1x']:.2f} RR={sum_base['RR']:.2f}")
        print(f"  pert_trK : trades={sum_pert['trades']} WR={sum_pert['WR']:.1f} "
              f"PnL={sum_pert['PnL_1x']:+.2f} daily={sum_pert['daily_PnL_1x']:+.4f}")
        print(f"  pert_eD5 : trades={sum_exitd['trades']} WR={sum_exitd['WR']:.1f} "
              f"PnL={sum_exitd['PnL_1x']:+.2f} daily={sum_exitd['daily_PnL_1x']:+.4f}")
        print(f"  drift tK : fraction={drift['drift_fraction']:.3f} "
              f"unmA={drift['unmatched_a']} unmB={drift['unmatched_b']}")
        print(f"  drift eD : fraction={drift_exitd['drift_fraction']:.3f} "
              f"unmA={drift_exitd['unmatched_a']} unmB={drift_exitd['unmatched_b']}")
        print(f"  3-way baseline: train={tws_base['train']['PnL_1x']:+.2f} "
              f"val={tws_base['val']['PnL_1x']:+.2f} test={tws_base['test']['PnL_1x']:+.2f}")

    # GO gate evaluation — use max(trail_K drift, exit_delay drift) as conservative
    current = next(r for r in results['sweep'] if r['min_bars_between'] == 2)
    current_daily = current['baseline']['daily_PnL_1x']
    current_drift_tK = current['drift_trail_K']['drift_fraction']
    current_drift_eD = current['drift_exit_delay']['drift_fraction']
    current_drift = max(current_drift_tK, current_drift_eD)

    for r in results['sweep']:
        tws = r['three_way_baseline']
        r_drift = max(r['drift_trail_K']['drift_fraction'],
                      r['drift_exit_delay']['drift_fraction'])
        c1 = r['baseline']['daily_PnL_1x'] >= current_daily
        c2 = r_drift < 0.5 * current_drift
        c3 = all(tws[k]['PnL_1x'] > 0 for k in ('train', 'val', 'test'))
        r['max_drift'] = round(r_drift, 4)
        r['go_gate'] = {
            'daily_not_worse': c1,
            'drift_halved': c2,
            'three_way_all_positive': c3,
            'overall': c1 and c2 and c3,
        }

    results['meta']['current_baseline_daily_PnL_1x'] = current_daily
    results['meta']['current_baseline_drift_fraction_trail_K'] = current_drift_tK
    results['meta']['current_baseline_drift_fraction_exit_delay'] = current_drift_eD
    results['meta']['current_baseline_drift_fraction_max'] = current_drift

    # Summary table
    print("\n" + "=" * 130)
    print(f"{'N':>3} {'trades':>7} {'WR':>6} {'PnL_1x':>9} {'daily':>8} {'MDD':>7} "
          f"{'dr_tK':>7} {'dr_eD':>7} {'train':>7} {'val':>7} {'test':>7} {'go':<4}")
    print("-" * 130)
    for r in results['sweep']:
        b = r['baseline']
        d_tK = r['drift_trail_K']
        d_eD = r['drift_exit_delay']
        tw = r['three_way_baseline']
        g = r['go_gate']
        flags = ''.join(['+' if g['daily_not_worse'] else '-',
                         '+' if g['drift_halved'] else '-',
                         '+' if g['three_way_all_positive'] else '-'])
        print(f"{r['min_bars_between']:>3} {b['trades']:>7} {b['WR']:>6.1f} "
              f"{b['PnL_1x']:>+9.2f} {b['daily_PnL_1x']:>+8.4f} {b['MDD_1x']:>7.2f} "
              f"{d_tK['drift_fraction']:>7.3f} {d_eD['drift_fraction']:>7.3f} "
              f"{tw['train']['PnL_1x']:>+7.2f} {tw['val']['PnL_1x']:>+7.2f} "
              f"{tw['test']['PnL_1x']:>+7.2f} {flags:<4}")
    print("=" * 130)
    print("Legend: dr_tK = drift vs trail_K=2.45 perturbation")
    print("        dr_eD = drift vs exit_delay_every=5 perturbation (realistic intrabar-tick mimic)")
    print("        go flags = [daily_not_worse][drift<50%current][3-way_all_positive]")

    # Verdict
    winners = [r for r in results['sweep'] if r['go_gate']['overall']]
    print(f"\nConfigs passing all 3 GO criteria: {len(winners)}")
    for w in winners:
        print(f"  N={w['min_bars_between']}: daily={w['baseline']['daily_PnL_1x']:+.4f}, "
              f"drift={w['drift']['drift_fraction']:.3f}")

    # Save
    results['generated_at'] = datetime.now().isoformat()
    out_path = ROOT / 'results' / 'entry_sequencing_drift_20260424.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
