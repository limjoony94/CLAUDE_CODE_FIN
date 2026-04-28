"""M3-R28 — R21 + TOD session filter with proper train/test split.

R27 found 4 positive hours (3, 9, 14, 18 UTC) post-hoc on full data.
R28: identify positive hours from train (60%), apply filter on test (40%) holdout.
Tests if TOD selection survives out-of-sample.
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_round20_dynamic_scalping import prepare_5m_data
from m3_round21_pattern_structure import (add_sma200_1h, entry_psi_prime, run_bt_structural)


def trade_summary_with_trades(trades):
    if not trades: return None
    nets = [t['net_pct'] for t in trades]
    grosses = [t['gross_pct'] for t in trades]
    days = (pd.to_datetime(trades[-1]['exit_ts']) - pd.to_datetime(trades[0]['entry_ts'])).days
    if days == 0: days = 1
    wins = sum(1 for x in nets if x > 0)
    n = len(nets)
    win_pnls = [x for x in nets if x > 0]
    loss_pnls = [x for x in nets if x <= 0]
    rr = abs((sum(win_pnls)/max(1,len(win_pnls))) / (sum(loss_pnls)/max(1,len(loss_pnls)))) if loss_pnls else float('inf')
    return {
        'n': n, 'days': days, 'per_day': round(n/days, 3),
        'sum_net': round(sum(nets), 2), 'avg_net': round(sum(nets)/n, 4),
        'avg_gross': round(sum(grosses)/n, 4),
        'wr_pct': round(100*wins/n, 2), 'rr': round(rr, 3),
        'daily_net': round(sum(nets)/days, 4),
        'trades': trades,
    }


def hour_breakdown(trades):
    by_hour = {}
    for t in trades:
        ts = pd.to_datetime(t['entry_ts'])
        h = ts.hour
        by_hour.setdefault(h, []).append(t['net_pct'])
    out = {}
    for h, arr in by_hour.items():
        out[h] = {'n': len(arr), 'avg_net': float(np.mean(arr)),
                   'wr_pct': float(np.mean(np.array(arr) > 0) * 100),
                   'sum_net': float(np.sum(arr))}
    return out


def main():
    df, h1, h4, valid = prepare_5m_data()
    df = add_sma200_1h(df)
    valid = valid & (~df['sma200_long'].isna()).values

    n_total = len(df)
    train_end = int(n_total * 0.6)

    df_tr = df.iloc[:train_end].reset_index(drop=True)
    df_te = df.iloc[train_end:].reset_index(drop=True)
    h1_tr, h1_te = h1[:train_end], h1[train_end:]
    h4_tr, h4_te = h4[:train_end], h4[train_end:]
    v_tr, v_te = valid[:train_end], valid[train_end:]
    print(f"Train: {train_end} bars ({train_end/(24*12):.0f}d), Test: {n_total-train_end} bars ({(n_total-train_end)/(24*12):.0f}d)\n")

    # Step 1: Run R21 on train, identify positive hours
    print("=" * 80); print("Step 1: R21 train run + TOD breakdown"); print("=" * 80)
    sigs_tr = entry_psi_prime(df_tr, h1_tr, h4_tr, v_tr)
    print(f"  Train R21 signals: {len(sigs_tr)}")
    trades_tr = run_bt_structural(df_tr, sigs_tr, friction_tp=0.04, friction_sl=0.07)
    s_tr = trade_summary_with_trades(trades_tr)
    print(f"  Train: n={s_tr['n']}, daily={s_tr['daily_net']:+.4f}%, WR={s_tr['wr_pct']}%, avg_gross={s_tr['avg_gross']:+.4f}%")

    train_tod = hour_breakdown(s_tr['trades'])
    print(f"\n  Train TOD breakdown:")
    print(f"    {'hour':>4} {'n':>5} {'avg_net':>10} {'WR':>6}")
    train_pos_hours = []
    for h in sorted(train_tod.keys()):
        info = train_tod[h]
        if info['n'] >= 5 and info['avg_net'] > 0:
            train_pos_hours.append(h)
            marker = '⭐'
        else:
            marker = '  '
        print(f"    {h:>4} {info['n']:>5} {info['avg_net']:>+9.4f}% {info['wr_pct']:>5.1f}% {marker}")
    print(f"\n  Train positive hours (n≥5, avg_net>0): {train_pos_hours}")

    if len(train_pos_hours) == 0:
        print("\n  No train positive hours — cannot apply filter.")
        return

    # Step 2: Apply train-derived filter to test (OOS)
    print(f"\n{'='*80}\nStep 2: Apply train-derived hours {train_pos_hours} to TEST (OOS)\n{'='*80}")
    sigs_te_full = entry_psi_prime(df_te, h1_te, h4_te, v_te)
    # Filter sigs to only train_pos_hours
    timestamps_te = pd.to_datetime(df_te['timestamp'])
    sigs_te_filtered = [(i, d) for i, d in sigs_te_full if timestamps_te.iloc[i].hour in train_pos_hours]
    print(f"  Test full signals: {len(sigs_te_full)}")
    print(f"  Test filtered (TOD): {len(sigs_te_filtered)}")

    # Run BT on full test (no filter) and filtered test
    trades_te_full = run_bt_structural(df_te, sigs_te_full, friction_tp=0.04, friction_sl=0.07)
    trades_te_filt = run_bt_structural(df_te, sigs_te_filtered, friction_tp=0.04, friction_sl=0.07)

    s_te_full = trade_summary_with_trades(trades_te_full)
    s_te_filt = trade_summary_with_trades(trades_te_filt)

    print(f"\n  Test FULL (no filter): ", end="")
    if s_te_full:
        print(f"n={s_te_full['n']}, daily={s_te_full['daily_net']:+.4f}%, WR={s_te_full['wr_pct']}%, avg_g={s_te_full['avg_gross']:+.4f}%")
    print(f"  Test FILTERED (TOD={train_pos_hours}): ", end="")
    if s_te_filt:
        print(f"n={s_te_filt['n']}, daily={s_te_filt['daily_net']:+.4f}%, WR={s_te_filt['wr_pct']}%, avg_g={s_te_filt['avg_gross']:+.4f}%")

    # Step 3: Compare filter improvement
    if s_te_full and s_te_filt:
        print(f"\n  Improvement from filter:")
        print(f"    daily_net: {s_te_full['daily_net']:+.4f}% → {s_te_filt['daily_net']:+.4f}% (Δ={s_te_filt['daily_net']-s_te_full['daily_net']:+.4f})")
        print(f"    WR: {s_te_full['wr_pct']}% → {s_te_filt['wr_pct']}% (Δ={s_te_filt['wr_pct']-s_te_full['wr_pct']:+.2f}pp)")
        print(f"    avg_gross: {s_te_full['avg_gross']:+.4f}% → {s_te_filt['avg_gross']:+.4f}% (Δ={s_te_filt['avg_gross']-s_te_full['avg_gross']:+.4f})")

    # Step 4: Test TOD breakdown OOS — do "positive train hours" remain positive in test?
    if s_te_full:
        test_tod = hour_breakdown(s_te_full['trades'])
        print(f"\n{'='*80}\nStep 4: Train positive hours' performance in TEST\n{'='*80}")
        print(f"  {'hour':>4} {'train_avg':>11} {'train_n':>8} {'test_avg':>11} {'test_n':>8} {'consistent':>11}")
        consistent = 0; total_checked = 0
        for h in train_pos_hours:
            tr_info = train_tod[h]
            te_info = test_tod.get(h, {'n': 0, 'avg_net': None})
            cons = (te_info['n'] >= 3 and te_info['avg_net'] is not None and te_info['avg_net'] > 0)
            print(f"  {h:>4} {tr_info['avg_net']:>+10.4f}% {tr_info['n']:>8} "
                  f"{te_info.get('avg_net', 0):>+10.4f}% {te_info.get('n', 0):>8} "
                  f"{'YES' if cons else 'no':>11}")
            if te_info['n'] >= 3:
                total_checked += 1
                if cons: consistent += 1
        print(f"\n  Consistent positive train→test: {consistent}/{total_checked} (random expectation ~50% under null)")

    # Verdict
    print(f"\n{'='*80}\nM3-R28 VERDICT\n{'='*80}")
    print(f"  Test FULL daily: {s_te_full['daily_net'] if s_te_full else 'N/A':+.4f}%")
    print(f"  Test FILTERED daily: {s_te_filt['daily_net'] if s_te_filt else 'N/A':+.4f}%")
    print(f"  Filter helps OOS: {s_te_filt and s_te_full and s_te_filt['daily_net'] > s_te_full['daily_net']}")
    print(f"  Filter passes strict (≥+0.2%/day): {s_te_filt and s_te_filt['daily_net'] >= 0.2}")
    print(f"  Train→test consistency: {consistent}/{total_checked}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'train_pos_hours': train_pos_hours,
           'train_summary': {kk: vv for kk, vv in s_tr.items() if kk != 'trades'} if s_tr else None,
           'test_full_summary': {kk: vv for kk, vv in s_te_full.items() if kk != 'trades'} if s_te_full else None,
           'test_filtered_summary': {kk: vv for kk, vv in s_te_filt.items() if kk != 'trades'} if s_te_filt else None,
           'train_tod': train_tod,
           'test_tod': test_tod if s_te_full else None,
           'consistent': consistent if s_te_full else None,
           'total_checked': total_checked if s_te_full else None}
    p = ROOT / 'results' / f'm3_r28_tod_filter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
