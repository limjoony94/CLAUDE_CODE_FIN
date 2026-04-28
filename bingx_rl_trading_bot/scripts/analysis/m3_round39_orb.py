"""M3-R39 — Opening Range Breakout (5m + MTF) LOCKED OOS.

Pre-registered (claudedocs/m3_round39_orb_prereg.md, commit 39d6276).
Honest prior ~0% (6/6 prior FP, R38 inconclusive).
NEW: pre-run vacuity gate ≥0.5/day applied.

Locked params (theory-based, NO sweep):
  opening_range_minutes=60, body_min_ratio=0.4, volume_mult=1.0, max_entries_per_day=1

5m signals projected to 15m exit framework (consistent with R36/R37/R38).
Trend: 1h SMA200 + 4h EMA20/50 + 15m EMA20/50.

Pre-run vacuity gate: signal frequency ≥0.5/day required for tests to be conducted.
ALL 3 OOS tests required to pass: WF 5-fold, Bootstrap 1000×3d, Train/Test 60/40.
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_round35_15m_deep import add_15m_extras, trade_summary
from m3_round30_c1_production_exact import run_bt_c1_production
from m3_round29_c1_exact_revalidation import prepare_15m_data
from m3_round38_vwap_reversion import prepare_5m_15m_data


LOCKED = {
    'opening_range_minutes': 60,
    'break_buffer_pct': 0.0,
    'body_min_ratio': 0.4,
    'volume_mult': 1.0,
    'max_entries_per_day': 1,
}
FRICTION = 0.07
VACUITY_FLOOR_PER_DAY = 0.5


def add_5m_orb_features(df_5m):
    """Add 5m volume SMA20 if not present."""
    df = df_5m.copy()
    if 'vol_sma20_5m' not in df.columns:
        df['vol_sma20_5m'] = pd.Series(df['volume'].values).rolling(20, min_periods=20).mean().values
    df['session_date'] = pd.to_datetime(df['timestamp']).dt.date
    return df


def entry_orb_5m_to_15m(df_5m, df_15m, valid_15m):
    """Generate ORB signals at 5m granularity, project to 15m bar index."""
    df_5m = add_5m_orb_features(df_5m)

    op5 = df_5m['open'].values
    hi5 = df_5m['high'].values
    lo5 = df_5m['low'].values
    cl5 = df_5m['close'].values
    vol5 = df_5m['volume'].values
    vol_sma5 = df_5m['vol_sma20_5m'].values
    sess = df_5m['session_date'].values
    ts5 = pd.to_datetime(df_5m['timestamp'])

    # Trend filters from 15m frame
    sma_long_15m = df_15m['sma200_long'].fillna(False).astype(bool).values
    h4_long_15m = df_15m['htf4_long'].fillna(False).astype(bool).values
    ema20_15m = df_15m['ema20_15m'].values
    ema50_15m = df_15m['ema50_15m'].values

    # Map each 5m bar to its 15m bar index
    ts_5m_floor = ts5.dt.floor('15min')
    ts_15m = pd.to_datetime(df_15m['timestamp'])
    idx_15m_lookup = {t: i for i, t in enumerate(ts_15m)}
    bar_idx_for_5m = ts_5m_floor.map(idx_15m_lookup).values

    # Compute opening range per session (UTC 00:00)
    # First N minutes = first N/5 5m bars where bar's UTC time < OR_minutes from session start
    or_minutes = LOCKED['opening_range_minutes']
    or_bars = or_minutes // 5  # 12 5m bars

    # For each session_date, compute OR high/low (max/min of first or_bars)
    session_or = {}  # date -> (or_high, or_low, or_complete_index)
    for d, group in pd.DataFrame({'date': sess, 'high': hi5, 'low': lo5,
                                    'idx': np.arange(len(sess))}).groupby('date'):
        if len(group) < or_bars:
            continue
        first_n = group.iloc[:or_bars]
        or_h = float(first_n['high'].max())
        or_l = float(first_n['low'].min())
        or_complete_idx = int(first_n.iloc[-1]['idx'])  # last bar index of OR
        session_or[d] = (or_h, or_l, or_complete_idx)

    n5 = len(df_5m)
    sigs_15m = []
    seen_15m_idx = set()
    daily_entry_count = {}  # date -> count

    body_min = LOCKED['body_min_ratio']
    vmin = LOCKED['volume_mult']
    max_per_day = LOCKED['max_entries_per_day']

    for i in range(or_bars + 1, n5):
        d = sess[i]
        if d not in session_or:
            continue
        or_h, or_l, or_complete_idx = session_or[d]
        if i <= or_complete_idx:
            continue  # still within OR formation

        if any(pd.isna(x) for x in (op5[i], hi5[i], lo5[i], cl5[i], vol5[i], vol_sma5[i])):
            continue
        rng = hi5[i] - lo5[i]
        if rng <= 0:
            continue

        # Body filter
        body = cl5[i] - op5[i]
        if abs(body) / rng < body_min:
            continue

        # Volume confirmation
        if vol5[i] < vmin * vol_sma5[i]:
            continue

        # Direction: clean break of OR
        long_break = (cl5[i] > or_h) and (body > 0)
        short_break = (cl5[i] < or_l) and (body < 0)
        if not (long_break or short_break):
            continue

        # 15m bar lookup
        bidx = bar_idx_for_5m[i]
        if pd.isna(bidx):
            continue
        bidx = int(bidx)
        if bidx >= len(df_15m) or bidx < 0:
            continue
        if not valid_15m[bidx]:
            continue
        if bidx in seen_15m_idx:
            continue

        # Trend confluence
        sma_up = sma_long_15m[bidx]
        h4_up = h4_long_15m[bidx]
        if pd.isna(ema20_15m[bidx]) or pd.isna(ema50_15m[bidx]):
            continue
        ema_up = ema20_15m[bidx] > ema50_15m[bidx]
        ema_down = ema20_15m[bidx] < ema50_15m[bidx]

        long_setup = long_break and sma_up and h4_up and ema_up
        short_setup = short_break and (not sma_up) and (not h4_up) and ema_down

        if not (long_setup or short_setup):
            continue

        # Max entries per day
        cnt = daily_entry_count.get(d, 0)
        if cnt >= max_per_day:
            continue

        if long_setup:
            sigs_15m.append((bidx, 'LONG'))
        else:
            sigs_15m.append((bidx, 'SHORT'))
        seen_15m_idx.add(bidx)
        daily_entry_count[d] = cnt + 1

    sigs_15m.sort(key=lambda x: x[0])
    return sigs_15m


def main():
    print("=" * 100)
    print("M3-R39 — Opening Range Breakout (5m → 15m exit) LOCKED OOS")
    print("=" * 100)
    print(f"Locked params: {LOCKED}")
    print(f"Friction: {FRICTION}")
    print(f"Pre-reg: claudedocs/m3_round39_orb_prereg.md (commit 39d6276)")
    print(f"Honest prior: ~0% (6/6 FP + R38 inconclusive)")
    print(f"NEW: pre-run vacuity gate ≥{VACUITY_FLOOR_PER_DAY}/day\n")

    df_5m, df_15m, valid_15m = prepare_5m_15m_data()
    n_5m = len(df_5m)
    n_15m = len(df_15m)
    days = n_15m / 96
    print(f"5m bars: {n_5m:,} | 15m bars: {n_15m:,} | days: {days:.0f}\n")

    # ----------------------------------------------------------------------
    # PRE-RUN VACUITY GATE
    # ----------------------------------------------------------------------
    sigs_full = entry_orb_5m_to_15m(df_5m, df_15m, valid_15m)
    sig_per_day = len(sigs_full) / days
    print(f"Full-dataset signals: {len(sigs_full)} ({sig_per_day:.3f}/day)")

    if sig_per_day < VACUITY_FLOOR_PER_DAY:
        print(f"\n  ⚠️  VACUITY GATE FAIL — {sig_per_day:.3f}/day < {VACUITY_FLOOR_PER_DAY}/day")
        print(f"  R39 declared INCONCLUSIVE (vacuous test). Pile unchanged at 6/6.")
        out = {
            'date': datetime.now(timezone.utc).isoformat(),
            'pre_reg': 'claudedocs/m3_round39_orb_prereg.md',
            'pre_reg_commit': '39d6276',
            'locked_params': LOCKED,
            'friction': FRICTION,
            'full_signals_count': len(sigs_full),
            'signals_per_day': sig_per_day,
            'vacuity_gate_pass': False,
            'verdict': 'INCONCLUSIVE - vacuous test',
        }
        p = ROOT / 'results' / f'm3_r39_orb_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
        print(f"\nSaved: {p}")
        return

    print(f"  ✓ Vacuity gate PASS — proceed to OOS tests\n")

    ts15 = pd.to_datetime(df_15m['timestamp'])
    ts5 = pd.to_datetime(df_5m['timestamp'])

    # ----------------------------------------------------------------------
    # TEST 1: WF 5-fold expanding
    # ----------------------------------------------------------------------
    print("=" * 100)
    print("TEST 1 — WF 5-fold expanding (locked params, friction=0.07)")
    print("Pass criterion: ≥3/5 folds daily_net > 0")
    print("=" * 100)
    fold_size = n_15m // 6
    wf_results = []
    for fold_i in range(5):
        te_s = (fold_i + 1) * fold_size
        te_e = min(te_s + fold_size, n_15m)
        df_15m_f = df_15m.iloc[te_s:te_e].reset_index(drop=True)
        v_f = valid_15m[te_s:te_e]
        t_start = ts15.iloc[te_s]
        t_end = ts15.iloc[te_e - 1] if te_e <= n_15m else ts15.iloc[-1]
        mask5 = (ts5 >= t_start) & (ts5 <= t_end + pd.Timedelta(minutes=14))
        df_5m_f = df_5m.loc[mask5].reset_index(drop=True)
        sigs_f = entry_orb_5m_to_15m(df_5m_f, df_15m_f, v_f)
        trades = run_bt_c1_production(df_15m_f, sigs_f, friction=FRICTION)
        s_f = trade_summary(trades)
        if s_f is None:
            wf_results.append({'fold': fold_i+1, 'n': 0, 'daily': None, 'wr': None, 'rr': None, 'avg_g': None})
            print(f"  fold {fold_i+1}: n=0 (NO TRADES)")
            continue
        wf_results.append({'fold': fold_i+1, 'n': s_f['n'], 'daily': s_f['daily_net'],
                            'wr': s_f['wr_pct'], 'rr': s_f['rr'], 'avg_g': s_f['avg_gross']})
        print(f"  fold {fold_i+1}: n={s_f['n']:>3} daily={s_f['daily_net']:>+.4f}% WR={s_f['wr_pct']:>5.1f}% RR={s_f['rr']:>5.2f} avg_g={s_f['avg_gross']:>+.4f}%")
    wf_pos = sum(1 for r in wf_results if r['daily'] is not None and r['daily'] > 0)
    wf_pass = wf_pos >= 3
    print(f"\n  WF positive folds: {wf_pos}/5 → {'PASS' if wf_pass else 'FAIL'}")

    # ----------------------------------------------------------------------
    # TEST 2: Bootstrap
    # ----------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("TEST 2 — Bootstrap 1000 × 3-day (locked params, friction=0.07)")
    print("Pass criterion: pos_rate ≥ 50%")
    print("=" * 100)
    bars_per_3day_15m = 3 * 96
    max_start = n_15m - bars_per_3day_15m - 1
    random.seed(42)
    starts = random.sample(range(max_start), min(1000, max_start))
    cand_pnls = []
    for st in starts:
        en = st + bars_per_3day_15m
        df_15m_w = df_15m.iloc[st:en].reset_index(drop=True)
        v_w = valid_15m[st:en]
        t_start = ts15.iloc[st]
        t_end = ts15.iloc[en - 1] if en <= n_15m else ts15.iloc[-1]
        mask5 = (ts5 >= t_start) & (ts5 <= t_end + pd.Timedelta(minutes=14))
        df_5m_w = df_5m.loc[mask5].reset_index(drop=True)
        sigs_w = entry_orb_5m_to_15m(df_5m_w, df_15m_w, v_w)
        trades = run_bt_c1_production(df_15m_w, sigs_w, friction=FRICTION)
        cand_pnls.append(sum(t['net_pct'] for t in trades) if trades else 0)
    arr = np.array(cand_pnls)
    bs_mean = float(arr.mean())
    bs_pos_rate = float((arr > 0).mean())
    bs_p5 = float(np.percentile(arr, 5))
    bs_active = float((arr != 0).mean())
    bs_pass = bs_pos_rate >= 0.5
    print(f"  mean: {bs_mean:+.4f}%")
    print(f"  pos_rate (>0): {bs_pos_rate:.4f} ({int(bs_pos_rate*1000)}/1000)")
    print(f"  active windows (≠0): {bs_active:.4f} ({int(bs_active*1000)}/1000)")
    print(f"  p5: {bs_p5:+.4f}%")
    print(f"\n  Pos rate ≥ 0.50: {'PASS' if bs_pass else 'FAIL'}")

    # ----------------------------------------------------------------------
    # TEST 3: Train/Test
    # ----------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("TEST 3 — Train/Test 60/40 (locked params, friction=0.07)")
    print("Pass criterion: BOTH train AND test daily_net > 0")
    print("=" * 100)
    train_end = int(n_15m * 0.6)
    df_15m_tr = df_15m.iloc[:train_end].reset_index(drop=True)
    df_15m_te = df_15m.iloc[train_end:].reset_index(drop=True)
    v_tr = valid_15m[:train_end]
    v_te = valid_15m[train_end:]
    t_tr_end = ts15.iloc[train_end - 1]
    mask5_tr = ts5 <= t_tr_end + pd.Timedelta(minutes=14)
    mask5_te = ts5 > t_tr_end + pd.Timedelta(minutes=14)
    df_5m_tr = df_5m.loc[mask5_tr].reset_index(drop=True)
    df_5m_te = df_5m.loc[mask5_te].reset_index(drop=True)
    sigs_tr = entry_orb_5m_to_15m(df_5m_tr, df_15m_tr, v_tr)
    sigs_te = entry_orb_5m_to_15m(df_5m_te, df_15m_te, v_te)
    trades_tr = run_bt_c1_production(df_15m_tr, sigs_tr, friction=FRICTION)
    trades_te = run_bt_c1_production(df_15m_te, sigs_te, friction=FRICTION)
    s_tr = trade_summary(trades_tr)
    s_te = trade_summary(trades_te)

    if s_tr is None or s_te is None:
        n_tr_str = 'NULL' if s_tr is None else 'n=' + str(s_tr['n'])
        n_te_str = 'NULL' if s_te is None else 'n=' + str(s_te['n'])
        print(f"  train: {n_tr_str}")
        print(f"  test:  {n_te_str}")
        tt_pass = False
    else:
        print(f"  train: n={s_tr['n']:>3} daily={s_tr['daily_net']:>+.4f}% WR={s_tr['wr_pct']:>5.1f}% RR={s_tr['rr']:>5.2f} avg_g={s_tr['avg_gross']:>+.4f}%")
        print(f"  test:  n={s_te['n']:>3} daily={s_te['daily_net']:>+.4f}% WR={s_te['wr_pct']:>5.1f}% RR={s_te['rr']:>5.2f} avg_g={s_te['avg_gross']:>+.4f}%")
        tt_pass = (s_tr['daily_net'] > 0) and (s_te['daily_net'] > 0)
    print(f"\n  Train/Test both > 0: {'PASS' if tt_pass else 'FAIL'}")

    # FINAL VERDICT
    print(f"\n{'='*100}")
    print("M3-R39 FINAL VERDICT (pre-registered, all 3 required)")
    print(f"{'='*100}")
    print(f"  Vacuity gate (≥{VACUITY_FLOOR_PER_DAY}/day):              PASS  ({sig_per_day:.3f}/day)")
    print(f"  Test 1 (WF ≥3/5):                       {'PASS' if wf_pass else 'FAIL'}  ({wf_pos}/5)")
    print(f"  Test 2 (Bootstrap ≥50%):                {'PASS' if bs_pass else 'FAIL'}  ({bs_pos_rate:.4f})")
    print(f"  Test 3 (Train+Test sign-agree):          {'PASS' if tt_pass else 'FAIL'}")
    all_pass = wf_pass and bs_pass and tt_pass
    print(f"\n  OVERALL: {'ALL 3 PASS — call advisor before any claim' if all_pass else 'FAIL — 7th OOS negative'}")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg': 'claudedocs/m3_round39_orb_prereg.md',
        'pre_reg_commit': '39d6276',
        'locked_params': LOCKED,
        'friction': FRICTION,
        'full_signals_count': len(sigs_full),
        'signals_per_day': sig_per_day,
        'vacuity_gate_pass': True,
        'test_1_wf': {'folds': wf_results, 'pos_count': wf_pos, 'pass': wf_pass},
        'test_2_bootstrap': {'mean': bs_mean, 'pos_rate': bs_pos_rate,
                              'p5': bs_p5, 'active_rate': bs_active, 'pass': bs_pass},
        'test_3_train_test': {
            'train': {kk: vv for kk, vv in s_tr.items() if kk != 'trades'} if s_tr else None,
            'test': {kk: vv for kk, vv in s_te.items() if kk != 'trades'} if s_te else None,
            'pass': tt_pass,
        },
        'all_pass': bool(all_pass),
    }
    p = ROOT / 'results' / f'm3_r39_orb_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
