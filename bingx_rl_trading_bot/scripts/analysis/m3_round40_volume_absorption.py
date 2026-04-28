"""M3-R40 — Volume Absorption + Trend Continuation (5m + MTF) LOCKED OOS.

Pre-registered (claudedocs/m3_round40_volume_absorption_prereg.md, commit 57bad84).
Honest prior ~0% (7/7 prior FP + R38 inconclusive).

Locked params (theory-based):
  absorption_vol_mult=2.0, absorption_body_ratio_max=0.3, confirmation_body_min=0.4

Mechanism: high-volume small-body absorption bar → next bar trend continuation.
Theory: Wyckoff absorption + modern order flow proxy.
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
from m3_round38_vwap_reversion import prepare_5m_15m_data


LOCKED = {
    'absorption_vol_mult': 2.0,
    'absorption_body_ratio_max': 0.3,
    'confirmation_body_min': 0.4,
    'wick_imbalance_required': True,
}
FRICTION = 0.07
VACUITY_FLOOR_PER_DAY = 0.5


def entry_volume_absorption_5m(df_5m, df_15m, valid_15m):
    """Detect volume absorption + next-bar trend continuation."""
    df_5m = df_5m.copy()
    if 'vol_sma20_5m' not in df_5m.columns:
        df_5m['vol_sma20_5m'] = pd.Series(df_5m['volume'].values).rolling(20, min_periods=20).mean().values

    op5 = df_5m['open'].values
    hi5 = df_5m['high'].values
    lo5 = df_5m['low'].values
    cl5 = df_5m['close'].values
    vol5 = df_5m['volume'].values
    vol_sma5 = df_5m['vol_sma20_5m'].values

    sma_long_15m = df_15m['sma200_long'].fillna(False).astype(bool).values
    h4_long_15m = df_15m['htf4_long'].fillna(False).astype(bool).values
    ema20_15m = df_15m['ema20_15m'].values
    ema50_15m = df_15m['ema50_15m'].values

    ts5 = pd.to_datetime(df_5m['timestamp']).dt.floor('15min')
    ts_15m = pd.to_datetime(df_15m['timestamp'])
    idx_15m_lookup = {t: i for i, t in enumerate(ts_15m)}
    bar_idx_for_5m = ts5.map(idx_15m_lookup).values

    n5 = len(df_5m)
    sigs_15m = []
    seen_15m_idx = set()

    vol_mult = LOCKED['absorption_vol_mult']
    body_max = LOCKED['absorption_body_ratio_max']
    body_min_conf = LOCKED['confirmation_body_min']

    for i in range(50, n5 - 1):
        # Absorption bar at i
        if any(pd.isna(x) for x in (op5[i], hi5[i], lo5[i], cl5[i], vol5[i], vol_sma5[i])):
            continue
        rng_i = hi5[i] - lo5[i]
        if rng_i <= 0:
            continue

        body_i = cl5[i] - op5[i]
        body_ratio_i = abs(body_i) / rng_i
        if body_ratio_i > body_max:
            continue
        if vol5[i] < vol_mult * vol_sma5[i]:
            continue

        # Wick imbalance
        upper_wick = hi5[i] - max(cl5[i], op5[i])
        lower_wick = min(cl5[i], op5[i]) - lo5[i]
        long_absorption = lower_wick > upper_wick  # buyers absorbed
        short_absorption = upper_wick > lower_wick  # sellers absorbed

        if not (long_absorption or short_absorption):
            continue

        # Confirmation bar at i+1
        if any(pd.isna(x) for x in (op5[i+1], hi5[i+1], lo5[i+1], cl5[i+1])):
            continue
        rng_c = hi5[i+1] - lo5[i+1]
        if rng_c <= 0:
            continue
        body_c = cl5[i+1] - op5[i+1]
        if abs(body_c) / rng_c < body_min_conf:
            continue

        # Direction match
        long_setup_basic = long_absorption and (cl5[i+1] > cl5[i]) and (body_c > 0)
        short_setup_basic = short_absorption and (cl5[i+1] < cl5[i]) and (body_c < 0)

        if not (long_setup_basic or short_setup_basic):
            continue

        # 15m bar lookup at i+1 (entry signal index)
        bidx = bar_idx_for_5m[i+1]
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

        long_setup = long_setup_basic and sma_up and h4_up and ema_up
        short_setup = short_setup_basic and (not sma_up) and (not h4_up) and ema_down

        if long_setup:
            sigs_15m.append((bidx, 'LONG'))
            seen_15m_idx.add(bidx)
        elif short_setup:
            sigs_15m.append((bidx, 'SHORT'))
            seen_15m_idx.add(bidx)

    sigs_15m.sort(key=lambda x: x[0])
    return sigs_15m


def main():
    print("=" * 100)
    print("M3-R40 — Volume Absorption (5m → 15m exit) LOCKED OOS")
    print("=" * 100)
    print(f"Locked params: {LOCKED}")
    print(f"Friction: {FRICTION}")
    print(f"Pre-reg: claudedocs/m3_round40_volume_absorption_prereg.md (commit 57bad84)")
    print(f"Honest prior: ~0% (7/7 FP + R38 inconclusive)\n")

    df_5m, df_15m, valid_15m = prepare_5m_15m_data()
    n_5m = len(df_5m)
    n_15m = len(df_15m)
    days = n_15m / 96
    print(f"5m bars: {n_5m:,} | 15m bars: {n_15m:,} | days: {days:.0f}\n")

    sigs_full = entry_volume_absorption_5m(df_5m, df_15m, valid_15m)
    sig_per_day = len(sigs_full) / days
    print(f"Full-dataset signals: {len(sigs_full)} ({sig_per_day:.3f}/day)")

    if sig_per_day < VACUITY_FLOOR_PER_DAY:
        print(f"\n  ⚠️  VACUITY GATE FAIL — {sig_per_day:.3f}/day < {VACUITY_FLOOR_PER_DAY}/day")
        print(f"  R40 INCONCLUSIVE (vacuous test). Pile unchanged.")
        out = {
            'date': datetime.now(timezone.utc).isoformat(),
            'pre_reg': 'claudedocs/m3_round40_volume_absorption_prereg.md',
            'pre_reg_commit': '57bad84',
            'locked_params': LOCKED, 'friction': FRICTION,
            'full_signals_count': len(sigs_full), 'signals_per_day': sig_per_day,
            'vacuity_gate_pass': False, 'verdict': 'INCONCLUSIVE - vacuous test',
        }
        p = ROOT / 'results' / f'm3_r40_absorption_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
        print(f"\nSaved: {p}")
        return

    print(f"  ✓ Vacuity gate PASS\n")

    ts15 = pd.to_datetime(df_15m['timestamp'])
    ts5 = pd.to_datetime(df_5m['timestamp'])

    # TEST 1
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
        sigs_f = entry_volume_absorption_5m(df_5m_f, df_15m_f, v_f)
        trades = run_bt_c1_production(df_15m_f, sigs_f, friction=FRICTION)
        s_f = trade_summary(trades)
        if s_f is None:
            wf_results.append({'fold': fold_i+1, 'n': 0, 'daily': None})
            print(f"  fold {fold_i+1}: n=0 (NO TRADES)")
            continue
        wf_results.append({'fold': fold_i+1, 'n': s_f['n'], 'daily': s_f['daily_net'],
                            'wr': s_f['wr_pct'], 'rr': s_f['rr'], 'avg_g': s_f['avg_gross']})
        print(f"  fold {fold_i+1}: n={s_f['n']:>3} daily={s_f['daily_net']:>+.4f}% WR={s_f['wr_pct']:>5.1f}% RR={s_f['rr']:>5.2f} avg_g={s_f['avg_gross']:>+.4f}%")
    wf_pos = sum(1 for r in wf_results if r['daily'] is not None and r['daily'] > 0)
    wf_pass = wf_pos >= 3
    print(f"\n  WF positive folds: {wf_pos}/5 → {'PASS' if wf_pass else 'FAIL'}")

    # TEST 2
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
        sigs_w = entry_volume_absorption_5m(df_5m_w, df_15m_w, v_w)
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

    # TEST 3
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
    sigs_tr = entry_volume_absorption_5m(df_5m_tr, df_15m_tr, v_tr)
    sigs_te = entry_volume_absorption_5m(df_5m_te, df_15m_te, v_te)
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

    print(f"\n{'='*100}")
    print("M3-R40 FINAL VERDICT")
    print(f"{'='*100}")
    print(f"  Vacuity gate (≥{VACUITY_FLOOR_PER_DAY}/day):              PASS  ({sig_per_day:.3f}/day)")
    print(f"  Test 1 (WF ≥3/5):                       {'PASS' if wf_pass else 'FAIL'}  ({wf_pos}/5)")
    print(f"  Test 2 (Bootstrap ≥50%):                {'PASS' if bs_pass else 'FAIL'}  ({bs_pos_rate:.4f})")
    print(f"  Test 3 (Train+Test sign-agree):          {'PASS' if tt_pass else 'FAIL'}")
    all_pass = wf_pass and bs_pass and tt_pass
    print(f"\n  OVERALL: {'ALL 3 PASS — call advisor before any claim' if all_pass else 'FAIL — 8th OOS negative'}")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg': 'claudedocs/m3_round40_volume_absorption_prereg.md',
        'pre_reg_commit': '57bad84',
        'locked_params': LOCKED, 'friction': FRICTION,
        'full_signals_count': len(sigs_full), 'signals_per_day': sig_per_day,
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
    p = ROOT / 'results' / f'm3_r40_absorption_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
