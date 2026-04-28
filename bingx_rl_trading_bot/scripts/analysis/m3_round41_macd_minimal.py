"""M3-R41 — MACD Cross + 1h SMA200 Minimal Conjunction (5m + MTF) LOCKED OOS.

Pre-registered (claudedocs/m3_round41_macd_minimal_prereg.md, commit 4da8d38).
Process change: 3-condition minimal conjunction (after R38/R40 vacuity).

Locked params (theory-based):
  macd: (12, 26, 9) standard, body_min_ratio=0.4, trend filter: 1h SMA200 only
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
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'body_min_ratio': 0.4,
}
FRICTION = 0.07
VACUITY_FLOOR_PER_DAY = 0.5


def add_5m_macd(df_5m):
    df = df_5m.copy()
    cl = df['close'].values
    cl_ser = pd.Series(cl)
    ema_fast = cl_ser.ewm(span=LOCKED['macd_fast'], adjust=False).mean()
    ema_slow = cl_ser.ewm(span=LOCKED['macd_slow'], adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=LOCKED['macd_signal'], adjust=False).mean()
    df['macd'] = macd.values
    df['macd_signal'] = signal.values
    df['macd_hist'] = (macd - signal).values
    return df


def entry_macd_cross_5m(df_5m, df_15m, valid_15m):
    df_5m = add_5m_macd(df_5m)
    op5 = df_5m['open'].values
    hi5 = df_5m['high'].values
    lo5 = df_5m['low'].values
    cl5 = df_5m['close'].values
    macd = df_5m['macd'].values
    sig = df_5m['macd_signal'].values

    sma_long_15m = df_15m['sma200_long'].fillna(False).astype(bool).values

    ts5 = pd.to_datetime(df_5m['timestamp']).dt.floor('15min')
    ts_15m = pd.to_datetime(df_15m['timestamp'])
    idx_15m_lookup = {t: i for i, t in enumerate(ts_15m)}
    bar_idx_for_5m = ts5.map(idx_15m_lookup).values

    n5 = len(df_5m)
    sigs_15m = []
    seen_15m_idx = set()

    body_min = LOCKED['body_min_ratio']

    for i in range(LOCKED['macd_slow'] + 5, n5):
        if any(pd.isna(x) for x in (op5[i], hi5[i], lo5[i], cl5[i], macd[i], sig[i], macd[i-1], sig[i-1])):
            continue
        rng = hi5[i] - lo5[i]
        if rng <= 0:
            continue

        # Body filter at i
        body = cl5[i] - op5[i]
        if abs(body) / rng < body_min:
            continue

        # MACD cross detection
        bull_cross = (macd[i-1] <= sig[i-1]) and (macd[i] > sig[i])
        bear_cross = (macd[i-1] >= sig[i-1]) and (macd[i] < sig[i])
        if not (bull_cross or bear_cross):
            continue

        # Body direction agreement
        if bull_cross and body <= 0:
            continue
        if bear_cross and body >= 0:
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

        # Trend filter (1h SMA200 only, single filter)
        sma_up = sma_long_15m[bidx]

        long_setup = bull_cross and sma_up
        short_setup = bear_cross and (not sma_up)

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
    print("M3-R41 — MACD Cross + 1h SMA200 Minimal (5m → 15m exit) LOCKED OOS")
    print("=" * 100)
    print(f"Locked params: {LOCKED}")
    print(f"Friction: {FRICTION}")
    print(f"Pre-reg: claudedocs/m3_round41_macd_minimal_prereg.md (commit 4da8d38)\n")

    df_5m, df_15m, valid_15m = prepare_5m_15m_data()
    n_5m = len(df_5m)
    n_15m = len(df_15m)
    days = n_15m / 96
    print(f"5m bars: {n_5m:,} | 15m bars: {n_15m:,} | days: {days:.0f}\n")

    sigs_full = entry_macd_cross_5m(df_5m, df_15m, valid_15m)
    sig_per_day = len(sigs_full) / days
    print(f"Full-dataset signals: {len(sigs_full)} ({sig_per_day:.3f}/day)")

    if sig_per_day < VACUITY_FLOOR_PER_DAY:
        print(f"\n  ⚠️  VACUITY GATE FAIL — {sig_per_day:.3f}/day")
        out = {'date': datetime.now(timezone.utc).isoformat(), 'pre_reg_commit': '4da8d38',
               'locked_params': LOCKED, 'friction': FRICTION,
               'full_signals_count': len(sigs_full), 'signals_per_day': sig_per_day,
               'vacuity_gate_pass': False, 'verdict': 'INCONCLUSIVE'}
        p = ROOT / 'results' / f'm3_r41_macd_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
        print(f"\nSaved: {p}")
        return

    print(f"  ✓ Vacuity gate PASS\n")

    ts15 = pd.to_datetime(df_15m['timestamp'])
    ts5 = pd.to_datetime(df_5m['timestamp'])

    # TEST 1
    print("=" * 100)
    print("TEST 1 — WF 5-fold expanding")
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
        sigs_f = entry_macd_cross_5m(df_5m_f, df_15m_f, v_f)
        trades = run_bt_c1_production(df_15m_f, sigs_f, friction=FRICTION)
        s_f = trade_summary(trades)
        if s_f is None:
            wf_results.append({'fold': fold_i+1, 'n': 0, 'daily': None})
            print(f"  fold {fold_i+1}: n=0")
            continue
        wf_results.append({'fold': fold_i+1, 'n': s_f['n'], 'daily': s_f['daily_net'],
                            'wr': s_f['wr_pct'], 'rr': s_f['rr'], 'avg_g': s_f['avg_gross']})
        print(f"  fold {fold_i+1}: n={s_f['n']:>4} daily={s_f['daily_net']:>+.4f}% WR={s_f['wr_pct']:>5.1f}% RR={s_f['rr']:>5.2f} avg_g={s_f['avg_gross']:>+.4f}%")
    wf_pos = sum(1 for r in wf_results if r['daily'] is not None and r['daily'] > 0)
    wf_pass = wf_pos >= 3
    print(f"\n  WF positive folds: {wf_pos}/5 → {'PASS' if wf_pass else 'FAIL'}")

    # TEST 2
    print(f"\n{'='*100}\nTEST 2 — Bootstrap 1000 × 3-day\n{'='*100}")
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
        sigs_w = entry_macd_cross_5m(df_5m_w, df_15m_w, v_w)
        trades = run_bt_c1_production(df_15m_w, sigs_w, friction=FRICTION)
        cand_pnls.append(sum(t['net_pct'] for t in trades) if trades else 0)
    arr = np.array(cand_pnls)
    bs_mean = float(arr.mean())
    bs_pos_rate = float((arr > 0).mean())
    bs_p5 = float(np.percentile(arr, 5))
    bs_active = float((arr != 0).mean())
    bs_pass = bs_pos_rate >= 0.5
    print(f"  mean={bs_mean:+.4f}% pos_rate={bs_pos_rate:.4f} active={bs_active:.4f} p5={bs_p5:+.4f}%")
    print(f"  Pos rate ≥ 0.50: {'PASS' if bs_pass else 'FAIL'}")

    # TEST 3
    print(f"\n{'='*100}\nTEST 3 — Train/Test 60/40\n{'='*100}")
    train_end = int(n_15m * 0.6)
    df_15m_tr = df_15m.iloc[:train_end].reset_index(drop=True)
    df_15m_te = df_15m.iloc[train_end:].reset_index(drop=True)
    v_tr = valid_15m[:train_end]; v_te = valid_15m[train_end:]
    t_tr_end = ts15.iloc[train_end - 1]
    mask5_tr = ts5 <= t_tr_end + pd.Timedelta(minutes=14)
    mask5_te = ts5 > t_tr_end + pd.Timedelta(minutes=14)
    df_5m_tr = df_5m.loc[mask5_tr].reset_index(drop=True)
    df_5m_te = df_5m.loc[mask5_te].reset_index(drop=True)
    sigs_tr = entry_macd_cross_5m(df_5m_tr, df_15m_tr, v_tr)
    sigs_te = entry_macd_cross_5m(df_5m_te, df_15m_te, v_te)
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
        print(f"  train: n={s_tr['n']:>4} daily={s_tr['daily_net']:>+.4f}% WR={s_tr['wr_pct']:>5.1f}% RR={s_tr['rr']:>5.2f} avg_g={s_tr['avg_gross']:>+.4f}%")
        print(f"  test:  n={s_te['n']:>4} daily={s_te['daily_net']:>+.4f}% WR={s_te['wr_pct']:>5.1f}% RR={s_te['rr']:>5.2f} avg_g={s_te['avg_gross']:>+.4f}%")
        tt_pass = (s_tr['daily_net'] > 0) and (s_te['daily_net'] > 0)
    print(f"  Train/Test both > 0: {'PASS' if tt_pass else 'FAIL'}")

    print(f"\n{'='*100}\nM3-R41 FINAL VERDICT\n{'='*100}")
    print(f"  Vacuity gate: PASS ({sig_per_day:.3f}/day)")
    print(f"  Test 1 (WF ≥3/5):                       {'PASS' if wf_pass else 'FAIL'}  ({wf_pos}/5)")
    print(f"  Test 2 (Bootstrap ≥50%):                {'PASS' if bs_pass else 'FAIL'}  ({bs_pos_rate:.4f})")
    print(f"  Test 3 (Train+Test sign-agree):          {'PASS' if tt_pass else 'FAIL'}")
    all_pass = wf_pass and bs_pass and tt_pass
    print(f"  OVERALL: {'ALL 3 PASS — call advisor' if all_pass else 'FAIL — 8th OOS negative'}")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '4da8d38',
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
    p = ROOT / 'results' / f'm3_r41_macd_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
