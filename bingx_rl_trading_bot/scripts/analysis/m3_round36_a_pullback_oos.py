"""M3-R36 — A pullback (15m) LOCKED OOS verification.

Pre-registered (see claudedocs/m3_round36_a_pullback_oos_prereg.md).
Locked params: ema_dist_pct=0.5, volume_mult=1.0
Friction: 0.07
Mechanism: entry_pullback_15m from m3_round35_15m_deep
Exit: run_bt_c1_production from m3_round30_c1_production_exact

ALL 3 tests required to pass:
  1. WF 5-fold expanding: ≥3/5 folds daily_net > 0
  2. Bootstrap 1000 × 3-day: pos_rate ≥ 50%
  3. Train→Test 60/40: both daily_net > 0 (sign-agree)

ANY one fail → candidate dropped permanently.
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_round35_15m_deep import entry_pullback_15m, add_15m_extras, trade_summary
from m3_round30_c1_production_exact import run_bt_c1_production
from m3_round29_c1_exact_revalidation import prepare_15m_data


# LOCKED PARAMS — DO NOT CHANGE
LOCKED_PARAMS = {'ema_dist_pct': 0.5, 'volume_mult': 1.0}
FRICTION = 0.07


def main():
    print("=" * 100)
    print("M3-R36 — A Pullback 15m LOCKED OOS Verification (pre-registered)")
    print("=" * 100)
    print(f"Locked params: {LOCKED_PARAMS}")
    print(f"Friction: {FRICTION}")
    print(f"Pre-reg: claudedocs/m3_round36_a_pullback_oos_prereg.md\n")

    df, valid = prepare_15m_data()
    df = add_15m_extras(df)
    valid = (valid & (~df['sma200_long'].isna()).values & (~df['htf4_long'].isna()).values
              & (~pd.isna(df['ema20_15m']).values) & (~pd.isna(df['ema50_15m']).values)
              & (~pd.isna(df['volume_sma20']).values) & (~pd.isna(df['rsi14_15m']).values))
    n_total = len(df)
    print(f"15m bars: {n_total:,} | days: {n_total/96:.0f}\n")

    # ----------------------------------------------------------------------
    # TEST 1: WF 5-fold EXPANDING window
    # ----------------------------------------------------------------------
    print("=" * 100)
    print("TEST 1 — WF 5-fold expanding (locked params, friction=0.07)")
    print("Pass criterion: ≥3/5 folds daily_net > 0")
    print("=" * 100)

    # Expanding window: each fold trains on [0, train_end] and tests on [train_end, test_end]
    # Total span split into 6 segments. Train uses [0, k*seg], test uses [k*seg, (k+1)*seg], k=1..5.
    # Since params are locked (no per-fold tuning), we just need to verify each test segment.
    fold_size = n_total // 6
    wf_results = []
    for fold_i in range(5):
        te_s = (fold_i + 1) * fold_size
        te_e = min(te_s + fold_size, n_total)
        df_f = df.iloc[te_s:te_e].reset_index(drop=True)
        v_f = valid[te_s:te_e]
        sigs_f = entry_pullback_15m(df_f, v_f, params=LOCKED_PARAMS)
        trades = run_bt_c1_production(df_f, sigs_f, friction=FRICTION)
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
    # TEST 2: 3-day Random Window Bootstrap
    # ----------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("TEST 2 — Bootstrap 1000 × 3-day windows (locked params, friction=0.07)")
    print("Pass criterion: pos_rate ≥ 50%")
    print("=" * 100)

    bars_per_3day = 3 * 96
    max_start = n_total - bars_per_3day - 1
    random.seed(42)
    starts = random.sample(range(max_start), min(1000, max_start))
    cand_pnls = []
    for st in starts:
        en = st + bars_per_3day
        df_w = df.iloc[st:en].reset_index(drop=True)
        v_w = valid[st:en]
        sigs_w = entry_pullback_15m(df_w, v_w, params=LOCKED_PARAMS)
        trades = run_bt_c1_production(df_w, sigs_w, friction=FRICTION)
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
    # TEST 3: Train→Test 60/40 sign agreement
    # ----------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("TEST 3 — Train/Test 60/40 split (locked params, friction=0.07)")
    print("Pass criterion: both train and test daily_net > 0")
    print("=" * 100)

    train_end = int(n_total * 0.6)
    df_tr = df.iloc[:train_end].reset_index(drop=True)
    df_te = df.iloc[train_end:].reset_index(drop=True)
    v_tr = valid[:train_end]
    v_te = valid[train_end:]

    sigs_tr = entry_pullback_15m(df_tr, v_tr, params=LOCKED_PARAMS)
    sigs_te = entry_pullback_15m(df_te, v_te, params=LOCKED_PARAMS)
    trades_tr = run_bt_c1_production(df_tr, sigs_tr, friction=FRICTION)
    trades_te = run_bt_c1_production(df_te, sigs_te, friction=FRICTION)
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

    # ----------------------------------------------------------------------
    # FINAL VERDICT
    # ----------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("M3-R36 FINAL VERDICT (pre-registered, all 3 required)")
    print(f"{'='*100}")
    print(f"  Test 1 (WF 5-fold expanding ≥3/5):     {'PASS' if wf_pass else 'FAIL'}  ({wf_pos}/5)")
    print(f"  Test 2 (Bootstrap pos_rate ≥ 50%):    {'PASS' if bs_pass else 'FAIL'}  ({bs_pos_rate:.4f})")
    print(f"  Test 3 (Train+Test sign-agree):        {'PASS' if tt_pass else 'FAIL'}")
    all_pass = wf_pass and bs_pass and tt_pass
    print(f"\n  OVERALL: {'ALL PASS — candidate not falsified, deeper validation warranted' if all_pass else 'FAIL — A pullback breakthrough claim retracted'}")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg': 'claudedocs/m3_round36_a_pullback_oos_prereg.md',
        'locked_params': LOCKED_PARAMS,
        'friction': FRICTION,
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
    p = ROOT / 'results' / f'm3_r36_a_pullback_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
