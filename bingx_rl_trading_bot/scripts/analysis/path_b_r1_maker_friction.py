"""Path B R1-maker — XS Momentum at maker friction 0.04% (5-min advisor test).

Single scoped re-evaluation: only friction param changed from R1 (0.07 → 0.04).
NEW gate Test 4: avg_daily_net ≥ 0.05% (1× leverage, ~18% annualized).

Pre-reg: claudedocs/path_b_r1_maker_friction_prereg.md (commit a12559f)
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

import path_b_r1_xs_momentum as r1mod

# Override LOCKED friction (only change)
r1mod.LOCKED['friction_per_transaction'] = 0.04
NEW_DAILY_NET_GATE = 0.05  # %


def main():
    print("=" * 100)
    print("Path B R1-MAKER — XS Momentum at maker friction (0.04% per transaction)")
    print("=" * 100)
    print(f"Locked params: {r1mod.LOCKED}")
    print(f"NEW Test 4 gate: avg_daily_net ≥ {NEW_DAILY_NET_GATE}% (1× leverage)")
    print(f"Pre-reg: claudedocs/path_b_r1_maker_friction_prereg.md (commit a12559f)\n")

    prices = r1mod.load_pivot()
    n_dates = len(prices)
    print(f'Daily prices: {n_dates} days × {prices.shape[1]} coins\n')

    look = r1mod.LOCKED['lookback_days']
    trail_ret = (prices / prices.shift(look) - 1) * 100
    median_disp = trail_ret.std(axis=1).median()
    print(f"Dispersion gate: {median_disp:.2f}% (floor {r1mod.DISPERSION_FLOOR}%)")
    if median_disp < r1mod.DISPERSION_FLOOR:
        print("  ⚠️ DISPERSION FAIL")
        return
    print('  ✓ Dispersion PASS\n')

    # Full sample
    bt_full = r1mod.run_xs_momentum(prices, friction_pct=0.04)
    sumf = r1mod.summarize(bt_full)
    print("=== Full-sample BT (maker friction 0.04%) ===")
    for k, v in sumf.items():
        print(f"  {k}: {v}")
    daily_net_full = sumf['avg_daily_pct']
    edge_above_friction = sumf['avg_weekly_gross_pct'] - sumf['avg_weekly_friction_pct']
    print(f"\n  Edge above friction: {edge_above_friction:+.4f}%/wk")
    print(f"  ★ avg_daily_net (1×): {daily_net_full:+.4f}%/day")
    print(f"  ★ NEW Test 4 gate: ≥ {NEW_DAILY_NET_GATE}%/day → {'PASS' if daily_net_full >= NEW_DAILY_NET_GATE else 'FAIL'}")

    # Test 1: WF
    print(f"\n{'='*100}\nTEST 1 — WF 5-fold expanding (maker friction)\n{'='*100}")
    print(f"  {'fold':>4} {'days':>5} {'avg_wk_gross':>13} {'avg_wk_fric':>12} {'avg_wk_net':>11} {'avg_d_net':>10} {'sharpe':>8} {'max_dd':>9}")
    fold_size = n_dates // 6
    wf_results = []
    daily_nets_per_fold = []
    for fold_i in range(5):
        te_s = (fold_i + 1) * fold_size
        te_e = min(te_s + fold_size, n_dates)
        sub = prices.iloc[te_s:te_e]
        bt = r1mod.run_xs_momentum(sub, friction_pct=0.04)
        s = r1mod.summarize(bt)
        if s.get('n_days', 0) == 0:
            wf_results.append({'fold': fold_i+1, 'avg_weekly_net_pct': None})
            continue
        wf_results.append({'fold': fold_i+1, **s})
        daily_nets_per_fold.append(s['avg_daily_pct'])
        gate_wk = '✓' if s['avg_weekly_net_pct'] > 0 else '✗'
        gate_d = '✓' if s['avg_daily_pct'] >= NEW_DAILY_NET_GATE else '✗'
        print(f"  {fold_i+1:>4} {s['n_days']:>5} {s['avg_weekly_gross_pct']:>+12.4f}% {s['avg_weekly_friction_pct']:>+11.4f}% {s['avg_weekly_net_pct']:>+10.4f}% {gate_wk} {s['avg_daily_pct']:>+8.4f}% {gate_d} {s['sharpe_annualized']:>+7.2f} {s['max_dd_pct']:>+8.2f}%")
    wf_pos = sum(1 for r in wf_results if r.get('avg_weekly_net_pct') is not None and r['avg_weekly_net_pct'] > 0)
    wf_pass = wf_pos >= 3
    print(f"  WF positive folds: {wf_pos}/5 → {'PASS' if wf_pass else 'FAIL'}")
    folds_above_daily_gate = sum(1 for d in daily_nets_per_fold if d >= NEW_DAILY_NET_GATE)
    print(f"  Folds with avg_daily ≥ {NEW_DAILY_NET_GATE}%: {folds_above_daily_gate}/5")

    # Test 2 Bootstrap
    print(f"\n{'='*100}\nTEST 2 — Bootstrap 1000 × 30-day NET-return windows\n{'='*100}")
    net_returns = bt_full['net_pct'].values
    n_returns = len(net_returns)
    win = 30
    max_start = n_returns - win - 1
    random.seed(42)
    starts = random.sample(range(max(1, max_start)), min(1000, max(1, max_start)))
    pnls = []
    for st in starts:
        slice_ = net_returns[st:st + win]
        cum = float((1 + slice_ / 100).prod() - 1) * 100
        pnls.append(cum)
    arr = np.array(pnls)
    bs_mean = float(arr.mean())
    bs_pos_rate = float((arr > 0).mean())
    bs_p5 = float(np.percentile(arr, 5))
    bs_pass = bs_pos_rate >= 0.5
    print(f"  mean (30d cum): {bs_mean:+.4f}% pos_rate={bs_pos_rate:.4f} p5={bs_p5:+.4f}%")
    print(f"  Pos rate ≥ 0.50: {'PASS' if bs_pass else 'FAIL'}")

    # Test 3 Train/Test
    print(f"\n{'='*100}\nTEST 3 — Train/Test 60/40 (maker friction)\n{'='*100}")
    train_end = int(n_dates * 0.6)
    bt_tr = r1mod.run_xs_momentum(prices.iloc[:train_end], friction_pct=0.04)
    bt_te = r1mod.run_xs_momentum(prices.iloc[train_end:], friction_pct=0.04)
    s_tr = r1mod.summarize(bt_tr)
    s_te = r1mod.summarize(bt_te)
    print(f"  train: days={s_tr['n_days']} avg_d_net={s_tr['avg_daily_pct']:+.4f}% avg_wk_net={s_tr['avg_weekly_net_pct']:+.4f}% (gross={s_tr['avg_weekly_gross_pct']:+.4f}%, fric={s_tr['avg_weekly_friction_pct']:.4f}%) dd={s_tr['max_dd_pct']:+.2f}%")
    print(f"  test:  days={s_te['n_days']} avg_d_net={s_te['avg_daily_pct']:+.4f}% avg_wk_net={s_te['avg_weekly_net_pct']:+.4f}% (gross={s_te['avg_weekly_gross_pct']:+.4f}%, fric={s_te['avg_weekly_friction_pct']:.4f}%) dd={s_te['max_dd_pct']:+.2f}%")
    tt_pass = (s_tr['avg_weekly_net_pct'] > 0) and (s_te['avg_weekly_net_pct'] > 0)
    print(f"  Train+Test both > 0: {'PASS' if tt_pass else 'FAIL'}")

    # Test 4 NEW
    print(f"\n{'='*100}\nTEST 4 (NEW) — avg_daily_net ≥ {NEW_DAILY_NET_GATE}% on FULL SAMPLE\n{'='*100}")
    test4_pass = daily_net_full >= NEW_DAILY_NET_GATE
    print(f"  Full sample avg_daily_net: {daily_net_full:+.4f}%/day")
    print(f"  Gate: ≥ {NEW_DAILY_NET_GATE}%/day → {'PASS' if test4_pass else 'FAIL'}")
    print(f"  Annualized equivalent: {daily_net_full * 365:+.2f}%/yr")

    print(f"\n{'='*100}\nPath B R1-maker FINAL VERDICT\n{'='*100}")
    print(f"  Dispersion:                      PASS ({median_disp:.2f}%)")
    print(f"  Test 1 (WF ≥3/5):                {'PASS' if wf_pass else 'FAIL'}  ({wf_pos}/5)")
    print(f"  Test 2 (Bootstrap ≥50%):         {'PASS' if bs_pass else 'FAIL'}  ({bs_pos_rate:.4f})")
    print(f"  Test 3 (Train+Test sign-agree):  {'PASS' if tt_pass else 'FAIL'}")
    print(f"  Test 4 (avg_daily ≥ 0.05%):      {'PASS' if test4_pass else 'FAIL'}  ({daily_net_full:+.4f}%/day)")
    all_pass = wf_pass and bs_pass and tt_pass and test4_pass
    print(f"  OVERALL: {'ALL 4 PASS — paper trade candidate' if all_pass else 'FAIL — escalate to advisor'}")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': 'a12559f',
        'locked_params': r1mod.LOCKED,
        'friction_change': '0.07 → 0.04',
        'new_daily_net_gate': NEW_DAILY_NET_GATE,
        'dispersion_median_pct': float(median_disp),
        'full_sample': sumf,
        'avg_daily_net_pct_full': daily_net_full,
        'test_1_wf': {'folds': wf_results, 'pos_count': wf_pos, 'pass': wf_pass,
                       'folds_above_daily_gate': folds_above_daily_gate},
        'test_2_bootstrap_30d': {'mean_pct': bs_mean, 'pos_rate': bs_pos_rate, 'p5_pct': bs_p5, 'pass': bs_pass},
        'test_3_train_test': {'train': s_tr, 'test': s_te, 'pass': tt_pass},
        'test_4_daily_net': {'value_pct': daily_net_full, 'gate_pct': NEW_DAILY_NET_GATE, 'pass': test4_pass},
        'all_pass': bool(all_pass),
    }
    p = ROOT / 'results' / f'path_b_r1_maker_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
