"""M3-R9c — α N=4 fixed exit OOS verification (single test, pre-registered).

Pre-reg: claudedocs/m3_round9c_alpha_N4_prereg.md
Tests: WF 5-fold, 3-way split, bootstrap 200, friction sensitivity, robustness check.
ALL must pass — any FAIL → drop claim.
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_critique_pipeline import (prepare_all_data, run_bt_with_spec, trade_summary,
                                    entry_alpha)


# Pre-registered fixed N=4 exit params
ALPHA_N4_EXIT = {
    'use_sl': False, 'use_trail': False,
    'sl_atr_mult': 0.0, 'trail_k': 0.0,
    'emergency_pct': 1.5,
    'timeout_bars': 4,
    'min_bars_between': 2,
}

ALPHA_PARAMS = {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'atr_pctile': 70.0}


def get_spec():
    return {
        'name': 'α N=4 fixed exit',
        'entry_fn': entry_alpha,
        'parameters': ALPHA_PARAMS,
        'direction_by_trend': True,
        'exit_params': ALPHA_N4_EXIT,
    }


def main():
    print("Loading data...")
    df, h1, h4, base_valid, eth_valid, funding_valid = prepare_all_data()
    spec = get_spec()
    valid = eth_valid

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'spec': 'α entry + N=4 fixed timeout exit',
           'pre_reg': 'claudedocs/m3_round9c_alpha_N4_prereg.md',
           'pre_reg_friction': 0.04}

    # Test 0: Full-sample sanity check
    print("\n=== Test 0: Full sample @ friction 0.04 ===")
    trades_full = run_bt_with_spec(df, h1, h4, valid, spec, friction=0.04)
    s = trade_summary(trades_full, friction=0.04)
    print(f"  full: n={s['n']} daily={s['daily_net']:+.4f}% WR={s['wr_pct']:.1f}% RR={s['rr']:.2f}")
    out['full_sample'] = {'n': s['n'], 'daily': s['daily_net'], 'wr': s['wr_pct'], 'rr': s['rr'],
                            'sum_gross': s['sum_gross'], 'sum_net': s['sum_net']}

    # Test 1: WF 5-fold expanding
    print("\n=== Test 1: WF 5-fold @ friction 0.04 ===")
    n = len(df)
    fold_size = n // 6
    wf = []
    for fold_i in range(5):
        train_end = (fold_i + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        df_t = df.iloc[test_start:test_end].reset_index(drop=True)
        h1_t = h1[test_start:test_end]; h4_t = h4[test_start:test_end]; v_t = valid[test_start:test_end]
        trades = run_bt_with_spec(df_t, h1_t, h4_t, v_t, spec, friction=0.04)
        ss = trade_summary(trades, friction=0.04) if trades else None
        wf.append({'fold': fold_i + 1,
                    'daily': ss['daily_net'] if ss else None,
                    'n': ss['n'] if ss else 0,
                    'wr': ss['wr_pct'] if ss else None,
                    'rr': ss['rr'] if ss else None})
        print(f"  fold {fold_i+1}: daily={ss['daily_net']:+.4f}% n={ss['n']} WR={ss['wr_pct']:.1f}% RR={ss['rr']:.2f}" if ss else f"  fold {fold_i+1}: no trades")
    wf_pos = sum(1 for r in wf if r['daily'] is not None and r['daily'] > 0)
    wf_pass = wf_pos >= 3
    print(f"  → WF positive folds: {wf_pos}/5 [{'PASS' if wf_pass else 'FAIL'}]")
    out['wf'] = {'folds': wf, 'positive_count': wf_pos, 'pass': wf_pass}

    # Test 2: 3-way split
    print("\n=== Test 2: 3-way split (train/val/test) @ friction 0.04 ===")
    third = n // 3
    splits = {}
    for label, (s_st, s_en) in [('train', (0, third)), ('val', (third, 2*third)), ('test', (2*third, n))]:
        df_s = df.iloc[s_st:s_en].reset_index(drop=True)
        h1_s = h1[s_st:s_en]; h4_s = h4[s_st:s_en]; v_s = valid[s_st:s_en]
        trades = run_bt_with_spec(df_s, h1_s, h4_s, v_s, spec, friction=0.04)
        ss = trade_summary(trades, friction=0.04) if trades else None
        splits[label] = {'daily': ss['daily_net'] if ss else None,
                          'n': ss['n'] if ss else 0,
                          'wr': ss['wr_pct'] if ss else None,
                          'rr': ss['rr'] if ss else None}
        print(f"  {label}: daily={ss['daily_net']:+.4f}% n={ss['n']} WR={ss['wr_pct']:.1f}% RR={ss['rr']:.2f}" if ss else f"  {label}: no trades")
    test_daily = splits['test']['daily']
    test_pass = test_daily is not None and test_daily > 0
    print(f"  → test split positive: {test_pass} [{'PASS' if test_pass else 'FAIL'}]")
    out['three_way'] = {'splits': splits, 'test_positive': test_pass, 'pass': test_pass}

    # Test 3: Bootstrap 200 × 3-day
    print("\n=== Test 3: 3-day bootstrap × 200 windows @ friction 0.04 ===")
    random.seed(42)
    bars_per_3day = 3 * 24 * 4
    max_start = n - bars_per_3day - 1
    bs_pnls = []
    if max_start > 0:
        starts = random.sample(range(max_start), min(200, max_start))
        for st in starts:
            en = st + bars_per_3day
            df_w = df.iloc[st:en].reset_index(drop=True)
            h1_w = h1[st:en]; h4_w = h4[st:en]; v_w = valid[st:en]
            trades = run_bt_with_spec(df_w, h1_w, h4_w, v_w, spec, friction=0.04)
            cand_pnl = sum(t['net_pct'] for t in trades) if trades else 0
            bs_pnls.append(cand_pnl)
    bs_mean = sum(bs_pnls) / len(bs_pnls) if bs_pnls else 0
    bs_pos = sum(1 for p in bs_pnls if p > 0) / len(bs_pnls) if bs_pnls else 0
    bs_pass = bs_mean > 0 and bs_pos >= 0.5
    print(f"  bootstrap: n_windows={len(bs_pnls)} mean={bs_mean:+.4f}% pos_rate={bs_pos:.4f} [{'PASS' if bs_pass else 'FAIL'}]")
    out['bootstrap'] = {'n': len(bs_pnls), 'mean': bs_mean, 'pos_rate': bs_pos, 'pass': bs_pass}

    # Test 4: Friction sensitivity (0.02, 0.04, 0.06 must all positive daily)
    print("\n=== Test 4: Friction sensitivity 0.02 / 0.04 / 0.06 ===")
    fric_results = {}
    for f in (0.02, 0.04, 0.06):
        trades = run_bt_with_spec(df, h1, h4, valid, spec, friction=f)
        ss = trade_summary(trades, friction=f) if trades else None
        fric_results[f] = {'daily': ss['daily_net'] if ss else None}
        print(f"  friction {f:.2f}: daily={ss['daily_net']:+.4f}%" if ss else f"  friction {f:.2f}: no trades")
    fric_all_pos = all(r['daily'] is not None and r['daily'] > 0 for r in fric_results.values())
    print(f"  → all 3 frictions positive: {fric_all_pos} [{'PASS' if fric_all_pos else 'FAIL'}]")
    out['friction_sensitivity'] = {'results': fric_results, 'all_positive': fric_all_pos, 'pass': fric_all_pos}

    # Test 5: Robustness check (WR ≥40, RR ≥1.0, n ≥ 150)
    print("\n=== Test 5: Robustness (full sample @ f=0.04) ===")
    rob_pass = (s['wr_pct'] >= 40 and s['rr'] >= 1.0 and s['n'] >= 150)
    print(f"  WR={s['wr_pct']:.1f} (≥40), RR={s['rr']:.2f} (≥1.0), n={s['n']} (≥150) [{'PASS' if rob_pass else 'FAIL'}]")
    out['robustness'] = {'wr': s['wr_pct'], 'rr': s['rr'], 'n': s['n'], 'pass': rob_pass}

    # OVERALL
    all_pass = wf_pass and test_pass and bs_pass and fric_all_pos and rob_pass
    print("\n" + "=" * 80)
    print(f"M3-R9c — α N=4 OOS verdict: {'ALL PASS' if all_pass else 'FAIL'}")
    print("=" * 80)
    print(f"  Test 1 WF 5-fold (≥3 pos):       {'PASS' if wf_pass else 'FAIL'} ({wf_pos}/5)")
    print(f"  Test 2 3-way test split:         {'PASS' if test_pass else 'FAIL'}")
    print(f"  Test 3 Bootstrap mean+pos rate:  {'PASS' if bs_pass else 'FAIL'}")
    print(f"  Test 4 Friction sens (3/3 pos):  {'PASS' if fric_all_pos else 'FAIL'}")
    print(f"  Test 5 Robustness:               {'PASS' if rob_pass else 'FAIL'}")
    out['overall_pass'] = all_pass

    if all_pass:
        print("\n✓ Pre-reg MET. Real OOS-stable edge under adjusted criterion (friction 0.04).")
        print("  Next step: PDCA Plan (production path with maker-rebate infra). User decision.")
    else:
        print("\n✗ One or more tests failed. Drop claim per fix-impulse anti-pattern guard.")
        print("  Next step: Accept. Move to user options A / B / E.")

    p = ROOT / 'results' / f'm3_r9c_alpha_N4_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
