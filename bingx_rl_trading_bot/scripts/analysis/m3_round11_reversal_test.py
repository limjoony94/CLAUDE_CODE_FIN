"""M3-R11 — User reversal hypothesis test.

Q: 'Random에 지면 반대로 하면 random에 이긴다 아님?'
A: Only if train-test correlation is significantly NEGATIVE.

Test plan:
1. Re-run α/ι grid (1575 configs) saving ALL train + test results (no filter)
2. Compute Pearson correlation between train_daily and test_daily
3. Pick BOTTOM-10 train combos (worst train), check their test daily
4. If their test_daily > 0 consistently, reversal works (user's hypothesis correct)
5. Also pick BOTTOM-10 forward train, REVERSE direction, run on test
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_critique_pipeline import (prepare_all_data, run_bt_with_spec, trade_summary,
                                    rolling_pctile)
from m3_round2_critique import prepare_data_with_eth_break
from m3_round10_multidim_grid import (make_alpha_entry_param, make_iota_entry_param,
                                        make_fixed_exit, run_combo)


def make_alpha_entry_REVERSED(eth_thresh, btc_lag_thresh, atr_pctile):
    """α with LONG/SHORT directions FLIPPED."""
    def entry_fn(df, h1, h4, valid, params=None):
        n = len(df)
        btc_ret = df['btc_return'].values
        eth_ret = df['eth_return'].values
        atr = df['atr14'].values
        if atr_pctile == 70:
            atr_pctile_col = df['atr_pctile_70_200'].values
        else:
            atr_pctile_col = rolling_pctile(atr, 200, atr_pctile)
        sigs = []
        for i in range(2, n):
            if not valid[i]: continue
            if any(pd.isna(x) for x in (btc_ret[i - 1], eth_ret[i - 1], atr[i], atr_pctile_col[i])):
                continue
            if not (atr[i] > atr_pctile_col[i]): continue
            eth_up = eth_ret[i - 1] > eth_thresh
            btc_lag_up = btc_ret[i - 1] < btc_lag_thresh
            eth_down = eth_ret[i - 1] < -eth_thresh
            btc_lag_down = btc_ret[i - 1] > -btc_lag_thresh
            # FLIPPED: was LONG → SHORT, was SHORT → LONG
            if eth_up and btc_lag_up and h1[i] and h4[i]:
                sigs.append((i, 'SHORT'))  # ← reversed
            elif eth_down and btc_lag_down and (not h1[i]) and (not h4[i]):
                sigs.append((i, 'LONG'))  # ← reversed
        return sigs
    return entry_fn


def main():
    print("Loading data...")
    df, h1, h4, base_valid, eth_valid_ext, funding_valid = prepare_data_with_eth_break(lookback=24)
    n_total = len(df)
    train_end = int(n_total * 0.6)
    df_tr = df.iloc[:train_end].reset_index(drop=True)
    df_te = df.iloc[train_end:].reset_index(drop=True)
    h1_tr, h1_te = h1[:train_end], h1[train_end:]
    h4_tr, h4_te = h4[:train_end], h4[train_end:]
    valid_tr = eth_valid_ext[:train_end]
    valid_te = eth_valid_ext[train_end:]
    friction = 0.04

    # Run α grid saving ALL train + test (no filter)
    print(f"\nα grid 900 combos (no train filter)...")
    eth_thresh_grid = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
    btc_lag_grid = (0.00, 0.05, 0.10, 0.15, 0.20)
    atr_pctile_grid = (50, 60, 70, 80, 90)
    N_grid = (2, 4, 6, 8, 12, 16)

    all_results = []
    counter = 0
    for et, bl, ap, N in product(eth_thresh_grid, btc_lag_grid, atr_pctile_grid, N_grid):
        counter += 1
        entry_fn = make_alpha_entry_param(et, bl, ap)
        s_tr = run_combo(df_tr, h1_tr, h4_tr, valid_tr, entry_fn, N, friction)
        s_te = run_combo(df_te, h1_te, h4_te, valid_te, entry_fn, N, friction)
        if s_tr is None or s_te is None: continue
        if s_tr['n'] < 30 or s_te['n'] < 30: continue  # min sample
        all_results.append({
            'family': 'α', 'eth_thresh': et, 'btc_lag': bl, 'atr_pctile': ap, 'N': N,
            'train_daily': s_tr['daily_net'], 'train_n': s_tr['n'],
            'test_daily': s_te['daily_net'], 'test_n': s_te['n'],
        })
    print(f"  α: {counter} searched, {len(all_results)} with both train+test n≥30")

    # Pearson correlation across all combos
    train_arr = np.array([r['train_daily'] for r in all_results])
    test_arr = np.array([r['test_daily'] for r in all_results])
    if len(train_arr) > 2:
        corr = np.corrcoef(train_arr, test_arr)[0, 1]
        print(f"\n  Pearson correlation train_daily vs test_daily: {corr:+.4f}")

    # Sort by train_daily
    sorted_by_train = sorted(all_results, key=lambda r: r['train_daily'])

    print(f"\n{'=' * 80}\nBOTTOM-10 train (worst train daily)\n{'=' * 80}")
    print(f"{'eth':>4} {'btc':>5} {'atr':>4} {'N':>3} | {'train':>10} {'test':>10}")
    bottom_10 = sorted_by_train[:10]
    for r in bottom_10:
        print(f"{r['eth_thresh']:>4} {r['btc_lag']:>5} {r['atr_pctile']:>4} {r['N']:>3} | {r['train_daily']:>+9.4f}% {r['test_daily']:>+9.4f}%")
    bot_test_pos = sum(1 for r in bottom_10 if r['test_daily'] > 0)
    print(f"\n  Bottom-10 train, test_daily > 0 count: {bot_test_pos}/10")

    print(f"\n{'=' * 80}\nTOP-10 train (best train daily) [for reference]\n{'=' * 80}")
    top_10 = sorted_by_train[-10:][::-1]
    for r in top_10:
        print(f"{r['eth_thresh']:>4} {r['btc_lag']:>5} {r['atr_pctile']:>4} {r['N']:>3} | {r['train_daily']:>+9.4f}% {r['test_daily']:>+9.4f}%")
    top_test_pos = sum(1 for r in top_10 if r['test_daily'] > 0)
    print(f"\n  Top-10 train, test_daily > 0 count: {top_test_pos}/10")

    # User hypothesis check: REVERSE direction on bottom-10 train combos
    # If reversed gives positive test, user's intuition correct
    print(f"\n{'=' * 80}\nUSER HYPOTHESIS TEST: Reverse direction on bottom-10 train\n{'=' * 80}")
    print("(Bottom-10 train has worst train_daily. Reverse direction → expect train_daily flips sign)")
    print(f"{'eth':>4} {'btc':>5} {'atr':>4} {'N':>3} | {'orig_train':>10} {'rev_train':>10} {'orig_test':>10} {'rev_test':>10}")
    rev_train_pos = 0; rev_test_pos = 0
    rev_results = []
    for r in bottom_10:
        rev_fn = make_alpha_entry_REVERSED(r['eth_thresh'], r['btc_lag'], r['atr_pctile'])
        s_rev_tr = run_combo(df_tr, h1_tr, h4_tr, valid_tr, rev_fn, r['N'], friction)
        s_rev_te = run_combo(df_te, h1_te, h4_te, valid_te, rev_fn, r['N'], friction)
        rev_train = s_rev_tr['daily_net'] if s_rev_tr else None
        rev_test = s_rev_te['daily_net'] if s_rev_te else None
        if rev_train is not None and rev_train > 0: rev_train_pos += 1
        if rev_test is not None and rev_test > 0: rev_test_pos += 1
        rev_results.append({
            **r, 'rev_train_daily': rev_train, 'rev_test_daily': rev_test,
        })
        print(f"{r['eth_thresh']:>4} {r['btc_lag']:>5} {r['atr_pctile']:>4} {r['N']:>3} | "
              f"{r['train_daily']:>+9.4f}% {rev_train:>+9.4f}% {r['test_daily']:>+9.4f}% {rev_test:>+9.4f}%"
              if rev_train is not None and rev_test is not None
              else f"{r['eth_thresh']:>4} ... no trades")

    print(f"\n  Reversed combos: train_daily>0 in {rev_train_pos}/10 (sanity: should be near 10)")
    print(f"  Reversed combos: test_daily>0 in {rev_test_pos}/10 (USER HYPOTHESIS: should be ≥7)")

    # Verdict
    print(f"\n{'=' * 80}\nVERDICT\n{'=' * 80}")
    if corr < -0.3:
        print(f"  Pearson corr {corr:+.3f} < -0.3 → STRONG NEGATIVE — reversal hypothesis HOLDS")
    elif corr > 0.3:
        print(f"  Pearson corr {corr:+.3f} > 0.3 → POSITIVE — reversal would NOT help (real signal exists, my filter too loose)")
    else:
        print(f"  Pearson corr {corr:+.3f} ≈ 0 → train-test independent NOISE — reversal doesn't help systematically")

    if rev_test_pos >= 7:
        print(f"  Reversed bottom-10 → {rev_test_pos}/10 OOS positive → REVERSAL WORKS for selected subset")
    elif rev_test_pos >= 5:
        print(f"  Reversed bottom-10 → {rev_test_pos}/10 — moderate, possibly chance")
    else:
        print(f"  Reversed bottom-10 → {rev_test_pos}/10 — reversal does NOT systematically help")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'q': 'User: Random에 지면 반대로 하면 random에 이긴다 아님?',
           'pearson_corr_train_test': float(corr) if len(train_arr) > 2 else None,
           'all_combos_n': len(all_results),
           'bottom_10_train': bottom_10,
           'bottom_10_train_test_positive_count': bot_test_pos,
           'top_10_train': top_10,
           'top_10_train_test_positive_count': top_test_pos,
           'reversed_bottom_10': rev_results,
           'reversed_train_pos': rev_train_pos,
           'reversed_test_pos': rev_test_pos}
    p = ROOT / 'results' / f'm3_r11_reversal_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
