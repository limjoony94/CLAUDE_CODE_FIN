"""M3-R10 — Multi-dim parameter grid search with train/test OOS verification.

Pre-reg: claudedocs/m3_round10_multidim_prereg.md
- Train: first 60% (~432 days)
- Test: last 40% (~288 days)
- Grid: α (900 combos) + ι (675 combos) = ~1575 multi-dim configs
- Top-10 from train, OOS test, random control, bootstrap on best
- 5-condition pre-registered PASS criteria
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


# ---------- Generic α-family entry function (param-driven) ----------

def make_alpha_entry_param(eth_thresh, btc_lag_thresh, atr_pctile):
    """Returns entry_fn for α with params injected. Recompute atr_pctile column if non-default."""
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
            if eth_up and btc_lag_up and h1[i] and h4[i]:
                sigs.append((i, 'LONG'))
            elif eth_down and btc_lag_down and (not h1[i]) and (not h4[i]):
                sigs.append((i, 'SHORT'))
        return sigs
    return entry_fn


def make_iota_entry_param(eth_thresh, btc_lag_thresh, atr_pctile, eth_break_lookback):
    """Returns entry_fn for ι with multi-dim params."""
    def entry_fn(df, h1, h4, valid, params=None):
        n = len(df)
        btc_ret = df['btc_return'].values
        eth_ret = df['eth_return'].values
        eth_close = df['eth_close'].values
        atr = df['atr14'].values
        if atr_pctile == 70:
            atr_pctile_col = df['atr_pctile_70_200'].values
        else:
            atr_pctile_col = rolling_pctile(atr, 200, atr_pctile)
        if eth_break_lookback == 24:
            eth_high_prev = df['eth_high_24_prev'].values
            eth_low_prev = df['eth_low_24_prev'].values
        else:
            eth_high_prev = pd.Series(eth_close).rolling(eth_break_lookback, min_periods=eth_break_lookback).max().shift(1).values
            eth_low_prev = pd.Series(eth_close).rolling(eth_break_lookback, min_periods=eth_break_lookback).min().shift(1).values
        sigs = []
        for i in range(2, n):
            if not valid[i]: continue
            if any(pd.isna(x) for x in (btc_ret[i - 1], eth_ret[i - 1], atr[i], atr_pctile_col[i],
                                          eth_high_prev[i], eth_low_prev[i])):
                continue
            if not (atr[i] > atr_pctile_col[i]): continue
            eth_up = eth_ret[i - 1] > eth_thresh
            btc_lag_up = btc_ret[i - 1] < btc_lag_thresh
            eth_break_up = eth_close[i] > eth_high_prev[i]
            eth_down = eth_ret[i - 1] < -eth_thresh
            btc_lag_down = btc_ret[i - 1] > -btc_lag_thresh
            eth_break_down = eth_close[i] < eth_low_prev[i]
            if eth_up and btc_lag_up and eth_break_up and h1[i] and h4[i]:
                sigs.append((i, 'LONG'))
            elif eth_down and btc_lag_down and eth_break_down and (not h1[i]) and (not h4[i]):
                sigs.append((i, 'SHORT'))
        return sigs
    return entry_fn


def make_random_entry_param(target_n_per_combo, seed):
    """Random entry generator matching combo's expected sample size — for control."""
    def entry_fn(df, h1, h4, valid, params=None):
        random.seed(seed)
        eligible_idx = np.where(valid & (h1 & h4 | (~h1) & (~h4)))[0]
        eligible_idx = eligible_idx[(eligible_idx > 2) & (eligible_idx < len(df) - 50)]
        if len(eligible_idx) == 0:
            return []
        target = min(target_n_per_combo, len(eligible_idx))
        sampled = sorted(random.sample(eligible_idx.tolist(), target))
        out = []
        for idx in sampled:
            if h1[idx] and h4[idx]:
                out.append((idx, 'LONG'))
            elif (not h1[idx]) and (not h4[idx]):
                out.append((idx, 'SHORT'))
        return out
    return entry_fn


def make_fixed_exit(N):
    return {
        'use_sl': False, 'use_trail': False,
        'sl_atr_mult': 0.0, 'trail_k': 0.0,
        'emergency_pct': 1.5,
        'timeout_bars': N,
        'min_bars_between': 2,
    }


def run_combo(df, h1, h4, valid, entry_fn, N, friction):
    spec = {
        'name': 'combo',
        'entry_fn': entry_fn,
        'parameters': {},
        'direction_by_trend': True,
        'exit_params': make_fixed_exit(N),
    }
    trades = run_bt_with_spec(df, h1, h4, valid, spec, friction=friction)
    if not trades:
        return None
    s = trade_summary(trades, friction=friction)
    return s


def main():
    print("Loading data + ETH break columns...")
    df, h1, h4, base_valid, eth_valid_ext, funding_valid = prepare_data_with_eth_break(lookback=24)
    n_total = len(df)
    train_end = int(n_total * 0.6)
    print(f"  bars: {n_total:,} | train_end: {train_end:,} (60%)")
    print(f"  train days: {train_end / 96:.0f}, test days: {(n_total - train_end) / 96:.0f}\n")

    df_tr = df.iloc[:train_end].reset_index(drop=True)
    df_te = df.iloc[train_end:].reset_index(drop=True)
    h1_tr, h1_te = h1[:train_end], h1[train_end:]
    h4_tr, h4_te = h4[:train_end], h4[train_end:]
    valid_tr = eth_valid_ext[:train_end]
    valid_te = eth_valid_ext[train_end:]

    friction = 0.04

    # ---------- α grid ----------
    print("=" * 80); print("α FAMILY GRID (6 × 5 × 5 × 6 = 900 combos)"); print("=" * 80)
    eth_thresh_grid = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
    btc_lag_grid = (0.00, 0.05, 0.10, 0.15, 0.20)
    atr_pctile_grid = (50, 60, 70, 80, 90)
    N_grid = (2, 4, 6, 8, 12, 16)

    alpha_results = []
    total_combos = len(eth_thresh_grid) * len(btc_lag_grid) * len(atr_pctile_grid) * len(N_grid)
    counter = 0
    for et, bl, ap, N in product(eth_thresh_grid, btc_lag_grid, atr_pctile_grid, N_grid):
        counter += 1
        entry_fn = make_alpha_entry_param(et, bl, ap)
        s_tr = run_combo(df_tr, h1_tr, h4_tr, valid_tr, entry_fn, N, friction)
        if s_tr is None or s_tr['n'] < 50: continue
        # Train pass criteria
        if not (s_tr['daily_net'] > 0 and s_tr['wr_pct'] >= 40 and s_tr['rr'] >= 1.0):
            continue
        alpha_results.append({
            'family': 'α', 'eth_thresh': et, 'btc_lag': bl, 'atr_pctile': ap, 'N': N,
            'train_daily': s_tr['daily_net'], 'train_n': s_tr['n'],
            'train_wr': s_tr['wr_pct'], 'train_rr': s_tr['rr'],
        })
        if counter % 100 == 0:
            print(f"  progress {counter}/{total_combos} | train_pass count: {len(alpha_results)}")
    print(f"  α: {counter} combos searched, {len(alpha_results)} train-pass\n")

    # ---------- ι grid ----------
    print("=" * 80); print("ι FAMILY GRID (3 × 3 × 3 × 5 × 5 = 675 combos)"); print("=" * 80)
    iota_eth_grid = (0.20, 0.30, 0.40)
    iota_btc_grid = (0.05, 0.10, 0.15)
    iota_atr_grid = (60, 70, 80)
    iota_lookback_grid = (12, 18, 24, 30, 36)
    iota_N_grid = (4, 6, 8, 12, 16)

    iota_results = []
    total_iota = len(iota_eth_grid) * len(iota_btc_grid) * len(iota_atr_grid) * len(iota_lookback_grid) * len(iota_N_grid)
    counter = 0
    for et, bl, ap, lb, N in product(iota_eth_grid, iota_btc_grid, iota_atr_grid, iota_lookback_grid, iota_N_grid):
        counter += 1
        entry_fn = make_iota_entry_param(et, bl, ap, lb)
        s_tr = run_combo(df_tr, h1_tr, h4_tr, valid_tr, entry_fn, N, friction)
        if s_tr is None or s_tr['n'] < 50: continue
        if not (s_tr['daily_net'] > 0 and s_tr['wr_pct'] >= 40 and s_tr['rr'] >= 1.0):
            continue
        iota_results.append({
            'family': 'ι', 'eth_thresh': et, 'btc_lag': bl, 'atr_pctile': ap,
            'eth_break_lookback': lb, 'N': N,
            'train_daily': s_tr['daily_net'], 'train_n': s_tr['n'],
            'train_wr': s_tr['wr_pct'], 'train_rr': s_tr['rr'],
        })
        if counter % 100 == 0:
            print(f"  progress {counter}/{total_iota} | train_pass count: {len(iota_results)}")
    print(f"  ι: {counter} combos searched, {len(iota_results)} train-pass\n")

    train_pass_total = len(alpha_results) + len(iota_results)
    print(f"TOTAL train-pass: {train_pass_total}")

    # ---------- Top-10 selection ----------
    all_results = alpha_results + iota_results
    all_results_sorted = sorted(all_results, key=lambda r: -r['train_daily'])
    top10 = all_results_sorted[:10]
    print(f"\n{'=' * 80}\nTOP-10 from train\n{'=' * 80}")
    for i, r in enumerate(top10, 1):
        print(f"  #{i:>2} {r['family']} eth={r['eth_thresh']} btc={r['btc_lag']} atr={r['atr_pctile']}"
              + (f" lb={r['eth_break_lookback']}" if 'eth_break_lookback' in r else "")
              + f" N={r['N']} | train_daily={r['train_daily']:+.4f}% n={r['train_n']} WR={r['train_wr']:.1f} RR={r['train_rr']:.2f}")

    # ---------- OOS test on top-10 ----------
    print(f"\n{'=' * 80}\nOOS TEST on top-10\n{'=' * 80}")
    oos_results = []
    for r in top10:
        if r['family'] == 'α':
            entry_fn = make_alpha_entry_param(r['eth_thresh'], r['btc_lag'], r['atr_pctile'])
        else:
            entry_fn = make_iota_entry_param(r['eth_thresh'], r['btc_lag'], r['atr_pctile'], r['eth_break_lookback'])
        s_te = run_combo(df_te, h1_te, h4_te, valid_te, entry_fn, r['N'], friction)
        oos_pass = s_te is not None and s_te['daily_net'] > 0 and s_te['n'] >= 30
        oos_results.append({**r,
                             'test_daily': s_te['daily_net'] if s_te else None,
                             'test_n': s_te['n'] if s_te else 0,
                             'test_wr': s_te['wr_pct'] if s_te else None,
                             'oos_pass': oos_pass})
        marker = '✓' if oos_pass else '✗'
        if s_te:
            print(f"  {marker} {r['family']} N={r['N']} | train={r['train_daily']:+.4f} | test={s_te['daily_net']:+.4f} n_te={s_te['n']} WR_te={s_te['wr_pct']:.1f}")
        else:
            print(f"  {marker} {r['family']} N={r['N']} | test: no trades")
    oos_survivors = sum(1 for r in oos_results if r['oos_pass'])
    print(f"\n  OOS survivors: {oos_survivors}/10")

    # ---------- Random control: same N grid on random entries ----------
    print(f"\n{'=' * 80}\nRANDOM BASELINE CONTROL\n{'=' * 80}")
    random_results = []
    n_random_combos = 30  # 3 seeds × 5 N × 2 sample sizes
    for seed in (42, 123, 456):
        for N in (4, 6, 8, 12, 16):
            for target_n in (200, 400):
                entry_fn = make_random_entry_param(target_n, seed)
                s_tr = run_combo(df_tr, h1_tr, h4_tr, valid_tr, entry_fn, N, friction)
                if s_tr is None: continue
                # Run same on test
                s_te = run_combo(df_te, h1_te, h4_te, valid_te, entry_fn, N, friction)
                if s_te is None: continue
                random_results.append({
                    'seed': seed, 'N': N, 'target_n': target_n,
                    'train_daily': s_tr['daily_net'], 'test_daily': s_te['daily_net'],
                    'oos_pass': s_te['daily_net'] > 0 and s_te['n'] >= 30,
                })
    random_survivors = sum(1 for r in random_results if r['oos_pass'])
    print(f"  Random control: {random_survivors}/{len(random_results)} OOS positive (expected ~50% if 0-mean)")

    # ---------- Bootstrap on best OOS combo ----------
    best = max([r for r in oos_results if r['oos_pass']], key=lambda r: r['test_daily'], default=None)
    bs_pos_rate = None
    if best:
        print(f"\n{'=' * 80}\nBOOTSTRAP on best OOS combo: {best['family']} N={best['N']}\n{'=' * 80}")
        if best['family'] == 'α':
            entry_fn = make_alpha_entry_param(best['eth_thresh'], best['btc_lag'], best['atr_pctile'])
        else:
            entry_fn = make_iota_entry_param(best['eth_thresh'], best['btc_lag'], best['atr_pctile'], best['eth_break_lookback'])
        random.seed(42)
        bars_per_3day = 3 * 24 * 4
        max_start = n_total - bars_per_3day - 1
        starts = random.sample(range(max_start), min(200, max_start))
        bs_pnls = []
        for st in starts:
            en = st + bars_per_3day
            df_w = df.iloc[st:en].reset_index(drop=True)
            h1_w = h1[st:en]; h4_w = h4[st:en]; v_w = eth_valid_ext[st:en]
            s_w = run_combo(df_w, h1_w, h4_w, v_w, entry_fn, best['N'], friction)
            bs_pnls.append(s_w['sum_net'] if s_w else 0)
        bs_pos_rate = sum(1 for p in bs_pnls if p > 0) / len(bs_pnls) if bs_pnls else 0
        bs_mean = sum(bs_pnls) / len(bs_pnls) if bs_pnls else 0
        print(f"  bootstrap: n={len(bs_pnls)} mean={bs_mean:+.4f}% pos_rate={bs_pos_rate:.4f}")

    # ---------- 5-condition pre-reg check ----------
    print(f"\n{'=' * 80}\nPRE-REG 5-CONDITION CHECK\n{'=' * 80}")
    cond1_pass = train_pass_total >= 5
    cond2_pass = oos_survivors >= 5
    cond3_pass = random_survivors <= 2 if random_results else None
    cond4_pass = bs_pos_rate is not None and bs_pos_rate >= 0.30
    cond5_pass = None
    if best:
        rel_diff = abs(best['test_daily'] - best['train_daily']) / max(abs(best['train_daily']), 1e-6)
        cond5_pass = rel_diff < 0.50
    all_pass = all([cond1_pass, cond2_pass,
                     cond3_pass if cond3_pass is not None else False,
                     cond4_pass if cond4_pass is not None else False,
                     cond5_pass if cond5_pass is not None else False])
    print(f"  (1) Train pass ≥5         : {cond1_pass} ({train_pass_total})")
    print(f"  (2) OOS top-10 ≥5 survive : {cond2_pass} ({oos_survivors}/10)")
    print(f"  (3) Random control ≤2     : {cond3_pass} ({random_survivors}/{len(random_results)})")
    print(f"  (4) Bootstrap pos_rate ≥0.30 : {cond4_pass} ({bs_pos_rate})")
    print(f"  (5) train↔test stability  : {cond5_pass}")
    print(f"\n  OVERALL: {'ALL PASS — real multi-dim sweet spot' if all_pass else 'FAIL — drop claim per pre-reg'}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'pre_reg': 'claudedocs/m3_round10_multidim_prereg.md',
           'train_pass_total': train_pass_total,
           'top10': top10,
           'oos_results': oos_results,
           'oos_survivors': oos_survivors,
           'random_control': random_results,
           'random_survivors': random_survivors,
           'bootstrap_best': {'best_combo': best, 'pos_rate': bs_pos_rate},
           'conditions': {
               'cond1_train_pass_5': cond1_pass,
               'cond2_oos_survivors_5': cond2_pass,
               'cond3_random_le2': cond3_pass,
               'cond4_bootstrap_pos30': cond4_pass,
               'cond5_train_test_stability': cond5_pass,
               'all_pass': all_pass,
           }}
    p = ROOT / 'results' / f'm3_r10_multidim_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
