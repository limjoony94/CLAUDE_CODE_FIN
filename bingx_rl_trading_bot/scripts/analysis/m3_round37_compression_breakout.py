"""M3-R37 — Volatility Compression Breakout (NR7 + Bollinger Squeeze) LOCKED OOS.

Pre-registered (claudedocs/m3_round37_compression_breakout_prereg.md, commit 0661b09).
Honest prior ~0% (5/5 prior false positives).

Locked params (theory-based, NO sweep):
  compression_lookback=7 (NR7), bandwidth_lookback=20, bandwidth_pctile_max=0.20
  body_min_ratio=0.4, volume_mult=1.0, bb_period=20, bb_std=2.0

Exit: run_bt_c1_production (constant across rounds)
Friction: 0.07

ALL 3 OOS tests required: WF 5-fold, Bootstrap 1000×3d, Train/Test 60/40
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


LOCKED = {
    'compression_lookback': 7,
    'bandwidth_lookback': 20,
    'bandwidth_pctile_max': 0.20,
    'body_min_ratio': 0.4,
    'volume_mult': 1.0,
    'bb_period': 20,
    'bb_std': 2.0,
}
FRICTION = 0.07


def add_compression_features(df):
    """Add NR7 narrowing + Bollinger Bandwidth columns."""
    n = len(df)
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values

    # Bar range
    rng = hi - lo
    df['bar_range'] = rng

    # NR7: current bar is the narrowest of the last 7 (including itself)
    cl_lookback = LOCKED['compression_lookback']
    nr7 = np.zeros(n, dtype=bool)
    for i in range(cl_lookback - 1, n):
        window = rng[i - cl_lookback + 1: i + 1]
        nr7[i] = (rng[i] == window.min()) and (rng[i] > 0)
    df['nr7'] = nr7

    # Bollinger Bandwidth = (upper - lower) / middle
    bb_p = LOCKED['bb_period']
    bb_std = LOCKED['bb_std']
    cl_ser = pd.Series(cl)
    sma = cl_ser.rolling(bb_p, min_periods=bb_p).mean()
    sd = cl_ser.rolling(bb_p, min_periods=bb_p).std(ddof=0)
    upper = sma + bb_std * sd
    lower = sma - bb_std * sd
    bw = (upper - lower) / sma  # bandwidth normalized
    df['bb_bandwidth'] = bw.values

    # Bandwidth percentile within recent bandwidth_lookback bars (lower 20% threshold)
    bw_lookback = LOCKED['bandwidth_lookback']
    bw_ser = pd.Series(bw.values)
    # Each bar's bandwidth percentile rank within last 20 bars (0.0 = lowest, 1.0 = highest)
    bw_pctile = bw_ser.rolling(bw_lookback, min_periods=bw_lookback).rank(pct=True)
    df['bb_pctile'] = bw_pctile.values

    return df


def entry_compression_breakout_15m(df, valid, params=None):
    """NR7 + Bollinger Squeeze + body filter + 7-bar break direction."""
    p = LOCKED if params is None else params
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    rng = df['bar_range'].values
    nr7 = df['nr7'].values
    bb_pctile = df['bb_pctile'].values
    vol = df['volume'].values
    vol_sma = df['volume_sma20'].values

    # Prev 7-bar high/low (excluding current bar)
    cl_lookback = p['compression_lookback']
    high_prev = pd.Series(hi).rolling(cl_lookback, min_periods=cl_lookback).max().shift(1).values
    low_prev = pd.Series(lo).rolling(cl_lookback, min_periods=cl_lookback).min().shift(1).values

    sigs = []
    body_min = p['body_min_ratio']
    bw_max = p['bandwidth_pctile_max']
    vmin = p['volume_mult']

    for i in range(p['bandwidth_lookback'] + 5, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (cl[i], op[i], hi[i], lo[i], rng[i], bb_pctile[i],
                                     high_prev[i], low_prev[i], vol[i], vol_sma[i])):
            continue
        if rng[i] <= 0: continue

        # Volatility compression gate (squeeze)
        if bb_pctile[i] > bw_max: continue
        # NR7 narrowing on a *prior* bar (we want compression then a break,
        # so check NR7 within last 3 bars to allow a 1-2 bar "build-up")
        recent_nr7 = any(nr7[i - k] for k in range(1, 4) if i - k >= 0)
        if not recent_nr7: continue

        # Body filter on current bar
        body = cl[i] - op[i]
        if abs(body) / rng[i] < body_min: continue

        # Volume confirmation
        if vol[i] < vmin * vol_sma[i]: continue

        # Direction: must break the prior 7-bar range with body matching direction
        long_break = (cl[i] > high_prev[i]) and (body > 0)
        short_break = (cl[i] < low_prev[i]) and (body < 0)

        if long_break:
            sigs.append((i, 'LONG'))
        elif short_break:
            sigs.append((i, 'SHORT'))
    return sigs


def main():
    print("=" * 100)
    print("M3-R37 — Compression Breakout (NR7 + BB Squeeze) LOCKED OOS Verification")
    print("=" * 100)
    print(f"Locked params: {LOCKED}")
    print(f"Friction: {FRICTION}")
    print(f"Pre-reg: claudedocs/m3_round37_compression_breakout_prereg.md (commit 0661b09)")
    print(f"Honest prior: ~0% (5/5 prior FP under same envelope)\n")

    df, valid = prepare_15m_data()
    df = add_15m_extras(df)
    df = add_compression_features(df)
    valid = (valid & (~df['sma200_long'].isna()).values & (~df['htf4_long'].isna()).values
              & (~pd.isna(df['ema20_15m']).values) & (~pd.isna(df['ema50_15m']).values)
              & (~pd.isna(df['volume_sma20']).values)
              & (~pd.isna(df['bb_bandwidth']).values) & (~pd.isna(df['bb_pctile']).values))
    n_total = len(df)
    print(f"15m bars: {n_total:,} | days: {n_total/96:.0f}\n")

    # Sanity check: how many compression signals exist on full dataset?
    sigs_full = entry_compression_breakout_15m(df, valid)
    print(f"Full-dataset signals: {len(sigs_full)} ({len(sigs_full)/(n_total/96):.3f}/day)\n")

    # ----------------------------------------------------------------------
    # TEST 1: WF 5-fold expanding
    # ----------------------------------------------------------------------
    print("=" * 100)
    print("TEST 1 — WF 5-fold expanding (locked params, friction=0.07)")
    print("Pass criterion: ≥3/5 folds daily_net > 0")
    print("=" * 100)
    fold_size = n_total // 6
    wf_results = []
    for fold_i in range(5):
        te_s = (fold_i + 1) * fold_size
        te_e = min(te_s + fold_size, n_total)
        df_f = df.iloc[te_s:te_e].reset_index(drop=True)
        v_f = valid[te_s:te_e]
        sigs_f = entry_compression_breakout_15m(df_f, v_f)
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
    # TEST 2: Bootstrap 1000 × 3-day
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
        sigs_w = entry_compression_breakout_15m(df_w, v_w)
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
    # TEST 3: Train/Test 60/40
    # ----------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("TEST 3 — Train/Test 60/40 split (locked params, friction=0.07)")
    print("Pass criterion: BOTH train AND test daily_net > 0")
    print("=" * 100)
    train_end = int(n_total * 0.6)
    df_tr = df.iloc[:train_end].reset_index(drop=True)
    df_te = df.iloc[train_end:].reset_index(drop=True)
    v_tr = valid[:train_end]
    v_te = valid[train_end:]
    sigs_tr = entry_compression_breakout_15m(df_tr, v_tr)
    sigs_te = entry_compression_breakout_15m(df_te, v_te)
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
    print("M3-R37 FINAL VERDICT (pre-registered, all 3 required)")
    print(f"{'='*100}")
    print(f"  Test 1 (WF 5-fold expanding ≥3/5):     {'PASS' if wf_pass else 'FAIL'}  ({wf_pos}/5)")
    print(f"  Test 2 (Bootstrap pos_rate ≥ 50%):    {'PASS' if bs_pass else 'FAIL'}  ({bs_pos_rate:.4f})")
    print(f"  Test 3 (Train+Test sign-agree):        {'PASS' if tt_pass else 'FAIL'}")
    all_pass = wf_pass and bs_pass and tt_pass
    print(f"\n  OVERALL: {'ALL 3 PASS — must call advisor before any claim' if all_pass else 'FAIL — 6th OOS negative committed to evidence pile'}")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg': 'claudedocs/m3_round37_compression_breakout_prereg.md',
        'pre_reg_commit': '0661b09',
        'locked_params': LOCKED,
        'friction': FRICTION,
        'full_signals_count': len(sigs_full),
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
    p = ROOT / 'results' / f'm3_r37_compression_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
