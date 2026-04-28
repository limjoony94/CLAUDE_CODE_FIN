"""Trade-Tape R1 — Persistent Taker Imbalance + 1h Trend (1m signals → 15m exit) LOCKED OOS.

Pre-registered (claudedocs/trade_tape_r1_persistent_imbalance_prereg.md, commit 7fb7407).
Honest prior: trade-tape envelope distinct from R41-falsified OHLCV — no prior evidence either way.

Locked params (theory-based):
  imbalance_window_min=5, imbalance_threshold=0.30,
  body_min_ratio=0.4, trend_filter='1h_sma200', min_bars_between_15m=2

ALL 3 OOS tests required: WF 5-fold, Bootstrap 1000×3d, Train/Test 60/40.
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_round30_c1_production_exact import run_bt_c1_production
from m3_round35_15m_deep import trade_summary
from m3_round29_c1_exact_revalidation import prepare_15m_data


LOCKED = {
    'imbalance_window_min': 5,
    'imbalance_threshold': 0.30,
    'body_min_ratio': 0.4,
    'min_bars_between_15m': 2,
}
FRICTION = 0.07
VACUITY_FLOOR_PER_DAY = 0.5

FEATURES_PATH = ROOT / 'data' / 'btc_trade_features_1m.parquet'


def load_aligned_data():
    """Load 1m trade features + 15m OHLCV, aligned to overlap range."""
    feat = pd.read_parquet(FEATURES_PATH)
    feat['timestamp'] = pd.to_datetime(feat['timestamp'])
    if feat['timestamp'].dt.tz is not None:
        feat['timestamp'] = feat['timestamp'].dt.tz_localize(None)

    df_15m, valid_15m = prepare_15m_data()
    if df_15m['timestamp'].dt.tz is not None:
        df_15m['timestamp'] = df_15m['timestamp'].dt.tz_localize(None)

    # Add 1h SMA200 trend filter to 15m frame (same as R41)
    df_1h = df_15m.set_index('timestamp').resample('1h', label='left', closed='left').agg({
        'close': 'last'}).dropna().reset_index()
    df_1h['sma200'] = pd.Series(df_1h['close'].values).rolling(200, min_periods=200).mean().values
    df_1h['close_above_sma200'] = df_1h['close'] > df_1h['sma200']
    # Merge to 15m frame
    df_15m = df_15m.sort_values('timestamp').reset_index(drop=True)
    df_15m_with_h1 = pd.merge_asof(
        df_15m[['timestamp']], df_1h[['timestamp', 'close_above_sma200']].sort_values('timestamp'),
        on='timestamp', direction='backward', tolerance=pd.Timedelta('1h')
    )
    df_15m['sma200_long'] = df_15m_with_h1['close_above_sma200'].values

    # Update valid mask for trend filter
    valid_15m = valid_15m & (~df_15m['sma200_long'].isna()).values

    # Compute overlap range
    feat_min = feat['timestamp'].min()
    feat_max = feat['timestamp'].max()
    ohlcv_min = df_15m['timestamp'].min()
    ohlcv_max = df_15m['timestamp'].max()
    overlap_start = max(feat_min, ohlcv_min)
    overlap_end = min(feat_max, ohlcv_max)
    print(f'  feature range: {feat_min} → {feat_max}')
    print(f'  OHLCV range:   {ohlcv_min} → {ohlcv_max}')
    print(f'  overlap:       {overlap_start} → {overlap_end} ({(overlap_end - overlap_start).days} days)')

    # Trim both to overlap
    feat_o = feat[(feat['timestamp'] >= overlap_start) & (feat['timestamp'] <= overlap_end)].reset_index(drop=True)
    mask_15m = (df_15m['timestamp'] >= overlap_start) & (df_15m['timestamp'] <= overlap_end)
    df_15m_o = df_15m[mask_15m].reset_index(drop=True)
    valid_15m_o = valid_15m[mask_15m.values]

    return feat_o, df_15m_o, valid_15m_o


def entry_persistent_imbalance(feat: pd.DataFrame, df_15m: pd.DataFrame, valid_15m: np.ndarray):
    """Generate signals from 1-min trade-flow features, project to 15m bar index."""
    f = feat.copy()
    # Rolling sum of vol_buy, vol_sell, vol_total over window
    win = LOCKED['imbalance_window_min']
    f['rs_buy'] = f['vol_buy'].rolling(win, min_periods=win).sum()
    f['rs_sell'] = f['vol_sell'].rolling(win, min_periods=win).sum()
    f['rs_total'] = f['vol_total'].rolling(win, min_periods=win).sum()
    f['roll_imb'] = (f['rs_buy'] - f['rs_sell']) / f['rs_total'].replace(0, np.nan)

    # Body filter
    f['body_abs'] = (f['price_last'] - f['price_first']).abs()
    f['range'] = f['price_high'] - f['price_low']
    f['body_dir'] = np.sign(f['price_last'] - f['price_first'])

    # Map to 15m bar index
    ts15 = df_15m['timestamp'].values  # numpy datetime64
    idx_lookup = {pd.Timestamp(ts).to_pydatetime(): i for i, ts in enumerate(ts15)}
    f['floor_15m'] = pd.to_datetime(f['timestamp']).dt.floor('15min')

    # Trend filter for each 15m bar
    sma_long = df_15m['sma200_long'].fillna(False).astype(bool).values

    sigs = []
    seen_bars = set()
    threshold = LOCKED['imbalance_threshold']
    body_min = LOCKED['body_min_ratio']

    for row in f.itertuples(index=False):
        if pd.isna(row.roll_imb):
            continue
        if pd.isna(row.range) or row.range <= 0:
            continue
        # Body filter
        if (row.body_abs / row.range) < body_min:
            continue

        # 15m bar lookup
        bar_ts = row.floor_15m.to_pydatetime() if hasattr(row.floor_15m, 'to_pydatetime') else pd.Timestamp(row.floor_15m).to_pydatetime()
        bidx = idx_lookup.get(bar_ts)
        if bidx is None or bidx >= len(df_15m) or bidx < 0:
            continue
        if bidx in seen_bars:
            continue
        if not valid_15m[bidx]:
            continue

        sma_up = sma_long[bidx]

        long_setup = (row.roll_imb >= threshold) and (row.body_dir > 0) and sma_up
        short_setup = (row.roll_imb <= -threshold) and (row.body_dir < 0) and (not sma_up)

        if long_setup:
            sigs.append((bidx, 'LONG'))
            seen_bars.add(bidx)
        elif short_setup:
            sigs.append((bidx, 'SHORT'))
            seen_bars.add(bidx)

    sigs.sort(key=lambda x: x[0])
    return sigs


def main():
    print("=" * 100)
    print("Trade-Tape R1 — Persistent Imbalance + 1h Trend (1m → 15m exit) LOCKED OOS")
    print("=" * 100)
    print(f"Locked params: {LOCKED}")
    print(f"Friction: {FRICTION}")
    print(f"Pre-reg: claudedocs/trade_tape_r1_persistent_imbalance_prereg.md (commit 7fb7407)")

    feat, df_15m, valid_15m = load_aligned_data()
    n_15m = len(df_15m)
    n_feat = len(feat)
    days = n_15m / 96
    print(f'\n1m features: {n_feat:,} | 15m bars: {n_15m:,} | days: {days:.0f}\n')

    sigs_full = entry_persistent_imbalance(feat, df_15m, valid_15m)
    sig_per_day = len(sigs_full) / max(1, days)
    print(f'Full-dataset signals: {len(sigs_full)} ({sig_per_day:.3f}/day)')

    if sig_per_day < VACUITY_FLOOR_PER_DAY:
        print(f"\n  ⚠️  VACUITY GATE FAIL — {sig_per_day:.3f}/day < {VACUITY_FLOOR_PER_DAY}/day")
        out = {'date': datetime.now(timezone.utc).isoformat(), 'pre_reg_commit': '7fb7407',
               'locked_params': LOCKED, 'friction': FRICTION,
               'full_signals_count': len(sigs_full), 'signals_per_day': sig_per_day,
               'vacuity_gate_pass': False, 'verdict': 'INCONCLUSIVE'}
        p = ROOT / 'results' / f'trade_tape_r1_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
        print(f'\nSaved: {p}')
        return

    print(f'  ✓ Vacuity gate PASS\n')

    # TEST 1: WF 5-fold expanding
    print("=" * 100)
    print("TEST 1 — WF 5-fold expanding")
    print("=" * 100)
    fold_size = n_15m // 6
    wf_results = []
    feat_ts = pd.to_datetime(feat['timestamp']).values

    for fold_i in range(5):
        te_s = (fold_i + 1) * fold_size
        te_e = min(te_s + fold_size, n_15m)
        df_15m_f = df_15m.iloc[te_s:te_e].reset_index(drop=True)
        v_f = valid_15m[te_s:te_e]
        # Feature mask for this fold
        t_start = df_15m['timestamp'].iloc[te_s]
        t_end = df_15m['timestamp'].iloc[te_e - 1] if te_e <= n_15m else df_15m['timestamp'].iloc[-1]
        mask_f = (feat['timestamp'] >= t_start) & (feat['timestamp'] <= t_end + pd.Timedelta(minutes=14))
        feat_f = feat.loc[mask_f].reset_index(drop=True)

        sigs_f = entry_persistent_imbalance(feat_f, df_15m_f, v_f)
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

    # TEST 2: Bootstrap
    print(f"\n{'='*100}\nTEST 2 — Bootstrap 1000 × 3-day\n{'='*100}")
    bars_per_3day = 3 * 96
    max_start = n_15m - bars_per_3day - 1
    random.seed(42)
    starts = random.sample(range(max(1, max_start)), min(1000, max(1, max_start)))
    cand_pnls = []
    for st in starts:
        en = st + bars_per_3day
        df_15m_w = df_15m.iloc[st:en].reset_index(drop=True)
        v_w = valid_15m[st:en]
        t_start = df_15m['timestamp'].iloc[st]
        t_end = df_15m['timestamp'].iloc[en - 1] if en <= n_15m else df_15m['timestamp'].iloc[-1]
        mask_w = (feat['timestamp'] >= t_start) & (feat['timestamp'] <= t_end + pd.Timedelta(minutes=14))
        feat_w = feat.loc[mask_w].reset_index(drop=True)
        sigs_w = entry_persistent_imbalance(feat_w, df_15m_w, v_w)
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

    # TEST 3: Train/Test 60/40
    print(f"\n{'='*100}\nTEST 3 — Train/Test 60/40\n{'='*100}")
    train_end = int(n_15m * 0.6)
    df_15m_tr = df_15m.iloc[:train_end].reset_index(drop=True)
    df_15m_te = df_15m.iloc[train_end:].reset_index(drop=True)
    v_tr = valid_15m[:train_end]; v_te = valid_15m[train_end:]
    t_tr_end = df_15m['timestamp'].iloc[train_end - 1]
    feat_tr = feat[feat['timestamp'] <= t_tr_end + pd.Timedelta(minutes=14)].reset_index(drop=True)
    feat_te = feat[feat['timestamp'] > t_tr_end + pd.Timedelta(minutes=14)].reset_index(drop=True)

    sigs_tr = entry_persistent_imbalance(feat_tr, df_15m_tr, v_tr)
    sigs_te = entry_persistent_imbalance(feat_te, df_15m_te, v_te)
    trades_tr = run_bt_c1_production(df_15m_tr, sigs_tr, friction=FRICTION)
    trades_te = run_bt_c1_production(df_15m_te, sigs_te, friction=FRICTION)
    s_tr = trade_summary(trades_tr)
    s_te = trade_summary(trades_te)
    if s_tr is None or s_te is None:
        print(f"  train: {'NULL' if s_tr is None else 'n=' + str(s_tr['n'])}")
        print(f"  test:  {'NULL' if s_te is None else 'n=' + str(s_te['n'])}")
        tt_pass = False
    else:
        print(f"  train: n={s_tr['n']:>4} daily={s_tr['daily_net']:>+.4f}% WR={s_tr['wr_pct']:>5.1f}% RR={s_tr['rr']:>5.2f} avg_g={s_tr['avg_gross']:>+.4f}%")
        print(f"  test:  n={s_te['n']:>4} daily={s_te['daily_net']:>+.4f}% WR={s_te['wr_pct']:>5.1f}% RR={s_te['rr']:>5.2f} avg_g={s_te['avg_gross']:>+.4f}%")
        tt_pass = (s_tr['daily_net'] > 0) and (s_te['daily_net'] > 0)
    print(f"  Train/Test both > 0: {'PASS' if tt_pass else 'FAIL'}")

    print(f"\n{'='*100}\nTrade-Tape R1 FINAL VERDICT\n{'='*100}")
    print(f"  Vacuity gate: PASS ({sig_per_day:.3f}/day, {len(sigs_full)} signals)")
    print(f"  Test 1 (WF ≥3/5):                       {'PASS' if wf_pass else 'FAIL'}  ({wf_pos}/5)")
    print(f"  Test 2 (Bootstrap ≥50%):                {'PASS' if bs_pass else 'FAIL'}  ({bs_pos_rate:.4f})")
    print(f"  Test 3 (Train+Test sign-agree):          {'PASS' if tt_pass else 'FAIL'}")
    all_pass = wf_pass and bs_pass and tt_pass
    print(f"  OVERALL: {'ALL 3 PASS — call advisor before any claim' if all_pass else 'FAIL — 1st trade-tape OOS negative'}")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '7fb7407',
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
    p = ROOT / 'results' / f'trade_tape_r1_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
