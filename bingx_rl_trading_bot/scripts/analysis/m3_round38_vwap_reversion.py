"""M3-R38 — VWAP-anchored Mean Reversion (5m signals projected to 15m exit) LOCKED OOS.

Pre-registered (claudedocs/m3_round38_vwap_reversion_prereg.md, commit a257838).
Honest prior ~0% (6/6 prior FP same envelope, user explicit override).

Locked params (theory-based, NO sweep):
  vwap_dev_min_pct=0.5, body_min_ratio=0.4, volume_mult=1.0,
  wick_to_body_min=2.0, session_reset='UTC 00:00'

5m signal generation + 15m exit framework (consistent with R36/R37).
Trend: 1h SMA200 + 4h EMA20/50; Confluence: 15m EMA20/50.

ALL 3 OOS tests required: WF 5-fold, Bootstrap 1000×3d, Train/Test 60/40.
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
from m2_round1_screening import compute_ema, load_ohlcv


LOCKED = {
    'vwap_dev_min_pct': 0.5,
    'body_min_ratio': 0.4,
    'volume_mult': 1.0,
    'wick_to_body_min': 2.0,
    'session_reset_hour_utc': 0,
}
FRICTION = 0.07


def add_5m_vwap_features(df_5m):
    """Add session-anchored VWAP + deviation, plus 5m volume SMA on 5m timeframe."""
    df = df_5m.copy()
    ts = pd.to_datetime(df['timestamp'])
    # Session date (UTC, reset at 00:00)
    df['session_date'] = ts.dt.date

    cl = df['close'].values
    vol = df['volume'].values
    pv = cl * vol  # price * volume

    # Cumulative volume and pv per session, vectorized via groupby cumsum
    df['_pv'] = pv
    df['_v'] = vol
    df['cum_pv'] = df.groupby('session_date')['_pv'].cumsum()
    df['cum_v'] = df.groupby('session_date')['_v'].cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_v']
    df['dev_pct'] = (df['close'] - df['vwap']) / df['vwap'] * 100

    # 5m volume SMA20
    df['vol_sma20_5m'] = pd.Series(vol).rolling(20, min_periods=20).mean().values

    # 5m EMA20 / EMA50 (used for fast confirmation)
    df['ema20_5m'] = compute_ema(cl, 20)
    df['ema50_5m'] = compute_ema(cl, 50)

    df = df.drop(columns=['_pv', '_v', 'cum_pv', 'cum_v'])
    return df


def synthesize_15m_from_5m(df_5m):
    """Aggregate 5m to 15m bars (3 5m bars per 15m)."""
    df = df_5m.copy()
    ts = pd.to_datetime(df['timestamp'])
    # Floor to 15m grid
    df['t15'] = ts.dt.floor('15min')
    g = df.groupby('t15').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).reset_index().rename(columns={'t15': 'timestamp'})
    return g


def detect_rejection_at_5m(op, hi, lo, cl, i):
    """Bullish reversal (hammer/bull engulfing) or bearish (shooting star/bear engulfing)."""
    body = cl[i] - op[i]
    rng = hi[i] - lo[i]
    if rng <= 0:
        return False, False
    abs_body = abs(body)

    # Hammer: long lower wick (>= wick_to_body_min × |body|), small upper wick
    upper_wick = hi[i] - max(cl[i], op[i])
    lower_wick = min(cl[i], op[i]) - lo[i]
    is_hammer = (lower_wick >= LOCKED['wick_to_body_min'] * abs_body) and (upper_wick <= abs_body) and (body > 0)
    is_shooting = (upper_wick >= LOCKED['wick_to_body_min'] * abs_body) and (lower_wick <= abs_body) and (body < 0)

    # Engulfing (compares to prev bar)
    if i > 0:
        prev_body = cl[i-1] - op[i-1]
        bull_eng = (body > 0) and (prev_body < 0) and (cl[i] > op[i-1]) and (op[i] < cl[i-1])
        bear_eng = (body < 0) and (prev_body > 0) and (cl[i] < op[i-1]) and (op[i] > cl[i-1])
    else:
        bull_eng = bear_eng = False

    bullish_signal = is_hammer or bull_eng
    bearish_signal = is_shooting or bear_eng
    return bullish_signal, bearish_signal


def entry_vwap_reversion_5m_to_15m(df_5m, df_15m, valid_15m):
    """Generate signals at 5m granularity, project to 15m bar index."""
    op5 = df_5m['open'].values
    hi5 = df_5m['high'].values
    lo5 = df_5m['low'].values
    cl5 = df_5m['close'].values
    vol5 = df_5m['volume'].values
    vol_sma5 = df_5m['vol_sma20_5m'].values
    dev = df_5m['dev_pct'].values
    ema20_5m = df_5m['ema20_5m'].values
    ema50_5m = df_5m['ema50_5m'].values

    # Trend filters from 15m frame (aligned via timestamp)
    sma_long_15m = df_15m['sma200_long'].fillna(False).astype(bool).values
    h4_long_15m = df_15m['htf4_long'].fillna(False).astype(bool).values
    ema20_15m = df_15m['ema20_15m'].values
    ema50_15m = df_15m['ema50_15m'].values

    # Map each 5m timestamp to its 15m bar index
    ts_5m = pd.to_datetime(df_5m['timestamp']).dt.floor('15min')
    ts_15m = pd.to_datetime(df_15m['timestamp'])
    idx_15m_lookup = {t: i for i, t in enumerate(ts_15m)}
    bar_idx_for_5m = ts_5m.map(idx_15m_lookup).values

    n5 = len(df_5m)
    sigs_15m = []  # list of (15m_index, direction)
    seen_15m_idx = set()  # only one signal per 15m bar (first 5m signal wins)

    body_min = LOCKED['body_min_ratio']
    vmin = LOCKED['volume_mult']
    dev_min = LOCKED['vwap_dev_min_pct']

    for i in range(50, n5):
        # Defensive NaN
        if any(pd.isna(x) for x in (op5[i], hi5[i], lo5[i], cl5[i], dev[i],
                                     vol5[i], vol_sma5[i], ema20_5m[i], ema50_5m[i])):
            continue
        rng = hi5[i] - lo5[i]
        if rng <= 0:
            continue

        # 5m body filter
        body = cl5[i] - op5[i]
        if abs(body) / rng < body_min:
            continue

        # 5m volume confirmation
        if vol5[i] < vmin * vol_sma5[i]:
            continue

        # VWAP deviation gate
        if abs(dev[i]) < dev_min:
            continue

        # Rejection candle
        bullish, bearish = detect_rejection_at_5m(op5, hi5, lo5, cl5, i)
        if not (bullish or bearish):
            continue

        # 15m bar lookup
        bidx = bar_idx_for_5m[i]
        if pd.isna(bidx):
            continue
        bidx = int(bidx)
        if bidx in seen_15m_idx:
            continue
        if bidx >= len(df_15m) or bidx < 0:
            continue
        if not valid_15m[bidx]:
            continue

        # Trend filters (15m frame at the same bar)
        sma_up = sma_long_15m[bidx]
        h4_up = h4_long_15m[bidx]
        if pd.isna(ema20_15m[bidx]) or pd.isna(ema50_15m[bidx]):
            continue
        ema_up = ema20_15m[bidx] > ema50_15m[bidx]
        ema_down = ema20_15m[bidx] < ema50_15m[bidx]

        # Mean reversion: dev < 0 → expect rebound up (LONG); dev > 0 → expect down (SHORT)
        long_setup = (dev[i] <= -dev_min) and bullish and sma_up and h4_up and ema_up and (cl5[i] > ema20_5m[i])
        short_setup = (dev[i] >= dev_min) and bearish and (not sma_up) and (not h4_up) and ema_down and (cl5[i] < ema20_5m[i])

        if long_setup:
            sigs_15m.append((bidx, 'LONG'))
            seen_15m_idx.add(bidx)
        elif short_setup:
            sigs_15m.append((bidx, 'SHORT'))
            seen_15m_idx.add(bidx)

    # Sort by 15m bar index
    sigs_15m.sort(key=lambda x: x[0])
    return sigs_15m


def prepare_5m_15m_data():
    """Load both 5m and 15m frames, align timestamps."""
    # 15m frame already prepared
    df_15m, valid_15m = prepare_15m_data()
    df_15m = add_15m_extras(df_15m)
    valid_15m = (valid_15m & (~df_15m['sma200_long'].isna()).values
                  & (~df_15m['htf4_long'].isna()).values
                  & (~pd.isna(df_15m['ema20_15m']).values)
                  & (~pd.isna(df_15m['ema50_15m']).values)
                  & (~pd.isna(df_15m['volume_sma20']).values))

    # Load 5m raw OHLCV
    data_p = ROOT / 'data' / 'btc_5m_720days_binance.csv'
    df_5m = pd.read_csv(data_p)
    df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])

    # Normalize timezone: strip tz info from both frames so they align
    if df_5m['timestamp'].dt.tz is not None:
        df_5m['timestamp'] = df_5m['timestamp'].dt.tz_localize(None)
    if df_15m['timestamp'].dt.tz is not None:
        df_15m['timestamp'] = df_15m['timestamp'].dt.tz_localize(None)

    # Add 5m VWAP + features
    df_5m = add_5m_vwap_features(df_5m)
    return df_5m, df_15m, valid_15m


def main():
    print("=" * 100)
    print("M3-R38 — VWAP Reversion (5m signals → 15m exit) LOCKED OOS Verification")
    print("=" * 100)
    print(f"Locked params: {LOCKED}")
    print(f"Friction: {FRICTION}")
    print(f"Pre-reg: claudedocs/m3_round38_vwap_reversion_prereg.md (commit a257838)")
    print(f"Honest prior: ~0% (6/6 prior FP same envelope, user explicit override)\n")

    df_5m, df_15m, valid_15m = prepare_5m_15m_data()
    n_5m = len(df_5m)
    n_15m = len(df_15m)
    print(f"5m bars: {n_5m:,} | 15m bars: {n_15m:,} | days: {n_15m/96:.0f}\n")

    # Sanity check: full-data signal count
    sigs_full = entry_vwap_reversion_5m_to_15m(df_5m, df_15m, valid_15m)
    print(f"Full-dataset signals: {len(sigs_full)} ({len(sigs_full)/(n_15m/96):.3f}/day)\n")

    # ----------------------------------------------------------------------
    # TEST 1: WF 5-fold expanding (15m frame splits, 5m derived per-fold)
    # ----------------------------------------------------------------------
    print("=" * 100)
    print("TEST 1 — WF 5-fold expanding (locked params, friction=0.07)")
    print("Pass criterion: ≥3/5 folds daily_net > 0")
    print("=" * 100)

    fold_size = n_15m // 6
    wf_results = []

    # Map 15m bar timestamp ranges to 5m frame slices
    ts15 = pd.to_datetime(df_15m['timestamp'])
    ts5 = pd.to_datetime(df_5m['timestamp'])

    for fold_i in range(5):
        te_s = (fold_i + 1) * fold_size
        te_e = min(te_s + fold_size, n_15m)
        df_15m_f = df_15m.iloc[te_s:te_e].reset_index(drop=True)
        v_f = valid_15m[te_s:te_e]

        # Slice 5m to match 15m fold time range
        t_start = ts15.iloc[te_s]
        t_end = ts15.iloc[te_e - 1] if te_e <= n_15m else ts15.iloc[-1]
        mask5 = (ts5 >= t_start) & (ts5 <= t_end + pd.Timedelta(minutes=14))
        df_5m_f = df_5m.loc[mask5].reset_index(drop=True)

        sigs_f = entry_vwap_reversion_5m_to_15m(df_5m_f, df_15m_f, v_f)
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
    # TEST 2: Bootstrap 1000 × 3-day
    # ----------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("TEST 2 — Bootstrap 1000 × 3-day windows (locked params, friction=0.07)")
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
        sigs_w = entry_vwap_reversion_5m_to_15m(df_5m_w, df_15m_w, v_w)
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
    # TEST 3: Train/Test 60/40
    # ----------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("TEST 3 — Train/Test 60/40 split (locked params, friction=0.07)")
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

    sigs_tr = entry_vwap_reversion_5m_to_15m(df_5m_tr, df_15m_tr, v_tr)
    sigs_te = entry_vwap_reversion_5m_to_15m(df_5m_te, df_15m_te, v_te)
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

    # ----------------------------------------------------------------------
    # FINAL VERDICT
    # ----------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("M3-R38 FINAL VERDICT (pre-registered, all 3 required)")
    print(f"{'='*100}")
    print(f"  Test 1 (WF 5-fold expanding ≥3/5):     {'PASS' if wf_pass else 'FAIL'}  ({wf_pos}/5)")
    print(f"  Test 2 (Bootstrap pos_rate ≥ 50%):    {'PASS' if bs_pass else 'FAIL'}  ({bs_pos_rate:.4f})")
    print(f"  Test 3 (Train+Test sign-agree):        {'PASS' if tt_pass else 'FAIL'}")
    all_pass = wf_pass and bs_pass and tt_pass
    print(f"\n  OVERALL: {'ALL 3 PASS — must call advisor before any claim' if all_pass else 'FAIL — 7th OOS negative committed'}")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg': 'claudedocs/m3_round38_vwap_reversion_prereg.md',
        'pre_reg_commit': 'a257838',
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
    p = ROOT / 'results' / f'm3_r38_vwap_reversion_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
