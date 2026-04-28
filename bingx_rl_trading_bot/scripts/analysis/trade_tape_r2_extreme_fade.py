"""Trade-Tape R2 — Extreme Single-Bar Imbalance Fade (1m → 15m exit) LOCKED OOS.

Pre-registered (claudedocs/trade_tape_r2_extreme_fade_prereg.md, commit f584711).
Advisor structural opposite to R1: mean-reversion fade vs continuation.

Locked params (theory-based):
  imb_extreme_threshold=0.85, intensity_pctile=0.90, intensity_lookback_min=60
  NO body filter, NO MTF trend filter

ALL 3 OOS tests required: WF 5-fold, Bootstrap 1000×3d, Train/Test 60/40.
avg_gross logged at every fold per advisor (friction-floor pattern visibility).
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
from trade_tape_r1_persistent_imbalance import load_aligned_data


LOCKED = {
    'imb_extreme_threshold': 0.85,
    'intensity_pctile': 0.90,
    'intensity_lookback_min': 60,
    'min_bars_between_15m': 2,
}
FRICTION = 0.07
VACUITY_FLOOR_PER_DAY = 0.5


def entry_extreme_fade(feat: pd.DataFrame, df_15m: pd.DataFrame, valid_15m: np.ndarray):
    """Extreme single-bar imbalance + intensity → fade direction."""
    f = feat.copy()
    # Rolling 60-min trade_count p90 (causal — uses last 60 bars including current)
    win = LOCKED['intensity_lookback_min']
    f['tc_p90'] = f['trade_count'].rolling(win, min_periods=win).quantile(LOCKED['intensity_pctile'])
    # Project to 15m bar index
    ts15 = df_15m['timestamp'].values
    idx_lookup = {pd.Timestamp(ts).to_pydatetime(): i for i, ts in enumerate(ts15)}
    f['floor_15m'] = pd.to_datetime(f['timestamp']).dt.floor('15min')

    sigs = []
    seen_bars = set()
    threshold = LOCKED['imb_extreme_threshold']

    for row in f.itertuples(index=False):
        if pd.isna(row.vol_imbalance) or pd.isna(row.tc_p90):
            continue
        # Extreme imbalance gate
        if abs(row.vol_imbalance) < threshold:
            continue
        # Intensity gate
        if row.trade_count <= row.tc_p90:
            continue

        # Fade direction
        if row.vol_imbalance <= -threshold:
            direction = 'LONG'  # extreme sell exhaustion → fade up
        elif row.vol_imbalance >= threshold:
            direction = 'SHORT'  # extreme buy exhaustion → fade down
        else:
            continue

        bar_ts = row.floor_15m.to_pydatetime() if hasattr(row.floor_15m, 'to_pydatetime') else pd.Timestamp(row.floor_15m).to_pydatetime()
        bidx = idx_lookup.get(bar_ts)
        if bidx is None or bidx >= len(df_15m) or bidx < 0:
            continue
        if bidx in seen_bars:
            continue
        if not valid_15m[bidx]:
            continue

        sigs.append((bidx, direction))
        seen_bars.add(bidx)

    sigs.sort(key=lambda x: x[0])
    return sigs


def main():
    print("=" * 100)
    print("Trade-Tape R2 — Extreme Single-Bar Fade (1m → 15m exit) LOCKED OOS")
    print("=" * 100)
    print(f"Locked params: {LOCKED}")
    print(f"Friction: {FRICTION}")
    print(f"Pre-reg: claudedocs/trade_tape_r2_extreme_fade_prereg.md (commit f584711)")
    print("Theory: VPIN-style extreme exhaustion → mean reversion (advisor 'structural opposite' to R1)\n")

    feat, df_15m, valid_15m = load_aligned_data()
    n_15m = len(df_15m)
    n_feat = len(feat)
    days = n_15m / 96
    print(f'1m features: {n_feat:,} | 15m bars: {n_15m:,} | days: {days:.0f}\n')

    sigs_full = entry_extreme_fade(feat, df_15m, valid_15m)
    sig_per_day = len(sigs_full) / max(1, days)
    print(f'Full-dataset signals: {len(sigs_full)} ({sig_per_day:.3f}/day)')

    if sig_per_day < VACUITY_FLOOR_PER_DAY:
        print(f"\n  ⚠️  VACUITY GATE FAIL — {sig_per_day:.3f}/day")
        out = {'date': datetime.now(timezone.utc).isoformat(), 'pre_reg_commit': 'f584711',
               'locked_params': LOCKED, 'friction': FRICTION,
               'full_signals_count': len(sigs_full), 'signals_per_day': sig_per_day,
               'vacuity_gate_pass': False, 'verdict': 'INCONCLUSIVE'}
        p = ROOT / 'results' / f'trade_tape_r2_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
        print(f'\nSaved: {p}')
        return

    print(f'  ✓ Vacuity gate PASS\n')

    # TEST 1: WF 5-fold expanding (with prominent avg_gross logging per advisor)
    print("=" * 100)
    print("TEST 1 — WF 5-fold expanding (avg_gross prominent — friction floor visibility)")
    print("=" * 100)
    print(f"  {'fold':>4} {'n':>5} {'daily_net':>10} {'avg_gross':>12} {'WR':>6} {'RR':>5}")
    print(f"  {'----':>4} {'-----':>5} {'----------':>10} {'------------':>12} {'------':>6} {'-----':>5}")
    fold_size = n_15m // 6
    wf_results = []

    for fold_i in range(5):
        te_s = (fold_i + 1) * fold_size
        te_e = min(te_s + fold_size, n_15m)
        df_15m_f = df_15m.iloc[te_s:te_e].reset_index(drop=True)
        v_f = valid_15m[te_s:te_e]
        t_start = df_15m['timestamp'].iloc[te_s]
        t_end = df_15m['timestamp'].iloc[te_e - 1] if te_e <= n_15m else df_15m['timestamp'].iloc[-1]
        mask_f = (feat['timestamp'] >= t_start) & (feat['timestamp'] <= t_end + pd.Timedelta(minutes=14))
        feat_f = feat.loc[mask_f].reset_index(drop=True)

        sigs_f = entry_extreme_fade(feat_f, df_15m_f, v_f)
        trades = run_bt_c1_production(df_15m_f, sigs_f, friction=FRICTION)
        s_f = trade_summary(trades)
        if s_f is None:
            wf_results.append({'fold': fold_i+1, 'n': 0, 'daily': None})
            print(f"  {fold_i+1:>4} {0:>5} {'  N/A':>10} {'    N/A':>12} {'  N/A':>6} {' N/A':>5}")
            continue
        wf_results.append({'fold': fold_i+1, 'n': s_f['n'], 'daily': s_f['daily_net'],
                            'wr': s_f['wr_pct'], 'rr': s_f['rr'], 'avg_g': s_f['avg_gross']})
        # Friction band visualization: ✓ if avg_gross > 0.07, ⚠ if 0.03-0.07, ✗ if <0.03
        ag = s_f['avg_gross']
        gate = '✓' if ag >= 0.07 else ('⚠' if ag >= 0.03 else '✗')
        print(f"  {fold_i+1:>4} {s_f['n']:>5} {s_f['daily_net']:>+9.4f}% {s_f['avg_gross']:>+10.4f}% {gate} {s_f['wr_pct']:>5.1f}% {s_f['rr']:>4.2f}")
    wf_pos = sum(1 for r in wf_results if r['daily'] is not None and r['daily'] > 0)
    wf_pass = wf_pos >= 3
    print(f"\n  Friction floor reference: 0.07% (taker round-trip)")
    print(f"  Folds with avg_gross > friction: {sum(1 for r in wf_results if r.get('avg_g', 0) and r['avg_g'] >= 0.07)}/5")
    print(f"  WF positive folds: {wf_pos}/5 → {'PASS' if wf_pass else 'FAIL'}")

    # TEST 2
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
        sigs_w = entry_extreme_fade(feat_w, df_15m_w, v_w)
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
    print(f"\n{'='*100}\nTEST 3 — Train/Test 60/40 (avg_gross prominent)\n{'='*100}")
    train_end = int(n_15m * 0.6)
    df_15m_tr = df_15m.iloc[:train_end].reset_index(drop=True)
    df_15m_te = df_15m.iloc[train_end:].reset_index(drop=True)
    v_tr = valid_15m[:train_end]; v_te = valid_15m[train_end:]
    t_tr_end = df_15m['timestamp'].iloc[train_end - 1]
    feat_tr = feat[feat['timestamp'] <= t_tr_end + pd.Timedelta(minutes=14)].reset_index(drop=True)
    feat_te = feat[feat['timestamp'] > t_tr_end + pd.Timedelta(minutes=14)].reset_index(drop=True)
    sigs_tr = entry_extreme_fade(feat_tr, df_15m_tr, v_tr)
    sigs_te = entry_extreme_fade(feat_te, df_15m_te, v_te)
    trades_tr = run_bt_c1_production(df_15m_tr, sigs_tr, friction=FRICTION)
    trades_te = run_bt_c1_production(df_15m_te, sigs_te, friction=FRICTION)
    s_tr = trade_summary(trades_tr)
    s_te = trade_summary(trades_te)
    if s_tr is None or s_te is None:
        print(f"  train: {'NULL' if s_tr is None else 'n=' + str(s_tr['n'])}")
        print(f"  test:  {'NULL' if s_te is None else 'n=' + str(s_te['n'])}")
        tt_pass = False
    else:
        ag_tr = s_tr['avg_gross']; ag_te = s_te['avg_gross']
        gate_tr = '✓' if ag_tr >= 0.07 else ('⚠' if ag_tr >= 0.03 else '✗')
        gate_te = '✓' if ag_te >= 0.07 else ('⚠' if ag_te >= 0.03 else '✗')
        print(f"  train: n={s_tr['n']:>4} daily={s_tr['daily_net']:>+.4f}% avg_gross={s_tr['avg_gross']:>+.4f}% {gate_tr} WR={s_tr['wr_pct']:>5.1f}% RR={s_tr['rr']:>5.2f}")
        print(f"  test:  n={s_te['n']:>4} daily={s_te['daily_net']:>+.4f}% avg_gross={s_te['avg_gross']:>+.4f}% {gate_te} WR={s_te['wr_pct']:>5.1f}% RR={s_te['rr']:>5.2f}")
        tt_pass = (s_tr['daily_net'] > 0) and (s_te['daily_net'] > 0)
    print(f"  Train/Test both > 0: {'PASS' if tt_pass else 'FAIL'}")

    # FINAL
    print(f"\n{'='*100}\nTrade-Tape R2 FINAL VERDICT\n{'='*100}")
    print(f"  Vacuity gate:                    PASS ({sig_per_day:.3f}/day, {len(sigs_full)} signals)")
    print(f"  Test 1 (WF ≥3/5):                {'PASS' if wf_pass else 'FAIL'}  ({wf_pos}/5)")
    print(f"  Test 2 (Bootstrap ≥50%):         {'PASS' if bs_pass else 'FAIL'}  ({bs_pos_rate:.4f})")
    print(f"  Test 3 (Train+Test sign-agree):  {'PASS' if tt_pass else 'FAIL'}")
    all_pass = wf_pass and bs_pass and tt_pass
    print(f"  OVERALL: {'ALL 3 PASS — call advisor before any claim' if all_pass else 'FAIL'}")
    print(f"\n  Friction-floor pattern across rounds (advisor visibility):")
    print(f"    R41 (OHLCV):       avg_gross +0.0323% / +0.0342% (train/test) < 0.07%")
    print(f"    R1 (trade-tape):   avg_gross +0.0288% / +0.0499% (train/test) < 0.07%")
    if s_tr and s_te:
        print(f"    R2 (this round):   avg_gross {s_tr['avg_gross']:+.4f}% / {s_te['avg_gross']:+.4f}% (train/test) " +
              ("→ SAME basement" if max(s_tr['avg_gross'], s_te['avg_gross']) < 0.07 else "→ DIFFERENT — investigate"))

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': 'f584711',
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
    p = ROOT / 'results' / f'trade_tape_r2_oos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
