"""N8 — 4 audits to determine if +41%/720d is real edge or artifact.

Audit 1: Path-period dependence — 4 비겹침 180d windows
Audit 2: yfinance forward-fill weekend artifact
Audit 3: LONG-bias artifact (BTC buy-and-hold + shuffle test)  ← 가장 중요
Audit 4: Bootstrap interpretation already done

Goal: distinguish B-1 (real edge) / B-2 (LONG-bias artifact) / B-3 (cherry-pick by period)
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c2_design.n8_macro_regime_bt import fetch_btc_daily, fetch_macro, simulate, LOCKED


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / 'results'


def main():
    print('=' * 100)
    print('N8 — 4 Audits (real edge vs LONG-bias artifact vs cherry-pick)')
    print('=' * 100)

    btc = fetch_btc_daily()
    macro = fetch_macro()
    trades, df = simulate(btc, macro)
    df_t = pd.DataFrame(trades)
    df_t['close_ts'] = pd.to_datetime(df_t['close_ts'])
    df_t['enter_ts'] = pd.to_datetime(df_t['enter_ts'])

    # ============================================================
    # AUDIT 3 — LONG-bias artifact (most important)
    # ============================================================
    print('\n=== AUDIT 3 — LONG-bias artifact ===')
    btc_bh_ret = (btc.iloc[-1] / btc.iloc[0] - 1) * 100
    print(f'  BTC buy-and-hold 720d return: {btc_bh_ret:+.2f}%')

    days_long = (df['regime'] == 'RISK_ON').sum()
    days_short = df['regime'].isin(['RISK_OFF', 'USD_STRONG']).sum()
    days_neutral = (df['regime'] == 'NEUTRAL').sum()
    days_total = len(df)
    print(f'  RISK_ON days: {days_long} ({days_long/days_total*100:.1f}%)')
    print(f'  SHORT days (RISK_OFF + USD_STRONG): {days_short} ({days_short/days_total*100:.1f}%)')
    print(f'  NEUTRAL days: {days_neutral} ({days_neutral/days_total*100:.1f}%)')

    net_long_exposure = (days_long - days_short) / days_total
    expected_from_exposure = net_long_exposure * btc_bh_ret
    actual_cum_pct = float(df_t['net_pnl_pct'].sum())
    timing_edge = actual_cum_pct - expected_from_exposure
    print(f'  Net LONG exposure: {net_long_exposure:.3f}')
    print(f'  Expected from exposure × buy-hold: {expected_from_exposure:+.2f}%')
    print(f'  Actual N8 cum_net: {actual_cum_pct:+.2f}%')
    print(f'  Timing edge (actual - expected): {timing_edge:+.2f}%')
    if abs(timing_edge) < abs(expected_from_exposure) * 0.3:
        print(f'  → ⚠️  Timing edge < 30% of exposure-derived. LONG-bias artifact LIKELY.')
    elif timing_edge > 0 and timing_edge > 5:
        print(f'  → ✅ Significant timing edge (>5pp above buy-hold-weighted).')
    else:
        print(f'  → 🟡 Borderline. Further audit needed.')

    # Shuffle test
    print(f'\n  Shuffle test (regime randomized, 100 iter):')
    rng = np.random.default_rng(42)
    shuffle_cums = []
    for it in range(100):
        df_shuffle = df.copy()
        df_shuffle['regime'] = df['regime'].sample(frac=1, random_state=it).values
        # Re-simulate with shuffled regime
        capital = LOCKED['capital_usd']
        fric_per_round = 2 * (LOCKED['maker_fric_pct'] + LOCKED['slippage_pct']) / 100 * capital
        active = None
        cum = 0.0
        for ts, row in df_shuffle.iterrows():
            regime = row['regime']
            btc_p = row['BTC']
            if pd.isna(btc_p): continue
            target = None
            if regime == 'RISK_ON': target = 'LONG'
            elif regime in ('RISK_OFF', 'USD_STRONG'): target = 'SHORT'
            if active is None and target is not None:
                active = {'side': target, 'enter_price': btc_p}
            elif active is not None and target != active['side']:
                d_ret = (btc_p - active['enter_price']) / active['enter_price']
                pnl = capital * d_ret if active['side'] == 'LONG' else -capital * d_ret
                cum += (pnl - fric_per_round) / capital * 100
                active = {'side': target, 'enter_price': btc_p} if target else None
        shuffle_cums.append(cum)
    shuffle_arr = np.array(shuffle_cums)
    p_actual_beats_shuffle = float((shuffle_arr < actual_cum_pct).mean())
    print(f'  Shuffled cum_pct: mean={shuffle_arr.mean():+.2f}%, '
          f'median={np.median(shuffle_arr):+.2f}%, '
          f'std={shuffle_arr.std():.2f}%')
    print(f'  Actual {actual_cum_pct:+.2f}% percentile vs shuffle: '
          f'{p_actual_beats_shuffle*100:.1f}% windows beaten')
    if p_actual_beats_shuffle > 0.95:
        print(f'  → ✅ Actual significantly beats random regime (p<0.05). Real timing.')
    elif p_actual_beats_shuffle < 0.50:
        print(f'  → 🔴 Random regime achieves similar/better. Strategy is LONG-bias artifact.')
    else:
        print(f'  → 🟡 Borderline. Edge weak.')

    # ============================================================
    # AUDIT 1 — Period split
    # ============================================================
    print('\n=== AUDIT 1 — 4 비겹침 180d windows ===')
    span_min = df.index.min()
    span_max = df.index.max()
    windows = []
    for i in range(4):
        start = span_min + pd.Timedelta(days=i * 180)
        end = start + pd.Timedelta(days=180)
        if end > span_max:
            end = span_max
        windows.append((f'W{i+1}', start, end))
    for name, s, e in windows:
        in_win = df_t[(df_t['close_ts'] >= s) & (df_t['close_ts'] < e)]
        cum_w = float(in_win['net_pnl_pct'].sum())
        n_w = len(in_win)
        print(f'  {name} {s.date()} → {e.date()}  trades={n_w}  cum={cum_w:+.2f}%')

    n_pos_windows = sum(1 for name, s, e in windows
                         if float(df_t[(df_t['close_ts'] >= s) & (df_t['close_ts'] < e)]['net_pnl_pct'].sum()) > 0)
    print(f'  Positive windows: {n_pos_windows}/4')
    if n_pos_windows == 4:
        print(f'  → ✅ All 4 windows positive. Robust to time period.')
    elif n_pos_windows >= 2:
        print(f'  → 🟡 Partial robustness ({n_pos_windows}/4).')
    else:
        print(f'  → 🔴 Cherry-pick by period ({n_pos_windows}/4).')

    # ============================================================
    # AUDIT 2 — Weekend distribution
    # ============================================================
    print('\n=== AUDIT 2 — Weekend forward-fill artifact ===')
    df_t['close_weekday'] = df_t['close_ts'].dt.day_name()
    weekday_stats = df_t.groupby('close_weekday')['net_pnl_pct'].agg(['count', 'mean'])
    print(weekday_stats.to_string())
    weekend_trades = df_t[df_t['close_weekday'].isin(['Saturday', 'Sunday'])]
    weekday_trades = df_t[~df_t['close_weekday'].isin(['Saturday', 'Sunday'])]
    we_n = len(weekend_trades); we_mean = weekend_trades['net_pnl_pct'].mean() if we_n > 0 else 0
    wd_n = len(weekday_trades); wd_mean = weekday_trades['net_pnl_pct'].mean() if wd_n > 0 else 0
    print(f'  Weekend trades: n={we_n}, mean PnL={we_mean:+.2f}%')
    print(f'  Weekday trades: n={wd_n}, mean PnL={wd_mean:+.2f}%')
    if we_n > 0 and abs(we_mean - wd_mean) > 1.0:
        print(f'  → ⚠️  Weekend PnL diverges by {abs(we_mean-wd_mean):.2f}pp. Possible forward-fill artifact.')
    else:
        print(f'  → ✅ No significant weekend bias.')

    # ============================================================
    # Final verdict
    # ============================================================
    print('\n' + '=' * 100)
    print('VERDICT')
    print('=' * 100)
    if p_actual_beats_shuffle < 0.50 or abs(timing_edge) < abs(expected_from_exposure) * 0.3:
        print('  🔴 B-2: LONG-bias ARTIFACT confirmed. N8 cum_pct ≈ random regime + BTC strength.')
        print('  → N8 폐기. envelope-empty 결론 강화.')
    elif n_pos_windows < 3:
        print(f'  🔴 B-3: Cherry-pick by period ({n_pos_windows}/4 positive). N8 specific to 2024-2026 conditions.')
        print('  → N8 폐기.')
    elif p_actual_beats_shuffle > 0.95 and n_pos_windows >= 3:
        print('  🟢 B-1: REAL EDGE. Significantly beats random regime, robust across periods.')
        print('  → N8 develop path 가능. frequency expansion 검토.')
    else:
        print('  🟡 BORDERLINE. Edge weak.')
        print('  → 정직한 결론: marginal. Frequency expansion 가치 미정.')

    out_path = RESULTS / f'n8_audits_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'date': datetime.now(timezone.utc).isoformat(),
            'audit_3_long_bias': {
                'btc_buy_hold_pct': btc_bh_ret,
                'days_long_pct': days_long / days_total * 100,
                'days_short_pct': days_short / days_total * 100,
                'net_long_exposure': net_long_exposure,
                'expected_from_exposure_pct': expected_from_exposure,
                'actual_cum_pct': actual_cum_pct,
                'timing_edge_pct': timing_edge,
                'shuffle_mean_pct': float(shuffle_arr.mean()),
                'shuffle_median_pct': float(np.median(shuffle_arr)),
                'shuffle_std_pct': float(shuffle_arr.std()),
                'p_actual_beats_shuffle': p_actual_beats_shuffle,
            },
            'audit_1_period_split': {
                'n_pos_windows_of_4': n_pos_windows,
                'windows_summary': [
                    {'name': name,
                     'start': str(s), 'end': str(e),
                     'n_trades': int(len(df_t[(df_t['close_ts'] >= s) & (df_t['close_ts'] < e)])),
                     'cum_pct': float(df_t[(df_t['close_ts'] >= s) & (df_t['close_ts'] < e)]['net_pnl_pct'].sum())}
                    for name, s, e in windows
                ],
            },
            'audit_2_weekend': {
                'weekend_n': int(we_n), 'weekend_mean': float(we_mean),
                'weekday_n': int(wd_n), 'weekday_mean': float(wd_mean),
            },
        }, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
