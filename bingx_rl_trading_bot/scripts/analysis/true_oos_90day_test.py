#!/usr/bin/env python3
"""
True OOS Validation — 패턴 발굴에 사용하지 않은 기간으로 검증
=============================================================
훈련 데이터: 2025-05-05 ~ 2026-03-04 (303일)
테스트: 이 기간 이전/이후의 데이터를 BingX에서 다운로드
"""
import os, sys, time, warnings, json
import numpy as np
import pandas as pd
import yaml, ccxt
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.scanner.pattern_scanner import (
    build_signal_index, portfolio_npos, calc_stats_compound,
    compute_atr_ratio, compute_ema_slope, DEFAULT_TP_DECAY_RATE,
)

PATTERNS = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'dynamic_patterns.json')
API_KEYS = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_keys.yaml')
OUTPUT = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'true_oos_validation.json')

CASCADE_OFF = {
    'cascade_tighten_pct': 0, 'preemptive_cascade_pct': 999, 'preemptive_tighten_pct': 0,
    'direction_cap': 6, 'agg_risk_counter': 10.0, 'agg_risk_with': 15.0,
    'timeout_bars': 288, 'tp_decay_rate': DEFAULT_TP_DECAY_RATE,
    'vol_adapt_clamp': (0.0, 0.0),
    'momentum_threshold': 1.5, 'momentum_lookback': 3, 'momentum_cooldown': 12,
}


def download(start, end):
    with open(API_KEYS) as f:
        cfg = yaml.safe_load(f)
    k = cfg['bingx']['mainnet']
    ex = ccxt.bingx({'apiKey': k['api_key'], 'secret': k['secret_key'],
                     'options': {'defaultType': 'swap'}})
    s_ms = int(pd.Timestamp(start).timestamp() * 1000)
    e_ms = int(pd.Timestamp(end).timestamp() * 1000)
    all_c = []; since = s_ms
    while since < e_ms:
        c = ex.fetch_ohlcv('BTC/USDT:USDT', '5m', since=since, limit=1000)
        if not c: break
        all_c.extend([x for x in c if x[0] < e_ms])
        if c[-1][0] <= since: break
        since = c[-1][0] + 300000
        time.sleep(0.3)
    print(f"  Downloaded {len(all_c)} candles ({start} ~ {end})")
    return all_c


def to_df(candles):
    rows = [{'timestamp': pd.to_datetime(c[0], unit='ms').strftime('%Y-%m-%d %H:%M:%S'),
             'open': c[1], 'high': c[2], 'low': c[3], 'close': c[4], 'volume': c[5]}
            for c in candles]
    df = pd.DataFrame(rows)
    df['body'] = df['close'] - df['open']
    df['body_abs'] = df['body'].abs()
    df['avg_body_20'] = df['body_abs'].rolling(20, min_periods=1).mean()
    df['candle_type'] = [classify_candle(row, row['avg_body_20']).value
                         if hasattr(classify_candle(row, row['avg_body_20']), 'value')
                         else str(classify_candle(row, row['avg_body_20']))
                         for _, row in df.iterrows()]
    return df


def test(df, label, params):
    n = len(df)
    if n < 1000:
        print(f"  {label}: insufficient data ({n} bars)")
        return None
    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    types = df['candle_type'].values
    atr = compute_atr_ratio(h, l, c, atr_period=14, window=576)
    ema = compute_ema_slope(c, period=20, lookback=5)
    sidx = build_signal_index(types, n)

    with open(PATTERNS) as f:
        details = json.load(f)['pattern_details']
    sigs = []
    for key, pd_ in details.items():
        pat, d = pd_['pattern'], pd_['direction']
        tp = pd_.get('exc_stats', {}).get('mfe_median') or pd_['tp']
        sl = pd_['sl']
        for b in sidx.get(pat, []):
            sigs.append((b, pat, d, tp, sl))

    density = len(sigs) / (n / 288)
    if len(sigs) < 20:
        print(f"  {label}: too few signals ({len(sigs)})")
        return None

    start = 576
    t_list, st = portfolio_npos(sigs, o, h, l, c, n, atr, ema, start_bar=start, end_bar=n, **params)
    cs = calc_stats_compound(t_list)
    if t_list:
        w = [x['pnl_slot'] for x in t_list if x['pnl_slot'] > 0]
        lo = [x['pnl_slot'] for x in t_list if x['pnl_slot'] <= 0]
        cs['rr'] = round(np.mean(w) / abs(np.mean(lo)), 3) if w and lo else 0
        reasons = defaultdict(int)
        for x in t_list:
            reasons[x.get('reason', '?')] += 1
        cs['exit_reasons'] = dict(reasons)

    btc = (c[-1] / c[start] - 1) * 100
    print(f"\n  {label}: {n} bars ({n/288:.0f}d), {len(sigs)} sigs ({density:.1f}/d)")
    print(f"    BTC: ${c[start]:.0f} → ${c[-1]:.0f} ({btc:+.1f}%)")
    print(f"    Trades: {cs['trades']}, WR: {cs['wr']:.1f}%, PnL: {cs['pnl']:+.1f}%")
    print(f"    MDD: {cs['mdd']:.2f}%, R:R: {cs.get('rr',0):.3f}")
    if cs.get('exit_reasons'):
        print(f"    Exits: {cs['exit_reasons']}")
    cs['btc_return'] = round(btc, 1)
    cs['density'] = round(density, 1)
    cs['n_sigs'] = len(sigs)
    return cs


def main():
    print("=" * 65)
    print("TRUE OOS VALIDATION — Never-seen data periods")
    print("Patterns from: 2025-05-05 ~ 2026-03-04")
    print("Cascade: OFF (current production config)")
    print("=" * 65)

    R = {}

    # Period 1: Pre-training (2025-02 ~ 2025-05)
    print(f"\n{'─'*65}")
    print("PERIOD 1: PRE-TRAINING (2025-02-04 ~ 2025-05-04)")
    c1 = download('2025-02-04', '2025-05-04')
    if c1:
        r1 = test(to_df(c1), "Pre-training 90d", CASCADE_OFF)
        if r1: R['pre_training_90d'] = r1

    # Period 2: Deep history (2024-10 ~ 2025-01)
    print(f"\n{'─'*65}")
    print("PERIOD 2: DEEP HISTORY (2024-10-01 ~ 2025-01-31)")
    c2 = download('2024-10-01', '2025-01-31')
    if c2:
        r2 = test(to_df(c2), "Deep history 120d", CASCADE_OFF)
        if r2: R['deep_history_120d'] = r2

    # Period 3: Post-training (from existing data, last 23d)
    print(f"\n{'─'*65}")
    print("PERIOD 3: POST-TRAINING (2026-03-04 ~ 2026-03-27)")
    from scripts.scanner.pattern_scanner import load_and_classify
    df_full = load_and_classify(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_5m_270days_reclassified.csv'))
    r3 = test(df_full.iloc[-6549:].reset_index(drop=True), "Post-training 23d", CASCADE_OFF)
    if r3: R['post_training_23d'] = r3

    # Period 4: Training period itself (for reference)
    print(f"\n{'─'*65}")
    print("PERIOD 4: TRAINING (reference, 2025-05 ~ 2026-03)")
    r4 = test(df_full.iloc[:87315].reset_index(drop=True), "Training 303d", CASCADE_OFF)
    if r4: R['training_303d'] = r4

    # Summary
    print(f"\n{'='*65}")
    print("TRUE OOS SUMMARY")
    print("="*65)
    print(f"\n  {'Period':22s} | {'BTC':>7} | {'Trades':>6} {'WR':>5} {'PnL':>8} {'R:R':>5} {'MDD':>6} | {'Genuine?':>8}")
    print(f"  {'-'*80}")

    for name, r in R.items():
        if not r: continue
        is_oos = 'training' not in name
        genuine = "OOS ✅" if is_oos else "IS ref"
        flag = "✅" if r['pnl'] > 0 else "❌"
        print(f"  {flag}{name:21s} | {r['btc_return']:+6.1f}% | {r['trades']:>5} {r['wr']:4.1f}% {r['pnl']:+7.1f}% {r.get('rr',0):4.2f} {r['mdd']:5.2f}% | {genuine}")

    oos_results = {k: v for k, v in R.items() if 'training' not in k or 'post' in k}
    all_pos = all(r['pnl'] > 0 for r in oos_results.values())

    print(f"\n  OOS periods ALL profitable: {'YES ✅' if all_pos else 'NO ❌'}")
    if all_pos:
        print(f"  → System has GENUINE edge on UNSEEN data")
        print(f"  → Not overfit to training period")
    else:
        neg = [k for k, r in oos_results.items() if r['pnl'] <= 0]
        print(f"  → Negative: {neg} — edge may be period-specific")

    with open(OUTPUT, 'w') as f:
        json.dump(R, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT}")


if __name__ == '__main__':
    main()
