"""
Phase 0.2 보강: 15m role 재해석 — funnel 비교 (BT 아님)
========================================================
원본 M1-A의 15m 역할 = "EMA9>EMA21 strict alignment"는 내(어시스턴트)의 해석.
사용자 spec literal = "5m, 15m 캔들을 참고하여 스캘핑" — 'alignment 필수'는 명시 없음.

이 함수는 4가지 15m role 변형의 entry frequency funnel을 동시에 측정한다.
변형은 Phase 2 BT가 아니라 frequency 측정용. 최종 spec은 funnel 결과 + advisor
권고("data-driven decision")에 따라 결정.

15m 역할 후보:
  D1 (strict align): 15m EMA9 > 15m EMA21  ← 원래 plan §3 spec
  D2 (price > slow): 15m close > 15m EMA21  ← MA crossover 대신 price relative
  D3 (no veto)     : 15m EMA9 ≥ 15m EMA21 - 0.1% buffer  ← strongly-opposing-only veto
  D4 (15m omitted) : 15m 미사용                          ← baseline (가장 loose)

선택 원칙:
  - D_per_day ≥ 3.0 (margin for min_bars_between=2 dedupe → 실제 BT ≥2/day)
  - 가장 strict한 후보 선택 (selectivity 보존)
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# 동일 indicator 함수 재사용
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from m1_entry_frequency_20260427 import (load_ohlcv, compute_ema, compute_rsi,
                                           resample_to_4h)


def main():
    print("Loading...")
    df_5m = load_ohlcv(ROOT / 'data' / 'btc_5m_720days_binance.csv')
    df_15m = load_ohlcv(ROOT / 'data' / 'btc_15m_720days.csv')
    df_1h = load_ohlcv(ROOT / 'data' / 'btc_1h_720days.csv')
    df_4h = resample_to_4h(df_1h)

    # Indicators (5m/15m/1h/4h)
    df_5m['ema9'] = compute_ema(df_5m['close'].values, 9)
    df_5m['rsi14'] = compute_rsi(df_5m['close'].values, 14)
    df_5m['body_ratio'] = (df_5m['close'] - df_5m['open']).abs() / \
        (df_5m['high'] - df_5m['low']).replace(0, np.nan)

    df_15m['ema9'] = compute_ema(df_15m['close'].values, 9)
    df_15m['ema21'] = compute_ema(df_15m['close'].values, 21)
    df_15m['D1_long'] = df_15m['ema9'] > df_15m['ema21']
    df_15m['D2_long'] = df_15m['close'] > df_15m['ema21']
    df_15m['D3_long'] = df_15m['ema9'] >= df_15m['ema21'] * 0.999   # LONG: EMA9 within 0.1% above-or-near
    df_15m['D3_short'] = df_15m['ema9'] <= df_15m['ema21'] * 1.001   # SHORT: EMA9 within 0.1% below-or-near (separate column, not negation)

    df_1h['ema20'] = compute_ema(df_1h['close'].values, 20)
    df_1h['ema50'] = compute_ema(df_1h['close'].values, 50)
    df_1h['htf_long'] = df_1h['ema20'] > df_1h['ema50']

    df_4h['ema50'] = compute_ema(df_4h['close'].values, 50)
    df_4h['htf_long'] = df_4h['close'] > df_4h['ema50']

    # MTF causal merge
    df_5m['close_time'] = df_5m['timestamp'] + pd.Timedelta(minutes=5)

    def merge_htf(df_target, df_htf, htf_minutes, cols):
        df_htf = df_htf.copy()
        df_htf['close_time'] = df_htf['timestamp'] + pd.Timedelta(minutes=htf_minutes)
        df_htf = df_htf[['close_time'] + cols].sort_values('close_time')
        return pd.merge_asof(df_target.sort_values('close_time'), df_htf, on='close_time', direction='backward')

    df_5m = merge_htf(df_5m, df_1h.rename(columns={'htf_long': 'h1_long'}), 60, ['h1_long'])
    df_5m = merge_htf(df_5m, df_4h.rename(columns={'htf_long': 'h4_long'}), 240, ['h4_long'])
    df_5m = merge_htf(df_5m, df_15m, 15, ['D1_long', 'D2_long', 'D3_long', 'D3_short'])
    df_5m = df_5m.sort_values('timestamp').reset_index(drop=True)

    n = len(df_5m)
    rsi = df_5m['rsi14'].values
    close = df_5m['close'].values
    ema9_5m = df_5m['ema9'].values
    body_ratio = df_5m['body_ratio'].values

    h1_long = df_5m['h1_long'].fillna(False).astype(bool).values
    h4_long = df_5m['h4_long'].fillna(False).astype(bool).values
    h1_known = ~df_5m['h1_long'].isna()
    h4_known = ~df_5m['h4_long'].isna()

    rsi_min_lb = np.array([np.nan if i < 3 else rsi[i - 3:i].min() for i in range(n)])
    rsi_max_lb = np.array([np.nan if i < 3 else rsi[i - 3:i].max() for i in range(n)])

    long_rsi_cross = (rsi > 40) & (rsi_min_lb <= 40)
    short_rsi_cross = (rsi < 60) & (rsi_max_lb >= 60)
    body_ok = body_ratio > 0.4
    close_above_ema9 = close > ema9_5m
    close_below_ema9 = close < ema9_5m

    # A,B,C 공통
    A_long = h1_long & h4_long
    A_short = ((~h1_long) & h1_known.values) & ((~h4_long) & h4_known.values)

    B_long = A_long & long_rsi_cross
    B_short = A_short & short_rsi_cross

    C_long = B_long & body_ok & close_above_ema9
    C_short = B_short & body_ok & close_below_ema9

    # 15m role variants — direction-specific (symmetric)
    def d_filter_long(role):
        """LONG: 15m predicate (LONG side) true."""
        col = df_5m[f'{role}_long'].fillna(False).astype(bool).values
        known = (~df_5m[f'{role}_long'].isna()).values
        return col & known

    def d_filter_short(role):
        """SHORT: 15m predicate (SHORT side) true.
        D1/D2 SHORT = mirror of LONG via negation (binary alignment).
        D3 SHORT = separate column with symmetric buffer (NOT negation of D3_long)."""
        if role == 'D3':
            col = df_5m['D3_short'].fillna(False).astype(bool).values
            known = (~df_5m['D3_short'].isna()).values
            return col & known
        col_known = (~df_5m[f'{role}_long'].isna()).values
        col_actual = df_5m[f'{role}_long'].fillna(False).astype(bool).values
        return (~col_actual) & col_known

    usable_mask = (~pd.isna(rsi)) & (~pd.isna(ema9_5m)) & h1_known.values & h4_known.values
    n_usable = int(usable_mask.sum())
    days = n_usable * 5 / 60 / 24

    variants = []
    for label, role in [('D1 (15m EMA9>EMA21 strict)', 'D1'),
                        ('D2 (15m close>EMA21)', 'D2'),
                        ('D3 (15m EMA9≥EMA21·0.999, 0.1% buffer)', 'D3'),
                        ('D4 (15m omitted)', None)]:
        if role is None:
            d_long = C_long
            d_short = C_short
        else:
            d_long = C_long & d_filter_long(role)
            d_short = C_short & d_filter_short(role)
        n_long = int((d_long & usable_mask).sum())
        n_short = int((d_short & usable_mask).sum())
        n_total = int(((d_long | d_short) & usable_mask).sum())
        variants.append({
            'label': label,
            'role': role or 'OMITTED',
            'long_n': n_long,
            'short_n': n_short,
            'total_n': n_total,
            'long_per_day': round(n_long / days, 3),
            'short_per_day': round(n_short / days, 3),
            'total_per_day': round(n_total / days, 3),
        })

    # Funnel A,B,C 출력
    n_A = int(((A_long | A_short) & usable_mask).sum())
    n_B = int(((B_long | B_short) & usable_mask).sum())
    n_C = int(((C_long | C_short) & usable_mask).sum())

    print(f"\nUsable: {n_usable:,} bars / {days:.1f} days\n")
    print(f"  A. trend filter (1h+4h): {n_A:,} ({n_A/days:.2f}/day)")
    print(f"  B. + RSI cross         : {n_B:,} ({n_B/days:.2f}/day)")
    print(f"  C. + body + EMA9       : {n_C:,} ({n_C/days:.2f}/day)\n")

    print("=== 15m role 4 variants ===")
    print(f"{'role':<46} {'LONG/d':>7} {'SHORT/d':>8} {'TOTAL/d':>9} {'verdict':>10}")
    for v in variants:
        target_after_dedupe = v['total_per_day'] * 0.85  # 추정: dedupe로 ~15% 감소
        verdict = ('PASS' if target_after_dedupe >= 2.0 else
                   'BORDERLINE' if target_after_dedupe >= 1.5 else 'FAIL')
        print(f"{v['label']:<46} {v['long_per_day']:>7.2f} {v['short_per_day']:>8.2f} "
              f"{v['total_per_day']:>9.2f} {verdict:>10}")

    print("\n=== 권고 ===")
    print("Selectivity 보존 + criterion 7 통과 → D_per_day ≥ 2.5 (dedupe 후 ≥ 2.0)")
    pass_variants = [v for v in variants if v['total_per_day'] * 0.85 >= 2.0]
    if pass_variants:
        chosen = pass_variants[0]  # 가장 strict한 첫 번째 후보
        print(f"  CHOICE: {chosen['label']}")
        print(f"          {chosen['total_per_day']:.2f}/day raw → 추정 {chosen['total_per_day']*0.85:.2f}/day post-dedupe")
    else:
        print("  ALL FAIL — entry trigger 자체 (RSI cross + body + EMA9) 가 너무 strict.")
        print("  → 사용자 보고 필요")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'usable_days': days,
        'common_funnel': {'A': n_A, 'B': n_B, 'C': n_C,
                          'A_per_day': round(n_A/days, 3),
                          'B_per_day': round(n_B/days, 3),
                          'C_per_day': round(n_C/days, 3)},
        'variants_15m_role': variants,
    }
    p = ROOT / 'results' / f'm1_entry_15m_role_compare_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
