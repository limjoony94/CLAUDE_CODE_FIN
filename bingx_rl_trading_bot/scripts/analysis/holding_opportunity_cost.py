#!/usr/bin/env python3
"""
Holding Time + Missed Signal Analysis — TP/SL 범위와 기회비용

현재 TP 2.0% / SL 3.0%에서:
1. 트레이드당 평균 홀딩 시간
2. 홀딩 중 발생한 시그널 수 (놓친 기회)
3. 다양한 TP/SL에서의 비교
"""

import os
import sys
import json
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')


def load_data():
    df = pd.read_csv(DATA_FILE)
    with open(PATTERNS_FILE) as f:
        dp = json.load(f)
    return df, dp


def classify_all(df):
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    bodies = abs(closes - opens)
    avg_body = pd.Series(bodies).rolling(AVG_BODY_WINDOW, min_periods=1).mean().values
    types = []
    for i in range(len(df)):
        row = df.iloc[i]
        types.append(classify_candle(row, avg_body[i]).value)
    return types, opens, highs, lows, closes


def find_signals(types, long_pats, short_pats, n_bars):
    signals = []
    for i in range(2, n_bars - 1):
        pat3 = f'{types[i-2]}-{types[i-1]}-{types[i]}'
        if pat3 in long_pats:
            signals.append((i + 1, f'{pat3}_LONG', 'LONG'))
        if pat3 in short_pats:
            signals.append((i + 1, f'{pat3}_SHORT', 'SHORT'))
    return signals


def simulate_trades(signals, opens, highs, lows, tp_pct, sl_pct, n_bars):
    results = []
    sig_bars = [s[0] for s in signals]

    for sig_idx, (entry_bar, pat_key, direction) in enumerate(signals):
        if entry_bar >= n_bars:
            continue
        entry = opens[entry_bar]
        if direction == 'LONG':
            tp_price = entry * (1 + tp_pct / 100)
            sl_price = entry * (1 - sl_pct / 100)
        else:
            tp_price = entry * (1 - tp_pct / 100)
            sl_price = entry * (1 + sl_pct / 100)

        exit_bar = None
        exit_reason = 'TIMEOUT'
        for j in range(entry_bar, min(entry_bar + MAX_BARS, n_bars)):
            if direction == 'LONG':
                tp_dist = abs(tp_price - opens[j])
                sl_dist = abs(sl_price - opens[j])
                if highs[j] >= tp_price and lows[j] <= sl_price:
                    exit_bar = j
                    exit_reason = 'TP' if tp_dist <= sl_dist else 'SL'
                    break
                elif highs[j] >= tp_price:
                    exit_bar = j; exit_reason = 'TP'; break
                elif lows[j] <= sl_price:
                    exit_bar = j; exit_reason = 'SL'; break
            else:
                tp_dist = abs(tp_price - opens[j])
                sl_dist = abs(sl_price - opens[j])
                if lows[j] <= tp_price and highs[j] >= sl_price:
                    exit_bar = j
                    exit_reason = 'TP' if tp_dist <= sl_dist else 'SL'
                    break
                elif lows[j] <= tp_price:
                    exit_bar = j; exit_reason = 'TP'; break
                elif highs[j] >= sl_price:
                    exit_bar = j; exit_reason = 'SL'; break

        if exit_bar is None:
            continue

        holding_bars = exit_bar - entry_bar + 1
        holding_hours = holding_bars * 5 / 60

        # Count missed signals during holding
        missed = 0
        for other_idx in range(sig_idx + 1, len(signals)):
            other_bar = signals[other_idx][0]
            if other_bar > exit_bar:
                break
            if other_bar >= entry_bar:
                missed += 1

        # PnL
        if exit_reason == 'TP':
            pnl = tp_pct * LEVERAGE - FEE_PCT
        elif exit_reason == 'SL':
            pnl = -(sl_pct * LEVERAGE + FEE_PCT)
        else:
            continue

        results.append({
            'pat': pat_key,
            'entry_bar': entry_bar,
            'exit_bar': exit_bar,
            'holding_bars': holding_bars,
            'holding_hours': holding_hours,
            'exit_reason': exit_reason,
            'missed_signals': missed,
            'pnl': pnl
        })

    return results


def portfolio_sim(signals, opens, highs, lows, tp_pct, sl_pct, n_bars):
    """1-position-at-a-time 포트폴리오 시뮬레이션"""
    trades = []
    current_end = -1

    for sig_idx, (entry_bar, pat_key, direction) in enumerate(signals):
        if entry_bar >= n_bars or entry_bar <= current_end:
            continue

        entry = opens[entry_bar]
        if direction == 'LONG':
            tp_price = entry * (1 + tp_pct / 100)
            sl_price = entry * (1 - sl_pct / 100)
        else:
            tp_price = entry * (1 - tp_pct / 100)
            sl_price = entry * (1 + sl_pct / 100)

        exit_bar = None
        exit_reason = 'TIMEOUT'
        for j in range(entry_bar, min(entry_bar + MAX_BARS, n_bars)):
            if direction == 'LONG':
                tp_dist = abs(tp_price - opens[j])
                sl_dist = abs(sl_price - opens[j])
                if highs[j] >= tp_price and lows[j] <= sl_price:
                    exit_bar = j
                    exit_reason = 'TP' if tp_dist <= sl_dist else 'SL'
                    break
                elif highs[j] >= tp_price:
                    exit_bar = j; exit_reason = 'TP'; break
                elif lows[j] <= sl_price:
                    exit_bar = j; exit_reason = 'SL'; break
            else:
                tp_dist = abs(tp_price - opens[j])
                sl_dist = abs(sl_price - opens[j])
                if lows[j] <= tp_price and highs[j] >= sl_price:
                    exit_bar = j
                    exit_reason = 'TP' if tp_dist <= sl_dist else 'SL'
                    break
                elif lows[j] <= tp_price:
                    exit_bar = j; exit_reason = 'TP'; break
                elif highs[j] >= sl_price:
                    exit_bar = j; exit_reason = 'SL'; break

        if exit_bar is None:
            continue

        current_end = exit_bar
        if exit_reason == 'TP':
            pnl = tp_pct * LEVERAGE - FEE_PCT
        else:
            pnl = -(sl_pct * LEVERAGE + FEE_PCT)

        trades.append({
            'entry_bar': entry_bar,
            'exit_bar': exit_bar,
            'holding_bars': exit_bar - entry_bar + 1,
            'exit_reason': exit_reason,
            'pnl': pnl
        })

    return trades


def main():
    df, dp = load_data()
    types, opens, highs, lows, closes = classify_all(df)
    long_pats = set(dp['patterns']['long'])
    short_pats = set(dp['patterns']['short'])
    n_bars = len(df)
    total_hours = n_bars * 5 / 60

    signals = find_signals(types, long_pats, short_pats, n_bars)

    print('=' * 65)
    print('홀딩 시간 + 놓친 시그널 분석 (TP 2.0% / SL 3.0%)')
    print('=' * 65)

    # ── Phase 1: 현재 설정 분석 ──
    results = simulate_trades(signals, opens, highs, lows, 2.0, 3.0, n_bars)
    tr_df = pd.DataFrame(results)

    print(f'\n총 시그널: {len(signals)}개 / 270일 ({total_hours:.0f}시간)')
    print(f'시그널/일: {len(signals)/total_hours*24:.1f}')
    print(f'38시간 기대 시그널: {len(signals)/total_hours*38:.1f}개')
    print(f'\n완료된 트레이드(timeout 제외): {len(tr_df)}')

    print('\n--- 홀딩 시간 (TP 2.0 / SL 3.0) ---')
    print(f'  평균: {tr_df["holding_hours"].mean():.1f}시간 ({tr_df["holding_bars"].mean():.0f}봉)')
    print(f'  중앙값: {tr_df["holding_hours"].median():.1f}시간 ({tr_df["holding_bars"].median():.0f}봉)')
    print(f'  P75: {tr_df["holding_hours"].quantile(0.75):.1f}시간')
    print(f'  P90: {tr_df["holding_hours"].quantile(0.90):.1f}시간')
    print(f'  P95: {tr_df["holding_hours"].quantile(0.95):.1f}시간')
    print(f'  최대: {tr_df["holding_hours"].max():.1f}시간 ({tr_df["holding_bars"].max():.0f}봉)')

    for reason in ['TP', 'SL']:
        sub = tr_df[tr_df['exit_reason'] == reason]
        if len(sub) > 0:
            print(f'\n--- {reason} ({len(sub)}건, {len(sub)/len(tr_df)*100:.1f}%) ---')
            print(f'  평균 홀딩: {sub["holding_hours"].mean():.1f}시간 ({sub["holding_bars"].mean():.0f}봉)')
            print(f'  놓친 시그널 평균: {sub["missed_signals"].mean():.1f}개')
            print(f'  놓친 시그널 중앙값: {sub["missed_signals"].median():.0f}개')
            print(f'  놓친 시그널 최대: {sub["missed_signals"].max()}개')

    print('\n--- 놓친 시그널 분포 ---')
    for n in [0, 1, 2, 3, 5, 10, 20]:
        pct = (tr_df['missed_signals'] <= n).mean() * 100
        print(f'  {n}개 이하: {pct:.1f}%')

    print(f'\n전체 평균 놓친 시그널: {tr_df["missed_signals"].mean():.1f}개/트레이드')
    total_missed = tr_df['missed_signals'].sum()
    print(f'총 놓친 시그널: {total_missed}개 / {len(signals)}개')

    # 현재 포지션
    long_holding = tr_df[tr_df['holding_bars'] >= 400]
    print(f'\n--- 현재 포지션 참고 (456봉+ 홀딩) ---')
    print(f'400봉+ 트레이드: {len(long_holding)} / {len(tr_df)} ({len(long_holding)/len(tr_df)*100:.1f}%)')
    if len(long_holding) > 0:
        print(f'이 경우 놓친 시그널: 평균 {long_holding["missed_signals"].mean():.1f}개, 최대 {long_holding["missed_signals"].max()}개')

    # ── Phase 2: 다양한 TP/SL 비교 ──
    print('\n' + '=' * 65)
    print('Phase 2: TP/SL 크기별 비교 (포트폴리오 1-at-a-time)')
    print('=' * 65)

    configs = [
        (0.5, 0.75, 'Tight'),
        (0.8, 1.2,  'Small'),
        (1.0, 1.5,  'Medium'),
        (1.4, 2.0,  'Default v1.27'),
        (2.0, 3.0,  'Current v1.28'),
        (2.5, 3.5,  'Wider'),
        (3.0, 4.0,  'Very Wide'),
    ]

    print(f'\n{"TP/SL":>10} {"Label":>14} {"Trades":>7} {"WR%":>6} {"AvgHold":>10} '
          f'{"MedHold":>10} {"P95Hold":>10} {"Missed/T":>9} {"PnL%":>8} {"PnL/MDD":>8}')
    print('-' * 105)

    for tp, sl, label in configs:
        ptrades = portfolio_sim(signals, opens, highs, lows, tp, sl, n_bars)
        if len(ptrades) == 0:
            continue
        pt_df = pd.DataFrame(ptrades)

        # Also get missed signal info from individual trades
        ind_results = simulate_trades(signals, opens, highs, lows, tp, sl, n_bars)
        ind_df = pd.DataFrame(ind_results)

        wr = (pt_df['exit_reason'] == 'TP').mean() * 100
        avg_hold = ind_df['holding_hours'].mean() if len(ind_df) > 0 else 0
        med_hold = ind_df['holding_hours'].median() if len(ind_df) > 0 else 0
        p95_hold = ind_df['holding_hours'].quantile(0.95) if len(ind_df) > 0 else 0
        missed_avg = ind_df['missed_signals'].mean() if len(ind_df) > 0 else 0

        # Compound PnL
        equity = 100.0
        peak = 100.0
        mdd = 0.0
        for _, t in pt_df.iterrows():
            equity *= (1 + t['pnl'] / 100)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > mdd:
                mdd = dd

        total_pnl = equity - 100
        pnl_mdd = total_pnl / mdd if mdd > 0 else float('inf')

        be_wr = sl / (tp + sl) * 100

        tpsl_str = f'{tp}/{sl}'
        print(f'{tpsl_str:>10} {label:>14} {len(pt_df):>7} {wr:>5.1f}% '
              f'{avg_hold:>8.1f}h {med_hold:>8.1f}h {p95_hold:>8.1f}h '
              f'{missed_avg:>8.1f} {total_pnl:>7.1f}% {pnl_mdd:>7.1f}x')

    # ── Phase 3: 시간 점유율 ──
    print('\n' + '=' * 65)
    print('Phase 3: 시장 시간 점유율 (포트폴리오 기준)')
    print('=' * 65)

    for tp, sl, label in configs:
        ptrades = portfolio_sim(signals, opens, highs, lows, tp, sl, n_bars)
        if len(ptrades) == 0:
            continue
        pt_df = pd.DataFrame(ptrades)
        bars_in_trade = pt_df['holding_bars'].sum()
        occupancy = bars_in_trade / n_bars * 100
        idle_pct = 100 - occupancy
        print(f'  {tp}/{sl} ({label:>14}): 포지션 보유 {occupancy:.1f}% | 대기 {idle_pct:.1f}% | 트레이드 {len(pt_df)}개')


if __name__ == '__main__':
    main()
