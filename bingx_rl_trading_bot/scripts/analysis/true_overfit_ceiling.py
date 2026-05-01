"""True Overfit Ceiling — per-trade cherry-pick (사용자 질문 응답).

사용자: "과적합 시켰는데 왜 결과 저래? 발산해서 무한대 가까운 수익 나와야".

이전 L2 (+1.9%/day)는 "8 mechanism × 1 best config × per-day hindsight" — 약한 과적합.
진짜 무한 cherry-pick:

L5: Per-trade keep-winners-only (8 mech best config trades, net_pnl > 0만)
    매 mechanism의 winning trades sum / 720d = daily ceiling

L6: Per-day all-mechanism cherry-pick best WINNING trade
    매 day, 모든 8 mechanism의 모든 trades 중 best 1개 (winning만)
    + losing trades 모두 skip

L7: Sum of all winning trades from all 8 mechanisms
    모든 mechanism의 모든 winning trades 합산 (overlap 허용)
    = 진짜 무한 cherry-pick ceiling
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'


def load_all_trades():
    """Load all trades from each mechanism's best config."""
    from d3_portfolio_simulation import (
        BEST_CONFIGS, get_btc_1h, sim_r8b, sim_r37b, sim_r40b,
        simulate_with_signals, range_expansion_signals, volume_spike_signals,
        run_xs_momentum, load_pivot, sim_n8b, get_macro_data, run_xs_reversal,
    )

    df_1h = get_btc_1h()
    print('Building trades for each mechanism best-IS config...')

    trades_by_mech = {}
    trades_by_mech['R8b'] = sim_r8b(df_1h, BEST_CONFIGS['R8b'])
    trades_by_mech['R37b'] = sim_r37b(df_1h, BEST_CONFIGS['R37b'])
    trades_by_mech['R40b'] = sim_r40b(df_1h, BEST_CONFIGS['R40b'])
    trades_by_mech['Range'] = simulate_with_signals(df_1h, range_expansion_signals(df_1h, BEST_CONFIGS['Range']), BEST_CONFIGS['Range'])
    trades_by_mech['VolSpike'] = simulate_with_signals(df_1h, volume_spike_signals(df_1h, BEST_CONFIGS['VolSpike']), BEST_CONFIGS['VolSpike'])
    prices = load_pivot()
    trades_by_mech['R1b'] = run_xs_momentum(prices, BEST_CONFIGS['R1b'])
    trades_by_mech['R2b'] = run_xs_reversal(prices, BEST_CONFIGS['R2b'])
    macro_full = get_macro_data()
    trades_by_mech['N8b'] = sim_n8b(macro_full, BEST_CONFIGS['N8b'])

    return trades_by_mech


def main():
    print('=' * 100)
    print('True Overfit Ceiling — Per-trade cherry-pick (사용자 질문 응답)')
    print('=' * 100)

    trades_by_mech = load_all_trades()
    n_days = 721

    # Per-mechanism summary
    print('\n=== Per-mechanism trade stats ===')
    for mech, df in trades_by_mech.items():
        if len(df) == 0:
            print(f'  {mech}: no trades')
            continue
        n = len(df)
        winners = df[df['net_pnl_pct'] > 0]
        losers = df[df['net_pnl_pct'] <= 0]
        avg_w = winners['net_pnl_pct'].mean() if len(winners) > 0 else 0
        avg_l = losers['net_pnl_pct'].mean() if len(losers) > 0 else 0
        sum_w = winners['net_pnl_pct'].sum()
        sum_l = losers['net_pnl_pct'].sum()
        net = sum_w + sum_l
        print(f'  {mech}: n={n}, winners={len(winners)} ({len(winners)/n*100:.1f}%), '
              f'avg_w={avg_w:+.3f}%, avg_l={avg_l:+.3f}%, '
              f'sum_w={sum_w:+.2f}%, sum_l={sum_l:+.2f}%, net={net:+.2f}%')

    # ============================================================
    # L5: Per-trade keep-winners-only (per mechanism)
    # ============================================================
    print('\n' + '=' * 100)
    print('L5: PER-TRADE KEEP-WINNERS-ONLY (per mechanism)')
    print('=' * 100)
    print('  매 mechanism: 모든 winning trades sum, losers ignore')
    print()

    l5_per_mech_daily = {}
    for mech, df in trades_by_mech.items():
        if len(df) == 0:
            l5_per_mech_daily[mech] = 0
            continue
        winners = df[df['net_pnl_pct'] > 0]
        sum_w = winners['net_pnl_pct'].sum()
        daily = sum_w / n_days
        l5_per_mech_daily[mech] = daily
        print(f'  {mech}: winners sum = {sum_w:+.2f}%, daily = {daily:+.4f}%')

    # Total L5 (sum of all winning trades across all mechanisms)
    l5_total = sum(l5_per_mech_daily.values())
    print(f'\n  L5 TOTAL (sum across 8 mechanisms): {l5_total:+.4f}%/day')

    # ============================================================
    # L6: Per-day all-mechanism cherry-pick BEST winning trade
    # ============================================================
    print('\n' + '=' * 100)
    print('L6: PER-DAY ALL-MECHANISM CHERRY-PICK BEST WINNING TRADE')
    print('=' * 100)
    print('  매 day, 모든 mechanism trades 중 가장 큰 winner 1개만 take, losers all skip')
    print()

    # Combine all trades with mechanism labels
    all_trades_list = []
    for mech, df in trades_by_mech.items():
        if len(df) == 0:
            continue
        df = df.copy()
        df['mech'] = mech
        df['close_ts'] = pd.to_datetime(df['close_ts'], utc=True)
        df['date'] = df['close_ts'].dt.tz_convert('UTC').dt.normalize().dt.tz_localize(None)
        all_trades_list.append(df[['date', 'mech', 'gross_pct', 'net_pnl_pct']])
    all_trades_df = pd.concat(all_trades_list, ignore_index=True)
    print(f'  Total trades across 8 mechanisms: {len(all_trades_df):,}')

    # Per-day, take best WINNING trade only
    winners_only = all_trades_df[all_trades_df['net_pnl_pct'] > 0]
    print(f'  Winning trades: {len(winners_only):,} ({len(winners_only)/len(all_trades_df)*100:.1f}%)')

    daily_best = winners_only.groupby('date').apply(
        lambda g: g.nlargest(1, 'net_pnl_pct').iloc[0]['net_pnl_pct'] if len(g) > 0 else 0,
        include_groups=False,
    )
    l6_mean = daily_best.sum() / n_days  # spread across all 720d (active days only sum)
    l6_active = (daily_best > 0).sum()
    l6_active_mean = daily_best[daily_best > 0].mean()
    print(f'  Active days (any winner): {l6_active}')
    print(f'  Per active day mean: {l6_active_mean:+.4f}%')
    print(f'  Spread over 721d: {l6_mean:+.4f}%/day')

    # ============================================================
    # L7: SUM of ALL winning trades per day (모든 mechanism의 모든 winners 합산)
    # ============================================================
    print('\n' + '=' * 100)
    print('L7: SUM OF ALL WINNING TRADES PER DAY (전체 mechanism 합산)')
    print('=' * 100)
    print('  매 day, 모든 mechanism의 모든 winning trades sum, losers all skip')
    print()

    daily_sum_winners = winners_only.groupby('date')['net_pnl_pct'].sum()
    l7_total_sum = daily_sum_winners.sum()
    l7_daily = l7_total_sum / n_days
    print(f'  Total winning sum: {l7_total_sum:+.2f}%')
    print(f'  Daily mean (over 721d): {l7_daily:+.4f}%')

    # ============================================================
    # L8: SUM of ALL winning trades, NO friction
    # ============================================================
    print('\n' + '=' * 100)
    print('L8: SAME AS L7 BUT FRICTION 추가 (이미 net인 데이터에서 +0.07% 더 add)')
    print('=' * 100)
    # Note: gross_pct already > net_pnl_pct (friction baked in via -0.07-0.14)
    daily_sum_gross_winners = winners_only.groupby('date')['gross_pct'].sum()
    l8_total = daily_sum_gross_winners.sum()
    l8_daily = l8_total / n_days
    print(f'  L8 (gross instead of net): {l8_daily:+.4f}%/day')

    # ============================================================
    # VERDICT
    # ============================================================
    print('\n' + '=' * 100)
    print('TRUE OVERFIT CEILING — VERDICT')
    print('=' * 100)
    print(f'\nUser target: +0.20%/day\n')
    print(f'  L1 (sweep best per mech, in-sample):         ~+0.30%/day (single mech)')
    print(f'  L2 (8-mech per-day hindsight switcher):       +1.8975%/day')
    print(f'  L3a (8-mech fixed-weight max-mean):           +0.2338%/day')
    print(f'  L4 (weekly best-mech hindsight):              +0.9182%/day')
    print(f'  L5 (per-mech winners-only sum / 8 mech):      {l5_total:+.4f}%/day')
    print(f'  L6 (per-day BEST 1 winning trade across all): {l6_mean:+.4f}%/day')
    print(f'  L7 (per-day SUM all winning trades):          {l7_daily:+.4f}%/day')
    print(f'  L8 (L7 with gross_pct instead of net):        {l8_daily:+.4f}%/day')

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'mandate': 'true overfit ceiling — per-trade cherry-pick',
        'n_days': n_days,
        'mechanisms': list(trades_by_mech.keys()),
        'L5_per_mech_winners_only_daily': l5_per_mech_daily,
        'L5_total_daily': float(l5_total),
        'L6_best_winning_per_day_daily': float(l6_mean),
        'L6_active_days': int(l6_active),
        'L6_per_active_day_mean': float(l6_active_mean),
        'L7_sum_all_winners_daily': float(l7_daily),
        'L7_total_cumulative': float(l7_total_sum),
        'L8_sum_all_winners_gross_daily': float(l8_daily),
        'total_trades_8_mech': int(len(all_trades_df)),
        'winning_trades_8_mech': int(len(winners_only)),
        'win_rate': float(len(winners_only) / len(all_trades_df)) if len(all_trades_df) > 0 else 0,
    }
    out_path = RESULTS / f'true_overfit_ceiling_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
