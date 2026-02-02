"""
최근 7일 백테스트 신호 확인
"""
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Load backtest results
results_dir = Path(__file__).resolve().parents[2] / "results"
df = pd.read_csv(results_dir / "backtest_current_production_full_20251103_155037.csv")
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])

# Get last 7 days
cutoff = df['entry_time'].max() - timedelta(days=7)
df_recent = df[df['entry_time'] >= cutoff].copy()

print('=' * 80)
print('최근 7일 백테스트 거래 (신호 확률 포함)')
print('=' * 80)
print(f'기간: {cutoff.strftime("%Y-%m-%d %H:%M")} ~ {df["entry_time"].max().strftime("%Y-%m-%d %H:%M")}')
print(f'거래 수: {len(df_recent)}')
print()

for idx, row in df_recent.iterrows():
    entry_time = row['entry_time'].strftime('%Y-%m-%d %H:%M')
    side = row['side']
    long_prob = row['entry_long_prob']
    short_prob = row['entry_short_prob']
    pnl = row['pnl_usd']

    print(f'{entry_time} | {side:5s} | LONG: {long_prob:.4f} ({long_prob*100:.1f}%) | SHORT: {short_prob:.4f} ({short_prob*100:.1f}%) | P&L: ${pnl:+,.0f}')

print()
print('=' * 80)
print('신호 통계 (최근 7일)')
print('=' * 80)
print(f'LONG 평균 확률: {df_recent["entry_long_prob"].mean():.4f} ({df_recent["entry_long_prob"].mean()*100:.1f}%)')
print(f'SHORT 평균 확률: {df_recent["entry_short_prob"].mean():.4f} ({df_recent["entry_short_prob"].mean()*100:.1f}%)')
print()

print(f'LONG 거래 ({len(df_recent[df_recent["side"] == "LONG"])}건):')
for idx, row in df_recent[df_recent['side'] == 'LONG'].iterrows():
    entry_time = row['entry_time'].strftime('%Y-%m-%d %H:%M')
    long_prob = row['entry_long_prob']
    pnl = row['pnl_usd']
    exit_reason = row['exit_reason']
    print(f'  {entry_time} | LONG: {long_prob:.4f} ({long_prob*100:.1f}%) | Exit: {exit_reason:20s} | P&L: ${pnl:+,.0f}')

print()
print(f'SHORT 거래 ({len(df_recent[df_recent["side"] == "SHORT"])}건):')
for idx, row in df_recent[df_recent['side'] == 'SHORT'].iterrows():
    entry_time = row['entry_time'].strftime('%Y-%m-%d %H:%M')
    short_prob = row['entry_short_prob']
    pnl = row['pnl_usd']
    exit_reason = row['exit_reason']
    print(f'  {entry_time} | SHORT: {short_prob:.4f} ({short_prob*100:.1f}%) | Exit: {exit_reason:20s} | P&L: ${pnl:+,.0f}')

print()
print('=' * 80)
print('프로덕션과 비교')
print('=' * 80)
print()
print('프로덕션 최근 신호 (사용자 보고):')
print('  - 최근 거래: 손실 발생')
print('  - 현재 포지션: LONG 87.42%, SHORT 9.26%')
print()
print('백테스트 최근 7일:')
print(f'  - 총 거래: {len(df_recent)}건')
print(f'  - 승률: {len(df_recent[df_recent["pnl_usd"] > 0]) / len(df_recent) * 100:.1f}%')
print(f'  - 평균 LONG 확률: {df_recent["entry_long_prob"].mean()*100:.1f}%')
print(f'  - 평균 SHORT 확률: {df_recent["entry_short_prob"].mean()*100:.1f}%')
print(f'  - 총 수익: ${df_recent["pnl_usd"].sum():+,.0f}')
