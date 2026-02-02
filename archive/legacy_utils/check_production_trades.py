#!/usr/bin/env python3
"""Check production trades and performance"""

import json
from pathlib import Path

state_file = Path(__file__).parent.parent.parent / 'results' / 'opportunity_gating_bot_4x_state.json'

with open(state_file, 'r') as f:
    state = json.load(f)

print('=' * 80)
print('프로덕션 실제 성과')
print('=' * 80)
print(f'시작 잔고: ${state["initial_balance"]:.2f}')
print(f'현재 잔고: ${state["current_balance"]:.2f}')
loss = state["current_balance"] - state["initial_balance"]
loss_pct = (state["current_balance"] / state["initial_balance"] - 1) * 100
print(f'손실: ${loss:.2f} ({loss_pct:.2f}%)')
print(f'총 거래: {len(state["trades"])}개')
print()

# 거래 분석
if state['trades']:
    print('최근 15개 거래:')
    print(f'{"No":>3s} {"Side":>5s} {"Entry":>10s} {"Exit":>10s} {"P&L":>10s} {"Exit Reason":>20s} {"Time":>16s}')
    print('-' * 80)

    for i, trade in enumerate(state['trades'][-15:], 1):
        side = trade.get('side', '?')
        entry_p = trade.get('entry_price', 0)
        exit_p = trade.get('exit_price', 0)
        pnl = trade.get('pnl', 0)
        reason = trade.get('exit_reason', '?')[:20]
        entry_time = trade.get('entry_time', '')[:16] if 'entry_time' in trade else '?'

        idx = len(state["trades"]) - 15 + i
        print(f'{idx:3d} {side:>5s} ${entry_p:9,.0f} ${exit_p:9,.0f} ${pnl:9.2f} {reason:>20s} {entry_time:>16s}')

print()
print('-' * 80)

# 통계
long_trades = [t for t in state['trades'] if t.get('side') == 'LONG']
short_trades = [t for t in state['trades'] if t.get('side') == 'SHORT']
wins = [t for t in state['trades'] if t.get('pnl', 0) > 0]
losses = [t for t in state['trades'] if t.get('pnl', 0) < 0]

print(f'LONG: {len(long_trades)}개, SHORT: {len(short_trades)}개')
print(f'승: {len(wins)}개, 패: {len(losses)}개')
print(f'승률: {len(wins) / len(state["trades"]) * 100:.1f}%')

# Stop Loss 비율
sl_trades = [t for t in state['trades'] if 'Stop Loss' in t.get('exit_reason', '')]
print(f'Stop Loss: {len(sl_trades)}개 ({len(sl_trades) / len(state["trades"]) * 100:.1f}%)')

print('=' * 80)
