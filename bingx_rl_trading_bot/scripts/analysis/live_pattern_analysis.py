"""Deep pattern analysis of 15 live trades + current active."""
import json
from collections import Counter
from datetime import datetime

with open('results/c1_breakout_state.json') as f:
    d = json.load(f)

v26 = d['trade_history'][1:]  # skip old 10x bot

# Re-classify correctly with BUG#47 logic we have full data for:
# Use trade info to detect winner/loser/SL/trail
print('='*70)
print('  15거래 패턴 심층 분석')
print('='*70)
print()

# Trade categorization
zero_trail = []  # exit_price == entry_price (or very close)
win_trail = []   # pnl > 0.5%
loss_trail = []  # pnl < -0.5% AND not SL
sl_hit = []      # pnl ~ -2.5% range (SL at max SL distance)

for i, t in enumerate(v26, 2):
    ep = t['entry_price']; xp = t.get('exit_price', 0)
    diff_abs = abs(xp - ep)
    if diff_abs < 1:
        zero_trail.append((i, t))
    elif t['pnl_pct'] > 0.5:
        win_trail.append((i, t))
    elif 'SL' in t['reason'] and t['pnl_pct'] < -1.8:
        sl_hit.append((i, t))
    else:
        loss_trail.append((i, t))

print('분류 결과:')
print('  Win Trail  : {} trades — total {:+.2f}%'.format(len(win_trail), sum(t[1]['pnl_pct'] for t in win_trail)))
print('  Zero Trail : {} trades — total {:+.2f}% (수수료)'.format(len(zero_trail), sum(t[1]['pnl_pct'] for t in zero_trail)))
print('  Loss Trail : {} trades — total {:+.2f}%'.format(len(loss_trail), sum(t[1]['pnl_pct'] for t in loss_trail)))
print('  SL Hit     : {} trades — total {:+.2f}%'.format(len(sl_hit), sum(t[1]['pnl_pct'] for t in sl_hit)))
total = sum(t['pnl_pct'] for t in v26)
print('  Grand Total: {:+.2f}%'.format(total))
print()

# Direction analysis
longs = [t for t in v26 if t['direction'] == 'LONG']
shorts = [t for t in v26 if t['direction'] == 'SHORT']
long_pnl = sum(t['pnl_pct'] for t in longs)
short_pnl = sum(t['pnl_pct'] for t in shorts)
long_wins = sum(1 for t in longs if t['pnl_pct'] > 0)
short_wins = sum(1 for t in shorts if t['pnl_pct'] > 0)
print('방향별:')
print('  LONG  {} trades ({}W): PnL {:+.2f}%  ({:.0f}% WR)'.format(len(longs), long_wins, long_pnl, long_wins/len(longs)*100))
print('  SHORT {} trades ({}W): PnL {:+.2f}%  ({:.0f}% WR)'.format(len(shorts), short_wins, short_pnl, short_wins/len(shorts)*100))
print()

# Time-of-day analysis (rough - from entry_time if available)
print('보유 시간 분포:')
bars = [t.get('bars_held', 0) for t in v26]
print('  최단: {}봉 ({}시간)  평균: {:.1f}봉  최장: {}봉'.format(
    min(bars), min(bars)*15//60, sum(bars)/len(bars), max(bars)))
short_hold = [b for b in bars if b <= 5]
print('  짧은 보유(≤5봉): {}건 ({:.0f}%) — 즉시 반전 패턴'.format(len(short_hold), len(short_hold)/len(v26)*100))
print()

# Volatility cluster analysis - ATR at entry (from log would be better but let's approximate)
print('SL 거리 (변동성 프록시):')
# Recover SL distance from trade data if possible
# We only have entry/exit, not SL
# Instead, infer from max_loss pattern
print('  (SL 거리는 로그에서 분석 - 별도 확인)')
print()

# Sequential PnL pattern analysis
print('시간 순서 PnL (누적):')
cum = 0
for i, t in enumerate(v26, 2):
    cum += t['pnl_pct']
    bar = '▲' if t['pnl_pct'] > 0 else '▼'
    print('  #{:>2} {} {:+6.2f}% → 누적 {:+6.2f}%'.format(i, bar, t['pnl_pct'], cum))
print()

# Streak analysis
print('스트릭 분석:')
streaks = []
cur_streak = {'type': None, 'count': 0, 'pnl': 0}
for t in v26:
    win = t['pnl_pct'] > 0
    t_type = 'W' if win else 'L'
    if cur_streak['type'] == t_type:
        cur_streak['count'] += 1
        cur_streak['pnl'] += t['pnl_pct']
    else:
        if cur_streak['type']:
            streaks.append(dict(cur_streak))
        cur_streak = {'type': t_type, 'count': 1, 'pnl': t['pnl_pct']}
streaks.append(dict(cur_streak))

for s in streaks:
    print('  {}x{}: {:+.2f}%'.format(s['type'], s['count'], s['pnl']))
print()

# Key question: Are losses clustering?
# Compute rolling 5-trade PnL
print('5-trade 롤링 PnL:')
for i in range(len(v26) - 4):
    window = v26[i:i+5]
    wpnl = sum(t['pnl_pct'] for t in window)
    trade_nums = '#{}~#{}'.format(i+2, i+6)
    wins = sum(1 for t in window if t['pnl_pct']>0)
    print('  {}: {:+.2f}% ({}/5 win)'.format(trade_nums, wpnl, wins))
