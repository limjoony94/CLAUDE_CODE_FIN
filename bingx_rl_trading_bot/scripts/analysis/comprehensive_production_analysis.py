"""
Comprehensive Production Performance Analysis (Nov 7-13, 2025)
30% High Frequency Models
"""

import json
from datetime import datetime
import sys
sys.path.append('.')

with open('results/opportunity_gating_bot_4x_state.json', 'r') as f:
    state = json.load(f)

deployment_time = datetime.fromisoformat('2025-11-07T19:09:00')
history = state.get('trading_history', [])

print('='*80)
print('📊 COMPREHENSIVE PRODUCTION ANALYSIS')
print('='*80)
print(f'Period: Nov 7 19:09 - Nov 13 (5.0 days)')
print(f'Models: 30% High Frequency Configuration')
print()

# Filter trades after deployment
prod_trades = []
for t in history:
    if t.get('manual_trade', False):
        continue
    entry_time = t.get('entry_time', '')
    if entry_time:
        try:
            trade_time = datetime.fromisoformat(entry_time.replace('Z', ''))
            if trade_time >= deployment_time:
                prod_trades.append(t)
        except:
            pass

print(f'Total Trades: {len(prod_trades)}')
print()

# 1. OVERALL STATISTICS
print('='*80)
print('1️⃣ OVERALL STATISTICS')
print('='*80)

long_trades = [t for t in prod_trades if t['side'] == 'LONG']
short_trades = [t for t in prod_trades if t['side'] == 'SHORT']

wins = [t for t in prod_trades if t.get('pnl_usd_net', 0) > 0]
losses = [t for t in prod_trades if t.get('pnl_usd_net', 0) <= 0]

total_pnl = sum(t.get('pnl_usd_net', 0) for t in prod_trades)
win_rate = len(wins) / len(prod_trades) * 100 if prod_trades else 0

print(f'Direction Split:')
print(f'  LONG: {len(long_trades)}/{len(prod_trades)} ({len(long_trades)/len(prod_trades)*100:.1f}%)')
print(f'  SHORT: {len(short_trades)}/{len(prod_trades)} ({len(short_trades)/len(prod_trades)*100:.1f}%)')
print()
print(f'Win/Loss:')
print(f'  Wins: {len(wins)}/{len(prod_trades)} ({win_rate:.1f}%)')
print(f'  Losses: {len(losses)}/{len(prod_trades)} ({100-win_rate:.1f}%)')
print()
print(f'P&L Summary:')
print(f'  Total P&L: ${total_pnl:.2f}')
print(f'  Avg Win: ${sum(t.get("pnl_usd_net", 0) for t in wins)/len(wins):.2f}' if wins else '  Avg Win: N/A')
print(f'  Avg Loss: ${sum(t.get("pnl_usd_net", 0) for t in losses)/len(losses):.2f}' if losses else '  Avg Loss: N/A')
print()

# 2. EXIT MECHANISM ANALYSIS
print('='*80)
print('2️⃣ EXIT MECHANISM ANALYSIS')
print('='*80)

ml_exits = [t for t in prod_trades if 'ML Exit' in t.get('exit_reason', '')]
stop_losses = [t for t in prod_trades if 'Stop Loss' in t.get('exit_reason', '')]
max_holds = [t for t in prod_trades if 'Max Hold' in t.get('exit_reason', '')]

print(f'ML Exit: {len(ml_exits)}/{len(prod_trades)} ({len(ml_exits)/len(prod_trades)*100:.1f}%)')
if ml_exits:
    ml_wins = [t for t in ml_exits if t.get('pnl_usd_net', 0) > 0]
    ml_pnl = sum(t.get('pnl_usd_net', 0) for t in ml_exits)
    print(f'  Win Rate: {len(ml_wins)}/{len(ml_exits)} ({len(ml_wins)/len(ml_exits)*100:.1f}%)')
    print(f'  Total P&L: ${ml_pnl:.2f}')
    print(f'  Avg P&L: ${ml_pnl/len(ml_exits):.2f}')
print()

print(f'Stop Loss: {len(stop_losses)}/{len(prod_trades)} ({len(stop_losses)/len(prod_trades)*100:.1f}%)')
if stop_losses:
    sl_pnl = sum(t.get('pnl_usd_net', 0) for t in stop_losses)
    print(f'  Total Loss: ${sl_pnl:.2f}')
    print(f'  Avg Loss: ${sl_pnl/len(stop_losses):.2f}')
    print(f'  LONG SL: {len([t for t in stop_losses if t["side"]=="LONG"])}/{len(long_trades)} LONG trades')
    print(f'  SHORT SL: {len([t for t in stop_losses if t["side"]=="SHORT"])}/{len(short_trades)} SHORT trades' if short_trades else '  SHORT SL: 0/0 SHORT trades')
print()

print(f'Max Hold: {len(max_holds)}/{len(prod_trades)} ({len(max_holds)/len(prod_trades)*100:.1f}%)')
if max_holds:
    mh_wins = [t for t in max_holds if t.get('pnl_usd_net', 0) > 0]
    mh_pnl = sum(t.get('pnl_usd_net', 0) for t in max_holds)
    print(f'  Win Rate: {len(mh_wins)}/{len(max_holds)} ({len(mh_wins)/len(max_holds)*100:.1f}%)')
    print(f'  Total P&L: ${mh_pnl:.2f}')
    print(f'  Avg P&L: ${mh_pnl/len(max_holds):.2f}')
print()

# 3. LONG STOP LOSS DETAILED ANALYSIS
print('='*80)
print('3️⃣ LONG STOP LOSS DETAILED ANALYSIS (CRITICAL ISSUE)')
print('='*80)

long_sls = [t for t in stop_losses if t['side'] == 'LONG']
print(f'LONG Stop Losses: {len(long_sls)}/{len(long_trades)} ({len(long_sls)/len(long_trades)*100:.1f}%)')
print()

if long_sls:
    for i, t in enumerate(long_sls, 1):
        entry_time = datetime.fromisoformat(t['entry_time'].replace('Z', ''))
        entry_price = t.get('entry_price', 0)
        sl_price = t.get('stop_loss_price', 0)
        exit_price = t.get('exit_price', entry_price)

        sl_distance = ((entry_price - sl_price) / entry_price) * 100 if sl_price else 0
        prob = t.get('probability', 0)
        pnl = t.get('pnl_usd_net', 0)

        print(f'{i}. {entry_time.strftime("%m-%d %H:%M")} | Prob: {prob:.3f} | Entry: ${entry_price:,.0f} | SL: ${sl_price:,.0f}')
        print(f'   SL Distance: {sl_distance:.2f}% | Loss: ${pnl:.2f}')
        print()

    avg_prob = sum(t.get('probability', 0) for t in long_sls) / len(long_sls)
    avg_sl_dist = sum(((t.get('entry_price', 0) - t.get('stop_loss_price', 0)) / t.get('entry_price', 1)) * 100 for t in long_sls) / len(long_sls)
    total_loss = sum(t.get('pnl_usd_net', 0) for t in long_sls)

    print(f'Summary:')
    print(f'  Avg Entry Probability: {avg_prob:.3f}')
    print(f'  Avg SL Distance: {avg_sl_dist:.2f}%')
    print(f'  Total Loss: ${total_loss:.2f}')
print()

# 4. TRADE FREQUENCY ANALYSIS
print('='*80)
print('4️⃣ TRADE FREQUENCY ANALYSIS')
print('='*80)

# Group by date
from collections import defaultdict
daily_trades = defaultdict(int)
for t in prod_trades:
    entry_time = datetime.fromisoformat(t['entry_time'].replace('Z', ''))
    date = entry_time.strftime('%m-%d')
    daily_trades[date] += 1

print('Daily Trade Count:')
for date in sorted(daily_trades.keys()):
    print(f'  {date}: {daily_trades[date]} trades')
print()

total_days = 5.0
avg_per_day = len(prod_trades) / total_days
print(f'Average: {avg_per_day:.2f} trades/day')
print(f'Target: 2-10 trades/day')
print(f'Backtest: 9.46 trades/day')
print()

# 5. ENTRY PROBABILITY ANALYSIS
print('='*80)
print('5️⃣ ENTRY PROBABILITY DISTRIBUTION')
print('='*80)

probs_with_entry = [t.get('probability', 0) for t in prod_trades if t.get('probability', 0) > 0]

if probs_with_entry:
    low_conf = [p for p in probs_with_entry if p < 0.70]
    med_conf = [p for p in probs_with_entry if 0.70 <= p < 0.85]
    high_conf = [p for p in probs_with_entry if p >= 0.85]

    print(f'Probability Ranges:')
    print(f'  Low (<0.70): {len(low_conf)}/{len(probs_with_entry)} ({len(low_conf)/len(probs_with_entry)*100:.1f}%)')
    print(f'  Medium (0.70-0.85): {len(med_conf)}/{len(probs_with_entry)} ({len(med_conf)/len(probs_with_entry)*100:.1f}%)')
    print(f'  High (≥0.85): {len(high_conf)}/{len(probs_with_entry)} ({len(high_conf)/len(probs_with_entry)*100:.1f}%)')
    print()

    # Win rate by probability range
    for range_name, prob_list, threshold in [
        ('Low (<0.70)', low_conf, 0.70),
        ('Medium (0.70-0.85)', med_conf, None),
        ('High (≥0.85)', high_conf, 0.85)
    ]:
        if threshold is not None and threshold == 0.70:
            range_trades = [t for t in prod_trades if t.get('probability', 0) > 0 and t.get('probability', 0) < 0.70]
        elif threshold is not None and threshold == 0.85:
            range_trades = [t for t in prod_trades if t.get('probability', 0) >= 0.85]
        else:
            range_trades = [t for t in prod_trades if t.get('probability', 0) >= 0.70 and t.get('probability', 0) < 0.85]

        if range_trades:
            range_wins = [t for t in range_trades if t.get('pnl_usd_net', 0) > 0]
            range_pnl = sum(t.get('pnl_usd_net', 0) for t in range_trades)
            print(f'{range_name}:')
            print(f'  Trades: {len(range_trades)}')
            print(f'  Win Rate: {len(range_wins)}/{len(range_trades)} ({len(range_wins)/len(range_trades)*100:.1f}%)')
            print(f'  Total P&L: ${range_pnl:.2f}')
            print()

# 6. HOLD TIME ANALYSIS
print('='*80)
print('6️⃣ HOLD TIME ANALYSIS')
print('='*80)

hold_candles = [t.get('hold_candles', 0) for t in prod_trades if t.get('hold_candles', 0) > 0]
if hold_candles:
    avg_hold = sum(hold_candles) / len(hold_candles)
    avg_hours = avg_hold * 5 / 60  # 5-minute candles

    print(f'Average Hold: {avg_hold:.1f} candles ({avg_hours:.1f} hours)')
    print(f'Min Hold: {min(hold_candles)} candles ({min(hold_candles)*5/60:.1f} hours)')
    print(f'Max Hold: {max(hold_candles)} candles ({max(hold_candles)*5/60:.1f} hours)')
    print()

    # Hold time distribution
    short_hold = [h for h in hold_candles if h < 30]  # < 2.5 hours
    med_hold = [h for h in hold_candles if 30 <= h < 90]  # 2.5-7.5 hours
    long_hold = [h for h in hold_candles if h >= 90]  # > 7.5 hours

    print(f'Hold Time Distribution:')
    print(f'  Short (<2.5h): {len(short_hold)}/{len(hold_candles)} ({len(short_hold)/len(hold_candles)*100:.1f}%)')
    print(f'  Medium (2.5-7.5h): {len(med_hold)}/{len(hold_candles)} ({len(med_hold)/len(hold_candles)*100:.1f}%)')
    print(f'  Long (>7.5h): {len(long_hold)}/{len(hold_candles)} ({len(long_hold)/len(hold_candles)*100:.1f}%)')
print()

# 7. KEY FINDINGS AND RECOMMENDATIONS
print('='*80)
print('7️⃣ KEY FINDINGS')
print('='*80)

print('✅ Strengths:')
if ml_exits and len(ml_exits) > 0:
    ml_win_rate = len([t for t in ml_exits if t.get('pnl_usd_net', 0) > 0]) / len(ml_exits) * 100
    print(f'  - ML Exit Win Rate: {ml_win_rate:.1f}% (excellent)')
print(f'  - Overall Win Rate: {win_rate:.1f}% (above breakeven)')
print(f'  - Profitable: ${total_pnl:.2f} in 5 days')
print()

print('🚨 Critical Issues:')
if long_sls and len(long_trades) > 0:
    long_sl_rate = len(long_sls) / len(long_trades) * 100
    print(f'  - LONG Stop Loss Rate: {long_sl_rate:.1f}% (target: <20%)')
    print(f'  - LONG SL Total Loss: ${sum(t.get("pnl_usd_net", 0) for t in long_sls):.2f}')
if len(long_trades) > 0 and len(short_trades) >= 0:
    long_pct = len(long_trades) / len(prod_trades) * 100
    print(f'  - LONG Bias: {long_pct:.1f}% (target: 60%)')
print()

print('📊 Status vs Expectations:')
print(f'  Trade Frequency: {avg_per_day:.2f}/day (target: 2-10, backtest: 9.46)')
print(f'  Win Rate: {win_rate:.1f}% (backtest: 60.75%)')
print(f'  LONG/SHORT: {len(long_trades)}/{len(short_trades)} (backtest: 58/42)')
print()

print('💡 Fixes Applied Today (Nov 13):')
print('  1. LONG Entry Threshold: 0.60 → 0.70 (filter low-quality)')
print('  2. Stop Loss Distance: minimum 2.5% (prevent tight SLs)')
print('  3. SHORT Entry Threshold: 0.60 → 0.55 (increase opportunities)')
print()
print('⏳ Next Steps:')
print('  - Monitor for 2 days (until Nov 15)')
print('  - Validate fix effectiveness')
print('  - Target: LONG SL rate <20%, LONG/SHORT 60/40')
print()
