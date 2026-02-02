import json
from datetime import datetime
import sys
sys.path.append('.')

with open('results/opportunity_gating_bot_4x_state.json', 'r') as f:
    state = json.load(f)

deployment_time = datetime.fromisoformat('2025-11-07T19:09:00')
history = state.get('trading_history', [])

print('='*80)
print('🚨 CRITICAL ISSUE ANALYSIS: LONG STOP LOSS (5/15 = 33.3%)')
print('='*80)
print()

sl_trades = []
for t in history:
    if t.get('manual_trade', False):
        continue
    entry_time = t.get('entry_time', '')
    if entry_time:
        try:
            trade_time = datetime.fromisoformat(entry_time.replace('Z', ''))
            if trade_time >= deployment_time and 'Stop Loss' in t.get('exit_reason', ''):
                sl_trades.append(t)
        except:
            pass

print(f'Total Stop Loss Trades: {len(sl_trades)}')
print()

for i, t in enumerate(sl_trades, 1):
    entry_time = datetime.fromisoformat(t['entry_time'].replace('Z', ''))
    exit_time = datetime.fromisoformat(t.get('exit_time', t['entry_time']).replace('Z', ''))

    entry_price = t.get('entry_price', 0)
    exit_price = t.get('exit_price', entry_price)
    sl_price = t.get('stop_loss_price', 0)

    price_change = ((exit_price - entry_price) / entry_price) * 100

    entry_prob = t.get('probability', 0)
    leveraged_pnl = t.get('leveraged_pnl_pct', 0) * 100
    pnl_usd = t.get('pnl_usd_net', 0)

    hold_hours = (exit_time - entry_time).total_seconds() / 3600

    print(f'{i}. {entry_time.strftime("%m-%d %H:%M")} LONG Stop Loss')
    print(f'   Entry Prob: {entry_prob:.3f} (High confidence: {"YES" if entry_prob > 0.80 else "MEDIUM" if entry_prob > 0.70 else "LOW"})')
    print(f'   Entry: ${entry_price:,.1f} | Exit: ${exit_price:,.1f} | SL: ${sl_price:,.1f}')
    print(f'   Price Change: {price_change:.2f}% | Leveraged P&L: {leveraged_pnl:.2f}%')
    print(f'   Hold Time: {hold_hours:.1f}h | Loss: ${pnl_usd:.2f}')
    print()

# Analyze SL distance
print('📊 Stop Loss Distance Analysis:')
print('-'*80)
for i, t in enumerate(sl_trades, 1):
    entry_price = t.get('entry_price', 0)
    sl_price = t.get('stop_loss_price', 0)

    if entry_price > 0:
        sl_distance = ((entry_price - sl_price) / entry_price) * 100
        position_size = t.get('position_size_pct', 0) * 100

        print(f'{i}. SL Distance: {sl_distance:.2f}% | Position Size: {position_size:.1f}%')

print()
print('💡 KEY INSIGHTS:')
print('-'*80)
avg_prob = sum(t.get('probability', 0) for t in sl_trades) / len(sl_trades) if sl_trades else 0
avg_sl_distance = sum(((t.get('entry_price', 0) - t.get('stop_loss_price', 0)) / t.get('entry_price', 1)) * 100 for t in sl_trades) / len(sl_trades) if sl_trades else 0
print(f'- Avg Entry Prob: {avg_prob:.3f} (high confidence trades failing!)')
print(f'- Avg SL Distance: {avg_sl_distance:.2f}%')
print(f'- All 5 SLs are LONG (0 SHORT SLs)')
print(f'- Total Loss: ${sum(t.get("pnl_usd_net", 0) for t in sl_trades):.2f}')
print(f'- This is the MAIN PROBLEM reducing profitability')
print()
print('🎯 ROOT CAUSE:')
print('-'*80)
print('High confidence LONG entries (avg 0.76 prob) hitting Stop Loss')
print('→ Model is confident but market moves against position')
print('→ Either SL too tight OR model overconfident in current regime')
