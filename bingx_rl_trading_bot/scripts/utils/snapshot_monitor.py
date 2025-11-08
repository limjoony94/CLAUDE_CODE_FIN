"""Quick monitoring snapshot - display current bot status"""
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
state_file = PROJECT_ROOT / 'results' / 'phase4_testnet_trading_state.json'

with open(state_file, 'r') as f:
    state = json.load(f)

# Parse state
initial_balance = state['initial_balance']
current_balance = state['current_balance']
pnl = current_balance - initial_balance
pnl_pct = (pnl / initial_balance * 100) if initial_balance > 0 else 0

trades = state['trades']
open_trades = [t for t in trades if t['status'] == 'OPEN']
closed_trades = [t for t in trades if t['status'] == 'CLOSED']

print('=' * 80)
print('📊 BINGX DUAL MODEL BOT - MONITORING SNAPSHOT')
print('=' * 80)
print(f'Snapshot Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'State Updated: {state["timestamp"]}')
print()

# Bot Status
print('🤖 BOT STATUS')
print('-' * 80)
session_start = datetime.fromisoformat(state['session_start'])
runtime_hours = (datetime.now() - session_start).total_seconds() / 3600
print(f'Runtime: {runtime_hours:.1f} hours (since {session_start.strftime("%Y-%m-%d %H:%M")})')
print(f'Initial Balance: ${initial_balance:,.2f} USDT')
print(f'Current Balance: ${current_balance:,.2f} USDT')

pnl_emoji = '📈' if pnl >= 0 else '📉'
print(f'Session P&L: {pnl_emoji} ${pnl:+,.2f} ({pnl_pct:+.2f}%)')
print()

# Open Position
print('📈 OPEN POSITION')
print('-' * 80)
if len(open_trades) > 0:
    for trade in open_trades:
        side = trade['side']
        entry_price = trade['entry_price']
        quantity = trade['quantity']
        entry_time = datetime.fromisoformat(trade['entry_time'])
        hours_held = (datetime.now() - entry_time).total_seconds() / 3600
        probability = trade['probability']

        side_emoji = '🟢' if side == 'LONG' else '🔴'
        print(f'{side_emoji} {side} Position')
        print(f'   Entry: ${entry_price:,.2f} @ {entry_time.strftime("%H:%M:%S")}')
        print(f'   Quantity: {quantity:.4f} BTC')
        print(f'   Value: ${entry_price * quantity:,.2f} USDT')
        print(f'   Holding: {hours_held:.1f} hours (Max: 4.0h)')
        print(f'   Signal: {probability:.1%} confidence')
        print(f'   Size: {trade["position_size_pct"]*100:.1f}% of capital')
        print(f'   Regime: {trade["regime"]}')
        print(f'   Time Remaining: {max(0, 4.0 - hours_held):.1f} hours until Max Hold exit')
else:
    print('No open positions')
    print('   Waiting for LONG >= 0.7 OR SHORT >= 0.7 signal')

print()

# Trade Statistics
print('📊 TRADE STATISTICS')
print('-' * 80)
print(f'Total Trades: {len(trades)}')

if len(closed_trades) > 0:
    winning = len([t for t in closed_trades if t['pnl_usd_net'] > 0])
    win_rate = (winning / len(closed_trades)) * 100
    total_pnl = sum(t['pnl_usd_net'] for t in closed_trades)

    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']

    print(f'Closed: {len(closed_trades)}')
    print(f'   Wins: {winning} ({win_rate:.1f}%)')
    print(f'   Total P&L: ${total_pnl:+,.2f}')
    print()
    print(f'Direction:')
    print(f'   LONG: {len(long_trades)} ({len(long_trades)/len(trades)*100:.1f}%)')
    print(f'   SHORT: {len(short_trades)} ({len(short_trades)/len(trades)*100:.1f}%)')
else:
    print('No completed trades yet')

print()

# Expected vs Actual
print('🎯 EXPECTED (Backtest) vs ACTUAL')
print('-' * 80)
days_running = runtime_hours / 24
trades_per_week = (len(trades) / days_running * 7) if days_running > 0 else 0

print('Expected (Dual Model):')
print('   Return: +14.98% per 5 days')
print('   Win Rate: 66.2%')
print('   Trades/Week: ~26.2')
print('   LONG: 87.6% | SHORT: 12.4%')
print()
print(f'Actual (Live):')
print(f'   Return: {pnl_pct:+.2f}% ({runtime_hours:.1f}h runtime)')
if len(closed_trades) > 0:
    print(f'   Win Rate: {win_rate:.1f}%')
else:
    print(f'   Win Rate: N/A (no closed trades yet)')
print(f'   Trades/Week: {trades_per_week:.1f}')

long_pct = len([t for t in trades if t['side'] == 'LONG']) / len(trades) * 100 if len(trades) > 0 else 0
short_pct = len([t for t in trades if t['side'] == 'SHORT']) / len(trades) * 100 if len(trades) > 0 else 0
print(f'   LONG: {long_pct:.1f}% | SHORT: {short_pct:.1f}%')

print()
print('=' * 80)
print('💡 Tip: Run this script anytime to see current status')
print('   Command: python scripts/utils/snapshot_monitor.py')
