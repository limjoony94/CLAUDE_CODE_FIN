"""
Detailed state file analysis for performance metrics review
"""
import json
from pathlib import Path

# Load state file
state_file = Path(__file__).parent.parent.parent / 'results' / 'opportunity_gating_bot_4x_state.json'
with open(state_file, 'r') as f:
    state = json.load(f)

print('=' * 80)
print('STATE FILE DETAILED ANALYSIS')
print('=' * 80)
print()

print('=== Balance Summary ===')
print(f'Initial Balance: ${state["initial_balance"]:.2f}')
print(f'Current Balance: ${state["current_balance"]:.2f}')
print(f'Realized Balance: ${state.get("realized_balance", 0):.2f}')
print(f'Unrealized P&L: ${state["unrealized_pnl"]:.2f}')
print()

# Calculate equity
equity = state['current_balance'] + state['unrealized_pnl']
print(f'Calculated Equity: ${equity:.2f}')
print()

# Analyze trades
trades = state.get('trades', [])
print(f'Total Trades in Array: {len(trades)}')
print()

# Separate bot vs manual trades
bot_trades = [t for t in trades if not t.get('manual_trade', False)]
manual_trades = [t for t in trades if t.get('manual_trade', False)]

print(f'Bot-Managed Trades: {len(bot_trades)}')
print(f'Manual Trades: {len(manual_trades)}')
print()

# Manual trades impact
total_manual_pnl = 0
if manual_trades:
    print('=== Manual Trades Breakdown ===')
    for i, t in enumerate(manual_trades, 1):
        pnl = t.get('pnl_usd_net', 0)
        total_manual_pnl += pnl
        print(f'{i}. Entry: ${t["entry_price"]:.1f} → Exit: ${t["exit_price"]:.1f}')
        print(f'   P&L: ${pnl:.2f}, Fee: ${t.get("total_fee", 0):.2f}')
    print(f'Total Manual P&L: ${total_manual_pnl:.2f}')
    print()

# Balance reconciliation
balance_change = state['current_balance'] - state['initial_balance']
print('=== Balance Reconciliation ===')
print(f'Balance Change: ${balance_change:.2f}')
print(f'Manual Trades P&L: ${total_manual_pnl:.2f}')
print(f'Difference: ${balance_change - total_manual_pnl:.2f}')
print(f'Difference %: {(balance_change - total_manual_pnl) / abs(balance_change) * 100:.1f}% of change')
print()

# Calculate returns
initial = state['initial_balance']
print('=== Return Calculations (Current Monitor Logic) ===')
total_return = (equity - initial) / initial * 100
realized_return = balance_change / initial * 100
unrealized_return = state["unrealized_pnl"] / initial * 100

print(f'Total Return (Equity-based): {total_return:+.2f}%')
print(f'  = (${equity:.2f} - ${initial:.2f}) / ${initial:.2f}')
print()
print(f'Realized Return (Balance change): {realized_return:+.2f}%')
print(f'  = (${state["current_balance"]:.2f} - ${initial:.2f}) / ${initial:.2f}')
print()
print(f'Unrealized Return: {unrealized_return:+.2f}%')
print(f'  = ${state["unrealized_pnl"]:.2f} / ${initial:.2f}')
print()

# Verification
print(f'Verification: Realized + Unrealized = Total?')
print(f'  {realized_return:.2f}% + {unrealized_return:.2f}% = {realized_return + unrealized_return:.2f}%')
print(f'  Expected: {total_return:.2f}%')
print(f'  Match: {"✅ YES" if abs((realized_return + unrealized_return) - total_return) < 0.01 else "❌ NO"}')
print()

# Stats field
stats = state.get('stats', {})
print('=== Stats Field (Bot Performance Tracking) ===')
print(f'Total Trades (stats): {stats.get("total_trades", 0)}')
print(f'Total P&L USD (stats): ${stats.get("total_pnl_usd", 0):.2f}')
print(f'Total P&L % (stats): {stats.get("total_pnl_pct", 0):.2f}%')
print()

# Current position
position = state.get('position', {})
print('=== Current Position ===')
if position and position.get('status') == 'OPEN':
    print(f'Status: {position["status"]}')
    print(f'Side: {position["side"]}')
    print(f'Entry Price: ${position["entry_price"]:.2f}')
    print(f'Quantity: {position["quantity"]:.4f} BTC')
    print(f'Position Value: ${position["position_value"]:.2f}')
    print(f'Leveraged Value: ${position["leveraged_value"]:.2f}')
    print(f'Synced from Exchange: {position.get("synced_from_exchange", False)}')
else:
    print('No open position')
print()

# Reconciliation log
recon_log = state.get('reconciliation_log', [])
if recon_log:
    print('=== Reconciliation Log ===')
    for entry in recon_log:
        print(f'Time: {entry["timestamp"]}')
        print(f'Event: {entry["event"]}')
        print(f'Balance: ${entry["balance"]:.2f}')
        print(f'Notes: {entry.get("notes", "N/A")}')
        print()

print('=' * 80)
print('CONCLUSION')
print('=' * 80)
print()
print('1. Current Balance Source:')
print(f'   - Set at session start: ${initial:.2f} (2025-10-22 02:08:57)')
print(f'   - Reflects manual trades losses: ${total_manual_pnl:.2f}')
print(f'   - Bot-managed trades: 0')
print()
print('2. Performance Metrics Interpretation:')
print(f'   - Trades: 0 = Bot-managed trades only (correct)')
print(f'   - Realized Return: {realized_return:+.2f}% = Includes manual trades (correct)')
print(f'   - This creates confusion: "0 trades but negative return?"')
print()
print('3. Balance vs Equity:')
print(f'   - Current Balance: ${state["current_balance"]:.2f} (realized only)')
print(f'   - Equity: ${equity:.2f} (realized + unrealized)')
print(f'   - Difference: ${equity - state["current_balance"]:.2f} (unrealized)')
print()
print('4. Key Question:')
print('   Is current_balance from Exchange API or calculated internally?')
print('   → Need to check bot code for balance update logic')
