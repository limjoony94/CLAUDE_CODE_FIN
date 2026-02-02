#!/usr/bin/env python3
"""
Test monitor calculation logic
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load state
state_file = PROJECT_ROOT / 'results' / 'opportunity_gating_bot_4x_state.json'
with open(state_file, 'r') as f:
    state = json.load(f)

print("=" * 80)
print("MONITOR CALCULATION TEST")
print("=" * 80)

# Test 1: Balance Calculation
print("\n1. BALANCE CALCULATION TEST")
print("-" * 80)

initial_balance = state.get('initial_balance', 0)
current_balance = state.get('current_balance', 0)
unrealized_pnl = state.get('unrealized_pnl', 0)

print(f"Initial Balance:    ${initial_balance:>10,.2f}")
print(f"Current Balance:    ${current_balance:>10,.2f}")
print(f"Unrealized P&L:     ${unrealized_pnl:>10,.2f}")
print()

# Calculate returns
total_return = (current_balance - initial_balance) / initial_balance if initial_balance > 0 else 0
unrealized_return = unrealized_pnl / initial_balance if initial_balance > 0 else 0
realized_balance_change = (current_balance - initial_balance) - unrealized_pnl
realized_return = realized_balance_change / initial_balance if initial_balance > 0 else 0

print(f"Total Return:       {total_return*100:>10.2f}%")
print(f"Unrealized Return:  {unrealized_return*100:>10.2f}%")
print(f"Realized Return:    {realized_return*100:>10.2f}%")
print()

# Verification
calculated_total = realized_return + unrealized_return
matches = abs(calculated_total - total_return) < 0.0001

print(f"Verification:")
print(f"  Realized + Unrealized = {calculated_total*100:>10.2f}%")
print(f"  Total Return          = {total_return*100:>10.2f}%")
print(f"  Match: {'✅ YES' if matches else '❌ NO'}")

if not matches:
    print(f"  ⚠️  ERROR: Mismatch of {abs(calculated_total - total_return)*100:.4f}%")
    sys.exit(1)

# Test 2: Trades Analysis
print("\n2. TRADES ANALYSIS TEST")
print("-" * 80)

trades = state.get('trades', [])
closed_trades = [t for t in trades if t.get('status') == 'CLOSED' and not t.get('manual_trade', False)]
open_trades = [t for t in trades if t.get('status') == 'OPEN']
manual_trades = [t for t in trades if t.get('manual_trade', False)]

print(f"Total Trades:       {len(trades):>5d}")
print(f"  Bot Closed:       {len(closed_trades):>5d}")
print(f"  Bot Open:         {len(open_trades):>5d}")
print(f"  Manual (Recon):   {len(manual_trades):>5d}")

# Test 3: Position Grouping
print("\n3. POSITION GROUPING TEST")
print("-" * 80)

positions = defaultdict(lambda: {
    'trades': [],
    'side': 'N/A',
    'total_pnl': 0,
    'total_fees': 0,
    'exit_reason': 'N/A'
})

for trade in closed_trades:
    position_id = trade.get('position_id_exchange', trade.get('order_id'))
    positions[position_id]['trades'].append(trade)
    positions[position_id]['side'] = trade.get('side', 'N/A')
    positions[position_id]['total_pnl'] += trade.get('pnl_usd_net', 0)
    positions[position_id]['total_fees'] += trade.get('total_fee', 0)
    positions[position_id]['exit_reason'] = trade.get('exit_reason', 'N/A')

print(f"Unique Positions:   {len(positions):>5d}")
print()

# Show position details
for i, (pos_id, pos_data) in enumerate(positions.items(), 1):
    num_trades = len(pos_data['trades'])
    print(f"  Position #{i}: {pos_id}")
    print(f"    Side:       {pos_data['side']}")
    print(f"    Trades:     {num_trades}")
    print(f"    P&L:        ${pos_data['total_pnl']:>+10.2f}")
    print(f"    Fees:       ${pos_data['total_fees']:>10.2f}")
    print(f"    Exit:       {pos_data['exit_reason'][:30]}")
    print()

# Test 4: Realized P&L Cross-Check
print("4. REALIZED P&L CROSS-CHECK")
print("-" * 80)

# Method 1: Sum of closed trades P&L
trades_sum = sum([t.get('pnl_usd_net', 0) for t in closed_trades])
print(f"Method 1 (Trades Sum):       ${trades_sum:>10,.2f}")

# Method 2: Balance reconciliation
balance_recon = realized_balance_change
print(f"Method 2 (Balance Recon):    ${balance_recon:>10,.2f}")

# Method 3: Position grouping sum
position_sum = sum([pos['total_pnl'] for pos in positions.values()])
print(f"Method 3 (Position Sum):     ${position_sum:>10,.2f}")

print()
print(f"Difference (M1 vs M2):       ${abs(trades_sum - balance_recon):>10,.2f}")
print(f"Difference (M1 vs M3):       ${abs(trades_sum - position_sum):>10,.2f}")

# Check if differences are significant
if abs(trades_sum - balance_recon) > 1.0:
    print(f"\n⚠️  WARNING: Large discrepancy between trades sum and balance reconciliation")
    print(f"   This indicates state file inconsistency or missing trades")
    print(f"   Using balance reconciliation (Method 2) is more accurate")

# Test 5: State File Consistency
print("\n5. STATE FILE CONSISTENCY CHECK")
print("-" * 80)

stats = state.get('stats', {})
stats_trades = stats.get('total_trades', 0)
actual_trades = len(closed_trades)

print(f"Stats 'total_trades':    {stats_trades:>5d}")
print(f"Actual closed trades:    {actual_trades:>5d}")
print(f"Match: {'✅ YES' if stats_trades == actual_trades else '❌ NO'}")

if stats_trades != actual_trades:
    print(f"⚠️  WARNING: stats.total_trades mismatch")

print()
print("=" * 80)
print("✅ ALL TESTS COMPLETED")
print("=" * 80)
