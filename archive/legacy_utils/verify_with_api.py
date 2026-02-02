#!/usr/bin/env python3
"""Verify metrics using API ground truth"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.monitoring.quant_monitor import init_api_client, fetch_realtime_data
import json

STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"

print('='*80)
print('EXCHANGE API - GROUND TRUTH DATA')
print('='*80)

# Get API data
client = init_api_client()
if not client:
    print("Failed to initialize API client")
    sys.exit(1)

data = fetch_realtime_data(client, 'BTC-USDT')
if not data:
    print("Failed to fetch data")
    sys.exit(1)

print('\n📊 API DATA (Ground Truth from Exchange):')
api_balance = data['balance']
api_unrealized = data['position']['unrealized_pnl'] if data.get('position') else 0
api_net = api_balance + api_unrealized

print(f'   Current Balance: ${api_balance:,.2f}')
print(f'   Unrealized P&L: ${api_unrealized:+,.2f}')
print(f'   Net Balance: ${api_net:,.2f}  (Balance + Unrealized)')

if data.get('position'):
    pos = data['position']
    print(f'\n   Position:')
    print(f'      Side: {pos.get("side", "NONE")}')
    print(f'      Entry Price: ${pos.get("entry_price", 0):,.2f}')
    print(f'      Quantity: {pos.get("quantity", 0):.4f}')
    print(f'      Unrealized P&L: ${pos.get("unrealized_pnl", 0):+,.2f}')

# Load state file
with open(STATE_FILE, 'r') as f:
    state = json.load(f)

print('\n'+'='*80)
print('STATE FILE DATA')
print('='*80)
state_initial = state.get('initial_balance', 0)
state_current = state.get('current_balance', 0)
state_net = state.get('net_balance', 0)
state_realized = state.get('realized_balance', 0)
state_unrealized = state.get('unrealized_pnl', 0)

print(f'   initial_balance: ${state_initial:,.2f}')
print(f'   current_balance: ${state_current:,.2f}')
print(f'   net_balance: ${state_net:,.2f}')
print(f'   realized_balance: ${state_realized:,.2f}')
print(f'   unrealized_pnl: ${state_unrealized:+,.2f}')

# Comparison
print('\n'+'='*80)
print('❗ COMPARISON: API (Truth) vs STATE FILE')
print('='*80)

print(f'\n💰 Current Balance:')
print(f'   API (Truth): ${api_balance:,.2f}')
print(f'   State File:  ${state_current:,.2f}')
diff_balance = api_balance - state_current
print(f'   Difference:  ${diff_balance:+,.2f}')
if abs(diff_balance) > 1.0:
    print(f'   ⚠️ WARNING: Significant difference!')

print(f'\n📊 Unrealized P&L:')
print(f'   API (Truth): ${api_unrealized:+,.2f}')
print(f'   State File:  ${state_unrealized:+,.2f}')
diff_unrealized = api_unrealized - state_unrealized
print(f'   Difference:  ${diff_unrealized:+,.2f}')
if abs(diff_unrealized) > 1.0:
    print(f'   ⚠️ WARNING: Significant difference!')

print(f'\n💎 Net Balance (Current + Unrealized):')
print(f'   API (Truth): ${api_net:,.2f}')
print(f'   State File:  ${state_net:,.2f}')
diff_net = api_net - state_net
print(f'   Difference:  ${diff_net:+,.2f}')
if abs(diff_net) > 1.0:
    print(f'   ⚠️ WARNING: Significant difference!')

# Calculate correct metrics using API ground truth
print('\n'+'='*80)
print('✅ CORRECT METRICS (using API Ground Truth)')
print('='*80)

print(f'\n   Initial Balance: ${state_initial:,.2f}')
print(f'   API Net Balance: ${api_net:,.2f}')
print(f'   Change: ${api_net - state_initial:+,.2f}')

total_return_api = ((api_net - state_initial) / state_initial) * 100
unrealized_pct_api = (api_unrealized / state_initial) * 100
balance_change_api = api_balance - state_initial

print(f'\n📈 Total Return (using API):')
print(f'   Formula: (net_balance - initial) / initial')
print(f'   Calculation: (${api_net:,.2f} - ${state_initial:,.2f}) / ${state_initial:,.2f}')
print(f'   Result: {total_return_api:+.2f}%')

print(f'\n📉 Unrealized P&L % (using API):')
print(f'   Formula: unrealized_pnl / initial_balance')
print(f'   Calculation: ${api_unrealized:+,.2f} / ${state_initial:,.2f}')
print(f'   Result: {unrealized_pct_api:+.2f}%')

print(f'\n💵 Balance Change (using API):')
print(f'   Formula: current_balance - initial_balance')
print(f'   Calculation: ${api_balance:,.2f} - ${state_initial:,.2f}')
print(f'   Result: ${balance_change_api:+,.2f} ({(balance_change_api/state_initial)*100:+.2f}%)')

# Compare with what monitor should show
print('\n'+'='*80)
print('🎯 WHAT MONITOR SHOULD SHOW (with corrected code):')
print('='*80)
print(f'   Realized Return: +0.0% (no closed trades)')
print(f'   Unrealized P&L: {unrealized_pct_api:+.2f}%')
print(f'   Total Return: {total_return_api:+.2f}%')
print(f'   Balance Change: ${balance_change_api:+,.2f} ({(balance_change_api/state_initial)*100:+.2f}%)')
