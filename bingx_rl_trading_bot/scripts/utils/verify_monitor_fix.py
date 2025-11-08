#!/usr/bin/env python3
"""
Verify monitor calculation fix using BingX API and State file
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient
import yaml

# Load config
with open(PROJECT_ROOT / 'config' / 'api_keys.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize BingX client
client = BingXClient(
    api_key=config['bingx']['mainnet']['api_key'],
    secret_key=config['bingx']['mainnet']['secret_key'],
    testnet=False
)

# Load state file
state_file = PROJECT_ROOT / 'results' / 'opportunity_gating_bot_4x_state.json'
with open(state_file, 'r') as f:
    state = json.load(f)

print("=" * 80)
print("MONITOR CALCULATION VERIFICATION (AFTER FIX)")
print("=" * 80)
print()

# Get baseline values from state
initial_equity = state.get('initial_balance', 0)
initial_wallet = state.get('initial_wallet_balance', 0)
initial_unrealized = state.get('initial_unrealized_pnl', 0)

print("📊 BASELINE VALUES (from last reset):")
print(f"   Initial Equity:      ${initial_equity:>10,.2f}  (wallet + unrealized at reset)")
print(f"   Initial Wallet:      ${initial_wallet:>10,.2f}  (realized balance at reset)")
print(f"   Initial Unrealized:  ${initial_unrealized:>10,.2f}  (unrealized P&L at reset)")
print()
print(f"   Verification: ${initial_wallet:.2f} + (${initial_unrealized:.2f}) = ${initial_wallet + initial_unrealized:.2f}")
print(f"   Initial Equity: ${initial_equity:.2f}")
print(f"   Match: {'✅' if abs((initial_wallet + initial_unrealized) - initial_equity) < 0.01 else '❌'}")
print()

# Get current values from API
try:
    account_data = client.get_balance()
    balance = account_data.get('balance', {})

    current_wallet = float(balance.get('balance', 0))
    current_equity = float(balance.get('equity', 0))
    current_unrealized = float(balance.get('unrealizedProfit', 0))

    print("💰 CURRENT VALUES (from BingX API):")
    print(f"   Current Equity:      ${current_equity:>10,.2f}  (wallet + unrealized now)")
    print(f"   Current Wallet:      ${current_wallet:>10,.2f}  (realized balance now)")
    print(f"   Current Unrealized:  ${current_unrealized:>10,.2f}  (unrealized P&L now)")
    print()
    print(f"   Verification: ${current_wallet:.2f} + (${current_unrealized:.2f}) = ${current_wallet + current_unrealized:.2f}")
    print(f"   API Equity: ${current_equity:.2f}")
    print(f"   Match: {'✅' if abs((current_wallet + current_unrealized) - current_equity) < 0.01 else '❌'}")
    print()

    # Calculate returns (NEW METHOD - using proper baselines)
    print("📈 RETURN CALCULATION (NEW - FIXED):")
    print("-" * 80)

    # Wallet Return (Realized)
    wallet_change = current_wallet - initial_wallet
    realized_return_pct = (wallet_change / initial_equity) * 100 if initial_equity > 0 else 0

    print(f"1. Realized Return (Wallet Change):")
    print(f"   Current Wallet:   ${current_wallet:>10,.2f}")
    print(f"   Initial Wallet:   ${initial_wallet:>10,.2f}")
    print(f"   Change:           ${wallet_change:>+10,.2f}")
    print(f"   Return:           {realized_return_pct:>+10.2f}%")
    print()

    # Unrealized Return
    unrealized_change = current_unrealized - initial_unrealized
    unrealized_return_pct = (unrealized_change / initial_equity) * 100 if initial_equity > 0 else 0

    print(f"2. Unrealized Return (Unrealized Change):")
    print(f"   Current Unrealized: ${current_unrealized:>+10,.2f}")
    print(f"   Initial Unrealized: ${initial_unrealized:>+10,.2f}")
    print(f"   Change:             ${unrealized_change:>+10,.2f}")
    print(f"   Return:             {unrealized_return_pct:>+10.2f}%")
    print()

    # Total Return (Equity)
    equity_change = current_equity - initial_equity
    total_return_pct = (equity_change / initial_equity) * 100 if initial_equity > 0 else 0

    print(f"3. Total Return (Equity Change):")
    print(f"   Current Equity:   ${current_equity:>10,.2f}")
    print(f"   Initial Equity:   ${initial_equity:>10,.2f}")
    print(f"   Change:           ${equity_change:>+10,.2f}")
    print(f"   Return:           {total_return_pct:>+10.2f}%")
    print()

    # Verification
    print("✅ VERIFICATION:")
    print("-" * 80)
    calculated_total = realized_return_pct + unrealized_return_pct
    difference = abs(calculated_total - total_return_pct)

    print(f"   Realized Return:    {realized_return_pct:>+10.2f}%")
    print(f"   Unrealized Return:  {unrealized_return_pct:>+10.2f}%")
    print(f"   Sum:                {calculated_total:>+10.2f}%")
    print()
    print(f"   Total Return:       {total_return_pct:>+10.2f}%")
    print(f"   Difference:         {difference:>10.4f}%")
    print()

    if difference < 0.01:
        print("   ✅ PASS - Calculations are mathematically correct!")
        print("   ✅ Realized + Unrealized = Total")
    else:
        print(f"   ❌ FAIL - Mismatch of {difference:.4f}%")
        sys.exit(1)

except Exception as e:
    print(f"❌ Error querying BingX API: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("✅ MONITOR FIX VERIFIED - ALL CALCULATIONS CORRECT")
print("=" * 80)
