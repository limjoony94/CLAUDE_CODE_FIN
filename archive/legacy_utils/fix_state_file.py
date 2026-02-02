#!/usr/bin/env python3
"""
Fix state file issues:
1. Update initial_wallet_balance to match recent reset (354.02)
2. Remove duplicate open positions from trades array
3. Keep only the latest open position that matches the position field
"""

import json
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"
BACKUP_FILE = PROJECT_ROOT / "results" / f"opportunity_gating_bot_4x_state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def fix_state_file():
    """Fix state file issues"""

    # Load state file
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    # Backup original state
    with open(BACKUP_FILE, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"✅ Backed up state file to: {BACKUP_FILE.name}")

    # Issue 1: Fix initial_wallet_balance
    # Based on reconciliation_log entry 2025-10-30 10:18:39:
    # "balance_updated": 354.02
    old_initial_wallet = state.get('initial_wallet_balance', 0)
    new_initial_wallet = 354.02

    print(f"\n📝 Fixing initial_wallet_balance:")
    print(f"   Old: ${old_initial_wallet:,.2f}")
    print(f"   New: ${new_initial_wallet:,.2f}")

    state['initial_wallet_balance'] = new_initial_wallet
    state['initial_unrealized_pnl'] = 0  # Reset was done with no position initially

    # Issue 2: Remove duplicate open positions from trades array
    # The position field (state['position']) is the source of truth
    # Keep only the latest open position that matches it

    current_position = state.get('position')
    trades = state.get('trades', [])

    # Identify open trades in trades array
    open_trades = [t for t in trades if t.get('status') == 'OPEN']
    closed_trades = [t for t in trades if t.get('status') == 'CLOSED']

    print(f"\n📝 Cleaning up duplicate open positions:")
    print(f"   Found {len(open_trades)} open position(s) in trades array")
    print(f"   Found {len(closed_trades)} closed trades")

    if current_position and current_position.get('status') == 'OPEN':
        # Position field exists - find matching trade in array
        position_order_id = current_position.get('order_id')

        # Keep only the open trade that matches the position field
        matching_trades = [t for t in open_trades if t.get('order_id') == position_order_id]

        if matching_trades:
            print(f"   ✅ Keeping matching open position: {position_order_id}")
            # Rebuild trades array: closed trades + matching open trade
            state['trades'] = closed_trades + matching_trades
        else:
            print(f"   ⚠️  No matching trade found for position {position_order_id}")
            print(f"   Removing all open trades from array (position field is source of truth)")
            # Keep only closed trades
            state['trades'] = closed_trades
    else:
        print(f"   No open position in position field")
        print(f"   Removing all open trades from array")
        # Keep only closed trades
        state['trades'] = closed_trades

    # Update closed_trades count
    state['closed_trades'] = len(closed_trades)

    # Save fixed state file
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"\n✅ State file fixed successfully!")
    print(f"   Trades: {len(closed_trades)} closed + {len(state['trades']) - len(closed_trades)} open")
    print(f"   Initial wallet balance: ${new_initial_wallet:,.2f}")

    return state

if __name__ == "__main__":
    try:
        state = fix_state_file()

        # Verify fixes
        print(f"\n🔍 Verification:")
        print(f"   initial_wallet_balance: ${state.get('initial_wallet_balance', 0):,.2f}")
        print(f"   current_balance: ${state.get('current_balance', 0):,.2f}")

        wallet_change = state.get('current_balance', 0) - state.get('initial_wallet_balance', 0)
        wallet_change_pct = (wallet_change / state.get('initial_balance', 1)) * 100

        print(f"   Wallet change: ${wallet_change:+,.2f} ({wallet_change_pct:+.2f}%)")
        print(f"   Open positions in trades: {len([t for t in state.get('trades', []) if t.get('status') == 'OPEN'])}")

        position = state.get('position')
        if position and position.get('status') == 'OPEN':
            print(f"   Position field: OPEN ({position.get('side')} @ ${position.get('entry_price', 0):,.2f})")
        else:
            print(f"   Position field: CLOSED or None")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
