#!/usr/bin/env python3
"""
Reset Trading History - Option 1: Keep Positions, Clear History
Resets trade history while maintaining current open positions
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def reset_trading_history():
    """Reset trading history while keeping current positions"""

    state_path = project_root / "results" / "opportunity_gating_bot_4x_state.json"

    # Backup current state
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = project_root / "results" / f"opportunity_gating_bot_4x_state_backup_{backup_time}.json"

    with open(state_path, 'r') as f:
        state = json.load(f)

    # Save backup
    with open(backup_path, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"✅ Backup created: {backup_path.name}")

    # Current values before reset
    current_balance = state.get('current_balance', 0)
    current_position = state.get('position', {})
    if current_position is None:
        current_position = {}

    # Get unrealized P&L from open position
    unrealized_pnl = state.get('unrealized_pnl', 0)
    net_balance = current_balance + unrealized_pnl

    print(f"\n📊 Current State:")
    print(f"   Balance: ${current_balance:.2f}")
    print(f"   Unrealized P&L: ${unrealized_pnl:.2f}")
    print(f"   Net Balance: ${net_balance:.2f}")
    print(f"   Position: {current_position.get('status', 'UNKNOWN')}")
    if current_position.get('status') == 'OPEN':
        print(f"   Side: {current_position.get('side')}")
        print(f"   Entry: ${current_position.get('entry_price', 0):.2f}")

    # Reset trading history
    # ✅ FIXED 2025-10-26: Reverse calculate to "no position" baseline
    # When resetting with an open position, we need to remove the position's P&L
    # from the baseline to avoid treating existing losses as "normal"
    #
    # Logic:
    # - If position exists with unrealized P&L, reverse calculate the balance
    #   BEFORE entering that position (remove the P&L effect)
    # - Set initial_unrealized_pnl = 0 (no position baseline)
    # - This makes the current position's P&L show as actual performance
    #
    # Example:
    #   Current: Wallet $4589, Unrealized -$100, Equity $4489
    #   Baseline: Initial = $4589 (wallet), Unrealized = 0
    #   Result: Position P&L = -$100 (actual loss), Total Return = -2.2%
    now_utc = datetime.now(timezone.utc).isoformat()

    # Reverse calculate: baseline = wallet (position removed)
    state['initial_balance'] = current_balance  # Wallet = baseline (no position)
    state['initial_wallet_balance'] = current_balance  # Same as initial_balance
    state['initial_unrealized_pnl'] = 0  # No position baseline
    state['trades'] = []  # Clear all trade history
    state['closed_trades'] = 0  # Reset closed trades counter
    state['ledger'] = []  # Clear ledger

    # ✅ FIXED 2025-10-25: Update session_start to prevent reconciliation from loading old trades
    # This ensures reconciliation only fetches trades from this reset point forward
    state['session_start'] = now_utc

    # Reset stats
    state['stats'] = {
        'total_trades': 0,
        'long_trades': 0,
        'short_trades': 0,
        'wins': 0,
        'losses': 0,
        'total_pnl_usd': 0,
        'total_pnl_pct': 0
    }

    # Add reconciliation log entry
    reconciliation_entry = {
        'timestamp': now_utc,
        'event': 'trading_history_reset',
        'reason': 'Manual reset - Option 1: Keep positions, clear history (net_balance)',
        'balance': net_balance,
        'previous_balance': state.get('initial_balance', net_balance),
        'notes': f'Trade history reset on {now_utc}. Current position maintained. New baseline: ${net_balance:.2f} (balance ${current_balance:.2f} + unrealized ${unrealized_pnl:.2f})'
    }

    if 'reconciliation_log' not in state:
        state['reconciliation_log'] = []
    state['reconciliation_log'].append(reconciliation_entry)

    # Update timestamp
    state['timestamp'] = now_utc

    # Save reset state
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"\n✅ Trading History Reset Complete!")
    print(f"\n📋 New Session:")
    print(f"   Initial Balance (Net): ${net_balance:.2f}")
    print(f"     - Current Balance: ${current_balance:.2f}")
    print(f"     - Unrealized P&L: ${unrealized_pnl:.2f}")
    print(f"   Trades: 0")
    print(f"   Stats: Reset to 0")
    print(f"   Position: {'Maintained' if current_position.get('status') == 'OPEN' else 'None'}")

    print(f"\n⚠️ Important:")
    print(f"   - Current open position(s) are MAINTAINED")
    print(f"   - Bot will continue monitoring existing positions")
    print(f"   - New trades will start from clean slate")
    print(f"   - Backup saved: {backup_path.name}")

    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Trading History Reset - Option 1")
    print("Keep Positions, Clear History")
    print("=" * 60)

    try:
        reset_trading_history()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
