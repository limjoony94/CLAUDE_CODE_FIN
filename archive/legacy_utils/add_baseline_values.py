#!/usr/bin/env python3
"""
Add initial_wallet_balance and initial_unrealized_pnl from last reset
"""

import json
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def add_baseline_values():
    """Add wallet and unrealized baseline values from reset event"""

    state_path = project_root / "results" / "opportunity_gating_bot_4x_state.json"

    # Backup
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = project_root / "results" / f"opportunity_gating_bot_4x_state_backup_{backup_time}.json"

    with open(state_path, 'r') as f:
        state = json.load(f)

    # Save backup
    with open(backup_path, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"✅ Backup: {backup_path.name}")
    print()

    # Find last reset
    reconciliation_log = state.get('reconciliation_log', [])
    last_reset = None

    for log_entry in reversed(reconciliation_log):
        if log_entry.get('event') == 'trading_history_reset':
            last_reset = log_entry
            break

    if not last_reset:
        print("❌ No reset event found")
        return False

    # Extract from notes
    notes = last_reset.get('notes', '')
    # Format: "New baseline: $4481.96 (balance $4534.97 + unrealized $-53.00)"

    import re
    match = re.search(r'balance \$([0-9,.]+) \+ unrealized \$([0-9,.+-]+)', notes)

    if not match:
        print("❌ Could not parse reset notes")
        print(f"   Notes: {notes}")
        return False

    initial_wallet = float(match.group(1).replace(',', ''))
    initial_unrealized = float(match.group(2).replace(',', ''))

    print("📊 Extracted from Reset Notes:")
    print(f"   Timestamp: {last_reset.get('timestamp', 'N/A')[:19]}")
    print(f"   initial_balance (equity): ${last_reset.get('balance'):.2f}")
    print(f"   initial_wallet_balance: ${initial_wallet:.2f}")
    print(f"   initial_unrealized_pnl: ${initial_unrealized:.2f}")
    print()

    # Verify
    equity = last_reset.get('balance')
    calculated = initial_wallet + initial_unrealized
    if abs(equity - calculated) > 0.01:
        print(f"⚠️  Warning: Mismatch!")
        print(f"   Equity: ${equity:.2f}")
        print(f"   Calculated: ${calculated:.2f}")
        print()

    # Add values
    state['initial_wallet_balance'] = initial_wallet
    state['initial_unrealized_pnl'] = initial_unrealized

    # Save
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    print("✅ Baseline Values Added:")
    print(f"   initial_wallet_balance: ${initial_wallet:.2f}")
    print(f"   initial_unrealized_pnl: ${initial_unrealized:.2f}")
    print()
    print("🔄 Bot 재시작 권장")

    return True

if __name__ == "__main__":
    print("=" * 70)
    print("Add Baseline Values from Reset")
    print("=" * 70)
    print()

    try:
        add_baseline_values()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
