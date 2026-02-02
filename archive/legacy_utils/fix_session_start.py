#!/usr/bin/env python3
"""
Fix session_start bug: Update to last reset time and remove old trades
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def fix_session_start():
    """Fix session_start to match last trading_history_reset"""

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

    # Find last trading_history_reset time
    reset_time = None
    for log in state.get('reconciliation_log', []):
        if log['event'] == 'trading_history_reset':
            reset_time = log['timestamp']

    if not reset_time:
        print("❌ No trading_history_reset found in reconciliation_log")
        return False

    print(f"\n📊 Current State:")
    print(f"   session_start: {state.get('session_start')}")
    print(f"   Last reset: {reset_time}")
    print(f"   Trades: {len(state.get('trades', []))}")

    # Filter trades to only those AFTER reset time
    reset_dt = datetime.fromisoformat(reset_time.replace('+00:00', '').replace('Z', ''))
    trades_before = len(state.get('trades', []))

    filtered_trades = []
    for trade in state.get('trades', []):
        entry_time = trade.get('entry_time', '')
        if entry_time:
            # Parse entry_time (may have different formats)
            try:
                if '+' in entry_time or 'Z' in entry_time:
                    trade_dt = datetime.fromisoformat(entry_time.replace('+00:00', '').replace('Z', ''))
                else:
                    trade_dt = datetime.fromisoformat(entry_time)

                # Keep only trades AFTER reset
                if trade_dt >= reset_dt:
                    filtered_trades.append(trade)
                else:
                    print(f"   Removing: {entry_time} (before reset)")
            except Exception as e:
                print(f"   ⚠️ Could not parse time: {entry_time} - {e}")
                # Keep if cannot parse (safer)
                filtered_trades.append(trade)

    # Update state
    state['session_start'] = reset_time
    state['trades'] = filtered_trades
    state['timestamp'] = datetime.now().isoformat()

    # Save fixed state
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"\n✅ Session Start Fixed!")
    print(f"\n📋 Updated State:")
    print(f"   session_start: {reset_time}")
    print(f"   Trades: {trades_before} → {len(filtered_trades)}")
    print(f"   Removed: {trades_before - len(filtered_trades)} old trades")
    print(f"   Backup: {backup_path.name}")

    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Fix session_start Bug")
    print("Update to last reset time & remove old trades")
    print("=" * 60)

    try:
        fix_session_start()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
