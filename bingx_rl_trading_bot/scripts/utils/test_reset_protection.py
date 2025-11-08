#!/usr/bin/env python3
"""
Test script to verify that reset protection logic would work correctly.
Simulates what happens during bot startup reconciliation.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"

def test_reset_protection():
    """Test if recent reset would be detected and respected"""

    print("="*80)
    print("🧪 TESTING RESET PROTECTION LOGIC")
    print("="*80)

    # Load state file
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    print(f"\n📋 Current State:")
    print(f"   Initial Balance: ${state['initial_balance']:,.2f}")
    print(f"   Current Balance: ${state['current_balance']:,.2f}")
    print(f"   Trades: {len(state['trades'])}")
    print(f"   Closed Trades: {state['closed_trades']}")

    # Check reconciliation log for recent reset
    reconciliation_log = state.get('reconciliation_log', [])
    print(f"\n📊 Reconciliation Log Entries: {len(reconciliation_log)}")

    # Check last 5 entries for reset
    recent_reset_detected = False
    reset_time_since = None

    if reconciliation_log:
        recent_logs = reconciliation_log[-5:]
        for log_entry in reversed(recent_logs):
            print(f"\n   Event: {log_entry.get('event')}")
            print(f"   Timestamp: {log_entry.get('timestamp')}")

            if log_entry.get('event') == 'trading_history_reset':
                reset_time_str = log_entry.get('timestamp', '')
                try:
                    reset_time = datetime.fromisoformat(reset_time_str.replace('Z', '+00:00'))
                    now = datetime.now(reset_time.tzinfo) if reset_time.tzinfo else datetime.now()
                    time_since_reset = (now - reset_time).total_seconds()
                    reset_time_since = time_since_reset

                    print(f"   ⏰ Time Since Reset: {time_since_reset/60:.1f} minutes")

                    if time_since_reset < 3600:  # 1 hour
                        recent_reset_detected = True
                        print(f"   ✅ WITHIN 1-HOUR PROTECTION WINDOW")
                    else:
                        print(f"   ⚠️ OUTSIDE 1-HOUR PROTECTION WINDOW")
                    break
                except Exception as e:
                    print(f"   ❌ Error parsing timestamp: {e}")

    print(f"\n{'='*80}")
    print("🎯 TEST RESULTS:")
    print(f"{'='*80}")

    if recent_reset_detected:
        print(f"✅ PROTECTION ACTIVE - Reset detected {reset_time_since/60:.1f} minutes ago")
        print(f"✅ Balance reconciliation: Would PRESERVE initial_balance")
        print(f"✅ Trade reconciliation: Would SKIP trade fetching from exchange")
        print(f"\n📌 Expected Bot Behavior:")
        print(f"   - Initial balance stays at ${state['initial_balance']:,.2f}")
        print(f"   - Trades array stays empty (no re-population)")
        print(f"   - Current balance updated from exchange API")
        print(f"   - Position monitoring continues normally")
        return True
    else:
        if reset_time_since:
            print(f"⚠️ PROTECTION EXPIRED - Reset was {reset_time_since/3600:.1f} hours ago")
        else:
            print(f"⚠️ NO RECENT RESET DETECTED")
        print(f"⚠️ Balance reconciliation: Would ADJUST initial_balance")
        print(f"⚠️ Trade reconciliation: Would FETCH trades from exchange")
        return False

if __name__ == "__main__":
    try:
        protection_active = test_reset_protection()
        sys.exit(0 if protection_active else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
