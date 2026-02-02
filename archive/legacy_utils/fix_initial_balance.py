#!/usr/bin/env python3
"""
Fix initial_balance to correct value from last reset
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def fix_initial_balance():
    """Restore initial_balance to value from last reset"""

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
    print()

    # Find last trading_history_reset
    reconciliation_log = state.get('reconciliation_log', [])
    last_reset = None

    for log_entry in reversed(reconciliation_log):
        if log_entry.get('event') == 'trading_history_reset':
            last_reset = log_entry
            break

    if not last_reset:
        print("❌ No trading_history_reset event found in reconciliation_log")
        return False

    # Get the initial_balance from reset event
    correct_initial_balance = last_reset.get('balance')

    if correct_initial_balance is None:
        print("❌ No balance field in reset event")
        return False

    # Current values
    current_initial = state.get('initial_balance', 0)
    current_balance = state.get('current_balance', 0)

    print("📊 State Analysis:")
    print(f"   Last Reset: {last_reset.get('timestamp', 'N/A')[:19]}")
    print(f"   Reset Reason: {last_reset.get('reason', 'N/A')}")
    print()
    print(f"   Correct initial_balance (from reset): ${correct_initial_balance:,.2f}")
    print(f"   Current initial_balance (WRONG): ${current_initial:.2f}")
    print(f"   Difference: ${current_initial - correct_initial_balance:+,.2f}")
    print()
    print(f"   Current current_balance: ${current_balance:.2f}")
    print()

    # Fix initial_balance
    state['initial_balance'] = correct_initial_balance

    # Save fixed state
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    print("✅ initial_balance Fixed:")
    print(f"   ${current_initial:.2f} → ${correct_initial_balance:.2f}")
    print()
    print("📝 Note:")
    print("   initial_balance는 이제 reset 시점의 올바른 값입니다.")
    print("   앞으로 reconciliation은 initial_balance를 절대 조정하지 않습니다.")
    print()
    print("🔄 Bot 재시작 권장:")
    print("   수정된 initial_balance를 사용하려면 봇을 재시작하세요.")

    return True

if __name__ == "__main__":
    print("=" * 70)
    print("Fix initial_balance to Reset Value")
    print("=" * 70)
    print()

    try:
        fix_initial_balance()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
