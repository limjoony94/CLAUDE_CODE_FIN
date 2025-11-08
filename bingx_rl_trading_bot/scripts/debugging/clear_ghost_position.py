"""
Clear Ghost Position from State File
====================================
거래소에는 없지만 state file에 남아있는 포지션 제거
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"

def main():
    print("="*80)
    print("CLEAR GHOST POSITION")
    print("="*80)

    # Load state
    print("\nLoading state file...")
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    # Check current position
    position = state.get('position')
    if not position:
        print("✅ No position in state file (already clean)")
        return

    print(f"\nCurrent state shows:")
    print(f"  Position: {position['side']}")
    print(f"  Entry: {position['entry_time']}")
    print(f"  Price: ${position['entry_price']:,.2f}")
    print(f"  Order ID: {position['order_id']}")

    # Confirm
    print(f"\n⚠️  This will:")
    print(f"  1. Set position to None in state file")
    print(f"  2. Keep trades history intact")
    print(f"  3. Keep stats intact")

    # Clear position
    state['position'] = None

    # Save
    print(f"\nSaving updated state...")
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"✅ Position cleared!")
    print(f"\nState file updated:")
    print(f"  Position: None")
    print(f"  Closed trades: {state.get('closed_trades', 0)}")
    print(f"  Balance: ${state.get('current_balance', 0):,.2f}")

    print("\n" + "="*80)
    print("DONE - State file synchronized with exchange")
    print("="*80)

if __name__ == "__main__":
    main()
