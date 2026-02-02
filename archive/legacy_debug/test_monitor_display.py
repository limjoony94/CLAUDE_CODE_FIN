#!/usr/bin/env python3
"""Test monitor display function."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import json

# Load state file
state_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'opportunity_gating_bot_4x_state.json')
with open(state_path, 'r') as f:
    state = json.load(f)

print("="*100)
print("Testing display_recent_trades function")
print("="*100)

# Import and test the function
try:
    from scripts.monitoring import quant_monitor

    print("\n1. State loaded successfully")
    print(f"   Trades: {len(state.get('trades', []))}")

    print("\n2. Calling display_recent_trades...")
    quant_monitor.display_recent_trades(state)

    print("\n✅ Function executed successfully")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
