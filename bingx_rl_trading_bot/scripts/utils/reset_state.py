#!/usr/bin/env python3
"""
Reset trading state with current exchange balance
Creates clean state for new trading session
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.api.bingx_client import BingXClient
import yaml

def reset_state():
    """Reset state file with current balance from exchange"""

    print("=" * 80)
    print("🔄 Resetting Trading State")
    print("=" * 80)

    # Load API keys
    config_path = project_root / "config" / "api_keys.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize client
    client = BingXClient(
        api_key=config['bingx']['mainnet']['api_key'],
        secret_key=config['bingx']['mainnet']['secret_key'],
        testnet=False  # MAINNET
    )

    print("\n💰 Fetching current balance from exchange...")

    # Get current balance
    balance_info = client.get_balance()
    balance = float(balance_info['balance']['balance'])
    print(f"   Balance: ${balance:,.2f}")

    # Check for open positions
    print("\n📊 Checking for open positions...")
    positions = client.get_positions("BTC-USDT")

    has_open_positions = False
    for pos in positions:
        if pos and pos.get('positionAmt') != 0:
            has_open_positions = True
            qty = abs(float(pos['positionAmt']))
            side = "LONG" if float(pos['positionAmt']) > 0 else "SHORT"
            print(f"   ⚠️  OPEN POSITION: {side} {qty} BTC")

    if has_open_positions:
        print("\n❌ ERROR: Cannot reset state with open positions!")
        print("   Please close all positions first.")
        return

    print("   ✅ No open positions")

    # Confirm reset
    print(f"\n⚠️  This will RESET the state file with:")
    print(f"   - Initial Balance: ${balance:,.2f}")
    print(f"   - Session Start: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"   - All trade history: CLEARED")

    response = input("\nContinue? (yes/no): ")

    if response.lower() != 'yes':
        print("❌ Cancelled")
        return

    # Create new state
    now_utc = datetime.now(timezone.utc)

    new_state = {
        "session_start": now_utc.isoformat(),
        "initial_balance": balance,
        "current_balance": balance,
        "timestamp": now_utc.isoformat(),
        "position": {
            "status": "CLOSED"
        },
        "trades": [],
        "closed_trades": 0,
        "ledger": [],
        "reconciliation_log": [
            {
                "timestamp": now_utc.isoformat(),
                "event": "state_reset",
                "reason": "Clean reset with new Exit models (threshold 0.80)",
                "balance": balance,
                "previous_balance": balance,
                "notes": f"State reset on {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC. New session starting."
            }
        ],
        "latest_signals": {},
        "stats": {
            "total_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl_usd": 0,
            "total_pnl_pct": 0
        },
        "configuration": {
            "long_threshold": 0.70,
            "short_threshold": 0.70,
            "gate_threshold": 0.001,
            "ml_exit_threshold_base_long": 0.80,
            "ml_exit_threshold_base_short": 0.80,
            "emergency_stop_loss": 0.03,
            "emergency_max_hold_hours": 10.0,
            "leverage": 4,
            "long_avg_return": 0.0041,
            "short_avg_return": 0.0047,
            "fixed_take_profit": 0.03,
            "trailing_tp_activation": 0.02,
            "trailing_tp_drawdown": 0.10,
            "volatility_high": 0.02,
            "volatility_low": 0.01,
            "ml_threshold_high_vol": 0.65,
            "ml_threshold_low_vol": 0.75,
            "exit_strategy": "COMBINED"
        },
        "realized_balance": balance,
        "unrealized_pnl": 0.0
    }

    # Save new state
    state_path = project_root / "results" / "opportunity_gating_bot_4x_state.json"

    print(f"\n💾 Saving new state to: {state_path}")

    with open(state_path, 'w') as f:
        json.dump(new_state, f, indent=2)

    print("✅ State reset complete!")
    print(f"\n📊 New Session Details:")
    print(f"   Session Start: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"   Initial Balance: ${balance:,.2f}")
    print(f"   Configuration: EXIT threshold 0.80, SL -3%, Hold 10h")
    print(f"\n🚀 Ready to start bot with clean state!")

if __name__ == "__main__":
    reset_state()
