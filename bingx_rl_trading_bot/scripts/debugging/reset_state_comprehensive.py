"""
Comprehensive State File Reset
===============================
Reset all state records and synchronize with exchange reality

Purpose:
  - Clear old testnet data mixed with mainnet data
  - Reset initial_balance to actual exchange balance
  - Clear trade history
  - Synchronize positions with exchange
  - Prepare clean state for fresh session

WARNING: This will delete all historical trade records!
Backup will be created automatically.
"""

import sys
from pathlib import Path
import json
import yaml
from datetime import datetime
import shutil

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"
CONFIG_DIR = PROJECT_ROOT / "config"
SYMBOL = "BTC-USDT"

def load_api_keys():
    """Load API keys"""
    api_keys_file = CONFIG_DIR / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('testnet', {})
    return {}

def main():
    print("="*80)
    print("COMPREHENSIVE STATE RESET")
    print("="*80)
    print("\n⚠️  WARNING: This will:")
    print("   1. Delete all historical trade records")
    print("   2. Reset statistics to zero")
    print("   3. Synchronize state with exchange")
    print("   4. Create fresh session")
    print("\n📦 Backup will be created automatically")

    # Confirm
    print("\n" + "="*80)
    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Reset cancelled")
        return

    # Create backup
    print("\n[1] Creating backup...")
    backup_file = STATE_FILE.parent / f"opportunity_gating_bot_4x_state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy(STATE_FILE, backup_file)
    print(f"✅ Backup saved: {backup_file.name}")

    # Connect to exchange
    print("\n[2] Connecting to BingX (Mainnet)...")
    api_config = load_api_keys()
    client = BingXClient(
        api_key=api_config.get('api_key', ''),
        secret_key=api_config.get('secret_key', ''),
        testnet=False  # Mainnet
    )
    print("✅ Connected")

    # Get current exchange state
    print("\n[3] Fetching exchange state...")

    # Get balance
    try:
        balance_info = client.get_balance()
        actual_balance = float(balance_info.get('balance', {}).get('availableMargin', 0))
        equity = float(balance_info.get('balance', {}).get('equity', 0))
        unrealized_pnl = float(balance_info.get('balance', {}).get('unrealizedProfit', 0))

        print(f"   Balance:")
        print(f"     Available Margin: ${actual_balance:,.2f}")
        print(f"     Equity: ${equity:,.2f}")
        print(f"     Unrealized P&L: ${unrealized_pnl:,.2f}")
    except Exception as e:
        print(f"❌ Failed to get balance: {e}")
        return

    # Get positions
    try:
        positions = client.get_positions(SYMBOL)
        active_positions = []

        if positions and len(positions) > 0:
            print(f"\n   Positions:")
            for pos in positions:
                position_amt = float(pos.get('positionAmt', 0))
                if abs(position_amt) > 0.00001:  # Not dust
                    position_side = pos.get('positionSide', 'BOTH')
                    avg_price = float(pos.get('avgPrice', 0))
                    mark_price = float(pos.get('markPrice', 0))
                    unrealized = float(pos.get('unrealizedProfit', 0))
                    leverage = pos.get('leverage', 'N/A')

                    # Determine actual side
                    if position_side == "BOTH":
                        actual_side = "LONG" if position_amt > 0 else "SHORT"
                    else:
                        actual_side = position_side

                    print(f"     {actual_side}: {abs(position_amt):.8f} BTC @ ${avg_price:,.2f}")
                    print(f"       Mark Price: ${mark_price:,.2f}")
                    print(f"       Unrealized P&L: ${unrealized:,.2f}")
                    print(f"       Leverage: {leverage}x")

                    active_positions.append({
                        'side': actual_side,
                        'quantity': abs(position_amt),
                        'entry_price': avg_price,
                        'mark_price': mark_price,
                        'unrealized_pnl': unrealized
                    })
        else:
            print(f"   No open positions")

    except Exception as e:
        print(f"❌ Failed to get positions: {e}")
        return

    # Create clean state
    print("\n[4] Creating clean state...")

    current_time = datetime.now().isoformat()

    # Position data (if any)
    position_data = None
    if len(active_positions) > 0:
        # Take first position (should only be one in One-Way mode)
        pos = active_positions[0]

        position_data = {
            'status': 'OPEN',
            'side': pos['side'],
            'entry_time': current_time,  # Unknown actual entry time
            'entry_candle_time': 'unknown',
            'entry_price': pos['entry_price'],
            'entry_candle_idx': 0,
            'entry_long_prob': 0.0,  # Unknown
            'entry_short_prob': 0.0,  # Unknown
            'probability': 0.0,  # Unknown
            'position_size_pct': 0.0,  # Unknown
            'position_value': pos['quantity'] * pos['entry_price'] / 4,  # Estimate (4x leverage)
            'leveraged_value': pos['quantity'] * pos['entry_price'],
            'quantity': pos['quantity'],
            'order_id': 'adopted_from_exchange'  # Unknown
        }

        print(f"   Adopted position from exchange:")
        print(f"     {pos['side']}: {pos['quantity']:.8f} BTC @ ${pos['entry_price']:,.2f}")
        print(f"     ⚠️  Entry time, signals, and sizing are ESTIMATED")

    new_state = {
        'session_start': current_time,
        'initial_balance': actual_balance,  # Current balance as starting point
        'current_balance': actual_balance,
        'timestamp': current_time,
        'position': position_data,
        'trades': [position_data] if position_data else [],  # Keep current position in trades
        'closed_trades': 0,
        'ledger': [],  # Fresh ledger
        'reconciliation_log': [{
            'timestamp': current_time,
            'event': 'state_reset',
            'reason': 'Comprehensive reset - cleared old testnet/mainnet mixed data',
            'balance': actual_balance,
            'positions_adopted': len(active_positions)
        }],
        'latest_signals': {
            'entry': {},
            'exit': {}
        },
        'stats': {
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl_usd': 0.0,
            'total_pnl_pct': 0.0
        }
    }

    print(f"   ✅ Clean state created")
    print(f"     Session Start: {current_time}")
    print(f"     Initial Balance: ${actual_balance:,.2f}")
    print(f"     Position: {'Yes' if position_data else 'None'}")

    # Save new state
    print("\n[5] Saving new state...")
    with open(STATE_FILE, 'w') as f:
        json.dump(new_state, f, indent=2)
    print(f"✅ State file saved")

    # Summary
    print("\n" + "="*80)
    print("RESET SUMMARY")
    print("="*80)

    print(f"\n📋 Previous State:")
    with open(backup_file, 'r') as f:
        old_state = json.load(f)
    print(f"   Initial Balance: ${old_state.get('initial_balance', 0):,.2f} (was hardcoded)")
    print(f"   Current Balance: ${old_state.get('current_balance', 0):,.2f}")
    print(f"   Trades: {len(old_state.get('trades', []))} total")
    print(f"   Closed Trades: {old_state.get('closed_trades', 0)}")
    print(f"   Total P&L: ${old_state.get('stats', {}).get('total_pnl_usd', 0):,.2f}")

    print(f"\n✨ New State:")
    print(f"   Initial Balance: ${actual_balance:,.2f} (from exchange)")
    print(f"   Current Balance: ${actual_balance:,.2f}")
    print(f"   Trades: {len(new_state['trades'])} (current positions only)")
    print(f"   Closed Trades: 0")
    print(f"   Total P&L: $0.00 (fresh start)")

    print(f"\n📦 Backup Location:")
    print(f"   {backup_file}")

    print(f"\n✅ Reset Complete!")

    print(f"\n📝 Next Steps:")
    print(f"   1. ⚠️  If bot is running, restart it to load new state")
    print(f"   2. Monitor will now show correct balance and return (0%)")
    print(f"   3. New trades will be tracked cleanly from this point")
    print(f"   4. Ledger system ready for tracking bot operations")

    if len(active_positions) > 0:
        print(f"\n   ⚠️  WARNING: Adopted position has ESTIMATED entry time and signals")
        print(f"      Bot will manage it but historical data is incomplete")

    print("="*80)

if __name__ == "__main__":
    main()
