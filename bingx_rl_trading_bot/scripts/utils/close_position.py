#!/usr/bin/env python3
"""
Close existing position on BingX exchange
Used for manual position closure before state reset
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.api.bingx_client import BingXClient
import yaml

def close_position():
    """Close all open positions on BingX"""

    print("=" * 80)
    print("🔄 Closing Position on BingX")
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

    print("\n📊 Checking for open positions...")

    # Get current position
    position = client.get_position("BTC-USDT")

    if not position or position.get('positionAmt') == 0:
        print("✅ No open positions found")
        return

    # Display position details
    qty = abs(float(position['positionAmt']))
    side = "LONG" if float(position['positionAmt']) > 0 else "SHORT"
    entry_price = float(position['entryPrice'])
    unrealized_pnl = float(position['unrealizedProfit'])

    print(f"\n⚠️  OPEN POSITION FOUND:")
    print(f"   Side: {side}")
    print(f"   Quantity: {qty} BTC")
    print(f"   Entry Price: ${entry_price:,.2f}")
    print(f"   Unrealized P&L: ${unrealized_pnl:,.2f}")

    # Confirm closure
    print("\n⚠️  This will CLOSE the position with a MARKET order")
    response = input("Continue? (yes/no): ")

    if response.lower() != 'yes':
        print("❌ Cancelled")
        return

    # Close position
    print(f"\n🔄 Closing {side} position...")

    try:
        # For SHORT: BUY to close
        # For LONG: SELL to close
        close_side = "BUY" if side == "SHORT" else "SELL"

        result = client.close_position(
            symbol="BTC-USDT",
            side=side
        )

        print(f"✅ Position closed successfully!")
        print(f"   Order ID: {result.get('order_id', 'N/A')}")

        # Get updated balance
        balance = client.get_balance()
        print(f"\n💰 Final Balance: ${balance:,.2f}")

    except Exception as e:
        print(f"❌ Error closing position: {e}")
        raise

if __name__ == "__main__":
    close_position()
