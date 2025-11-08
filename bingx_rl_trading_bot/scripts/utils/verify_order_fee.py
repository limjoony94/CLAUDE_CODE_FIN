"""
Verify order fee from exchange for specific order ID

This script queries the exchange directly to get the actual fee for a specific order.
Used to verify P&L calculations and identify fee tracking issues.
"""
import sys
from pathlib import Path
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

def verify_order_fee(order_id: str):
    """
    Query exchange for specific order and display fee information

    Args:
        order_id: Order ID to look up (e.g., '1985136139019960321')
    """
    # Load API credentials
    config_path = PROJECT_ROOT / 'config' / 'api_keys.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    api_key = config['bingx']['mainnet']['api_key']
    secret_key = config['bingx']['mainnet']['secret_key']

    # Initialize client
    client = BingXClient(api_key=api_key, secret_key=secret_key, testnet=False)

    print(f"\n{'='*80}")
    print(f"🔍 VERIFYING ORDER FEE FROM EXCHANGE")
    print(f"{'='*80}")
    print(f"Order ID: {order_id}")
    print(f"Exchange: BingX Mainnet")
    print(f"Symbol: BTC/USDT:USDT")
    print(f"{'='*80}\n")

    # Method 1: fetch_my_trades
    print("📊 Method 1: fetch_my_trades (with larger limit)")
    print("-" * 80)

    try:
        # Try with larger limit to go further back
        trades = client.exchange.fetch_my_trades(
            symbol='BTC/USDT:USDT',
            limit=100  # Increase from 5 to 100
        )

        print(f"✅ Retrieved {len(trades)} recent trades")

        # Search for our order
        found = False
        for trade in trades:
            if str(trade.get('order')) == str(order_id):
                found = True
                print(f"\n✅ FOUND ORDER IN TRADE HISTORY!\n")
                print(f"Trade Details:")
                print(f"  Order ID: {trade.get('order')}")
                print(f"  Price: ${trade.get('price', 0):,.2f}")
                print(f"  Amount: {trade.get('amount', 0):.6f} BTC")
                print(f"  Side: {trade.get('side', 'N/A').upper()}")
                print(f"  Timestamp: {trade.get('datetime', 'N/A')}")

                # Extract fee information
                print(f"\nFee Information:")
                if 'fee' in trade and isinstance(trade['fee'], dict):
                    fee_cost = float(trade['fee'].get('cost', 0))
                    fee_currency = trade['fee'].get('currency', 'USDT')
                    print(f"  ✅ Fee (from 'fee' dict): ${fee_cost:.4f} {fee_currency}")
                else:
                    print(f"  ⚠️  No 'fee' dict found in trade")

                if 'info' in trade and 'commission' in trade['info']:
                    commission = abs(float(trade['info']['commission']))
                    print(f"  ✅ Commission (from 'info'): ${commission:.4f}")
                else:
                    print(f"  ⚠️  No 'commission' found in trade['info']")

                # Show full trade structure for debugging
                print(f"\nFull Trade Structure (for debugging):")
                import json
                print(json.dumps(trade, indent=2, default=str))
                break

        if not found:
            print(f"\n❌ Order {order_id} NOT FOUND in recent {len(trades)} trades")
            print(f"   The order may be older than the API window allows")
            print(f"   Consider using a different API method or checking position history")

    except Exception as e:
        print(f"\n❌ Error fetching trades: {e}")

    print(f"\n{'='*80}")
    print(f"💡 ANALYSIS")
    print(f"{'='*80}")
    print(f"If order was NOT found:")
    print(f"  • Order may be older than API trade history window")
    print(f"  • Try increasing limit further (current: 100)")
    print(f"  • Use position history API instead")
    print(f"  • Check if order was actually filled\n")
    print(f"If fee is $0.00:")
    print(f"  • Fee data may not be included in API response")
    print(f"  • Try fetching order details directly")
    print(f"  • Estimate based on position value × 0.05% (taker fee)")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_order_fee.py <order_id>")
        print("Example: python verify_order_fee.py 1985136139019960321")
        sys.exit(1)

    order_id = sys.argv[1]
    verify_order_fee(order_id)
