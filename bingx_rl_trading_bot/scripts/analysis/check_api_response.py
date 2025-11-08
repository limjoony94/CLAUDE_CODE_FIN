"""
Check BingX API Position History Response

This script fetches position history and shows the actual API response structure
to identify side mapping issues.
"""

import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient


def main():
    print("="*100)
    print(" "*35 + "🔍 API RESPONSE CHECK")
    print("="*100)

    # Load API keys
    print("\n📂 Loading configuration...")
    with open(PROJECT_ROOT / 'config' / 'api_keys.yaml') as f:
        config = yaml.safe_load(f)
        api_config = config['bingx']['mainnet']

    # Initialize client
    print("🔌 Connecting to BingX API...")
    client = BingXClient(
        api_key=api_config['api_key'],
        secret_key=api_config['secret_key'],
        testnet=False
    )

    # Fetch position history (last 7 days)
    print("📡 Fetching position history...")
    since = datetime.now() - timedelta(days=7)
    start_ts = int(since.timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000)

    try:
        response = client.exchange.swap_v1_private_get_trade_positionhistory({
            'symbol': 'BTC-USDT',
            'startTs': start_ts,
            'endTs': end_ts,
            'limit': 10
        })

        if response.get('code') == '0' and 'data' in response:
            positions = response['data'].get('positionHistory', [])
            print(f"✅ Fetched {len(positions)} positions\n")

            # Show structure of first position (if exists)
            if positions:
                print("="*100)
                print("FIRST POSITION (Raw API Response)")
                print("="*100)
                print(json.dumps(positions[0], indent=2))

                print("\n" + "="*100)
                print("ALL POSITIONS (Summary)")
                print("="*100)

                for i, pos in enumerate(positions, 1):
                    pos_id = pos.get('positionId')
                    pos_side = pos.get('positionSide', 'N/A')
                    action_side = pos.get('side', 'N/A')
                    entry_price = float(pos.get('avgPrice', 0))
                    exit_price = float(pos.get('avgClosePrice', 0))
                    net_profit = float(pos.get('netProfit', 0))
                    open_time = datetime.fromtimestamp(int(pos.get('openTime', 0)) / 1000)

                    print(f"\n{i}. Position ID: {pos_id}")
                    print(f"   positionSide: {pos_side}")
                    print(f"   side: {action_side}")
                    print(f"   Entry: ${entry_price:,.2f}")
                    print(f"   Exit: ${exit_price:,.2f}")
                    print(f"   Net P&L: ${net_profit:.2f}")
                    print(f"   Open Time: {open_time}")

                print("\n" + "="*100)
                print("SIDE MAPPING ANALYSIS")
                print("="*100)

                side_mapping_rules = {
                    'positionSide == "LONG"': 'Use "LONG"',
                    'positionSide == "SHORT"': 'Use "SHORT"',
                    'side == "BUY"': 'Map to "LONG"',
                    'side == "SELL"': 'Not a position side (closing LONG)',
                    'side == "Open-Short"': 'Map to "SHORT"',
                    'side == "Close-Short"': 'Not a position side (closing SHORT)'
                }

                print("\nCorrect Mapping Rules:")
                for condition, action in side_mapping_rules.items():
                    print(f"  {condition:30s} → {action}")

                print("\nActual Values in API Response:")
                unique_pos_sides = set(p.get('positionSide', 'N/A') for p in positions)
                unique_action_sides = set(p.get('side', 'N/A') for p in positions)

                print(f"  positionSide values: {sorted(unique_pos_sides)}")
                print(f"  side values: {sorted(unique_action_sides)}")

            else:
                print("⚠️  No positions found in last 7 days")

        else:
            print(f"❌ Unexpected response: {response}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
