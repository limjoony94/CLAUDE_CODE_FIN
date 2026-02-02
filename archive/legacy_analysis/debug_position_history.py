"""
Debug Position History - Compare API vs State File

BingX API Position Side Values:
- "BUY": Open LONG position → should map to "LONG"
- "SELL": Close LONG position → not a position side
- Open-Short: Opens SHORT position → should map to "SHORT"
- Close-Short: Close SHORT position → not a position side

This script:
1. Fetches position history from BingX API
2. Compares with state file trades
3. Identifies side mapping issues
4. Suggests corrections
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient


def load_api_keys() -> Dict:
    """Load API keys from config"""
    config_path = PROJECT_ROOT / "config" / "api_keys.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_state_file() -> Dict:
    """Load state file"""
    state_path = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"
    with open(state_path) as f:
        return json.load(f)


def format_timestamp(ts_ms: int) -> str:
    """Format Unix timestamp (ms) to readable string"""
    return datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


def analyze_position_side(position: Dict) -> Dict:
    """
    Analyze BingX position and determine correct side mapping

    BingX API Fields:
    - positionSide: "LONG" or "SHORT" (position direction)
    - side: "BUY", "SELL", "Open-Short", "Close-Short" (action taken)

    Correct Mapping:
    - If positionSide exists → use it directly
    - If side == "BUY" and no positionSide → assume "LONG"
    - If side == "Open-Short" and no positionSide → assume "SHORT"
    """
    pos_side = position.get('positionSide', '')
    action_side = position.get('side', '')

    # Determine correct position side
    if pos_side in ['LONG', 'SHORT']:
        correct_side = pos_side
        confidence = "HIGH"
    elif action_side == "BUY":
        correct_side = "LONG"
        confidence = "MEDIUM (inferred from BUY action)"
    elif action_side in ["Open-Short", "SHORT"]:
        correct_side = "SHORT"
        confidence = "MEDIUM (inferred from SHORT action)"
    else:
        correct_side = "UNKNOWN"
        confidence = "LOW"

    return {
        'positionSide': pos_side,
        'actionSide': action_side,
        'correctSide': correct_side,
        'confidence': confidence
    }


def main():
    """Main debug function"""
    print("="*100)
    print(" "*35 + "🔍 POSITION HISTORY DEBUG")
    print("="*100)

    # Load configuration
    print("\n📂 Loading configuration...")
    config = load_api_keys()
    mainnet_config = config['bingx']['mainnet']

    # Initialize client
    print("🔌 Connecting to BingX API (Mainnet)...")
    client = BingXClient(
        api_key=mainnet_config['api_key'],
        secret_key=mainnet_config['secret_key'],
        testnet=False
    )

    # Load state file
    print("📋 Loading state file...")
    state = load_state_file()
    state_trades = {t.get('position_id_exchange', t.get('order_id')): t for t in state.get('trades', [])}

    print(f"\n✅ State file loaded: {len(state_trades)} trades")

    # Fetch position history from API
    print("\n📡 Fetching position history from BingX API...")
    try:
        # Get all closed positions (last 100)
        response = client.perpetual_get_all_orders(symbol="BTC-USDT", limit=100)

        if not response or 'data' not in response:
            print("❌ Failed to fetch position history")
            return

        positions = response['data'].get('orders', [])
        print(f"✅ Fetched {len(positions)} positions from API")

        # Filter only closed positions
        closed_positions = [p for p in positions if p.get('status') == 'FILLED']
        print(f"📊 Found {len(closed_positions)} FILLED positions")

        # Analyze each position
        print("\n" + "="*100)
        print("POSITION ANALYSIS - API vs STATE FILE")
        print("="*100)

        issues_found = []

        for i, pos in enumerate(closed_positions[:10], 1):  # Check last 10
            order_id = pos.get('orderId')
            side_analysis = analyze_position_side(pos)

            print(f"\n{i}. Order ID: {order_id}")
            print(f"   Time: {format_timestamp(pos.get('time', 0))}")
            print(f"   Price: ${pos.get('price', 0):,.2f}")
            print(f"   Quantity: {pos.get('executedQty', 0)}")
            print(f"   API positionSide: {side_analysis['positionSide']}")
            print(f"   API actionSide: {side_analysis['actionSide']}")
            print(f"   ✅ Correct Side: {side_analysis['correctSide']} ({side_analysis['confidence']})")

            # Check if in state file
            if order_id in state_trades:
                state_trade = state_trades[order_id]
                state_side = state_trade.get('side')
                print(f"   📋 State File Side: {state_side}")

                if state_side != side_analysis['correctSide']:
                    issue = {
                        'order_id': order_id,
                        'api_side': side_analysis['correctSide'],
                        'state_side': state_side,
                        'api_position_side': side_analysis['positionSide'],
                        'api_action_side': side_analysis['actionSide']
                    }
                    issues_found.append(issue)
                    print(f"   ⚠️  MISMATCH: State '{state_side}' vs Correct '{side_analysis['correctSide']}'")
                else:
                    print(f"   ✅ MATCH: Side is correct")
            else:
                print(f"   ℹ️  Not found in state file (may be old trade)")

        # Summary
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)

        if issues_found:
            print(f"\n⚠️  Found {len(issues_found)} side mapping issues:\n")
            for issue in issues_found:
                print(f"Order ID: {issue['order_id']}")
                print(f"  API Position Side: {issue['api_position_side']}")
                print(f"  API Action Side: {issue['api_action_side']}")
                print(f"  Correct Side: {issue['api_side']}")
                print(f"  State Side: {issue['state_side']} ❌")
                print(f"  Fix: Change '{issue['state_side']}' → '{issue['api_side']}'")
                print()
        else:
            print("\n✅ No side mapping issues found!")

        # Recommendations
        print("\n" + "="*100)
        print("RECOMMENDATIONS")
        print("="*100)

        print("""
1. BingX API Side Mapping Rules:
   - positionSide field is reliable when present → use directly
   - If side == "BUY" → map to "LONG"
   - If side == "Open-Short" → map to "SHORT"
   - If side == "SELL" or "Close-Short" → not a position side (exit action)

2. Fix Trade Reconciliation:
   - Update reconcile_trades.py to use correct mapping
   - Map "BUY" → "LONG" when creating reconciled trades
   - Map "Open-Short" → "SHORT" when creating reconciled trades

3. State File Corrections:
   - Manually fix existing trades with incorrect sides
   - Or run reconciliation again with fixed mapping
        """)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
