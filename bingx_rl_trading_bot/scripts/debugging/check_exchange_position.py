"""
Check Actual Exchange Position vs State File
=============================================
거래소 API로 실제 포지션 확인 후 state file과 비교
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

# =============================================================================
# CONFIGURATION
# =============================================================================

STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"
CONFIG_DIR = PROJECT_ROOT / "config"

def load_api_keys():
    """Load API keys"""
    api_keys_file = CONFIG_DIR / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('testnet', {})  # Same key for both testnet/mainnet
    return {}

def load_state():
    """Load state file"""
    with open(STATE_FILE, 'r') as f:
        return json.load(f)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("EXCHANGE POSITION CHECK - State vs Reality")
    print("="*80)

    # Load state
    print("\n[1] Loading state file...")
    state = load_state()
    state_position = state.get('position')

    if state_position:
        print(f"✅ State file shows OPEN position:")
        print(f"   Side: {state_position['side']}")
        print(f"   Entry Time: {state_position['entry_time']}")
        print(f"   Entry Price: ${state_position['entry_price']:,.2f}")
        print(f"   Quantity: {state_position['quantity']:.8f} BTC")
        print(f"   Position Value: ${state_position['position_value']:,.2f}")
        print(f"   Leveraged Value: ${state_position['leveraged_value']:,.2f}")
        print(f"   Order ID: {state_position['order_id']}")
    else:
        print(f"❌ State file shows NO open position")

    # Connect to exchange
    print(f"\n[2] Connecting to BingX...")
    api_config = load_api_keys()
    client = BingXClient(
        api_key=api_config.get('api_key', ''),
        secret_key=api_config.get('secret_key', ''),
        testnet=False  # Changed to Mainnet
    )
    print(f"✅ Connected to BingX (Mainnet)")

    # Get account balance
    print(f"\n[3] Checking account balance...")
    try:
        balance_info = client.get_balance()
        balance = float(balance_info.get('balance', {}).get('availableMargin', 0))
        equity = float(balance_info.get('balance', {}).get('equity', 0))
        unrealized_profit = float(balance_info.get('balance', {}).get('unrealizedProfit', 0))

        print(f"   Available Margin: ${balance:,.2f}")
        print(f"   Equity: ${equity:,.2f}")
        print(f"   Unrealized P&L: ${unrealized_profit:,.2f}")

        state_balance = state.get('current_balance')
        print(f"\n   State File Balance: ${state_balance:,.2f}")

        if abs(balance - state_balance) > 1.0:
            print(f"   ⚠️ MISMATCH: ${abs(balance - state_balance):,.2f} difference")
        else:
            print(f"   ✅ Balance matches")

    except Exception as e:
        print(f"   ❌ Failed to get balance: {e}")

    # Get open positions
    print(f"\n[4] Checking open positions on exchange...")
    try:
        positions = client.get_positions("BTC-USDT")

        if not positions or len(positions) == 0:
            print(f"   ❌ NO OPEN POSITIONS on exchange")

            if state_position:
                print(f"\n   🚨 CRITICAL DISCREPANCY:")
                print(f"      State file: OPEN position")
                print(f"      Exchange: NO position")
                print(f"      → Position may have been closed but state not updated")

        else:
            print(f"   ✅ Found {len(positions)} position(s)")

            for i, pos in enumerate(positions, 1):
                print(f"\n   Position #{i} (from Exchange):")

                symbol = pos.get('symbol', 'N/A')
                position_side = pos.get('positionSide', 'N/A')
                position_amt = float(pos.get('positionAmt', 0))
                entry_price = float(pos.get('avgPrice', 0))
                mark_price = float(pos.get('markPrice', 0))
                unrealized_pnl = float(pos.get('unrealizedProfit', 0))
                leverage = pos.get('leverage', 'N/A')

                print(f"      Symbol: {symbol}")
                print(f"      Side: {position_side}")
                print(f"      Position Amount: {position_amt:.8f} BTC")
                print(f"      Entry Price: ${entry_price:,.2f}")
                print(f"      Mark Price: ${mark_price:,.2f}")
                print(f"      Unrealized P&L: ${unrealized_pnl:,.2f}")
                print(f"      Leverage: {leverage}x")

                # Compare with state
                if state_position:
                    print(f"\n   Comparison with State File:")

                    # Side
                    state_side = state_position['side']
                    if position_side == "BOTH":
                        # One-way mode: check position amount sign
                        actual_side = "LONG" if position_amt > 0 else "SHORT"
                    else:
                        actual_side = position_side

                    if actual_side == state_side:
                        print(f"      Side: ✅ Match ({actual_side})")
                    else:
                        print(f"      Side: ❌ MISMATCH (Exchange: {actual_side}, State: {state_side})")

                    # Entry Price
                    state_entry = state_position['entry_price']
                    price_diff = abs(entry_price - state_entry)
                    if price_diff < 1.0:
                        print(f"      Entry Price: ✅ Match (${entry_price:,.2f})")
                    else:
                        print(f"      Entry Price: ⚠️ DIFF ${price_diff:,.2f} (Exchange: ${entry_price:,.2f}, State: ${state_entry:,.2f})")

                    # Quantity
                    state_qty = state_position['quantity']
                    qty_diff = abs(abs(position_amt) - state_qty)
                    if qty_diff < 0.00001:
                        print(f"      Quantity: ✅ Match ({abs(position_amt):.8f} BTC)")
                    else:
                        print(f"      Quantity: ❌ MISMATCH (Exchange: {abs(position_amt):.8f}, State: {state_qty:.8f})")

                    # P&L calculation
                    if actual_side == "LONG":
                        price_change_pct = (mark_price - entry_price) / entry_price
                    else:
                        price_change_pct = (entry_price - mark_price) / entry_price

                    leveraged_pnl_pct = price_change_pct * 4

                    print(f"\n      Current P&L:")
                    print(f"         Price Change: {price_change_pct*100:+.4f}%")
                    print(f"         Leveraged (4x): {leveraged_pnl_pct*100:+.4f}%")
                    print(f"         Unrealized P&L: ${unrealized_pnl:+,.2f}")

    except Exception as e:
        print(f"   ❌ Failed to get positions: {e}")

    # Get recent orders
    print(f"\n[5] Checking recent orders...")
    try:
        # Check if the order ID in state exists
        if state_position:
            order_id = state_position['order_id']
            print(f"   State file Order ID: {order_id}")
            print(f"   Entry time: {state_position['entry_time']}")

            # Try to get order history
            # Note: BingX API might have limits on historical data

    except Exception as e:
        print(f"   ❌ Failed to get order history: {e}")

    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if state_position and not positions:
        print("🚨 CRITICAL ISSUE:")
        print("   State file shows OPEN position")
        print("   Exchange shows NO position")
        print("")
        print("Possible causes:")
        print("   1. Position was closed manually on exchange")
        print("   2. Position was closed by bot but state not updated")
        print("   3. Position was liquidated")
        print("   4. API connection to wrong account")
        print("")
        print("Action required:")
        print("   1. Clear the position in state file")
        print("   2. Check bot logs for close order")
        print("   3. Review trade history on exchange")

    elif state_position and positions:
        print("✅ Position exists on both state and exchange")
        print("   Check comparison details above for any mismatches")

    elif not state_position and not positions:
        print("✅ No position on both state and exchange")
        print("   System is in sync")

    else:
        print("⚠️ Unusual state:")
        print("   No position in state but position exists on exchange")

    print("="*80)

if __name__ == "__main__":
    main()
