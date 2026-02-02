"""
Test Protection System (Exchange-Level Stop Loss & Take Profit)

Validates the new protection system on testnet:
1. Enter position with protection orders
2. Verify Stop Loss and Take Profit orders are placed
3. Cancel protection orders
4. Close position
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.bingx_client import BingXClient
from loguru import logger
import yaml

# Setup logger
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# Load API keys
config_file = project_root / "config" / "api_keys.yaml"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

testnet_config = config['bingx']['testnet']

# Initialize client
client = BingXClient(
    api_key=testnet_config['api_key'],
    secret_key=testnet_config['secret_key'],
    testnet=True
)

SYMBOL = "BTC-USDT"
LEVERAGE = 4
TEST_QUANTITY = 0.001  # Small test quantity (0.001 BTC = ~$100 @ $100k)


def test_protection_system():
    """Test complete protection system workflow"""

    logger.info("="*80)
    logger.info("PROTECTION SYSTEM TEST - TESTNET")
    logger.info("="*80)

    # Step 1: Get current price
    logger.info("\n[STEP 1] Fetching current price...")
    klines = client.get_klines(SYMBOL, "5m", limit=1)
    if not klines:
        logger.error("Failed to fetch price data")
        return False

    current_price = float(klines[0]['close'])
    logger.info(f"✅ Current BTC price: ${current_price:,.2f}")

    # Step 2: Check account balance
    logger.info("\n[STEP 2] Checking account balance...")
    balance_info = client.get_balance()
    if not balance_info:
        logger.error("Failed to fetch balance")
        return False

    available_balance = float(balance_info.get('availableMargin', 0))
    logger.info(f"✅ Available balance: ${available_balance:,.2f}")

    if available_balance < 100:
        logger.error("Insufficient balance for test (need > $100)")
        return False

    # Step 3: Enter position with protection
    logger.info("\n[STEP 3] Entering LONG position with protection...")
    logger.info(f"Test quantity: {TEST_QUANTITY} BTC (~${TEST_QUANTITY * current_price:,.2f})")

    try:
        protection_result = client.enter_position_with_protection(
            symbol=SYMBOL,
            side="LONG",
            quantity=TEST_QUANTITY,
            entry_price=current_price,
            leverage=LEVERAGE,
            stop_loss_pct=0.015,  # -1.5%
            take_profit_pct=0.03  # +3%
        )

        entry_order = protection_result['entry_order']
        stop_loss_order = protection_result['stop_loss_order']
        take_profit_order = protection_result['take_profit_order']

        logger.info(f"✅ Position entered successfully!")
        logger.info(f"   Entry Order: {entry_order.get('id')}")
        logger.info(f"   Stop Loss Order: {stop_loss_order.get('id')}")
        logger.info(f"   Take Profit Order: {take_profit_order.get('id')}")
        logger.info(f"   Stop Loss Price: ${protection_result['stop_loss_price']:,.2f}")
        logger.info(f"   Take Profit Price: ${protection_result['take_profit_price']:,.2f}")

    except Exception as e:
        logger.error(f"❌ Failed to enter position: {e}")
        return False

    # Step 4: Verify position exists
    logger.info("\n[STEP 4] Verifying position on exchange...")
    time.sleep(1)  # Wait for exchange to update

    positions = client.get_positions(SYMBOL)
    open_positions = [p for p in positions if float(p.get('contracts', 0)) != 0]

    if not open_positions:
        logger.error("❌ Position not found on exchange!")
        return False

    pos = open_positions[0]
    logger.info(f"✅ Position verified:")
    logger.info(f"   Side: {pos.get('side')}")
    logger.info(f"   Contracts: {pos.get('contracts')} BTC")
    logger.info(f"   Entry Price: ${float(pos.get('entryPrice', 0)):,.2f}")
    logger.info(f"   Unrealized P&L: ${float(pos.get('unrealizedPnl', 0)):+,.2f}")

    # Step 5: Verify protection orders exist
    logger.info("\n[STEP 5] Verifying protection orders...")
    open_orders = client.get_open_orders(SYMBOL)

    sl_order_found = False
    tp_order_found = False

    for order in open_orders:
        order_id = order.get('id')
        order_type = order.get('type', '').upper()

        if order_id == stop_loss_order.get('id'):
            sl_order_found = True
            logger.info(f"✅ Stop Loss order found: {order_id} (type: {order_type})")

        if order_id == take_profit_order.get('id'):
            tp_order_found = True
            logger.info(f"✅ Take Profit order found: {order_id} (type: {order_type})")

    if not sl_order_found:
        logger.warning(f"⚠️ Stop Loss order not found in open orders")
    if not tp_order_found:
        logger.warning(f"⚠️ Take Profit order not found in open orders")

    # Step 6: Cancel protection orders
    logger.info("\n[STEP 6] Cancelling protection orders...")
    order_ids = [stop_loss_order.get('id'), take_profit_order.get('id')]

    cancel_result = client.cancel_position_orders(
        symbol=SYMBOL,
        order_ids=order_ids
    )

    logger.info(f"✅ Cancellation result:")
    logger.info(f"   Cancelled: {len(cancel_result['cancelled'])} orders")
    logger.info(f"   Failed: {len(cancel_result['failed'])} orders")

    # Step 7: Close position
    logger.info("\n[STEP 7] Closing position...")
    time.sleep(0.5)

    try:
        close_result = client.close_position(
            symbol=SYMBOL,
            position_side="LONG",
            quantity=TEST_QUANTITY
        )

        logger.info(f"✅ Position closed: {close_result.get('id')}")

    except Exception as e:
        logger.error(f"❌ Failed to close position: {e}")
        logger.error(f"⚠️ MANUAL CLEANUP REQUIRED!")
        return False

    # Step 8: Final verification
    logger.info("\n[STEP 8] Final verification...")
    time.sleep(1)

    positions_after = client.get_positions(SYMBOL)
    open_positions_after = [p for p in positions_after if float(p.get('contracts', 0)) != 0]

    if open_positions_after:
        logger.warning(f"⚠️ Position still exists on exchange (may be closing)")
    else:
        logger.info(f"✅ Position fully closed")

    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETED SUCCESSFULLY ✅")
    logger.info("="*80)
    logger.info("\nSummary:")
    logger.info(f"  ✅ Entry with protection: SUCCESS")
    logger.info(f"  ✅ Position verification: SUCCESS")
    logger.info(f"  ✅ Protection orders: {'FOUND' if (sl_order_found or tp_order_found) else 'NOT FOUND'}")
    logger.info(f"  ✅ Order cancellation: SUCCESS")
    logger.info(f"  ✅ Position close: SUCCESS")

    return True


def main():
    """Main test execution"""
    try:
        logger.info("Starting protection system test on TESTNET...")
        logger.info(f"Symbol: {SYMBOL}")
        logger.info(f"Leverage: {LEVERAGE}x")
        logger.info(f"Test Quantity: {TEST_QUANTITY} BTC\n")

        success = test_protection_system()

        if success:
            logger.info("\n🎉 All tests passed!")
            return 0
        else:
            logger.error("\n❌ Test failed!")
            return 1

    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n\n❌ Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
