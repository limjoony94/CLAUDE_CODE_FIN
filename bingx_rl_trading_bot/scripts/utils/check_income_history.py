#!/usr/bin/env python3
"""
Check BingX income history to find source of wallet change
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient
import yaml

# Load config
with open(PROJECT_ROOT / 'config' / 'api_keys.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize BingX client
client = BingXClient(
    api_key=config['bingx']['mainnet']['api_key'],
    secret_key=config['bingx']['mainnet']['secret_key'],
    testnet=False
)

print("=" * 80)
print("INCOME HISTORY CHECK")
print("=" * 80)
print()

# Check from reset time
from datetime import timezone
reset_time = datetime.fromisoformat("2025-10-24T23:30:01.284839+00:00")
now = datetime.now(timezone.utc)

print(f"Reset time (UTC): {reset_time}")
print(f"Current time (UTC): {now}")
print(f"Period: {(now - reset_time).total_seconds() / 3600:.1f} hours")
print()

try:
    # Get income history
    # BingX API: GET /openApi/swap/v2/user/income
    # incomeType: FUNDING_FEE, REALIZED_PNL, COMMISSION, TRANSFER, etc.

    income_history = client.exchange.fetch_income_history(
        symbol='BTC/USDT:USDT',
        since=int(reset_time.timestamp() * 1000),
        limit=100
    )

    print(f"📊 Income Records since reset: {len(income_history)}")
    print()

    if not income_history:
        print("   No income records found!")
        print()
        print("   Possible reasons:")
        print("   1. Exchange doesn't provide income API")
        print("   2. Records not available yet")
        print("   3. No actual income during period")
        sys.exit(0)

    # Categorize income
    funding_fees = []
    realized_pnl = []
    commission = []
    transfers = []
    other = []

    total = 0.0

    for record in income_history:
        income_type = record.get('info', {}).get('incomeType', 'UNKNOWN')
        amount = float(record.get('amount', 0))
        timestamp = record.get('timestamp', 0)

        total += amount

        if income_type == 'FUNDING_FEE':
            funding_fees.append((timestamp, amount))
        elif income_type == 'REALIZED_PNL':
            realized_pnl.append((timestamp, amount))
        elif income_type == 'COMMISSION':
            commission.append((timestamp, amount))
        elif income_type == 'TRANSFER':
            transfers.append((timestamp, amount))
        else:
            other.append((timestamp, income_type, amount))

    # Display by category
    print("=" * 80)
    print("INCOME BREAKDOWN")
    print("=" * 80)
    print()

    if funding_fees:
        print(f"💰 FUNDING FEES ({len(funding_fees)} records):")
        funding_total = sum(amt for _, amt in funding_fees)
        for ts, amt in sorted(funding_fees):
            dt = datetime.fromtimestamp(ts / 1000)
            print(f"   {dt.strftime('%Y-%m-%d %H:%M:%S')}: ${amt:>+10.6f}")
        print(f"   Total: ${funding_total:>+10.2f}")
        print()

    if realized_pnl:
        print(f"📈 REALIZED P&L ({len(realized_pnl)} records):")
        pnl_total = sum(amt for _, amt in realized_pnl)
        for ts, amt in sorted(realized_pnl):
            dt = datetime.fromtimestamp(ts / 1000)
            print(f"   {dt.strftime('%Y-%m-%d %H:%M:%S')}: ${amt:>+10.2f}")
        print(f"   Total: ${pnl_total:>+10.2f}")
        print()

    if commission:
        print(f"💸 COMMISSION ({len(commission)} records):")
        commission_total = sum(amt for _, amt in commission)
        for ts, amt in sorted(commission):
            dt = datetime.fromtimestamp(ts / 1000)
            print(f"   {dt.strftime('%Y-%m-%d %H:%M:%S')}: ${amt:>+10.2f}")
        print(f"   Total: ${commission_total:>+10.2f}")
        print()

    if transfers:
        print(f"🔄 TRANSFERS ({len(transfers)} records):")
        transfer_total = sum(amt for _, amt in transfers)
        for ts, amt in sorted(transfers):
            dt = datetime.fromtimestamp(ts / 1000)
            print(f"   {dt.strftime('%Y-%m-%d %H:%M:%S')}: ${amt:>+10.2f}")
        print(f"   Total: ${transfer_total:>+10.2f}")
        print()

    if other:
        print(f"❓ OTHER ({len(other)} records):")
        other_total = sum(amt for _, _, amt in other)
        for ts, itype, amt in sorted(other):
            dt = datetime.fromtimestamp(ts / 1000)
            print(f"   {dt.strftime('%Y-%m-%d %H:%M:%S')} [{itype}]: ${amt:>+10.2f}")
        print(f"   Total: ${other_total:>+10.2f}")
        print()

    print("=" * 80)
    print(f"💵 TOTAL INCOME: ${total:>+10.2f}")
    print("=" * 80)
    print()

    # Compare with wallet change
    print("COMPARISON:")
    print(f"   Wallet change (from State): ${54.20:>+10.2f}")
    print(f"   Income total (from API):    ${total:>+10.2f}")
    print(f"   Difference:                 ${abs(54.20 - total):>10.2f}")
    print()

    if abs(54.20 - total) < 0.50:
        print("   ✅ MATCH - Income history explains wallet change!")
    else:
        print("   ⚠️  MISMATCH - Income history doesn't fully explain wallet change")
        print("   Possible reasons:")
        print("   - API doesn't return all income types")
        print("   - Manual transfer not recorded")
        print("   - Timing difference")

except AttributeError:
    print("❌ BingX API doesn't support fetch_income_history()")
    print()
    print("Alternative: Check trade history for closed positions")

    try:
        # Try getting closed orders
        orders = client.exchange.fetch_closed_orders(
            symbol='BTC/USDT:USDT',
            since=int(reset_time.timestamp() * 1000),
            limit=100
        )

        print(f"📋 Closed Orders since reset: {len(orders)}")
        print()

        if orders:
            total_pnl = 0
            for order in orders:
                if order.get('info', {}).get('profit'):
                    profit = float(order['info']['profit'])
                    total_pnl += profit
                    print(f"   Order {order['id']}: ${profit:>+10.2f}")

            print()
            print(f"   Total P&L: ${total_pnl:>+10.2f}")
            print(f"   Wallet change: ${54.20:>+10.2f}")
            print(f"   Difference: ${abs(54.20 - total_pnl):>10.2f}")

    except Exception as e2:
        print(f"   Also failed to get closed orders: {e2}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
