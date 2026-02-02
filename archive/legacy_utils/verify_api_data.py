#!/usr/bin/env python3
"""
Verify API Data - Get Ground Truth from Exchange
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"

def get_api_ground_truth():
    """Get actual data from exchange API"""

    print("=" * 80)
    print("EXCHANGE API - GROUND TRUTH DATA")
    print("=" * 80)

    # Initialize client
    client = BingXClient()

    # Get account balance
    print("\n📊 ACCOUNT BALANCE (from API):")
    try:
        balance_info = client.fetch_balance()
        print(f"   Available Balance: ${balance_info['free']:,.2f}")
        print(f"   Total Balance: ${balance_info['total']:,.2f}")
        print(f"   Used (Margin): ${balance_info['used']:,.2f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        balance_info = None

    # Get positions
    print("\n📈 POSITIONS (from API):")
    try:
        positions = client.fetch_positions()
        if positions:
            for pos in positions:
                if pos['contracts'] > 0:  # Only show open positions
                    print(f"   Symbol: {pos['symbol']}")
                    print(f"   Side: {pos['side']}")
                    print(f"   Size: {pos['contracts']} contracts")
                    print(f"   Entry Price: ${pos['entryPrice']:,.2f}")
                    print(f"   Mark Price: ${pos['markPrice']:,.2f}")
                    print(f"   Unrealized P&L: ${pos['unrealizedPnl']:,.2f}")
                    print(f"   Liquidation Price: ${pos.get('liquidationPrice', 0):,.2f}")
        else:
            print("   No open positions")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        positions = []

    # Load state file for comparison
    print("\n" + "=" * 80)
    print("STATE FILE DATA (for comparison)")
    print("=" * 80)

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    print(f"\n   initial_balance: ${state.get('initial_balance', 0):,.2f}")
    print(f"   current_balance: ${state.get('current_balance', 0):,.2f}")
    print(f"   net_balance: ${state.get('net_balance', 0):,.2f}")
    print(f"   realized_balance: ${state.get('realized_balance', 0):,.2f}")
    print(f"   unrealized_pnl: ${state.get('unrealized_pnl', 0):,.2f}")

    position = state.get('position', {})
    if position and position.get('status') == 'OPEN':
        print(f"\n   Position Status: {position.get('status')}")
        print(f"   Position Side: {position.get('side')}")
        print(f"   Entry Price: ${position.get('entry_price', 0):,.2f}")
        print(f"   Quantity: {position.get('quantity', 0)}")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: API vs STATE FILE")
    print("=" * 80)

    if balance_info:
        api_balance = balance_info['total']
        state_current = state.get('current_balance', 0)
        diff_balance = api_balance - state_current

        print(f"\n💰 Balance:")
        print(f"   API Total Balance: ${api_balance:,.2f}")
        print(f"   State current_balance: ${state_current:,.2f}")
        print(f"   Difference: ${diff_balance:+,.2f}")
        if abs(diff_balance) > 1.0:
            print(f"   ⚠️ WARNING: Significant difference!")

    if positions:
        for pos in positions:
            if pos['contracts'] > 0:
                api_unrealized = pos['unrealizedPnl']
                state_unrealized = state.get('unrealized_pnl', 0)
                diff_unrealized = api_unrealized - state_unrealized

                print(f"\n📊 Unrealized P&L:")
                print(f"   API Unrealized P&L: ${api_unrealized:,.2f}")
                print(f"   State unrealized_pnl: ${state_unrealized:,.2f}")
                print(f"   Difference: ${diff_unrealized:+,.2f}")
                if abs(diff_unrealized) > 1.0:
                    print(f"   ⚠️ WARNING: Significant difference!")

    # Calculate correct metrics using API data
    print("\n" + "=" * 80)
    print("CORRECT METRICS (using API data)")
    print("=" * 80)

    if balance_info and positions:
        initial_balance = state.get('initial_balance', 0)
        api_total_balance = balance_info['total']
        api_unrealized_pnl = sum(p['unrealizedPnl'] for p in positions if p['contracts'] > 0)

        # Net balance = total balance (which already includes unrealized)
        # OR if total balance doesn't include unrealized, we need to add it
        # Let's check by comparing with state

        print(f"\n   Initial Balance: ${initial_balance:,.2f}")
        print(f"   API Total Balance: ${api_total_balance:,.2f}")
        print(f"   API Unrealized P&L: ${api_unrealized_pnl:,.2f}")

        # Calculate metrics
        print(f"\n   CALCULATIONS:")

        # Check if API total balance includes unrealized
        # Method 1: Total balance is equity (includes unrealized)
        print(f"\n   Method 1 (Total Balance = Equity):")
        total_return_1 = (api_total_balance - initial_balance) / initial_balance * 100
        print(f"      Total Return: {total_return_1:+.2f}%")
        print(f"      Formula: (${api_total_balance:,.2f} - ${initial_balance:,.2f}) / ${initial_balance:,.2f}")

        # Method 2: Total balance is cash, need to add unrealized
        print(f"\n   Method 2 (Total Balance = Cash, add Unrealized):")
        net_balance_2 = api_total_balance + api_unrealized_pnl
        total_return_2 = (net_balance_2 - initial_balance) / initial_balance * 100
        print(f"      Net Balance: ${net_balance_2:,.2f}")
        print(f"      Total Return: {total_return_2:+.2f}%")
        print(f"      Formula: (${net_balance_2:,.2f} - ${initial_balance:,.2f}) / ${initial_balance:,.2f}")

        # Unrealized P&L percentage
        unrealized_pct = (api_unrealized_pnl / initial_balance) * 100
        print(f"\n   Unrealized P&L %:")
        print(f"      {unrealized_pct:+.2f}%")
        print(f"      Formula: ${api_unrealized_pnl:,.2f} / ${initial_balance:,.2f}")

if __name__ == "__main__":
    get_api_ground_truth()
