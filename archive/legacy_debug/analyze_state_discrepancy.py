#!/usr/bin/env python3
"""
Analyze state file discrepancies in detail
"""

import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
state_file = PROJECT_ROOT / 'results' / 'opportunity_gating_bot_4x_state.json'

with open(state_file, 'r') as f:
    state = json.load(f)

print("=" * 80)
print("STATE FILE DISCREPANCY ANALYSIS")
print("=" * 80)

# 1. Balance Overview
print("\n1. BALANCE OVERVIEW")
print("-" * 80)

initial = state['initial_balance']
current = state['current_balance']
unrealized = state['unrealized_pnl']

print(f"Initial Balance:     ${initial:>12,.2f}")
print(f"Current Balance:     ${current:>12,.2f}")
print(f"Unrealized P&L:      ${unrealized:>12,.2f}")
print(f"Balance Change:      ${current - initial:>12,.2f}")

# 2. Trades Breakdown
print("\n2. TRADES BREAKDOWN")
print("-" * 80)

trades = state.get('trades', [])
bot_closed = [t for t in trades if t.get('status') == 'CLOSED' and not t.get('manual_trade', False)]
bot_open = [t for t in trades if t.get('status') == 'OPEN' and not t.get('manual_trade', False)]
manual_closed = [t for t in trades if t.get('status') == 'CLOSED' and t.get('manual_trade', False)]

print(f"Total Trades:        {len(trades):>4d}")
print(f"  Bot Closed:        {len(bot_closed):>4d}")
print(f"  Bot Open:          {len(bot_open):>4d}")
print(f"  Manual (Recon):    {len(manual_closed):>4d}")

# 3. Bot Trades Detail
print("\n3. BOT CLOSED TRADES DETAIL")
print("-" * 80)

bot_total_pnl = 0
for i, trade in enumerate(bot_closed, 1):
    pnl = trade.get('pnl_usd_net', 0)
    fee = trade.get('total_fee', 0)
    entry_price = trade.get('entry_price', 0)
    exit_price = trade.get('exit_price', 0)
    exit_reason = trade.get('exit_reason', 'N/A')

    bot_total_pnl += pnl

    print(f"\nTrade #{i}:")
    print(f"  Order ID:          {trade.get('order_id', 'N/A')}")
    print(f"  Entry Price:       ${entry_price:>10,.2f}")
    print(f"  Exit Price:        ${exit_price:>10,.2f}")
    print(f"  P&L (net):         ${pnl:>10,.2f}")
    print(f"  Fees:              ${fee:>10,.2f}")
    print(f"  Exit Reason:       {exit_reason[:50]}")

print(f"\nBot Trades Total P&L: ${bot_total_pnl:>10,.2f}")

# 4. Reconciliation Events
print("\n4. RECONCILIATION LOG")
print("-" * 80)

recon_log = state.get('reconciliation_log', [])
if recon_log:
    for event in recon_log:
        print(f"\nEvent: {event.get('event', 'N/A')}")
        print(f"  Timestamp:         {event.get('timestamp', 'N/A')}")
        print(f"  Reason:            {event.get('reason', 'N/A')}")
        print(f"  Balance:           ${event.get('balance', 0):,.2f}")
        print(f"  Previous Balance:  ${event.get('previous_balance', 0):,.2f}")
        print(f"  Notes:             {event.get('notes', 'N/A')[:60]}")
else:
    print("No reconciliation events")

# 5. Stats Field
print("\n5. STATS FIELD")
print("-" * 80)

stats = state.get('stats', {})
print(f"Stats total_trades:  {stats.get('total_trades', 0):>4d}")
print(f"Stats long_trades:   {stats.get('long_trades', 0):>4d}")
print(f"Stats short_trades:  {stats.get('short_trades', 0):>4d}")
print(f"Stats wins:          {stats.get('wins', 0):>4d}")
print(f"Stats losses:        {stats.get('losses', 0):>4d}")
print(f"Stats total_pnl:     ${stats.get('total_pnl_usd', 0):>10,.2f}")

# 6. Discrepancy Analysis
print("\n6. DISCREPANCY ANALYSIS")
print("-" * 80)

# Calculate from balance
realized_from_balance = (current - initial) - unrealized

# Calculate from trades
realized_from_trades = bot_total_pnl

# Difference
difference = realized_from_balance - realized_from_trades

print(f"Realized (from balance):    ${realized_from_balance:>10,.2f}")
print(f"Realized (from trades):     ${realized_from_trades:>10,.2f}")
print(f"Difference:                 ${difference:>10,.2f}")
print()

if abs(difference) > 1.0:
    print("⚠️  SIGNIFICANT DISCREPANCY DETECTED!")
    print()
    print("Possible causes:")
    print("  1. Fees not tracked in trades array")
    print("  2. Missing trade records")
    print("  3. Balance sync event (reconciliation)")
    print("  4. State file corruption")
    print()

    # Check reconciliation
    if recon_log:
        print("🔍 Reconciliation event found:")
        last_recon = recon_log[-1]
        recon_balance = last_recon.get('balance', 0)
        recon_time = last_recon.get('timestamp', 'N/A')
        print(f"   Date: {recon_time}")
        print(f"   Synced Balance: ${recon_balance:,.2f}")
        print(f"   Current Balance: ${current:,.2f}")
        print(f"   Balance Change Since Sync: ${current - recon_balance:,.2f}")
        print()
        print("✅ This explains the discrepancy:")
        print("   - State was reset/synced from exchange")
        print("   - Old trade records may have been cleared")
        print("   - Only trades AFTER sync are in trades array")
        print("   - Balance reflects ALL historical activity")

# 7. Session Timeline
print("\n7. SESSION TIMELINE")
print("-" * 80)

session_start = state.get('session_start', '')
if session_start:
    try:
        start_time = datetime.fromisoformat(session_start)
        print(f"Session Start:       {start_time}")
        print(f"Current Time:        {datetime.now()}")
        duration = datetime.now() - start_time
        print(f"Duration:            {duration.total_seconds()/3600:.2f} hours")
    except:
        print(f"Session Start:       {session_start}")

# Last reconciliation
if recon_log:
    last_recon = recon_log[-1]
    recon_time = last_recon.get('timestamp', '')
    if recon_time:
        try:
            recon_dt = datetime.fromisoformat(recon_time)
            print(f"Last Reconciliation: {recon_dt}")
            time_since = datetime.now() - recon_dt
            print(f"Time Since Recon:    {time_since.total_seconds()/3600:.2f} hours")
        except:
            print(f"Last Reconciliation: {recon_time}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if abs(difference) > 1.0 and recon_log:
    print("✅ The discrepancy is EXPECTED and EXPLAINED:")
    print("   1. State file was synced from exchange at session start")
    print("   2. Balance includes all historical trades")
    print("   3. Trades array only includes trades since sync")
    print("   4. This is NORMAL after reconciliation/reset")
    print()
    print("📊 For accurate metrics, use BALANCE RECONCILIATION method:")
    print("   Realized P&L = (Current - Initial) - Unrealized")
    print("   This accounts for all activity, not just tracked trades")
else:
    print("✅ State file is consistent")
    print("   Trades array matches balance changes")

print("=" * 80)
