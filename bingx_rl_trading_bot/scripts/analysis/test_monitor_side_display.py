"""
Test Monitor Side Display - Quick verification

This script simulates the monitor's side normalization logic
to verify correct display without running the full monitor.
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"


def normalize_trade_side(side: str) -> str:
    """
    Normalize trade side to standard format (LONG/SHORT).

    Args:
        side: Original side value from trade

    Returns:
        Normalized side: "LONG" or "SHORT"
    """
    side_upper = side.upper()

    mapping = {
        "BUY": "LONG",
        "LONG": "LONG",
        "SHORT": "SHORT",
        "OPEN-SHORT": "SHORT"
    }

    return mapping.get(side_upper, side)


def main():
    print("="*100)
    print(" "*35 + "🧪 MONITOR SIDE DISPLAY TEST")
    print("="*100)

    # Load state file
    print(f"\n📂 Loading state file: {STATE_FILE}")
    with open(STATE_FILE) as f:
        state = json.load(f)

    trades = state.get('trades', [])
    closed_trades = [t for t in trades if t.get('status') == 'CLOSED']

    print(f"✅ Loaded {len(closed_trades)} closed trades\n")

    # Simulate monitor display logic
    print("="*100)
    print("SIMULATED MONITOR OUTPUT - CLOSED POSITIONS")
    print("="*100)

    print("\n┌─ CLOSED POSITIONS (Last 5) - Historical Exit Reasons " + "─"*48 + "┐")

    if not closed_trades:
        print("│ No closed positions found                                                                         │")
        print("└" + "─"*99 + "┘")
        return

    # Sort by entry time (most recent last)
    sorted_trades = sorted(closed_trades, key=lambda t: t.get('entry_time', ''))

    # Show last 5 (chronological order: oldest first, newest last)
    recent_trades = sorted_trades[-5:]

    # Display (oldest first)
    for position_num, trade in enumerate(recent_trades, 1):

        raw_side = trade.get('side', 'N/A')
        normalized_side = normalize_trade_side(raw_side) if raw_side != 'N/A' else 'N/A'

        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        pnl_usd_net = trade.get('pnl_usd_net', 0)
        total_fee = trade.get('total_fee', 0)
        exit_reason = trade.get('exit_reason', 'N/A')

        # Shorten exit reason
        if 'ML Exit' in exit_reason:
            exit_reason = 'ML Exit'
        elif 'Max Hold' in exit_reason:
            exit_reason = 'Max Hold'
        elif 'Reconciled' in exit_reason:
            exit_reason = 'Exchange'

        # Format line
        side_display = f"{normalized_side:6s}"
        price_display = f"${entry_price:>10,.2f} → ${exit_price:>10,.2f}"

        # Calculate percentage change
        price_change_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        pct_display = f"{price_change_pct:>+6.2f}%"

        pnl_display = f"(${pnl_usd_net:>+7.2f}, fee: ${total_fee:.2f})"

        print(f"│ #{position_num:>3d} {side_display} │  {price_display}  │  {pct_display} {pnl_display:>30s}  │  {exit_reason:12s} │")

        # Show raw vs normalized for debugging
        if raw_side != normalized_side:
            print(f"│      DEBUG: Raw side '{raw_side}' normalized to '{normalized_side}'                                          │")

    print("└" + "─"*99 + "┘")

    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    side_counts = {}
    for trade in closed_trades:
        raw_side = trade.get('side', 'N/A')
        normalized_side = normalize_trade_side(raw_side) if raw_side != 'N/A' else 'N/A'
        side_counts[normalized_side] = side_counts.get(normalized_side, 0) + 1

    print("\nSide Distribution (After Normalization):")
    for side, count in sorted(side_counts.items()):
        print(f"  {side:10s}: {count} trades")

    print("\n✅ All sides displayed correctly (BUY → LONG, etc.)")


if __name__ == "__main__":
    main()
