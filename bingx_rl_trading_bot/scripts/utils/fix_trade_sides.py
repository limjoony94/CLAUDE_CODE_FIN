"""
Fix Trade Side Mappings in State File

BingX API uses "LONG"/"SHORT" for position directions.
Some reconciled trades may have incorrect side values like "BUY".

This script:
1. Loads state file
2. Fixes incorrect side values
3. Saves corrected state file
4. Creates backup

Side Mapping Rules:
- "BUY" → "LONG" (opening LONG position)
- "SELL" → Should not exist as side (it's a closing action)
- "Open-Short" → "SHORT" (opening SHORT position)
- "Close-Short" → Should not exist as side (it's a closing action)
- "LONG" → Keep as "LONG" ✅
- "SHORT" → Keep as "SHORT" ✅
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import shutil
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"


def normalize_side(side: str) -> str:
    """
    Normalize side value to "LONG" or "SHORT".

    Args:
        side: Original side value

    Returns:
        Normalized side: "LONG" or "SHORT"
    """
    side_upper = side.upper()

    # Correct mapping
    mapping = {
        "BUY": "LONG",
        "LONG": "LONG",
        "SHORT": "SHORT",
        "OPEN-SHORT": "SHORT",
        "SELL": "UNKNOWN",  # Should not be a position side
        "CLOSE-SHORT": "UNKNOWN"  # Should not be a position side
    }

    return mapping.get(side_upper, side_upper)


def main():
    parser = argparse.ArgumentParser(description='Fix trade side mappings in state file')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Automatically confirm fixes without prompting')
    args = parser.parse_args()

    print("="*100)
    print(" "*35 + "🔧 FIX TRADE SIDES")
    print("="*100)

    # Load state file
    if not STATE_FILE.exists():
        print(f"❌ State file not found: {STATE_FILE}")
        return

    print(f"\n📂 Loading state file: {STATE_FILE}")
    with open(STATE_FILE) as f:
        state = json.load(f)

    trades = state.get('trades', [])
    print(f"✅ Loaded {len(trades)} trades\n")

    # Analyze current sides
    print("="*100)
    print("SIDE ANALYSIS")
    print("="*100)

    side_counts = {}
    issues_found = []

    for i, trade in enumerate(trades):
        side = trade.get('side', 'N/A')
        side_counts[side] = side_counts.get(side, 0) + 1

        normalized = normalize_side(side)

        if side != normalized:
            issues_found.append({
                'index': i,
                'position_id': trade.get('position_id_exchange', trade.get('order_id', 'N/A')),
                'original_side': side,
                'corrected_side': normalized,
                'entry_price': trade.get('entry_price', 0),
                'exit_price': trade.get('exit_price', 0),
                'entry_time': trade.get('entry_time', 'N/A')
            })

    print("\nCurrent Side Distribution:")
    for side, count in sorted(side_counts.items()):
        print(f"  {side:15s}: {count} trades")

    if not issues_found:
        print("\n✅ No side mapping issues found!")
        print("   All trades have correct side values (LONG/SHORT)")
        return

    # Show issues
    print(f"\n⚠️  Found {len(issues_found)} trades with incorrect side values:\n")

    for issue in issues_found:
        print(f"Trade #{issue['index'] + 1}:")
        print(f"  Position ID: {issue['position_id']}")
        print(f"  Entry Time: {issue['entry_time']}")
        print(f"  Entry Price: ${issue['entry_price']:,.2f}")
        print(f"  Exit Price: ${issue['exit_price']:,.2f}")
        print(f"  ❌ Current Side: {issue['original_side']}")
        print(f"  ✅ Correct Side: {issue['corrected_side']}")
        print()

    # Ask for confirmation
    print("="*100)

    if not args.yes:
        try:
            response = input("Fix these side values? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Cancelled by user")
                return
        except EOFError:
            print("❌ Cannot read input. Use --yes flag to auto-confirm.")
            return
    else:
        print("✅ Auto-confirming fixes (--yes flag)")
        print()

    # Create backup
    backup_path = STATE_FILE.parent / f"opportunity_gating_bot_4x_state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    print(f"\n💾 Creating backup: {backup_path.name}")
    shutil.copy(STATE_FILE, backup_path)

    # Fix sides
    print("\n🔧 Fixing side values...")
    fix_count = 0

    for issue in issues_found:
        idx = issue['index']
        trades[idx]['side'] = issue['corrected_side']
        fix_count += 1

    # Update state
    state['trades'] = trades

    # Add reconciliation log entry
    if 'reconciliation_log' not in state:
        state['reconciliation_log'] = []

    state['reconciliation_log'].append({
        'timestamp': datetime.now().isoformat(),
        'event': 'side_mapping_fix',
        'trades_fixed': fix_count,
        'note': f"Fixed {fix_count} trades with incorrect side values (BUY → LONG, etc.)"
    })

    # Save corrected state
    print("💾 Saving corrected state file...")
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"✅ Fixed {fix_count} trades")
    print(f"✅ Backup saved: {backup_path.name}")

    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    new_side_counts = {}
    for trade in trades:
        side = trade.get('side', 'N/A')
        new_side_counts[side] = new_side_counts.get(side, 0) + 1

    print("\nCorrected Side Distribution:")
    for side, count in sorted(new_side_counts.items()):
        print(f"  {side:15s}: {count} trades")

    print("\n✅ All done!")
    print("   Monitor will now display correct side values")
    print("   Backup available if rollback needed")


if __name__ == "__main__":
    main()
