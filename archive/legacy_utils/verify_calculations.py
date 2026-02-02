#!/usr/bin/env python3
"""
Verify Monitor Calculations - Compare State File vs Monitor Display
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"

def verify_calculations():
    """Verify all monitor calculations"""

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    print("=" * 80)
    print("STATE FILE VALUES")
    print("=" * 80)
    initial = state['initial_balance']
    current = state['current_balance']
    net = state['net_balance']
    realized = state['realized_balance']
    unrealized = state['unrealized_pnl']

    print(f"initial_balance:    ${initial:,.2f}")
    print(f"current_balance:    ${current:,.2f}")
    print(f"net_balance:        ${net:,.2f}")
    print(f"realized_balance:   ${realized:,.2f}")
    print(f"unrealized_pnl:     ${unrealized:,.2f}")

    print("\n" + "=" * 80)
    print("EXPECTED CALCULATIONS")
    print("=" * 80)

    print("\n1. Balance Change (should use realized_balance):")
    balance_change = realized - initial
    balance_change_pct = (balance_change / initial) * 100 if initial > 0 else 0
    print(f"   realized_balance - initial_balance")
    print(f"   ${realized:,.2f} - ${initial:,.2f} = ${balance_change:+,.2f}")
    print(f"   Percentage: {balance_change_pct:+.2f}%")

    print("\n2. Unrealized P&L Percentage:")
    unrealized_pct = (unrealized / initial) * 100 if initial > 0 else 0
    print(f"   unrealized_pnl / initial_balance")
    print(f"   ${unrealized:,.2f} / ${initial:,.2f} = {unrealized_pct:+.2f}%")

    print("\n3. Total Return Percentage:")
    total_return = net - initial
    total_return_pct = (total_return / initial) * 100 if initial > 0 else 0
    print(f"   (net_balance - initial_balance) / initial_balance")
    print(f"   (${net:,.2f} - ${initial:,.2f}) / ${initial:,.2f}")
    print(f"   = {total_return_pct:+.2f}%")

    print("\n4. Balance Change from current_balance (alternative calculation):")
    current_change = current - initial
    current_change_pct = (current_change / initial) * 100 if initial > 0 else 0
    print(f"   current_balance - initial_balance")
    print(f"   ${current:,.2f} - ${initial:,.2f} = ${current_change:+,.2f}")
    print(f"   Percentage: {current_change_pct:+.2f}%")

    print("\n" + "=" * 80)
    print("COMPARISON: Monitor Display (from user)")
    print("=" * 80)
    print("\nUser reported:")
    print("   ROI (vs Balance): -0.95%")
    print("   P&L: -$43.20")
    print("   Unrealized P&L: -1.1%")
    print("   Balance Change: +1% ($+28.39)")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    issues = []

    # Check Balance Change
    if abs(balance_change - 28.39) > 0.5:
        issues.append({
            'metric': 'Balance Change',
            'expected': f'${balance_change:+,.2f}',
            'displayed': '$+28.39',
            'issue': f'Mismatch: ${abs(balance_change - 28.39):.2f} difference'
        })

    # Check Unrealized P&L %
    if abs(unrealized_pct - (-1.1)) > 0.1:
        issues.append({
            'metric': 'Unrealized P&L %',
            'expected': f'{unrealized_pct:+.2f}%',
            'displayed': '-1.1%',
            'issue': f'Mismatch: {abs(unrealized_pct - (-1.1)):.2f}% difference'
        })

    # Check Total Return %
    if abs(total_return_pct - (-1.1)) > 0.1:
        issues.append({
            'metric': 'Total Return %',
            'expected': f'{total_return_pct:+.2f}%',
            'displayed': '-1.1%',
            'issue': f'Difference: {abs(total_return_pct - (-1.1)):.2f}%'
        })

    if issues:
        print("\n⚠️ MISMATCHES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"\n{i}. {issue['metric']}:")
            print(f"   Expected: {issue['expected']}")
            print(f"   Displayed: {issue['displayed']}")
            print(f"   Issue: {issue['issue']}")
    else:
        print("\n✅ All calculations match!")

    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"\nBalance Change should be: ${balance_change:+,.2f} ({balance_change_pct:+.2f}%)")
    print(f"Unrealized P&L % should be: {unrealized_pct:+.2f}%")
    print(f"Total Return % should be: {total_return_pct:+.2f}%")

if __name__ == "__main__":
    verify_calculations()
