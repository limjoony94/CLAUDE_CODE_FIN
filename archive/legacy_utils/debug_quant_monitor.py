#!/usr/bin/env python3
"""
Quant Monitor Debug Script
Deep analysis of calculation logic and data accuracy
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"

def debug_state():
    """Analyze current state for monitor calculations"""

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    print("="*80)
    print("QUANT MONITOR DEBUG - STATE ANALYSIS")
    print("="*80)

    # Balance Analysis
    print("\n📊 BALANCE ANALYSIS:")
    initial = state.get('initial_balance', 0)
    current = state.get('current_balance', 0)
    net_balance = state.get('net_balance', 0)
    realized_balance = state.get('realized_balance', 0)
    unrealized_pnl = state.get('unrealized_pnl', 0)

    print(f"   Initial Balance:    ${initial:,.2f}")
    print(f"   Current Balance:    ${current:,.2f}")
    print(f"   Net Balance:        ${net_balance:,.2f}")
    print(f"   Realized Balance:   ${realized_balance:,.2f}")
    print(f"   Unrealized P&L:     ${unrealized_pnl:+,.2f}")
    print(f"\n   Balance Change:     ${current - initial:+,.2f}")
    print(f"   Net - Initial:      ${net_balance - initial:+,.2f}")
    print(f"   Realized - Initial: ${realized_balance - initial:+,.2f}")

    # Trade Analysis
    print("\n📈 TRADE ANALYSIS:")
    trades = state.get('trades', [])
    closed_trades = [t for t in trades if t.get('status') == 'CLOSED']

    print(f"   Total Trades:       {len(trades)}")
    print(f"   Closed Trades:      {len(closed_trades)}")
    print(f"   State Closed Count: {state.get('closed_trades', 0)}")

    if closed_trades:
        print("\n   Closed Trades Detail:")
        total_pnl_gross = 0
        total_pnl_net = 0
        total_fees = 0

        for i, trade in enumerate(closed_trades, 1):
            pnl_gross = trade.get('pnl_usd', 0)
            pnl_net = trade.get('pnl_usd_net', 0)
            fee = trade.get('total_fee', 0)

            print(f"      Trade {i}:")
            print(f"         P&L (gross):  ${pnl_gross:+,.2f}")
            print(f"         Fee:          ${fee:+,.2f}")
            print(f"         P&L (net):    ${pnl_net:+,.2f}")
            print(f"         Reconciled:   {trade.get('exchange_reconciled', False)}")
            print(f"         Manual:       {trade.get('manual_trade', False)}")

            total_pnl_gross += pnl_gross
            total_pnl_net += pnl_net
            total_fees += fee

        print(f"\n   TOTALS:")
        print(f"      Total P&L (gross): ${total_pnl_gross:+,.2f}")
        print(f"      Total Fees:        ${total_fees:+,.2f}")
        print(f"      Total P&L (net):   ${total_pnl_net:+,.2f}")

    # Ledger Analysis
    print("\n📔 LEDGER ANALYSIS:")
    ledger = state.get('ledger', [])
    print(f"   Ledger Entries: {len(ledger)}")

    if ledger:
        ledger_sum = sum(entry.get('amount', 0) for entry in ledger)
        print(f"   Ledger Sum:     ${ledger_sum:+,.2f}")

    # Stats Analysis
    print("\n📊 STATS ANALYSIS:")
    stats = state.get('stats', {})
    print(f"   Total Trades:   {stats.get('total_trades', 0)}")
    print(f"   Wins:           {stats.get('wins', 0)}")
    print(f"   Losses:         {stats.get('losses', 0)}")
    print(f"   Total P&L:      ${stats.get('total_pnl_usd', 0):+,.2f}")
    print(f"   Total P&L %:    {stats.get('total_pnl_pct', 0):+.2f}%")

    # Calculation Verification
    print("\n" + "="*80)
    print("CALCULATION VERIFICATION")
    print("="*80)

    # Verify Balance Change calculation
    print("\n1. BALANCE CHANGE (quant_monitor uses 'net_balance'):")
    print(f"   net_balance - initial_balance = ${net_balance - initial:+,.2f}")
    print(f"   current_balance - initial_balance = ${current - initial:+,.2f}")
    print(f"   realized_balance - initial_balance = ${realized_balance - initial:+,.2f}")

    # Verify Realized Return calculation
    print("\n2. REALIZED RETURN (quant_monitor sums pnl_usd from closed trades):")
    if closed_trades:
        pnl_usd_sum = sum(t.get('pnl_usd', 0) for t in closed_trades)
        pnl_net_sum = sum(t.get('pnl_usd_net', 0) for t in closed_trades)

        print(f"   Sum of pnl_usd:     ${pnl_usd_sum:+,.2f} (gross, excludes fees)")
        print(f"   Sum of pnl_usd_net: ${pnl_net_sum:+,.2f} (net, includes fees)")

        print(f"\n   Realized Return (gross): {pnl_usd_sum / initial * 100:+.2f}%")
        print(f"   Realized Return (net):   {pnl_net_sum / initial * 100:+.2f}%")

    # Issues Found
    print("\n" + "="*80)
    print("🔍 POTENTIAL ISSUES")
    print("="*80)

    issues = []

    # Issue 1: Realized return uses pnl_usd (gross) instead of pnl_usd_net
    if closed_trades:
        pnl_usd_sum = sum(t.get('pnl_usd', 0) for t in closed_trades)
        pnl_net_sum = sum(t.get('pnl_usd_net', 0) for t in closed_trades)
        if abs(pnl_usd_sum - pnl_net_sum) > 0.01:
            issues.append({
                'id': 1,
                'severity': 'MEDIUM',
                'issue': 'Realized Return calculation',
                'problem': 'Uses pnl_usd (gross) instead of pnl_usd_net',
                'impact': f'Shows ${pnl_usd_sum:+,.2f} instead of ${pnl_net_sum:+,.2f}',
                'fix': 'Change line 540 to use pnl_usd_net for accurate net returns'
            })

    # Issue 2: Balance change label was wrong (already fixed)
    issues.append({
        'id': 2,
        'severity': 'HIGH',
        'issue': 'Balance Change label',
        'problem': 'Was labeled as "Fees Impact" (incorrect)',
        'impact': 'Users thought fees could be positive',
        'fix': 'FIXED: Changed to "Balance Change (Realized only)"'
    })

    # Issue 3: net_balance vs realized_balance confusion
    if net_balance != realized_balance:
        issues.append({
            'id': 3,
            'severity': 'LOW',
            'issue': 'Balance fields inconsistency',
            'problem': f'net_balance (${net_balance:,.2f}) != realized_balance (${realized_balance:,.2f})',
            'impact': 'Unclear which balance to use for calculations',
            'fix': 'Clarify balance field usage or consolidate'
        })

    # Display issues
    for issue in issues:
        print(f"\n⚠️  Issue #{issue['id']} - Severity: {issue['severity']}")
        print(f"   Problem:  {issue['problem']}")
        print(f"   Impact:   {issue['impact']}")
        print(f"   Fix:      {issue['fix']}")

    print("\n" + "="*80)
    print(f"TOTAL ISSUES FOUND: {len(issues)}")
    print("="*80)

if __name__ == "__main__":
    debug_state()
