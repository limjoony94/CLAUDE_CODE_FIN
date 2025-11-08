#!/usr/bin/env python3
"""
Fix Duplicate Positions in State File
Removes bot trades that have been replaced by exchange-reconciled trades.
"""
import json
from pathlib import Path
from datetime import datetime

def fix_duplicate_positions(state_file_path):
    """Remove duplicate positions, keeping exchange-reconciled version"""
    print('='*80)
    print('Fixing Duplicate Positions in State File')
    print('='*80)

    # Load state file
    with open(state_file_path, 'r') as f:
        state = json.load(f)

    trades = state.get('trades', [])
    print(f'\nTotal trades before: {len(trades)}')

    # Find position_id_exchange duplicates
    position_ids = {}
    duplicates = []

    for i, trade in enumerate(trades):
        pos_id = trade.get('position_id_exchange')
        if pos_id:
            if pos_id in position_ids:
                duplicates.append({
                    'position_id': pos_id,
                    'first_idx': position_ids[pos_id],
                    'second_idx': i,
                    'first_trade': trades[position_ids[pos_id]],
                    'second_trade': trade
                })
            else:
                position_ids[pos_id] = i

    print(f'Found {len(duplicates)} duplicate positions')

    if not duplicates:
        print('\n✅ No duplicates found!')
        return

    # For each duplicate, keep reconciled version, remove bot version
    indices_to_remove = []

    for dup in duplicates:
        first = dup['first_trade']
        second = dup['second_trade']

        print(f'\n{"="*80}')
        print(f'Duplicate Position ID: {dup["position_id"]}')
        print(f'{"="*80}')

        print(f'\nFirst Trade (Index {dup["first_idx"]}):')
        print(f'  Order ID: {first.get("order_id")}')
        print(f'  Entry Price: ${first.get("entry_price", 0):,.2f}')
        print(f'  Exit Price: ${first.get("exit_price", 0):,.2f}')
        print(f'  Entry Fee: ${first.get("entry_fee", 0):.2f}')
        print(f'  Exit Fee: ${first.get("exit_fee", 0):.2f}')
        print(f'  Total Fee: ${first.get("total_fee", 0):.2f}')
        print(f'  Net P&L: ${first.get("pnl_usd_net", 0):.2f}')
        print(f'  Exchange Reconciled: {first.get("exchange_reconciled", False)}')
        print(f'  Manual Trade: {first.get("manual_trade", False)}')

        print(f'\nSecond Trade (Index {dup["second_idx"]}):')
        print(f'  Order ID: {second.get("order_id")}')
        print(f'  Entry Price: ${second.get("entry_price", 0):,.2f}')
        print(f'  Exit Price: ${second.get("exit_price", 0):,.2f}')
        print(f'  Entry Fee: ${second.get("entry_fee", 0):.2f}')
        print(f'  Exit Fee: ${second.get("exit_fee", 0):.2f}')
        print(f'  Total Fee: ${second.get("total_fee", 0):.2f}')
        print(f'  Net P&L: ${second.get("pnl_usd_net", 0):.2f}')
        print(f'  Exchange Reconciled: {second.get("exchange_reconciled", False)}')
        print(f'  Manual Trade: {second.get("manual_trade", False)}')

        # Decide which to keep
        if second.get('exchange_reconciled'):
            # Second is reconciled - keep it, remove first
            indices_to_remove.append(dup['first_idx'])
            print(f'\n✅ Decision: Keep Second (reconciled), Remove First')
        elif first.get('exchange_reconciled'):
            # First is reconciled - keep it, remove second
            indices_to_remove.append(dup['second_idx'])
            print(f'\n✅ Decision: Keep First (reconciled), Remove Second')
        else:
            # Neither is reconciled - keep the one with fees
            if first.get('entry_fee', 0) > 0 and second.get('entry_fee', 0) == 0:
                indices_to_remove.append(dup['second_idx'])
                print(f'\n✅ Decision: Keep First (has fees), Remove Second')
            elif second.get('entry_fee', 0) > 0 and first.get('entry_fee', 0) == 0:
                indices_to_remove.append(dup['first_idx'])
                print(f'\n✅ Decision: Keep Second (has fees), Remove First')
            else:
                # Keep the newer one
                indices_to_remove.append(dup['first_idx'])
                print(f'\n✅ Decision: Keep Second (newer), Remove First')

    # Remove duplicates
    print(f'\n{"="*80}')
    print(f'Removing {len(indices_to_remove)} duplicate trades...')

    # Sort indices in reverse order to avoid index shifting
    indices_to_remove.sort(reverse=True)

    for idx in indices_to_remove:
        removed_trade = trades.pop(idx)
        print(f'  ❌ Removed: Order ID {removed_trade.get("order_id")} (Index {idx})')

    state['trades'] = trades

    # Backup original
    backup_file = state_file_path.with_suffix('.json.bak')
    with open(state_file_path, 'r') as f:
        with open(backup_file, 'w') as bf:
            bf.write(f.read())

    print(f'\n💾 Backup saved: {backup_file.name}')

    # Save updated state
    with open(state_file_path, 'w') as f:
        json.dump(state, f, indent=2)

    print(f'✅ State file updated!')
    print(f'\nTotal trades after: {len(state["trades"])}')
    print(f'Trades removed: {len(indices_to_remove)}')

    # Summary
    print(f'\n{"="*80}')
    print('Summary:')
    print(f'{"="*80}')

    closed_trades = [t for t in state['trades'] if t.get('status') == 'CLOSED']
    reconciled_trades = [t for t in closed_trades if t.get('exchange_reconciled')]
    bot_trades = [t for t in closed_trades if not t.get('manual_trade')]
    manual_trades = [t for t in closed_trades if t.get('manual_trade')]

    total_net_pnl = sum(t.get('pnl_usd_net', 0) for t in closed_trades)
    bot_net_pnl = sum(t.get('pnl_usd_net', 0) for t in bot_trades)
    manual_net_pnl = sum(t.get('pnl_usd_net', 0) for t in manual_trades)

    print(f'\nClosed Trades: {len(closed_trades)}')
    print(f'  - Bot Trades: {len(bot_trades)} (Net P&L: ${bot_net_pnl:.2f})')
    print(f'  - Manual Trades: {len(manual_trades)} (Net P&L: ${manual_net_pnl:.2f})')
    print(f'Exchange Reconciled: {len(reconciled_trades)}')
    print(f'\nTotal Net P&L: ${total_net_pnl:.2f}')
    print(f'{"="*80}')

if __name__ == '__main__':
    state_file = Path(__file__).parent.parent.parent / 'results' / 'opportunity_gating_bot_4x_state.json'
    fix_duplicate_positions(state_file)
