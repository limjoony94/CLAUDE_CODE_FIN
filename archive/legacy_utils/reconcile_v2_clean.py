#!/usr/bin/env python3
"""
Reconcile with V2 API and keep only reconciled trades
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.exchange_reconciliation_v2 import reconcile_state_from_exchange_v2
from api.bingx_client import BingXClient
import yaml
import json
from datetime import datetime

def main():
    # Load API client
    config = yaml.safe_load(open('config/api_keys.yaml'))
    client = BingXClient(
        config['bingx']['mainnet']['api_key'],
        config['bingx']['mainnet']['secret_key'],
        testnet=False
    )

    # Load state
    state_file = 'results/opportunity_gating_bot_4x_state.json'
    with open(state_file) as f:
        state = json.load(f)

    print('='*80)
    print('RECONCILE V2 + CLEANUP')
    print('='*80)
    print(f'\nBefore: {len(state["trades"])} trades')

    # Backup
    backup_file = f'{state_file}.backup_v2_clean_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    with open(backup_file, 'w') as f:
        json.dump(state, f, indent=2)
    print(f'💾 Backup: {backup_file}')

    # V2 Reconciliation
    print('\n' + '='*80)
    updated, new = reconcile_state_from_exchange_v2(state, client, days=7)
    print('='*80)

    # Keep only reconciled
    all_trades = state['trades']
    reconciled = [t for t in all_trades if t.get('exchange_reconciled', False)]
    non_reconciled = [t for t in all_trades if not t.get('exchange_reconciled', False)]

    print(f'\n📋 CLEANUP:')
    print(f'   Reconciled: {len(reconciled)} ✅')
    print(f'   Non-reconciled: {len(non_reconciled)}')

    if non_reconciled:
        print(f'\n🗑️  Removing {len(non_reconciled)} non-reconciled trades...')
        state['trades'] = reconciled

    # Save
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    print(f'\n✅ FINAL RESULT:')
    print(f'   Total trades: {len(state["trades"])}')
    print(f'   All reconciled with exchange ground truth ✅')

    print(f'\nFinal trades:')
    for i, t in enumerate(state['trades'], 1):
        print(f'   {i}. Order: {t.get("order_id")}, History ID: {t.get("position_history_id")}, P&L: ${t.get("pnl_usd_net", 0):.2f}')

    print('\n' + '='*80)

if __name__ == '__main__':
    main()
