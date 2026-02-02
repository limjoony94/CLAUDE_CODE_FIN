#!/usr/bin/env python3
"""
Test V2 Reconciliation Integration in Bot Startup
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.exchange_reconciliation_v2 import reconcile_state_from_exchange_v2
from src.api.bingx_client import BingXClient
import yaml
import json
from datetime import datetime

def test_v2_integration():
    """Test V2 reconciliation as it would run in bot startup"""

    print('='*80)
    print('V2 RECONCILIATION BOT INTEGRATION TEST')
    print('='*80)

    # Load API client
    print('\n1. Loading API client...')
    config = yaml.safe_load(open('config/api_keys.yaml'))
    client = BingXClient(
        config['bingx']['mainnet']['api_key'],
        config['bingx']['mainnet']['secret_key'],
        testnet=False
    )
    print('✅ Client loaded')

    # Load state
    print('\n2. Loading state file...')
    state_file = 'results/opportunity_gating_bot_4x_state.json'
    with open(state_file) as f:
        state = json.load(f)

    trades_before = len(state.get('trades', []))
    print(f'✅ State loaded: {trades_before} trades')

    # Run V2 reconciliation (as bot would)
    print('\n3. Running V2 reconciliation...')
    print('🔄 Reconciling state from exchange (V2 - Position History API)...')

    try:
        updated_count, new_count = reconcile_state_from_exchange_v2(
            state=state,
            api_client=client,
            bot_start_time=state.get('start_time'),
            days=7
        )

        print('\n4. Results:')
        if updated_count > 0 or new_count > 0:
            print(f'✅ State reconciled: {updated_count} updated, {new_count} new trades')
        else:
            print(f'ℹ️  No reconciliation needed (all trades up to date)')

        trades_after = len(state.get('trades', []))
        print(f'   Trades: {trades_before} → {trades_after}')

        # Verify all trades are reconciled
        reconciled = [t for t in state['trades'] if t.get('exchange_reconciled', False)]
        print(f'   Reconciled: {len(reconciled)}/{trades_after}')

        if len(reconciled) == trades_after:
            print('\n✅ SUCCESS: All trades reconciled with exchange ground truth')
        else:
            print(f'\n⚠️  WARNING: {trades_after - len(reconciled)} trades not reconciled')

        # Show sample trade
        if state['trades']:
            print('\n5. Sample Trade (First):')
            t = state['trades'][0]
            print(f'   Order ID: {t.get("order_id")}')
            print(f'   History ID: {t.get("position_history_id")}')
            print(f'   P&L: ${t.get("pnl_usd_net", 0):.2f}')
            print(f'   Reconciled: {"✅" if t.get("exchange_reconciled") else "❌"}')

        print('\n' + '='*80)
        print('V2 INTEGRATION TEST COMPLETE')
        print('='*80)

        return True

    except Exception as e:
        print(f'\n❌ RECONCILIATION FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_v2_integration()
    sys.exit(0 if success else 1)
