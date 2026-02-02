#!/usr/bin/env python3
"""
Sync State File with Exchange Balance
Retrieves current balance from BingX exchange and updates state file.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import yaml
import json
from datetime import datetime
from api.bingx_client import BingXClient

def sync_balance():
    # Load API keys
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_keys.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize mainnet client
    api_key = config['bingx']['mainnet']['api_key']
    secret_key = config['bingx']['mainnet']['secret_key']
    client = BingXClient(api_key, secret_key, testnet=False)

    # Get balance from exchange
    balance_info = client.get_balance()
    exchange_balance = float(balance_info['balance']['balance'])
    available_margin = float(balance_info['balance']['availableMargin'])

    print(f'✅ Successfully retrieved balance from BingX mainnet')
    print(f'')
    print(f'📊 Exchange Balance:')
    print(f'  Total Balance: ${exchange_balance:.2f}')
    print(f'  Available Margin: ${available_margin:.2f}')
    print(f'')

    # Read current state
    state_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'opportunity_gating_bot_4x_state.json')
    with open(state_path, 'r') as f:
        state = json.load(f)

    old_balance = state['current_balance']
    timestamp = datetime.now().isoformat()

    # Update state with exchange balance
    state['session_start'] = timestamp
    state['initial_balance'] = exchange_balance
    state['current_balance'] = exchange_balance
    state['realized_balance'] = exchange_balance
    state['timestamp'] = timestamp
    state['position'] = None
    state['trades'] = []
    state['closed_trades'] = 0
    state['ledger'] = []
    state['unrealized_pnl'] = 0.0
    state['stats'] = {
        'total_trades': 0,
        'long_trades': 0,
        'short_trades': 0,
        'wins': 0,
        'losses': 0,
        'total_pnl_usd': 0.0,
        'total_pnl_pct': 0.0
    }
    state['latest_signals'] = {
        'entry': {
            'long_prob': 0.0,
            'short_prob': 0.0,
            'long_threshold': 0.65,
            'short_threshold': 0.7
        },
        'exit': {}
    }
    state['reconciliation_log'] = [
        {
            'timestamp': timestamp,
            'event': 'exchange_sync_reset',
            'reason': 'User requested reset - Synced with exchange balance',
            'balance': exchange_balance,
            'previous_balance': old_balance,
            'previous_backup': f'opportunity_gating_bot_4x_state_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            'notes': f'Balance synced from BingX exchange: ${exchange_balance:.2f}, all trade records cleared'
        }
    ]

    # Write updated state
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    print('✅ State file synced with exchange balance!')
    print(f'')
    print(f'📊 Sync Summary:')
    print(f'  Previous Balance: ${old_balance:.2f}')
    print(f'  Exchange Balance: ${exchange_balance:.2f}')
    print(f'  Change: ${exchange_balance - old_balance:+.2f}')
    print(f'')
    print(f'  Initial Balance: ${exchange_balance:.2f} (updated)')
    print(f'  Trades: 0 (cleared)')
    print(f'  Position: None')
    print(f'  Stats: All reset to 0')
    print(f'  Configuration: Preserved')
    print(f'')
    print(f'🔄 Session Start: {timestamp}')

if __name__ == '__main__':
    sync_balance()
