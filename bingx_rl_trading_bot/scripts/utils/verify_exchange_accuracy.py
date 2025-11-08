#!/usr/bin/env python3
"""
Verify State File Accuracy Against Exchange API
Direct API verification of closed positions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import yaml
from datetime import datetime, timedelta
from src.api.bingx_client import BingXClient

def verify_exchange_accuracy():
    """Direct API verification of closed positions vs state file"""

    print('='*80)
    print('EXCHANGE ACCURACY VERIFICATION')
    print('='*80)
    print()

    # Load API keys from config
    config_path = Path(__file__).parent.parent.parent / 'config' / 'api_keys.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create API client (mainnet)
    api_key = config['bingx']['mainnet']['api_key']
    secret_key = config['bingx']['mainnet']['secret_key']
    client = BingXClient(api_key=api_key, secret_key=secret_key, testnet=False)

    # Load state file
    state_file = Path(__file__).parent.parent.parent / 'results' / 'opportunity_gating_bot_4x_state.json'
    with open(state_file, 'r') as f:
        state = json.load(f)

    state_trades = state.get('trades', [])
    closed_state_trades = [t for t in state_trades if t.get('status') == 'CLOSED']

    print(f'State File Closed Trades: {len(closed_state_trades)}')
    print()

    # Fetch position history from exchange (last 7 days)
    print('🔍 Fetching position history from exchange...')
    print()

    symbol = 'BTC-USDT'
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)

    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    try:
        # Direct API call to fetch position history
        ccxt_symbol = 'BTC/USDT:USDT'
        history = client.exchange.fetch_position_history(
            symbol=ccxt_symbol,
            since=start_ts,
            limit=100,
            params={'startTs': start_ts, 'endTs': end_ts}
        )

        print(f'✅ Exchange returned {len(history)} closed positions')
        print()

        # Extract closed positions
        closed_positions = []
        for pos in history:
            info = pos.get('info', {})
            position_id = str(info.get('positionId', ''))

            if position_id:
                closed_positions.append({
                    'position_id': position_id,
                    'side': 'BUY' if float(info.get('closePositionAmt', 0)) > 0 else 'SELL',
                    'entry_price': float(info.get('avgOpenPrice', 0)),
                    'exit_price': float(info.get('avgClosePrice', 0)),
                    'quantity': abs(float(info.get('closePositionAmt', 0))),
                    'entry_time': info.get('createTime', ''),
                    'close_time': info.get('updateTime', ''),
                    'realized_pnl': float(info.get('realisedProfit', 0)),
                    'commission': abs(float(info.get('commission', 0))),
                    'net_profit': float(info.get('netProfit', 0))
                })

        print(f'📊 Parsed {len(closed_positions)} closed positions from exchange')
        print()

        # Match state file trades with exchange positions
        print('='*80)
        print('VERIFICATION: State File vs Exchange')
        print('='*80)
        print()

        matches = 0
        mismatches = 0
        missing_in_state = 0
        missing_in_exchange = 0

        # Create lookup by position ID
        exchange_by_id = {p['position_id']: p for p in closed_positions}
        state_by_id = {t.get('position_id_exchange'): t for t in closed_state_trades if t.get('position_id_exchange')}

        # Check each exchange position
        for pos_id, exchange_pos in exchange_by_id.items():
            print(f'Position ID: {pos_id}')
            print(f'  Exchange Side: {exchange_pos["side"]}')

            if pos_id in state_by_id:
                state_trade = state_by_id[pos_id]

                # Normalize side for comparison
                state_side = state_trade.get('side', '')
                if state_side == 'LONG':
                    state_side = 'BUY'
                elif state_side == 'SHORT':
                    state_side = 'SELL'

                # Compare key fields
                entry_price_match = abs(exchange_pos['entry_price'] - state_trade.get('entry_price', 0)) < 0.01
                exit_price_match = abs(exchange_pos['exit_price'] - state_trade.get('exit_price', 0)) < 0.01
                pnl_match = abs(exchange_pos['realized_pnl'] - state_trade.get('pnl_usd', 0)) < 0.01
                net_pnl_match = abs(exchange_pos['net_profit'] - state_trade.get('pnl_usd_net', 0)) < 0.01

                print(f'  State Side: {state_side}')
                print(f'  Entry Price: Exchange ${exchange_pos["entry_price"]:,.2f} vs State ${state_trade.get("entry_price", 0):,.2f} {"✅" if entry_price_match else "❌"}')
                print(f'  Exit Price: Exchange ${exchange_pos["exit_price"]:,.2f} vs State ${state_trade.get("exit_price", 0):,.2f} {"✅" if exit_price_match else "❌"}')
                print(f'  Realized P&L: Exchange ${exchange_pos["realized_pnl"]:.2f} vs State ${state_trade.get("pnl_usd", 0):.2f} {"✅" if pnl_match else "❌"}')
                print(f'  Net P&L: Exchange ${exchange_pos["net_profit"]:.2f} vs State ${state_trade.get("pnl_usd_net", 0):.2f} {"✅" if net_pnl_match else "❌"}')
                print(f'  Total Fees: Exchange ${exchange_pos["commission"]:.2f} vs State ${state_trade.get("total_fee", 0):.2f}')

                if entry_price_match and exit_price_match and pnl_match and net_pnl_match:
                    print('  ✅ MATCH - All fields accurate!')
                    matches += 1
                else:
                    print('  ❌ MISMATCH - Discrepancy detected!')
                    mismatches += 1

            else:
                print(f'  ⚠️  NOT IN STATE FILE')
                missing_in_state += 1

            print()

        # Check for state trades not in exchange
        for pos_id, state_trade in state_by_id.items():
            if pos_id not in exchange_by_id:
                print(f'Position ID: {pos_id}')
                print(f'  ⚠️  IN STATE FILE BUT NOT IN EXCHANGE (may be older than 7 days)')
                missing_in_exchange += 1
                print()

        # Summary
        print('='*80)
        print('VERIFICATION SUMMARY')
        print('='*80)
        print()
        print(f'Exchange Positions: {len(exchange_by_id)}')
        print(f'State File Trades: {len(state_by_id)}')
        print()
        print(f'✅ Perfect Matches: {matches}')
        print(f'❌ Mismatches: {mismatches}')
        print(f'⚠️  Missing in State: {missing_in_state}')
        print(f'⚠️  Missing in Exchange (>7 days): {missing_in_exchange}')
        print()

        if mismatches == 0 and missing_in_state == 0:
            print('🎉 VERIFICATION PASSED - All positions match exchange exactly!')
        else:
            print('⚠️  VERIFICATION ISSUES - Discrepancies found!')

        print('='*80)

    except Exception as e:
        print(f'❌ Error fetching position history: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    verify_exchange_accuracy()
