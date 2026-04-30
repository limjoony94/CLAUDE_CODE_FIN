"""R26 — Cancel all open orders + close any open positions for clean restart."""
import os
import sys
import yaml
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import ccxt
from scripts.production.r26_grid.config import load_api_keys


def main():
    config_keys = load_api_keys()
    ex = ccxt.bingx({
        'apiKey': config_keys['api_key'],
        'secret': config_keys['secret'],
        'enableRateLimit': True,
        'options': {'defaultType': 'swap', 'recvWindow': 10000},
    })
    ex.load_markets()
    symbol = 'BTC/USDT:USDT'

    print('=' * 60)
    print('R26 Clean Reset — cancel orders + close positions')
    print('=' * 60)

    # 1. Cancel all open orders for symbol
    print('\n[1] Cancelling open orders...')
    try:
        orders = ex.fetch_open_orders(symbol)
        print(f'  Open orders: {len(orders)}')
        for o in orders:
            try:
                ex.cancel_order(o['id'], symbol)
                print(f'  Cancelled: {o["id"]} ({o["side"]} {o["amount"]} @ {o["price"]})')
            except Exception as e:
                print(f'  Failed cancel {o["id"]}: {e}')
    except Exception as e:
        print(f'  fetch_open_orders failed: {e}')

    # 2. Close open positions (market)
    print('\n[2] Closing open positions...')
    try:
        positions = ex.fetch_positions([symbol])
        for p in positions:
            contracts = float(p.get('contracts') or 0)
            if abs(contracts) > 0:
                side_close = 'sell' if p.get('side') == 'long' else 'buy'
                qty = abs(contracts)
                print(f'  Closing: side={p.get("side")}, qty={qty}, '
                      f'pnl={p.get("unrealizedPnl")}')
                if side_close == 'sell':
                    order = ex.create_market_sell_order(symbol, qty,
                        {'positionSide': 'BOTH', 'reduceOnly': True})
                else:
                    order = ex.create_market_buy_order(symbol, qty,
                        {'positionSide': 'BOTH', 'reduceOnly': True})
                print(f'  Close order: {order.get("id")}')
            else:
                print(f'  No open position (contracts={contracts})')
    except Exception as e:
        print(f'  Position close error: {e}')

    # 3. Reset state file
    print('\n[3] Resetting state file...')
    state_path = Path('results/r26_grid_state.json')
    if state_path.exists():
        state_path.unlink()
        print(f'  Deleted: {state_path}')
    else:
        print('  (state file did not exist)')

    print('\n' + '=' * 60)
    print('Clean reset complete. Ready to restart bot.')
    print('=' * 60)


if __name__ == '__main__':
    main()
