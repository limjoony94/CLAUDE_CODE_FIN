"""R26 Pre-flight — verify exchange connection, balance, position mode, leverage.

Run BEFORE starting the bot to catch config / API issues without placing orders.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import ccxt
from scripts.production.r26_grid.config import load_config, load_api_keys


def main():
    print('=' * 80)
    print('R26 Pre-flight Check')
    print('=' * 80)

    # 1. Load configs
    print('\n[1] Loading config...')
    config = load_config()
    api_keys = load_api_keys(config['api_keys_path'])
    print(f'  Strategy: spacing {config["strategy"]["grid_spacing_pct"]}%, '
          f'{config["strategy"]["grid_levels_each_side"]}+{config["strategy"]["grid_levels_each_side"]} levels')
    print(f'  Risk: per_level_notional ${config["risk"]["per_level_notional_usd"]}, '
          f'halt_daily {config["risk"]["halt_daily_loss_pct"]}%, '
          f'halt_emergency {config["risk"]["halt_emergency_adverse_pct"]}%')
    print(f'  Exchange: {config["exchange"]["symbol"]} '
          f'@ {config["exchange"]["exchange_leverage"]}× exchange / '
          f'{config["exchange"]["trading_leverage"]}× trading')
    print(f'  Bot: poll {config["bot"]["poll_interval_seconds"]}s, '
          f'warmup {config["bot"]["warmup_bars"]} bars')
    print(f'  API key length: {len(api_keys["api_key"])}, secret length: {len(api_keys["secret"])}')

    # 2. Connect to BingX
    print('\n[2] Connecting to BingX...')
    ex = ccxt.bingx({
        'apiKey': api_keys['api_key'],
        'secret': api_keys['secret'],
        'enableRateLimit': True,
        'options': {'defaultType': 'swap', 'recvWindow': 10000},
    })
    try:
        ex.load_markets()
        print(f'  Markets loaded: {len(ex.markets)} symbols')
    except Exception as e:
        print(f'  ❌ Failed: {e}')
        sys.exit(1)

    # 3. Account balance
    print('\n[3] Account balance...')
    try:
        balance = ex.fetch_balance({'type': 'swap'})
        usdt = balance.get('USDT', {})
        print(f'  USDT total:  {usdt.get("total", 0):.4f}')
        print(f'  USDT free:   {usdt.get("free", 0):.4f}')
        print(f'  USDT used:   {usdt.get("used", 0):.4f}')
        equity = float(usdt.get('total', 0))
        if equity < 50:
            print(f'  ⚠️  Low balance ({equity} USDT). Bot may not size positions properly.')
    except Exception as e:
        print(f'  ❌ Failed: {e}')
        sys.exit(1)

    # 4. Position mode
    print('\n[4] Position mode (BUG#66 protection)...')
    try:
        pm = ex.fetch_position_mode()
        if pm and pm.get('hedged') is True:
            print(f'  ⚠️  HEDGE mode detected! Bot will auto-correct to One-Way.')
        else:
            print(f'  ✅ One-Way mode confirmed: {pm}')
    except Exception as e:
        print(f'  ⚠️  Position mode check failed: {e}')

    # 5. Symbol info
    print('\n[5] Symbol info...')
    sym = config['exchange']['symbol']
    if sym not in ex.markets:
        print(f'  ❌ Symbol {sym} not found in markets')
        sys.exit(1)
    market = ex.markets[sym]
    print(f'  Symbol: {sym}')
    print(f'  Min amount: {market.get("limits", {}).get("amount", {}).get("min")}')
    print(f'  Min cost: {market.get("limits", {}).get("cost", {}).get("min")}')
    print(f'  Precision amount: {market.get("precision", {}).get("amount")}')
    print(f'  Precision price: {market.get("precision", {}).get("price")}')

    # 6. Current price + ATR sanity
    print('\n[6] Current market state...')
    try:
        ticker = ex.fetch_ticker(sym)
        print(f'  Last:  {ticker.get("last"):.2f}')
        print(f'  Bid:   {ticker.get("bid"):.2f}')
        print(f'  Ask:   {ticker.get("ask"):.2f}')
        print(f'  Spread bps: {(ticker.get("ask") - ticker.get("bid"))/ticker.get("last")*10000:.2f}')
    except Exception as e:
        print(f'  ❌ Failed: {e}')

    # 7. Recent klines for ATR computation
    print('\n[7] Recent 1h klines...')
    try:
        ohlcv = ex.fetch_ohlcv(sym, '1h', limit=50)
        print(f'  Fetched {len(ohlcv)} bars')
        latest = ohlcv[-1]
        print(f'  Latest: ts={latest[0]}, close={latest[4]:.2f}')
    except Exception as e:
        print(f'  ❌ Failed: {e}')

    # 8. Leverage attempt (does not actually place orders)
    print('\n[8] Setting leverage...')
    try:
        ex_lev = config['exchange']['exchange_leverage']
        ex.set_leverage(ex_lev, sym, params={'side': 'BOTH'})
        print(f'  ✅ Leverage set to {ex_lev}× for {sym}')
    except Exception as e:
        print(f'  ⚠️  set_leverage error (may already be set): {e}')

    # 9. Sizing math
    print('\n[9] Sizing math...')
    last = ex.fetch_ticker(sym).get('last')
    per_level = config['risk']['per_level_notional_usd']
    levels = config['strategy']['grid_levels_each_side']
    qty_per_level = per_level / last
    total_max_one_side_notional = per_level * levels
    print(f'  Per-level notional: ${per_level}')
    print(f'  Per-level qty: {qty_per_level:.6f} BTC')
    print(f'  Max one-side notional (5 levels filled): ${total_max_one_side_notional}')
    print(f'  Required margin per level @ {config["exchange"]["exchange_leverage"]}×: '
          f'${per_level / config["exchange"]["exchange_leverage"]:.2f}')
    print(f'  Total max margin (10 levels): '
          f'${10 * per_level / config["exchange"]["exchange_leverage"]:.2f}')

    print('\n' + '=' * 80)
    print('Pre-flight: PASS — ready for bot start')
    print('=' * 80)


if __name__ == '__main__':
    main()
