"""Close all open BTC/USDT positions on BingX."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import ccxt, yaml

with open('config/api_keys.yaml') as f:
    keys = yaml.safe_load(f)

exchange = ccxt.bingx({
    'apiKey': keys['bingx']['mainnet']['api_key'],
    'secret': keys['bingx']['mainnet']['secret_key'],
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

# Get open positions
positions = exchange.fetch_positions(['BTC/USDT:USDT'])
print(f"Found {len(positions)} position entries")

for pos in positions:
    side = pos['side']  # 'long' or 'short'
    contracts = abs(float(pos['contracts'] or 0))
    unrealized = float(pos.get('unrealizedPnl', 0) or 0)
    entry = float(pos.get('entryPrice', 0) or 0)

    if contracts <= 0:
        continue

    print(f"\n  {side.upper()}: {contracts} contracts, entry=${entry:.1f}, unrealPnL=${unrealized:.2f}")

    # Close: sell to close long, buy to close short
    close_side = 'sell' if side == 'long' else 'buy'
    position_side = 'LONG' if side == 'long' else 'SHORT'

    try:
        order = exchange.create_order(
            symbol='BTC/USDT:USDT',
            type='market',
            side=close_side,
            amount=contracts,
            params={
                'positionSide': position_side
            }
        )
        print(f"  ✅ CLOSED {side.upper()} {contracts} contracts")
        time.sleep(1)
    except Exception as e:
        print(f"  ❌ Error closing {side}: {e}")

# Verify
time.sleep(2)
positions_after = exchange.fetch_positions(['BTC/USDT:USDT'])
open_after = [p for p in positions_after if abs(float(p['contracts'] or 0)) > 0]
print(f"\nRemaining positions: {len(open_after)}")
if not open_after:
    print("✅ ALL POSITIONS CLOSED")
