"""Close positions + switch to One-way mode"""
import ccxt, yaml, time

with open('config/api_keys.yaml') as f:
    keys = yaml.safe_load(f)

ex = ccxt.bingx({
    'apiKey': keys['bingx']['mainnet']['api_key'],
    'secret': keys['bingx']['mainnet']['secret_key'],
    'enableRateLimit': True,
    'options': {'defaultType': 'swap', 'recvWindow': 60000}
})

# Time sync
st = ex.fetch_time()
lt = int(time.time() * 1000)
off = st - lt
ex.nonce = lambda: int(time.time() * 1000) + off
print(f"Time offset: {off}ms")

# 1. Close all positions
print("\n--- Closing positions ---")
positions = ex.fetch_positions(['BTC/USDT:USDT'])
for pos in positions:
    side = pos['side']
    contracts = abs(float(pos['contracts'] or 0))
    if contracts <= 0:
        continue
    close_side = 'sell' if side == 'long' else 'buy'
    position_side = 'LONG' if side == 'long' else 'SHORT'
    entry = float(pos.get('entryPrice', 0) or 0)
    pnl = float(pos.get('unrealizedPnl', 0) or 0)
    print(f"  Closing {side.upper()} {contracts} @ ${entry:.0f} (PnL: ${pnl:.2f})")
    try:
        ex.create_order('BTC/USDT:USDT', 'market', close_side, contracts,
                        params={'positionSide': position_side})
        print(f"  Closed.")
        time.sleep(1)
    except Exception as e:
        print(f"  Error: {e}")

# Verify
time.sleep(2)
remaining = [p for p in ex.fetch_positions(['BTC/USDT:USDT'])
             if abs(float(p['contracts'] or 0)) > 0]
print(f"\nRemaining: {len(remaining)}")

if len(remaining) > 0:
    print("Cannot switch mode with open positions!")
    exit(1)

# 2. Switch to One-way mode
print("\n--- Switching to One-way mode ---")
try:
    ex.set_position_mode(False, 'BTC/USDT:USDT')  # False = One-way
    print("Switched to One-way mode.")
except Exception as e:
    # Try direct API call
    try:
        result = ex.private_post_v1_trade_set_dual_side_position({
            'dualSidePosition': 'false'
        })
        print(f"Direct API: {result}")
    except Exception as e2:
        print(f"Switch error: {e2}")
        print("May need to switch manually in BingX app.")

# Verify mode
print("\nDone. Bot code updated for One-way mode.")
