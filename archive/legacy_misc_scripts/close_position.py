#!/usr/bin/env python3
"""Close existing testnet position"""

from bingx import BingxSync
import yaml
from pathlib import Path

# Load API keys
config_file = Path('config/api_keys.yaml')
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
    api_config = config['bingx']['testnet']

# Initialize exchange
exchange = BingxSync({
    'apiKey': api_config['api_key'],
    'secret': api_config['secret_key'],
    'enableRateLimit': True,
    'options': {'defaultType': 'swap', 'test': True},
    'timeout': 30000
})
exchange.set_sandbox_mode(True)

# Close position
print('현재 포지션 확인 중...')
positions = exchange.fetch_positions(['BTC/USDT:USDT'])

# Close if exists
for pos in positions:
    if pos['contracts'] != 0:
        print(f"\n청산 시작:")
        print(f"  Side: {pos['side']}")
        print(f"  Amount: {pos['contracts']} BTC")
        print(f"  Entry: ${pos['entryPrice']:.2f}")
        print(f"  Mark Price: ${pos['markPrice']:.2f}")
        print(f"  Unrealized PnL: ${pos['unrealizedPnl']:.4f}")

        # Create close order (One-way mode, no positionSide needed)
        side = 'sell' if pos['side'] == 'long' else 'buy'
        order = exchange.create_order(
            symbol='BTC/USDT:USDT',
            type='market',
            side=side,
            amount=abs(pos['contracts'])
        )

        print(f'\n✅ 청산 완료!')
        print(f'Order ID: {order.get("id")}')
        print(f'Status: {order.get("status")}')
        break
else:
    print('포지션 없음')
