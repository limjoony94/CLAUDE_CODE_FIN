#!/usr/bin/env python3
"""
Check real exchange data via API
"""
import sys
from pathlib import Path
import yaml
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

# Load API keys
config_path = PROJECT_ROOT / 'config' / 'api_keys.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

api_key = config['bingx']['mainnet']['api_key']
secret_key = config['bingx']['mainnet']['secret_key']

# Initialize client
client = BingXClient(api_key, secret_key, testnet=False)

print('=== 거래소 실시간 데이터 ===\n')

# Get balance
balance_data = client.get_balance()
balance = float(balance_data.get('balance', {}).get('balance', 0))
print(f'잔고 (Balance): ${balance:,.2f}')

# Get positions
positions = client.get_positions('BTC-USDT')
if positions and len(positions) > 0:
    pos = positions[0]
    side = pos.get("positionSide", "UNKNOWN")  # Use positionSide field directly
    qty = abs(float(pos.get("positionAmt", 0)))
    entry_price = float(pos.get("entryPrice", 0))  # Use entryPrice field
    unrealized_pnl = float(pos.get("unrealizedProfit", 0))
    leverage = int(float(pos.get("leverage", 1)))

    print(f'\n포지션 (Position):')
    print(f'  Side: {side}')
    print(f'  Quantity: {qty}')
    print(f'  Entry Price: ${entry_price:,.2f}')
    print(f'  Unrealized P&L: ${unrealized_pnl:,.2f}')
    print(f'  Leverage: {leverage}x')

    unrealized = unrealized_pnl
else:
    print(f'\n포지션: 없음')
    unrealized = 0

# Get current price
ticker = client.get_ticker('BTC-USDT')
current_price = float(ticker.get('lastPrice', 0))
print(f'\n현재가 (Current Price): ${current_price:,.2f}')

# Calculate equity
equity = balance + unrealized
print(f'\n총 자산 (Equity): ${equity:,.2f}')
print(f'  = Balance ${balance:,.2f} + Unrealized ${unrealized:,.2f}')

# Compare with state file
print('\n=== State 파일과 비교 ===\n')
state_file = PROJECT_ROOT / 'results' / 'opportunity_gating_bot_4x_state.json'
with open(state_file, 'r') as f:
    state = json.load(f)

state_balance = state.get('current_balance', 0)
state_unrealized = state.get('unrealized_pnl', 0)
state_initial = state.get('initial_balance', 0)

print(f'State 파일:')
print(f'  Initial Balance: ${state_initial:,.2f}')
print(f'  Current Balance: ${state_balance:,.2f}')
print(f'  Unrealized P&L: ${state_unrealized:,.2f}')

print(f'\n차이 (Exchange - State):')
print(f'  Balance 차이: ${balance - state_balance:,.2f}')
print(f'  Unrealized 차이: ${unrealized - state_unrealized:,.2f}')

# Calculate returns
if state_initial > 0:
    state_total_return = ((state_balance + state_unrealized - state_initial) / state_initial) * 100
    exchange_total_return = ((balance + unrealized - state_initial) / state_initial) * 100

    print(f'\n수익률:')
    print(f'  State 파일 기준: {state_total_return:+.2f}%')
    print(f'  Exchange 기준: {exchange_total_return:+.2f}%')
    print(f'  차이: {exchange_total_return - state_total_return:+.2f}%p')
