#!/usr/bin/env python3
"""
Check actual fees from exchange API
"""

import sys
from pathlib import Path
from datetime import datetime
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.api.bingx_client import BingXClient

def check_fees():
    """Fetch actual trade history and fees from exchange"""

    # Load API keys
    with open(project_root / 'config' / 'api_keys.yaml', 'r') as f:
        config = yaml.safe_load(f)
        api_key = config['bingx']['mainnet']['api_key']
        secret_key = config['bingx']['mainnet']['secret_key']

    client = BingXClient(api_key, secret_key, testnet=False)

    # Reset time: 2025-10-25 03:46:44
    reset_time_ms = int(datetime(2025, 10, 25, 3, 46, 0).timestamp() * 1000)

    print('=' * 80)
    print('📋 Reset 이후 거래 내역 및 수수료 조회')
    print('   Reset 시각: 2025-10-25 03:46:44 KST')
    print('=' * 80)
    print()

    try:
        # Fetch trades
        trades = client.exchange.fetch_my_trades(
            symbol='BTC/USDT:USDT',
            since=reset_time_ms,
            limit=100
        )

        print(f'✅ 조회 완료: {len(trades)} 거래 발견')
        print()

        if len(trades) == 0:
            print('⚠️  Reset 이후 거래 내역 없음')
            print()
            print('   Balance 감소 원인:')
            print('   1. 거래 수수료 아님 (거래 없음)')
            print('   2. Funding Fee 가능성 낮음 (4.5시간 경과, 다음 12:00)')
            print('   3. 기타 원인 조사 필요')
            return

        # Analyze trades
        total_fee_usdt = 0

        for i, trade in enumerate(trades, 1):
            ts = datetime.fromtimestamp(trade['timestamp'] / 1000)
            side = trade['side'].upper()
            price = trade['price']
            amount = trade['amount']
            cost = trade['cost']

            # Fee info
            fee = trade.get('fee', {})
            if isinstance(fee, dict):
                fee_cost = float(fee.get('cost', 0))
                fee_currency = fee.get('currency', 'USDT')
            else:
                fee_cost = 0
                fee_currency = 'USDT'

            if fee_currency == 'USDT':
                total_fee_usdt += fee_cost

            # Format output
            time_str = ts.strftime('%m-%d %H:%M:%S')
            price_str = f'${price:>10,.2f}'
            amount_str = f'{amount:.4f} BTC'
            cost_str = f'${cost:>8,.2f}'
            fee_str = f'${fee_cost:>6.2f}'

            print(f'{i}. {time_str} | {side:4s} | Price: {price_str} | '
                  f'Amount: {amount_str} | Cost: {cost_str} | Fee: {fee_str}')

        print()
        print('=' * 80)
        print(f'📊 총 거래 수수료 (USDT): ${total_fee_usdt:.2f}')
        print()

        # Compare with balance change
        balance_change = -18.47  # From API
        print(f'💵 Balance Change: ${balance_change:.2f}')
        print(f'📊 Trading Fees: ${total_fee_usdt:.2f}')
        print(f'❓ Unexplained: ${balance_change - (-total_fee_usdt):.2f}')

        if abs(total_fee_usdt - abs(balance_change)) < 1.0:
            print('\n✅ Balance 감소는 거래 수수료로 설명됨')
        else:
            print('\n⚠️  Balance 감소가 거래 수수료만으로 설명되지 않음')
            print('   추가 원인 조사 필요 (Funding Fee, 기타 수수료 등)')

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_fees()
