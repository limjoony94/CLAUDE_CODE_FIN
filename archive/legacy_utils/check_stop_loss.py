#!/usr/bin/env python3
"""
Stop Loss 주문 확인 스크립트
현재 거래소에 설정된 Stop Loss 주문이 있는지 확인합니다.
"""

import sys
import os
import yaml
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.api.bingx_client import BingXClient

SYMBOL = "BTC-USDT"

def load_api_keys():
    """Load API keys from config/api_keys.yaml"""
    project_root = Path(__file__).parent.parent.parent
    api_keys_file = project_root / "config" / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('mainnet', {})
    return {}

def main():
    api_config = load_api_keys()
    api_key = api_config.get('api_key', os.getenv("BINGX_API_KEY", ""))
    api_secret = api_config.get('secret_key', os.getenv("BINGX_API_SECRET", ""))

    client = BingXClient(api_key=api_key, secret_key=api_secret, testnet=False)

    print("🔍 Stop Loss 주문 확인")
    print("=" * 80)

    # Get open orders
    try:
        orders = client.exchange.fetch_open_orders('BTC/USDT:USDT')

        print(f"\n📊 현재 미체결 주문: {len(orders)}개\n")

        if not orders:
            print("⚠️  미체결 주문이 없습니다!")
            print("   현재 포지션에 Stop Loss가 설정되지 않았을 수 있습니다.\n")
            return

        stop_loss_found = False
        for idx, order in enumerate(orders, 1):
            order_type = order.get('type', 'N/A')
            side = order.get('side', 'N/A')
            price = order.get('price', 0)
            stop_price = order.get('stopPrice', 0)
            amount = order.get('amount', 0)
            order_id = order.get('id', 'N/A')

            print(f"주문 #{idx}:")
            print(f"  ID: {order_id}")
            print(f"  타입: {order_type}")
            print(f"  방향: {side}")
            if stop_price:
                print(f"  ⚠️  Stop Price: ${stop_price:,.2f}")
                stop_loss_found = True
            if price:
                print(f"  Limit Price: ${price:,.2f}")
            print(f"  수량: {amount} BTC")
            print()

        if stop_loss_found:
            print("✅ Stop Loss 주문이 설정되어 있습니다.\n")
        else:
            print("⚠️  Stop Loss 주문이 없습니다!")
            print("   수동으로 진입한 포지션에는 Stop Loss가 자동 설정되지 않습니다.\n")

    except Exception as e:
        print(f"❌ 오류 발생: {e}\n")

if __name__ == "__main__":
    main()
