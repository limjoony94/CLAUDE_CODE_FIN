#!/usr/bin/env python3
"""
주문 구조 디버깅 스크립트
실제 API 응답 구조를 확인합니다.
"""

import sys
import os
import yaml
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.api.bingx_client import BingXClient

SYMBOL = "BTC/USDT:USDT"

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

    print("🔍 주문 데이터 구조 디버깅")
    print("=" * 80)
    print()

    try:
        # Fetch open orders
        orders = client.exchange.fetch_open_orders(SYMBOL)

        print(f"📊 총 {len(orders)}개 미체결 주문\n")

        for idx, order in enumerate(orders, 1):
            print(f"주문 #{idx}:")
            print(f"  원본 데이터 (JSON):")
            print(json.dumps(order, indent=4, default=str))
            print()

            # 중요 필드들 개별 확인
            print(f"  📌 주요 필드:")
            print(f"     type: {order.get('type')}")
            print(f"     stopPrice: {order.get('stopPrice')}")
            print(f"     price: {order.get('price')}")
            print(f"     side: {order.get('side')}")
            print(f"     amount: {order.get('amount')}")

            # info 필드 확인 (CCXT raw data)
            if 'info' in order:
                print(f"  📌 Raw API 응답 (info):")
                print(json.dumps(order['info'], indent=4, default=str))

            print("=" * 80)
            print()

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
