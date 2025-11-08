#!/usr/bin/env python3
"""
주문 내역 확인 스크립트
SHORT 포지션 진입 및 Stop Loss 주문 내역을 확인합니다.
"""

import sys
import os
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.api.bingx_client import BingXClient

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

    print("🔍 최근 주문 내역 확인 (SHORT 포지션 관련)")
    print("=" * 80)
    print()

    try:
        # Fetch recent orders (last 20)
        orders = client.exchange.fetch_orders('BTC/USDT:USDT', limit=20)

        print(f"📊 총 {len(orders)}개 주문 조회됨\n")

        short_entry_found = False
        stop_loss_found = False

        # Filter for recent SHORT entry and STOP orders
        for order in reversed(orders):  # Show newest first
            order_time = datetime.fromtimestamp(order['timestamp'] / 1000)
            order_type = order.get('type', 'N/A')
            side = order.get('side', 'N/A')
            status = order.get('status', 'N/A')
            amount = order.get('amount', 0)
            price = order.get('price', 0)
            stop_price = order.get('stopPrice', 0)
            order_id = order.get('id', 'N/A')

            # Show SHORT related orders only
            if side == 'sell' or 'stop' in order_type.lower():
                print(f"시간: {order_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  ID: {order_id}")
                print(f"  타입: {order_type}")
                print(f"  방향: {side}")
                print(f"  상태: {status}")
                print(f"  수량: {amount} BTC")

                if price:
                    print(f"  가격: ${price:,.2f}")
                if stop_price:
                    print(f"  ⚠️  Stop Price: ${stop_price:,.2f}")
                    stop_loss_found = True

                if side == 'sell' and status == 'closed':
                    short_entry_found = True

                print()

        print("\n" + "=" * 80)
        print("📝 분석:")
        print(f"  SHORT 진입 주문: {'✅ 발견' if short_entry_found else '❌ 없음'}")
        print(f"  Stop Loss 주문: {'✅ 발견' if stop_loss_found else '❌ 없음'}")

        if short_entry_found and not stop_loss_found:
            print("\n⚠️  SHORT 진입은 있지만 Stop Loss 주문이 없습니다!")
            print("   가능한 원인:")
            print("   1. 완전 수동 진입 (거래소에서 직접)")
            print("   2. 봇 진입 시 Stop Loss 설정 실패")
            print("   3. Stop Loss 주문이 취소됨")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
