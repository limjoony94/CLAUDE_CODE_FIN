"""
BingX Testnet 거래 히스토리 확인 스크립트
Trade #2의 실제 청산 기록을 거래소에서 확인
"""
import sys
import os
import yaml
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

def load_api_keys():
    """Load API keys from config file"""
    api_keys_file = PROJECT_ROOT / "config" / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('testnet', {})
    return {}

def main():
    print("=" * 80)
    print("BingX Testnet 거래 히스토리 확인")
    print("=" * 80)
    print()

    # Get API credentials from config file
    api_config = load_api_keys()
    api_key = api_config.get('api_key', os.getenv("BINGX_API_KEY", ""))
    api_secret = api_config.get('secret_key', os.getenv("BINGX_API_SECRET", ""))

    if not api_key or not api_secret:
        print("❌ API credentials not found!")
        print("   Check config/api_keys.yaml or set BINGX_API_KEY and BINGX_API_SECRET environment variables.")
        return

    # Initialize client
    client = BingXClient(
        api_key=api_key,
        secret_key=api_secret,
        testnet=True
    )
    print("✅ BingX Testnet Client initialized")
    print()

    # Get recent trades
    print("📊 최근 거래 기록 조회 중...")

    # BingX API: Get trade history
    # fetch_my_trades() or fetch_closed_orders()
    try:
        # Get closed orders from last 24 hours
        since = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000)

        # BingX perpetual swap symbol format: BTC/USDT:USDT
        closed_orders = client.exchange.fetch_closed_orders(
            symbol='BTC/USDT:USDT',
            since=since,
            limit=100
        )

        print(f"✅ 총 {len(closed_orders)}개의 청산된 주문 발견")
        print()

        if closed_orders:
            print("=" * 120)
            print(f"{'Time':<20} {'Side':<6} {'Type':<8} {'Qty':<12} {'Price':<15} {'Status':<10} {'Order ID'}")
            print("=" * 120)

            for order in closed_orders:
                timestamp = datetime.fromtimestamp(order['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                side = order['side']
                order_type = order['type']
                amount = order['amount']
                price = order.get('price', order.get('average', 'N/A'))
                status = order['status']
                order_id = order.get('id', 'N/A')

                print(f"{timestamp:<20} {side:<6} {order_type:<8} {amount:<12.4f} {price:<15} {status:<10} {order_id}")

        print()
        print("=" * 120)
        print("Trade #2 관련 주문 찾기:")
        print("  진입: 2025-10-15 18:00:13, LONG 0.5866 BTC @ $112,892.50")
        print("=" * 120)
        print()

        # Find Trade #2 entry
        trade2_entry = None
        for order in closed_orders:
            if order['side'] == 'buy' and abs(order['amount'] - 0.5866) < 0.01:
                timestamp = datetime.fromtimestamp(order['timestamp'] / 1000)
                if timestamp.hour == 18 and timestamp.minute == 0:
                    trade2_entry = order
                    print(f"✅ Trade #2 진입 주문 발견:")
                    print(f"   시간: {datetime.fromtimestamp(order['timestamp'] / 1000)}")
                    print(f"   수량: {order['amount']} BTC")
                    print(f"   가격: ${order.get('price', order.get('average', 'N/A'))}")
                    print(f"   주문 ID: {order.get('id', 'N/A')}")
                    print()
                    break

        if trade2_entry:
            # Find corresponding exit
            print("🔍 Trade #2 청산 주문 찾기...")
            for order in closed_orders:
                if order['side'] == 'sell' and abs(order['amount'] - 0.5866) < 0.01:
                    timestamp = datetime.fromtimestamp(order['timestamp'] / 1000)
                    print(f"✅ 청산 주문 발견:")
                    print(f"   시간: {timestamp}")
                    print(f"   수량: {order['amount']} BTC")
                    print(f"   가격: ${order.get('price', order.get('average', 'N/A'))}")
                    print(f"   주문 ID: {order.get('id', 'N/A')}")
                    print(f"   상태: {order['status']}")

                    # Compare with our record
                    print()
                    print("=" * 80)
                    print("비교:")
                    print(f"  우리 기록: 20:38:53에 역고아로 자동 청산, Exit $111,945.60")
                    print(f"  거래소 기록: {timestamp}에 청산, Exit ${order.get('price', order.get('average', 'N/A'))}")
                    print("=" * 80)
                    break
        else:
            print("⚠️ Trade #2 진입 주문을 찾지 못했습니다.")
            print("   가능한 이유:")
            print("   1. 24시간 이전 거래 (조회 범위 확장 필요)")
            print("   2. 다른 수량으로 체결됨")
            print("   3. 거래소에서 이미 삭제됨")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()

    print()

if __name__ == "__main__":
    main()
