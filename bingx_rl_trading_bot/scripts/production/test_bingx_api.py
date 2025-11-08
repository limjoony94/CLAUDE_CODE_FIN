"""
Test BingX API Connection and Real-time Data Collection

목표: BingX API가 정상 작동하는지 확인
- Public API (no credentials needed)
- 5분 캔들 데이터 수집
- 실시간 가격 확인
"""

import requests
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


class BingXAPITester:
    """BingX Public API Tester"""

    def __init__(self, use_testnet=False):
        if use_testnet:
            self.base_url = "https://open-api-vst.bingx.com"  # Testnet
            print("🧪 Using BingX Testnet API")
        else:
            self.base_url = "https://open-api.bingx.com"  # Production (public endpoints)
            print("🌐 Using BingX Production API (public endpoints only)")

    def test_connection(self):
        """Test basic API connection"""
        print("\n" + "=" * 80)
        print("Test 1: API Connection")
        print("=" * 80)

        try:
            url = f"{self.base_url}/openApi/swap/v2/server/time"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                server_time = data.get('data', {}).get('serverTime', 0)
                server_datetime = datetime.fromtimestamp(server_time / 1000)

                print(f"✅ API Connection Successful!")
                print(f"   Server Time: {server_datetime}")
                print(f"   Response: {data}")
                return True
            else:
                print(f"❌ API Connection Failed!")
                print(f"   Status Code: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

        except Exception as e:
            print(f"❌ API Connection Error: {e}")
            return False

    def get_klines(self, symbol="BTC-USDT", interval="5m", limit=100):
        """Get candlestick data"""
        print("\n" + "=" * 80)
        print("Test 2: Get 5-minute Candlestick Data")
        print("=" * 80)

        try:
            url = f"{self.base_url}/openApi/swap/v3/quote/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }

            print(f"Requesting: {url}")
            print(f"Parameters: {params}")

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data.get('code') == 0 and 'data' in data:
                    klines = data['data']

                    if len(klines) > 0:
                        # Parse to DataFrame (BingX returns list of dicts with keys: open, high, low, close, volume, time)
                        df = pd.DataFrame(klines)

                        # Rename 'time' to 'timestamp' and convert to datetime
                        df = df.rename(columns={'time': 'timestamp'})
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                        # Convert types (BingX returns strings)
                        df[['open', 'high', 'low', 'close', 'volume']] = \
                            df[['open', 'high', 'low', 'close', 'volume']].astype(float)

                        # Reorder columns to match expected format
                        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

                        print(f"✅ Candlestick Data Retrieved Successfully!")
                        print(f"   Symbol: {symbol}")
                        print(f"   Interval: {interval}")
                        print(f"   Candles: {len(df)}")
                        print(f"\n   Latest 5 Candles:")
                        print(df.tail(5).to_string(index=False))

                        # Current price
                        current_price = df['close'].iloc[-1]
                        print(f"\n   Current BTC Price: ${current_price:,.2f}")

                        return df
                    else:
                        print(f"⚠️ No candlestick data returned")
                        return None
                else:
                    print(f"❌ API Error!")
                    print(f"   Code: {data.get('code')}")
                    print(f"   Message: {data.get('msg')}")
                    return None
            else:
                print(f"❌ Request Failed!")
                print(f"   Status Code: {response.status_code}")
                print(f"   Response: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Get Klines Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def test_realtime_updates(self, symbol="BTC-USDT", interval="5m", iterations=3):
        """Test real-time price updates"""
        print("\n" + "=" * 80)
        print("Test 3: Real-time Price Updates")
        print("=" * 80)
        print(f"Monitoring {symbol} every 10 seconds for {iterations} iterations...")

        prices = []

        for i in range(iterations):
            try:
                df = self.get_klines(symbol, interval, limit=1)

                if df is not None and len(df) > 0:
                    current_price = df['close'].iloc[-1]
                    current_time = datetime.now()
                    prices.append((current_time, current_price))

                    print(f"\n[{i+1}/{iterations}] {current_time.strftime('%H:%M:%S')}: ${current_price:,.2f}")

                    if i < iterations - 1:
                        print("   Waiting 10 seconds...")
                        time.sleep(10)
                else:
                    print(f"⚠️ Failed to get price data")

            except Exception as e:
                print(f"❌ Error: {e}")

        if len(prices) > 1:
            print("\n" + "=" * 80)
            print("Price Movement Summary:")
            print("=" * 80)

            for i, (timestamp, price) in enumerate(prices):
                if i > 0:
                    prev_price = prices[i-1][1]
                    change = price - prev_price
                    change_pct = (change / prev_price) * 100

                    print(f"{timestamp.strftime('%H:%M:%S')}: ${price:,.2f} "
                          f"({change:+,.2f}, {change_pct:+.2f}%)")
                else:
                    print(f"{timestamp.strftime('%H:%M:%S')}: ${price:,.2f} (baseline)")

            total_change = prices[-1][1] - prices[0][1]
            total_change_pct = (total_change / prices[0][1]) * 100

            print(f"\nTotal Change: ${total_change:+,.2f} ({total_change_pct:+.2f}%)")

            print("\n✅ Real-time updates working!")

        return prices


def main():
    """Main test suite"""
    print("=" * 80)
    print("BingX API Connection Test")
    print("=" * 80)
    print("비판적 검증: 실제 API가 작동하는지 확인")
    print("=" * 80)

    # Test with production API (public endpoints)
    tester = BingXAPITester(use_testnet=False)

    # Test 1: Connection
    if not tester.test_connection():
        print("\n❌ API connection failed. Stopping tests.")
        return

    # Test 2: Get Klines
    df = tester.get_klines(symbol="BTC-USDT", interval="5m", limit=100)

    if df is None:
        print("\n❌ Failed to get candlestick data. Stopping tests.")
        return

    # Save to file for inspection
    output_file = PROJECT_ROOT / "data" / "test_bingx_latest.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Data saved to: {output_file}")

    # Test 3: Real-time updates
    print("\n⏳ Starting real-time update test...")
    prices = tester.test_realtime_updates(symbol="BTC-USDT", interval="5m", iterations=3)

    # Final verdict
    print("\n" + "=" * 80)
    print("🎯 Test Summary")
    print("=" * 80)

    if prices and len(prices) > 1:
        print("✅ BingX API is working correctly!")
        print("✅ Real-time 5-minute candlestick data available!")
        print("✅ Ready for Sweet-2 paper trading with live data!")
        print("\nNext Steps:")
        print("1. Set up BingX Testnet account (optional, for virtual trading)")
        print("2. Configure API credentials in environment variables")
        print("3. Run Sweet-2 paper trading bot with live data")
    else:
        print("⚠️ Some tests failed. Review errors above.")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Verify BingX API is not blocked by firewall")
        print("3. Try again later (API may be temporarily unavailable)")


if __name__ == "__main__":
    main()
