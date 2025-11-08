"""
Direct BingX API Test

Test actual API call to verify:
1. API is accessible
2. Data structure is correct
3. No inf/NaN values in response
"""

import requests
import pandas as pd
import json
from datetime import datetime

print("=" * 80)
print("BingX API Direct Test")
print("=" * 80)

# BingX API endpoint
url = "https://open-api.bingx.com/openApi/swap/v3/quote/klines"
params = {
    "symbol": "BTC-USDT",
    "interval": "5m",
    "limit": 10  # Just test with 10 candles first
}

print(f"\nAPI Request:")
print(f"  URL: {url}")
print(f"  Params: {params}")
print()

try:
    response = requests.get(url, params=params, timeout=10)

    print(f"Response Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()

        print(f"\nResponse Structure:")
        print(f"  Keys: {list(data.keys())}")
        print(f"  Code: {data.get('code')}")
        print(f"  Data type: {type(data.get('data'))}")

        if data.get('code') == 0 and 'data' in data:
            klines = data['data']
            print(f"  Candles count: {len(klines)}")

            print(f"\n{'=' * 80}")
            print("First 3 Candles (Raw):")
            print(f"{'=' * 80}")
            for i, candle in enumerate(klines[:3]):
                print(f"\nCandle {i+1}:")
                print(json.dumps(candle, indent=2))

            # Try to convert to DataFrame (like bot does)
            print(f"\n{'=' * 80}")
            print("DataFrame Conversion Test:")
            print(f"{'=' * 80}")

            df = pd.DataFrame(klines)
            print(f"\nDataFrame shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst 3 rows:")
            print(df.head(3))

            # Check for inf/NaN
            print(f"\n{'=' * 80}")
            print("Data Quality Check:")
            print(f"{'=' * 80}")

            # Check each column
            for col in df.columns:
                null_count = df[col].isnull().sum()
                print(f"  {col}: {null_count} nulls")

                # Check if numeric column has inf
                if col in ['open', 'high', 'low', 'close', 'volume', 'time']:
                    try:
                        numeric_df = pd.to_numeric(df[col], errors='coerce')
                        inf_count = (numeric_df == float('inf')).sum() + (numeric_df == float('-inf')).sum()
                        print(f"    -> {inf_count} inf values")
                    except:
                        pass

            # Try timestamp conversion (like bot does)
            print(f"\n{'=' * 80}")
            print("Timestamp Conversion Test:")
            print(f"{'=' * 80}")

            try:
                df_test = df.copy()
                df_test = df_test.rename(columns={'time': 'timestamp'})
                print(f"\nBefore conversion:")
                print(f"  Sample timestamps: {df_test['timestamp'].head(3).tolist()}")
                print(f"  Data type: {df_test['timestamp'].dtype}")

                # THIS IS WHERE THE ERROR MIGHT OCCUR
                df_test['timestamp'] = pd.to_datetime(df_test['timestamp'], unit='ms')
                print(f"\nAfter conversion:")
                print(f"  Success! ✅")
                print(f"  Sample: {df_test['timestamp'].head(3).tolist()}")

            except Exception as e:
                print(f"\n❌ ERROR during timestamp conversion:")
                print(f"  {type(e).__name__}: {e}")
                print(f"\nThis is the error the bot is experiencing!")

                # Debug: Check raw timestamp values
                print(f"\nDebug - Raw timestamp values:")
                print(df['time'].head(10))
                print(f"\nData type: {df['time'].dtype}")
                print(f"Unique values check: {df['time'].nunique()} unique out of {len(df)}")

            # Try type conversion (like bot does)
            print(f"\n{'=' * 80}")
            print("Type Conversion Test:")
            print(f"{'=' * 80}")

            try:
                df_test = df.copy()
                df_test[['open', 'high', 'low', 'close', 'volume']] = \
                    df_test[['open', 'high', 'low', 'close', 'volume']].astype(float)
                print(f"✅ Successfully converted to float")
                print(f"\nSample values:")
                print(df_test[['open', 'high', 'low', 'close', 'volume']].head(3))
            except Exception as e:
                print(f"❌ ERROR during type conversion:")
                print(f"  {type(e).__name__}: {e}")

        else:
            print(f"❌ API returned error code: {data.get('code')}")
            print(f"Message: {data.get('msg')}")

    else:
        print(f"❌ HTTP Error: {response.status_code}")
        print(f"Response: {response.text[:500]}")

except Exception as e:
    print(f"❌ Exception: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'=' * 80}")
print("Test Complete")
print(f"{'=' * 80}")
