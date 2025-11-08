#!/usr/bin/env python3
"""Check funding fees from BingX API."""

import sys
import os
import json
from datetime import datetime, timedelta
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from api.bingx_client import BingXClient

def main():
    # Load API keys
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_keys.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use mainnet
    api_key = config['bingx']['mainnet']['api_key']
    secret_key = config['bingx']['mainnet']['secret_key']

    # Initialize client
    client = BingXClient(api_key, secret_key, testnet=False)

    print('=' * 100)
    print('FUNDING FEE HISTORY')
    print('=' * 100)

    # Try to fetch income history (which includes funding fees)
    try:
        # CCXT doesn't have a standard method for funding fees, need to use private API
        # Try fetch_funding_history if available
        if hasattr(client.exchange, 'fetch_funding_history'):
            funding_history = client.exchange.fetch_funding_history(
                symbol='BTC/USDT:USDT',
                limit=20
            )

            print(f"\nFound {len(funding_history)} funding fee records")

            for i, record in enumerate(funding_history[:10], 1):
                timestamp = datetime.fromtimestamp(record.get('timestamp', 0) / 1000)
                amount = record.get('amount', 0)
                print(f"\n#{i} {timestamp.strftime('%Y-%m-%d %H:%M:%S')} | ${amount:+.4f} USDT")
                print(f"   Raw: {json.dumps(record, indent=2, default=str)}")

        else:
            print("\n⚠️  fetch_funding_history not available in CCXT")
            print("Trying direct API call...")

            # Try using private API directly
            # BingX endpoint: GET /openApi/swap/v2/user/income
            response = client.exchange.privateGetOpenApiSwapV2UserIncome({
                'symbol': 'BTC-USDT',
                'incomeType': 'FUNDING_FEE',
                'limit': 20
            })

            print(f"\nRaw API response:")
            print(json.dumps(response, indent=2))

    except Exception as e:
        print(f"\nError fetching funding fees: {e}")
        print(f"\nTrying alternative method...")

        # Try fetch_ledger or fetch_transactions
        try:
            if hasattr(client.exchange, 'fetch_ledger'):
                ledger = client.exchange.fetch_ledger(limit=50)
                print(f"\nLedger entries: {len(ledger)}")

                # Filter for funding fees
                funding_fees = [entry for entry in ledger if 'funding' in entry.get('type', '').lower()]
                print(f"Funding fee entries: {len(funding_fees)}")

                for i, entry in enumerate(funding_fees[:10], 1):
                    timestamp = datetime.fromtimestamp(entry.get('timestamp', 0) / 1000)
                    amount = entry.get('amount', 0)
                    print(f"\n#{i} {timestamp.strftime('%Y-%m-%d %H:%M:%S')} | ${amount:+.4f}")
                    print(f"   Type: {entry.get('type')}")
                    print(f"   Raw: {json.dumps(entry, indent=2, default=str)}")

        except Exception as e2:
            print(f"Error with alternative method: {e2}")

if __name__ == '__main__':
    main()
