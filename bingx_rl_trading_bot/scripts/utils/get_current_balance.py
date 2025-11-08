"""
Get Current Balance from BingX Exchange
=========================================

Fetches current USDT balance from exchange API.
"""

import sys
from pathlib import Path
import yaml
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

def load_api_keys():
    """Load API keys from config/api_keys.yaml"""
    api_keys_file = PROJECT_ROOT / "config" / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('mainnet', {})
    return {}

def get_current_balance():
    """Fetch current balance from exchange"""
    print("="*60)
    print("FETCHING CURRENT BALANCE FROM BINGX")
    print("="*60)

    # Load API keys
    api_config = load_api_keys()
    api_key = api_config.get('api_key', os.getenv("BINGX_API_KEY", ""))
    api_secret = api_config.get('secret_key', os.getenv("BINGX_API_SECRET", ""))

    if not api_key or not api_secret:
        print("\n❌ Error: API keys not found!")
        return None

    # Initialize client
    client = BingXClient(api_key=api_key, secret_key=api_secret, testnet=False)

    # Get balance
    try:
        balance_data = client.get_balance()
        balance_str = balance_data['balance']['balance']
        balance = float(balance_str)

        print(f"\n✅ Current Balance Retrieved:")
        print(f"   Total Balance: ${balance:,.4f} USDT")
        print(f"   Rounded: ${balance:,.2f} USDT")

        return balance

    except Exception as e:
        print(f"\n❌ Error fetching balance: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    balance = get_current_balance()
    if balance:
        print(f"\n📊 Balance: {balance}")
