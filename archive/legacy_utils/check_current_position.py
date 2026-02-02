"""현재 포지션 상태 확인"""
import sys
import os
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

def load_api_keys():
    api_keys_file = PROJECT_ROOT / "config" / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('testnet', {})
    return {}

# Get API credentials
api_config = load_api_keys()
api_key = api_config.get('api_key', os.getenv("BINGX_API_KEY", ""))
api_secret = api_config.get('secret_key', os.getenv("BINGX_API_SECRET", ""))

client = BingXClient(
    api_key=api_key,
    secret_key=api_secret,
    testnet=True
)

print("=" * 80)
print("현재 포지션 상태 확인")
print("=" * 80)
print()

# Get current positions
positions = client.get_positions("BTC-USDT")

if positions:
    for pos in positions:
        print(f"포지션 발견:")
        print(f"  방향: {pos['positionSide']}")
        print(f"  수량: {pos['positionAmt']} BTC")
        print(f"  진입가: ${float(pos['entryPrice']):,.2f}")
        print(f"  미실현 손익: ${float(pos['unrealizedProfit']):,.2f}")
else:
    print("✅ 현재 포지션 없음")
