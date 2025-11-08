"""
Position Calculation & Market Analysis Debugging
================================================
직접 실행해서 모니터 계산과 실제 거래소 상태 비교
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

# =============================================================================
# CONFIGURATION
# =============================================================================

STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"
CONFIG_DIR = PROJECT_ROOT / "config"

LEVERAGE = 4

def load_api_keys():
    """Load API keys"""
    api_keys_file = CONFIG_DIR / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('testnet', {})
    return {}

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def load_state():
    """Load current state"""
    with open(STATE_FILE, 'r') as f:
        return json.load(f)

def calculate_position_pnl(position, current_price):
    """Calculate position P&L manually"""
    entry_price = position['entry_price']
    quantity = position['quantity']
    side = position['side']
    position_value = position['position_value']

    # Price change
    if side == 'LONG':
        price_change_pct = (current_price - entry_price) / entry_price
    else:  # SHORT
        price_change_pct = (entry_price - current_price) / entry_price

    # Leveraged P&L
    leveraged_pnl_pct = price_change_pct * LEVERAGE
    pnl_usd = position_value * leveraged_pnl_pct

    return {
        'price_change_pct': price_change_pct,
        'leveraged_pnl_pct': leveraged_pnl_pct,
        'pnl_usd': pnl_usd,
        'entry_price': entry_price,
        'current_price': current_price,
        'quantity': quantity
    }

def calculate_holding_time(entry_time_str):
    """Calculate holding time"""
    entry_time = datetime.fromisoformat(entry_time_str)
    now = datetime.now()
    duration = (now - entry_time).total_seconds() / 3600
    return duration

def main():
    """Main debugging function"""
    print("="*80)
    print("POSITION CALCULATION & MARKET ANALYSIS DEBUGGING")
    print("="*80)

    # Load state
    print("\n[1] Loading state file...")
    state = load_state()

    print(f"✅ State loaded")
    print(f"   Session Start: {state['session_start']}")
    print(f"   Initial Balance: ${state['initial_balance']:,.2f}")
    print(f"   Current Balance: ${state['current_balance']:,.2f}")
    print(f"   Timestamp: {state['timestamp']}")

    # Check position
    position = state.get('position')
    if not position:
        print("\n❌ No open position")
        return

    print(f"\n[2] Current Position (from state file)")
    print(f"   Side: {position['side']}")
    print(f"   Entry Time: {position['entry_time']}")
    print(f"   Entry Price: ${position['entry_price']:,.2f}")
    print(f"   Quantity: {position['quantity']:.8f} BTC")
    print(f"   Position Value: ${position['position_value']:,.2f}")
    print(f"   Leveraged Value: ${position['leveraged_value']:,.2f}")
    print(f"   Order ID: {position['order_id']}")

    # Calculate holding time
    holding_hours = calculate_holding_time(position['entry_time'])
    print(f"\n[3] Holding Time Analysis")
    print(f"   Entry Time: {position['entry_time']}")
    print(f"   Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Duration: {holding_hours:.2f} hours")
    print(f"   Max Hold: 8.0 hours (emergency)")
    print(f"   Time Remaining: {8.0 - holding_hours:.2f} hours")

    # Initialize API client
    print(f"\n[4] Connecting to BingX API...")
    api_config = load_api_keys()
    client = BingXClient(
        api_key=api_config.get('api_key', ''),
        secret_key=api_config.get('secret_key', ''),
        testnet=True
    )
    print(f"✅ Connected to BingX (Testnet)")

    # Get current market price
    print(f"\n[5] Fetching current market data...")
    try:
        klines = client.get_klines("BTC-USDT", "1m", limit=1)
        current_price = float(klines[0]['close'])
        print(f"✅ Current Price: ${current_price:,.2f}")
    except Exception as e:
        print(f"❌ Failed to get price: {e}")
        return

    # Calculate P&L
    print(f"\n[6] P&L Calculation (Manual)")
    pnl = calculate_position_pnl(position, current_price)

    print(f"   Entry Price: ${pnl['entry_price']:,.2f}")
    print(f"   Current Price: ${pnl['current_price']:,.2f}")
    print(f"   Price Change: {pnl['price_change_pct']*100:+.4f}%")
    print(f"   ")
    print(f"   Unleveraged P&L: {pnl['price_change_pct']*100:+.4f}%")
    print(f"   Leveraged P&L (4x): {pnl['leveraged_pnl_pct']*100:+.4f}%")
    print(f"   P&L USD: ${pnl['pnl_usd']:+,.2f}")

    # Get actual position from exchange
    print(f"\n[7] Fetching actual position from BingX...")
    try:
        positions = client.get_positions("BTC-USDT")
        print(f"✅ Positions retrieved: {len(positions)}")

        for pos in positions:
            print(f"\n   Position from Exchange:")
            print(f"   Symbol: {pos.get('symbol', 'N/A')}")
            print(f"   Side: {pos.get('positionSide', 'N/A')}")
            print(f"   Position Amount: {pos.get('positionAmt', 0)}")
            print(f"   Entry Price: ${float(pos.get('avgPrice', 0)):,.2f}")
            print(f"   Mark Price: ${float(pos.get('markPrice', 0)):,.2f}")
            print(f"   Unrealized PNL: ${float(pos.get('unrealizedProfit', 0)):,.2f}")
            print(f"   Leverage: {pos.get('leverage', 'N/A')}")

    except Exception as e:
        print(f"❌ Failed to get positions: {e}")

    # Get balance
    print(f"\n[8] Account Balance")
    try:
        balance_info = client.get_balance()
        balance = float(balance_info.get('balance', {}).get('availableMargin', 0))
        equity = float(balance_info.get('balance', {}).get('equity', 0))

        print(f"   Available Margin: ${balance:,.2f}")
        print(f"   Equity: ${equity:,.2f}")
        print(f"   State File Balance: ${state['current_balance']:,.2f}")

        if abs(balance - state['current_balance']) > 1.0:
            print(f"   ⚠️ Balance mismatch: ${abs(balance - state['current_balance']):,.2f}")
        else:
            print(f"   ✅ Balance matches")

    except Exception as e:
        print(f"❌ Failed to get balance: {e}")

    # Total Return Analysis
    print(f"\n[9] Total Return Analysis")
    initial = state['initial_balance']
    current = state['current_balance']
    total_return = (current - initial) / initial

    print(f"   Initial Balance: ${initial:,.2f}")
    print(f"   Current Balance: ${current:,.2f}")
    print(f"   Total Return: {total_return*100:+.2f}%")

    print(f"\n   Closed Trades: {state.get('closed_trades', 0)}")
    print(f"   Win/Loss: {state['stats']['wins']}/{state['stats']['losses']}")
    print(f"   Total P&L: ${state['stats']['total_pnl_usd']:+,.2f}")

    # Market Regime Analysis
    print(f"\n[10] Market Regime Analysis")
    print(f"   Latest Signals:")
    entry_signals = state['latest_signals']['entry']
    print(f"   LONG Prob: {entry_signals['long_prob']:.4f} (threshold: {entry_signals['long_threshold']})")
    print(f"   SHORT Prob: {entry_signals['short_prob']:.4f} (threshold: {entry_signals['short_threshold']})")

    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Position exists: {position['side']} @ ${position['entry_price']:,.2f}")
    print(f"✅ Current Price: ${current_price:,.2f}")
    print(f"✅ Leveraged P&L: {pnl['leveraged_pnl_pct']*100:+.2f}% (${pnl['pnl_usd']:+,.2f})")
    print(f"✅ Holding Time: {holding_hours:.2f}h / 8.0h max")
    print(f"✅ Account Balance: ${current:,.2f}")
    print(f"✅ Total Return: {total_return*100:+.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()
