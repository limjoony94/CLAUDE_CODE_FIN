"""
Track Entry Signals from Production Bot Logs
============================================

Parse production bot logs to find all ENTRY signals and compare with current price.

Usage:
    python scripts/analysis/track_entry_signals.py

Output:
    - Console report with signal analysis
    - JSON file with detailed signal tracking
"""

import sys
from pathlib import Path
import re
from datetime import datetime, timedelta
import json
import yaml
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

# =============================================================================
# CONFIGURATION
# =============================================================================

LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
STATE_FILE = RESULTS_DIR / "opportunity_gating_bot_4x_state.json"

# Thresholds
LONG_THRESHOLD = 0.80
SHORT_THRESHOLD = 0.80


# =============================================================================
# LOG PARSING
# =============================================================================

def parse_bot_logs(log_file):
    """Parse production bot logs for ENTRY signals"""
    print(f"📖 Parsing log file: {log_file.name}")

    signals = []

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"  Total lines: {len(lines):,}")

    # Pattern to match signal check logs
    # Example: 2025-10-26 00:25:10,712 - INFO - [Candle 00:20:00 KST] Price: $111,412.9 | Balance: $4,589.17 | LONG: 0.7670 | SHORT: 0.0134
    signal_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'  # Timestamp
        r'.*?'
        r'Price: \$([\d,]+\.?\d*)'  # Price
        r'.*?'
        r'LONG: ([\d.]+)'  # LONG probability
        r'.*?'
        r'SHORT: ([\d.]+)'  # SHORT probability
    )

    # Pattern to match actual entries
    # Example: ✅ LONG Position Opened
    entry_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
        r'.*?'
        r'(LONG|SHORT) Position Opened'
    )

    # Parse all signals
    for line in lines:
        match = signal_pattern.search(line)
        if match:
            timestamp_str = match.group(1)
            price_str = match.group(2).replace(',', '')
            price = float(price_str)
            long_prob = float(match.group(3))
            short_prob = float(match.group(4))

            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

            # Determine if entry signal
            long_entry = long_prob >= LONG_THRESHOLD
            short_entry = short_prob >= SHORT_THRESHOLD

            # Opportunity gating for SHORT
            if short_entry:
                long_ev = long_prob * 0.0041
                short_ev = short_prob * 0.0047
                opportunity_cost = short_ev - long_ev
                short_entry = opportunity_cost > 0.001

            signal = {
                'timestamp': timestamp,
                'timestamp_str': timestamp_str,
                'long_prob': long_prob,
                'short_prob': short_prob,
                'price': price,
                'long_entry': long_entry,
                'short_entry': short_entry,
                'direction': None
            }

            if long_entry:
                signal['direction'] = 'LONG'
            elif short_entry:
                signal['direction'] = 'SHORT'

            signals.append(signal)

    # Parse actual entries
    actual_entries = []
    for line in lines:
        match = entry_pattern.search(line)
        if match:
            timestamp_str = match.group(1)
            direction = match.group(2)
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

            actual_entries.append({
                'timestamp': timestamp,
                'direction': direction
            })

    print(f"  ✅ Parsed {len(signals):,} signal checks")
    print(f"  ✅ Found {len(actual_entries)} actual entries")

    return signals, actual_entries


def filter_entry_signals(signals):
    """Filter for only valid ENTRY signals"""
    entry_signals = [s for s in signals if s['direction'] is not None]

    print(f"\n📊 Entry Signals Summary:")
    print(f"  Total checks: {len(signals):,}")
    print(f"  Entry signals: {len(entry_signals)}")

    if entry_signals:
        long_count = sum(1 for s in entry_signals if s['direction'] == 'LONG')
        short_count = sum(1 for s in entry_signals if s['direction'] == 'SHORT')

        print(f"    LONG: {long_count}")
        print(f"    SHORT: {short_count}")

    return entry_signals


# =============================================================================
# PRICE ANALYSIS
# =============================================================================

def fetch_current_price():
    """Fetch current BTC price from exchange"""
    print(f"\n💰 Fetching current price...")

    try:
        # Load API keys
        CONFIG_DIR = PROJECT_ROOT / "config"
        with open(CONFIG_DIR / "api_keys.yaml", 'r') as f:
            config = yaml.safe_load(f)

        api_keys = config['bingx']['mainnet']

        client = BingXClient(
            api_key=api_keys['api_key'],
            secret_key=api_keys['secret_key'],
            testnet=False
        )

        # Get current ticker price
        ticker = client.exchange.fetch_ticker('BTC/USDT:USDT')

        if ticker and 'last' in ticker:
            current_price = float(ticker['last'])
            print(f"  ✅ Current price: ${current_price:,.2f}")
            return current_price
        else:
            print(f"  ❌ Failed to fetch price")
            return None

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def analyze_signals(entry_signals, current_price):
    """Analyze potential returns for each entry signal"""
    print(f"\n📈 Analyzing Signals vs Current Price (${current_price:,.2f})")

    results = []

    for signal in entry_signals:
        entry_price = signal['price']
        direction = signal['direction']

        # Calculate potential return
        if direction == 'LONG':
            price_change = (current_price - entry_price) / entry_price
        else:  # SHORT
            price_change = (entry_price - current_price) / entry_price

        # 4x leverage
        leveraged_return = price_change * 4

        result = {
            **signal,
            'current_price': current_price,
            'price_change_pct': price_change * 100,
            'leveraged_return_pct': leveraged_return * 100,
            'profit_loss': 'PROFIT' if leveraged_return > 0 else 'LOSS',
            'hours_ago': (datetime.now() - signal['timestamp']).total_seconds() / 3600
        }

        results.append(result)

    return results


def load_actual_trades():
    """Load actual trades from state file"""
    print(f"\n📋 Loading Actual Trades...")

    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)

        trades = state.get('trade_history', [])
        print(f"  ✅ Loaded {len(trades)} trades from state")

        return trades

    except Exception as e:
        print(f"  ⚠️ Could not load trades: {e}")
        return []


def match_signals_to_trades(signal_results, actual_trades):
    """Match entry signals to actual executed trades"""
    print(f"\n🔗 Matching Signals to Trades...")

    # Parse trade timestamps
    for trade in actual_trades:
        if 'entry_time' in trade:
            trade['entry_timestamp'] = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))

    matched = 0
    unmatched_signals = []

    for result in signal_results:
        signal_time = result['timestamp']

        # Find matching trade within 1 minute
        found = False
        for trade in actual_trades:
            if 'entry_timestamp' not in trade:
                continue

            trade_time = trade['entry_timestamp']
            time_diff = abs((trade_time - signal_time).total_seconds())

            if time_diff < 60 and trade['direction'] == result['direction']:
                result['executed'] = True
                result['actual_entry_price'] = trade['entry_price']
                found = True
                matched += 1
                break

        if not found:
            result['executed'] = False
            unmatched_signals.append(result)

    print(f"  ✅ Matched: {matched} signals")
    print(f"  ⚠️ Unmatched: {len(unmatched_signals)} signals (not executed)")

    return signal_results, unmatched_signals


# =============================================================================
# REPORTING
# =============================================================================

def print_signal_report(signal_results, unmatched_signals):
    """Print detailed signal tracking report"""

    print("\n" + "="*80)
    print("ENTRY SIGNAL TRACKING REPORT")
    print("="*80)

    # Overall statistics
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Entry Signals: {len(signal_results)}")

    executed = [s for s in signal_results if s.get('executed', False)]
    unexecuted = [s for s in signal_results if not s.get('executed', False)]

    print(f"  Executed: {len(executed)}")
    print(f"  Not Executed: {len(unexecuted)}")

    if signal_results:
        long_signals = [s for s in signal_results if s['direction'] == 'LONG']
        short_signals = [s for s in signal_results if s['direction'] == 'SHORT']

        print(f"\n  By Direction:")
        print(f"    LONG: {len(long_signals)}")
        print(f"    SHORT: {len(short_signals)}")

    # Potential returns analysis
    if unexecuted:
        print(f"\n💰 Potential Returns (Unexecuted Signals):")

        profitable = [s for s in unexecuted if s['leveraged_return_pct'] > 0]
        losing = [s for s in unexecuted if s['leveraged_return_pct'] <= 0]

        print(f"  Profitable: {len(profitable)} / {len(unexecuted)} ({len(profitable)/len(unexecuted)*100:.1f}%)")
        print(f"  Losing: {len(losing)} / {len(unexecuted)}")

        if unexecuted:
            avg_return = sum(s['leveraged_return_pct'] for s in unexecuted) / len(unexecuted)
            max_return = max(s['leveraged_return_pct'] for s in unexecuted)
            min_return = min(s['leveraged_return_pct'] for s in unexecuted)

            print(f"\n  Average Return: {avg_return:+.2f}%")
            print(f"  Best Return: {max_return:+.2f}%")
            print(f"  Worst Return: {min_return:+.2f}%")

    # Recent signals (last 10)
    print(f"\n📋 Recent Entry Signals (Last 10):")
    print("-" * 80)

    recent = sorted(signal_results, key=lambda x: x['timestamp'], reverse=True)[:10]

    for i, signal in enumerate(recent, 1):
        timestamp = signal['timestamp_str']
        direction = signal['direction']
        prob = signal['long_prob'] if direction == 'LONG' else signal['short_prob']
        entry_price = signal['price']
        current_price = signal['current_price']
        price_change = signal['price_change_pct']
        leveraged_return = signal['leveraged_return_pct']
        hours_ago = signal['hours_ago']
        executed = signal.get('executed', False)

        status = "✅ EXECUTED" if executed else "⏸️ SKIPPED"
        profit_emoji = "📈" if leveraged_return > 0 else "📉"

        print(f"\n{i}. {timestamp} ({hours_ago:.1f}h ago)")
        print(f"   Direction: {direction} | Prob: {prob:.4f} | {status}")
        print(f"   Entry: ${entry_price:,.2f} → Current: ${current_price:,.2f}")
        print(f"   {profit_emoji} Price Change: {price_change:+.2f}%")
        print(f"   💰 4x Return: {leveraged_return:+.2f}%")

    # Best missed opportunities
    if unexecuted:
        print(f"\n🎯 Top 5 Missed Opportunities:")
        print("-" * 80)

        top_missed = sorted(unexecuted, key=lambda x: x['leveraged_return_pct'], reverse=True)[:5]

        for i, signal in enumerate(top_missed, 1):
            timestamp = signal['timestamp_str']
            direction = signal['direction']
            prob = signal['long_prob'] if direction == 'LONG' else signal['short_prob']
            entry_price = signal['price']
            leveraged_return = signal['leveraged_return_pct']
            hours_ago = signal['hours_ago']

            print(f"\n{i}. {timestamp} ({hours_ago:.1f}h ago)")
            print(f"   {direction} @ ${entry_price:,.2f} | Prob: {prob:.4f}")
            print(f"   💰 Potential Return: {leveraged_return:+.2f}%")


def save_results(signal_results, output_file):
    """Save detailed results to JSON"""
    print(f"\n💾 Saving results to {output_file.name}...")

    # Convert datetime to string for JSON serialization
    results_json = []
    for result in signal_results:
        result_copy = result.copy()
        result_copy['timestamp'] = result_copy['timestamp'].isoformat()
        results_json.append(result_copy)

    output = {
        'timestamp': datetime.now().isoformat(),
        'total_signals': len(signal_results),
        'executed': sum(1 for s in signal_results if s.get('executed', False)),
        'unexecuted': sum(1 for s in signal_results if not s.get('executed', False)),
        'signals': results_json
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  ✅ Results saved")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution"""
    print("="*80)
    print("ENTRY SIGNAL TRACKING ANALYSIS")
    print("="*80)
    print(f"Start Time: {datetime.now()}")

    # Find latest log file
    log_files = sorted(LOGS_DIR.glob("opportunity_gating_bot_4x_*.log"))
    if not log_files:
        print("❌ No log files found")
        return

    latest_log = log_files[-1]

    # Parse logs
    signals, actual_entries = parse_bot_logs(latest_log)

    # Filter entry signals
    entry_signals = filter_entry_signals(signals)

    if not entry_signals:
        print("\n⚠️ No entry signals found in logs")
        return

    # Fetch current price
    current_price = fetch_current_price()

    if current_price is None:
        print("\n❌ Cannot proceed without current price")
        return

    # Analyze signals
    signal_results = analyze_signals(entry_signals, current_price)

    # Load actual trades
    actual_trades = load_actual_trades()

    # Match signals to trades
    signal_results, unmatched_signals = match_signals_to_trades(signal_results, actual_trades)

    # Print report
    print_signal_report(signal_results, unmatched_signals)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"entry_signal_tracking_{timestamp}.json"
    save_results(signal_results, output_file)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
