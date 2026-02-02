#!/usr/bin/env python3
"""
Real-time monitoring script for Sweet-2 Paper Trading Bot
"""

import time
from pathlib import Path
from datetime import datetime
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / f"phase4_dynamic_paper_trading_{datetime.now().strftime('%Y%m%d')}.log"

def parse_log_line(line):
    """Parse key information from log line"""
    info = {}

    # Current Price
    if "Current Price:" in line:
        match = re.search(r'\$([0-9,]+\.\d+)', line)
        if match:
            info['price'] = match.group(1)

    # XGBoost Prob
    if "XGBoost Prob:" in line:
        match = re.search(r'XGBoost Prob: ([\d.]+)', line)
        if match:
            info['xgb_prob'] = match.group(1)

    # Tech Signal
    if "Tech Signal:" in line:
        match = re.search(r'Tech Signal: (\w+)', line)
        if match:
            info['tech_signal'] = match.group(1)

    # Should Enter
    if "Should Enter:" in line:
        match = re.search(r'Should Enter: (\w+)', line)
        if match:
            info['should_enter'] = match.group(1)

    # Live data
    if "Live data from BingX API" in line:
        match = re.search(r'(\d+) candles', line)
        if match:
            info['candles'] = match.group(1)

    # Simulation data (warning)
    if "Simulation data from file" in line:
        info['warning'] = 'USING SIMULATION DATA'

    # Trade executed
    if "📈 Entered LONG" in line or "🔔 Entered LONG" in line:
        info['trade'] = 'ENTRY'

    if "✅ Closed position" in line:
        info['trade'] = 'EXIT'

    # Errors
    if "ERROR" in line or "Failed to get live data" in line:
        info['error'] = line.strip()

    return info

def get_latest_status():
    """Get latest status from log file"""
    if not LOG_FILE.exists():
        return None

    status = {
        'price': 'N/A',
        'xgb_prob': 'N/A',
        'tech_signal': 'N/A',
        'should_enter': 'N/A',
        'data_source': 'Unknown',
        'last_update': 'N/A',
        'warnings': [],
        'trades': []
    }

    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            # Get last 100 lines for analysis
            recent_lines = lines[-100:] if len(lines) > 100 else lines

            for line in recent_lines:
                info = parse_log_line(line)

                if 'price' in info:
                    status['price'] = info['price']

                if 'xgb_prob' in info:
                    status['xgb_prob'] = info['xgb_prob']

                if 'tech_signal' in info:
                    status['tech_signal'] = info['tech_signal']

                if 'should_enter' in info:
                    status['should_enter'] = info['should_enter']

                if 'candles' in info:
                    status['data_source'] = f'✅ BingX API ({info["candles"]} candles)'

                if 'warning' in info:
                    status['data_source'] = '⚠️ SIMULATION FILE'
                    status['warnings'].append(info['warning'])

                if 'trade' in info:
                    status['trades'].append(info['trade'])

                if 'error' in info:
                    status['warnings'].append(info['error'])

                # Get timestamp
                if line.strip():
                    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match:
                        status['last_update'] = timestamp_match.group(1)

    except Exception as e:
        print(f"Error reading log: {e}")
        return None

    return status

def print_status(status):
    """Print formatted status"""
    print("\n" + "="*80)
    print(f"🤖 Phase 4 Dynamic Paper Trading Bot - Real-time Monitor")
    print("="*80)
    print(f"⏰ Last Update:    {status['last_update']}")
    print(f"💰 Current Price:  ${status['price']}")
    print(f"📊 Data Source:    {status['data_source']}")
    print("-"*80)
    print(f"🎯 XGBoost Prob:   {status['xgb_prob']} (threshold: 0.7)")
    print(f"📈 Tech Signal:    {status['tech_signal']}")
    print(f"✅ Should Enter:   {status['should_enter']}")
    print("-"*80)

    if status['trades']:
        print(f"📊 Recent Trades:  {len(status['trades'])} ({', '.join(status['trades'][-5:])})")
    else:
        print(f"📊 Recent Trades:  0 (waiting for setup)")

    if status['warnings']:
        print("-"*80)
        print("⚠️  WARNINGS:")
        for warning in status['warnings'][-3:]:  # Show last 3 warnings
            print(f"   {warning}")

    print("="*80)
    print(f"Next check in 30 seconds... (Press Ctrl+C to stop)")

def main():
    """Main monitoring loop"""
    print("\n🚀 Starting Phase 4 Dynamic Paper Trading Bot Monitor...")
    print(f"📁 Log file: {LOG_FILE}")
    print("\n" + "="*80)

    if not LOG_FILE.exists():
        print(f"❌ Log file not found: {LOG_FILE}")
        print("Make sure the bot is running.")
        return

    try:
        while True:
            status = get_latest_status()

            if status:
                print_status(status)
            else:
                print("❌ Failed to get status")

            time.sleep(30)  # Check every 30 seconds

    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
