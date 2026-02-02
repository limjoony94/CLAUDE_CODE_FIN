"""
Trade Reconciliation Script
===========================

Run trade reconciliation to detect and record manual trades.

Usage:
    python scripts/analysis/reconcile_trades.py [--days DAYS]

Author: Claude Code
Date: 2025-10-19
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient
from src.monitoring.trade_reconciliation import TradeReconciliation
import yaml
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Reconcile bot trades with exchange API')
    parser.add_argument('--days', type=int, default=7,
                       help='Days of history to check (default: 7)')
    args = parser.parse_args()

    # Load API keys
    with open(PROJECT_ROOT / 'config' / 'api_keys.yaml', 'r') as f:
        config = yaml.safe_load(f)
        api_config = config['bingx']['mainnet']

    # Initialize client
    client = BingXClient(api_config['api_key'], api_config['secret_key'], testnet=False)

    # Initialize reconciliation system
    state_file = PROJECT_ROOT / 'results' / 'opportunity_gating_bot_4x_state.json'
    reconciler = TradeReconciliation(client, state_file)

    # Run reconciliation
    logger.info(f"Running reconciliation for last {args.days} days...")
    result = reconciler.reconcile(lookback_days=args.days)

    # Print results
    print("\n" + "="*70)
    print("RECONCILIATION RESULT")
    print("="*70)

    if result.get('status') == 'success':
        print(f"✅ Manual trades found: {result['manual_trades_found']}")
        print(f"✅ Total manual P&L: ${result['total_manual_pnl']:.2f}")

        if result['trades_added']:
            print("\n📋 Added trades:")
            for i, trade in enumerate(result['trades_added'], 1):
                status = trade['status']
                pnl = trade.get('pnl_usd_net', 0)
                side = trade.get('side', 'N/A')
                entry_price = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
                print(f"   {i}. Position {trade['position_id_exchange']}")
                print(f"      Side: {side}")
                print(f"      Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}")
                print(f"      Status: {status}")
                print(f"      Net P&L: ${pnl:.2f}")

    elif result.get('status') == 'clean':
        print("✅ No manual trades detected - bot state is clean")

    elif result.get('status') == 'no_orders':
        print("⚠️  No exchange orders found in lookback period")

    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")

    print("="*70)


if __name__ == '__main__':
    main()
