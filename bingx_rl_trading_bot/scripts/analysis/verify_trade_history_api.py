"""
API로 실제 거래 내역 확인 스크립트

목적:
- BingX API에서 직접 거래 내역 가져오기
- State file과 교차 검증
- 00:24~04:24 기간의 모든 거래 확인
- 4분 gap (-$108.69) 미스터리 해결
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
import yaml

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient
from src.api.exceptions import BingXAPIError

# Configuration
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"

def load_api_keys():
    """Load API keys from config"""
    api_keys_file = CONFIG_DIR / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('testnet', {})
    return {}

def get_trade_history(client, symbol="BTC-USDT", start_time=None, end_time=None):
    """
    Get trade history from BingX API

    Args:
        client: BingX client
        symbol: Trading pair
        start_time: Start timestamp (ms)
        end_time: End timestamp (ms)

    Returns:
        List of trades
    """
    try:
        # BingX API: Get trades for symbol
        # Note: BingX uses CCXT, which has fetch_my_trades method

        # Convert datetime to milliseconds
        if start_time and isinstance(start_time, datetime):
            start_time = int(start_time.timestamp() * 1000)
        if end_time and isinstance(end_time, datetime):
            end_time = int(end_time.timestamp() * 1000)

        # Fetch trades
        params = {}
        if start_time:
            params['since'] = start_time

        trades = client.exchange.fetch_my_trades(
            symbol=symbol,
            since=start_time,
            params=params
        )

        logger.info(f"✅ Fetched {len(trades)} trades from API")
        return trades

    except Exception as e:
        logger.error(f"Failed to fetch trade history: {e}")
        return []

def get_funding_fee_history(client, symbol="BTC-USDT", start_time=None, end_time=None):
    """
    Get funding fee history from BingX API

    Args:
        client: BingX client
        symbol: Trading pair
        start_time: Start timestamp (ms)
        end_time: End timestamp (ms)

    Returns:
        List of funding fees
    """
    try:
        # Convert datetime to milliseconds
        if start_time and isinstance(start_time, datetime):
            start_time = int(start_time.timestamp() * 1000)
        if end_time and isinstance(end_time, datetime):
            end_time = int(end_time.timestamp() * 1000)

        # BingX API: Get funding fee history
        # Note: May require specific API call, check BingX docs

        # Try fetching funding fees (method may vary)
        try:
            funding_fees = client.exchange.fetch_funding_history(
                symbol=symbol,
                since=start_time
            )
            logger.info(f"✅ Fetched {len(funding_fees)} funding fees from API")
            return funding_fees
        except AttributeError:
            logger.warning("fetch_funding_history not available, trying alternative method")

            # Alternative: Check if there's a private API method
            # This depends on BingX API structure
            return []

    except Exception as e:
        logger.error(f"Failed to fetch funding fee history: {e}")
        return []

def analyze_trade_history(trades, target_start, target_end):
    """
    Analyze trade history for specific time period

    Args:
        trades: List of trades from API
        target_start: Target period start (datetime)
        target_end: Target period end (datetime)

    Returns:
        Dict with analysis results
    """
    if not trades:
        logger.warning("No trades to analyze")
        return {}

    # Convert to DataFrame
    df = pd.DataFrame(trades)

    # Parse timestamps
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        logger.error("No timestamp column found in trades")
        return {}

    # Filter to target period
    mask = (df['datetime'] >= target_start) & (df['datetime'] <= target_end)
    period_trades = df[mask].copy()

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Trade History Analysis")
    logger.info(f"{'=' * 80}")
    logger.info(f"Target Period: {target_start} → {target_end}")
    logger.info(f"Total Trades (All Time): {len(df)}")
    logger.info(f"Trades in Period: {len(period_trades)}")
    logger.info(f"")

    if len(period_trades) == 0:
        logger.warning("⚠️ No trades found in target period!")
        return {
            'total_trades': len(df),
            'period_trades': 0,
            'period_pnl': 0.0
        }

    # Calculate P&L for period
    if 'fee' in period_trades.columns and 'cost' in period_trades.columns:
        period_fees = period_trades['fee'].apply(lambda x: float(x['cost']) if isinstance(x, dict) else 0).sum()
        period_value = period_trades['cost'].sum()

        logger.info(f"Period Summary:")
        logger.info(f"  Total Value: ${period_value:,.2f}")
        logger.info(f"  Total Fees: ${period_fees:,.2f}")
        logger.info(f"")

    # Print trade details
    logger.info(f"Trade Details:")
    for idx, trade in period_trades.iterrows():
        trade_time = trade['datetime']
        side = trade.get('side', 'N/A')
        amount = trade.get('amount', 0)
        price = trade.get('price', 0)
        cost = trade.get('cost', 0)
        fee = trade.get('fee', {})
        fee_cost = float(fee.get('cost', 0)) if isinstance(fee, dict) else 0

        logger.info(f"  {trade_time}: {side.upper()} {amount:.4f} BTC @ ${price:,.2f} (${cost:,.2f}, fee: ${fee_cost:.2f})")

    logger.info(f"{'=' * 80}")

    return {
        'total_trades': len(df),
        'period_trades': len(period_trades),
        'period_df': period_trades
    }

def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("BingX API Trade History Verification")
    logger.info("=" * 80)

    # Load API keys
    api_config = load_api_keys()
    api_key = api_config.get('api_key', os.getenv("BINGX_API_KEY", ""))
    api_secret = api_config.get('secret_key', os.getenv("BINGX_API_SECRET", ""))

    if not api_key or not api_secret:
        logger.error("❌ API credentials not found!")
        logger.error("Set credentials in config/api_keys.yaml or environment variables")
        return

    # Initialize client
    try:
        client = BingXClient(
            api_key=api_key,
            secret_key=api_secret,
            testnet=True,
            timeout=30
        )
        logger.success("✅ BingX Testnet Client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize client: {e}")
        return

    # Test connection
    if not client.ping():
        logger.error("❌ Failed to connect to BingX API")
        return
    logger.success("✅ BingX API connection verified")

    # Define target period (00:24~04:24 on 2025-10-14)
    target_start = datetime(2025, 10, 14, 0, 24, 0)
    target_end = datetime(2025, 10, 14, 4, 24, 0)

    logger.info(f"\n🔍 Analyzing period: {target_start} → {target_end}")
    logger.info(f"   Duration: 4 hours")
    logger.info(f"   Looking for: Missing trades, funding fees, balance changes")
    logger.info("")

    # Get trade history
    logger.info("Fetching trade history from API...")
    trades = get_trade_history(
        client,
        symbol="BTC-USDT",
        start_time=target_start - timedelta(hours=1),  # 1 hour buffer before
        end_time=target_end + timedelta(hours=1)  # 1 hour buffer after
    )

    if trades:
        analysis = analyze_trade_history(trades, target_start, target_end)

        # Save to CSV
        if analysis.get('period_df') is not None and len(analysis['period_df']) > 0:
            output_file = RESULTS_DIR / f"api_trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            analysis['period_df'].to_csv(output_file, index=False)
            logger.success(f"\n✅ Trade history saved to: {output_file}")
    else:
        logger.warning("⚠️ No trades returned from API")

    # Get funding fee history
    logger.info("\nFetching funding fee history from API...")
    funding_fees = get_funding_fee_history(
        client,
        symbol="BTC-USDT",
        start_time=target_start - timedelta(hours=1),
        end_time=target_end + timedelta(hours=1)
    )

    if funding_fees:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Funding Fee History")
        logger.info(f"{'=' * 80}")

        df_fees = pd.DataFrame(funding_fees)
        if 'timestamp' in df_fees.columns:
            df_fees['datetime'] = pd.to_datetime(df_fees['timestamp'], unit='ms')

            # Filter to target period
            mask = (df_fees['datetime'] >= target_start) & (df_fees['datetime'] <= target_end)
            period_fees = df_fees[mask].copy()

            logger.info(f"Total Funding Fees (All Time): {len(df_fees)}")
            logger.info(f"Funding Fees in Period: {len(period_fees)}")

            if len(period_fees) > 0:
                for idx, fee in period_fees.iterrows():
                    fee_time = fee.get('datetime', 'N/A')
                    amount = fee.get('amount', 0)
                    logger.info(f"  {fee_time}: ${amount:+,.2f}")

                total_funding = period_fees['amount'].sum() if 'amount' in period_fees.columns else 0
                logger.info(f"\nTotal Funding Fees in Period: ${total_funding:+,.2f}")

            logger.info(f"{'=' * 80}")

            # Save to CSV
            output_file = RESULTS_DIR / f"api_funding_fees_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            period_fees.to_csv(output_file, index=False)
            logger.success(f"\n✅ Funding fees saved to: {output_file}")
    else:
        logger.warning("⚠️ No funding fees returned from API (may not be available)")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Verification Complete!")
    logger.info("=" * 80)
    logger.info("\nNext Steps:")
    logger.info("1. Compare API trades with State file trades")
    logger.info("2. Check for missing trades in 00:24~04:24 period")
    logger.info("3. Verify funding fees match balance changes")
    logger.info("4. Investigate 4-minute gap (-$108.69) with actual API data")

if __name__ == "__main__":
    main()
