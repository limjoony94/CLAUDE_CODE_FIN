"""최대한 많은 데이터 수집"""

import sys
from pathlib import Path
import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd
from loguru import logger

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():

    logger.info("="*60)
    logger.info("MAXIMUM DATA COLLECTION - MAINNET")
    logger.info("="*60)
    logger.info("⚠️ Collecting REAL mainnet data (NOT testnet!)")

    # OHLCV 데이터는 public이므로 API 키 불필요
    exchange = ccxt.bingx({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })

    # Mainnet mode (sandbox mode 비활성화)
    logger.info("✅ Mainnet mode enabled")

    symbol = 'BTC/USDT:USDT'
    timeframe = '5m'

    # 역순 수집 전략: 최신부터 과거로
    logger.info(f"Attempting to collect maximum historical {timeframe} mainnet data")
    logger.info("Strategy: Fetch backwards from recent to past")

    all_candles = []
    batch_count = 0
    max_batches = 50  # 최대 50회 요청 (50,000 캔들 가능)

    # 첫 배치: 최신 데이터
    until = None  # 최신부터 시작

    try:
        while batch_count < max_batches:
            batch_count += 1

            if until:
                logger.info(f"Batch {batch_count}/{max_batches} - Until {datetime.fromtimestamp(until/1000)}")
            else:
                logger.info(f"Batch {batch_count}/{max_batches} - Fetching latest data")

            try:
                # BingX에서 과거 데이터를 가져오기 위해 'since' 사용
                # since를 점점 과거로 이동
                params = {
                    'limit': 1000
                }

                if until:
                    # until 이전 데이터 가져오기 (역순)
                    # until - (1000 candles * 5 minutes * 60 seconds * 1000 ms)
                    since_for_batch = until - (1000 * 5 * 60 * 1000) - 1
                    params['since'] = since_for_batch

                candles = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    **params
                )

                if not candles or len(candles) == 0:
                    logger.info("No more historical data available")
                    break

                # 중복 제거: 이미 가져온 타임스탬프는 제외
                if all_candles:
                    existing_timestamps = {c[0] for c in all_candles}
                    new_candles = [c for c in candles if c[0] not in existing_timestamps]
                else:
                    new_candles = candles

                if not new_candles:
                    logger.info("No new candles (all duplicates)")
                    break

                all_candles.extend(new_candles)
                logger.info(f"  Fetched {len(new_candles)} new candles (Total: {len(all_candles)})")

                # 다음 배치를 위해 가장 오래된 타임스탬프 저장
                until = min(c[0] for c in new_candles)

                # Rate limit 준수
                time.sleep(exchange.rateLimit / 1000)

            except Exception as api_error:
                logger.error(f"API Error: {api_error}")
                break

    except Exception as e:
        logger.error(f"Error: {str(e)}")

    if not all_candles:
        logger.error("No data collected!")
        return

    # DataFrame 변환
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp'], keep='last')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 저장
    save_dir = project_root / 'data' / 'historical'
    filepath = save_dir / 'BTCUSDT_5m_max.csv'
    df.to_csv(filepath, index=False)

    actual_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400

    logger.info("\n" + "="*60)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total Candles:  {len(df):>10,}")
    logger.info(f"Date Range:     {df['timestamp'].min()} → {df['timestamp'].max()}")
    logger.info(f"Actual Days:    {actual_days:>10.2f}")
    logger.info(f"Saved to:       {filepath}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
