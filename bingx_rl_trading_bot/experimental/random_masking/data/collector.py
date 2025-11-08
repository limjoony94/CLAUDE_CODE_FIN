"""
OHLCV Data Collection and Technical Indicator Calculation

Reuses BingX API infrastructure from main project.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from loguru import logger
import ta  # Technical Analysis library

# Add parent directory to path to import from main project
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

class BinanceCollector:
    """
    Cryptocurrency data collector using BingX/Binance API

    Features:
    - Historical OHLCV data collection
    - Technical indicator calculation
    - Efficient storage (parquet format)
    - Real-time data updates
    """

    def __init__(
        self,
        symbols: List[str] = ['BTC-USDT'],
        interval: str = '5m',
        start_date: str = '2022-01-01',
        api_type: str = 'bingx'  # 'bingx' or 'binance'
    ):
        """
        Initialize data collector

        Args:
            symbols: List of trading pairs (e.g., ['BTC-USDT', 'ETH-USDT'])
            interval: Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')
            start_date: Start date for historical data (YYYY-MM-DD)
            api_type: API to use ('bingx' or 'binance')
        """
        self.symbols = symbols
        self.interval = interval
        self.start_date = pd.to_datetime(start_date)
        self.api_type = api_type

        # Initialize API client (reuse from main project if available)
        self._init_api_client()

        logger.info(f"Initialized {api_type.upper()} collector for {symbols}")
        logger.info(f"Interval: {interval}, Start date: {start_date}")

    def _init_api_client(self):
        """Initialize API client"""
        try:
            if self.api_type == 'bingx':
                # Import from main project
                from scripts.data.fetch_historical_bingx import fetch_ohlcv_bingx
                self.fetch_func = fetch_ohlcv_bingx
            else:
                # Use ccxt for Binance
                import ccxt
                self.exchange = ccxt.binance({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'}
                })
                self.fetch_func = self._fetch_binance

        except ImportError as e:
            logger.error(f"Failed to initialize API client: {e}")
            logger.info("Install ccxt: pip install ccxt")
            raise

    def _fetch_binance(self, symbol: str, since: int, limit: int = 1000):
        """Fetch data from Binance using ccxt"""
        ohlcv = self.exchange.fetch_ohlcv(
            symbol.replace('-', '/'),
            timeframe=self.interval,
            since=since,
            limit=limit
        )
        return ohlcv

    def fetch_historical(self, symbol: str) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol

        Args:
            symbol: Trading pair (e.g., 'BTC-USDT')

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Fetching historical data for {symbol}...")

        all_data = []
        current_time = int(self.start_date.timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)

        # Calculate interval in milliseconds
        interval_ms = self._get_interval_ms()

        while current_time < end_time:
            try:
                # Fetch batch
                if self.api_type == 'bingx':
                    # Use main project's fetch function
                    data = self.fetch_func(
                        symbol=symbol.replace('-', ''),
                        interval=self.interval,
                        start_time=current_time,
                        limit=1000
                    )
                else:
                    data = self.fetch_func(symbol, current_time, 1000)

                if not data:
                    break

                all_data.extend(data)

                # Update timestamp for next batch
                current_time = data[-1][0] + interval_ms

                logger.debug(f"Fetched {len(data)} candles, up to {pd.to_datetime(current_time, unit='ms')}")

            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')

        logger.info(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        return df

    def _get_interval_ms(self) -> int:
        """Convert interval string to milliseconds"""
        interval_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        return interval_map.get(self.interval, 5 * 60 * 1000)

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV data

        Indicators added:
        - RSI (14, 28)
        - MACD (12, 26, 9)
        - Bollinger Bands (20, 2)
        - ATR (14)
        - EMA (9, 21, 50, 200)
        - Stochastic RSI
        - Volume features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicators added
        """
        logger.info("Calculating technical indicators...")

        df = df.copy()

        # RSI
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_28'] = ta.momentum.RSIIndicator(df['close'], window=28).rsi()

        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()

        # ATR
        df['atr_14'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()

        # EMAs
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()

        # Stochastic RSI
        stoch_rsi = ta.momentum.StochRSIIndicator(df['close'])
        df['stoch_rsi'] = stoch_rsi.stochrsi()
        df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
        df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()

        # Volume features
        df['volume_ema_20'] = ta.trend.EMAIndicator(df['volume'], window=20).ema_indicator()
        df['volume_ratio'] = df['volume'] / df['volume_ema_20']

        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_1h'] = df['close'].pct_change(periods=12)  # 12 * 5min = 1h

        # Drop NaN rows (from indicator calculation)
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)

        logger.info(f"Added technical indicators. Dropped {dropped} NaN rows. Remaining: {len(df)} rows")

        return df

    def fetch_and_process(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch and process data for symbol(s)

        Args:
            symbol: Specific symbol to fetch. If None, fetches all configured symbols.

        Returns:
            Processed DataFrame with OHLCV + technical indicators
        """
        if symbol:
            symbols = [symbol]
        else:
            symbols = self.symbols

        all_data = {}

        for sym in symbols:
            # Fetch OHLCV
            df = self.fetch_historical(sym)

            # Add technical indicators
            df = self.add_technical_indicators(df)

            # Add symbol column
            df['symbol'] = sym

            all_data[sym] = df

            logger.info(f"Processed {sym}: {len(df)} rows, {len(df.columns)} features")

        # Combine if multiple symbols
        if len(all_data) == 1:
            return list(all_data.values())[0]
        else:
            return pd.concat(all_data.values(), keys=all_data.keys())

    def save_to_parquet(self, df: pd.DataFrame, filename: str):
        """
        Save DataFrame to parquet format (efficient storage)

        Args:
            df: DataFrame to save
            filename: Output filename (should end with .parquet)
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save
        df.to_parquet(filename, compression='snappy')

        # Log file size
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        logger.info(f"Saved {len(df)} rows to {filename} ({size_mb:.2f} MB)")

    def update_data(self, existing_file: str) -> pd.DataFrame:
        """
        Update existing data file with new candles

        Args:
            existing_file: Path to existing parquet file

        Returns:
            Updated DataFrame
        """
        # Load existing data
        df_old = pd.read_parquet(existing_file)
        last_timestamp = df_old.index[-1]

        logger.info(f"Updating data from {last_timestamp}...")

        # Temporarily update start_date
        original_start = self.start_date
        self.start_date = last_timestamp

        # Fetch new data
        df_new = self.fetch_and_process()

        # Restore original start_date
        self.start_date = original_start

        # Combine (remove duplicates)
        df_combined = pd.concat([df_old, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        df_combined = df_combined.sort_index()

        logger.info(f"Added {len(df_new)} new rows. Total: {len(df_combined)} rows")

        return df_combined


if __name__ == '__main__':
    # Example usage
    collector = BinanceCollector(
        symbols=['BTC-USDT'],
        interval='5m',
        start_date='2024-01-01',
        api_type='bingx'
    )

    data = collector.fetch_and_process()
    collector.save_to_parquet(data, '../../../data/raw/btc_5m.parquet')

    print(f"\nData shape: {data.shape}")
    print(f"\nColumns: {list(data.columns)}")
    print(f"\nFirst 5 rows:\n{data.head()}")
