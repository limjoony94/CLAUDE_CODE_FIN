"""
Data Caching System for Incremental Candle Collection

Solves the critical issue where bot never reaches 1440 candles
by persistently storing and accumulating historical data.

Problem: API returns max 500 candles, but bot needs 1440 for ML models
Solution: Cache candles to CSV, append new ones incrementally
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger


class DataCache:
    """
    Persistent data cache for incremental candle accumulation

    Features:
    - CSV-based storage (simple, reliable)
    - Automatic deduplication by timestamp
    - Incremental append (only new candles)
    - Thread-safe file operations
    """

    def __init__(self, cache_dir: Path, symbol: str, timeframe: str):
        """
        Initialize data cache

        Args:
            cache_dir: Directory for cache files
            symbol: Trading symbol (e.g., "BTC-USDT")
            timeframe: Candle timeframe (e.g., "5m")
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache file path: data/cache/BTC-USDT_5m.csv
        safe_symbol = symbol.replace("-", "")  # BTC-USDT -> BTCUSDT
        self.cache_file = self.cache_dir / f"{safe_symbol}_{timeframe}.csv"

        self.symbol = symbol
        self.timeframe = timeframe

        logger.info(f"📦 Data Cache initialized: {self.cache_file}")

        # Load existing cache if available
        self._cache_df = self._load_cache()

        if self._cache_df is not None and len(self._cache_df) > 0:
            logger.success(f"   Loaded {len(self._cache_df)} cached candles")
            logger.info(f"   Date range: {self._cache_df['timestamp'].iloc[0]} → {self._cache_df['timestamp'].iloc[-1]}")
        else:
            logger.info(f"   No existing cache (will create new)")

    def _load_cache(self) -> pd.DataFrame:
        """Load cache from CSV if exists"""
        if not self.cache_file.exists():
            return None

        try:
            df = pd.read_csv(self.cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Sort by timestamp (ensure chronological order)
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Remove duplicates (keep latest)
            df = df.drop_duplicates(subset=['timestamp'], keep='last')

            return df

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None

    def update(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Update cache with new candles

        Args:
            new_df: New candles from API (DataFrame with timestamp column)

        Returns:
            Combined DataFrame (old + new, deduplicated)
        """
        if new_df is None or len(new_df) == 0:
            logger.warning("No new data to cache")
            return self._cache_df if self._cache_df is not None else pd.DataFrame()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(new_df['timestamp']):
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])

        # Combine with existing cache
        if self._cache_df is not None and len(self._cache_df) > 0:
            combined = pd.concat([self._cache_df, new_df], ignore_index=True)
        else:
            combined = new_df.copy()

        # Sort and deduplicate
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
        after_dedup = len(combined)

        duplicates_removed = before_dedup - after_dedup
        new_candles = len(combined) - (len(self._cache_df) if self._cache_df is not None else 0)

        if new_candles > 0:
            logger.success(f"📦 Cache updated: +{new_candles} new candles (total: {len(combined)})")
            if duplicates_removed > 0:
                logger.debug(f"   Removed {duplicates_removed} duplicates")
        else:
            logger.debug(f"📦 Cache unchanged: {len(combined)} candles (no new data)")

        # Update internal cache
        self._cache_df = combined

        # Save to disk
        self._save_cache()

        return combined

    def _save_cache(self):
        """Save cache to CSV"""
        if self._cache_df is None or len(self._cache_df) == 0:
            return

        try:
            # Save with timestamp as string (for CSV compatibility)
            save_df = self._cache_df.copy()
            save_df['timestamp'] = save_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            save_df.to_csv(self.cache_file, index=False)
            logger.debug(f"   Cache saved: {self.cache_file}")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get(self, limit: int = None) -> pd.DataFrame:
        """
        Get cached data

        Args:
            limit: Maximum number of recent candles to return (None = all)

        Returns:
            DataFrame with cached candles (most recent first if limited)
        """
        if self._cache_df is None or len(self._cache_df) == 0:
            return pd.DataFrame()

        if limit is not None and limit > 0:
            return self._cache_df.tail(limit).reset_index(drop=True)

        return self._cache_df.copy()

    def count(self) -> int:
        """Get number of cached candles"""
        return len(self._cache_df) if self._cache_df is not None else 0

    def clear(self):
        """Clear cache (delete file)"""
        if self.cache_file.exists():
            self.cache_file.unlink()
            logger.warning(f"🗑️ Cache cleared: {self.cache_file}")

        self._cache_df = None
