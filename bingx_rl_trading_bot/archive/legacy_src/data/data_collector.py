"""데이터 수집 모듈"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

from ..api.bingx_client import BingXClient
from ..api.exceptions import BingXAPIError


class DataCollector:
    """
    BingX에서 캔들 데이터 수집
    """

    def __init__(
        self,
        client: BingXClient,
        symbol: str = "BTC-USDT",
        interval: str = "5m",
        data_dir: str = None
    ):
        """
        Args:
            client: BingX API 클라이언트
            symbol: 거래 쌍
            interval: 캔들 간격
            data_dir: 데이터 저장 디렉토리
        """
        self.client = client
        self.symbol = symbol
        self.interval = interval

        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / 'data' / 'historical'

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def collect_historical_data(
        self,
        days: int = 30,
        save: bool = True
    ) -> pd.DataFrame:
        """
        과거 데이터 수집

        Note: BingX API는 최대 1000개 캔들만 반환하므로,
        최근 데이터부터 수집합니다.

        Args:
            days: 수집할 일수 (참고용, 실제로는 API 제한에 따라 조정됨)
            save: 데이터 저장 여부

        Returns:
            캔들 데이터프레임
        """
        logger.info(f"Collecting historical {self.interval} data for {self.symbol}")
        logger.info(f"Target: {days} days (API may limit to ~1000 candles)")

        # BingX API는 최대 1000개까지만 지원
        # 5분 캔들 기준: 1000개 = 약 3.47일
        max_candles = 1000

        try:
            # 최근 1000개 캔들 수집
            logger.info(f"Requesting {max_candles} recent candles...")
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=max_candles
            )

            if not klines:
                logger.error("No data received from API")
                return pd.DataFrame()

            logger.info(f"Collected {len(klines)} candles")

            # 데이터프레임 변환
            df = self._parse_klines(klines)

            if save and not df.empty:
                self._save_data(df)

            # 실제 수집된 기간 계산
            if not df.empty:
                time_range = df['timestamp'].max() - df['timestamp'].min()
                actual_days = time_range.total_seconds() / (24 * 60 * 60)
                logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                logger.info(f"Actual days collected: {actual_days:.2f} days")

            return df

        except BingXAPIError as e:
            logger.error(f"API error: {e.message}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def collect_recent_data(
        self,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        최근 데이터 수집

        Args:
            limit: 수집할 캔들 개수

        Returns:
            캔들 데이터프레임
        """
        try:
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=limit
            )

            df = self._parse_klines(klines)
            logger.info(f"Collected {len(df)} recent candles")
            return df

        except Exception as e:
            logger.error(f"Failed to collect recent data: {str(e)}")
            return pd.DataFrame()

    def update_data(self) -> pd.DataFrame:
        """
        저장된 데이터 업데이트 (최신 캔들 추가)

        Returns:
            업데이트된 데이터프레임
        """
        # 기존 데이터 로드
        existing_df = self.load_data()

        if existing_df.empty:
            # 기존 데이터 없으면 전체 수집
            return self.collect_historical_data()

        # 마지막 타임스탬프
        last_timestamp = existing_df['timestamp'].max()
        last_time_ms = int(last_timestamp.timestamp() * 1000)

        # 현재 시간까지의 데이터 수집
        current_time_ms = int(time.time() * 1000)

        try:
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                start_time=last_time_ms,
                end_time=current_time_ms,
                limit=1000
            )

            new_df = self._parse_klines(klines)

            if not new_df.empty:
                # 중복 제거 후 병합
                combined_df = pd.concat([existing_df, new_df])
                combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

                self._save_data(combined_df)
                logger.info(f"Updated {len(new_df)} new candles")
                return combined_df
            else:
                logger.info("No new data to update")
                return existing_df

        except Exception as e:
            logger.error(f"Failed to update data: {str(e)}")
            return existing_df

    def load_data(self, filename: str = None) -> pd.DataFrame:
        """
        저장된 데이터 로드

        Args:
            filename: 파일명 (None이면 기본 파일명 사용)

        Returns:
            캔들 데이터프레임
        """
        if filename is None:
            filename = f"{self.symbol.replace('-', '')}_{self.interval}.csv"

        filepath = self.data_dir / filename

        if not filepath.exists():
            logger.warning(f"Data file not found: {filepath}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            logger.info(f"Loaded {len(df)} candles from {filename}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return pd.DataFrame()

    def _parse_klines(self, klines: List[dict]) -> pd.DataFrame:
        """
        API 응답을 데이터프레임으로 변환

        Args:
            klines: API 응답 데이터

        Returns:
            캔들 데이터프레임
        """
        if not klines:
            return pd.DataFrame()

        data = []
        for kline in klines:
            # BingX API 응답 구조에 따라 조정 필요
            data.append({
                'timestamp': pd.to_datetime(kline['time'], unit='ms'),
                'open': float(kline['open']),
                'high': float(kline['high']),
                'low': float(kline['low']),
                'close': float(kline['close']),
                'volume': float(kline['volume']),
            })

        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def _save_data(self, df: pd.DataFrame, filename: str = None) -> None:
        """
        데이터 저장

        Args:
            df: 저장할 데이터프레임
            filename: 파일명 (None이면 기본 파일명 사용)
        """
        if filename is None:
            filename = f"{self.symbol.replace('-', '')}_{self.interval}.csv"

        filepath = self.data_dir / filename

        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")

    def _interval_to_milliseconds(self, interval: str) -> int:
        """
        간격 문자열을 밀리초로 변환

        Args:
            interval: 간격 문자열 (1m, 5m, 15m, 1h, 1d)

        Returns:
            밀리초
        """
        unit = interval[-1]
        value = int(interval[:-1])

        if unit == 'm':
            return value * 60 * 1000
        elif unit == 'h':
            return value * 60 * 60 * 1000
        elif unit == 'd':
            return value * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Invalid interval: {interval}")

    def get_latest_price(self) -> float:
        """
        최신 가격 조회

        Returns:
            현재 가격
        """
        try:
            ticker = self.client.get_ticker(self.symbol)
            return float(ticker['lastPrice'])
        except Exception as e:
            logger.error(f"Failed to get latest price: {str(e)}")
            return 0.0
