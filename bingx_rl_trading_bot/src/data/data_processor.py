"""데이터 전처리 모듈"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    """
    데이터 전처리 및 정규화
    """

    def __init__(self, lookback_window: int = 50):
        """
        Args:
            lookback_window: 과거 데이터 참조 윈도우
        """
        self.lookback_window = lookback_window
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.fitted = False

    def prepare_data(
        self,
        df: pd.DataFrame,
        fit: bool = False
    ) -> pd.DataFrame:
        """
        데이터 전처리

        Args:
            df: 원본 데이터프레임
            fit: 스케일러 피팅 여부

        Returns:
            전처리된 데이터프레임
        """
        if df.empty:
            logger.warning("Empty dataframe provided")
            return df

        df = df.copy()

        # 결측치 제거
        df = df.dropna()

        # 기본 가격 특성
        df['price_change'] = df['close'].pct_change()
        df['high_low_spread'] = (df['high'] - df['low']) / df['low']
        df['close_open_spread'] = (df['close'] - df['open']) / df['open']

        # 볼륨 특성
        df['volume_change'] = df['volume'].pct_change()

        # 이동평균
        for period in [5, 10, 20]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()

        # 결측치 다시 제거 (롤링 계산으로 인한 NaN)
        df = df.dropna()

        # 정규화 (선택)
        if fit:
            self._fit_scaler(df)

        if self.fitted:
            df = self._normalize(df)

        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: list = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 시퀀스 생성 (강화학습용)

        Args:
            df: 데이터프레임
            feature_columns: 사용할 특성 컬럼

        Returns:
            (특성 배열, 타임스탬프 배열)
        """
        if feature_columns is None:
            # 기본 특성 컬럼
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'price_change', 'high_low_spread', 'close_open_spread',
                'volume_change'
            ]

        # 사용 가능한 컬럼만 선택
        available_columns = [col for col in feature_columns if col in df.columns]

        if not available_columns:
            logger.error("No valid feature columns found")
            return np.array([]), np.array([])

        # 특성 배열 생성
        features = df[available_columns].values

        # 타임스탬프 배열
        timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df))

        return features, timestamps

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        데이터 분할 (train/val/test)

        Args:
            df: 데이터프레임
            train_ratio: 훈련 데이터 비율
            val_ratio: 검증 데이터 비율

        Returns:
            (train_df, val_df, test_df)
        """
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size + val_size].copy()
        test_df = df.iloc[train_size + val_size:].copy()

        logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        return train_df, val_df, test_df

    def _fit_scaler(self, df: pd.DataFrame) -> None:
        """
        스케일러 피팅

        Args:
            df: 데이터프레임
        """
        # 정규화할 컬럼 (가격, 볼륨)
        scale_columns = ['open', 'high', 'low', 'close', 'volume']
        scale_columns = [col for col in scale_columns if col in df.columns]

        if scale_columns:
            self.scaler.fit(df[scale_columns])
            self.fitted = True
            logger.info("Scaler fitted")

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 정규화

        Args:
            df: 데이터프레임

        Returns:
            정규화된 데이터프레임
        """
        df = df.copy()
        scale_columns = ['open', 'high', 'low', 'close', 'volume']
        scale_columns = [col for col in scale_columns if col in df.columns]

        if scale_columns and self.fitted:
            df[scale_columns] = self.scaler.transform(df[scale_columns])

        return df

    def inverse_transform_price(self, normalized_prices: np.ndarray) -> np.ndarray:
        """
        정규화된 가격을 원래 스케일로 변환

        Args:
            normalized_prices: 정규화된 가격 배열

        Returns:
            원래 스케일의 가격 배열
        """
        if not self.fitted:
            return normalized_prices

        # close 가격 컬럼의 인덱스 (일반적으로 3)
        dummy = np.zeros((len(normalized_prices), 5))
        dummy[:, 3] = normalized_prices

        inverse = self.scaler.inverse_transform(dummy)
        return inverse[:, 3]

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        수익률 계산

        Args:
            df: 데이터프레임

        Returns:
            수익률이 추가된 데이터프레임
        """
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        column: str = 'close',
        n_std: float = 3.0
    ) -> pd.DataFrame:
        """
        이상치 제거

        Args:
            df: 데이터프레임
            column: 이상치를 확인할 컬럼
            n_std: 표준편차 배수 (기본 3σ)

        Returns:
            이상치가 제거된 데이터프레임
        """
        df = df.copy()
        mean = df[column].mean()
        std = df[column].std()

        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std

        before_len = len(df)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        after_len = len(df)

        removed = before_len - after_len
        if removed > 0:
            logger.info(f"Removed {removed} outliers from {column}")

        return df
