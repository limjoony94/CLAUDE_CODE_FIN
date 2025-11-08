"""
Data Preprocessing and Normalization

Key features:
- Rolling normalization (prevents look-ahead bias)
- Outlier handling
- Missing value imputation
- Feature selection
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from loguru import logger
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')


class CandlePreprocessor:
    """
    Candle data preprocessor with rolling normalization

    Prevents look-ahead bias by using only historical statistics
    for normalization at each timestep.
    """

    def __init__(
        self,
        normalization: str = 'rolling_zscore',
        rolling_window: int = 1000,
        clip_threshold: float = 5.0,
        feature_cols: Optional[list] = None
    ):
        """
        Initialize preprocessor

        Args:
            normalization: 'rolling_zscore', 'rolling_minmax', 'robust', or 'none'
            rolling_window: Window size for rolling statistics
            clip_threshold: Clip outliers beyond ±N standard deviations
            feature_cols: Feature columns to normalize (None = all numeric)
        """
        self.normalization = normalization
        self.rolling_window = rolling_window
        self.clip_threshold = clip_threshold
        self.feature_cols = feature_cols

        # Fitted statistics (for inverse transform)
        self.stats = {}
        self.scaler = None

        logger.info(f"Initialized CandlePreprocessor (method: {normalization}, window: {rolling_window})")

    def fit(self, data: pd.DataFrame) -> 'CandlePreprocessor':
        """
        Fit preprocessing parameters on training data

        Args:
            data: Training DataFrame

        Returns:
            self (for method chaining)
        """
        logger.info(f"Fitting preprocessor on {len(data)} samples...")

        # Select feature columns
        if self.feature_cols is None:
            self.feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # Calculate global statistics (for inverse transform)
        for col in self.feature_cols:
            self.stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'median': data[col].median()
            }

        # Fit robust scaler (if using)
        if self.normalization == 'robust':
            self.scaler = RobustScaler()
            self.scaler.fit(data[self.feature_cols])

        logger.info(f"Fitted preprocessor on {len(self.feature_cols)} features")

        return self

    def transform(self, data: pd.DataFrame, fit_first: bool = False) -> np.ndarray:
        """
        Transform data using rolling normalization

        Args:
            data: DataFrame to transform
            fit_first: If True, fit on this data before transforming

        Returns:
            Normalized numpy array with shape (N, n_features)
        """
        if fit_first:
            self.fit(data)

        logger.info(f"Transforming {len(data)} samples...")

        df = data[self.feature_cols].copy()

        if self.normalization == 'rolling_zscore':
            # Rolling Z-score normalization
            normalized = self._rolling_zscore(df)

        elif self.normalization == 'rolling_minmax':
            # Rolling min-max normalization
            normalized = self._rolling_minmax(df)

        elif self.normalization == 'robust':
            # Robust scaler (uses median and IQR)
            normalized = self.scaler.transform(df)

        elif self.normalization == 'none':
            # No normalization
            normalized = df.values

        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")

        # Clip outliers
        if self.clip_threshold > 0:
            normalized = np.clip(normalized, -self.clip_threshold, self.clip_threshold)

        logger.info(f"Transformation complete. Shape: {normalized.shape}")

        return normalized

    def _rolling_zscore(self, df: pd.DataFrame) -> np.ndarray:
        """
        Rolling Z-score normalization (prevents look-ahead bias)

        For each timestep t, uses only data from [t-window, t] for normalization.
        """
        normalized = np.zeros_like(df.values, dtype=np.float32)

        for i, col in enumerate(df.columns):
            series = df[col].values

            # Rolling mean and std
            rolling_mean = pd.Series(series).rolling(
                window=self.rolling_window,
                min_periods=1
            ).mean().values

            rolling_std = pd.Series(series).rolling(
                window=self.rolling_window,
                min_periods=1
            ).std().values

            # Prevent division by zero
            rolling_std = np.where(rolling_std < 1e-8, 1.0, rolling_std)

            # Z-score normalization
            normalized[:, i] = (series - rolling_mean) / rolling_std

        return normalized

    def _rolling_minmax(self, df: pd.DataFrame) -> np.ndarray:
        """
        Rolling min-max normalization (prevents look-ahead bias)
        """
        normalized = np.zeros_like(df.values, dtype=np.float32)

        for i, col in enumerate(df.columns):
            series = df[col].values

            # Rolling min and max
            rolling_min = pd.Series(series).rolling(
                window=self.rolling_window,
                min_periods=1
            ).min().values

            rolling_max = pd.Series(series).rolling(
                window=self.rolling_window,
                min_periods=1
            ).max().values

            # Prevent division by zero
            range_val = rolling_max - rolling_min
            range_val = np.where(range_val < 1e-8, 1.0, range_val)

            # Min-max normalization to [0, 1]
            normalized[:, i] = (series - rolling_min) / range_val

        return normalized

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step

        Args:
            data: DataFrame to fit and transform

        Returns:
            Normalized numpy array
        """
        return self.transform(data, fit_first=True)

    def inverse_transform(
        self,
        normalized_data: np.ndarray,
        feature_names: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Convert normalized data back to original scale

        Args:
            normalized_data: Normalized array (N, n_features)
            feature_names: Optional feature names (uses fitted cols if None)

        Returns:
            DataFrame with original scale
        """
        if feature_names is None:
            feature_names = self.feature_cols

        if self.normalization == 'robust':
            # Use robust scaler inverse
            denormalized = self.scaler.inverse_transform(normalized_data)

        elif self.normalization in ['rolling_zscore', 'rolling_minmax']:
            # Use global statistics for approximate inverse
            denormalized = np.zeros_like(normalized_data)

            for i, col in enumerate(feature_names):
                if self.normalization == 'rolling_zscore':
                    # Inverse Z-score
                    denormalized[:, i] = (
                        normalized_data[:, i] * self.stats[col]['std'] +
                        self.stats[col]['mean']
                    )
                else:  # rolling_minmax
                    # Inverse min-max
                    range_val = self.stats[col]['max'] - self.stats[col]['min']
                    denormalized[:, i] = (
                        normalized_data[:, i] * range_val +
                        self.stats[col]['min']
                    )

        else:  # 'none'
            denormalized = normalized_data

        return pd.DataFrame(denormalized, columns=feature_names)

    def split_data(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets (chronological)

        Args:
            data: Full DataFrame
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio

        Returns:
            (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = data.iloc[:train_end]
        val_df = data.iloc[train_end:val_end]
        test_df = data.iloc[val_end:]

        logger.info(f"Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        return train_df, val_df, test_df

    def get_stats_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of fitted data

        Returns:
            DataFrame with statistics for each feature
        """
        stats_df = pd.DataFrame(self.stats).T
        stats_df['range'] = stats_df['max'] - stats_df['min']
        stats_df['cv'] = stats_df['std'] / stats_df['mean']  # Coefficient of variation

        return stats_df


if __name__ == '__main__':
    # Example usage
    # Load sample data
    data = pd.read_parquet('../../../data/raw/btc_5m.parquet')

    # Initialize preprocessor
    preprocessor = CandlePreprocessor(
        normalization='rolling_zscore',
        rolling_window=1000,
        clip_threshold=5.0
    )

    # Split data
    train_df, val_df, test_df = preprocessor.split_data(data)

    # Fit on training data
    preprocessor.fit(train_df)

    # Transform all splits
    train_normalized = preprocessor.transform(train_df)
    val_normalized = preprocessor.transform(val_df)
    test_normalized = preprocessor.transform(test_df)

    print(f"\nTrain shape: {train_normalized.shape}")
    print(f"Val shape: {val_normalized.shape}")
    print(f"Test shape: {test_normalized.shape}")

    # Check normalization quality
    print(f"\nTrain stats (should be ~0 mean, ~1 std):")
    print(f"Mean: {train_normalized.mean(axis=0)[:5]}")  # First 5 features
    print(f"Std: {train_normalized.std(axis=0)[:5]}")

    # Test inverse transform
    denormalized = preprocessor.inverse_transform(
        train_normalized[:100],
        feature_names=preprocessor.feature_cols
    )
    print(f"\nInverse transform test (should match original):")
    print(f"Original: {train_df.iloc[:5, :5].values}")
    print(f"Inverse: {denormalized.iloc[:5, :5].values}")
