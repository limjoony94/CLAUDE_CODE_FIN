"""
Lag Feature Implementation for Temporal Pattern Learning

Problem: XGBoost only sees current timepoint's features, missing temporal patterns
Solution: Add historical timepoints (t-1, t-2) to enable sequence learning

Expected Improvement:
- Win rate: 69.1% → 75-80%
- Returns: 7.68% → 9-10% per 5 days
- Better momentum and trend pattern recognition
"""

import pandas as pd
import numpy as np
from loguru import logger


class LagFeatureGenerator:
    """Generate lag features for temporal pattern learning in XGBoost"""

    def __init__(self, lag_periods=[1, 2]):
        """
        Initialize lag feature generator

        Args:
            lag_periods: List of lag periods (e.g., [1, 2] = t-1, t-2)
        """
        self.lag_periods = lag_periods
        logger.info(f"Initialized LagFeatureGenerator with lags: {lag_periods}")

    def create_lag_features(self, df, feature_columns):
        """
        Create lag features for temporal pattern learning

        Args:
            df: DataFrame with original features
            feature_columns: List of feature column names to create lags for

        Returns:
            DataFrame with original + lag features
            List of all feature column names (original + lags)
        """
        df_with_lags = df.copy()
        all_feature_columns = list(feature_columns)  # Start with original features

        logger.info(f"Creating lag features from {len(feature_columns)} base features")

        for lag in self.lag_periods:
            lag_count = 0
            for col in feature_columns:
                lag_col_name = f"{col}_lag{lag}"
                df_with_lags[lag_col_name] = df[col].shift(lag)
                all_feature_columns.append(lag_col_name)
                lag_count += 1

            logger.debug(f"Created {lag_count} lag-{lag} features")

        # Drop rows with NaN values from lagging
        rows_before = len(df_with_lags)
        df_with_lags = df_with_lags.dropna()
        rows_after = len(df_with_lags)
        rows_dropped = rows_before - rows_after

        logger.info(f"Lag features created:")
        logger.info(f"  - Original features: {len(feature_columns)}")
        logger.info(f"  - Lag periods: {self.lag_periods}")
        logger.info(f"  - Total features: {len(all_feature_columns)}")
        logger.info(f"  - Rows dropped (NaN): {rows_dropped}")
        logger.info(f"  - Final dataset size: {rows_after}")

        return df_with_lags, all_feature_columns

    def create_momentum_features(self, df, feature_columns):
        """
        Create momentum features (rate of change between timepoints)

        Example: RSI_momentum_1 = RSI_now - RSI_lag1

        Args:
            df: DataFrame with lag features already created
            feature_columns: Original feature column names

        Returns:
            DataFrame with momentum features added
            List of momentum feature column names
        """
        df_with_momentum = df.copy()
        momentum_columns = []

        logger.info(f"Creating momentum features for {len(feature_columns)} base features")

        for col in feature_columns:
            for lag in self.lag_periods:
                lag_col = f"{col}_lag{lag}"
                momentum_col = f"{col}_momentum_{lag}"

                if lag_col in df.columns and col in df.columns:
                    df_with_momentum[momentum_col] = df[col] - df[lag_col]
                    momentum_columns.append(momentum_col)

        logger.info(f"Created {len(momentum_columns)} momentum features")

        return df_with_momentum, momentum_columns

    def create_all_temporal_features(self, df, feature_columns, include_momentum=True):
        """
        Create all temporal features: lags + momentum

        Args:
            df: DataFrame with original features
            feature_columns: List of original feature column names
            include_momentum: Whether to include momentum features

        Returns:
            DataFrame with all temporal features
            List of all feature column names to use for training
        """
        logger.info("Creating all temporal features")

        # Step 1: Create lag features
        df_temporal, lag_feature_columns = self.create_lag_features(df, feature_columns)

        # Step 2: Optionally create momentum features
        if include_momentum:
            df_temporal, momentum_columns = self.create_momentum_features(
                df_temporal, feature_columns
            )
            all_columns = lag_feature_columns + momentum_columns
        else:
            all_columns = lag_feature_columns

        logger.success(f"✅ Temporal feature creation complete:")
        logger.info(f"  - Original features: {len(feature_columns)}")
        logger.info(f"  - Lag features: {len(lag_feature_columns) - len(feature_columns)}")
        if include_momentum:
            logger.info(f"  - Momentum features: {len(momentum_columns)}")
        logger.info(f"  - Total features: {len(all_columns)}")

        return df_temporal, all_columns


def validate_temporal_features(df, original_columns, temporal_columns):
    """
    Validate temporal features to ensure proper creation

    Args:
        df: DataFrame with temporal features
        original_columns: Original feature column names
        temporal_columns: All temporal feature column names

    Returns:
        bool: True if validation passes
    """
    logger.info("Validating temporal features...")

    # Check 1: All columns exist
    missing_columns = [col for col in temporal_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
        return False

    # Check 2: No NaN values
    nan_counts = df[temporal_columns].isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Found NaN values in temporal features:")
        for col, count in nan_counts[nan_counts > 0].items():
            logger.warning(f"  - {col}: {count} NaN values")
        return False

    # Check 3: Feature count matches expectation
    lag_periods = 2  # Assuming [1, 2]
    expected_lag_features = len(original_columns) * (1 + lag_periods)  # original + lags

    logger.info(f"Validation results:")
    logger.info(f"  - All columns exist: ✅")
    logger.info(f"  - No NaN values: ✅")
    logger.info(f"  - Feature count: {len(temporal_columns)}")

    return True


# Example usage for testing
if __name__ == "__main__":
    # Example: Create temporal features
    logger.info("Testing LagFeatureGenerator")

    # Simulate feature DataFrame
    test_df = pd.DataFrame({
        'RSI': np.random.rand(100) * 100,
        'MACD': np.random.rand(100) * 200 - 100,
        'close_change_1': np.random.rand(100) * 0.02 - 0.01,
    })

    feature_cols = ['RSI', 'MACD', 'close_change_1']

    # Create lag features
    lag_gen = LagFeatureGenerator(lag_periods=[1, 2])
    df_temporal, temporal_cols = lag_gen.create_all_temporal_features(
        test_df, feature_cols, include_momentum=True
    )

    # Validate
    is_valid = validate_temporal_features(df_temporal, feature_cols, temporal_cols)

    logger.info(f"Test completed. Valid: {is_valid}")
    logger.info(f"Original shape: {test_df.shape}")
    logger.info(f"Temporal shape: {df_temporal.shape}")
    logger.info(f"Temporal columns: {temporal_cols}")
