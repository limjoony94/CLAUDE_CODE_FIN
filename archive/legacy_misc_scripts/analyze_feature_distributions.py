"""
Feature Distribution Analysis: Training vs Production

Purpose: Identify if model predictions differ because feature distributions
have shifted between training and production.

Root Cause Investigation:
- Backtest signal rate: 6.12%
- Production signal rate: 19.4% (3.17x higher!)
- Need to verify if features are calculated consistently
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

logger.add(PROJECT_ROOT / "logs" / "feature_distribution_analysis.log")


def load_training_data():
    """Load training data features and statistics"""
    # Try to load processed training data
    train_paths = [
        DATA_DIR / "btc_5m_features.csv",
        DATA_DIR / "train_data.csv",
        DATA_DIR / "btc_usdt_5m.csv"
    ]

    for path in train_paths:
        if path.exists():
            logger.info(f"Loading training data from {path}")
            df = pd.read_csv(path)
            return df

    raise FileNotFoundError(f"No training data found in {DATA_DIR}")


def load_scaler_stats(direction='LONG'):
    """Load scaler statistics (min/max from training)"""
    scaler_path = MODELS_DIR / f"xgboost_v4_{direction.lower()}_entry_scaler.pkl"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return {
        'min': scaler.data_min_,
        'max': scaler.data_max_,
        'feature_range': scaler.feature_range
    }


def load_feature_columns(direction='LONG'):
    """Load feature column names"""
    features_path = MODELS_DIR / f"xgboost_v4_{direction.lower()}_entry_features.txt"

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    with open(features_path, 'r') as f:
        features = [line.strip() for line in f if line.strip()]

    return features


def get_production_features():
    """
    Get recent production features from live bot

    This requires the bot to be running and logging features.
    For now, we'll use the latest candles from BingX.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from scripts.production.phase4_dynamic_testnet_trading import Phase4DynamicTestnetBot
    from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
    from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
    import ccxt

    logger.info("Fetching live production data from BingX...")

    # Initialize exchange
    exchange = ccxt.bingx({'enableRateLimit': True})

    # Fetch recent candles (last 1000 for feature calculation)
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=1440)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Calculate features (same as training)
    logger.info("Calculating features...")
    df = calculate_features(df)

    # Add advanced features
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)

    # Drop NaN
    df = df.ffill().dropna()

    logger.info(f"Production data: {len(df)} candles with features")

    return df


def compare_distributions(training_df, production_df, feature_columns, scaler_stats):
    """
    Compare feature distributions between training and production

    Metrics:
    1. Mean difference
    2. Std difference
    3. KS test (Kolmogorov-Smirnov) - tests if distributions are different
    4. Outlier rate (% of production values outside training range)
    5. Scaler compatibility (% outside scaler min/max)
    """

    results = []

    for i, feature in enumerate(feature_columns):
        if feature not in training_df.columns or feature not in production_df.columns:
            logger.warning(f"Feature {feature} not found in data, skipping")
            continue

        # Get feature values
        train_values = training_df[feature].dropna().values
        prod_values = production_df[feature].dropna().values

        if len(train_values) == 0 or len(prod_values) == 0:
            logger.warning(f"Feature {feature} has no valid values, skipping")
            continue

        # Calculate statistics
        train_mean = np.mean(train_values)
        train_std = np.std(train_values)
        train_min = np.min(train_values)
        train_max = np.max(train_values)

        prod_mean = np.mean(prod_values)
        prod_std = np.std(prod_values)
        prod_min = np.min(prod_values)
        prod_max = np.max(prod_values)

        # Mean difference (in standard deviations)
        mean_diff_std = (prod_mean - train_mean) / train_std if train_std > 0 else 0

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(train_values, prod_values)

        # Outlier rate (production values outside training range)
        outliers = ((prod_values < train_min) | (prod_values > train_max)).sum()
        outlier_rate = outliers / len(prod_values) if len(prod_values) > 0 else 0

        # Scaler compatibility (outside scaler min/max)
        scaler_min = scaler_stats['min'][i]
        scaler_max = scaler_stats['max'][i]
        scaler_outliers = ((prod_values < scaler_min) | (prod_values > scaler_max)).sum()
        scaler_outlier_rate = scaler_outliers / len(prod_values) if len(prod_values) > 0 else 0

        # Assess severity
        severity = 'OK'
        if abs(mean_diff_std) > 2.0 or ks_pvalue < 0.01 or outlier_rate > 0.1:
            severity = 'WARNING'
        if abs(mean_diff_std) > 3.0 or ks_pvalue < 0.001 or outlier_rate > 0.2:
            severity = 'CRITICAL'

        results.append({
            'feature': feature,
            'train_mean': train_mean,
            'train_std': train_std,
            'train_range': f"[{train_min:.4f}, {train_max:.4f}]",
            'prod_mean': prod_mean,
            'prod_std': prod_std,
            'prod_range': f"[{prod_min:.4f}, {prod_max:.4f}]",
            'mean_diff_std': mean_diff_std,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'outlier_rate': outlier_rate,
            'scaler_outlier_rate': scaler_outlier_rate,
            'severity': severity
        })

    return pd.DataFrame(results)


def generate_report(comparison_df, output_path):
    """Generate analysis report"""

    report = []
    report.append("=" * 100)
    report.append("FEATURE DISTRIBUTION ANALYSIS: Training vs Production")
    report.append("=" * 100)
    report.append("")

    # Summary statistics
    total_features = len(comparison_df)
    ok_features = (comparison_df['severity'] == 'OK').sum()
    warning_features = (comparison_df['severity'] == 'WARNING').sum()
    critical_features = (comparison_df['severity'] == 'CRITICAL').sum()

    report.append("SUMMARY:")
    report.append(f"  Total Features: {total_features}")
    report.append(f"  OK: {ok_features} ({ok_features/total_features*100:.1f}%)")
    report.append(f"  WARNING: {warning_features} ({warning_features/total_features*100:.1f}%)")
    report.append(f"  CRITICAL: {critical_features} ({critical_features/total_features*100:.1f}%)")
    report.append("")

    # Critical features (sorted by severity)
    critical_df = comparison_df[comparison_df['severity'] == 'CRITICAL'].sort_values('mean_diff_std', ascending=False)

    if len(critical_df) > 0:
        report.append("=" * 100)
        report.append("🚨 CRITICAL FEATURES (Distribution Shift Detected)")
        report.append("=" * 100)
        report.append("")

        for idx, row in critical_df.iterrows():
            report.append(f"Feature: {row['feature']}")
            report.append(f"  Training Mean: {row['train_mean']:.4f} ± {row['train_std']:.4f}")
            report.append(f"  Production Mean: {row['prod_mean']:.4f} ± {row['prod_std']:.4f}")
            report.append(f"  Mean Difference: {row['mean_diff_std']:.2f} std deviations")
            report.append(f"  KS Test: statistic={row['ks_statistic']:.4f}, p-value={row['ks_pvalue']:.6f}")
            report.append(f"  Outlier Rate: {row['outlier_rate']*100:.1f}% (outside training range)")
            report.append(f"  Scaler Outliers: {row['scaler_outlier_rate']*100:.1f}% (outside scaler range)")
            report.append("")

    # Warning features
    warning_df = comparison_df[comparison_df['severity'] == 'WARNING'].sort_values('mean_diff_std', ascending=False)

    if len(warning_df) > 0:
        report.append("=" * 100)
        report.append("⚠️  WARNING FEATURES (Moderate Shift)")
        report.append("=" * 100)
        report.append("")

        for idx, row in warning_df.head(10).iterrows():  # Top 10
            report.append(f"Feature: {row['feature']}")
            report.append(f"  Mean Difference: {row['mean_diff_std']:.2f} std deviations")
            report.append(f"  Outlier Rate: {row['outlier_rate']*100:.1f}%")
            report.append("")

    # Root cause analysis
    report.append("=" * 100)
    report.append("ROOT CAUSE ANALYSIS")
    report.append("=" * 100)
    report.append("")

    if critical_features > 0:
        report.append("❌ CRITICAL DISTRIBUTION SHIFT DETECTED")
        report.append("")
        report.append("Possible Causes:")
        report.append("1. Feature Calculation Bug: Production code differs from training")
        report.append("2. Market Regime Shift: Current market significantly different from training period")
        report.append("3. Data Source Change: BingX API returning different values")
        report.append("4. Lookback Window Issue: Production using different window sizes")
        report.append("")
        report.append("Recommended Actions:")
        report.append("1. Verify feature calculation code matches training exactly")
        report.append("2. Check for any recent code changes in feature engineering")
        report.append("3. Compare raw OHLCV data (training vs production)")
        report.append("4. Consider retraining models with recent data")
        report.append("")

    elif warning_features > total_features * 0.3:  # >30% warnings
        report.append("⚠️  MODERATE DISTRIBUTION SHIFT")
        report.append("")
        report.append("Market conditions may have changed since training.")
        report.append("Consider monitoring performance and retraining if issues persist.")
        report.append("")

    else:
        report.append("✅ DISTRIBUTIONS MATCH (Within Acceptable Range)")
        report.append("")
        report.append("Feature distributions are consistent between training and production.")
        report.append("High signal rate is NOT due to feature calculation differences.")
        report.append("")

    # Full comparison table
    report.append("=" * 100)
    report.append("DETAILED COMPARISON TABLE")
    report.append("=" * 100)
    report.append("")

    # Format as table
    report.append(f"{'Feature':<30} {'Mean Diff':<12} {'KS p-value':<12} {'Outlier %':<12} {'Status':<10}")
    report.append("-" * 100)

    for idx, row in comparison_df.iterrows():
        feature_short = row['feature'][:28]
        severity_symbol = {'OK': '✅', 'WARNING': '⚠️ ', 'CRITICAL': '🚨'}[row['severity']]

        report.append(
            f"{feature_short:<30} "
            f"{row['mean_diff_std']:>10.2f}σ "
            f"{row['ks_pvalue']:>11.6f} "
            f"{row['outlier_rate']*100:>10.1f}% "
            f"{severity_symbol} {row['severity']}"
        )

    report.append("=" * 100)

    # Write report
    report_text = "\n".join(report)

    with open(output_path, 'w') as f:
        f.write(report_text)

    logger.success(f"Report saved to {output_path}")

    # Print to console
    print("\n" + report_text)

    return report_text


def main():
    """Run feature distribution analysis"""

    logger.info("Starting Feature Distribution Analysis...")

    try:
        # Load data
        logger.info("Step 1: Loading training data...")
        training_df = load_training_data()

        logger.info("Step 2: Loading production data...")
        production_df = get_production_features()

        logger.info("Step 3: Loading scaler statistics...")
        scaler_stats = load_scaler_stats('LONG')

        logger.info("Step 4: Loading feature columns...")
        feature_columns = load_feature_columns('LONG')

        logger.info(f"Analyzing {len(feature_columns)} features...")

        # Compare distributions
        logger.info("Step 5: Comparing distributions...")
        comparison_df = compare_distributions(
            training_df,
            production_df,
            feature_columns,
            scaler_stats
        )

        # Generate report
        logger.info("Step 6: Generating report...")
        output_path = PROJECT_ROOT / "claudedocs" / "FEATURE_DISTRIBUTION_ANALYSIS_20251016.md"
        generate_report(comparison_df, output_path)

        logger.success("✅ Analysis Complete!")
        logger.info(f"Report: {output_path}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
