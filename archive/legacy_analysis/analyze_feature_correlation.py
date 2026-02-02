"""
Feature Correlation Analysis for Production Models

Purpose:
1. Calculate correlation matrix for all production features
2. Detect redundant features (correlation > 0.8)
3. Identify feature groups with high correlation
4. Generate recommendations for feature reduction

Author: Claude Code
Date: 2025-10-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "claudedocs"

# Production model feature lists
LONG_ENTRY_FEATURES = """close_change_1
close_change_3
volume_ma_ratio
rsi
macd
macd_signal
macd_diff
bb_high
bb_mid
bb_low
distance_to_support_pct
distance_to_resistance_pct
num_support_touches
num_resistance_touches
upper_trendline_slope
lower_trendline_slope
price_vs_upper_trendline_pct
price_vs_lower_trendline_pct
rsi_bullish_divergence
rsi_bearish_divergence
macd_bullish_divergence
macd_bearish_divergence
double_top
double_bottom
higher_highs_lows
lower_highs_lows
volume_ma_ratio
volume_price_correlation
price_volume_trend
body_to_range_ratio
upper_shadow_ratio
lower_shadow_ratio
bullish_engulfing
bearish_engulfing
hammer
shooting_star
doji
distance_from_recent_high_pct
bearish_candle_count
red_candle_volume_ratio
strong_selling_pressure
price_momentum_near_resistance
rsi_from_recent_peak
consecutive_up_candles""".split('\n')

SHORT_ENTRY_FEATURES = """rsi_deviation
rsi_direction
rsi_extreme
macd_strength
macd_direction
macd_divergence_abs
price_distance_ma20
price_direction_ma20
price_distance_ma50
price_direction_ma50
volatility
atr_pct
atr
negative_momentum
negative_acceleration
down_candle_ratio
down_candle_body
lower_low_streak
resistance_rejection_count
bearish_divergence
volume_decline_ratio
distribution_signal
down_candle
lower_low
near_resistance
rejection_from_resistance
volume_on_decline
volume_on_advance
bear_market_strength
trend_strength
downtrend_confirmed
volatility_asymmetry
below_support
support_breakdown
panic_selling
downside_volatility
upside_volatility
ema_12""".split('\n')

EXIT_FEATURES = """rsi
macd
macd_signal
atr
ema_12
trend_strength
volatility_regime
volume_surge
price_acceleration
volume_ratio
price_vs_ma20
price_vs_ma50
volatility_20
rsi_slope
rsi_overbought
rsi_oversold
rsi_divergence
macd_histogram_slope
macd_crossover
macd_crossunder
higher_high
lower_low
near_resistance
near_support
bb_position""".split('\n')


def load_and_prepare_data(n_candles=10000):
    """Load historical data and calculate all features"""
    print("="*80)
    print("Loading and Preparing Data")
    print("="*80)

    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    print(f"\nLoading: {data_file}")

    df = pd.read_csv(data_file)
    print(f"Original data: {len(df)} rows")

    # Use last N candles for analysis
    df = df.tail(n_candles).copy()
    print(f"Using last {n_candles} candles")

    # Calculate all features
    print("\nCalculating all features...")
    df = calculate_all_features(df)

    # Clean NaN
    df = df.dropna()
    print(f"After cleaning: {len(df)} rows")

    return df


def analyze_correlation(df, feature_list, model_name, threshold=0.8):
    """
    Analyze feature correlation and detect redundancy

    Args:
        df: DataFrame with all features
        feature_list: List of features to analyze
        model_name: Name of the model (for reporting)
        threshold: Correlation threshold for redundancy detection

    Returns:
        correlation_matrix, redundant_pairs, report
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {model_name} Features")
    print(f"{'='*80}")

    # Filter features that exist in df
    available_features = [f for f in feature_list if f in df.columns]
    missing_features = [f for f in feature_list if f not in df.columns]

    if missing_features:
        print(f"\n⚠️  Missing features ({len(missing_features)}):")
        for f in missing_features:
            print(f"   - {f}")

    print(f"\nAnalyzing {len(available_features)} features")

    # Calculate correlation matrix
    corr_matrix = df[available_features].corr()

    # Find high correlations (excluding diagonal)
    high_corr_pairs = []

    for i in range(len(available_features)):
        for j in range(i+1, len(available_features)):
            corr_value = corr_matrix.iloc[i, j]

            if abs(corr_value) >= threshold:
                high_corr_pairs.append({
                    'feature_1': available_features[i],
                    'feature_2': available_features[j],
                    'correlation': corr_value,
                    'abs_correlation': abs(corr_value)
                })

    # Sort by absolute correlation
    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x['abs_correlation'], reverse=True)

    # Generate report
    report = []
    report.append(f"\n{'='*80}")
    report.append(f"{model_name} - Correlation Analysis Report")
    report.append(f"{'='*80}")
    report.append(f"\nTotal Features: {len(available_features)}")
    report.append(f"Missing Features: {len(missing_features)}")
    report.append(f"Correlation Threshold: {threshold}")
    report.append(f"High Correlation Pairs Found: {len(high_corr_pairs)}")

    if high_corr_pairs:
        report.append(f"\n{'='*80}")
        report.append("High Correlation Pairs (Redundancy Detected)")
        report.append(f"{'='*80}")

        for idx, pair in enumerate(high_corr_pairs, 1):
            report.append(f"\n{idx}. Correlation: {pair['correlation']:.4f}")
            report.append(f"   Feature 1: {pair['feature_1']}")
            report.append(f"   Feature 2: {pair['feature_2']}")

            # Recommendation
            if abs(pair['correlation']) > 0.95:
                report.append(f"   ⚠️  SEVERE REDUNDANCY (>0.95) - Consider removing one")
            elif abs(pair['correlation']) > 0.9:
                report.append(f"   ⚠️  HIGH REDUNDANCY (>0.9) - Likely redundant")
            else:
                report.append(f"   ⚠️  MODERATE REDUNDANCY (>0.8) - Review needed")
    else:
        report.append("\n✅ No high correlation pairs found - Features are independent")

    # Feature groups (cluster features with high mutual correlation)
    if len(high_corr_pairs) > 0:
        report.append(f"\n{'='*80}")
        report.append("Potential Feature Groups (High Mutual Correlation)")
        report.append(f"{'='*80}")

        # Simple clustering: features appearing in multiple pairs
        feature_freq = {}
        for pair in high_corr_pairs:
            f1, f2 = pair['feature_1'], pair['feature_2']
            feature_freq[f1] = feature_freq.get(f1, 0) + 1
            feature_freq[f2] = feature_freq.get(f2, 0) + 1

        # Features with 3+ connections
        hub_features = {f: count for f, count in feature_freq.items() if count >= 3}

        if hub_features:
            report.append("\nHub Features (3+ high correlations):")
            for f, count in sorted(hub_features.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  - {f}: {count} connections")
                # Show connected features
                connected = set()
                for pair in high_corr_pairs:
                    if pair['feature_1'] == f:
                        connected.add(pair['feature_2'])
                    elif pair['feature_2'] == f:
                        connected.add(pair['feature_1'])
                report.append(f"    Connected to: {', '.join(list(connected)[:5])}")
        else:
            report.append("\nNo hub features found (all features have <3 connections)")

    # Recommendations
    report.append(f"\n{'='*80}")
    report.append("Recommendations")
    report.append(f"{'='*80}")

    if len(high_corr_pairs) == 0:
        report.append("\n✅ Current feature set is well-designed with minimal redundancy")
        report.append("✅ No immediate action needed")
    else:
        report.append(f"\n⚠️  Found {len(high_corr_pairs)} redundant pairs")

        # Count by severity
        severe = sum(1 for p in high_corr_pairs if p['abs_correlation'] > 0.95)
        high = sum(1 for p in high_corr_pairs if 0.9 < p['abs_correlation'] <= 0.95)
        moderate = sum(1 for p in high_corr_pairs if 0.8 < p['abs_correlation'] <= 0.9)

        report.append(f"   - Severe (>0.95): {severe} pairs")
        report.append(f"   - High (0.9-0.95): {high} pairs")
        report.append(f"   - Moderate (0.8-0.9): {moderate} pairs")

        report.append("\nRecommended Actions:")
        report.append("1. Review severe redundancy pairs first")
        report.append("2. Consider removing one feature from each pair")
        report.append("3. Prioritize keeping features with:")
        report.append("   - Higher importance scores")
        report.append("   - Better interpretability")
        report.append("   - Fewer connections to other features")

        potential_reduction = len(high_corr_pairs)
        current_count = len(available_features)
        report.append(f"\nPotential feature reduction: {current_count} → {current_count - potential_reduction}")
        report.append(f"Reduction: -{potential_reduction} features ({potential_reduction/current_count*100:.1f}%)")

    return corr_matrix, high_corr_pairs, report


def visualize_correlation(corr_matrix, high_corr_pairs, model_name, output_dir):
    """
    Create correlation heatmap visualizations

    Args:
        corr_matrix: Correlation matrix
        high_corr_pairs: List of high correlation pairs
        model_name: Model name for title
        output_dir: Directory to save plots
    """
    print(f"\nGenerating visualizations for {model_name}...")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Full correlation matrix
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=axes[0]
    )
    axes[0].set_title(f'{model_name} - Full Correlation Matrix', fontsize=14, fontweight='bold')

    # Plot 2: Highlight high correlations
    mask = np.abs(corr_matrix) < 0.8
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap='Reds',
        vmin=0.8,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=axes[1]
    )
    axes[1].set_title(f'{model_name} - High Correlations (>0.8)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    output_file = output_dir / f"correlation_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")

    plt.close()

    # Create distribution plot of correlations
    if len(high_corr_pairs) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        corr_values = [p['abs_correlation'] for p in high_corr_pairs]

        ax.hist(corr_values, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(x=0.8, color='orange', linestyle='--', label='Threshold (0.8)')
        ax.axvline(x=0.9, color='red', linestyle='--', label='High (0.9)')
        ax.axvline(x=0.95, color='darkred', linestyle='--', label='Severe (0.95)')

        ax.set_xlabel('Absolute Correlation', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{model_name} - Distribution of High Correlations', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Save
        output_file = output_dir / f"correlation_distribution_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")

        plt.close()


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("FEATURE CORRELATION ANALYSIS - PRODUCTION MODELS")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_and_prepare_data(n_candles=10000)

    # Analyze each model
    results = {}

    # 1. LONG Entry Model
    corr_long, pairs_long, report_long = analyze_correlation(
        df, LONG_ENTRY_FEATURES, "LONG Entry Model", threshold=0.8
    )
    results['long_entry'] = {
        'correlation': corr_long,
        'pairs': pairs_long,
        'report': report_long
    }

    # Print report
    print("\n".join(report_long))

    # Visualize
    visualize_correlation(corr_long, pairs_long, "LONG Entry Model", OUTPUT_DIR)

    # 2. SHORT Entry Model
    corr_short, pairs_short, report_short = analyze_correlation(
        df, SHORT_ENTRY_FEATURES, "SHORT Entry Model", threshold=0.8
    )
    results['short_entry'] = {
        'correlation': corr_short,
        'pairs': pairs_short,
        'report': report_short
    }

    # Print report
    print("\n".join(report_short))

    # Visualize
    visualize_correlation(corr_short, pairs_short, "SHORT Entry Model", OUTPUT_DIR)

    # 3. Exit Model
    corr_exit, pairs_exit, report_exit = analyze_correlation(
        df, EXIT_FEATURES, "Exit Model", threshold=0.8
    )
    results['exit'] = {
        'correlation': corr_exit,
        'pairs': pairs_exit,
        'report': report_exit
    }

    # Print report
    print("\n".join(report_exit))

    # Visualize
    visualize_correlation(corr_exit, pairs_exit, "Exit Model", OUTPUT_DIR)

    # Combined summary
    print("\n" + "="*80)
    print("COMBINED SUMMARY - ALL MODELS")
    print("="*80)

    total_features = len(LONG_ENTRY_FEATURES) + len(SHORT_ENTRY_FEATURES) + len(EXIT_FEATURES)
    total_pairs = len(pairs_long) + len(pairs_short) + len(pairs_exit)

    print(f"\nTotal Features Analyzed: {total_features}")
    print(f"  - LONG Entry: {len(LONG_ENTRY_FEATURES)} features")
    print(f"  - SHORT Entry: {len(SHORT_ENTRY_FEATURES)} features")
    print(f"  - Exit: {len(EXIT_FEATURES)} features")

    print(f"\nTotal High Correlation Pairs: {total_pairs}")
    print(f"  - LONG Entry: {len(pairs_long)} pairs")
    print(f"  - SHORT Entry: {len(pairs_short)} pairs")
    print(f"  - Exit: {len(pairs_exit)} pairs")

    if total_pairs > 0:
        print("\n⚠️  Redundancy detected across models")
        print(f"Potential feature reduction: {total_pairs} features")
        print(f"Reduction percentage: {total_pairs/total_features*100:.1f}%")
    else:
        print("\n✅ All models have well-designed feature sets")
        print("✅ No significant redundancy detected")

    # Save combined report
    report_file = OUTPUT_DIR / "FEATURE_CORRELATION_ANALYSIS_20251023.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Feature Correlation Analysis - Production Models\n")
        f.write(f"**Date**: 2025-10-23\n")
        f.write(f"**Status**: ✅ Analysis Complete\n\n")
        f.write("---\n\n")

        # Write each model's report
        f.write("## LONG Entry Model\n\n")
        f.write("\n".join(report_long))
        f.write("\n\n")

        f.write("## SHORT Entry Model\n\n")
        f.write("\n".join(report_short))
        f.write("\n\n")

        f.write("## Exit Model\n\n")
        f.write("\n".join(report_exit))
        f.write("\n\n")

        # Combined summary
        f.write("## Combined Summary\n\n")
        f.write(f"Total Features: {total_features}\n")
        f.write(f"Total High Correlation Pairs: {total_pairs}\n")
        f.write(f"Potential Reduction: {total_pairs} features ({total_pairs/total_features*100:.1f}%)\n\n")

        # Visualizations
        f.write("## Visualizations\n\n")
        f.write("- `correlation_matrix_long_entry_model.png`\n")
        f.write("- `correlation_matrix_short_entry_model.png`\n")
        f.write("- `correlation_matrix_exit_model.png`\n")
        f.write("- `correlation_distribution_*.png` (if redundancy found)\n")

    print(f"\n✅ Full report saved: {report_file}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nNext Steps:")
    print("1. Review correlation matrices and heatmaps")
    print("2. Examine high correlation pairs")
    print("3. Decide which features to remove")
    print("4. Retrain models with reduced feature set")
    print("5. Validate performance improvement")


if __name__ == "__main__":
    main()
