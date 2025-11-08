"""
Statistical Significance Testing for Ablation Study

Performs bootstrap hypothesis testing to validate whether random masking
performance improvements are statistically significant.

Requirements:
- p-value < 0.05 for statistical significance
- Tests null hypothesis: "Random masking does NOT improve performance"

Usage:
    # Test latest ablation results
    python experiments/statistical_test.py

    # Test specific results
    python experiments/statistical_test.py --results results/ablation/20250108_120000/ablation_comparison.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from scipy import stats
import json


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Statistical significance testing')

    parser.add_argument(
        '--results',
        type=str,
        default=None,
        help='Path to ablation_comparison.csv (if None, finds latest)'
    )

    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=1000,
        help='Number of bootstrap samples'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=0.95,
        help='Confidence level (default: 0.95 for p < 0.05)'
    )

    return parser.parse_args()


def find_latest_results():
    """Find latest ablation study results"""
    results_dir = Path('results/ablation')
    if not results_dir.exists():
        return None

    # Find all ablation directories
    ablation_dirs = sorted(results_dir.glob('*/ablation_comparison.csv'))
    if not ablation_dirs:
        return None

    return ablation_dirs[-1]


def load_equity_curves(results_df):
    """Load equity curves for each experiment"""
    equity_curves = {}

    for _, row in results_df.iterrows():
        config = row['config']
        output_dir = Path(row['output_dir'])

        equity_path = output_dir / 'equity_curve.csv'
        if equity_path.exists():
            equity = pd.read_csv(equity_path, index_col=0)
            equity_curves[config] = equity
        else:
            logger.warning(f"Equity curve not found for {config}")

    return equity_curves


def calculate_returns(equity_curve):
    """Calculate returns from equity curve"""
    returns = equity_curve['balance'].pct_change().dropna()
    return returns.values


def bootstrap_sharpe_difference(returns_a, returns_b, n_bootstrap=1000):
    """
    Bootstrap test for difference in Sharpe ratios

    H0: Sharpe(A) <= Sharpe(B)
    H1: Sharpe(A) > Sharpe(B)

    Returns:
        p_value: Probability of observing difference under null hypothesis
        ci_lower, ci_upper: Confidence interval for Sharpe difference
    """
    def sharpe_ratio(returns):
        if len(returns) == 0 or returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * (252 ** 0.5)

    # Observed difference
    sharpe_a = sharpe_ratio(returns_a)
    sharpe_b = sharpe_ratio(returns_b)
    observed_diff = sharpe_a - sharpe_b

    # Bootstrap samples
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample_a = np.random.choice(returns_a, size=len(returns_a), replace=True)
        sample_b = np.random.choice(returns_b, size=len(returns_b), replace=True)

        # Calculate Sharpe difference
        sharpe_a_boot = sharpe_ratio(sample_a)
        sharpe_b_boot = sharpe_ratio(sample_b)
        bootstrap_diffs.append(sharpe_a_boot - sharpe_b_boot)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Calculate p-value (one-tailed test)
    # Proportion of bootstrap samples where difference <= 0
    p_value = (bootstrap_diffs <= 0).mean()

    # Calculate confidence interval
    alpha = 0.05
    ci_lower = np.percentile(bootstrap_diffs, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)

    return p_value, (ci_lower, ci_upper), observed_diff


def test_all_vs_baseline(equity_curves, baseline_config='baseline_config', n_bootstrap=1000):
    """Test all configurations vs baseline"""
    if baseline_config not in equity_curves:
        logger.error(f"Baseline {baseline_config} not found in results")
        return None

    baseline_returns = calculate_returns(equity_curves[baseline_config])

    results = []

    logger.info(f"\n{'='*70}")
    logger.info("STATISTICAL SIGNIFICANCE TESTING")
    logger.info(f"{'='*70}")
    logger.info(f"\nBaseline: {baseline_config}")
    logger.info(f"Bootstrap samples: {n_bootstrap}")
    logger.info(f"Significance level: α = 0.05")

    for config, equity in equity_curves.items():
        if config == baseline_config:
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: {config} vs {baseline_config}")
        logger.info(f"{'='*70}")

        returns = calculate_returns(equity)

        # Bootstrap test
        p_value, (ci_lower, ci_upper), observed_diff = bootstrap_sharpe_difference(
            returns, baseline_returns, n_bootstrap
        )

        # Determine significance
        is_significant = p_value < 0.05

        logger.info(f"\nResults:")
        logger.info(f"  Observed Sharpe difference: {observed_diff:.4f}")
        logger.info(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Significant at α=0.05: {'✅ YES' if is_significant else '❌ NO'}")

        if is_significant:
            logger.info(f"  ✅ {config} performs SIGNIFICANTLY better than baseline")
        else:
            logger.info(f"  ⚠️  {config} improvement is NOT statistically significant")

        results.append({
            'config': config,
            'sharpe_diff': observed_diff,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': is_significant
        })

    return pd.DataFrame(results)


def run_pairwise_tests(equity_curves, n_bootstrap=1000):
    """Run pairwise tests between all configurations"""
    configs = list(equity_curves.keys())
    n_configs = len(configs)

    logger.info(f"\n{'='*70}")
    logger.info("PAIRWISE COMPARISONS")
    logger.info(f"{'='*70}")

    pairwise_results = []

    for i in range(n_configs):
        for j in range(i + 1, n_configs):
            config_a = configs[i]
            config_b = configs[j]

            returns_a = calculate_returns(equity_curves[config_a])
            returns_b = calculate_returns(equity_curves[config_b])

            p_value, (ci_lower, ci_upper), observed_diff = bootstrap_sharpe_difference(
                returns_a, returns_b, n_bootstrap
            )

            pairwise_results.append({
                'config_a': config_a,
                'config_b': config_b,
                'sharpe_diff': observed_diff,
                'p_value': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': p_value < 0.05
            })

    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_df = pairwise_df.sort_values('p_value')

    logger.info("\n" + pairwise_df.to_string(index=False))

    return pairwise_df


def main():
    """Main statistical testing pipeline"""
    args = parse_args()

    # Find results
    if args.results:
        results_path = Path(args.results)
    else:
        results_path = find_latest_results()

    if not results_path or not results_path.exists():
        logger.error("No ablation results found!")
        logger.error("Run ablation study first: python experiments/run_ablation.py")
        return

    logger.info(f"Loading results from: {results_path}")

    # Load results
    results_df = pd.read_csv(results_path)

    logger.info(f"Found {len(results_df)} experiments:")
    for config in results_df['config']:
        logger.info(f"  - {config}")

    # Load equity curves
    equity_curves = load_equity_curves(results_df)

    if len(equity_curves) < 2:
        logger.error("Need at least 2 experiments for statistical testing")
        return

    # Test vs baseline
    baseline_tests = test_all_vs_baseline(
        equity_curves,
        baseline_config='baseline_config',
        n_bootstrap=args.n_bootstrap
    )

    if baseline_tests is not None:
        # Save baseline test results
        output_dir = results_path.parent
        baseline_tests.to_csv(output_dir / 'statistical_tests_baseline.csv', index=False)
        logger.info(f"\n✅ Baseline tests saved to {output_dir / 'statistical_tests_baseline.csv'}")

    # Pairwise tests
    pairwise_tests = run_pairwise_tests(equity_curves, n_bootstrap=args.n_bootstrap)

    # Save pairwise test results
    output_dir = results_path.parent
    pairwise_tests.to_csv(output_dir / 'statistical_tests_pairwise.csv', index=False)
    logger.info(f"✅ Pairwise tests saved to {output_dir / 'statistical_tests_pairwise.csv'}")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("STATISTICAL TESTING COMPLETE")
    logger.info(f"{'='*70}")

    if baseline_tests is not None:
        significant_configs = baseline_tests[baseline_tests['significant']]

        logger.info(f"\n📊 Configurations with significant improvement over baseline:")
        if len(significant_configs) > 0:
            for _, row in significant_configs.iterrows():
                logger.info(f"  ✅ {row['config']}: Sharpe +{row['sharpe_diff']:.4f}, p={row['p_value']:.4f}")

            # Best configuration
            best = significant_configs.iloc[0]
            logger.info(f"\n🏆 BEST VALIDATED CONFIGURATION: {best['config']}")
            logger.info(f"  Sharpe improvement: +{best['sharpe_diff']:.4f}")
            logger.info(f"  p-value: {best['p_value']:.4f}")
            logger.info(f"  95% CI: [{best['ci_lower']:.4f}, {best['ci_upper']:.4f}]")
        else:
            logger.warning("  ⚠️  No configurations show statistically significant improvement")
            logger.warning("  Consider:")
            logger.warning("    - Collecting more data")
            logger.warning("    - Trying different masking ratios")
            logger.warning("    - Adjusting hyperparameters")

    logger.info(f"\n✅ All results saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
