"""
Ablation Study Runner

Automatically runs all ablation experiments and compares results:
1. Baseline (pure forecasting, no masking)
2. Proposed (40-40-20 random masking curriculum)
3. Variant A (70-30-0 infill-heavy)
4. Variant B (30-70-0 forecast-heavy)

Usage:
    python experiments/run_ablation.py --data-path data/raw/crypto_candles.parquet
    python experiments/run_ablation.py  # Will collect data automatically
"""

import argparse
import subprocess
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
import json


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run complete ablation study')

    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to pre-collected data (if None, will collect)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (cuda/cpu/auto)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/ablation',
        help='Directory to save ablation results'
    )

    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline training (if already completed)'
    )

    return parser.parse_args()


def run_experiment(config_path, data_path, device, output_dir):
    """Run a single experiment"""
    config_name = Path(config_path).stem
    logger.info(f"\n{'='*70}")
    logger.info(f"Running experiment: {config_name}")
    logger.info(f"{'='*70}")

    # Build command
    cmd = [
        'python', 'main.py',
        '--config', config_path,
        '--device', device,
        '--output-dir', output_dir
    ]

    if data_path:
        cmd.extend(['--data-path', data_path])

    # Run experiment
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Experiment {config_name} failed!")
        logger.error(result.stderr)
        return None

    logger.info(f"✅ Experiment {config_name} completed")

    # Find latest results directory
    output_path = Path(output_dir)
    experiment_dirs = sorted(output_path.glob(f"*{config_name}*"))
    if not experiment_dirs:
        logger.error(f"Could not find results for {config_name}")
        return None

    latest_dir = experiment_dirs[-1]

    # Load results
    results = {
        'config': config_name,
        'output_dir': str(latest_dir)
    }

    # Load metrics from trades.csv
    trades_path = latest_dir / 'trades.csv'
    if trades_path.exists():
        trades = pd.read_csv(trades_path)
        results['num_trades'] = len(trades)
        if len(trades) > 0:
            results['win_rate'] = (trades['pnl'] > 0).mean()
            results['avg_pnl'] = trades['pnl'].mean()
            results['total_pnl'] = trades['pnl'].sum()

    # Load equity curve
    equity_path = latest_dir / 'equity_curve.csv'
    if equity_path.exists():
        equity = pd.read_csv(equity_path, index_col=0)
        results['final_balance'] = equity['balance'].iloc[-1]
        results['total_return'] = (equity['balance'].iloc[-1] / equity['balance'].iloc[0] - 1)

        # Calculate Sharpe ratio
        returns = equity['balance'].pct_change().dropna()
        results['sharpe_ratio'] = returns.mean() / returns.std() * (252 ** 0.5) if len(returns) > 0 else 0

        # Calculate max drawdown
        cummax = equity['balance'].cummax()
        drawdown = (equity['balance'] - cummax) / cummax
        results['max_drawdown'] = drawdown.min()

    logger.info(f"Results for {config_name}:")
    logger.info(f"  Total Return: {results.get('total_return', 0):.2%}")
    logger.info(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  Win Rate: {results.get('win_rate', 0):.2%}")
    logger.info(f"  Max Drawdown: {results.get('max_drawdown', 0):.2%}")

    return results


def compare_results(all_results, output_dir):
    """Compare all experiment results"""
    logger.info(f"\n{'='*70}")
    logger.info("ABLATION STUDY RESULTS COMPARISON")
    logger.info(f"{'='*70}")

    # Create comparison DataFrame
    comparison = pd.DataFrame(all_results)

    # Sort by Sharpe ratio
    comparison = comparison.sort_values('sharpe_ratio', ascending=False)

    # Print table
    logger.info("\n" + comparison.to_string(index=False))

    # Save comparison
    output_path = Path(output_dir)
    comparison.to_csv(output_path / 'ablation_comparison.csv', index=False)
    comparison.to_json(output_path / 'ablation_comparison.json', orient='records', indent=2)

    # Find best configuration
    best = comparison.iloc[0]
    baseline = comparison[comparison['config'] == 'baseline_config'].iloc[0] if 'baseline_config' in comparison['config'].values else None

    logger.info(f"\n🏆 BEST CONFIGURATION: {best['config']}")
    logger.info(f"  Sharpe Ratio: {best['sharpe_ratio']:.2f}")
    logger.info(f"  Total Return: {best['total_return']:.2%}")
    logger.info(f"  Win Rate: {best['win_rate']:.2%}")

    if baseline is not None:
        improvement = (best['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100
        logger.info(f"\n📊 Improvement over baseline: {improvement:.1f}%")

        if improvement >= 10:
            logger.info("✅ SUCCESS: Random masking improves performance by >10%!")
        else:
            logger.info("⚠️  Random masking does not meet 10% improvement threshold")

    # Generate summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'best_config': best['config'],
        'best_sharpe': float(best['sharpe_ratio']),
        'baseline_sharpe': float(baseline['sharpe_ratio']) if baseline is not None else None,
        'improvement_pct': float(improvement) if baseline is not None else None,
        'all_results': all_results
    }

    with open(output_path / 'ablation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n✅ Comparison saved to {output_path}")

    return comparison, summary


def main():
    """Main ablation study pipeline"""
    args = parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger.add(output_dir / 'ablation_study.log')

    logger.info("=" * 70)
    logger.info("RANDOM MASKING ABLATION STUDY")
    logger.info("=" * 70)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Data path: {args.data_path or 'Will collect'}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Skip baseline: {args.skip_baseline}")

    # Define experiments to run
    experiments = [
        'configs/baseline_config.yaml',
        'configs/proposed_config.yaml',
        'configs/variant_infill_heavy.yaml',
        'configs/variant_forecast_heavy.yaml'
    ]

    if args.skip_baseline:
        experiments = experiments[1:]  # Skip baseline
        logger.info("\n⚠️  Skipping baseline (assuming already completed)")

    # Run all experiments
    all_results = []

    for config_path in experiments:
        if not Path(config_path).exists():
            logger.warning(f"Config {config_path} not found, skipping")
            continue

        results = run_experiment(
            config_path=config_path,
            data_path=args.data_path,
            device=args.device,
            output_dir=str(output_dir / 'experiments')
        )

        if results:
            all_results.append(results)

    # Compare results
    if len(all_results) > 1:
        comparison, summary = compare_results(all_results, output_dir)

        logger.info("\n" + "=" * 70)
        logger.info("ABLATION STUDY COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\n✅ Ran {len(all_results)} experiments")
        logger.info(f"✅ Results saved to: {output_dir}")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Review results: {output_dir / 'ablation_comparison.csv'}")
        logger.info(f"  2. Check visualizations in experiment directories")
        logger.info(f"  3. Run statistical tests: python experiments/statistical_test.py")
        logger.info("=" * 70)
    else:
        logger.error("Not enough experiments completed for comparison")


if __name__ == '__main__':
    main()
