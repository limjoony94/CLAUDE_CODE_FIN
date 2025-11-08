"""
Experiment Progress Tracker

Monitors and displays progress of all running and completed experiments.

Usage:
    python experiments/track_progress.py
    python experiments/track_progress.py --watch  # Continuous monitoring
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from loguru import logger
import pandas as pd


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Track experiment progress')

    parser.add_argument(
        '--watch',
        action='store_true',
        help='Continuous monitoring mode'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Update interval in seconds (default: 60)'
    )

    return parser.parse_args()


def find_experiment_dirs():
    """Find all experiment directories"""
    results_dir = Path('results/experiments')
    if not results_dir.exists():
        return []

    return sorted(results_dir.glob('*_*'))  # Match timestamp pattern


def get_experiment_status(exp_dir):
    """Get status of a single experiment"""
    status = {
        'name': exp_dir.name,
        'start_time': datetime.fromtimestamp(exp_dir.stat().st_ctime),
        'status': 'unknown',
        'progress': 0.0,
        'current_epoch': 0,
        'total_epochs': 0,
        'best_val_loss': None,
        'elapsed_time': None,
        'eta': None
    }

    # Check for completion markers
    if (exp_dir / 'backtest_results.json').exists():
        status['status'] = 'completed'
        status['progress'] = 100.0

        # Load final results
        with open(exp_dir / 'backtest_results.json') as f:
            results = json.load(f)
            status['sharpe_ratio'] = results.get('sharpe_ratio')
            status['total_return'] = results.get('total_return')

    elif (exp_dir / 'train_history.json').exists():
        # Training completed but backtest not yet run
        status['status'] = 'backtesting'
        status['progress'] = 90.0

        with open(exp_dir / 'train_history.json') as f:
            history = json.load(f)
            status['current_epoch'] = len(history.get('train_loss', []))
            status['best_val_loss'] = min(history.get('val_loss', [float('inf')]))

    elif (exp_dir / 'logs').exists():
        # Training in progress
        status['status'] = 'training'

        # Parse latest log file
        log_files = sorted((exp_dir / 'logs').glob('*.log'))
        if log_files:
            log_file = log_files[-1]
            try:
                with open(log_file) as f:
                    lines = f.readlines()

                # Look for epoch progress
                for line in reversed(lines):
                    if 'Epoch' in line and '/' in line:
                        # Parse: "Epoch 25/100: ..."
                        parts = line.split('Epoch')[1].split(':')[0].strip()
                        current, total = parts.split('/')
                        status['current_epoch'] = int(current)
                        status['total_epochs'] = int(total)
                        status['progress'] = (int(current) / int(total)) * 100
                        break

            except Exception as e:
                logger.debug(f"Error parsing log: {e}")

    else:
        status['status'] = 'preparing'
        status['progress'] = 5.0

    # Calculate elapsed time
    elapsed = datetime.now() - status['start_time']
    status['elapsed_time'] = str(elapsed).split('.')[0]  # Remove microseconds

    # Estimate ETA
    if status['current_epoch'] > 0 and status['total_epochs'] > 0:
        time_per_epoch = elapsed.total_seconds() / status['current_epoch']
        remaining_epochs = status['total_epochs'] - status['current_epoch']
        eta_seconds = time_per_epoch * remaining_epochs
        status['eta'] = str(pd.Timedelta(seconds=eta_seconds)).split('.')[0]

    return status


def display_progress(experiments):
    """Display progress of all experiments"""
    print("\n" + "=" * 120)
    print(f"EXPERIMENT PROGRESS TRACKER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)

    if not experiments:
        print("\n⚠️  No experiments found in results/experiments/")
        print("   Run: python main.py --config configs/baseline_config.yaml")
        return

    # Group by status
    by_status = {
        'completed': [],
        'backtesting': [],
        'training': [],
        'preparing': [],
        'unknown': []
    }

    for exp in experiments:
        status = get_experiment_status(exp)
        by_status[status['status']].append(status)

    # Display completed experiments
    if by_status['completed']:
        print(f"\n✅ COMPLETED ({len(by_status['completed'])})")
        print("-" * 120)
        for exp in by_status['completed']:
            sharpe = exp.get('sharpe_ratio', 0)
            ret = exp.get('total_return', 0)
            print(f"  {exp['name'][:50]:50s} | Sharpe: {sharpe:6.2f} | Return: {ret*100:6.2f}% | Time: {exp['elapsed_time']}")

    # Display training experiments
    if by_status['training']:
        print(f"\n🔄 TRAINING ({len(by_status['training'])})")
        print("-" * 120)
        for exp in by_status['training']:
            progress_bar = "█" * int(exp['progress'] / 5) + "░" * (20 - int(exp['progress'] / 5))
            epoch_info = f"Epoch {exp['current_epoch']}/{exp['total_epochs']}"
            eta = f"ETA: {exp['eta']}" if exp['eta'] else ""
            print(f"  {exp['name'][:40]:40s} | [{progress_bar}] {exp['progress']:5.1f}% | {epoch_info:15s} | {eta}")

    # Display backtesting experiments
    if by_status['backtesting']:
        print(f"\n📊 BACKTESTING ({len(by_status['backtesting'])})")
        print("-" * 120)
        for exp in by_status['backtesting']:
            val_loss = exp.get('best_val_loss', 0)
            print(f"  {exp['name'][:50]:50s} | Best Val Loss: {val_loss:.4f} | Epochs: {exp['current_epoch']}")

    # Display preparing experiments
    if by_status['preparing']:
        print(f"\n⏳ PREPARING ({len(by_status['preparing'])})")
        print("-" * 120)
        for exp in by_status['preparing']:
            print(f"  {exp['name'][:50]:50s} | Started: {exp['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")

    # Summary
    print("\n" + "=" * 120)
    print(f"SUMMARY: {len(by_status['completed'])} completed, {len(by_status['training'])} training, "
          f"{len(by_status['backtesting'])} backtesting, {len(by_status['preparing'])} preparing")
    print("=" * 120 + "\n")


def main():
    """Main progress tracking loop"""
    args = parse_args()

    try:
        while True:
            experiments = find_experiment_dirs()
            display_progress(experiments)

            if not args.watch:
                break

            # Wait before next update
            print(f"Refreshing in {args.interval} seconds... (Press Ctrl+C to stop)")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")


if __name__ == '__main__':
    main()
