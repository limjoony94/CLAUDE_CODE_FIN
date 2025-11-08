"""
Quick Ablation Test with Synthetic Data

Tests all 4 configurations with synthetic data to validate pipeline
before running full experiments with real data.

Usage:
    python scripts/quick_ablation_test.py
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
import json

# Add parent directory to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root.parent))

from random_masking.data.preprocessor import CandlePreprocessor
from random_masking.data.masking_strategy import RandomMaskingStrategy
from random_masking.data.dataset import CandleDataset, collate_fn_train
from random_masking.models.predictor import CandlePredictor
from random_masking.training.trainer import Trainer
from random_masking.evaluation.backtester import Backtester, BacktestConfig
from torch.utils.data import DataLoader


def generate_synthetic_data(n_candles=10000, seed=42):
    """Generate realistic synthetic cryptocurrency data"""
    np.random.seed(seed)

    dates = pd.date_range('2022-01-01', periods=n_candles, freq='5min')

    # Realistic price movement with trend + noise
    trend = np.linspace(100, 120, n_candles)
    random_walk = np.cumsum(np.random.randn(n_candles) * 0.5)
    close_prices = trend + random_walk

    # OHLCV data
    data = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(n_candles) * 0.001),
        'high': close_prices * (1 + np.abs(np.random.randn(n_candles)) * 0.003),
        'low': close_prices * (1 - np.abs(np.random.randn(n_candles)) * 0.003),
        'close': close_prices,
        'volume': np.random.rand(n_candles) * 1000
    }, index=dates)

    # Add basic features
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    data['volatility'] = data['returns'].rolling(20).std()
    data['volume_change'] = data['volume'].pct_change()

    return data


def run_quick_experiment(config_name, masking_enabled, masking_ratios, data, device):
    """Run a single quick experiment"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {config_name}")
    logger.info(f"Masking: {masking_ratios if masking_enabled else 'Disabled'}")
    logger.info(f"{'='*70}")

    # Preprocessing
    preprocessor = CandlePreprocessor(normalization='rolling_zscore', rolling_window=1000)
    data_normalized = preprocessor.fit_transform(data)

    # Drop NaN rows from rolling window warmup
    valid_mask = ~np.isnan(data_normalized).any(axis=1)
    data_normalized = data_normalized[valid_mask]
    logger.info(f"After dropping NaN rows: {len(data_normalized)} samples")

    # Split
    train_end = int(len(data_normalized) * 0.7)
    val_end = int(len(data_normalized) * 0.85)

    train_data = data_normalized[:train_end]
    val_data = data_normalized[train_end:val_end]
    test_data = data_normalized[val_end:]

    # Test DataFrame for backtesting
    test_data_df = pd.DataFrame(
        test_data,
        index=data.index[val_end:val_end + len(test_data)],
        columns=data.columns
    )

    # Masking strategy
    masking_strategy = None
    if masking_enabled:
        masking_strategy = RandomMaskingStrategy(
            seq_len=100,
            mask_ratio_infill=masking_ratios[0],
            mask_ratio_forecast=masking_ratios[1],
            mask_ratio_sparse=masking_ratios[2]
        )

    # Datasets
    train_dataset = CandleDataset(
        data=train_data,
        seq_len=100,
        pred_len=10,
        mode='train',
        masking_strategy=masking_strategy
    )

    val_dataset = CandleDataset(
        data=val_data,
        seq_len=100,
        pred_len=10,
        mode='val'
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_train if masking_enabled else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    # Model
    model = CandlePredictor(
        input_dim=data_normalized.shape[1],
        hidden_dim=128,  # Smaller for quick test
        n_layers=4,      # Fewer layers for quick test
        n_heads=4,
        ff_dim=512,
        dropout=0.1,
        use_uncertainty_head=True
    )

    logger.info(f"Model: {model.count_parameters():,} parameters")

    # Validate data has no NaN/Inf
    if np.any(np.isnan(train_data)) or np.any(np.isinf(train_data)):
        logger.error("Training data contains NaN or Inf values!")
        raise ValueError("Invalid training data")

    # Training config - conservative settings for stability
    training_config = {
        'epochs': 3,  # Only 3 epochs for quick test
        'learning_rate': 5e-5,  # Lower LR to prevent NaN
        'weight_decay': 1e-6,   # Reduced weight decay
        'patience': 10,
        'gradient_clip': 0.5,   # Stronger gradient clipping
        'checkpoint_dir': f'results/quick_test/{config_name}/checkpoints',
        'log_dir': f'results/quick_test/{config_name}/logs',
        'use_tensorboard': False
    }

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device
    )

    logger.info("Training started...")
    history = trainer.train()

    logger.info(f"Training complete: Best val loss = {min(history['val_loss']):.4f}")

    # Backtest
    backtest_config = BacktestConfig(
        initial_capital=10000,
        max_position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.06,
        max_hold_candles=48,
        slippage_pct=0.0005,
        fee_pct=0.0004,
        leverage=1.0,
        kelly_fraction=0.25
    )

    backtester = Backtester(model, backtest_config, device=device)
    results = backtester.run(test_data_df, seq_len=100, verbose=False)

    logger.info(f"\nBacktest Results:")
    logger.info(f"  Sharpe Ratio: {results.sharpe_ratio:.4f}")
    logger.info(f"  Total Return: {results.total_return:.2%}")
    logger.info(f"  Max Drawdown: {results.max_drawdown_pct:.2%}")
    logger.info(f"  Win Rate: {results.win_rate:.2%}")
    logger.info(f"  Total Trades: {len(results.trades)}")

    return {
        'config': config_name,
        'masking': f"{masking_ratios[0]:.1f}-{masking_ratios[1]:.1f}-{masking_ratios[2]:.1f}" if masking_enabled else "None",
        'train_loss': history['train_loss'][-1],
        'val_loss': min(history['val_loss']),
        'sharpe_ratio': results.sharpe_ratio,
        'total_return': results.total_return,
        'max_drawdown': results.max_drawdown_pct,
        'win_rate': results.win_rate,
        'num_trades': len(results.trades)
    }


def main():
    """Main quick ablation test"""
    logger.info("="*70)
    logger.info("QUICK ABLATION TEST - Synthetic Data")
    logger.info("="*70)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"\nDevice: {device}")

    # Generate data
    logger.info("\nGenerating synthetic data...")
    data = generate_synthetic_data(n_candles=10000, seed=42)
    logger.info(f"Generated {len(data)} candles")

    # Experiments to run
    experiments = [
        ('baseline', False, (0.0, 0.0, 0.0)),
        ('proposed_40_40_20', True, (0.4, 0.4, 0.2)),
        ('infill_heavy_70_30_0', True, (0.7, 0.3, 0.0)),
        ('forecast_heavy_30_70_0', True, (0.3, 0.7, 0.0))
    ]

    # Run experiments
    results = []
    for config_name, masking_enabled, masking_ratios in experiments:
        result = run_quick_experiment(
            config_name, masking_enabled, masking_ratios, data, device
        )
        results.append(result)

    # Compare results
    logger.info("\n" + "="*70)
    logger.info("RESULTS COMPARISON")
    logger.info("="*70)

    df = pd.DataFrame(results)
    df = df.sort_values('sharpe_ratio', ascending=False)

    print("\n" + df.to_string(index=False))

    # Find best
    best = df.iloc[0]
    baseline = df[df['config'] == 'baseline'].iloc[0]

    improvement = (best['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'] * 100

    logger.info(f"\n🏆 BEST: {best['config']}")
    logger.info(f"   Sharpe: {best['sharpe_ratio']:.4f}")
    logger.info(f"   Return: {best['total_return']:.2%}")
    logger.info(f"   Improvement over baseline: {improvement:+.1f}%")

    # Save results
    output_dir = Path('results/quick_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / 'quick_ablation_results.csv', index=False)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'best_config': best['config'],
        'baseline_sharpe': float(baseline['sharpe_ratio']),
        'best_sharpe': float(best['sharpe_ratio']),
        'improvement_pct': float(improvement),
        'all_results': results
    }

    with open(output_dir / 'quick_ablation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n✅ Results saved to {output_dir}")
    logger.info("="*70)

    if improvement >= 5:
        logger.info("\n✅ Quick test SUCCESSFUL - Random masking shows promise!")
        logger.info("   Proceed with full experiments on real data")
    else:
        logger.info("\n⚠️  Quick test shows marginal improvement")
        logger.info("   May need to adjust hyperparameters or masking ratios")


if __name__ == '__main__':
    main()
