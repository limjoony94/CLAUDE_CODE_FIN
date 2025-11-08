"""
Main Training Script for Random Masking Candle Predictor

This script trains models with configuration from YAML files and supports:
- Baseline (pure forecasting) training
- Random masking curriculum training
- Variant configurations for ablation studies

Usage:
    # Baseline training
    python main.py --config configs/baseline_config.yaml

    # Proposed random masking training
    python main.py --config configs/proposed_config.yaml

    # Variant training
    python main.py --config configs/variant_infill_heavy.yaml
    python main.py --config configs/variant_forecast_heavy.yaml
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime

# Setup path for package imports
script_dir = Path(__file__).parent.absolute()
experimental_dir = script_dir.parent.absolute()
sys.path.insert(0, str(experimental_dir))

from random_masking.data.collector import BinanceCollector
from random_masking.data.preprocessor import CandlePreprocessor
from random_masking.data.masking_strategy import RandomMaskingStrategy
from random_masking.data.dataset import CandleDataset, collate_fn_train
from random_masking.models.predictor import CandlePredictor
from random_masking.training.trainer import Trainer
from random_masking.evaluation.backtester import Backtester, BacktestConfig
from random_masking.evaluation.visualizer import ResultsVisualizer
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Random Masking Candle Predictor - Training')

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to pre-collected data (overrides config)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (cuda/cpu/auto)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/experiments',
        help='Directory to save results'
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collect_data(config, data_path=None):
    """Collect or load historical data"""
    if data_path and Path(data_path).exists():
        logger.info(f"Loading pre-collected data from {data_path}")
        data = pd.read_parquet(data_path)
        return data

    logger.info("Collecting data from Binance...")
    collector = BinanceCollector(exchange='binance')

    # Collect data for all symbols
    all_data = {}
    for symbol in config['data']['symbols']:
        logger.info(f"Fetching {symbol} data...")

        # Calculate required candles based on date range
        start_date = pd.to_datetime(config['data']['train_start'])
        end_date = pd.to_datetime(config['data']['test_end'])

        symbol_data = collector.fetch_historical(
            symbol=symbol,
            timeframe=config['data']['interval'],
            since=start_date,
            limit=None  # Fetch all available
        )

        all_data[symbol] = symbol_data

    # Combine data (for multi-asset training, concatenate)
    # For now, just use first symbol
    data = all_data[config['data']['symbols'][0]]

    logger.info(f"✅ Collected {len(data)} candles")
    logger.info(f"   Date range: {data.index[0]} to {data.index[-1]}")

    return data


def split_data(data, config):
    """Split data into train/val/test based on config"""
    train_start = pd.to_datetime(config['data']['train_start'])
    train_end = pd.to_datetime(config['data']['train_end'])
    val_start = pd.to_datetime(config['data']['val_start'])
    val_end = pd.to_datetime(config['data']['val_end'])
    test_start = pd.to_datetime(config['data']['test_start'])
    test_end = pd.to_datetime(config['data']['test_end'])

    train_data = data[(data.index >= train_start) & (data.index < train_end)]
    val_data = data[(data.index >= val_start) & (data.index < val_end)]
    test_data = data[(data.index >= test_start) & (data.index < test_end)]

    logger.info(f"Data split:")
    logger.info(f"  Train: {len(train_data)} candles ({train_start} to {train_end})")
    logger.info(f"  Val: {len(val_data)} candles ({val_start} to {val_end})")
    logger.info(f"  Test: {len(test_data)} candles ({test_start} to {test_end})")

    return train_data, val_data, test_data


def create_datasets(train_data, val_data, test_data, config):
    """Create PyTorch datasets with optional masking"""
    seq_len = config['data']['seq_len']
    pred_len = config['data']['pred_len']

    # Create masking strategy if enabled
    masking_strategy = None
    if config['masking']['enabled']:
        logger.info("Creating random masking strategy...")
        masking_strategy = RandomMaskingStrategy(
            seq_len=seq_len,
            mask_ratio_infill=config['masking']['ratio_infill'],
            mask_ratio_forecast=config['masking']['ratio_forecast'],
            mask_ratio_sparse=config['masking']['ratio_sparse']
        )

        logger.info(f"✅ Masking ratios:")
        logger.info(f"   Infill: {config['masking']['ratio_infill']:.1%}")
        logger.info(f"   Forecast: {config['masking']['ratio_forecast']:.1%}")
        logger.info(f"   Sparse: {config['masking']['ratio_sparse']:.1%}")
    else:
        logger.info("⚠️  Masking DISABLED - Pure forecasting mode")

    # Create datasets
    train_dataset = CandleDataset(
        data=train_data,
        seq_len=seq_len,
        pred_len=pred_len,
        mode='train',
        masking_strategy=masking_strategy
    )

    val_dataset = CandleDataset(
        data=val_data,
        seq_len=seq_len,
        pred_len=pred_len,
        mode='val'
    )

    test_dataset = CandleDataset(
        data=test_data,
        seq_len=seq_len,
        pred_len=pred_len,
        mode='test'
    )

    logger.info(f"✅ Datasets created:")
    logger.info(f"   Train: {len(train_dataset)} sequences")
    logger.info(f"   Val: {len(val_dataset)} sequences")
    logger.info(f"   Test: {len(test_dataset)} sequences")

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, config):
    """Create PyTorch dataloaders"""
    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 4)
    pin_memory = config['data'].get('pin_memory', True)

    # Use collate function for training if masking enabled
    collate_fn = collate_fn_train if config['masking']['enabled'] else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info(f"✅ DataLoaders created:")
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")

    return train_loader, val_loader


def create_model(config):
    """Create CandlePredictor model"""
    model = CandlePredictor(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        ff_dim=config['model']['ff_dim'],
        dropout=config['model']['dropout'],
        use_uncertainty_head=config['model'].get('uncertainty_head', True)
    )

    logger.info(f"✅ Model created:")
    logger.info(f"   Parameters: {model.count_parameters():,}")
    logger.info(f"   Hidden dim: {config['model']['hidden_dim']}")
    logger.info(f"   Layers: {config['model']['n_layers']}")
    logger.info(f"   Attention heads: {config['model']['n_heads']}")

    return model


def train_model(model, train_loader, val_loader, config, device, output_dir):
    """Train the model"""
    training_config = {
        'epochs': config['training']['epochs'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'patience': config['training']['patience'],
        'gradient_clip': config['training']['grad_clip'],
        'checkpoint_dir': str(output_dir / 'checkpoints'),
        'log_dir': str(output_dir / 'logs'),
        'use_tensorboard': config['logging'].get('tensorboard', True)
    }

    logger.info(f"Training configuration:")
    logger.info(f"  Epochs: {training_config['epochs']}")
    logger.info(f"  Learning rate: {training_config['learning_rate']}")
    logger.info(f"  Patience: {training_config['patience']}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device
    )

    logger.info("Starting training...")
    history = trainer.train()

    logger.info(f"✅ Training complete!")
    logger.info(f"   Epochs: {len(history['train_loss'])}")
    logger.info(f"   Best val loss: {min(history['val_loss']):.4f}")
    logger.info(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"   Final val loss: {history['val_loss'][-1]:.4f}")

    return history


def evaluate_model(model, test_data, config, device, output_dir):
    """Evaluate model with backtesting"""
    logger.info("Running backtest evaluation...")

    backtest_config = BacktestConfig(
        initial_capital=config['evaluation']['backtest']['initial_capital'],
        max_position_size=config['evaluation']['backtest']['max_position_size'],
        stop_loss_pct=config['evaluation']['backtest']['stop_loss_pct'],
        take_profit_pct=config['evaluation']['backtest']['take_profit_pct'],
        max_hold_candles=config['evaluation']['backtest'].get('max_hold_candles', 48),
        slippage_pct=config['evaluation']['backtest']['slippage_pct'],
        fee_pct=config['evaluation']['backtest']['commission_pct'],
        leverage=config['evaluation']['backtest']['leverage'],
        kelly_fraction=config['evaluation']['backtest'].get('kelly_fraction', 0.25)
    )

    backtester = Backtester(model, backtest_config, device=device)
    results = backtester.run(test_data, seq_len=config['data']['seq_len'], verbose=True)

    # Print results
    logger.info(f"\n📊 BACKTEST RESULTS:")
    logger.info(f"\nReturns:")
    logger.info(f"  Total Return: {results.total_return:.2%}")
    logger.info(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    logger.info(f"  Max Drawdown: {results.max_drawdown_pct:.2%}")

    logger.info(f"\nTrades:")
    logger.info(f"  Total Trades: {len(results.trades)}")
    logger.info(f"  Win Rate: {results.win_rate:.2%}")
    logger.info(f"  Profit Factor: {results.profit_factor:.2f}")

    # Save results
    results.equity_curve.to_csv(output_dir / 'equity_curve.csv')
    if results.trades:
        trades_df = pd.DataFrame(results.trades)
        trades_df.to_csv(output_dir / 'trades.csv', index=False)

    # Create visualizations
    visualizer = ResultsVisualizer()
    visualizer.create_full_report(
        equity_curve=results.equity_curve,
        trades=results.trades,
        predictions_df=results.predictions_df,
        metrics=results.metrics,
        save_dir=str(output_dir / 'visualizations')
    )

    logger.info(f"✅ Results saved to {output_dir}")

    return results


def main():
    """Main training pipeline"""
    args = parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = load_config(args.config)
    experiment_name = config['experiment']['name']
    output_dir = Path(args.output_dir) / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger.add(output_dir / 'training.log')

    logger.info("=" * 70)
    logger.info(f"Random Masking Candle Predictor - Training")
    logger.info(f"Experiment: {experiment_name}")
    logger.info("=" * 70)

    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info(f"\nConfiguration:")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Experiment: {experiment_name}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Output: {output_dir}")

    # Save config to output directory
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Step 1: Collect/Load Data
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("=" * 70)
    data = collect_data(config, args.data_path)

    # Step 2: Preprocessing
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: PREPROCESSING")
    logger.info("=" * 70)
    preprocessor = CandlePreprocessor(
        normalization=config['data']['normalization'],
        rolling_window=config['data'].get('rolling_window', 1000)
    )
    data_normalized = preprocessor.fit_transform(data)
    logger.info(f"✅ Preprocessing complete ({data_normalized.shape})")

    # Step 3: Split Data
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: DATA SPLITTING")
    logger.info("=" * 70)
    train_data, val_data, test_data = split_data(data_normalized, config)

    # Create test DataFrame for backtesting
    test_start_idx = len(train_data) + len(val_data)
    test_data_df = pd.DataFrame(
        test_data,
        index=data.index[test_start_idx:test_start_idx + len(test_data)],
        columns=data.columns
    )

    # Step 4: Create Datasets
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: DATASET CREATION")
    logger.info("=" * 70)
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_data, val_data, test_data, config
    )

    # Step 5: Create DataLoaders
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)

    # Step 6: Create Model
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: MODEL CREATION")
    logger.info("=" * 70)
    model = create_model(config)

    # Step 7: Train Model
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: TRAINING")
    logger.info("=" * 70)
    history = train_model(model, train_loader, val_loader, config, device, output_dir)

    # Step 8: Evaluate Model
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: BACKTESTING")
    logger.info("=" * 70)
    results = evaluate_model(model, test_data_df, config, device, output_dir)

    # Step 9: Final Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\n✅ Experiment '{experiment_name}' completed successfully!")
    logger.info(f"\nKey Metrics:")
    logger.info(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    logger.info(f"  Total Return: {results.total_return:.2%}")
    logger.info(f"  Win Rate: {results.win_rate:.2%}")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
