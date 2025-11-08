"""
Main Training Script for Random Masking Candle Predictor

Usage:
    python train.py --config ../configs/training_config.yaml --symbol BTC/USDT

Features:
- Data collection and preprocessing
- Random masking curriculum
- Model training with checkpointing
- TensorBoard logging
- Model evaluation
"""

import argparse
import torch
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from loguru import logger
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.collector import BinanceCollector
from data.preprocessor import CandlePreprocessor
from data.masking_strategy import RandomMaskingStrategy
from data.dataset import CandleDataset
from models.predictor import CandlePredictor
from training.trainer import Trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Random Masking Candle Predictor')

    parser.add_argument(
        '--config',
        type=str,
        default='../configs/training_config.yaml',
        help='Path to training configuration file'
    )

    parser.add_argument(
        '--model-config',
        type=str,
        default='../configs/model_config.yaml',
        help='Path to model configuration file'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Trading symbol'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        default='5m',
        help='Candle timeframe'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=20000,
        help='Number of candles to fetch'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (cuda/cpu/auto)'
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collect_and_preprocess_data(
    symbol: str,
    timeframe: str,
    limit: int
):
    """
    Collect and preprocess data

    Returns:
        Preprocessed dataframe
    """
    logger.info(f"Collecting data for {symbol} ({timeframe})...")

    # Collect
    collector = BinanceCollector(exchange='binance')
    data = collector.fetch_historical(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit
    )

    logger.info(f"Collected {len(data)} candles")
    logger.info(f"  Date range: {data.index[0]} to {data.index[-1]}")
    logger.info(f"  Features: {data.shape[1]}")

    # Preprocess
    logger.info("Preprocessing data...")
    preprocessor = CandlePreprocessor(method='rolling', rolling_window=1000)
    data_normalized = preprocessor.fit_transform(data)

    logger.info(f"Preprocessing complete")

    return data_normalized


def create_datasets(
    data,
    model_config: dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
):
    """
    Create train/val/test datasets

    Args:
        data: Preprocessed data
        model_config: Model configuration
        train_ratio: Training data ratio
        val_ratio: Validation data ratio

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Split data
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train_data)} ({train_ratio:.1%})")
    logger.info(f"  Val: {len(val_data)} ({val_ratio:.1%})")
    logger.info(f"  Test: {len(test_data)} ({1-train_ratio-val_ratio:.1%})")

    # Create masking strategy
    masking_strategy = RandomMaskingStrategy(
        seq_len=model_config['data']['seq_len'],
        pred_len=model_config['data']['pred_len'],
        infill_ratio=model_config['masking']['infill_ratio'],
        forecast_ratio=model_config['masking']['forecast_ratio'],
        sparse_ratio=model_config['masking']['sparse_ratio'],
        min_mask_len=model_config['masking']['min_mask_len'],
        max_mask_len=model_config['masking']['max_mask_len']
    )

    # Create datasets
    train_dataset = CandleDataset(
        train_data,
        seq_len=model_config['data']['seq_len'],
        pred_len=model_config['data']['pred_len'],
        mode='train',
        masking_strategy=masking_strategy
    )

    val_dataset = CandleDataset(
        val_data,
        seq_len=model_config['data']['seq_len'],
        pred_len=model_config['data']['pred_len'],
        mode='val'
    )

    test_dataset = CandleDataset(
        test_data,
        seq_len=model_config['data']['seq_len'],
        pred_len=model_config['data']['pred_len'],
        mode='test'
    )

    return train_dataset, val_dataset, test_dataset


def main():
    """Main training pipeline"""
    # Parse arguments
    args = parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path('../logs') / f'training_{timestamp}.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(log_file, rotation="100 MB")

    logger.info("=" * 60)
    logger.info("Random Masking Candle Predictor - Training")
    logger.info("=" * 60)

    # Load configurations
    train_config = load_config(args.config)
    model_config = load_config(args.model_config)

    logger.info(f"\nConfigurations loaded:")
    logger.info(f"  Training config: {args.config}")
    logger.info(f"  Model config: {args.model_config}")

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info(f"  Device: {device}")

    # Collect and preprocess data
    data = collect_and_preprocess_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit
    )

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        data,
        model_config,
        train_ratio=0.7,
        val_ratio=0.15
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['training']['batch_size'],
        shuffle=True,
        num_workers=train_config['training'].get('num_workers', 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['training']['batch_size'] * 2,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"\nDataLoaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")

    # Create model
    model = CandlePredictor(
        input_dim=model_config['model']['input_dim'],
        hidden_dim=model_config['model']['hidden_dim'],
        n_layers=model_config['model']['n_layers'],
        n_heads=model_config['model']['n_heads'],
        ff_dim=model_config['model']['ff_dim'],
        dropout=model_config['model']['dropout'],
        max_seq_len=model_config['data']['seq_len'],
        use_uncertainty_head=True
    )

    logger.info(f"\nModel created:")
    logger.info(f"  Parameters: {model.count_parameters():,}")
    logger.info(f"  Input dim: {model_config['model']['input_dim']}")
    logger.info(f"  Hidden dim: {model_config['model']['hidden_dim']}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config['training'],
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info(f"\nStarting training...")
    logger.info(f"  Epochs: {train_config['training']['epochs']}")
    logger.info(f"  Learning rate: {train_config['training']['learning_rate']}")
    logger.info(f"  Patience: {train_config['training']['patience']}")

    history = trainer.train()

    logger.info(f"\nTraining complete!")
    logger.info(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"  Total epochs: {len(history['train_loss'])}")

    # Save final summary
    summary_path = Path(train_config['training']['checkpoint_dir']) / 'training_summary.yaml'
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'training_date': timestamp,
        'total_epochs': len(history['train_loss']),
        'best_val_loss': float(trainer.best_val_loss),
        'final_train_loss': float(history['train_loss'][-1]),
        'model_parameters': model.count_parameters(),
        'device': device
    }

    with open(summary_path, 'w') as f:
        yaml.dump(summary, f)

    logger.info(f"\nTraining summary saved to: {summary_path}")
    logger.info("=" * 60)
    logger.info("Training pipeline complete! ✅")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
