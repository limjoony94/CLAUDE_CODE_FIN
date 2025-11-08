"""
Complete Pipeline Demonstration

This script demonstrates the full workflow:
1. Data collection from Binance
2. Preprocessing and normalization
3. Model training with random masking curriculum
4. Backtesting and performance evaluation
5. Result visualization

Usage:
    python demo_pipeline.py [--real-data] [--device cuda]
"""

import argparse
import torch
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from loguru import logger
from datetime import datetime
from torch.utils.data import DataLoader

# Setup path for package imports
script_dir = Path(__file__).parent.absolute()
experimental_dir = script_dir.parent.parent.absolute()
sys.path.insert(0, str(experimental_dir))

from random_masking.data.collector import BinanceCollector
from random_masking.data.preprocessor import CandlePreprocessor
from random_masking.data.masking_strategy import RandomMaskingStrategy
from random_masking.data.dataset import CandleDataset, collate_fn_train
from random_masking.models.predictor import CandlePredictor
from random_masking.training.trainer import Trainer
from random_masking.evaluation.backtester import Backtester, BacktestConfig
from random_masking.evaluation.visualizer import ResultsVisualizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Random Masking Candle Predictor - Complete Demo')

    parser.add_argument(
        '--real-data',
        action='store_true',
        help='Use real Binance data (default: synthetic)'
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
        default=5000,
        help='Number of candles to fetch'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Training epochs'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (cuda/cpu/auto)'
    )

    parser.add_argument(
        '--save-dir',
        type=str,
        default='../results/demo',
        help='Directory to save results'
    )

    return parser.parse_args()


def main():
    """Main demo pipeline"""
    args = parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.add(save_dir / 'demo.log')

    logger.info("=" * 70)
    logger.info("Random Masking Candle Predictor - Complete Demo")
    logger.info("=" * 70)

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info(f"\nConfiguration:")
    logger.info(f"  Use real data: {args.real_data}")
    logger.info(f"  Symbol: {args.symbol}")
    logger.info(f"  Timeframe: {args.timeframe}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Save directory: {save_dir}")

    # =========================================================================
    # STEP 1: DATA COLLECTION
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("=" * 70)

    if args.real_data:
        logger.info(f"\nFetching real data from Binance...")
        collector = BinanceCollector(exchange='binance')
        data = collector.fetch_historical(
            symbol=args.symbol,
            timeframe=args.timeframe,
            limit=args.limit
        )
    else:
        logger.info(f"\nGenerating synthetic data...")
        dates = pd.date_range('2025-01-01', periods=args.limit, freq='5min')

        # Synthetic price with trend and noise
        trend = np.linspace(100, 120, args.limit)
        noise = np.cumsum(np.random.randn(args.limit) * 0.5)
        close_prices = trend + noise

        data = pd.DataFrame({
            'open': close_prices * (1 + np.random.randn(args.limit) * 0.001),
            'high': close_prices * (1 + np.abs(np.random.randn(args.limit)) * 0.002),
            'low': close_prices * (1 - np.abs(np.random.randn(args.limit)) * 0.002),
            'close': close_prices,
            'volume': np.random.rand(args.limit) * 1000
        }, index=dates)

        # Add derived features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility'] = data['returns'].rolling(20).std()
        data['volume_change'] = data['volume'].pct_change()

    logger.info(f"✅ Collected {len(data)} candles")
    logger.info(f"   Date range: {data.index[0]} to {data.index[-1]}")
    logger.info(f"   Features: {data.shape[1]}")
    logger.info(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    # =========================================================================
    # STEP 2: PREPROCESSING
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: PREPROCESSING")
    logger.info("=" * 70)

    logger.info("\nNormalizing data with rolling window...")
    preprocessor = CandlePreprocessor(normalization='rolling_zscore', rolling_window=1000)
    data_normalized = preprocessor.fit_transform(data)

    logger.info(f"✅ Preprocessing complete")
    logger.info(f"   Shape: {data_normalized.shape}")
    logger.info(f"   NaN values: {np.isnan(data_normalized).sum()}")

    # =========================================================================
    # STEP 3: DATASET CREATION
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: DATASET CREATION")
    logger.info("=" * 70)

    # Split data
    train_end = int(len(data_normalized) * 0.7)
    val_end = int(len(data_normalized) * 0.85)

    train_data = data_normalized[:train_end]
    val_data = data_normalized[train_end:val_end]
    test_data = data_normalized[val_end:]

    # Create DataFrame from normalized test data with timestamps for backtesting
    test_data_df = pd.DataFrame(
        test_data,
        index=data.index[val_end:],
        columns=data.columns
    )

    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train_data)} samples (70%)")
    logger.info(f"  Val: {len(val_data)} samples (15%)")
    logger.info(f"  Test: {len(test_data)} samples (15%)")

    # Create masking strategy
    logger.info(f"\nCreating random masking strategy...")
    masking_strategy = RandomMaskingStrategy(
        seq_len=100,
        mask_ratio_infill=0.4,
        mask_ratio_forecast=0.4,
        mask_ratio_sparse=0.2
    )

    logger.info(f"✅ Masking strategy created:")
    logger.info(f"   Infill ratio: 40%")
    logger.info(f"   Forecast ratio: 40%")
    logger.info(f"   Sparse ratio: 20%")

    # Create datasets
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

    test_dataset = CandleDataset(
        data=test_data,
        seq_len=100,
        pred_len=10,
        mode='test'
    )

    logger.info(f"✅ Datasets created:")
    logger.info(f"   Train: {len(train_dataset)} sequences")
    logger.info(f"   Val: {len(val_dataset)} sequences")
    logger.info(f"   Test: {len(test_dataset)} sequences")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_train
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")

    # =========================================================================
    # STEP 4: MODEL CREATION
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: MODEL CREATION")
    logger.info("=" * 70)

    logger.info("\nCreating CandlePredictor model...")
    model = CandlePredictor(
        input_dim=data_normalized.shape[1],
        hidden_dim=256,
        n_layers=6,
        n_heads=8,
        ff_dim=1024,
        dropout=0.1,
        use_uncertainty_head=True
    )

    logger.info(f"✅ Model created:")
    logger.info(f"   Parameters: {model.count_parameters():,}")
    logger.info(f"   Input dim: {data_normalized.shape[1]}")
    logger.info(f"   Hidden dim: 256")
    logger.info(f"   Layers: 6")
    logger.info(f"   Attention heads: 8")

    # =========================================================================
    # STEP 5: TRAINING
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: TRAINING")
    logger.info("=" * 70)

    config = {
        'epochs': args.epochs,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'patience': 10,
        'gradient_clip': 1.0,
        'checkpoint_dir': str(save_dir / 'checkpoints'),
        'log_dir': str(save_dir / 'logs'),
        'use_tensorboard': True
    }

    logger.info(f"\nTraining configuration:")
    logger.info(f"  Epochs: {config['epochs']}")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  Patience: {config['patience']}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    logger.info(f"\nStarting training...")
    history = trainer.train()

    logger.info(f"\n✅ Training complete!")
    logger.info(f"   Epochs: {len(history['train_loss'])}")
    logger.info(f"   Best val loss: {min(history['val_loss']):.4f}")
    logger.info(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"   Final val loss: {history['val_loss'][-1]:.4f}")

    # =========================================================================
    # STEP 6: BACKTESTING
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: BACKTESTING")
    logger.info("=" * 70)

    logger.info("\nConfiguring backtest...")
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

    logger.info(f"  Initial capital: ${backtest_config.initial_capital:,.2f}")
    logger.info(f"  Max position: {backtest_config.max_position_size:.1%}")
    logger.info(f"  Stop loss: {backtest_config.stop_loss_pct:.2%}")
    logger.info(f"  Take profit: {backtest_config.take_profit_pct:.2%}")

    logger.info(f"\nRunning backtest on test set...")
    backtester = Backtester(model, backtest_config, device=device)
    results = backtester.run(test_data_df, seq_len=100, verbose=True)

    # =========================================================================
    # STEP 7: RESULTS
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: RESULTS")
    logger.info("=" * 70)

    logger.info(f"\n📊 PERFORMANCE SUMMARY:")
    logger.info(f"\nReturns:")
    logger.info(f"  Total Return: {results.total_return:.2%}")
    logger.info(f"  Annualized Return: {results.metrics.get('annualized_return', 0):.2%}")
    logger.info(f"  Monthly Return: {results.metrics.get('monthly_return', 0):.2%}")

    logger.info(f"\nRisk Metrics:")
    logger.info(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    logger.info(f"  Sortino Ratio: {results.metrics.get('sortino_ratio', 0):.2f}")
    logger.info(f"  Max Drawdown: {results.max_drawdown_pct:.2%}")
    logger.info(f"  Volatility: {results.metrics.get('volatility', 0):.2%}")

    logger.info(f"\nTrade Statistics:")
    logger.info(f"  Total Trades: {len(results.trades)}")
    logger.info(f"  Win Rate: {results.win_rate:.2%}")
    logger.info(f"  Profit Factor: {results.profit_factor:.2f}")
    logger.info(f"  Avg Win: ${results.metrics.get('avg_win', 0):.2f}")
    logger.info(f"  Avg Loss: ${results.metrics.get('avg_loss', 0):.2f}")

    if results.predictions_df is not None:
        logger.info(f"\nPrediction Metrics:")
        logger.info(f"  MSE: {results.metrics.get('mse', 0):.4f}")
        logger.info(f"  MAE: {results.metrics.get('mae', 0):.4f}")
        logger.info(f"  Directional Accuracy: {results.metrics.get('directional_accuracy', 0):.2%}")

    # =========================================================================
    # STEP 8: VISUALIZATION
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: VISUALIZATION")
    logger.info("=" * 70)

    logger.info(f"\nGenerating visualizations...")
    visualizer = ResultsVisualizer()

    visualizer.create_full_report(
        equity_curve=results.equity_curve,
        trades=results.trades,
        predictions_df=results.predictions_df,
        metrics=results.metrics,
        save_dir=str(save_dir / 'visualizations')
    )

    logger.info(f"✅ Visualizations saved to {save_dir / 'visualizations'}")

    # =========================================================================
    # STEP 9: SAVE RESULTS
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 9: SAVE RESULTS")
    logger.info("=" * 70)

    logger.info(f"\nSaving results...")

    # Save equity curve
    results.equity_curve.to_csv(save_dir / 'equity_curve.csv')

    # Save trades
    if results.trades:
        trades_df = pd.DataFrame(results.trades)
        trades_df.to_csv(save_dir / 'trades.csv', index=False)

    # Save predictions
    if results.predictions_df is not None:
        results.predictions_df.to_csv(save_dir / 'predictions.csv')

    # Save metrics
    import yaml
    with open(save_dir / 'metrics.yaml', 'w') as f:
        yaml.dump(results.metrics, f)

    logger.info(f"✅ Results saved to {save_dir}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 70)

    logger.info(f"\n✅ Pipeline completed successfully!")
    logger.info(f"\nResults location: {save_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review visualizations in: {save_dir / 'visualizations'}")
    logger.info(f"  2. Check metrics: {save_dir / 'metrics.yaml'}")
    logger.info(f"  3. Analyze trades: {save_dir / 'trades.csv'}")
    logger.info(f"  4. View logs: {save_dir / 'demo.log'}")

    logger.info("\n" + "=" * 70)


if __name__ == '__main__':
    main()
