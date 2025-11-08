"""
Evaluation Script for Random Masking Candle Predictor

Usage:
    python evaluate.py --checkpoint ../checkpoints/best_model.pt --symbol BTC/USDT

Features:
- Load trained model
- Run backtest on test data
- Generate comprehensive visualizations
- Export results and metrics
"""

import argparse
import torch
import pandas as pd
import yaml
from pathlib import Path
from loguru import logger
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.collector import BinanceCollector
from data.preprocessor import CandlePreprocessor
from models.predictor import CandlePredictor
from evaluation.backtester import Backtester, BacktestConfig
from evaluation.visualizer import ResultsVisualizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Random Masking Candle Predictor')

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='../configs/trading_config.yaml',
        help='Path to trading configuration'
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
        help='Number of candles for backtest'
    )

    parser.add_argument(
        '--save-dir',
        type=str,
        default='../results/evaluation',
        help='Directory to save results'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (cuda/cpu/auto)'
    )

    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = 'cpu'
) -> CandlePredictor:
    """
    Load model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded CandlePredictor model
    """
    logger.info(f"Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get('config', {})

    # Create model with same architecture
    model = CandlePredictor(
        input_dim=config.get('input_dim', 15),
        hidden_dim=config.get('hidden_dim', 256),
        n_layers=config.get('n_layers', 6),
        n_heads=config.get('n_heads', 8),
        ff_dim=config.get('ff_dim', 1024),
        dropout=config.get('dropout', 0.1),
        use_uncertainty_head=True
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded successfully")
    logger.info(f"  Parameters: {model.count_parameters():,}")
    logger.info(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")

    return model


def main():
    """Main evaluation pipeline"""
    # Parse arguments
    args = parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    log_file = save_dir / 'evaluation.log'
    logger.add(log_file)

    logger.info("=" * 60)
    logger.info("Random Masking Candle Predictor - Evaluation")
    logger.info("=" * 60)

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info(f"\nDevice: {device}")

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, device=device)

    # Load trading config
    with open(args.config, 'r') as f:
        trading_config = yaml.safe_load(f)

    logger.info(f"\nTrading config loaded from: {args.config}")

    # Collect test data
    logger.info(f"\nCollecting test data for {args.symbol}...")

    collector = BinanceCollector(exchange='binance')
    data = collector.fetch_historical(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit
    )

    logger.info(f"Collected {len(data)} candles")
    logger.info(f"  Date range: {data.index[0]} to {data.index[-1]}")

    # Preprocess
    logger.info("Preprocessing data...")
    preprocessor = CandlePreprocessor(method='rolling', rolling_window=1000)
    data_normalized = preprocessor.fit_transform(data)

    # Create backtest config
    backtest_config = BacktestConfig(
        initial_capital=trading_config['capital']['initial_capital'],
        max_position_size=trading_config['capital']['max_position_size'],
        stop_loss_pct=trading_config['risk']['stop_loss'],
        take_profit_pct=trading_config['risk']['take_profit'],
        max_hold_candles=trading_config['risk'].get('max_hold_candles', 48),
        slippage_pct=trading_config['fees']['slippage'],
        fee_pct=trading_config['fees']['taker_fee'],
        leverage=trading_config['position']['leverage'],
        kelly_fraction=trading_config['position']['kelly_fraction']
    )

    logger.info(f"\nBacktest configuration:")
    logger.info(f"  Initial capital: ${backtest_config.initial_capital:,.2f}")
    logger.info(f"  Max position size: {backtest_config.max_position_size:.1%}")
    logger.info(f"  Stop loss: {backtest_config.stop_loss_pct:.2%}")
    logger.info(f"  Take profit: {backtest_config.take_profit_pct:.2%}")
    logger.info(f"  Leverage: {backtest_config.leverage:.1f}x")

    # Run backtest
    logger.info(f"\nRunning backtest...")

    backtester = Backtester(model, backtest_config, device=device)
    results = backtester.run(data_normalized, seq_len=100, verbose=True)

    # Save results
    logger.info(f"\nSaving results to {save_dir}...")

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
    metrics_path = save_dir / 'metrics.yaml'
    with open(metrics_path, 'w') as f:
        yaml.dump(results.metrics, f)

    # Create visualizations
    logger.info(f"\nGenerating visualizations...")

    visualizer = ResultsVisualizer()
    visualizer.create_full_report(
        equity_curve=results.equity_curve,
        trades=results.trades,
        predictions_df=results.predictions_df,
        metrics=results.metrics,
        save_dir=str(save_dir)
    )

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*60}")

    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Total Return: {results.total_return:.2%}")
    logger.info(f"  Annualized Return: {results.metrics.get('annualized_return', 0):.2%}")
    logger.info(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    logger.info(f"  Sortino Ratio: {results.metrics.get('sortino_ratio', 0):.2f}")
    logger.info(f"  Max Drawdown: {results.max_drawdown_pct:.2%}")

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

    logger.info(f"\nResults saved to: {save_dir}")
    logger.info("=" * 60)
    logger.info("Evaluation complete! ✅")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
