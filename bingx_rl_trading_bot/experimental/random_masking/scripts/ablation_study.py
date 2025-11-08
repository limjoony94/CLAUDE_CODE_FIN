"""
Ablation Study: Random Masking vs Pure Forecasting

Compares:
1. Baseline: Pure forecasting (no infilling tasks)
2. Proposed: Random masking curriculum (infilling + forecasting)

Metrics:
- Prediction accuracy (MSE, MAE, directional accuracy)
- Trading performance (Sharpe, return, drawdown)
- Sample efficiency (performance vs training steps)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from loguru import logger
import yaml
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.collector import BinanceCollector
from data.preprocessor import CandlePreprocessor
from data.masking_strategy import RandomMaskingStrategy
from data.dataset import CandleDataset
from models.predictor import CandlePredictor
from training.trainer import Trainer
from evaluation.backtester import Backtester, BacktestConfig
from evaluation.visualizer import ResultsVisualizer


class AblationStudy:
    """
    Ablation study framework

    Compares:
    - Random masking curriculum (proposed)
    - Pure forecasting baseline
    - Different masking ratios
    """

    def __init__(
        self,
        config_path: str = '../configs/model_config.yaml',
        save_dir: str = '../results/ablation'
    ):
        """
        Initialize ablation study

        Args:
            config_path: Path to model configuration
            save_dir: Directory to save results
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {}

        logger.info(f"Initialized AblationStudy:")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Save directory: {save_dir}")

    def run_experiment(
        self,
        name: str,
        masking_config: Dict,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Dict:
        """
        Run single experiment

        Args:
            name: Experiment name
            masking_config: Masking strategy config
            train_data: Training data
            val_data: Validation data
            test_data: Test data

        Returns:
            Results dict
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment: {name}")
        logger.info(f"{'='*60}")

        # Create masking strategy
        if masking_config['enabled']:
            masking_strategy = RandomMaskingStrategy(
                seq_len=self.config['data']['seq_len'],
                pred_len=self.config['data']['pred_len'],
                **masking_config['params']
            )
        else:
            masking_strategy = None  # Pure forecasting

        # Create datasets
        train_dataset = CandleDataset(
            train_data,
            seq_len=self.config['data']['seq_len'],
            pred_len=self.config['data']['pred_len'],
            mode='train',
            masking_strategy=masking_strategy
        )

        val_dataset = CandleDataset(
            val_data,
            seq_len=self.config['data']['seq_len'],
            pred_len=self.config['data']['pred_len'],
            mode='val'
        )

        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0
        )

        # Create model
        model = CandlePredictor(
            input_dim=self.config['model']['input_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            n_layers=self.config['model']['n_layers'],
            n_heads=self.config['model']['n_heads'],
            ff_dim=self.config['model']['ff_dim'],
            dropout=self.config['model']['dropout'],
            use_uncertainty_head=True
        )

        # Training config
        train_config = {
            'epochs': 50,  # Reduced for ablation
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'patience': 10,
            'gradient_clip': 1.0,
            'checkpoint_dir': str(self.save_dir / name / 'checkpoints'),
            'log_dir': str(self.save_dir / name / 'logs'),
            'use_tensorboard': True
        }

        # Train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config,
            device=self.device
        )

        history = trainer.train()

        # Evaluate on test set
        logger.info(f"\nEvaluating {name} on test set...")

        # Backtesting
        backtest_config = BacktestConfig(
            initial_capital=10000,
            max_position_size=0.1,
            stop_loss_pct=0.02,
            take_profit_pct=0.06
        )

        backtester = Backtester(model, backtest_config, device=self.device)
        backtest_results = backtester.run(test_data, verbose=False)

        # Save results
        results = {
            'name': name,
            'masking_config': masking_config,
            'training_history': history,
            'backtest_results': {
                'total_return': backtest_results.total_return,
                'sharpe_ratio': backtest_results.sharpe_ratio,
                'max_drawdown_pct': backtest_results.max_drawdown_pct,
                'win_rate': backtest_results.win_rate,
                'profit_factor': backtest_results.profit_factor,
                'total_trades': len(backtest_results.trades)
            },
            'prediction_metrics': {
                'mse': backtest_results.metrics.get('mse', 0),
                'mae': backtest_results.metrics.get('mae', 0),
                'directional_accuracy': backtest_results.metrics.get('directional_accuracy', 0)
            }
        }

        self.results[name] = results

        logger.info(f"\n{name} Results:")
        logger.info(f"  Training - Best Val Loss: {min(history['val_loss']):.4f}")
        logger.info(f"  Backtest - Return: {results['backtest_results']['total_return']:.2%}")
        logger.info(f"  Backtest - Sharpe: {results['backtest_results']['sharpe_ratio']:.2f}")
        logger.info(f"  Prediction - Directional Acc: {results['prediction_metrics']['directional_accuracy']:.2%}")

        return results

    def run_full_ablation(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ):
        """
        Run full ablation study with multiple experiments

        Args:
            data: Full dataset
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
        """
        # Split data
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]

        logger.info(f"\nData split:")
        logger.info(f"  Train: {len(train_data)} samples ({train_ratio:.1%})")
        logger.info(f"  Val: {len(val_data)} samples ({val_ratio:.1%})")
        logger.info(f"  Test: {len(test_data)} samples ({1-train_ratio-val_ratio:.1%})")

        # Experiments
        experiments = [
            {
                'name': 'baseline_pure_forecasting',
                'config': {
                    'enabled': False,
                    'params': {}
                }
            },
            {
                'name': 'proposed_random_masking',
                'config': {
                    'enabled': True,
                    'params': {
                        'infill_ratio': 0.4,
                        'forecast_ratio': 0.4,
                        'sparse_ratio': 0.2,
                        'min_mask_len': 10,
                        'max_mask_len': 30
                    }
                }
            },
            {
                'name': 'heavy_infilling',
                'config': {
                    'enabled': True,
                    'params': {
                        'infill_ratio': 0.7,
                        'forecast_ratio': 0.2,
                        'sparse_ratio': 0.1,
                        'min_mask_len': 10,
                        'max_mask_len': 30
                    }
                }
            },
            {
                'name': 'heavy_forecasting',
                'config': {
                    'enabled': True,
                    'params': {
                        'infill_ratio': 0.2,
                        'forecast_ratio': 0.7,
                        'sparse_ratio': 0.1,
                        'min_mask_len': 10,
                        'max_mask_len': 30
                    }
                }
            }
        ]

        # Run experiments
        for exp in experiments:
            self.run_experiment(
                name=exp['name'],
                masking_config=exp['config'],
                train_data=train_data,
                val_data=val_data,
                test_data=test_data
            )

        # Compare results
        self.compare_results()

    def compare_results(self):
        """Compare and visualize ablation results"""
        logger.info(f"\n{'='*60}")
        logger.info("ABLATION STUDY RESULTS COMPARISON")
        logger.info(f"{'='*60}")

        # Create comparison table
        comparison = []

        for name, results in self.results.items():
            comparison.append({
                'Experiment': name,
                'Total Return': f"{results['backtest_results']['total_return']:.2%}",
                'Sharpe Ratio': f"{results['backtest_results']['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{results['backtest_results']['max_drawdown_pct']:.2%}",
                'Win Rate': f"{results['backtest_results']['win_rate']:.2%}",
                'Directional Acc': f"{results['prediction_metrics']['directional_accuracy']:.2%}",
                'MSE': f"{results['prediction_metrics']['mse']:.4f}",
                'Total Trades': results['backtest_results']['total_trades']
            })

        df_comparison = pd.DataFrame(comparison)

        print("\n" + df_comparison.to_string(index=False))

        # Save comparison
        comparison_path = self.save_dir / f'ablation_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_comparison.to_csv(comparison_path, index=False)
        logger.info(f"\nComparison saved to: {comparison_path}")

        # Identify best performer
        best_return = max(self.results.items(), key=lambda x: x[1]['backtest_results']['total_return'])
        best_sharpe = max(self.results.items(), key=lambda x: x[1]['backtest_results']['sharpe_ratio'])

        logger.info(f"\n🏆 BEST PERFORMERS:")
        logger.info(f"  Highest Return: {best_return[0]} ({best_return[1]['backtest_results']['total_return']:.2%})")
        logger.info(f"  Highest Sharpe: {best_sharpe[0]} ({best_sharpe[1]['backtest_results']['sharpe_ratio']:.2f})")


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Random Masking Ablation Study")
    logger.info("=" * 60)

    # Collect data
    logger.info("\nCollecting historical data...")
    collector = BinanceCollector(exchange='binance')
    data = collector.fetch_historical(
        symbol='BTC/USDT',
        timeframe='5m',
        limit=10000
    )

    # Preprocess
    logger.info("Preprocessing data...")
    preprocessor = CandlePreprocessor(method='rolling')
    data_normalized = preprocessor.fit_transform(data)

    # Run ablation study
    logger.info("\nStarting ablation study...")
    ablation = AblationStudy(
        config_path='../configs/model_config.yaml',
        save_dir='../results/ablation'
    )

    ablation.run_full_ablation(
        data=data_normalized,
        train_ratio=0.7,
        val_ratio=0.15
    )

    logger.info("\n" + "=" * 60)
    logger.info("Ablation study complete! ✅")
    logger.info("=" * 60)
