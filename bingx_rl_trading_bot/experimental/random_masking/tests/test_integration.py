"""
Integration Tests for Random Masking Candle Predictor

End-to-end pipeline validation:
1. Data Collection → Preprocessing → Dataset Creation
2. Model Creation → Training → Checkpointing
3. Evaluation → Backtesting → Visualization
4. Complete pipeline with random masking curriculum
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import tempfile
import shutil

from random_masking.data.collector import BinanceCollector
from random_masking.data.preprocessor import CandlePreprocessor
from random_masking.data.masking_strategy import RandomMaskingStrategy
from random_masking.data.dataset import CandleDataset
from random_masking.models.predictor import CandlePredictor
from random_masking.training.trainer import Trainer
from random_masking.evaluation.backtester import Backtester, BacktestConfig
from random_masking.evaluation.visualizer import ResultsVisualizer
from random_masking.trading.signal_generator import SignalGenerator
from random_masking.trading.risk_manager import RiskManager


class IntegrationTester:
    """
    Comprehensive integration testing framework

    Tests:
    - Data pipeline (collection → preprocessing → dataset)
    - Model pipeline (creation → training → inference)
    - Evaluation pipeline (backtesting → metrics → visualization)
    - Full end-to-end pipeline
    """

    def __init__(self, use_real_data: bool = False):
        """
        Initialize integration tester

        Args:
            use_real_data: If True, fetch real data from Binance
                          If False, use synthetic data (faster)
        """
        self.use_real_data = use_real_data
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.test_results = {}

        logger.info(f"Initialized IntegrationTester:")
        logger.info(f"  Use real data: {use_real_data}")
        logger.info(f"  Device: {self.device}")

    def test_data_pipeline(self) -> bool:
        """
        Test data collection → preprocessing → dataset creation

        Returns:
            True if all tests pass
        """
        logger.info("\n" + "=" * 60)
        logger.info("TEST 1: DATA PIPELINE")
        logger.info("=" * 60)

        try:
            # Step 1: Data Collection
            logger.info("\n1.1 Testing data collection...")

            if self.use_real_data:
                collector = BinanceCollector(exchange='binance')
                data = collector.fetch_historical(
                    symbol='BTC/USDT',
                    timeframe='5m',
                    limit=1000
                )
            else:
                # Synthetic data
                dates = pd.date_range('2025-01-01', periods=1000, freq='5min')
                data = pd.DataFrame({
                    'open': 100 + np.cumsum(np.random.randn(1000) * 0.1),
                    'high': 101 + np.cumsum(np.random.randn(1000) * 0.1),
                    'low': 99 + np.cumsum(np.random.randn(1000) * 0.1),
                    'close': 100 + np.cumsum(np.random.randn(1000) * 0.1),
                    'volume': np.random.rand(1000) * 1000
                }, index=dates)

                # Add derived features
                data['returns'] = data['close'].pct_change()
                data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
                data['volatility'] = data['returns'].rolling(20).std()
                data['volume_change'] = data['volume'].pct_change()

            assert len(data) == 1000, "Data collection failed: wrong length"
            assert data.shape[1] >= 5, "Data collection failed: missing columns"
            logger.info(f"✅ Data collection: {len(data)} candles, {data.shape[1]} features")

            # Step 2: Preprocessing
            logger.info("\n1.2 Testing preprocessing...")
            preprocessor = CandlePreprocessor(normalization='rolling_zscore', rolling_window=100)
            data_normalized = preprocessor.fit_transform(data)

            assert len(data_normalized) == len(data), "Preprocessing changed data length"
            assert not data_normalized.isnull().any().any(), "Preprocessing created NaN values"
            logger.info(f"✅ Preprocessing: {len(data_normalized)} candles normalized")

            # Step 3: Masking Strategy
            logger.info("\n1.3 Testing masking strategy...")
            masking_strategy = RandomMaskingStrategy(
                seq_len=100,
                pred_len=10,
                infill_ratio=0.4,
                forecast_ratio=0.4,
                sparse_ratio=0.2,
                min_mask_len=10,
                max_mask_len=30
            )

            test_mask = masking_strategy.create_mask()
            assert test_mask.shape[0] == 100, "Mask creation failed: wrong length"
            logger.info(f"✅ Masking strategy: {test_mask.sum()} positions masked")

            # Step 4: Dataset Creation
            logger.info("\n1.4 Testing dataset creation...")
            dataset = CandleDataset(
                data_normalized,
                seq_len=100,
                pred_len=10,
                mode='train',
                masking_strategy=masking_strategy
            )

            assert len(dataset) > 0, "Dataset creation failed: empty dataset"

            # Test data loading
            sample = dataset[0]
            assert 'sequence' in sample, "Dataset missing 'sequence'"
            assert 'target' in sample, "Dataset missing 'target'"
            assert 'mask' in sample, "Dataset missing 'mask'"
            assert 'task_type' in sample, "Dataset missing 'task_type'"

            logger.info(f"✅ Dataset creation: {len(dataset)} samples")
            logger.info(f"   Sample shape: {sample['sequence'].shape}")
            logger.info(f"   Task type: {sample['task_type']}")

            self.test_results['data_pipeline'] = {
                'passed': True,
                'data_shape': data.shape,
                'dataset_length': len(dataset),
                'sample_shape': sample['sequence'].shape
            }

            logger.info("\n✅ DATA PIPELINE TEST: PASSED")
            return True

        except Exception as e:
            logger.error(f"\n❌ DATA PIPELINE TEST: FAILED")
            logger.error(f"Error: {str(e)}")
            self.test_results['data_pipeline'] = {'passed': False, 'error': str(e)}
            return False

    def test_model_pipeline(self) -> bool:
        """
        Test model creation → training → inference

        Returns:
            True if all tests pass
        """
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: MODEL PIPELINE")
        logger.info("=" * 60)

        try:
            # Step 1: Model Creation
            logger.info("\n2.1 Testing model creation...")
            model = CandlePredictor(
                input_dim=15,
                hidden_dim=64,  # Small for testing
                n_layers=2,
                n_heads=4,
                ff_dim=256,
                dropout=0.1,
                use_uncertainty_head=True
            )

            param_count = model.count_parameters()
            logger.info(f"✅ Model created: {param_count:,} parameters")

            # Step 2: Forward Pass
            logger.info("\n2.2 Testing forward pass...")
            test_sequence = torch.randn(2, 100, 15)  # Batch of 2
            test_mask = torch.zeros(2, 100).bool()
            test_mask[:, 50:60] = True  # Mask middle section

            output = model(test_sequence, test_mask, task_types=['infill'])

            assert 'predictions' in output, "Model output missing 'predictions'"
            assert 'uncertainty' in output, "Model output missing 'uncertainty'"
            assert output['predictions'].shape == (2, 100, 15), "Wrong prediction shape"
            logger.info(f"✅ Forward pass: {output['predictions'].shape}")

            # Step 3: Prediction Methods
            logger.info("\n2.3 Testing prediction methods...")

            # Test infill prediction
            infill_result = model.predict(
                test_sequence[:1],
                n_samples=5,
                task_types=['infill']
            )
            assert 'mean' in infill_result, "Infill prediction missing 'mean'"
            assert 'std' in infill_result, "Infill prediction missing 'std'"
            logger.info(f"✅ Infill prediction: mean={infill_result['mean'].shape}, std={infill_result['std'].shape}")

            # Test forecast prediction
            forecast_result = model.predict(
                test_sequence[:1],
                n_samples=5,
                task_types=['forecast']
            )
            logger.info(f"✅ Forecast prediction: mean={forecast_result['mean'].shape}, std={forecast_result['std'].shape}")

            # Step 4: Quick Training Test (1 epoch)
            logger.info("\n2.4 Testing training loop...")

            # Create minimal dataset
            dates = pd.date_range('2025-01-01', periods=500, freq='5min')
            data = pd.DataFrame({
                'open': 100 + np.cumsum(np.random.randn(500) * 0.1),
                'high': 101 + np.cumsum(np.random.randn(500) * 0.1),
                'low': 99 + np.cumsum(np.random.randn(500) * 0.1),
                'close': 100 + np.cumsum(np.random.randn(500) * 0.1),
                'volume': np.random.rand(500) * 1000,
                'returns': np.random.randn(500) * 0.01,
                'log_returns': np.random.randn(500) * 0.01,
                'volatility': np.abs(np.random.randn(500) * 0.02),
                'volume_change': np.random.randn(500) * 0.1
            }, index=dates)

            preprocessor = CandlePreprocessor(normalization='rolling_zscore', rolling_window=100)
            data_normalized = preprocessor.fit_transform(data)

            masking_strategy = RandomMaskingStrategy(
                seq_len=100, pred_len=10,
                infill_ratio=0.4, forecast_ratio=0.4, sparse_ratio=0.2
            )

            train_dataset = CandleDataset(
                data_normalized, seq_len=100, pred_len=10,
                mode='train', masking_strategy=masking_strategy
            )

            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

            # Create temp directory for checkpoints
            temp_dir = tempfile.mkdtemp()

            try:
                config = {
                    'epochs': 1,  # Just 1 epoch for testing
                    'learning_rate': 1e-4,
                    'weight_decay': 1e-5,
                    'patience': 10,
                    'gradient_clip': 1.0,
                    'checkpoint_dir': temp_dir,
                    'log_dir': temp_dir,
                    'use_tensorboard': False  # Disable for testing
                }

                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=None,  # No validation for quick test
                    config=config,
                    device=self.device
                )

                # Train for 1 epoch
                history = trainer.train()

                assert len(history['train_loss']) == 1, "Training history wrong length"
                logger.info(f"✅ Training: Loss = {history['train_loss'][0]:.4f}")

            finally:
                # Cleanup temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)

            self.test_results['model_pipeline'] = {
                'passed': True,
                'parameters': param_count,
                'forward_pass': True,
                'training': True
            }

            logger.info("\n✅ MODEL PIPELINE TEST: PASSED")
            return True

        except Exception as e:
            logger.error(f"\n❌ MODEL PIPELINE TEST: FAILED")
            logger.error(f"Error: {str(e)}")
            self.test_results['model_pipeline'] = {'passed': False, 'error': str(e)}
            return False

    def test_evaluation_pipeline(self) -> bool:
        """
        Test backtesting → metrics → visualization

        Returns:
            True if all tests pass
        """
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: EVALUATION PIPELINE")
        logger.info("=" * 60)

        try:
            # Step 1: Create test model and data
            logger.info("\n3.1 Preparing test data...")

            dates = pd.date_range('2025-01-01', periods=500, freq='5min')
            data = pd.DataFrame({
                'open': 100 + np.cumsum(np.random.randn(500) * 0.1),
                'high': 101 + np.cumsum(np.random.randn(500) * 0.1),
                'low': 99 + np.cumsum(np.random.randn(500) * 0.1),
                'close': 100 + np.cumsum(np.random.randn(500) * 0.1),
                'volume': np.random.rand(500) * 1000,
                'returns': np.random.randn(500) * 0.01,
                'log_returns': np.random.randn(500) * 0.01,
                'volatility': np.abs(np.random.randn(500) * 0.02),
                'volume_change': np.random.randn(500) * 0.1
            }, index=dates)

            preprocessor = CandlePreprocessor(normalization='rolling_zscore', rolling_window=100)
            data_normalized = preprocessor.fit_transform(data)

            model = CandlePredictor(
                input_dim=15, hidden_dim=64, n_layers=2, n_heads=4,
                ff_dim=256, dropout=0.1, use_uncertainty_head=True
            )
            model.eval()

            logger.info(f"✅ Test data prepared: {len(data)} candles")

            # Step 2: Signal Generation
            logger.info("\n3.2 Testing signal generation...")
            signal_generator = SignalGenerator(
                min_confidence=0.6,
                max_uncertainty_pct=0.02,
                min_price_move_pct=0.001
            )

            test_prediction = np.array([100, 101, 99, 100.5, 1000])
            test_uncertainty = np.array([0.5, 0.5, 0.5, 0.5, 10])
            test_candle = {'close': 100.0}

            signal = signal_generator.generate_signal(
                test_prediction,
                test_uncertainty,
                test_candle
            )

            assert 'action' in signal, "Signal missing 'action'"
            assert 'confidence' in signal, "Signal missing 'confidence'"
            logger.info(f"✅ Signal generation: {signal['action']} (confidence: {signal['confidence']:.2%})")

            # Step 3: Risk Management
            logger.info("\n3.3 Testing risk management...")

            from dataclasses import dataclass

            @dataclass
            class TestConfig:
                max_position_size: float = 0.1
                kelly_fraction: float = 0.25
                leverage: float = 1.0

            risk_manager = RiskManager(TestConfig())

            position_size = risk_manager.calculate_position_size(
                capital=10000,
                confidence=0.7,
                current_price=100.0,
                method='confidence_weighted'
            )

            assert position_size > 0, "Position sizing failed"
            logger.info(f"✅ Risk management: Position size = {position_size:.4f}")

            # Step 4: Backtesting
            logger.info("\n3.4 Testing backtesting...")

            backtest_config = BacktestConfig(
                initial_capital=10000,
                max_position_size=0.1,
                stop_loss_pct=0.02,
                take_profit_pct=0.06,
                slippage_pct=0.0005,
                fee_pct=0.0004
            )

            backtester = Backtester(model, backtest_config, device=self.device)
            results = backtester.run(data_normalized, seq_len=100, verbose=False)

            assert hasattr(results, 'equity_curve'), "Results missing equity_curve"
            assert hasattr(results, 'trades'), "Results missing trades"
            assert hasattr(results, 'metrics'), "Results missing metrics"

            logger.info(f"✅ Backtesting: {len(results.trades)} trades executed")
            logger.info(f"   Total return: {results.total_return:.2%}")
            logger.info(f"   Sharpe ratio: {results.sharpe_ratio:.2f}")

            # Step 5: Visualization (without display)
            logger.info("\n3.5 Testing visualization...")

            temp_dir = tempfile.mkdtemp()

            try:
                visualizer = ResultsVisualizer()

                # Test individual plots (save without showing)
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend

                visualizer.plot_equity_curve(
                    results.equity_curve,
                    results.trades,
                    save_path=str(Path(temp_dir) / 'equity.png')
                )

                if results.trades:
                    visualizer.plot_trade_analysis(
                        results.trades,
                        save_path=str(Path(temp_dir) / 'trades.png')
                    )

                logger.info(f"✅ Visualization: Plots saved to {temp_dir}")

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            self.test_results['evaluation_pipeline'] = {
                'passed': True,
                'trades': len(results.trades),
                'return': results.total_return,
                'sharpe': results.sharpe_ratio
            }

            logger.info("\n✅ EVALUATION PIPELINE TEST: PASSED")
            return True

        except Exception as e:
            logger.error(f"\n❌ EVALUATION PIPELINE TEST: FAILED")
            logger.error(f"Error: {str(e)}")
            self.test_results['evaluation_pipeline'] = {'passed': False, 'error': str(e)}
            return False

    def test_full_pipeline(self) -> bool:
        """
        Test complete end-to-end pipeline

        Returns:
            True if all tests pass
        """
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: FULL END-TO-END PIPELINE")
        logger.info("=" * 60)

        try:
            logger.info("\n4.1 Running complete pipeline...")

            # Full pipeline: Data → Train → Evaluate

            # 1. Data
            dates = pd.date_range('2025-01-01', periods=1000, freq='5min')
            data = pd.DataFrame({
                'open': 100 + np.cumsum(np.random.randn(1000) * 0.1),
                'high': 101 + np.cumsum(np.random.randn(1000) * 0.1),
                'low': 99 + np.cumsum(np.random.randn(1000) * 0.1),
                'close': 100 + np.cumsum(np.random.randn(1000) * 0.1),
                'volume': np.random.rand(1000) * 1000,
                'returns': np.random.randn(1000) * 0.01,
                'log_returns': np.random.randn(1000) * 0.01,
                'volatility': np.abs(np.random.randn(1000) * 0.02),
                'volume_change': np.random.randn(1000) * 0.1
            }, index=dates)

            preprocessor = CandlePreprocessor(normalization='rolling_zscore', rolling_window=100)
            data_normalized = preprocessor.fit_transform(data)

            # 2. Split data
            train_end = int(len(data_normalized) * 0.7)
            val_end = int(len(data_normalized) * 0.85)

            train_data = data_normalized.iloc[:train_end]
            val_data = data_normalized.iloc[train_end:val_end]
            test_data = data_normalized.iloc[val_end:]

            logger.info(f"   Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

            # 3. Create datasets
            masking_strategy = RandomMaskingStrategy(
                seq_len=100, pred_len=10,
                infill_ratio=0.4, forecast_ratio=0.4, sparse_ratio=0.2
            )

            train_dataset = CandleDataset(
                train_data, seq_len=100, pred_len=10,
                mode='train', masking_strategy=masking_strategy
            )

            val_dataset = CandleDataset(
                val_data, seq_len=100, pred_len=10, mode='val'
            )

            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

            # 4. Train model
            model = CandlePredictor(
                input_dim=15, hidden_dim=64, n_layers=2, n_heads=4,
                ff_dim=256, dropout=0.1, use_uncertainty_head=True
            )

            temp_dir = tempfile.mkdtemp()

            try:
                config = {
                    'epochs': 2,  # Minimal epochs for testing
                    'learning_rate': 1e-4,
                    'weight_decay': 1e-5,
                    'patience': 10,
                    'gradient_clip': 1.0,
                    'checkpoint_dir': temp_dir,
                    'log_dir': temp_dir,
                    'use_tensorboard': False
                }

                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=config,
                    device=self.device
                )

                history = trainer.train()

                logger.info(f"   Training complete: {len(history['train_loss'])} epochs")
                logger.info(f"   Final train loss: {history['train_loss'][-1]:.4f}")
                logger.info(f"   Final val loss: {history['val_loss'][-1]:.4f}")

                # 5. Evaluate
                backtest_config = BacktestConfig(
                    initial_capital=10000,
                    max_position_size=0.1,
                    stop_loss_pct=0.02,
                    take_profit_pct=0.06
                )

                backtester = Backtester(model, backtest_config, device=self.device)
                results = backtester.run(test_data, seq_len=100, verbose=False)

                logger.info(f"   Backtest complete: {len(results.trades)} trades")
                logger.info(f"   Total return: {results.total_return:.2%}")
                logger.info(f"   Win rate: {results.win_rate:.2%}")

                self.test_results['full_pipeline'] = {
                    'passed': True,
                    'epochs': len(history['train_loss']),
                    'final_val_loss': history['val_loss'][-1],
                    'trades': len(results.trades),
                    'return': results.total_return
                }

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            logger.info("\n✅ FULL PIPELINE TEST: PASSED")
            return True

        except Exception as e:
            logger.error(f"\n❌ FULL PIPELINE TEST: FAILED")
            logger.error(f"Error: {str(e)}")
            self.test_results['full_pipeline'] = {'passed': False, 'error': str(e)}
            return False

    def run_all_tests(self) -> bool:
        """
        Run all integration tests

        Returns:
            True if all tests pass
        """
        logger.info("\n" + "=" * 70)
        logger.info("RANDOM MASKING CANDLE PREDICTOR - INTEGRATION TESTS")
        logger.info("=" * 70)

        all_passed = True

        # Run tests
        all_passed &= self.test_data_pipeline()
        all_passed &= self.test_model_pipeline()
        all_passed &= self.test_evaluation_pipeline()
        all_passed &= self.test_full_pipeline()

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("TEST SUMMARY")
        logger.info("=" * 70)

        for test_name, results in self.test_results.items():
            status = "✅ PASSED" if results['passed'] else "❌ FAILED"
            logger.info(f"{test_name}: {status}")

            if not results['passed']:
                logger.error(f"  Error: {results.get('error', 'Unknown')}")

        logger.info("\n" + "=" * 70)
        if all_passed:
            logger.info("✅ ALL INTEGRATION TESTS PASSED")
        else:
            logger.error("❌ SOME TESTS FAILED")
        logger.info("=" * 70)

        return all_passed


if __name__ == '__main__':
    # Run integration tests
    logger.info("Starting integration tests...")

    # Use synthetic data for faster testing
    # Set use_real_data=True to test with real Binance data
    tester = IntegrationTester(use_real_data=False)

    success = tester.run_all_tests()

    if success:
        logger.info("\n✅ Integration tests completed successfully!")
        logger.info("System is ready for use.")
    else:
        logger.error("\n❌ Some integration tests failed!")
        logger.error("Please review the error messages above.")
