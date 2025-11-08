"""
Live Trading Script for Random Masking Candle Predictor

WARNING: This is a template for live trading. Use at your own risk!

Usage:
    python trade_live.py --checkpoint ../checkpoints/best_model.pt --symbol BTC/USDT --mode paper

Features:
- Real-time data collection
- Model inference
- Signal generation
- Order execution (paper/live)
- Position management
- Performance monitoring
"""

import argparse
import torch
import time
import yaml
from pathlib import Path
from loguru import logger
from datetime import datetime
import ccxt

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.collector import BinanceCollector
from data.preprocessor import CandlePreprocessor
from models.predictor import CandlePredictor
from trading.signal_generator import SignalGenerator
from trading.risk_manager import RiskManager


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Live Trading with Random Masking Candle Predictor')

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
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
        '--mode',
        type=str,
        default='paper',
        choices=['paper', 'live'],
        help='Trading mode (paper/live)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (cuda/cpu/auto)'
    )

    return parser.parse_args()


class LiveTrader:
    """
    Live trading bot

    WARNING: Live trading involves real financial risk!
    """

    def __init__(
        self,
        model: CandlePredictor,
        signal_generator: SignalGenerator,
        risk_manager: RiskManager,
        config: dict,
        symbol: str,
        timeframe: str,
        mode: str = 'paper',
        device: str = 'cpu'
    ):
        """
        Initialize live trader

        Args:
            model: Trained CandlePredictor
            signal_generator: Signal generator
            risk_manager: Risk manager
            config: Trading configuration
            symbol: Trading symbol
            timeframe: Candle timeframe
            mode: 'paper' or 'live'
            device: Device for model inference
        """
        self.model = model.to(device)
        self.model.eval()

        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.config = config
        self.symbol = symbol
        self.timeframe = timeframe
        self.mode = mode
        self.device = device

        # Initialize exchange
        self.exchange = self._init_exchange()

        # Data collector
        self.collector = BinanceCollector(exchange='binance')
        self.preprocessor = CandlePreprocessor(method='rolling', rolling_window=1000)

        # Trading state
        self.position = None
        self.trades_history = []

        # Paper trading state
        if mode == 'paper':
            self.paper_capital = config['capital']['initial_capital']
            logger.info(f"Paper trading mode - Initial capital: ${self.paper_capital:,.2f}")

        logger.info(f"LiveTrader initialized:")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Timeframe: {timeframe}")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Device: {device}")

    def _init_exchange(self):
        """Initialize exchange connection"""
        if self.mode == 'paper':
            # Paper trading - read-only connection
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'
                }
            })
        else:
            # Live trading - requires API keys
            # ⚠️ DANGER: Real money at risk!
            logger.warning("⚠️ LIVE TRADING MODE - REAL MONEY AT RISK!")

            # Load API keys from config (NEVER hardcode!)
            api_key = self.config.get('api_key')
            api_secret = self.config.get('api_secret')

            if not api_key or not api_secret:
                raise ValueError("API keys required for live trading!")

            exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'
                }
            })

        return exchange

    def get_latest_data(self, seq_len: int = 100):
        """Fetch latest candle data"""
        # Fetch historical data
        data = self.collector.fetch_historical(
            symbol=self.symbol,
            timeframe=self.timeframe,
            limit=seq_len + 1000  # Extra for preprocessing
        )

        # Preprocess
        data_normalized = self.preprocessor.fit_transform(data)

        # Get last seq_len candles
        latest_sequence = data_normalized.iloc[-seq_len:].values

        return latest_sequence, data.iloc[-1]

    def run_prediction(self, sequence):
        """Run model prediction"""
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            result = self.model.predict(
                sequence_tensor,
                n_samples=10,
                task_type='forecast'
            )

        # Get next candle prediction
        mean_pred = result['mean'][0, -1, :].cpu().numpy()
        uncertainty = result['std'][0, -1, :].cpu().numpy()

        return mean_pred, uncertainty

    def execute_signal(self, signal: dict, current_candle: dict):
        """Execute trading signal"""
        action = signal['action']

        if action == 'hold':
            return

        logger.info(f"\nSignal: {action.upper()}")
        logger.info(f"  Confidence: {signal['confidence']:.2%}")
        logger.info(f"  Predicted price: ${signal['predicted_price']:.2f}")

        if self.mode == 'paper':
            self._execute_paper_trade(signal, current_candle)
        else:
            self._execute_live_trade(signal, current_candle)

    def _execute_paper_trade(self, signal: dict, current_candle: dict):
        """Execute trade in paper trading mode"""
        price = current_candle['close']

        # Calculate position size
        size = self.risk_manager.calculate_position_size(
            capital=self.paper_capital,
            confidence=signal['confidence'],
            current_price=price,
            method='confidence_weighted'
        )

        # Simulate trade
        logger.info(f"\n📄 PAPER TRADE:")
        logger.info(f"  Side: {signal['action']}")
        logger.info(f"  Price: ${price:.2f}")
        logger.info(f"  Size: {size:.4f} units")
        logger.info(f"  Value: ${price * size:.2f}")

        # Update paper capital (simplified - no fees/slippage)
        # In real implementation, track positions and calculate P&L

    def _execute_live_trade(self, signal: dict, current_candle: dict):
        """
        Execute trade in live mode

        ⚠️ WARNING: THIS WILL PLACE REAL ORDERS!
        """
        logger.warning("⚠️ LIVE TRADE EXECUTION - REAL MONEY!")

        # Get account balance
        balance = self.exchange.fetch_balance()

        # Calculate position size
        capital = balance['USDT']['free']
        price = current_candle['close']

        size = self.risk_manager.calculate_position_size(
            capital=capital,
            confidence=signal['confidence'],
            current_price=price,
            method='confidence_weighted'
        )

        # Create order
        side = 'buy' if signal['action'] == 'long' else 'sell'

        logger.info(f"\n💰 LIVE ORDER:")
        logger.info(f"  Side: {side}")
        logger.info(f"  Price: ${price:.2f}")
        logger.info(f"  Size: {size:.4f}")

        # ⚠️ UNCOMMENT AT YOUR OWN RISK!
        # order = self.exchange.create_market_order(
        #     symbol=self.symbol,
        #     side=side,
        #     amount=size
        # )
        # logger.info(f"Order placed: {order['id']}")

        logger.warning("⚠️ Order execution commented out for safety!")

    def run(self, update_interval: int = 60):
        """
        Main trading loop

        Args:
            update_interval: Seconds between updates (60 = 1 minute)
        """
        logger.info(f"\n{'='*60}")
        logger.info("LIVE TRADING STARTED")
        logger.info(f"{'='*60}")
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Update interval: {update_interval}s")
        logger.info(f"Press Ctrl+C to stop")
        logger.info(f"{'='*60}\n")

        try:
            while True:
                # Get latest data
                sequence, current_candle = self.get_latest_data(seq_len=100)

                # Run prediction
                mean_pred, uncertainty = self.run_prediction(sequence)

                # Generate signal
                signal = self.signal_generator.generate_signal(
                    mean_pred,
                    uncertainty,
                    current_candle
                )

                # Log status
                logger.info(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
                logger.info(f"Price: ${current_candle['close']:.2f}")
                logger.info(f"Prediction: ${mean_pred[3]:.2f} (±${uncertainty[3]:.2f})")
                logger.info(f"Signal: {signal['action'].upper()} ({signal['confidence']:.2%})")

                # Execute signal
                if signal['action'] != 'hold':
                    self.execute_signal(signal, current_candle)

                # Wait for next update
                time.sleep(update_interval)

        except KeyboardInterrupt:
            logger.info(f"\n{'='*60}")
            logger.info("Trading stopped by user")
            logger.info(f"{'='*60}")


def main():
    """Main live trading pipeline"""
    args = parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path('../logs') / f'live_trading_{timestamp}.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(log_file)

    logger.info("=" * 60)
    logger.info("Random Masking Candle Predictor - Live Trading")
    logger.info("=" * 60)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Load model
    logger.info(f"\nLoading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = CandlePredictor(
        input_dim=checkpoint['config'].get('input_dim', 15),
        hidden_dim=checkpoint['config'].get('hidden_dim', 256),
        n_layers=checkpoint['config'].get('n_layers', 6),
        n_heads=checkpoint['config'].get('n_heads', 8),
        ff_dim=checkpoint['config'].get('ff_dim', 1024),
        dropout=checkpoint['config'].get('dropout', 0.1),
        use_uncertainty_head=True
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Model loaded successfully")

    # Create signal generator
    signal_generator = SignalGenerator(
        min_confidence=0.6,
        max_uncertainty_pct=0.02,
        min_price_move_pct=0.001
    )

    # Create risk manager
    from dataclasses import dataclass

    @dataclass
    class TradingConfig:
        max_position_size: float = config['capital']['max_position_size']
        kelly_fraction: float = config['position']['kelly_fraction']
        leverage: float = config['position']['leverage']

    risk_manager = RiskManager(TradingConfig())

    # Create live trader
    trader = LiveTrader(
        model=model,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        config=config,
        symbol=args.symbol,
        timeframe=args.timeframe,
        mode=args.mode,
        device=device
    )

    # Run
    trader.run(update_interval=60)  # Update every minute


if __name__ == '__main__':
    main()
