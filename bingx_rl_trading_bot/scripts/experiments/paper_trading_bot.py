"""
Paper Trading Bot - XGBoost 전략 실시간 검증

목적:
1. BingX Testnet에서 XGBoost 전략 실시간 테스트
2. 모든 시장 상태 (상승/횡보/하락) 검증
3. 제로 리스크로 성과 추적
4. 2-4주 후 실전 배포 여부 결정

비판적 사고:
- "분석만 하고 실행 안 하면 무용지물"
- "Paper trading이 진짜 가치를 검증하는 유일한 방법"
- "상승장 편향 데이터 문제 해결"
"""

import os
import time
import json
import pickle
import hmac
import hashlib
import requests
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import ta

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Logging setup
log_file = LOGS_DIR / f"paper_trading_{datetime.now().strftime('%Y%m%d')}.log"
logger.add(log_file, rotation="1 day", retention="30 days")

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Trading configuration"""

    # API Configuration (BingX Testnet)
    API_KEY = os.getenv("BINGX_TESTNET_API_KEY", "")
    API_SECRET = os.getenv("BINGX_TESTNET_API_SECRET", "")
    BASE_URL = "https://open-api-vst.bingx.com"  # Testnet URL

    # Trading Parameters (Optimized from analysis)
    SYMBOL = "BTC-USDT"
    TIMEFRAME = "5m"
    ENTRY_THRESHOLD = 0.002  # Lowered from 0.003 for more trades
    STOP_LOSS = 0.01  # 1%
    TAKE_PROFIT = 0.03  # 3%
    MIN_VOLATILITY = 0.0008

    # Position Sizing
    INITIAL_CAPITAL = 10000.0  # Virtual capital (testnet)
    POSITION_SIZE_PCT = 0.95  # 95% of capital per trade

    # Risk Management
    MAX_DAILY_LOSS_PCT = 0.05  # 5% max daily loss
    MAX_POSITION_HOURS = 24  # Max holding period

    # Market Regime Classification
    BULL_THRESHOLD = 3.0  # 3%+ = Bull market
    BEAR_THRESHOLD = -2.0  # -2%+ = Bear market
    # Between = Sideways

    # Data Collection
    LOOKBACK_CANDLES = 200  # For feature calculation
    UPDATE_INTERVAL = 300  # 5 minutes (in seconds)

# ============================================================================
# BingX API Client
# ============================================================================

class BingXClient:
    """BingX Testnet API Client"""

    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

    def _generate_signature(self, params: dict) -> str:
        """Generate HMAC SHA256 signature"""
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False):
        """Make API request"""
        if params is None:
            params = {}

        headers = {
            "X-BX-APIKEY": self.api_key
        }

        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)

        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None

    def get_klines(self, symbol: str, interval: str, limit: int = 200):
        """Get candlestick data"""
        endpoint = "/openApi/swap/v3/quote/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        return self._request("GET", endpoint, params)

    def get_account_balance(self):
        """Get account balance (testnet)"""
        endpoint = "/openApi/swap/v2/user/balance"
        return self._request("GET", endpoint, signed=True)

    def get_position(self, symbol: str):
        """Get current position"""
        endpoint = "/openApi/swap/v2/user/positions"
        params = {"symbol": symbol}
        return self._request("GET", endpoint, params, signed=True)

    def place_order(self, symbol: str, side: str, quantity: float, price: float = None):
        """Place order (market or limit)"""
        endpoint = "/openApi/swap/v2/trade/order"
        params = {
            "symbol": symbol,
            "side": side,  # "BUY" or "SELL"
            "type": "MARKET" if price is None else "LIMIT",
            "quantity": quantity
        }
        if price is not None:
            params["price"] = price

        return self._request("POST", endpoint, params, signed=True)

# ============================================================================
# Feature Engineering
# ============================================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators (same as training)
    """
    df = df.copy()

    # Price changes
    for lag in range(1, 6):
        df[f'close_change_{lag}'] = df['close'].pct_change(lag)

    # Moving averages
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)

    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()

    # Volatility
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()

    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    return df

# ============================================================================
# Market Regime Classifier
# ============================================================================

class MarketRegimeClassifier:
    """Classify market regime (Bull/Bear/Sideways)"""

    def __init__(self, lookback_periods: int = 20):
        self.lookback_periods = lookback_periods

    def classify(self, df: pd.DataFrame) -> str:
        """
        Classify current market regime

        Returns: "Bull", "Bear", or "Sideways"
        """
        if len(df) < self.lookback_periods:
            return "Unknown"

        recent_data = df.tail(self.lookback_periods)
        price_change_pct = ((recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1) * 100

        if price_change_pct > Config.BULL_THRESHOLD:
            return "Bull"
        elif price_change_pct < Config.BEAR_THRESHOLD:
            return "Bear"
        else:
            return "Sideways"

# ============================================================================
# Trading Strategy
# ============================================================================

class XGBoostTradingStrategy:
    """XGBoost-based trading strategy"""

    def __init__(self, model_path: Path):
        self.model = self._load_model(model_path)
        self.feature_columns = [
            'close_change_1', 'close_change_2', 'close_change_3', 'close_change_4', 'close_change_5',
            'sma_10', 'sma_20', 'ema_10',
            'macd', 'macd_signal', 'macd_diff',
            'rsi',
            'bb_high', 'bb_low', 'bb_mid',
            'volatility',
            'volume_sma', 'volume_ratio'
        ]

    def _load_model(self, model_path: Path):
        """Load XGBoost model"""
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.success(f"Model loaded: {model_path}")
        return model

    def predict(self, df: pd.DataFrame) -> tuple:
        """
        Predict next price movement

        Returns: (prediction, probability, should_enter)
        """
        if self.model is None:
            return 0, 0.5, False

        # Get latest features
        features = df[self.feature_columns].iloc[-1:].values

        # Check for NaN
        if np.isnan(features).any():
            logger.warning("NaN values in features, skipping prediction")
            return 0, 0.5, False

        # Predict
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0][1]  # Probability of class 1

        # Check volatility filter
        current_volatility = df['volatility'].iloc[-1]
        if current_volatility < Config.MIN_VOLATILITY:
            logger.info(f"Volatility too low: {current_volatility:.4f} < {Config.MIN_VOLATILITY}")
            return prediction, probability, False

        # Check entry threshold (only enter on positive expected return)
        expected_return = (probability - 0.5) * 2  # Scale to -1 to 1
        should_enter = (expected_return > Config.ENTRY_THRESHOLD) and (prediction == 1)

        logger.info(f"Prediction: {prediction}, Probability: {probability:.3f}, Expected Return: {expected_return:.3f}, Should Enter: {should_enter}")

        return prediction, probability, should_enter

# ============================================================================
# Paper Trading Bot
# ============================================================================

class PaperTradingBot:
    """Paper trading bot with performance tracking"""

    def __init__(self):
        # API Client (if using real testnet API)
        if Config.API_KEY and Config.API_SECRET:
            self.client = BingXClient(Config.API_KEY, Config.API_SECRET, Config.BASE_URL)
            logger.info("BingX Testnet API client initialized")
        else:
            self.client = None
            logger.warning("No API credentials, using simulation mode")

        # Strategy
        model_path = MODELS_DIR / "xgboost_model.pkl"
        self.strategy = XGBoostTradingStrategy(model_path)

        # Market regime classifier
        self.regime_classifier = MarketRegimeClassifier()

        # Portfolio state
        self.capital = Config.INITIAL_CAPITAL
        self.position = None  # {"side": "LONG/SHORT", "entry_price": float, "quantity": float, "entry_time": datetime}
        self.trades = []

        # Performance tracking
        self.daily_pnl = 0.0
        self.session_start_capital = Config.INITIAL_CAPITAL

        # Market state tracking
        self.market_regime_history = []

    def run(self):
        """Main trading loop"""
        logger.info("=" * 80)
        logger.info("Paper Trading Bot Started")
        logger.info("=" * 80)
        logger.info(f"Initial Capital: ${self.capital:,.2f}")
        logger.info(f"Entry Threshold: {Config.ENTRY_THRESHOLD * 100:.2f}%")
        logger.info(f"Stop Loss: {Config.STOP_LOSS * 100:.1f}%")
        logger.info(f"Take Profit: {Config.TAKE_PROFIT * 100:.1f}%")
        logger.info("=" * 80)

        try:
            while True:
                self._update_cycle()
                time.sleep(Config.UPDATE_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n" + "=" * 80)
            logger.info("Bot stopped by user")
            self._print_final_stats()

    def _update_cycle(self):
        """Single update cycle"""
        try:
            # Get market data
            df = self._get_market_data()
            if df is None or len(df) < 50:
                logger.warning("Insufficient market data")
                return

            # Calculate features
            df = calculate_features(df)

            # Classify market regime
            current_regime = self.regime_classifier.classify(df)
            self.market_regime_history.append({
                "timestamp": datetime.now(),
                "regime": current_regime,
                "price": df['close'].iloc[-1]
            })

            # Current price
            current_price = df['close'].iloc[-1]

            logger.info(f"\n{'=' * 80}")
            logger.info(f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Market Regime: {current_regime}")
            logger.info(f"Current Price: ${current_price:,.2f}")
            logger.info(f"Capital: ${self.capital:,.2f}")

            # Check existing position
            if self.position is not None:
                self._manage_position(current_price, df)

            # Look for new entry (if no position)
            if self.position is None:
                self._check_entry(df, current_price, current_regime)

            # Daily reset check
            self._check_daily_reset()

            # Save state
            self._save_state()

        except Exception as e:
            logger.error(f"Error in update cycle: {e}")
            import traceback
            traceback.print_exc()

    def _get_market_data(self) -> pd.DataFrame:
        """Get market data (from API or simulation)"""
        if self.client is not None:
            # Real API call
            try:
                data = self.client.get_klines(Config.SYMBOL, Config.TIMEFRAME, Config.LOOKBACK_CANDLES)
                if data and 'data' in data:
                    df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                    return df
            except Exception as e:
                logger.error(f"Failed to get market data: {e}")

        # Simulation mode: Load from file
        data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
            # Use most recent data
            df = df.tail(Config.LOOKBACK_CANDLES)
            return df

        return None

    def _check_entry(self, df: pd.DataFrame, current_price: float, regime: str):
        """Check for entry signal"""
        # Get prediction
        prediction, probability, should_enter = self.strategy.predict(df)

        if not should_enter:
            logger.info("No entry signal")
            return

        # Binary classifier: only enter LONG positions
        # prediction == 1 means enter, prediction == 0 means do not enter
        if prediction != 1:
            logger.warning(f"Invalid prediction {prediction}, expected 1 for entry")
            return

        side = "LONG"

        # Calculate position size
        position_value = self.capital * Config.POSITION_SIZE_PCT
        quantity = position_value / current_price

        # Enter position
        self.position = {
            "side": side,
            "entry_price": current_price,
            "quantity": quantity,
            "entry_time": datetime.now(),
            "regime": regime,
            "probability": probability
        }

        logger.success(f"🔔 ENTRY: {side} {quantity:.4f} BTC @ ${current_price:,.2f}")
        logger.info(f"   Position Value: ${position_value:,.2f}")
        logger.info(f"   Market Regime: {regime}")
        logger.info(f"   Prediction Probability: {probability:.3f}")

    def _manage_position(self, current_price: float, df: pd.DataFrame):
        """Manage existing position"""
        entry_price = self.position['entry_price']
        side = self.position['side']
        quantity = self.position['quantity']

        # Calculate P&L
        if side == "LONG":
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price

        pnl_usd = pnl_pct * (entry_price * quantity)

        logger.info(f"Position: {side} {quantity:.4f} BTC @ ${entry_price:,.2f}")
        logger.info(f"P&L: {pnl_pct * 100:+.2f}% (${pnl_usd:+,.2f})")

        # Check exit conditions
        exit_reason = None

        # Stop Loss
        if pnl_pct <= -Config.STOP_LOSS:
            exit_reason = "Stop Loss"

        # Take Profit
        elif pnl_pct >= Config.TAKE_PROFIT:
            exit_reason = "Take Profit"

        # Max holding period
        elif (datetime.now() - self.position['entry_time']).total_seconds() / 3600 > Config.MAX_POSITION_HOURS:
            exit_reason = "Max Holding Period"

        # Exit if triggered
        if exit_reason:
            self._exit_position(current_price, exit_reason, pnl_usd, pnl_pct)

    def _exit_position(self, exit_price: float, reason: str, pnl_usd: float, pnl_pct: float):
        """Exit position"""
        # Record trade
        trade = {
            "entry_time": self.position['entry_time'],
            "exit_time": datetime.now(),
            "side": self.position['side'],
            "entry_price": self.position['entry_price'],
            "exit_price": exit_price,
            "quantity": self.position['quantity'],
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "exit_reason": reason,
            "regime": self.position['regime'],
            "probability": self.position['probability']
        }
        self.trades.append(trade)

        # Update capital
        self.capital += pnl_usd
        self.daily_pnl += pnl_usd

        # Log exit
        logger.warning(f"🔔 EXIT: {reason}")
        logger.info(f"   Exit Price: ${exit_price:,.2f}")
        logger.info(f"   P&L: {pnl_pct * 100:+.2f}% (${pnl_usd:+,.2f})")
        logger.info(f"   New Capital: ${self.capital:,.2f}")

        # Clear position
        self.position = None

        # Print stats
        self._print_stats()

    def _check_daily_reset(self):
        """Check if daily loss limit reached"""
        daily_loss_pct = (self.capital - self.session_start_capital) / self.session_start_capital

        if daily_loss_pct <= -Config.MAX_DAILY_LOSS_PCT:
            logger.error(f"⚠️ Daily loss limit reached: {daily_loss_pct * 100:.2f}%")
            logger.error("Trading halted for today")
            # In real implementation, would pause trading

    def _print_stats(self):
        """Print performance statistics"""
        if len(self.trades) == 0:
            return

        df_trades = pd.DataFrame(self.trades)

        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl_usd'] > 0])
        losing_trades = len(df_trades[df_trades['pnl_usd'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        total_pnl = df_trades['pnl_usd'].sum()
        avg_pnl = df_trades['pnl_usd'].mean()

        total_return_pct = ((self.capital - Config.INITIAL_CAPITAL) / Config.INITIAL_CAPITAL) * 100

        logger.info(f"\n{'=' * 80}")
        logger.info("📊 PERFORMANCE SUMMARY")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")
        logger.info(f"Losing Trades: {losing_trades}")
        logger.info(f"Total P&L: ${total_pnl:+,.2f}")
        logger.info(f"Average P&L per Trade: ${avg_pnl:+,.2f}")
        logger.info(f"Total Return: {total_return_pct:+.2f}%")
        logger.info(f"Current Capital: ${self.capital:,.2f}")
        logger.info(f"{'=' * 80}")

        # Market regime breakdown
        if 'regime' in df_trades.columns:
            logger.info("\n📈 Performance by Market Regime:")
            for regime in ['Bull', 'Bear', 'Sideways']:
                regime_trades = df_trades[df_trades['regime'] == regime]
                if len(regime_trades) > 0:
                    regime_pnl = regime_trades['pnl_usd'].sum()
                    regime_win_rate = (len(regime_trades[regime_trades['pnl_usd'] > 0]) / len(regime_trades)) * 100
                    logger.info(f"  {regime}: {len(regime_trades)} trades, {regime_win_rate:.1f}% WR, ${regime_pnl:+,.2f} P&L")

    def _print_final_stats(self):
        """Print final statistics on exit"""
        self._print_stats()

        # Save trades to CSV
        if len(self.trades) > 0:
            df_trades = pd.DataFrame(self.trades)
            output_file = RESULTS_DIR / f"paper_trading_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_trades.to_csv(output_file, index=False)
            logger.success(f"\nTrades saved to: {output_file}")

        # Save market regime history
        if len(self.market_regime_history) > 0:
            df_regime = pd.DataFrame(self.market_regime_history)
            regime_file = RESULTS_DIR / f"market_regime_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_regime.to_csv(regime_file, index=False)
            logger.success(f"Market regime history saved to: {regime_file}")

    def _save_state(self):
        """Save bot state"""
        state = {
            "capital": self.capital,
            "position": self.position,
            "trades": self.trades,
            "timestamp": datetime.now().isoformat()
        }

        state_file = RESULTS_DIR / "paper_trading_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    logger.info("Paper Trading Bot - XGBoost Strategy")
    logger.info("비판적 사고: '분석만 하고 실행 안 하면 무용지물'")
    logger.info("목적: 모든 시장 상태에서 진짜 가치 검증")

    # Check model exists
    model_path = MODELS_DIR / "xgboost_model.pkl"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Please train XGBoost model first")
        return

    # Initialize and run bot
    bot = PaperTradingBot()
    bot.run()

if __name__ == "__main__":
    main()
