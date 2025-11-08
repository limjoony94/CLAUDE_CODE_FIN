"""
Sweet-2 Paper Trading Bot - Updated to Phase 4 Base Model

목표: Phase 4 Base Model 실시간 검증
- XGBoost Phase 4 Base Model (37 features, BEST MODEL)
- Statistically validated: 7.68% per 5 days vs Buy & Hold
- Sweet-2 Thresholds: xgb_strong=0.7, xgb_moderate=0.6, tech_strength=0.75
- 5분 캔들 기반

Expected Performance (from statistically validated backtesting):
- vs Buy & Hold: +7.68% per 5 days (~1.26% per 2 days)
- Trade Frequency: ~15 per 5-day window
- Win Rate: 69.1%
- Sharpe Ratio: 11.88
- Max Drawdown: 0.90%
- Statistical Power: 88.3% (confident)

Critical Validation Goals:
- Week 1: Maintain >65% win rate, positive vs B&H
- Week 2: Validate 7%+ returns per 5 days
- Decision: Deploy to live if validated, iterate if partial
"""

import os
import time
import json
import pickle
import requests  # For BingX API calls
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import ta

# Project paths
# Always use parent.parent.parent since file is in scripts/production/
PROJECT_ROOT = Path(__file__).parent.parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Import project modules
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.production.technical_strategy import TechnicalStrategy
from scripts.production.backtest_hybrid_v4 import HybridStrategy

# Create directories
for dir_path in [RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Logging setup
log_file = LOGS_DIR / f"sweet2_paper_trading_{datetime.now().strftime('%Y%m%d')}.log"
logger.add(log_file, rotation="1 day", retention="30 days")

# ============================================================================
# Sweet-2 Configuration
# ============================================================================

class Sweet2Config:
    """Sweet-2 Paper Trading Configuration"""

    # Sweet-2 Thresholds (비판적으로 검증된 최적값)
    XGB_THRESHOLD_STRONG = 0.7
    XGB_THRESHOLD_MODERATE = 0.6
    TECH_STRENGTH_THRESHOLD = 0.75

    # Expected Metrics (Phase 4 Base - statistically validated)
    EXPECTED_TRADES_PER_WEEK = 21.0  # 15 per 5 days = ~21 per week
    EXPECTED_WIN_RATE = 69.1
    EXPECTED_VS_BH = 7.68  # per 5 days
    EXPECTED_PER_TRADE_NET = 0.512  # 7.68% / 15 trades

    # Targets (최소 달성 목표 - Phase 4 Base baseline)
    TARGET_TRADES_PER_WEEK = (14.0, 28.0)  # 10-20 per 5 days = 14-28 per week
    TARGET_WIN_RATE = 60.0  # minimum (69.1% expected)
    TARGET_VS_BH = 5.0  # minimum 5% per 5 days (7.68% expected)
    TARGET_PER_TRADE_NET = 0.35  # minimum (0.512% expected)

    # API Configuration (BingX)
    API_KEY = os.getenv("BINGX_API_KEY", "")
    API_SECRET = os.getenv("BINGX_API_SECRET", "")
    BASE_URL = "https://open-api.bingx.com"
    USE_TESTNET = os.getenv("BINGX_USE_TESTNET", "true").lower() == "true"

    # Trading Parameters
    SYMBOL = "BTC-USDT"
    TIMEFRAME = "5m"
    STOP_LOSS = 0.01  # 1%
    TAKE_PROFIT = 0.03  # 3%
    MAX_HOLDING_HOURS = 4  # Based on backtesting

    # Position Sizing
    INITIAL_CAPITAL = 10000.0  # Virtual capital
    POSITION_SIZE_PCT = 0.95  # 95% of capital per trade

    # Risk Management
    MAX_DAILY_LOSS_PCT = 0.05  # 5% max daily loss
    TRANSACTION_COST = 0.0006  # 0.06% taker + 0.06% maker = 0.12% per round trip

    # Data Collection
    LOOKBACK_CANDLES = 500  # For feature calculation (increased for Advanced Features stability)
    UPDATE_INTERVAL = 300  # 5 minutes (in seconds)

    # Market Regime Classification
    BULL_THRESHOLD = 3.0  # 3%+ = Bull market
    BEAR_THRESHOLD = -2.0  # -2%+ = Bear market

# ============================================================================
# Sweet-2 Paper Trading Bot
# ============================================================================

class Sweet2PaperTradingBot:
    """
    Sweet-2 Paper Trading Bot

    Validates Sweet-2 configuration in real-time with 5-minute candles
    """

    def __init__(self):
        # Load XGBoost Phase 4 Base model (BEST MODEL - 37 features, 7.68% returns per 5 days)
        model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
        feature_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"

        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost Phase 4 Base model not found: {model_path}")

        with open(model_path, 'rb') as f:
            self.xgboost_model = pickle.load(f)

        with open(feature_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]

        logger.success(f"✅ XGBoost Phase 4 Base model loaded: {len(self.feature_columns)} features")

        # Initialize Advanced Technical Features (for Phase 4)
        self.adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
        logger.success("✅ Advanced Technical Features initialized")

        # Initialize Technical Strategy
        self.technical_strategy = TechnicalStrategy()
        logger.success("✅ Technical Strategy initialized")

        # Initialize Hybrid Strategy with Sweet-2 thresholds
        self.hybrid_strategy = HybridStrategy(
            xgboost_model=self.xgboost_model,
            feature_columns=self.feature_columns,
            technical_strategy=self.technical_strategy,
            xgb_threshold_strong=Sweet2Config.XGB_THRESHOLD_STRONG,
            xgb_threshold_moderate=Sweet2Config.XGB_THRESHOLD_MODERATE,
            tech_strength_threshold=Sweet2Config.TECH_STRENGTH_THRESHOLD
        )
        logger.success(f"✅ Sweet-2 Hybrid Strategy initialized")
        logger.info(f"   XGB Strong: {Sweet2Config.XGB_THRESHOLD_STRONG}")
        logger.info(f"   XGB Moderate: {Sweet2Config.XGB_THRESHOLD_MODERATE}")
        logger.info(f"   Tech Strength: {Sweet2Config.TECH_STRENGTH_THRESHOLD}")

        # Portfolio state
        self.capital = Sweet2Config.INITIAL_CAPITAL
        self.initial_capital = Sweet2Config.INITIAL_CAPITAL
        self.position = None
        self.trades = []

        # Buy & Hold tracking
        self.bh_btc_quantity = 0.0
        self.bh_entry_price = 0.0
        self.bh_initialized = False

        # Performance tracking
        self.session_start = datetime.now()
        self.session_start_capital = Sweet2Config.INITIAL_CAPITAL

        # Market state tracking
        self.market_regime_history = []

        logger.info("=" * 80)
        logger.info("Sweet-2 Paper Trading Bot Initialized")
        logger.info("=" * 80)
        network = "TESTNET ✅" if Sweet2Config.USE_TESTNET else "MAINNET ⚠️"
        logger.info(f"Network: BingX {network}")
        logger.info(f"Initial Capital: ${self.capital:,.2f} (Virtual/Paper Trading)")
        logger.info(f"Expected Performance:")
        logger.info(f"  - vs B&H: +{Sweet2Config.EXPECTED_VS_BH:.2f}%")
        logger.info(f"  - Win Rate: {Sweet2Config.EXPECTED_WIN_RATE:.1f}%")
        logger.info(f"  - Trades/Week: {Sweet2Config.EXPECTED_TRADES_PER_WEEK:.1f}")
        logger.info(f"  - Per-trade Net: +{Sweet2Config.EXPECTED_PER_TRADE_NET:.3f}%")
        logger.info("=" * 80)

    def run(self):
        """Main trading loop"""
        logger.info("🚀 Starting Sweet-2 Paper Trading...")
        logger.info(f"Update Interval: {Sweet2Config.UPDATE_INTERVAL}s (5 minutes)")

        try:
            while True:
                self._update_cycle()
                time.sleep(Sweet2Config.UPDATE_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n" + "=" * 80)
            logger.info("Bot stopped by user")
            self._print_final_stats()

    def _update_cycle(self):
        """Single update cycle (every 5 minutes)"""
        try:
            # Get market data
            df = self._get_market_data()
            if df is None or len(df) < Sweet2Config.LOOKBACK_CANDLES:
                logger.warning("Insufficient market data")
                return

            # Calculate XGBoost baseline features
            df = calculate_features(df)

            # Calculate advanced technical features (Phase 4)
            df = self.adv_features.calculate_all_features(df)

            # Calculate technical indicators
            df = self.technical_strategy.calculate_indicators(df)

            # Handle NaN values (forward fill for indicator stabilization)
            rows_before = len(df)
            df = df.ffill()  # Forward fill for initial periods
            df = df.dropna()  # Drop remaining NaN (if any)
            rows_after = len(df)

            logger.info(f"Data rows: {rows_before} → {rows_after} after NaN handling")

            if len(df) < 50:
                logger.warning(f"Too few rows after NaN handling ({len(df)} < 50)")
                logger.warning("Waiting for more data to stabilize indicators...")
                return

            # Current price
            current_idx = len(df) - 1
            current_price = df['close'].iloc[current_idx]

            # Initialize Buy & Hold if first run
            if not self.bh_initialized:
                self._initialize_buy_hold(current_price)

            # Classify market regime
            current_regime = self._classify_market_regime(df)
            self.market_regime_history.append({
                "timestamp": datetime.now(),
                "regime": current_regime,
                "price": current_price
            })

            logger.info(f"\n{'=' * 80}")
            logger.info(f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Market Regime: {current_regime}")
            logger.info(f"Current Price: ${current_price:,.2f}")
            logger.info(f"Strategy Capital: ${self.capital:,.2f}")

            # Manage existing position
            if self.position is not None:
                self._manage_position(current_price, df, current_idx)

            # Look for new entry (if no position)
            if self.position is None:
                self._check_entry(df, current_idx, current_price, current_regime)

            # Print weekly stats (with current API price for accurate B&H comparison)
            self._print_stats(current_price=current_price)

            # Save state
            self._save_state()

        except Exception as e:
            logger.error(f"Error in update cycle: {e}")
            import traceback
            traceback.print_exc()

    def _get_market_data(self):
        """Get market data (live API or simulation)"""
        # Try to get real-time data from BingX API first
        try:
            # Select API URL based on testnet flag
            if Sweet2Config.USE_TESTNET:
                base_url = "https://open-api-vst.bingx.com"  # Testnet
                logger.debug("Using BingX Testnet API")
            else:
                base_url = Sweet2Config.BASE_URL  # Mainnet
                logger.debug("Using BingX Mainnet API")

            url = f"{base_url}/openApi/swap/v3/quote/klines"
            params = {
                "symbol": "BTC-USDT",
                "interval": "5m",
                "limit": min(Sweet2Config.LOOKBACK_CANDLES, 500)  # BingX API limit is 500
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data.get('code') == 0 and 'data' in data:
                    klines = data['data']

                    # Parse to DataFrame (BingX returns list of dicts with keys: open, high, low, close, volume, time)
                    df = pd.DataFrame(klines)

                    # Rename 'time' to 'timestamp' and convert to datetime
                    df = df.rename(columns={'time': 'timestamp'})
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                    # Convert types (BingX returns strings)
                    df[['open', 'high', 'low', 'close', 'volume']] = \
                        df[['open', 'high', 'low', 'close', 'volume']].astype(float)

                    # Reorder columns to match expected format
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

                    # CRITICAL FIX: BingX API returns candles in REVERSE order (newest first)
                    # Sort by timestamp to ensure chronological order (oldest → newest)
                    df = df.sort_values('timestamp').reset_index(drop=True)

                    network = "TESTNET" if Sweet2Config.USE_TESTNET else "MAINNET"
                    latest_price = df['close'].iloc[-1]  # Now correctly gets the LATEST candle
                    latest_time = df['timestamp'].iloc[-1]
                    logger.info(f"✅ Live data from BingX API ({network}): {len(df)} candles")
                    logger.info(f"   Latest: ${latest_price:,.2f} @ {latest_time.strftime('%Y-%m-%d %H:%M')}")
                    return df

        except Exception as e:
            logger.warning(f"⚠️ Failed to get live data from API: {e}")
            logger.info("Falling back to simulation mode (file data)")

        # Fallback: simulation mode (from file)
        data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
            # Use most recent data
            df = df.tail(Sweet2Config.LOOKBACK_CANDLES)
            logger.info(f"📁 Simulation data from file: {len(df)} candles")
            return df

        logger.error("❌ No market data available (API failed and file not found)")
        return None

    def _initialize_buy_hold(self, current_price):
        """Initialize Buy & Hold baseline"""
        self.bh_entry_price = current_price
        self.bh_btc_quantity = self.initial_capital / current_price
        self.bh_initialized = True

        logger.success(f"📊 Buy & Hold Baseline Initialized:")
        logger.info(f"   Bought {self.bh_btc_quantity:.6f} BTC @ ${current_price:,.2f}")
        logger.info(f"   Initial Value: ${self.initial_capital:,.2f}")

    def _classify_market_regime(self, df):
        """Classify current market regime"""
        # Use last 20 candles (100 minutes)
        lookback = 20
        if len(df) < lookback:
            return "Unknown"

        recent_data = df.tail(lookback)
        start_price = recent_data['close'].iloc[0]
        end_price = recent_data['close'].iloc[-1]
        price_change_pct = ((end_price / start_price) - 1) * 100

        if price_change_pct > Sweet2Config.BULL_THRESHOLD:
            return "Bull"
        elif price_change_pct < Sweet2Config.BEAR_THRESHOLD:
            return "Bear"
        else:
            return "Sideways"

    def _check_entry(self, df, idx, current_price, regime):
        """Check for entry signal using Sweet-2 Hybrid Strategy"""
        # Get Hybrid Strategy signal
        should_enter, confidence, xgb_prob, tech_signal, tech_strength = \
            self.hybrid_strategy.should_enter(df, idx)

        logger.info(f"Signal Check:")
        logger.info(f"  XGBoost Prob: {xgb_prob:.3f}")
        logger.info(f"  Tech Signal: {tech_signal} (strength: {tech_strength:.3f})")
        logger.info(f"  Should Enter: {should_enter} ({confidence if confidence else 'N/A'})")

        if not should_enter:
            return

        # Calculate position size
        position_value = self.capital * Sweet2Config.POSITION_SIZE_PCT
        quantity = position_value / current_price

        # Enter position
        self.position = {
            "entry_idx": idx,
            "entry_price": current_price,
            "quantity": quantity,
            "entry_time": datetime.now(),
            "regime": regime,
            "xgb_prob": xgb_prob,
            "tech_signal": tech_signal,
            "tech_strength": tech_strength,
            "confidence": confidence
        }

        logger.success(f"🔔 ENTRY: LONG {quantity:.4f} BTC @ ${current_price:,.2f}")
        logger.info(f"   Position Value: ${position_value:,.2f}")
        logger.info(f"   Confidence: {confidence}")
        logger.info(f"   Market Regime: {regime}")
        logger.info(f"   XGBoost Prob: {xgb_prob:.3f}")
        logger.info(f"   Tech Strength: {tech_strength:.3f}")

    def _manage_position(self, current_price, df, current_idx):
        """Manage existing position"""
        entry_price = self.position['entry_price']
        entry_idx = self.position['entry_idx']
        quantity = self.position['quantity']

        # Calculate P&L
        pnl_pct = (current_price - entry_price) / entry_price
        pnl_usd = pnl_pct * (entry_price * quantity)

        # Calculate holding time
        hours_held = (current_idx - entry_idx) / 12  # 5 min candles -> 12 per hour

        logger.info(f"Position: LONG {quantity:.4f} BTC @ ${entry_price:,.2f}")
        logger.info(f"P&L: {pnl_pct * 100:+.2f}% (${pnl_usd:+,.2f})")
        logger.info(f"Holding: {hours_held:.1f} hours")

        # Check exit conditions
        exit_reason = None

        # Stop Loss
        if pnl_pct <= -Sweet2Config.STOP_LOSS:
            exit_reason = "Stop Loss"

        # Take Profit
        elif pnl_pct >= Sweet2Config.TAKE_PROFIT:
            exit_reason = "Take Profit"

        # Max holding period
        elif hours_held >= Sweet2Config.MAX_HOLDING_HOURS:
            exit_reason = "Max Holding"

        # Exit if triggered
        if exit_reason:
            self._exit_position(current_price, exit_reason, pnl_usd, pnl_pct)

    def _exit_position(self, exit_price, reason, pnl_usd, pnl_pct):
        """Exit position"""
        # Calculate transaction costs
        entry_cost = self.position['entry_price'] * self.position['quantity'] * Sweet2Config.TRANSACTION_COST
        exit_cost = exit_price * self.position['quantity'] * Sweet2Config.TRANSACTION_COST
        total_cost = entry_cost + exit_cost

        # Net P&L after costs
        net_pnl_usd = pnl_usd - total_cost

        # Record trade
        trade = {
            "entry_time": self.position['entry_time'],
            "exit_time": datetime.now(),
            "entry_price": self.position['entry_price'],
            "exit_price": exit_price,
            "quantity": self.position['quantity'],
            "pnl_pct": pnl_pct,
            "pnl_usd_gross": pnl_usd,
            "transaction_cost": total_cost,
            "pnl_usd_net": net_pnl_usd,
            "exit_reason": reason,
            "regime": self.position['regime'],
            "xgb_prob": self.position['xgb_prob'],
            "tech_signal": self.position['tech_signal'],
            "tech_strength": self.position['tech_strength'],
            "confidence": self.position['confidence']
        }
        self.trades.append(trade)

        # Update capital
        self.capital += net_pnl_usd

        # Log exit
        logger.warning(f"🔔 EXIT: {reason}")
        logger.info(f"   Exit Price: ${exit_price:,.2f}")
        logger.info(f"   Gross P&L: {pnl_pct * 100:+.2f}% (${pnl_usd:+,.2f})")
        logger.info(f"   Transaction Cost: ${total_cost:.2f}")
        logger.info(f"   Net P&L: ${net_pnl_usd:+,.2f}")
        logger.info(f"   New Capital: ${self.capital:,.2f}")

        # Clear position
        self.position = None

        # Print stats after each trade (exit_price is current price)
        self._print_stats(current_price=exit_price)

    def _print_stats(self, current_price=None):
        """Print current performance statistics"""
        if len(self.trades) == 0:
            logger.info(f"\n📊 No trades yet")
            return

        df_trades = pd.DataFrame(self.trades)

        # Basic stats
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl_usd_net'] > 0])
        losing_trades = len(df_trades[df_trades['pnl_usd_net'] <= 0])
        win_rate = (winning_trades / total_trades) * 100

        # Returns
        total_net_pnl = df_trades['pnl_usd_net'].sum()
        total_return_pct = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        avg_pnl_per_trade = total_net_pnl / total_trades

        # Per-trade net profit percentage
        avg_net_pnl_pct = (avg_pnl_per_trade / (self.initial_capital * Sweet2Config.POSITION_SIZE_PCT)) * 100

        # Buy & Hold comparison - FIXED: Use current API price, not old file price!
        if self.bh_initialized and current_price is not None:
            current_btc_price = current_price
            bh_value = self.bh_btc_quantity * current_btc_price
            bh_return_pct = ((bh_value - self.initial_capital) / self.initial_capital) * 100
            vs_bh = total_return_pct - bh_return_pct
        else:
            bh_return_pct = 0.0
            vs_bh = 0.0
            current_btc_price = 0.0

        # Time-based metrics
        days_running = (datetime.now() - self.session_start).total_seconds() / 86400
        trades_per_week = (total_trades / days_running) * 7 if days_running > 0 else 0

        logger.info(f"\n{'=' * 80}")
        logger.info("📊 SWEET-2 PERFORMANCE SUMMARY")
        logger.info(f"{'=' * 80}")
        logger.info(f"Session Duration: {days_running:.1f} days")
        logger.info(f"")
        logger.info(f"Trading Performance:")
        logger.info(f"  Total Trades: {total_trades}")
        logger.info(f"  Winning: {winning_trades} ({win_rate:.1f}%) {'✅' if win_rate >= Sweet2Config.TARGET_WIN_RATE else '⚠️'}")
        logger.info(f"  Losing: {losing_trades}")
        logger.info(f"  Trades/Week: {trades_per_week:.1f} {'✅' if Sweet2Config.TARGET_TRADES_PER_WEEK[0] <= trades_per_week <= Sweet2Config.TARGET_TRADES_PER_WEEK[1] else '⚠️'}")
        logger.info(f"")
        logger.info(f"Returns:")
        logger.info(f"  Total Net P&L: ${total_net_pnl:+,.2f}")
        logger.info(f"  Total Return: {total_return_pct:+.2f}%")
        logger.info(f"  Per-trade Net: {avg_net_pnl_pct:+.3f}% {'✅' if avg_net_pnl_pct > 0 else '❌'}")
        logger.info(f"  Current Capital: ${self.capital:,.2f}")
        logger.info(f"")
        logger.info(f"vs Buy & Hold:")
        logger.info(f"  B&H Return: {bh_return_pct:+.2f}% (BTC @ ${current_btc_price:,.2f})")
        logger.info(f"  Strategy Return: {total_return_pct:+.2f}%")
        logger.info(f"  Difference: {vs_bh:+.2f}% {'✅' if vs_bh > 0 else '⚠️'}")
        logger.info(f"")
        logger.info(f"Sweet-2 Targets:")
        logger.info(f"  Win Rate: {win_rate:.1f}% (target: >{Sweet2Config.TARGET_WIN_RATE:.0f}%)")
        logger.info(f"  Trades/Week: {trades_per_week:.1f} (target: {Sweet2Config.TARGET_TRADES_PER_WEEK[0]:.0f}-{Sweet2Config.TARGET_TRADES_PER_WEEK[1]:.0f})")
        logger.info(f"  vs B&H: {vs_bh:+.2f}% (target: >{Sweet2Config.TARGET_VS_BH:.0f}%)")
        logger.info(f"  Per-trade Net: {avg_net_pnl_pct:+.3f}% (target: >{Sweet2Config.TARGET_PER_TRADE_NET:.0f}%)")
        logger.info(f"{'=' * 80}")

        # Regime breakdown
        if 'regime' in df_trades.columns:
            logger.info(f"\n📈 Performance by Market Regime:")
            for regime in ['Bull', 'Bear', 'Sideways']:
                regime_trades = df_trades[df_trades['regime'] == regime]
                if len(regime_trades) > 0:
                    regime_pnl = regime_trades['pnl_usd_net'].sum()
                    regime_win_rate = (len(regime_trades[regime_trades['pnl_usd_net'] > 0]) / len(regime_trades)) * 100
                    logger.info(f"  {regime}: {len(regime_trades)} trades, {regime_win_rate:.1f}% WR, ${regime_pnl:+,.2f} P&L")

    def _print_final_stats(self):
        """Print final statistics on exit"""
        self._print_stats()

        # Save trades to CSV
        if len(self.trades) > 0:
            df_trades = pd.DataFrame(self.trades)
            output_file = RESULTS_DIR / f"sweet2_paper_trading_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_trades.to_csv(output_file, index=False)
            logger.success(f"\n✅ Trades saved to: {output_file}")

        # Save market regime history
        if len(self.market_regime_history) > 0:
            df_regime = pd.DataFrame(self.market_regime_history)
            regime_file = RESULTS_DIR / f"sweet2_market_regime_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_regime.to_csv(regime_file, index=False)
            logger.success(f"✅ Market regime history saved to: {regime_file}")

        # Final verdict
        if len(self.trades) >= 10:
            df_trades = pd.DataFrame(self.trades)
            win_rate = (len(df_trades[df_trades['pnl_usd_net'] > 0]) / len(df_trades)) * 100
            total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100

            logger.info(f"\n{'=' * 80}")
            logger.info("🎯 SWEET-2 VALIDATION VERDICT")
            logger.info(f"{'=' * 80}")

            if win_rate >= Sweet2Config.TARGET_WIN_RATE and total_return > 0:
                logger.success("✅ SWEET-2 VALIDATION SUCCESSFUL!")
                logger.info("   Proceed to small live deployment (3-5% capital)")
            elif win_rate >= 50 and total_return >= -0.5:
                logger.warning("⚠️ PARTIAL SUCCESS")
                logger.info("   Continue paper trading for 1 more week or")
                logger.info("   Implement regime-specific thresholds")
            else:
                logger.error("❌ VALIDATION FAILED")
                logger.info("   Implement 15-minute features or")
                logger.info("   Review and adjust Sweet-2 thresholds")

            logger.info(f"{'=' * 80}")

    def _save_state(self):
        """Save bot state"""
        state = {
            "capital": self.capital,
            "position": self.position,
            "trades_count": len(self.trades),
            "bh_btc_quantity": self.bh_btc_quantity,
            "bh_entry_price": self.bh_entry_price,
            "session_start": self.session_start.isoformat(),
            "timestamp": datetime.now().isoformat()
        }

        state_file = RESULTS_DIR / "sweet2_paper_trading_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    logger.info("=" * 80)
    logger.info("Sweet-2 Paper Trading Bot - Phase 4 Base Model")
    logger.info("비판적 사고: '통계적으로 검증된 최고의 모델을 실시간으로 테스트'")
    logger.info("=" * 80)

    # Check model exists
    model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
    if not model_path.exists():
        logger.error(f"XGBoost Phase 4 Base model not found: {model_path}")
        logger.error("Please train XGBoost Phase 4 model first")
        return

    # Initialize and run bot
    try:
        bot = Sweet2PaperTradingBot()
        bot.run()
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
