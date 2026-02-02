"""
Sweet-2 Leverage 2x Paper Trading Bot

목표: 0.5%/day 수익 달성 (레버리지 2배)
- Sweet-2 Configuration + 2x Leverage
- Expected: 0.46%/day (168%/year)
- Risk: 청산 리스크 존재, Stop Loss 필수

⚠️ WARNING: 레버리지는 수익과 손실을 모두 증폭시킵니다
- 이익 2배 = 손실도 2배
- Stop Loss 엄격히 준수 필요
"""

import os
import time
import json
import pickle
import requests
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import ta

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Import project modules
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.technical_strategy import TechnicalStrategy
from scripts.production.backtest_hybrid_v4 import HybridStrategy

# Create directories
for dir_path in [RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Logging setup
log_file = LOGS_DIR / f"sweet2_leverage_2x_{datetime.now().strftime('%Y%m%d')}.log"
logger.add(log_file, rotation="1 day", retention="30 days")

# ============================================================================
# Sweet-2 Leverage 2x Configuration
# ============================================================================

class Sweet2Leverage2xConfig:
    """Sweet-2 Leverage 2x Paper Trading Configuration"""

    # ⚡ LEVERAGE SETTING ⚡
    LEVERAGE = 2.0  # 2x leverage for 0.46%/day target

    # Sweet-2 Thresholds (검증된 최적값)
    XGB_THRESHOLD_STRONG = 0.7
    XGB_THRESHOLD_MODERATE = 0.6
    TECH_STRENGTH_THRESHOLD = 0.75

    # Expected Metrics (with 2x leverage)
    EXPECTED_DAILY_RETURN = 0.46  # 0.230% * 2 = 0.46%
    EXPECTED_ANNUAL_RETURN = 168.0  # 84% * 2 = 168%
    EXPECTED_TRADES_PER_WEEK = 2.5
    EXPECTED_WIN_RATE = 54.3

    # Targets
    TARGET_DAILY_RETURN = 0.4  # minimum 0.4%/day
    TARGET_WIN_RATE = 52.0
    TARGET_VS_BH = 0.0

    # API Configuration
    API_KEY = os.getenv("BINGX_API_KEY", "")
    API_SECRET = os.getenv("BINGX_API_SECRET", "")
    BASE_URL = "https://open-api.bingx.com"
    USE_TESTNET = os.getenv("BINGX_USE_TESTNET", "true").lower() == "true"

    # Trading Parameters
    SYMBOL = "BTC-USDT"
    TIMEFRAME = "5m"

    # ⚠️ LEVERAGE RISK MANAGEMENT ⚠️
    STOP_LOSS = 0.005  # 0.5% (tighter due to 2x leverage)
    TAKE_PROFIT = 0.02  # 2% (lower target, faster exits)
    MAX_HOLDING_HOURS = 4

    # Liquidation protection
    LIQUIDATION_THRESHOLD = 0.50  # 50% loss = liquidation (2x leverage)
    EMERGENCY_STOP_LOSS = 0.01  # 1% emergency exit

    # Position Sizing (with leverage)
    INITIAL_CAPITAL = 10000.0
    POSITION_SIZE_PCT = 0.95  # 95% of capital per trade
    # Actual exposure = INITIAL_CAPITAL * POSITION_SIZE_PCT * LEVERAGE

    # Risk Management
    MAX_DAILY_LOSS_PCT = 0.03  # 3% max daily loss (tighter with leverage)
    MAX_CONSECUTIVE_LOSSES = 3  # Stop after 3 consecutive losses

    # Transaction Cost (Maker fees)
    TRANSACTION_COST = 0.0002  # 0.02% maker fee (using limit orders)

    # Data Collection
    LOOKBACK_CANDLES = 300
    UPDATE_INTERVAL = 300  # 5 minutes

    # Market Regime
    BULL_THRESHOLD = 3.0
    BEAR_THRESHOLD = -2.0

# ============================================================================
# Sweet-2 Leverage 2x Paper Trading Bot
# ============================================================================

class Sweet2Leverage2xBot:
    """
    Sweet-2 Leverage 2x Paper Trading Bot

    목표: 0.46%/day with 2x leverage
    ⚠️ 리스크: 청산 가능성, 손실 2배 증폭
    """

    def __init__(self):
        # Load XGBoost Phase 2 model
        model_path = MODELS_DIR / "xgboost_v3_lookahead3_thresh1_phase2.pkl"
        feature_path = MODELS_DIR / "xgboost_v3_lookahead3_thresh1_phase2_features.txt"

        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost Phase 2 model not found: {model_path}")

        with open(model_path, 'rb') as f:
            self.xgboost_model = pickle.load(f)

        with open(feature_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]

        logger.success(f"✅ XGBoost Phase 2 model loaded: {len(self.feature_columns)} features")

        # Initialize strategies
        self.technical_strategy = TechnicalStrategy()
        logger.success("✅ Technical Strategy initialized")

        self.hybrid_strategy = HybridStrategy(
            xgboost_model=self.xgboost_model,
            feature_columns=self.feature_columns,
            technical_strategy=self.technical_strategy,
            xgb_threshold_strong=Sweet2Leverage2xConfig.XGB_THRESHOLD_STRONG,
            xgb_threshold_moderate=Sweet2Leverage2xConfig.XGB_THRESHOLD_MODERATE,
            tech_strength_threshold=Sweet2Leverage2xConfig.TECH_STRENGTH_THRESHOLD
        )
        logger.success(f"✅ Sweet-2 Hybrid Strategy initialized")

        # Portfolio state
        self.capital = Sweet2Leverage2xConfig.INITIAL_CAPITAL
        self.initial_capital = Sweet2Leverage2xConfig.INITIAL_CAPITAL
        self.position = None
        self.trades = []

        # Risk tracking
        self.consecutive_losses = 0
        self.daily_loss = 0.0
        self.last_reset_date = datetime.now().date()

        # Buy & Hold tracking
        self.bh_btc_quantity = 0.0
        self.bh_entry_price = 0.0
        self.bh_initialized = False

        # Performance tracking
        self.session_start = datetime.now()
        self.market_regime_history = []

        logger.info("=" * 80)
        logger.info("⚡ Sweet-2 LEVERAGE 2x Paper Trading Bot Initialized ⚡")
        logger.info("=" * 80)
        logger.warning(f"⚠️ LEVERAGE: {Sweet2Leverage2xConfig.LEVERAGE}x")
        logger.warning(f"⚠️ RISK: Profits AND losses are DOUBLED")
        logger.warning(f"⚠️ Liquidation at: {Sweet2Leverage2xConfig.LIQUIDATION_THRESHOLD * 100}% loss")
        logger.info(f"Initial Capital: ${self.capital:,.2f}")
        logger.info(f"Expected Performance (2x leverage):")
        logger.info(f"  - Daily Return: {Sweet2Leverage2xConfig.EXPECTED_DAILY_RETURN}%")
        logger.info(f"  - Annual Return: {Sweet2Leverage2xConfig.EXPECTED_ANNUAL_RETURN}%")
        logger.info(f"  - Trades/Week: {Sweet2Leverage2xConfig.EXPECTED_TRADES_PER_WEEK:.1f}")
        logger.info(f"Risk Management:")
        logger.info(f"  - Stop Loss: {Sweet2Leverage2xConfig.STOP_LOSS * 100}%")
        logger.info(f"  - Take Profit: {Sweet2Leverage2xConfig.TAKE_PROFIT * 100}%")
        logger.info(f"  - Max Daily Loss: {Sweet2Leverage2xConfig.MAX_DAILY_LOSS_PCT * 100}%")
        logger.info(f"  - Max Consecutive Losses: {Sweet2Leverage2xConfig.MAX_CONSECUTIVE_LOSSES}")
        logger.info("=" * 80)

    def run(self):
        """Main trading loop"""
        logger.info("🚀 Starting Sweet-2 Leverage 2x Paper Trading...")
        logger.info(f"Update Interval: {Sweet2Leverage2xConfig.UPDATE_INTERVAL}s (5 minutes)")

        try:
            while True:
                self._update_cycle()
                time.sleep(Sweet2Leverage2xConfig.UPDATE_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n" + "=" * 80)
            logger.info("Bot stopped by user")
            self._print_final_stats()

    def _update_cycle(self):
        """Single update cycle (every 5 minutes)"""
        try:
            # Reset daily loss counter
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_loss = 0.0
                self.last_reset_date = current_date
                logger.info("📅 New trading day - daily loss counter reset")

            # Check if daily loss limit reached
            if self.daily_loss >= Sweet2Leverage2xConfig.MAX_DAILY_LOSS_PCT * self.initial_capital:
                logger.error(f"🚨 Daily loss limit reached: ${self.daily_loss:.2f}")
                logger.error("⏸️ Trading paused for today")
                return

            # Check if consecutive losses limit reached
            if self.consecutive_losses >= Sweet2Leverage2xConfig.MAX_CONSECUTIVE_LOSSES:
                logger.error(f"🚨 Consecutive losses limit reached: {self.consecutive_losses}")
                logger.error("⏸️ Please review strategy and restart manually")
                return

            # Get market data
            df = self._get_market_data()
            if df is None or len(df) < Sweet2Leverage2xConfig.LOOKBACK_CANDLES:
                logger.warning("Insufficient market data")
                return

            # Calculate features
            df = calculate_features(df)
            df = self.technical_strategy.calculate_indicators(df)

            # Handle NaN
            rows_before = len(df)
            df = df.ffill()
            df = df.dropna()
            rows_after = len(df)

            logger.info(f"Data rows: {rows_before} → {rows_after} after NaN handling")

            if len(df) < 50:
                logger.warning(f"Too few rows after NaN handling ({len(df)} < 50)")
                return

            # Current state
            current_idx = len(df) - 1
            current_price = df['close'].iloc[current_idx]

            # Initialize Buy & Hold
            if not self.bh_initialized:
                self._initialize_buy_hold(current_price)

            # Market regime
            current_regime = self._classify_market_regime(df)
            self.market_regime_history.append({
                "timestamp": datetime.now(),
                "regime": current_regime,
                "price": current_price
            })

            logger.info(f"\n{'=' * 80}")
            logger.info(f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"⚡ LEVERAGE: {Sweet2Leverage2xConfig.LEVERAGE}x")
            logger.info(f"Market Regime: {current_regime}")
            logger.info(f"Current Price: ${current_price:,.2f}")
            logger.info(f"Capital: ${self.capital:,.2f}")
            logger.info(f"Daily Loss: ${self.daily_loss:.2f} / ${Sweet2Leverage2xConfig.MAX_DAILY_LOSS_PCT * self.initial_capital:.2f}")
            logger.info(f"Consecutive Losses: {self.consecutive_losses} / {Sweet2Leverage2xConfig.MAX_CONSECUTIVE_LOSSES}")

            # Manage position
            if self.position is not None:
                self._manage_position(current_price, df, current_idx)

            # Check entry
            if self.position is None:
                self._check_entry(df, current_idx, current_price, current_regime)

            # Print stats
            self._print_stats()

            # Save state
            self._save_state()

        except Exception as e:
            logger.error(f"Error in update cycle: {e}")
            import traceback
            traceback.print_exc()

    def _get_market_data(self):
        """Get market data from BingX API"""
        try:
            url = "https://open-api.bingx.com/openApi/swap/v3/quote/klines"
            params = {
                "symbol": "BTC-USDT",
                "interval": "5m",
                "limit": min(Sweet2Leverage2xConfig.LOOKBACK_CANDLES, 500)
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data.get('code') == 0 and 'data' in data:
                    klines = data['data']

                    df = pd.DataFrame(klines)
                    df = df.rename(columns={'time': 'timestamp'})
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df[['open', 'high', 'low', 'close', 'volume']] = \
                        df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

                    logger.info(f"✅ Live data from BingX API: {len(df)} candles, Latest: ${df['close'].iloc[-1]:,.2f}")
                    return df

        except Exception as e:
            logger.warning(f"⚠️ Failed to get live data: {e}")

        # Fallback
        data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
            df = df.tail(Sweet2Leverage2xConfig.LOOKBACK_CANDLES)
            logger.info(f"📁 Simulation data: {len(df)} candles")
            return df

        return None

    def _initialize_buy_hold(self, current_price):
        """Initialize Buy & Hold baseline"""
        self.bh_entry_price = current_price
        self.bh_btc_quantity = self.initial_capital / current_price
        self.bh_initialized = True

        logger.success(f"📊 Buy & Hold Baseline Initialized:")
        logger.info(f"   {self.bh_btc_quantity:.6f} BTC @ ${current_price:,.2f}")

    def _classify_market_regime(self, df):
        """Classify market regime"""
        lookback = 20
        if len(df) < lookback:
            return "Unknown"

        recent_data = df.tail(lookback)
        start_price = recent_data['close'].iloc[0]
        end_price = recent_data['close'].iloc[-1]
        price_change_pct = ((end_price / start_price) - 1) * 100

        if price_change_pct > Sweet2Leverage2xConfig.BULL_THRESHOLD:
            return "Bull"
        elif price_change_pct < Sweet2Leverage2xConfig.BEAR_THRESHOLD:
            return "Bear"
        else:
            return "Sideways"

    def _check_entry(self, df, idx, current_price, regime):
        """Check for entry signal"""
        should_enter, confidence, xgb_prob, tech_signal, tech_strength = \
            self.hybrid_strategy.should_enter(df, idx)

        logger.info(f"Signal Check:")
        logger.info(f"  XGBoost Prob: {xgb_prob:.3f}")
        logger.info(f"  Tech Signal: {tech_signal} (strength: {tech_strength:.3f})")
        logger.info(f"  Should Enter: {should_enter}")

        if not should_enter:
            return

        # Calculate position size WITH LEVERAGE
        base_position_value = self.capital * Sweet2Leverage2xConfig.POSITION_SIZE_PCT
        leveraged_position_value = base_position_value * Sweet2Leverage2xConfig.LEVERAGE
        quantity = leveraged_position_value / current_price

        # Enter position
        self.position = {
            "entry_idx": idx,
            "entry_price": current_price,
            "quantity": quantity,
            "base_value": base_position_value,
            "leveraged_value": leveraged_position_value,
            "entry_time": datetime.now(),
            "regime": regime,
            "xgb_prob": xgb_prob,
            "tech_signal": tech_signal,
            "tech_strength": tech_strength,
            "confidence": confidence
        }

        logger.success(f"🔔 ENTRY: LONG {quantity:.4f} BTC @ ${current_price:,.2f}")
        logger.warning(f"   ⚡ LEVERAGE: {Sweet2Leverage2xConfig.LEVERAGE}x")
        logger.info(f"   Base Value: ${base_position_value:,.2f}")
        logger.info(f"   Leveraged Value: ${leveraged_position_value:,.2f}")
        logger.info(f"   Confidence: {confidence}")

    def _manage_position(self, current_price, df, current_idx):
        """Manage existing position with leverage"""
        entry_price = self.position['entry_price']
        entry_idx = self.position['entry_idx']
        quantity = self.position['quantity']
        base_value = self.position['base_value']

        # Calculate P&L (AMPLIFIED by leverage)
        price_change_pct = (current_price - entry_price) / entry_price
        leveraged_pnl_pct = price_change_pct * Sweet2Leverage2xConfig.LEVERAGE
        leveraged_pnl_usd = leveraged_pnl_pct * base_value

        # Holding time
        hours_held = (current_idx - entry_idx) / 12

        logger.info(f"Position: LONG {quantity:.4f} BTC @ ${entry_price:,.2f}")
        logger.info(f"⚡ Leveraged P&L: {leveraged_pnl_pct * 100:+.2f}% (${leveraged_pnl_usd:+,.2f})")
        logger.info(f"Holding: {hours_held:.1f} hours")

        # Exit conditions
        exit_reason = None

        # LIQUIDATION CHECK (critical!)
        if leveraged_pnl_pct <= -Sweet2Leverage2xConfig.LIQUIDATION_THRESHOLD:
            exit_reason = "LIQUIDATION"
            logger.error("🚨 LIQUIDATION TRIGGERED!")

        # Emergency Stop Loss
        elif leveraged_pnl_pct <= -Sweet2Leverage2xConfig.EMERGENCY_STOP_LOSS:
            exit_reason = "Emergency Stop"

        # Regular Stop Loss
        elif leveraged_pnl_pct <= -Sweet2Leverage2xConfig.STOP_LOSS:
            exit_reason = "Stop Loss"

        # Take Profit
        elif leveraged_pnl_pct >= Sweet2Leverage2xConfig.TAKE_PROFIT:
            exit_reason = "Take Profit"

        # Max holding
        elif hours_held >= Sweet2Leverage2xConfig.MAX_HOLDING_HOURS:
            exit_reason = "Max Holding"

        if exit_reason:
            self._exit_position(current_price, exit_reason, leveraged_pnl_usd, leveraged_pnl_pct)

    def _exit_position(self, exit_price, reason, leveraged_pnl_usd, leveraged_pnl_pct):
        """Exit position and update capital"""
        # Transaction costs (on leveraged value)
        leveraged_value = self.position['leveraged_value']
        entry_cost = leveraged_value * Sweet2Leverage2xConfig.TRANSACTION_COST
        exit_cost = (exit_price / self.position['entry_price']) * leveraged_value * Sweet2Leverage2xConfig.TRANSACTION_COST
        total_cost = entry_cost + exit_cost

        # Net P&L
        net_pnl_usd = leveraged_pnl_usd - total_cost

        # Record trade
        trade = {
            "entry_time": self.position['entry_time'],
            "exit_time": datetime.now(),
            "entry_price": self.position['entry_price'],
            "exit_price": exit_price,
            "quantity": self.position['quantity'],
            "leverage": Sweet2Leverage2xConfig.LEVERAGE,
            "leveraged_pnl_pct": leveraged_pnl_pct,
            "leveraged_pnl_usd": leveraged_pnl_usd,
            "transaction_cost": total_cost,
            "pnl_usd_net": net_pnl_usd,
            "exit_reason": reason,
            "regime": self.position['regime']
        }
        self.trades.append(trade)

        # Update capital
        self.capital += net_pnl_usd

        # Risk tracking
        if net_pnl_usd < 0:
            self.consecutive_losses += 1
            self.daily_loss += abs(net_pnl_usd)
        else:
            self.consecutive_losses = 0

        # Log exit
        logger.warning(f"🔔 EXIT: {reason}")
        logger.info(f"   Exit Price: ${exit_price:,.2f}")
        logger.info(f"   ⚡ Leveraged P&L: {leveraged_pnl_pct * 100:+.2f}% (${leveraged_pnl_usd:+,.2f})")
        logger.info(f"   Transaction Cost: ${total_cost:.2f}")
        logger.info(f"   Net P&L: ${net_pnl_usd:+,.2f}")
        logger.info(f"   New Capital: ${self.capital:,.2f}")

        if net_pnl_usd < 0:
            logger.warning(f"   Consecutive Losses: {self.consecutive_losses}")

        # Clear position
        self.position = None
        self._print_stats()

    def _print_stats(self):
        """Print performance stats"""
        if len(self.trades) == 0:
            logger.info(f"\n📊 No trades yet")
            return

        df_trades = pd.DataFrame(self.trades)

        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl_usd_net'] > 0])
        win_rate = (winning_trades / total_trades) * 100

        total_net_pnl = df_trades['pnl_usd_net'].sum()
        total_return_pct = ((self.capital - self.initial_capital) / self.initial_capital) * 100

        # Time metrics
        days_running = (datetime.now() - self.session_start).total_seconds() / 86400
        daily_return = total_return_pct / days_running if days_running > 0 else 0
        trades_per_week = (total_trades / days_running) * 7 if days_running > 0 else 0

        logger.info(f"\n{'=' * 80}")
        logger.info("📊 SWEET-2 LEVERAGE 2x PERFORMANCE")
        logger.info(f"{'=' * 80}")
        logger.info(f"Session: {days_running:.1f} days")
        logger.info(f"")
        logger.info(f"Trading:")
        logger.info(f"  Total Trades: {total_trades}")
        logger.info(f"  Win Rate: {win_rate:.1f}% {'✅' if win_rate >= 50 else '⚠️'}")
        logger.info(f"  Trades/Week: {trades_per_week:.1f}")
        logger.info(f"")
        logger.info(f"Returns:")
        logger.info(f"  Total: {total_return_pct:+.2f}%")
        logger.info(f"  Daily Avg: {daily_return:+.2f}% {'✅' if daily_return >= Sweet2Leverage2xConfig.TARGET_DAILY_RETURN else '⚠️'}")
        logger.info(f"  Current Capital: ${self.capital:,.2f}")
        logger.info(f"")
        logger.info(f"Target: {Sweet2Leverage2xConfig.TARGET_DAILY_RETURN}%/day (0.46% expected)")
        logger.info(f"{'=' * 80}")

    def _print_final_stats(self):
        """Print final stats"""
        self._print_stats()

        if len(self.trades) > 0:
            df_trades = pd.DataFrame(self.trades)
            output_file = RESULTS_DIR / f"sweet2_leverage_2x_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_trades.to_csv(output_file, index=False)
            logger.success(f"\n✅ Trades saved to: {output_file}")

    def _save_state(self):
        """Save bot state"""
        state = {
            "capital": self.capital,
            "leverage": Sweet2Leverage2xConfig.LEVERAGE,
            "position": self.position,
            "trades_count": len(self.trades),
            "consecutive_losses": self.consecutive_losses,
            "daily_loss": self.daily_loss,
            "timestamp": datetime.now().isoformat()
        }

        state_file = RESULTS_DIR / "sweet2_leverage_2x_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    logger.info("=" * 80)
    logger.info("⚡ Sweet-2 LEVERAGE 2x Paper Trading Bot ⚡")
    logger.info("비판적 경고: '레버리지는 양날의 검입니다'")
    logger.info("=" * 80)

    model_path = MODELS_DIR / "xgboost_v3_lookahead3_thresh1_phase2.pkl"
    if not model_path.exists():
        logger.error(f"XGBoost Phase 2 model not found: {model_path}")
        return

    try:
        bot = Sweet2Leverage2xBot()
        bot.run()
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
