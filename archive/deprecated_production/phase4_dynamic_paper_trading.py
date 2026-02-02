"""
Phase 4 Dynamic Sizing Paper Trading Bot

목표: Phase 4 Dynamic Sizing 전략 실시간 검증
- XGBoost Phase 4 Base Model (37 features)
- Dynamic Position Sizing (20-95% adaptive)
- Statistically validated: 4.56% per window vs Buy & Hold (3.6x better than Sweet-2!)
- 5분 캔들 기반

Expected Performance (from statistically validated backtesting):
- vs Buy & Hold: +4.56% per window
- Trade Frequency: ~13.2 per window
- Win Rate: 69.1%
- Sharpe Ratio: 11.88
- Position Size: Dynamic 20-95% (avg 56.3%)

Critical Validation Goals:
- Week 1: Maintain >65% win rate, positive vs B&H
- Week 2: Validate 4%+ returns per window
- Decision: Deploy to live if validated, iterate if partial
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
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.production.technical_strategy import TechnicalStrategy
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

# Create directories
for dir_path in [RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Logging setup
log_file = LOGS_DIR / f"phase4_dynamic_paper_trading_{datetime.now().strftime('%Y%m%d')}.log"
logger.add(log_file, rotation="1 day", retention="30 days")

# ============================================================================
# Phase 4 Dynamic Configuration
# ============================================================================

class Phase4DynamicConfig:
    """Phase 4 Dynamic Position Sizing Configuration"""

    # XGBoost Threshold (0.7 = best from backtesting)
    XGB_THRESHOLD = 0.7

    # Expected Metrics (Phase 4 Dynamic - statistically validated)
    EXPECTED_TRADES_PER_WEEK = 21.0
    EXPECTED_WIN_RATE = 69.1
    EXPECTED_VS_BH = 4.56  # per window
    EXPECTED_AVG_POSITION = 56.3  # average position size %

    # Targets
    TARGET_WIN_RATE = 60.0
    TARGET_VS_BH = 3.0
    TARGET_AVG_POSITION = (40.0, 70.0)  # reasonable range

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
    MAX_HOLDING_HOURS = 4

    # Initial Capital
    INITIAL_CAPITAL = 10000.0

    # Dynamic Position Sizing Parameters
    BASE_POSITION_PCT = 0.50  # 50% base (conservative)
    MAX_POSITION_PCT = 0.95   # 95% maximum
    MIN_POSITION_PCT = 0.20   # 20% minimum
    SIGNAL_WEIGHT = 0.4
    VOLATILITY_WEIGHT = 0.3
    REGIME_WEIGHT = 0.2
    STREAK_WEIGHT = 0.1

    # Risk Management
    MAX_DAILY_LOSS_PCT = 0.05
    TRANSACTION_COST = 0.0006

    # Data Collection
    LOOKBACK_CANDLES = 500
    UPDATE_INTERVAL = 300  # 5 minutes

    # Market Regime Classification
    BULL_THRESHOLD = 3.0
    BEAR_THRESHOLD = -2.0

# ============================================================================
# Phase 4 Dynamic Paper Trading Bot
# ============================================================================

class Phase4DynamicPaperTradingBot:
    """
    Phase 4 Dynamic Position Sizing Paper Trading Bot

    Validates dynamic sizing strategy in real-time with 5-minute candles
    """

    def __init__(self):
        # Load XGBoost Phase 4 Base model
        model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
        feature_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"

        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost Phase 4 model not found: {model_path}")

        with open(model_path, 'rb') as f:
            self.xgboost_model = pickle.load(f)

        with open(feature_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]

        logger.success(f"✅ XGBoost Phase 4 Base model loaded: {len(self.feature_columns)} features")

        # Initialize Advanced Technical Features
        self.adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
        logger.success("✅ Advanced Technical Features initialized")

        # Initialize Dynamic Position Sizer
        self.position_sizer = DynamicPositionSizer(
            base_position_pct=Phase4DynamicConfig.BASE_POSITION_PCT,
            max_position_pct=Phase4DynamicConfig.MAX_POSITION_PCT,
            min_position_pct=Phase4DynamicConfig.MIN_POSITION_PCT,
            signal_weight=Phase4DynamicConfig.SIGNAL_WEIGHT,
            volatility_weight=Phase4DynamicConfig.VOLATILITY_WEIGHT,
            regime_weight=Phase4DynamicConfig.REGIME_WEIGHT,
            streak_weight=Phase4DynamicConfig.STREAK_WEIGHT
        )
        logger.success("✅ Dynamic Position Sizer initialized")
        logger.info(f"   Position Range: {Phase4DynamicConfig.MIN_POSITION_PCT*100:.0f}% - {Phase4DynamicConfig.MAX_POSITION_PCT*100:.0f}%")
        logger.info(f"   Base Position: {Phase4DynamicConfig.BASE_POSITION_PCT*100:.0f}%")

        # Portfolio state
        self.capital = Phase4DynamicConfig.INITIAL_CAPITAL
        self.initial_capital = Phase4DynamicConfig.INITIAL_CAPITAL
        self.position = None
        self.trades = []

        # Buy & Hold tracking
        self.bh_btc_quantity = 0.0
        self.bh_entry_price = 0.0
        self.bh_initialized = False

        # Performance tracking
        self.session_start = datetime.now()
        self.market_regime_history = []

        logger.info("=" * 80)
        logger.info("Phase 4 Dynamic Position Sizing Bot Initialized")
        logger.info("=" * 80)
        network = "TESTNET ✅" if Phase4DynamicConfig.USE_TESTNET else "MAINNET ⚠️"
        logger.info(f"Network: BingX {network}")
        logger.info(f"Initial Capital: ${self.capital:,.2f} (Virtual/Paper Trading)")
        logger.info(f"Trading Mode: LONG + SHORT ✅ (Futures)")
        logger.info(f"  LONG Entry: XGBoost Prob >= {Phase4DynamicConfig.XGB_THRESHOLD}")
        logger.info(f"  SHORT Entry: XGBoost Prob <= {1 - Phase4DynamicConfig.XGB_THRESHOLD}")
        logger.info(f"Expected Performance:")
        logger.info(f"  - vs B&H: +{Phase4DynamicConfig.EXPECTED_VS_BH:.2f}%")
        logger.info(f"  - Win Rate: {Phase4DynamicConfig.EXPECTED_WIN_RATE:.1f}%")
        logger.info(f"  - Avg Position: {Phase4DynamicConfig.EXPECTED_AVG_POSITION:.1f}%")
        logger.info("=" * 80)

    def run(self):
        """Main trading loop"""
        logger.info("🚀 Starting Phase 4 Dynamic Paper Trading...")
        logger.info(f"Update Interval: {Phase4DynamicConfig.UPDATE_INTERVAL}s (5 minutes)")

        try:
            while True:
                self._update_cycle()
                time.sleep(Phase4DynamicConfig.UPDATE_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n" + "=" * 80)
            logger.info("Bot stopped by user")
            self._print_final_stats()

    def _update_cycle(self):
        """Single update cycle (every 5 minutes)"""
        try:
            # Get market data
            df = self._get_market_data()
            if df is None or len(df) < Phase4DynamicConfig.LOOKBACK_CANDLES:
                logger.warning("Insufficient market data")
                return

            # Calculate features
            df = calculate_features(df)
            df = self.adv_features.calculate_all_features(df)

            # Handle NaN values
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

            # Print stats (with current API price for accurate B&H comparison)
            self._print_stats(current_price=current_price)

            # Save state
            self._save_state()

        except Exception as e:
            logger.error(f"Error in update cycle: {e}")
            import traceback
            traceback.print_exc()

    def _get_market_data(self):
        """Get market data (live API or simulation)"""
        try:
            # Select API URL based on testnet flag
            if Phase4DynamicConfig.USE_TESTNET:
                base_url = "https://open-api-vst.bingx.com"
                logger.debug("Using BingX Testnet API")
            else:
                base_url = Phase4DynamicConfig.BASE_URL
                logger.debug("Using BingX Mainnet API")

            url = f"{base_url}/openApi/swap/v3/quote/klines"
            params = {
                "symbol": "BTC-USDT",
                "interval": "5m",
                "limit": min(Phase4DynamicConfig.LOOKBACK_CANDLES, 500)
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

                    # CRITICAL FIX: BingX API returns candles in REVERSE order (newest first)
                    # Sort by timestamp to ensure chronological order (oldest → newest)
                    df = df.sort_values('timestamp').reset_index(drop=True)

                    network = "TESTNET" if Phase4DynamicConfig.USE_TESTNET else "MAINNET"
                    latest_price = df['close'].iloc[-1]  # Now correctly gets the LATEST candle
                    latest_time = df['timestamp'].iloc[-1]
                    logger.info(f"✅ Live data from BingX API ({network}): {len(df)} candles")
                    logger.info(f"   Latest: ${latest_price:,.2f} @ {latest_time.strftime('%Y-%m-%d %H:%M')}")
                    return df

        except Exception as e:
            logger.warning(f"⚠️ Failed to get live data from API: {e}")

        # Fallback: simulation mode
        data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
            df = df.tail(Phase4DynamicConfig.LOOKBACK_CANDLES)
            logger.info(f"📁 Simulation data from file: {len(df)} candles")
            return df

        logger.error("❌ No market data available")
        return None

    def _initialize_buy_hold(self, current_price):
        """Initialize Buy & Hold baseline"""
        self.bh_entry_price = current_price
        self.bh_btc_quantity = self.initial_capital / current_price
        self.bh_initialized = True

        logger.success(f"📊 Buy & Hold Baseline Initialized (for comparison):")
        logger.info(f"   Bought {self.bh_btc_quantity:.6f} BTC @ ${current_price:,.2f}")

    def _classify_market_regime(self, df):
        """Classify current market regime"""
        lookback = 20
        if len(df) < lookback:
            return "Unknown"

        recent_data = df.tail(lookback)
        start_price = recent_data['close'].iloc[0]
        end_price = recent_data['close'].iloc[-1]
        price_change_pct = ((end_price / start_price) - 1) * 100

        if price_change_pct > Phase4DynamicConfig.BULL_THRESHOLD:
            return "Bull"
        elif price_change_pct < Phase4DynamicConfig.BEAR_THRESHOLD:
            return "Bear"
        else:
            return "Sideways"

    def _check_entry(self, df, idx, current_price, regime):
        """Check for entry signal using Phase 4 Dynamic Strategy (LONG/SHORT)"""
        # Get XGBoost prediction
        features = df[self.feature_columns].iloc[idx:idx+1].values

        if np.isnan(features).any():
            logger.warning("NaN in features, skipping entry check")
            return

        probability = self.xgboost_model.predict_proba(features)[0][1]

        logger.info(f"Signal Check:")
        logger.info(f"  XGBoost Prob: {probability:.3f}")
        logger.info(f"  LONG Threshold: >= {Phase4DynamicConfig.XGB_THRESHOLD:.1f}")
        logger.info(f"  SHORT Threshold: <= {1 - Phase4DynamicConfig.XGB_THRESHOLD:.1f}")

        # Determine entry direction
        side = None
        signal_strength = None

        if probability >= Phase4DynamicConfig.XGB_THRESHOLD:
            # LONG signal
            side = "LONG"
            signal_strength = probability
        elif probability <= (1 - Phase4DynamicConfig.XGB_THRESHOLD):
            # SHORT signal
            side = "SHORT"
            signal_strength = 1 - probability  # Inverse for SHORT
        else:
            logger.info(f"  Should Enter: False (prob {probability:.3f} in neutral zone)")
            return

        # Calculate volatility
        current_volatility = df['atr_pct'].iloc[idx] if 'atr_pct' in df.columns else 0.01
        avg_volatility = df['atr_pct'].iloc[max(0, idx-50):idx].mean() if 'atr_pct' in df.columns else 0.01

        # Calculate dynamic position size
        sizing_result = self.position_sizer.calculate_position_size(
            capital=self.capital,
            signal_strength=signal_strength,
            current_volatility=current_volatility,
            avg_volatility=avg_volatility,
            market_regime=regime,
            recent_trades=self.trades[-10:] if len(self.trades) > 0 else [],
            leverage=1.0  # No leverage for paper trading
        )

        position_size_pct = sizing_result['position_size_pct']
        position_value = sizing_result['position_value']
        quantity = position_value / current_price

        # Enter position
        self.position = {
            "side": side,
            "entry_idx": idx,
            "entry_price": current_price,
            "quantity": quantity,
            "position_size_pct": position_size_pct,
            "position_value": position_value,
            "entry_time": datetime.now(),
            "regime": regime,
            "probability": probability,
            "signal_strength": signal_strength,
            "sizing_factors": sizing_result['factors']
        }

        logger.success(f"🔔 ENTRY: {side} {quantity:.4f} BTC @ ${current_price:,.2f}")
        logger.info(f"   Position Size: {position_size_pct*100:.1f}% (Dynamic)")
        logger.info(f"   Position Value: ${position_value:,.2f}")
        logger.info(f"   Market Regime: {regime}")
        logger.info(f"   XGBoost Prob: {probability:.3f}")
        logger.info(f"   Signal Strength: {signal_strength:.3f}")
        logger.info(f"   Sizing Factors: {sizing_result['factors']}")

    def _manage_position(self, current_price, df, current_idx):
        """Manage existing position (LONG/SHORT)"""
        side = self.position['side']
        entry_price = self.position['entry_price']
        entry_idx = self.position['entry_idx']
        quantity = self.position['quantity']

        # Calculate P&L based on position side
        if side == "LONG":
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price

        pnl_usd = pnl_pct * (entry_price * quantity)

        # Calculate holding time
        hours_held = (current_idx - entry_idx) / 12

        logger.info(f"Position: {side} {quantity:.4f} BTC @ ${entry_price:,.2f}")
        logger.info(f"P&L: {pnl_pct * 100:+.2f}% (${pnl_usd:+,.2f})")
        logger.info(f"Holding: {hours_held:.1f} hours")

        # Check exit conditions
        exit_reason = None

        if pnl_pct <= -Phase4DynamicConfig.STOP_LOSS:
            exit_reason = "Stop Loss"
        elif pnl_pct >= Phase4DynamicConfig.TAKE_PROFIT:
            exit_reason = "Take Profit"
        elif hours_held >= Phase4DynamicConfig.MAX_HOLDING_HOURS:
            exit_reason = "Max Holding"

        if exit_reason:
            self._exit_position(current_price, exit_reason, pnl_usd, pnl_pct)

    def _exit_position(self, exit_price, reason, pnl_usd, pnl_pct):
        """Exit position"""
        # Calculate transaction costs
        entry_cost = self.position['entry_price'] * self.position['quantity'] * Phase4DynamicConfig.TRANSACTION_COST
        exit_cost = exit_price * self.position['quantity'] * Phase4DynamicConfig.TRANSACTION_COST
        total_cost = entry_cost + exit_cost

        # Net P&L after costs
        net_pnl_usd = pnl_usd - total_cost

        # Record trade
        trade = {
            "side": self.position['side'],
            "entry_time": self.position['entry_time'],
            "exit_time": datetime.now(),
            "entry_price": self.position['entry_price'],
            "exit_price": exit_price,
            "quantity": self.position['quantity'],
            "position_size_pct": self.position['position_size_pct'],
            "pnl_pct": pnl_pct,
            "pnl_usd_gross": pnl_usd,
            "transaction_cost": total_cost,
            "pnl_usd_net": net_pnl_usd,
            "exit_reason": reason,
            "regime": self.position['regime'],
            "probability": self.position['probability'],
            "signal_strength": self.position.get('signal_strength', self.position['probability']),
            "sizing_factors": self.position['sizing_factors']
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
        win_rate = (winning_trades / total_trades) * 100

        # Returns
        total_net_pnl = df_trades['pnl_usd_net'].sum()
        total_return_pct = ((self.capital - self.initial_capital) / self.initial_capital) * 100

        # Average position size
        avg_position_size = df_trades['position_size_pct'].mean() * 100

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

        # Position side breakdown
        long_trades = len(df_trades[df_trades.get('side', 'LONG') == 'LONG'])
        short_trades = len(df_trades[df_trades.get('side', 'LONG') == 'SHORT'])

        logger.info(f"\n{'=' * 80}")
        logger.info("📊 PHASE 4 DYNAMIC PERFORMANCE SUMMARY (LONG + SHORT)")
        logger.info(f"{'=' * 80}")
        logger.info(f"Session Duration: {days_running:.1f} days")
        logger.info(f"")
        logger.info(f"Trading Performance:")
        logger.info(f"  Total Trades: {total_trades} (LONG: {long_trades}, SHORT: {short_trades})")
        logger.info(f"  Winning: {winning_trades} ({win_rate:.1f}%) {'✅' if win_rate >= Phase4DynamicConfig.TARGET_WIN_RATE else '⚠️'}")
        logger.info(f"  Trades/Week: {trades_per_week:.1f}")
        logger.info(f"  Avg Position: {avg_position_size:.1f}% {'✅' if Phase4DynamicConfig.TARGET_AVG_POSITION[0] <= avg_position_size <= Phase4DynamicConfig.TARGET_AVG_POSITION[1] else '⚠️'}")
        logger.info(f"")
        logger.info(f"Returns:")
        logger.info(f"  Total Net P&L: ${total_net_pnl:+,.2f}")
        logger.info(f"  Total Return: {total_return_pct:+.2f}%")
        logger.info(f"  Current Capital: ${self.capital:,.2f}")
        logger.info(f"")
        logger.info(f"vs Buy & Hold:")
        logger.info(f"  B&H Return: {bh_return_pct:+.2f}% (BTC @ ${current_btc_price:,.2f})")
        logger.info(f"  Strategy Return: {total_return_pct:+.2f}%")
        logger.info(f"  Difference: {vs_bh:+.2f}% {'✅' if vs_bh > 0 else '⚠️'}")
        logger.info(f"")
        logger.info(f"Phase 4 Dynamic Targets:")
        logger.info(f"  Win Rate: {win_rate:.1f}% (target: >{Phase4DynamicConfig.TARGET_WIN_RATE:.0f}%)")
        logger.info(f"  vs B&H: {vs_bh:+.2f}% (target: >{Phase4DynamicConfig.TARGET_VS_BH:.0f}%)")
        logger.info(f"  Avg Position: {avg_position_size:.1f}% (target: {Phase4DynamicConfig.TARGET_AVG_POSITION[0]:.0f}-{Phase4DynamicConfig.TARGET_AVG_POSITION[1]:.0f}%)")
        logger.info(f"{'=' * 80}")

    def _print_final_stats(self):
        """Print final statistics on exit"""
        self._print_stats()

        # Save trades to CSV
        if len(self.trades) > 0:
            df_trades = pd.DataFrame(self.trades)
            output_file = RESULTS_DIR / f"phase4_dynamic_paper_trading_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_trades.to_csv(output_file, index=False)
            logger.success(f"\n✅ Trades saved to: {output_file}")

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

        state_file = RESULTS_DIR / "phase4_dynamic_paper_trading_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    logger.info("=" * 80)
    logger.info("Phase 4 Dynamic Position Sizing Paper Trading Bot")
    logger.info("비판적 사고: '동적 포지션 조절로 리스크 대비 수익률 최적화'")
    logger.info("=" * 80)

    # Check model exists
    model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
    if not model_path.exists():
        logger.error(f"XGBoost Phase 4 model not found: {model_path}")
        return

    # Initialize and run bot
    try:
        bot = Phase4DynamicPaperTradingBot()
        bot.run()
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
