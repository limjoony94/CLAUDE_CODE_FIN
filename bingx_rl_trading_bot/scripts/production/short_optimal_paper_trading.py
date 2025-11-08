"""
SHORT Strategy - Production Paper Trading Bot

Configuration: Optimal balance between frequency and profitability
- Model: 3-class XGBoost (Phase 4)
- Threshold: 0.6 (balanced for practical frequency)
- Stop Loss: 1.5%
- Take Profit: 6.0%
- Risk-Reward Ratio: 1:4

Expected Performance:
- Win Rate: ~30-35% (lower than 0.7 threshold but more trades)
- Trades per month: ~8-12 (practical frequency)
- Expected Value: Positive (with 1:4 R:R)
"""

import time
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient
from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

# Directories
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Trading Configuration
THRESHOLD = 0.6  # Balanced threshold for practical frequency
STOP_LOSS = 0.015  # 1.5%
TAKE_PROFIT = 0.06  # 6.0%
MAX_HOLDING_HOURS = 4
POSITION_SIZE_PCT = 0.95
CHECK_INTERVAL = 300  # 5 minutes

# Paper Trading Settings
INITIAL_CAPITAL = 10000.0
paper_capital = INITIAL_CAPITAL
paper_position = None


class ShortPaperTradingBot:
    """SHORT Strategy Paper Trading Bot"""

    def __init__(self):
        self.client = BingXClient(testnet=True)
        self.capital = INITIAL_CAPITAL
        self.position = None
        self.trades = []

        # Load model
        model_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3.pkl"
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        # Load features
        feature_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3_features.txt"
        with open(feature_file, 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]

        # Advanced features calculator
        self.adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)

        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = LOGS_DIR / f"short_paper_trading_{timestamp}.log"

        self.log("="*80)
        self.log("SHORT Strategy Paper Trading Bot - STARTED")
        self.log("="*80)
        self.log(f"Configuration:")
        self.log(f"  Model: 3-class XGBoost (Phase 4)")
        self.log(f"  Threshold: {THRESHOLD}")
        self.log(f"  Stop Loss: {STOP_LOSS*100:.1f}%")
        self.log(f"  Take Profit: {TAKE_PROFIT*100:.1f}%")
        self.log(f"  Risk-Reward: 1:{TAKE_PROFIT/STOP_LOSS:.1f}")
        self.log(f"  Max Holding: {MAX_HOLDING_HOURS} hours")
        self.log(f"  Position Size: {POSITION_SIZE_PCT*100:.0f}%")
        self.log(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        self.log("="*80)

    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + "\n")

    def get_market_data(self):
        """Fetch latest market data"""
        try:
            candles = self.client.get_kline_data("BTC-USDT", "5m", limit=500)

            if not candles:
                self.log("⚠️ No candles received")
                return None

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})

            self.log(f"✅ Fetched {len(df)} candles")
            return df

        except Exception as e:
            self.log(f"❌ Error fetching data: {e}")
            return None

    def calculate_all_features(self, df):
        """Calculate all features for model"""
        try:
            # Baseline features
            df = calculate_features(df)

            # Advanced features
            df = self.adv_features.calculate_all_features(df)

            # Handle NaN
            df = df.ffill().dropna()

            self.log(f"✅ Features calculated: {len(df)} rows after processing")
            return df

        except Exception as e:
            self.log(f"❌ Error calculating features: {e}")
            return None

    def get_signal(self, df):
        """Get SHORT trading signal"""
        try:
            # Get latest features
            features = df[self.feature_columns].iloc[-1:].values

            if np.isnan(features).any():
                self.log("⚠️ NaN in features, skipping signal")
                return None, None, None

            # Get 3-class probabilities
            probs = self.model.predict_proba(features)[0]
            neutral_prob = probs[0]
            long_prob = probs[1]
            short_prob = probs[2]

            self.log(f"📊 Probabilities - NEUTRAL: {neutral_prob:.3f}, LONG: {long_prob:.3f}, SHORT: {short_prob:.3f}")

            # SHORT signal if probability >= threshold
            if short_prob >= THRESHOLD:
                return "SHORT", short_prob, df['close'].iloc[-1]

            return None, None, None

        except Exception as e:
            self.log(f"❌ Error getting signal: {e}")
            return None, None, None

    def check_exit(self, current_price):
        """Check if position should exit"""
        if self.position is None:
            return False, None

        entry_price = self.position['entry_price']
        entry_time = self.position['entry_time']

        # Calculate P&L (SHORT: profit when price goes down)
        pnl_pct = (entry_price - current_price) / entry_price

        # Check exit conditions
        if pnl_pct <= -STOP_LOSS:
            return True, "Stop Loss"
        elif pnl_pct >= TAKE_PROFIT:
            return True, "Take Profit"

        # Check holding time
        hours_held = (datetime.now() - entry_time).total_seconds() / 3600
        if hours_held >= MAX_HOLDING_HOURS:
            return True, "Max Holding"

        return False, None

    def enter_position(self, signal, prob, price):
        """Enter SHORT position (paper trading)"""
        position_value = self.capital * POSITION_SIZE_PCT
        quantity = position_value / price

        self.position = {
            'side': signal,
            'entry_price': price,
            'entry_time': datetime.now(),
            'quantity': quantity,
            'probability': prob
        }

        self.log("="*80)
        self.log(f"🔴 SHORT POSITION ENTERED")
        self.log(f"  Entry Price: ${price:,.2f}")
        self.log(f"  Quantity: {quantity:.6f} BTC")
        self.log(f"  Position Value: ${position_value:,.2f}")
        self.log(f"  SHORT Probability: {prob:.3f}")
        self.log(f"  Stop Loss: ${price * (1 + STOP_LOSS):,.2f} ({STOP_LOSS*100:.1f}%)")
        self.log(f"  Take Profit: ${price * (1 - TAKE_PROFIT):,.2f} ({TAKE_PROFIT*100:.1f}%)")
        self.log("="*80)

    def exit_position(self, exit_price, exit_reason):
        """Exit position (paper trading)"""
        entry_price = self.position['entry_price']
        quantity = self.position['quantity']

        # Calculate P&L
        pnl_pct = (entry_price - exit_price) / entry_price
        pnl_usd = pnl_pct * (entry_price * quantity)

        # Update capital
        self.capital += pnl_usd

        # Record trade
        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': datetime.now(),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': pnl_usd,
            'exit_reason': exit_reason,
            'probability': self.position['probability']
        }
        self.trades.append(trade)

        self.log("="*80)
        self.log(f"🟢 SHORT POSITION EXITED - {exit_reason}")
        self.log(f"  Exit Price: ${exit_price:,.2f}")
        self.log(f"  Entry Price: ${entry_price:,.2f}")
        self.log(f"  P&L: {pnl_pct*100:+.2f}% (${pnl_usd:+,.2f})")
        self.log(f"  New Capital: ${self.capital:,.2f}")
        self.log(f"  Total Return: {((self.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL)*100:+.2f}%")

        # Trade stats
        if len(self.trades) > 0:
            wins = len([t for t in self.trades if t['pnl_usd'] > 0])
            win_rate = wins / len(self.trades) * 100
            self.log(f"  Win Rate: {win_rate:.1f}% ({wins}/{len(self.trades)})")

        self.log("="*80)

        self.position = None

    def run(self):
        """Main trading loop"""
        self.log("🚀 Bot running... Press Ctrl+C to stop")

        try:
            while True:
                self.log("-" * 80)

                # Get market data
                df = self.get_market_data()
                if df is None:
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Calculate features
                df = self.calculate_all_features(df)
                if df is None or len(df) < 100:
                    time.sleep(CHECK_INTERVAL)
                    continue

                current_price = df['close'].iloc[-1]
                self.log(f"💰 Current Price: ${current_price:,.2f}")

                # Check existing position
                if self.position is not None:
                    should_exit, exit_reason = self.check_exit(current_price)

                    if should_exit:
                        self.exit_position(current_price, exit_reason)
                    else:
                        entry_price = self.position['entry_price']
                        pnl_pct = (entry_price - current_price) / entry_price
                        hours = (datetime.now() - self.position['entry_time']).total_seconds() / 3600
                        self.log(f"📍 Holding SHORT: P&L {pnl_pct*100:+.2f}% | {hours:.1f}h")

                # Look for entry
                else:
                    signal, prob, price = self.get_signal(df)

                    if signal == "SHORT":
                        self.enter_position(signal, prob, price)
                    else:
                        self.log("⏸️  No entry signal (waiting for SHORT >= 0.6)")

                # Wait
                self.log(f"⏳ Next check in {CHECK_INTERVAL//60} minutes...")
                time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            self.log("\n" + "="*80)
            self.log("🛑 Bot stopped by user")
            self.log_final_stats()

        except Exception as e:
            self.log(f"\n❌ Critical error: {e}")
            self.log_final_stats()

    def log_final_stats(self):
        """Log final statistics"""
        self.log("="*80)
        self.log("FINAL STATISTICS")
        self.log("="*80)

        self.log(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        self.log(f"Final Capital: ${self.capital:,.2f}")
        self.log(f"Total Return: {((self.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL)*100:+.2f}%")

        if len(self.trades) > 0:
            wins = len([t for t in self.trades if t['pnl_usd'] > 0])
            losses = len(self.trades) - wins
            win_rate = wins / len(self.trades) * 100

            avg_win = np.mean([t['pnl_pct'] for t in self.trades if t['pnl_usd'] > 0]) if wins > 0 else 0
            avg_loss = np.mean([t['pnl_pct'] for t in self.trades if t['pnl_usd'] <= 0]) if losses > 0 else 0

            self.log(f"\nTrades: {len(self.trades)} ({wins}W / {losses}L)")
            self.log(f"Win Rate: {win_rate:.1f}%")
            self.log(f"Avg Win: +{avg_win:.2f}%")
            self.log(f"Avg Loss: {avg_loss:.2f}%")

            if wins > 0 and losses > 0:
                profit_factor = abs(avg_win * wins / (avg_loss * losses))
                self.log(f"Profit Factor: {profit_factor:.2f}")
        else:
            self.log("\nNo trades executed")

        self.log("="*80)


if __name__ == "__main__":
    bot = ShortPaperTradingBot()
    bot.run()
