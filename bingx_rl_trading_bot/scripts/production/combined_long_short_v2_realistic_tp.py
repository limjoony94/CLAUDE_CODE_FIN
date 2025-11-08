"""
LONG + SHORT Combined Strategy - V2 with Realistic TP Targets

🔧 IMPROVEMENT: Adjusted Take Profit targets based on actual 4h performance

Analysis of First 3 Trades (Oct 12, 00:32-10:45):
  - ALL trades hit Max Holding (4h) timeout
  - ZERO trades reached original TP targets
  - Gap: Original TPs were 7-42x higher than actual peaks

Root Cause:
  - Backtest used 2-5 day windows (longer price movements)
  - Production has 4-hour max holding constraint
  - 4h volatility much lower than multi-day volatility

Changes Made:
  LONG TP: 3.0% → 1.5% (realistic for 4h, still 1.5:1 R/R)
  SHORT TP: 6.0% → 3.0% (realistic for 4h, still 2:1 R/R)

Expected Impact:
  ✅ Higher probability of hitting TP
  ✅ Better win rate (fewer Max Hold exits)
  ✅ Faster capital rotation
  ⚠️ Lower per-trade reward (but higher frequency)

Configuration: 70/30 allocation (LONG/SHORT)
Network: BingX Testnet
"""

import time
import pandas as pd
import numpy as np
import pickle
import requests
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

# Directories
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Capital Allocation
INITIAL_CAPITAL = 10000.0
LONG_ALLOCATION = 0.70  # 70% to LONG
SHORT_ALLOCATION = 0.30  # 30% to SHORT

# LONG Configuration - ✅ UPDATED TP
LONG_THRESHOLD = 0.7
LONG_STOP_LOSS = 0.01  # 1%
LONG_TAKE_PROFIT = 0.015  # 1.5% ← UPDATED from 3.0% (realistic for 4h)
LONG_MAX_HOLDING_HOURS = 4
LONG_POSITION_SIZE_PCT = 0.95  # 95% of LONG allocation

# SHORT Configuration - ✅ UPDATED TP
SHORT_THRESHOLD = 0.4  # Optimal threshold from validation
SHORT_STOP_LOSS = 0.015  # 1.5%
SHORT_TAKE_PROFIT = 0.03  # 3.0% ← UPDATED from 6.0% (realistic for 4h)
SHORT_MAX_HOLDING_HOURS = 4
SHORT_POSITION_SIZE_PCT = 0.95  # 95% of SHORT allocation

CHECK_INTERVAL = 300  # 5 minutes

# BingX API Configuration
BINGX_TESTNET_URL = "https://open-api-vst.bingx.com"
BINGX_MAINNET_URL = "https://open-api.bingx.com"
USE_TESTNET = True


class CombinedLongShortBotV2:
    """LONG + SHORT Combined Strategy Bot - V2 with Realistic TP"""

    def __init__(self):
        # API configuration
        self.base_url = BINGX_TESTNET_URL if USE_TESTNET else BINGX_MAINNET_URL

        # Capital allocation
        self.total_capital = INITIAL_CAPITAL
        self.long_capital = INITIAL_CAPITAL * LONG_ALLOCATION
        self.short_capital = INITIAL_CAPITAL * SHORT_ALLOCATION

        # Positions
        self.long_position = None
        self.short_position = None

        # Trade history
        self.long_trades = []
        self.short_trades = []

        # Load LONG model
        long_model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
        with open(long_model_file, 'rb') as f:
            self.long_model = pickle.load(f)

        long_feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
        with open(long_feature_file, 'r') as f:
            self.long_feature_columns = [line.strip() for line in f.readlines()]

        # Load SHORT model
        short_model_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3.pkl"
        with open(short_model_file, 'rb') as f:
            self.short_model = pickle.load(f)

        short_feature_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3_features.txt"
        with open(short_feature_file, 'r') as f:
            self.short_feature_columns = [line.strip() for line in f.readlines()]

        # Advanced features calculator
        self.adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)

        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = LOGS_DIR / f"combined_v2_realistic_{timestamp}.log"

        self.log("="*80)
        self.log("LONG + SHORT V2 - REALISTIC TP TARGETS - STARTED")
        self.log("="*80)
        self.log("🔧 IMPROVEMENT: Take Profit targets adjusted based on actual data")
        self.log("")
        self.log("Analysis of First 3 Trades (4h max holding):")
        self.log("  ❌ 0/3 trades reached original TP targets")
        self.log("  ⏳ 3/3 trades hit Max Holding timeout")
        self.log("  📊 Actual peaks: LONG +0.07%, SHORT +0.82%")
        self.log("  ⚠️ Original TPs: LONG 3.0%, SHORT 6.0% (7-42x higher!)")
        self.log("")
        self.log("Root Cause:")
        self.log("  - Backtest used 2-5 day windows")
        self.log("  - Production has 4-hour constraint")
        self.log("  - Much lower volatility in 4h vs multi-day")
        self.log("")
        self.log("✅ ADJUSTMENTS MADE:")
        self.log(f"  LONG TP: 3.0% → {LONG_TAKE_PROFIT*100:.1f}% (achievable in 4h)")
        self.log(f"  SHORT TP: 6.0% → {SHORT_TAKE_PROFIT*100:.1f}% (achievable in 4h)")
        self.log("")
        self.log(f"Capital Allocation:")
        self.log(f"  Total: ${INITIAL_CAPITAL:,.2f}")
        self.log(f"  LONG (70%): ${self.long_capital:,.2f}")
        self.log(f"  SHORT (30%): ${self.short_capital:,.2f}")
        self.log("")
        self.log(f"LONG Configuration:")
        self.log(f"  Model: Phase 4 Base")
        self.log(f"  Threshold: {LONG_THRESHOLD}")
        self.log(f"  Stop Loss: {LONG_STOP_LOSS*100:.1f}%")
        self.log(f"  Take Profit: {LONG_TAKE_PROFIT*100:.1f}% ✅ UPDATED")
        self.log(f"  Risk/Reward: 1:{LONG_TAKE_PROFIT/LONG_STOP_LOSS:.1f}")
        self.log(f"  Max Holding: {LONG_MAX_HOLDING_HOURS}h")
        self.log("")
        self.log(f"SHORT Configuration:")
        self.log(f"  Model: 3-class Phase 4")
        self.log(f"  Threshold: {SHORT_THRESHOLD}")
        self.log(f"  Stop Loss: {SHORT_STOP_LOSS*100:.1f}%")
        self.log(f"  Take Profit: {SHORT_TAKE_PROFIT*100:.1f}% ✅ UPDATED")
        self.log(f"  Risk/Reward: 1:{SHORT_TAKE_PROFIT/SHORT_STOP_LOSS:.1f}")
        self.log(f"  Max Holding: {SHORT_MAX_HOLDING_HOURS}h")
        self.log("")
        self.log(f"Expected Improvements:")
        self.log(f"  ✅ Higher TP hit rate")
        self.log(f"  ✅ Fewer Max Hold exits")
        self.log(f"  ✅ Faster capital rotation")
        self.log(f"  ✅ Better win rate")
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
            url = f"{self.base_url}/openApi/swap/v3/quote/klines"
            params = {
                "symbol": "BTC-USDT",
                "interval": "5m",
                "limit": 500
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data.get('code') == 0 and 'data' in data:
                    klines = data['data']

                    # Parse to DataFrame
                    df = pd.DataFrame(klines)

                    # Rename and convert
                    df = df.rename(columns={'time': 'timestamp'})
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df[['open', 'high', 'low', 'close', 'volume']] = \
                        df[['open', 'high', 'low', 'close', 'volume']].astype(float)

                    # Reorder columns
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

                    # Sort chronologically
                    df = df.sort_values('timestamp').reset_index(drop=True)

                    network = "TESTNET" if USE_TESTNET else "MAINNET"
                    self.log(f"✅ Live data from BingX API ({network}): {len(df)} candles")
                    return df

            self.log("⚠️ No candles received")
            return None

        except Exception as e:
            self.log(f"❌ Error fetching data: {e}")
            return None

    def calculate_all_features(self, df):
        """Calculate all features"""
        try:
            # Baseline features
            df = calculate_features(df)

            # Advanced features
            df = self.adv_features.calculate_all_features(df)

            # Handle NaN
            df = df.ffill().dropna()

            return df

        except Exception as e:
            self.log(f"❌ Error calculating features: {e}")
            return None

    def get_long_signal(self, df):
        """Get LONG signal"""
        try:
            features = df[self.long_feature_columns].iloc[-1:].values

            if np.isnan(features).any():
                return None, None, None

            prob = self.long_model.predict_proba(features)[0][1]

            if prob >= LONG_THRESHOLD:
                return "LONG", prob, df['close'].iloc[-1]

            return None, None, None

        except Exception as e:
            self.log(f"❌ Error getting LONG signal: {e}")
            return None, None, None

    def get_short_signal(self, df):
        """Get SHORT signal"""
        try:
            features = df[self.short_feature_columns].iloc[-1:].values

            if np.isnan(features).any():
                return None, None, None

            probs = self.short_model.predict_proba(features)[0]
            short_prob = probs[2]  # Class 2 = SHORT

            if short_prob >= SHORT_THRESHOLD:
                return "SHORT", short_prob, df['close'].iloc[-1]

            return None, None, None

        except Exception as e:
            self.log(f"❌ Error getting SHORT signal: {e}")
            return None, None, None

    def check_long_exit(self, current_price):
        """Check if LONG position should exit"""
        if self.long_position is None:
            return False, None

        entry_price = self.long_position['entry_price']
        entry_time = self.long_position['entry_time']

        # LONG P&L
        pnl_pct = (current_price - entry_price) / entry_price

        # Check exit conditions
        if pnl_pct <= -LONG_STOP_LOSS:
            return True, "Stop Loss"
        elif pnl_pct >= LONG_TAKE_PROFIT:
            return True, "Take Profit"

        # Check holding time
        hours_held = (datetime.now() - entry_time).total_seconds() / 3600
        if hours_held >= LONG_MAX_HOLDING_HOURS:
            return True, "Max Holding"

        return False, None

    def check_short_exit(self, current_price):
        """Check if SHORT position should exit"""
        if self.short_position is None:
            return False, None

        entry_price = self.short_position['entry_price']
        entry_time = self.short_position['entry_time']

        # SHORT P&L
        pnl_pct = (entry_price - current_price) / entry_price

        # Check exit conditions
        if pnl_pct <= -SHORT_STOP_LOSS:
            return True, "Stop Loss"
        elif pnl_pct >= SHORT_TAKE_PROFIT:
            return True, "Take Profit"

        # Check holding time
        hours_held = (datetime.now() - entry_time).total_seconds() / 3600
        if hours_held >= SHORT_MAX_HOLDING_HOURS:
            return True, "Max Holding"

        return False, None

    def enter_long_position(self, prob, price):
        """Enter LONG position"""
        position_value = self.long_capital * LONG_POSITION_SIZE_PCT
        quantity = position_value / price

        self.long_position = {
            'side': 'LONG',
            'entry_price': price,
            'entry_time': datetime.now(),
            'quantity': quantity,
            'probability': prob
        }

        self.log("="*80)
        self.log(f"🟢 LONG POSITION ENTERED")
        self.log(f"  Entry Price: ${price:,.2f}")
        self.log(f"  Quantity: {quantity:.6f} BTC")
        self.log(f"  Position Value: ${position_value:,.2f}")
        self.log(f"  LONG Probability: {prob:.3f}")
        self.log(f"  Stop Loss: ${price * (1 - LONG_STOP_LOSS):,.2f} (-{LONG_STOP_LOSS*100:.1f}%)")
        self.log(f"  Take Profit: ${price * (1 + LONG_TAKE_PROFIT):,.2f} (+{LONG_TAKE_PROFIT*100:.1f}%)")
        self.log(f"  Risk/Reward: 1:{LONG_TAKE_PROFIT/LONG_STOP_LOSS:.1f}")
        self.log("="*80)

    def enter_short_position(self, prob, price):
        """Enter SHORT position"""
        position_value = self.short_capital * SHORT_POSITION_SIZE_PCT
        quantity = position_value / price

        self.short_position = {
            'side': 'SHORT',
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
        self.log(f"  Stop Loss: ${price * (1 + SHORT_STOP_LOSS):,.2f} (+{SHORT_STOP_LOSS*100:.1f}%)")
        self.log(f"  Take Profit: ${price * (1 - SHORT_TAKE_PROFIT):,.2f} (-{SHORT_TAKE_PROFIT*100:.1f}%)")
        self.log(f"  Risk/Reward: 1:{SHORT_TAKE_PROFIT/SHORT_STOP_LOSS:.1f}")
        self.log("="*80)

    def exit_long_position(self, exit_price, exit_reason):
        """Exit LONG position"""
        entry_price = self.long_position['entry_price']
        quantity = self.long_position['quantity']

        # Calculate P&L
        pnl_pct = (exit_price - entry_price) / entry_price
        pnl_usd = pnl_pct * (entry_price * quantity)

        # Update capital
        self.long_capital += pnl_usd

        # Record trade
        trade = {
            'entry_time': self.long_position['entry_time'],
            'exit_time': datetime.now(),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': pnl_usd,
            'exit_reason': exit_reason,
            'probability': self.long_position['probability']
        }
        self.long_trades.append(trade)

        self.log("="*80)
        self.log(f"🟢 LONG POSITION EXITED - {exit_reason}")
        self.log(f"  Exit Price: ${exit_price:,.2f}")
        self.log(f"  Entry Price: ${entry_price:,.2f}")
        self.log(f"  P&L: {pnl_pct*100:+.2f}% (${pnl_usd:+,.2f})")
        self.log(f"  LONG Capital: ${self.long_capital:,.2f}")
        self.log(f"  Total Capital: ${self.long_capital + self.short_capital:,.2f}")

        # Stats
        if len(self.long_trades) > 0:
            wins = len([t for t in self.long_trades if t['pnl_usd'] > 0])
            win_rate = wins / len(self.long_trades) * 100
            self.log(f"  LONG Win Rate: {win_rate:.1f}% ({wins}/{len(self.long_trades)})")

        self.log("="*80)

        self.long_position = None

    def exit_short_position(self, exit_price, exit_reason):
        """Exit SHORT position"""
        entry_price = self.short_position['entry_price']
        quantity = self.short_position['quantity']

        # Calculate P&L
        pnl_pct = (entry_price - exit_price) / entry_price
        pnl_usd = pnl_pct * (entry_price * quantity)

        # Update capital
        self.short_capital += pnl_usd

        # Record trade
        trade = {
            'entry_time': self.short_position['entry_time'],
            'exit_time': datetime.now(),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': pnl_usd,
            'exit_reason': exit_reason,
            'probability': self.short_position['probability']
        }
        self.short_trades.append(trade)

        self.log("="*80)
        self.log(f"🔴 SHORT POSITION EXITED - {exit_reason}")
        self.log(f"  Exit Price: ${exit_price:,.2f}")
        self.log(f"  Entry Price: ${entry_price:,.2f}")
        self.log(f"  P&L: {pnl_pct*100:+.2f}% (${pnl_usd:+,.2f})")
        self.log(f"  SHORT Capital: ${self.short_capital:,.2f}")
        self.log(f"  Total Capital: ${self.long_capital + self.short_capital:,.2f}")

        # Stats
        if len(self.short_trades) > 0:
            wins = len([t for t in self.short_trades if t['pnl_usd'] > 0])
            win_rate = wins / len(self.short_trades) * 100
            self.log(f"  SHORT Win Rate: {win_rate:.1f}% ({wins}/{len(self.short_trades)})")

        self.log("="*80)

        self.short_position = None

    def run(self):
        """Main trading loop"""
        self.log("🚀 Bot V2 running... Press Ctrl+C to stop")

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

                # Check LONG position
                if self.long_position is not None:
                    should_exit, exit_reason = self.check_long_exit(current_price)

                    if should_exit:
                        self.exit_long_position(current_price, exit_reason)
                    else:
                        entry_price = self.long_position['entry_price']
                        pnl_pct = (current_price - entry_price) / entry_price
                        hours = (datetime.now() - self.long_position['entry_time']).total_seconds() / 3600
                        self.log(f"🟢 Holding LONG: P&L {pnl_pct*100:+.2f}% | {hours:.1f}h")

                # Check SHORT position
                if self.short_position is not None:
                    should_exit, exit_reason = self.check_short_exit(current_price)

                    if should_exit:
                        self.exit_short_position(current_price, exit_reason)
                    else:
                        entry_price = self.short_position['entry_price']
                        pnl_pct = (entry_price - current_price) / entry_price
                        hours = (datetime.now() - self.short_position['entry_time']).total_seconds() / 3600
                        self.log(f"🔴 Holding SHORT: P&L {pnl_pct*100:+.2f}% | {hours:.1f}h")

                # Look for LONG entry
                if self.long_position is None:
                    signal, prob, price = self.get_long_signal(df)

                    if signal == "LONG":
                        self.enter_long_position(prob, price)
                    else:
                        prob_str = f"{prob:.3f}" if prob is not None else "N/A"
                        self.log(f"⏸️  LONG: No entry signal (prob: {prob_str})")

                # Look for SHORT entry
                if self.short_position is None:
                    signal, prob, price = self.get_short_signal(df)

                    if signal == "SHORT":
                        self.enter_short_position(prob, price)
                    else:
                        prob_str = f"{prob:.3f}" if prob is not None else "N/A"
                        self.log(f"⏸️  SHORT: No entry signal (prob: {prob_str})")

                # Portfolio status
                total_capital = self.long_capital + self.short_capital
                total_return = ((total_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
                self.log(f"💼 Portfolio: ${total_capital:,.2f} ({total_return:+.2f}%)")

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
        self.log("FINAL STATISTICS - V2 REALISTIC TP")
        self.log("="*80)

        total_capital = self.long_capital + self.short_capital
        total_return = ((total_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

        self.log(f"\nCapital:")
        self.log(f"  Initial: ${INITIAL_CAPITAL:,.2f}")
        self.log(f"  LONG: ${self.long_capital:,.2f} ({((self.long_capital - INITIAL_CAPITAL * LONG_ALLOCATION) / (INITIAL_CAPITAL * LONG_ALLOCATION))*100:+.2f}%)")
        self.log(f"  SHORT: ${self.short_capital:,.2f} ({((self.short_capital - INITIAL_CAPITAL * SHORT_ALLOCATION) / (INITIAL_CAPITAL * SHORT_ALLOCATION))*100:+.2f}%)")
        self.log(f"  Total: ${total_capital:,.2f} ({total_return:+.2f}%)")

        # LONG stats
        if len(self.long_trades) > 0:
            long_wins = len([t for t in self.long_trades if t['pnl_usd'] > 0])
            long_win_rate = long_wins / len(self.long_trades) * 100
            tp_exits = len([t for t in self.long_trades if t['exit_reason'] == "Take Profit"])

            self.log(f"\nLONG Trades: {len(self.long_trades)} ({long_wins}W / {len(self.long_trades)-long_wins}L)")
            self.log(f"  Win Rate: {long_win_rate:.1f}%")
            self.log(f"  TP Exits: {tp_exits}/{len(self.long_trades)} ({tp_exits/len(self.long_trades)*100:.1f}%)")
        else:
            self.log(f"\nLONG Trades: No trades executed")

        # SHORT stats
        if len(self.short_trades) > 0:
            short_wins = len([t for t in self.short_trades if t['pnl_usd'] > 0])
            short_win_rate = short_wins / len(self.short_trades) * 100
            tp_exits = len([t for t in self.short_trades if t['exit_reason'] == "Take Profit"])

            self.log(f"\nSHORT Trades: {len(self.short_trades)} ({short_wins}W / {len(self.short_trades)-short_wins}L)")
            self.log(f"  Win Rate: {short_win_rate:.1f}%")
            self.log(f"  TP Exits: {tp_exits}/{len(self.short_trades)} ({tp_exits/len(self.short_trades)*100:.1f}%)")
        else:
            self.log(f"\nSHORT Trades: No trades executed")

        self.log("")
        self.log("✅ V2 IMPROVEMENTS:")
        self.log(f"  - LONG TP: 3.0% → {LONG_TAKE_PROFIT*100:.1f}%")
        self.log(f"  - SHORT TP: 6.0% → {SHORT_TAKE_PROFIT*100:.1f}%")
        self.log(f"  - Expected: Higher TP hit rate, better win rate")

        self.log("="*80)


if __name__ == "__main__":
    bot = CombinedLongShortBotV2()
    bot.run()
