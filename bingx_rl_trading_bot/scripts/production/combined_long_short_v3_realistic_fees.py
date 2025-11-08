"""
LONG + SHORT Combined Strategy - V3 with Realistic Fees & Slippage

🔧 IMPROVEMENT: Added realistic trading costs (slippage + fees)

V2 → V3 Changes:
  ✅ Slippage simulation: 0.02% per trade (entry/exit)
  ✅ Trading fees: 0.05% per trade (BingX Taker fee)
  ✅ Total cost per round trip: ~0.14%

Why This Matters:
  - V2 (paper only) was too optimistic
  - Trade #2: +0.07% paper → likely LOSS with real costs
  - Need realistic results before testnet/mainnet

Expected Impact:
  ⚠️ Lower win rate (some small wins → losses)
  ⚠️ Lower total returns (~0.14% per trade cost)
  ✅ More realistic performance expectations
  ✅ Better preparation for real trading

Configuration: 70/30 allocation (LONG/SHORT)
Network: Paper Trading with Simulated Costs
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

# LONG Configuration
LONG_THRESHOLD = 0.7
LONG_STOP_LOSS = 0.01  # 1%
LONG_TAKE_PROFIT = 0.015  # 1.5%
LONG_MAX_HOLDING_HOURS = 4
LONG_POSITION_SIZE_PCT = 0.95  # 95% of LONG allocation

# SHORT Configuration
SHORT_THRESHOLD = 0.4
SHORT_STOP_LOSS = 0.015  # 1.5%
SHORT_TAKE_PROFIT = 0.03  # 3.0%
SHORT_MAX_HOLDING_HOURS = 4
SHORT_POSITION_SIZE_PCT = 0.95  # 95% of SHORT allocation

# 🆕 V3: Realistic Trading Costs
SLIPPAGE_PCT = 0.0002  # 0.02% slippage per trade
TAKER_FEE_PCT = 0.0005  # 0.05% BingX Taker fee
TOTAL_COST_PER_TRADE = (SLIPPAGE_PCT + TAKER_FEE_PCT) * 2  # Entry + Exit = 0.14%

CHECK_INTERVAL = 300  # 5 minutes

# BingX API Configuration
BINGX_TESTNET_URL = "https://open-api-vst.bingx.com"
BINGX_MAINNET_URL = "https://open-api.bingx.com"
USE_TESTNET = True


class CombinedLongShortBotV3:
    """LONG + SHORT Combined Strategy Bot - V3 with Realistic Fees"""

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

        # 🆕 V3: Cost tracking
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0

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
        self.log_file = LOGS_DIR / f"combined_v3_realistic_fees_{timestamp}.log"

        self.log("="*80)
        self.log("LONG + SHORT V3 - REALISTIC FEES & SLIPPAGE - STARTED")
        self.log("="*80)
        self.log("🔧 IMPROVEMENT: Added realistic trading costs")
        self.log("")
        self.log("V2 vs V3 Difference:")
        self.log("  V2: Paper trading (no costs)")
        self.log("  V3: Paper + simulated slippage & fees")
        self.log("")
        self.log("Simulated Costs:")
        self.log(f"  Slippage: {SLIPPAGE_PCT*100:.3f}% per trade")
        self.log(f"  Trading Fee: {TAKER_FEE_PCT*100:.3f}% per trade (BingX Taker)")
        self.log(f"  Total per round trip: {TOTAL_COST_PER_TRADE*100:.3f}%")
        self.log("")
        self.log("Expected Impact:")
        self.log("  - Trade #2 (V2: +0.07%) → likely LOSS in V3")
        self.log("  - Win rate: Lower (small wins → losses)")
        self.log("  - Returns: -0.14% per trade compared to V2")
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
        self.log(f"  Take Profit: {LONG_TAKE_PROFIT*100:.1f}%")
        self.log(f"  Risk/Reward: 1:{LONG_TAKE_PROFIT/LONG_STOP_LOSS:.1f}")
        self.log(f"  Max Holding: {LONG_MAX_HOLDING_HOURS}h")
        self.log("")
        self.log(f"SHORT Configuration:")
        self.log(f"  Model: 3-class Phase 4")
        self.log(f"  Threshold: {SHORT_THRESHOLD}")
        self.log(f"  Stop Loss: {SHORT_STOP_LOSS*100:.1f}%")
        self.log(f"  Take Profit: {SHORT_TAKE_PROFIT*100:.1f}%")
        self.log(f"  Risk/Reward: 1:{SHORT_TAKE_PROFIT/SHORT_STOP_LOSS:.1f}")
        self.log(f"  Max Holding: {SHORT_MAX_HOLDING_HOURS}h")
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

                    network = "TESTNET DATA" if USE_TESTNET else "MAINNET DATA"
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

        entry_price = self.long_position['actual_entry_price']  # 🆕 V3: Use actual entry price
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

        entry_price = self.short_position['actual_entry_price']  # 🆕 V3: Use actual entry price
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
        """Enter LONG position with slippage & fees"""
        # 🆕 V3: Apply slippage (LONG buys at higher price)
        actual_entry_price = price * (1 + SLIPPAGE_PCT)
        slippage_cost = (actual_entry_price - price) * (self.long_capital * LONG_POSITION_SIZE_PCT / price)

        # Calculate position value and fees
        position_value = self.long_capital * LONG_POSITION_SIZE_PCT
        entry_fee = position_value * TAKER_FEE_PCT
        position_value_after_fee = position_value - entry_fee

        quantity = position_value_after_fee / actual_entry_price

        # Track costs
        self.total_slippage_cost += slippage_cost
        self.total_fees_paid += entry_fee

        self.long_position = {
            'side': 'LONG',
            'signal_price': price,  # Original signal price
            'actual_entry_price': actual_entry_price,  # Price after slippage
            'entry_time': datetime.now(),
            'quantity': quantity,
            'probability': prob,
            'entry_fee': entry_fee,
            'slippage_cost': slippage_cost
        }

        self.log("="*80)
        self.log(f"🟢 LONG POSITION ENTERED")
        self.log(f"  Signal Price: ${price:,.2f}")
        self.log(f"  Actual Entry: ${actual_entry_price:,.2f} (slippage: ${actual_entry_price-price:.2f})")
        self.log(f"  Quantity: {quantity:.6f} BTC")
        self.log(f"  Position Value: ${position_value:,.2f}")
        self.log(f"  Entry Fee: ${entry_fee:.2f} ({TAKER_FEE_PCT*100:.3f}%)")
        self.log(f"  LONG Probability: {prob:.3f}")
        self.log(f"  Stop Loss: ${actual_entry_price * (1 - LONG_STOP_LOSS):,.2f} (-{LONG_STOP_LOSS*100:.1f}%)")
        self.log(f"  Take Profit: ${actual_entry_price * (1 + LONG_TAKE_PROFIT):,.2f} (+{LONG_TAKE_PROFIT*100:.1f}%)")
        self.log("="*80)

    def enter_short_position(self, prob, price):
        """Enter SHORT position with slippage & fees"""
        # 🆕 V3: Apply slippage (SHORT sells at lower price, worse for SHORT)
        # For SHORT, we want to sell, so slippage makes us sell at LOWER price
        actual_entry_price = price * (1 - SLIPPAGE_PCT)
        slippage_cost = (price - actual_entry_price) * (self.short_capital * SHORT_POSITION_SIZE_PCT / price)

        # Calculate position value and fees
        position_value = self.short_capital * SHORT_POSITION_SIZE_PCT
        entry_fee = position_value * TAKER_FEE_PCT
        position_value_after_fee = position_value - entry_fee

        quantity = position_value_after_fee / actual_entry_price

        # Track costs
        self.total_slippage_cost += slippage_cost
        self.total_fees_paid += entry_fee

        self.short_position = {
            'side': 'SHORT',
            'signal_price': price,  # Original signal price
            'actual_entry_price': actual_entry_price,  # Price after slippage
            'entry_time': datetime.now(),
            'quantity': quantity,
            'probability': prob,
            'entry_fee': entry_fee,
            'slippage_cost': slippage_cost
        }

        self.log("="*80)
        self.log(f"🔴 SHORT POSITION ENTERED")
        self.log(f"  Signal Price: ${price:,.2f}")
        self.log(f"  Actual Entry: ${actual_entry_price:,.2f} (slippage: ${price-actual_entry_price:.2f})")
        self.log(f"  Quantity: {quantity:.6f} BTC")
        self.log(f"  Position Value: ${position_value:,.2f}")
        self.log(f"  Entry Fee: ${entry_fee:.2f} ({TAKER_FEE_PCT*100:.3f}%)")
        self.log(f"  SHORT Probability: {prob:.3f}")
        self.log(f"  Stop Loss: ${actual_entry_price * (1 + SHORT_STOP_LOSS):,.2f} (+{SHORT_STOP_LOSS*100:.1f}%)")
        self.log(f"  Take Profit: ${actual_entry_price * (1 - SHORT_TAKE_PROFIT):,.2f} (-{SHORT_TAKE_PROFIT*100:.1f}%)")
        self.log("="*80)

    def exit_long_position(self, exit_price, exit_reason):
        """Exit LONG position with slippage & fees"""
        entry_price = self.long_position['actual_entry_price']
        quantity = self.long_position['quantity']
        entry_fee = self.long_position['entry_fee']
        entry_slippage = self.long_position['slippage_cost']

        # 🆕 V3: Apply slippage (LONG sells at lower price)
        actual_exit_price = exit_price * (1 - SLIPPAGE_PCT)
        exit_slippage_cost = (exit_price - actual_exit_price) * quantity

        # Calculate exit value and fee
        exit_value = quantity * actual_exit_price
        exit_fee = exit_value * TAKER_FEE_PCT
        exit_value_after_fee = exit_value - exit_fee

        # Track costs
        self.total_slippage_cost += exit_slippage_cost
        self.total_fees_paid += exit_fee

        # Calculate P&L
        entry_cost = entry_price * quantity + entry_fee  # What we paid
        pnl_usd = exit_value_after_fee - entry_cost
        pnl_pct = pnl_usd / entry_cost

        # Total costs for this trade
        total_trade_cost = entry_fee + exit_fee + entry_slippage + exit_slippage_cost

        # Update capital
        self.long_capital += pnl_usd

        # Record trade
        trade = {
            'entry_time': self.long_position['entry_time'],
            'exit_time': datetime.now(),
            'signal_entry_price': self.long_position['signal_price'],
            'actual_entry_price': entry_price,
            'signal_exit_price': exit_price,
            'actual_exit_price': actual_exit_price,
            'quantity': quantity,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': pnl_usd,
            'total_costs': total_trade_cost,
            'fees_paid': entry_fee + exit_fee,
            'slippage_cost': entry_slippage + exit_slippage_cost,
            'exit_reason': exit_reason,
            'probability': self.long_position['probability']
        }
        self.long_trades.append(trade)

        self.log("="*80)
        self.log(f"🟢 LONG POSITION EXITED - {exit_reason}")
        self.log(f"  Signal Exit: ${exit_price:,.2f}")
        self.log(f"  Actual Exit: ${actual_exit_price:,.2f} (slippage: ${exit_price-actual_exit_price:.2f})")
        self.log(f"  Entry Price: ${entry_price:,.2f}")
        self.log(f"  P&L: {pnl_pct*100:+.2f}% (${pnl_usd:+,.2f})")
        self.log(f"  Costs: ${total_trade_cost:.2f} (fees: ${entry_fee+exit_fee:.2f}, slippage: ${entry_slippage+exit_slippage_cost:.2f})")
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
        """Exit SHORT position with slippage & fees"""
        entry_price = self.short_position['actual_entry_price']
        quantity = self.short_position['quantity']
        entry_fee = self.short_position['entry_fee']
        entry_slippage = self.short_position['slippage_cost']

        # 🆕 V3: Apply slippage (SHORT buys back at higher price, worse for SHORT)
        actual_exit_price = exit_price * (1 + SLIPPAGE_PCT)
        exit_slippage_cost = (actual_exit_price - exit_price) * quantity

        # Calculate exit value and fee
        exit_value = quantity * actual_exit_price
        exit_fee = exit_value * TAKER_FEE_PCT

        # Track costs
        self.total_slippage_cost += exit_slippage_cost
        self.total_fees_paid += exit_fee

        # Calculate P&L (SHORT: profit when price goes down)
        entry_value = entry_price * quantity  # What we sold for
        pnl_usd = entry_value - exit_value - entry_fee - exit_fee
        pnl_pct = pnl_usd / entry_value

        # Total costs for this trade
        total_trade_cost = entry_fee + exit_fee + entry_slippage + exit_slippage_cost

        # Update capital
        self.short_capital += pnl_usd

        # Record trade
        trade = {
            'entry_time': self.short_position['entry_time'],
            'exit_time': datetime.now(),
            'signal_entry_price': self.short_position['signal_price'],
            'actual_entry_price': entry_price,
            'signal_exit_price': exit_price,
            'actual_exit_price': actual_exit_price,
            'quantity': quantity,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': pnl_usd,
            'total_costs': total_trade_cost,
            'fees_paid': entry_fee + exit_fee,
            'slippage_cost': entry_slippage + exit_slippage_cost,
            'exit_reason': exit_reason,
            'probability': self.short_position['probability']
        }
        self.short_trades.append(trade)

        self.log("="*80)
        self.log(f"🔴 SHORT POSITION EXITED - {exit_reason}")
        self.log(f"  Signal Exit: ${exit_price:,.2f}")
        self.log(f"  Actual Exit: ${actual_exit_price:,.2f} (slippage: ${actual_exit_price-exit_price:.2f})")
        self.log(f"  Entry Price: ${entry_price:,.2f}")
        self.log(f"  P&L: {pnl_pct*100:+.2f}% (${pnl_usd:+,.2f})")
        self.log(f"  Costs: ${total_trade_cost:.2f} (fees: ${entry_fee+exit_fee:.2f}, slippage: ${entry_slippage+exit_slippage_cost:.2f})")
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
        self.log("🚀 Bot V3 running... Press Ctrl+C to stop")

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
                        entry_price = self.long_position['actual_entry_price']
                        pnl_pct = (current_price - entry_price) / entry_price
                        hours = (datetime.now() - self.long_position['entry_time']).total_seconds() / 3600
                        self.log(f"🟢 Holding LONG: P&L {pnl_pct*100:+.2f}% | {hours:.1f}h")

                # Check SHORT position
                if self.short_position is not None:
                    should_exit, exit_reason = self.check_short_exit(current_price)

                    if should_exit:
                        self.exit_short_position(current_price, exit_reason)
                    else:
                        entry_price = self.short_position['actual_entry_price']
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
                self.log(f"💸 Total Costs: ${self.total_fees_paid + self.total_slippage_cost:.2f} (fees: ${self.total_fees_paid:.2f}, slippage: ${self.total_slippage_cost:.2f})")

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
        self.log("FINAL STATISTICS - V3 REALISTIC FEES")
        self.log("="*80)

        total_capital = self.long_capital + self.short_capital
        total_return = ((total_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

        self.log(f"\nCapital:")
        self.log(f"  Initial: ${INITIAL_CAPITAL:,.2f}")
        self.log(f"  LONG: ${self.long_capital:,.2f} ({((self.long_capital - INITIAL_CAPITAL * LONG_ALLOCATION) / (INITIAL_CAPITAL * LONG_ALLOCATION))*100:+.2f}%)")
        self.log(f"  SHORT: ${self.short_capital:,.2f} ({((self.short_capital - INITIAL_CAPITAL * SHORT_ALLOCATION) / (INITIAL_CAPITAL * SHORT_ALLOCATION))*100:+.2f}%)")
        self.log(f"  Total: ${total_capital:,.2f} ({total_return:+.2f}%)")

        self.log(f"\nCosts:")
        self.log(f"  Total Fees: ${self.total_fees_paid:.2f}")
        self.log(f"  Total Slippage: ${self.total_slippage_cost:.2f}")
        self.log(f"  Total Costs: ${self.total_fees_paid + self.total_slippage_cost:.2f}")

        # LONG stats
        if len(self.long_trades) > 0:
            long_wins = len([t for t in self.long_trades if t['pnl_usd'] > 0])
            long_win_rate = long_wins / len(self.long_trades) * 100
            tp_exits = len([t for t in self.long_trades if t['exit_reason'] == "Take Profit"])
            total_long_costs = sum([t['total_costs'] for t in self.long_trades])

            self.log(f"\nLONG Trades: {len(self.long_trades)} ({long_wins}W / {len(self.long_trades)-long_wins}L)")
            self.log(f"  Win Rate: {long_win_rate:.1f}%")
            self.log(f"  TP Exits: {tp_exits}/{len(self.long_trades)} ({tp_exits/len(self.long_trades)*100:.1f}%)")
            self.log(f"  Total Costs: ${total_long_costs:.2f}")
        else:
            self.log(f"\nLONG Trades: No trades executed")

        # SHORT stats
        if len(self.short_trades) > 0:
            short_wins = len([t for t in self.short_trades if t['pnl_usd'] > 0])
            short_win_rate = short_wins / len(self.short_trades) * 100
            tp_exits = len([t for t in self.short_trades if t['exit_reason'] == "Take Profit"])
            total_short_costs = sum([t['total_costs'] for t in self.short_trades])

            self.log(f"\nSHORT Trades: {len(self.short_trades)} ({short_wins}W / {len(self.short_trades)-short_wins}L)")
            self.log(f"  Win Rate: {short_win_rate:.1f}%")
            self.log(f"  TP Exits: {tp_exits}/{len(self.short_trades)} ({tp_exits/len(self.short_trades)*100:.1f}%)")
            self.log(f"  Total Costs: ${total_short_costs:.2f}")
        else:
            self.log(f"\nSHORT Trades: No trades executed")

        self.log("")
        self.log("✅ V3 IMPROVEMENTS:")
        self.log(f"  - Added slippage: {SLIPPAGE_PCT*100:.3f}% per trade")
        self.log(f"  - Added fees: {TAKER_FEE_PCT*100:.3f}% per trade")
        self.log(f"  - Total cost: {TOTAL_COST_PER_TRADE*100:.3f}% per round trip")
        self.log(f"  - Result: More realistic performance expectations")

        self.log("="*80)


if __name__ == "__main__":
    bot = CombinedLongShortBotV3()
    bot.run()
