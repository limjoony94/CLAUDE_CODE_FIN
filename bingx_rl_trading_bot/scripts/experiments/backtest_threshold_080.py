"""
Backtest with Threshold 0.80 Models - Full Data Period
=======================================================

New threshold 0.80 configuration (2025-10-27):
- Entry Models: Trained 2025-10-27 with threshold 0.80
- Exit Models: Trained 2025-10-27 with threshold 0.80
- LONG/SHORT Entry Threshold: 0.80
- ML Exit Threshold: 0.80 (LONG/SHORT)
- Stop Loss: -3%
- Max Hold: 120 candles (10h)
- Leverage: 4x

Date Range: Full available data (2025-07-01 to 2025-10-26)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

# =============================================================================
# PRODUCTION CONFIGURATION
# =============================================================================

# Entry Thresholds (0.80)
LONG_THRESHOLD = 0.80
SHORT_THRESHOLD = 0.80
GATE_THRESHOLD = 0.001

# Exit Parameters
ML_EXIT_THRESHOLD_LONG = 0.80
ML_EXIT_THRESHOLD_SHORT = 0.80
EMERGENCY_STOP_LOSS = 0.03  # -3%
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Trading
LEVERAGE = 4
INITIAL_CAPITAL = 10000
TAKER_FEE = 0.0005  # 0.05% BingX taker fee

# Model Paths (THRESHOLD 0.80 - 2025-10-27)
MODELS_DIR = PROJECT_ROOT / "models"
LONG_ENTRY_MODEL = MODELS_DIR / "xgboost_long_trade_outcome_full_optimized_20251027_020339.pkl"
LONG_ENTRY_SCALER = MODELS_DIR / "xgboost_long_trade_outcome_full_optimized_20251027_020339_scaler.pkl"
SHORT_ENTRY_MODEL = MODELS_DIR / "xgboost_short_trade_outcome_full_optimized_20251027_020339.pkl"
SHORT_ENTRY_SCALER = MODELS_DIR / "xgboost_short_trade_outcome_full_optimized_20251027_020339_scaler.pkl"
LONG_EXIT_MODEL = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251027_023257.pkl"
LONG_EXIT_SCALER = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251027_023257_scaler.pkl"
SHORT_EXIT_MODEL = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251027_023919.pkl"
SHORT_EXIT_SCALER = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251027_023919_scaler.pkl"

print("=" * 80)
print("BACKTEST: PRODUCTION SETTINGS - FULL DATA PERIOD")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nConfiguration:")
print(f"  Entry: LONG={LONG_THRESHOLD}, SHORT={SHORT_THRESHOLD}, Gate={GATE_THRESHOLD}")
print(f"  Exit: ML={ML_EXIT_THRESHOLD_LONG}, SL=-{EMERGENCY_STOP_LOSS*100}%, MaxHold={EMERGENCY_MAX_HOLD_TIME/12}h")
print(f"  Leverage: {LEVERAGE}x")
print("=" * 80)

# =============================================================================
# LOAD MODELS
# =============================================================================

print("\nLoading models...")
long_entry_model = joblib.load(LONG_ENTRY_MODEL)
long_entry_scaler = joblib.load(LONG_ENTRY_SCALER)
short_entry_model = joblib.load(SHORT_ENTRY_MODEL)
short_entry_scaler = joblib.load(SHORT_ENTRY_SCALER)
long_exit_model = joblib.load(LONG_EXIT_MODEL)
long_exit_scaler = joblib.load(LONG_EXIT_SCALER)
short_exit_model = joblib.load(SHORT_EXIT_MODEL)
short_exit_scaler = joblib.load(SHORT_EXIT_SCALER)

# Load feature lists (Threshold 0.80 - 2025-10-27)
long_entry_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_optimized_20251027_020339_features.txt"
with open(long_entry_features_path, 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

short_entry_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_optimized_20251027_020339_features.txt"
with open(short_entry_features_path, 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines() if line.strip()]

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251027_023257_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines()]

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251027_023919_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines()]

print("✅ All models loaded")
print(f"  LONG Entry features: {len(long_entry_features)}")
print(f"  SHORT Entry features: {len(short_entry_features)}")
print(f"  LONG Exit features: {len(long_exit_features)}")
print(f"  SHORT Exit features: {len(short_exit_features)}")

# =============================================================================
# LOAD DATA (FULL PERIOD)
# =============================================================================

print("\nLoading data (full period)...")
data_path = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_updated.csv"
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Total days: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
print(f"  Total candles: {len(df):,}")

# =============================================================================
# CALCULATE FEATURES
# =============================================================================

print("\nCalculating features...")
df_features = calculate_all_features_enhanced_v2(df)
print(f"✅ Features calculated ({len(df_features)} rows)")

# Prepare exit features (add market context features)
print("\nPreparing exit features...")
df_features = prepare_exit_features(df_features)
print(f"✅ Exit features prepared ({len(df_features)} rows)")

# =============================================================================
# BACKTEST LOGIC
# =============================================================================

def dynamic_position_size(prob, min_size=0.20, max_size=0.95):
    """Dynamic position sizing based on signal strength"""
    if prob < 0.65:
        return min_size
    elif prob >= 0.85:
        return max_size
    else:
        # Linear interpolation
        return min_size + (max_size - min_size) * ((prob - 0.65) / (0.85 - 0.65))

def run_backtest():
    """Run backtest with production settings"""

    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df_features)):
        row = df_features.iloc[i]
        timestamp = row['timestamp']
        price = row['close']

        # === POSITION MANAGEMENT ===
        if position is not None:
            side = position['side']
            entry_price = position['entry_price']
            entry_time = position['entry_time']
            position_size_pct = position['position_size_pct']

            # Calculate P&L
            if side == 'LONG':
                price_change_pct = (price - entry_price) / entry_price
            else:  # SHORT
                price_change_pct = (entry_price - price) / entry_price

            leveraged_pnl_pct = price_change_pct * LEVERAGE

            # Check exit conditions
            should_exit = False
            exit_reason = None

            # 1. Emergency Stop Loss (-3% balance)
            if leveraged_pnl_pct <= -EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = "STOP_LOSS"

            # 2. Emergency Max Hold (10 hours = 120 candles)
            elif i - position['entry_index'] >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = "MAX_HOLD"

            # 3. ML Exit
            else:
                # Get current features for exit prediction
                if side == 'LONG':
                    X_exit = row[long_exit_features].values.reshape(1, -1)
                    X_exit_scaled = long_exit_scaler.transform(X_exit)
                    exit_prob = long_exit_model.predict_proba(X_exit_scaled)[0][1]

                    if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                        should_exit = True
                        exit_reason = "ML_EXIT"

                else:  # SHORT
                    X_exit = row[short_exit_features].values.reshape(1, -1)
                    X_exit_scaled = short_exit_scaler.transform(X_exit)
                    exit_prob = short_exit_model.predict_proba(X_exit_scaled)[0][1]

                    if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                        should_exit = True
                        exit_reason = "ML_EXIT"

            # Execute exit
            if should_exit:
                pnl_usd = capital * position_size_pct * leveraged_pnl_pct
                capital += pnl_usd

                hold_time = i - position['entry_index']

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'position_size_pct': position_size_pct,
                    'pnl_pct': leveraged_pnl_pct * 100,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'hold_time': hold_time
                })

                position = None

        # === ENTRY SIGNALS ===
        if position is None:
            # Get entry probabilities
            X_long = row[long_entry_features].values.reshape(1, -1)
            X_long_scaled = long_entry_scaler.transform(X_long)
            long_prob = long_entry_model.predict_proba(X_long_scaled)[0][1]

            X_short = row[short_entry_features].values.reshape(1, -1)
            X_short_scaled = short_entry_scaler.transform(X_short)
            short_prob = short_entry_model.predict_proba(X_short_scaled)[0][1]

            # Opportunity cost gating
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN
            opportunity_cost = short_ev - long_ev

            # Entry logic
            if long_prob >= LONG_THRESHOLD:
                side = 'LONG'
                entry_prob = long_prob
            elif short_prob >= SHORT_THRESHOLD and opportunity_cost > GATE_THRESHOLD:
                side = 'SHORT'
                entry_prob = short_prob
            else:
                continue

            # Dynamic position sizing
            position_size_pct = dynamic_position_size(entry_prob)

            position = {
                'side': side,
                'entry_price': price,
                'entry_time': timestamp,
                'entry_index': i,
                'entry_prob': entry_prob,
                'position_size_pct': position_size_pct
            }

    return capital, trades

def run_single_window_backtest(df_window):
    """Run backtest on a single window (5 days = 1440 candles)"""
    
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df_window)):
        row = df_window.iloc[i]
        timestamp = row['timestamp']
        price = row['close']

        # === POSITION MANAGEMENT ===
        if position is not None:
            side = position['side']
            entry_price = position['entry_price']
            entry_time = position['entry_time']
            position_size_pct = position['position_size_pct']

            # Calculate P&L
            if side == 'LONG':
                price_change_pct = (price - entry_price) / entry_price
            else:  # SHORT
                price_change_pct = (entry_price - price) / entry_price

            leveraged_pnl_pct = price_change_pct * LEVERAGE
            
            # P&L in USD (based on INITIAL capital, not growing capital)
            leveraged_value = INITIAL_CAPITAL * position_size_pct * LEVERAGE
            pnl_usd = leveraged_value * price_change_pct

            # Check exit conditions
            should_exit = False
            exit_reason = None

            # 1. Emergency Stop Loss (-3% balance)
            if leveraged_pnl_pct <= -EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = "STOP_LOSS"

            # 2. Emergency Max Hold (10 hours = 120 candles)
            elif i - position['entry_index'] >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = "MAX_HOLD"

            # 3. ML Exit
            else:
                # Get current features for exit prediction
                if side == 'LONG':
                    X_exit = row[long_exit_features].values.reshape(1, -1)
                    X_exit_scaled = long_exit_scaler.transform(X_exit)
                    exit_prob = long_exit_model.predict_proba(X_exit_scaled)[0][1]

                    if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                        should_exit = True
                        exit_reason = "ML_EXIT"

                else:  # SHORT
                    X_exit = row[short_exit_features].values.reshape(1, -1)
                    X_exit_scaled = short_exit_scaler.transform(X_exit)
                    exit_prob = short_exit_model.predict_proba(X_exit_scaled)[0][1]

                    if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                        should_exit = True
                        exit_reason = "ML_EXIT"

            # Execute exit
            if should_exit:
                # Calculate commissions (0.05% per trade)
                entry_commission = leveraged_value * TAKER_FEE
                exit_commission = position['quantity'] * price * TAKER_FEE if 'quantity' in position else leveraged_value * TAKER_FEE
                total_commission = entry_commission + exit_commission
                
                # Net P&L after fees
                net_pnl_usd = pnl_usd - total_commission
                capital += net_pnl_usd

                hold_time = i - position['entry_index']

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'position_size_pct': position_size_pct,
                    'pnl_pct': leveraged_pnl_pct * 100,
                    'pnl_usd': pnl_usd,
                    'commission_usd': total_commission,
                    'net_pnl_usd': net_pnl_usd,
                    'exit_reason': exit_reason,
                    'hold_time': hold_time
                })

                position = None

        # === ENTRY SIGNALS ===
        if position is None:
            # Get entry probabilities
            X_long = row[long_entry_features].values.reshape(1, -1)
            X_long_scaled = long_entry_scaler.transform(X_long)
            long_prob = long_entry_model.predict_proba(X_long_scaled)[0][1]

            X_short = row[short_entry_features].values.reshape(1, -1)
            X_short_scaled = short_entry_scaler.transform(X_short)
            short_prob = short_entry_model.predict_proba(X_short_scaled)[0][1]

            # Opportunity cost gating
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN
            opportunity_cost = short_ev - long_ev

            # Entry logic
            if long_prob >= LONG_THRESHOLD:
                side = 'LONG'
                entry_prob = long_prob
            elif short_prob >= SHORT_THRESHOLD and opportunity_cost > GATE_THRESHOLD:
                side = 'SHORT'
                entry_prob = short_prob
            else:
                continue

            # Dynamic position sizing
            position_size_pct = dynamic_position_size(entry_prob)
            leveraged_value = INITIAL_CAPITAL * position_size_pct * LEVERAGE
            quantity = leveraged_value / price  # BTC quantity

            position = {
                'side': side,
                'entry_price': price,
                'entry_time': timestamp,
                'entry_index': i,
                'entry_prob': entry_prob,
                'position_size_pct': position_size_pct,
                'leveraged_value': leveraged_value,
                'quantity': quantity
            }

    return trades, capital

def run_108_window_backtest():
    """Run 108-window backtest (each window = 5 days = 1440 candles)"""
    
    window_size = 1440  # 5 days (288 candles/day × 5)
    step_size = 288     # 1 day step (20% overlap)
    num_windows = (len(df_features) - window_size) // step_size
    
    print(f"\n108-Window Backtest Configuration:")
    print(f"  Window Size: {window_size} candles (5 days)")
    print(f"  Step Size: {step_size} candles (1 day)")
    print(f"  Total Windows: {num_windows}")
    print(f"  Each window starts with: ${INITIAL_CAPITAL:,.0f}")
    
    window_results = []
    all_trades = []
    
    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size
        
        if end_idx > len(df_features):
            break
        
        window_df = df_features.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # Run backtest for this window
        trades, final_capital = run_single_window_backtest(window_df)
        
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            all_trades.extend(trades)
            
            # Calculate window metrics
            total_return_pct = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            long_trades = trades_df[trades_df['side'] == 'LONG']
            short_trades = trades_df[trades_df['side'] == 'SHORT']
            
            window_results.append({
                'window': window_idx,
                'total_trades': len(trades),
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'return_pct': total_return_pct,
                'final_capital': final_capital,
                'win_rate': (trades_df['net_pnl_usd'] > 0).mean() * 100,
                'ml_exit_rate': (trades_df['exit_reason'] == 'ML_EXIT').mean() * 100,
                'avg_hold_time': trades_df['hold_time'].mean()
            })
    
    return pd.DataFrame(window_results), all_trades

# =============================================================================
# RUN BACKTEST
# =============================================================================

print("\nRunning 108-window backtest...")
window_results_df, all_trades = run_108_window_backtest()

# =============================================================================
# ANALYZE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("BACKTEST RESULTS - 108 WINDOW METHODOLOGY")
print("=" * 80)

if len(window_results_df) > 0:
    # Window-level statistics
    mean_return = window_results_df['return_pct'].mean()
    std_return = window_results_df['return_pct'].std()
    median_return = window_results_df['return_pct'].median()
    
    # Sharpe Ratio (annualized)
    # 5-day windows = 73 windows/year, sqrt(73) = 8.544
    sharpe_ratio = (mean_return / std_return) * 8.544 if std_return > 0 else 0
    
    print(f"\nWindow Performance (5-day windows, n={len(window_results_df)}):")
    print(f"  Mean Return: {mean_return:+.2f}%")
    print(f"  Std Return: {std_return:.2f}%")
    print(f"  Median Return: {median_return:+.2f}%")
    print(f"  Sharpe Ratio (annualized): {sharpe_ratio:.3f}")
    print(f"  Positive Windows: {(window_results_df['return_pct'] > 0).sum()}/{len(window_results_df)} ({(window_results_df['return_pct'] > 0).mean()*100:.1f}%)")
    
    # Trade statistics
    total_trades = len(all_trades)
    avg_trades_per_window = window_results_df['total_trades'].mean()
    
    df_all_trades = pd.DataFrame(all_trades)
    wins = df_all_trades[df_all_trades['net_pnl_usd'] > 0]
    losses = df_all_trades[df_all_trades['net_pnl_usd'] <= 0]
    
    print(f"\nTrade Statistics (All Windows):")
    print(f"  Total Trades: {total_trades:,}")
    print(f"  Avg Trades/Window: {avg_trades_per_window:.1f}")
    print(f"  LONG: {len(df_all_trades[df_all_trades['side'] == 'LONG'])} ({len(df_all_trades[df_all_trades['side'] == 'LONG'])/len(df_all_trades)*100:.1f}%)")
    print(f"  SHORT: {len(df_all_trades[df_all_trades['side'] == 'SHORT'])} ({len(df_all_trades[df_all_trades['side'] == 'SHORT'])/len(df_all_trades)*100:.1f}%)")
    
    print(f"\nWin Rate: {len(wins)/len(df_all_trades)*100:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Avg Win: ${wins['net_pnl_usd'].mean():.2f}" if len(wins) > 0 else "Avg Win: N/A")
    print(f"Avg Loss: ${losses['net_pnl_usd'].mean():.2f}" if len(losses) > 0 else "Avg Loss: N/A")
    print(f"Avg Hold: {df_all_trades['hold_time'].mean():.1f} candles ({df_all_trades['hold_time'].mean()/12:.1f}h)")
    
    # Profit factor
    if len(losses) > 0 and losses['net_pnl_usd'].sum() != 0:
        profit_factor = abs(wins['net_pnl_usd'].sum() / losses['net_pnl_usd'].sum())
        print(f"Profit Factor: {profit_factor:.2f}x")
    
    print(f"\nExit Reasons:")
    for reason, count in df_all_trades['exit_reason'].value_counts().items():
        print(f"  {reason}: {count} ({count/len(df_all_trades)*100:.1f}%)")
    
    # Risk metrics (from window returns)
    print(f"\nRisk Metrics (Window-Based):")
    max_window_loss = window_results_df['return_pct'].min()
    max_window_gain = window_results_df['return_pct'].max()
    print(f"  Worst Window: {max_window_loss:.2f}%")
    print(f"  Best Window: {max_window_gain:.2f}%")
    print(f"  Best Trade: ${df_all_trades['net_pnl_usd'].max():.2f}")
    print(f"  Worst Trade: ${df_all_trades['net_pnl_usd'].min():.2f}")
    
    # Performance projection
    print(f"\nPerformance Projection:")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Expected 5-day Return: {mean_return:+.2f}%")
    print(f"  Projected 5-day Capital: ${INITIAL_CAPITAL * (1 + mean_return/100):,.0f}")
    print(f"  Conservative 5-day (30% degradation): ${INITIAL_CAPITAL * (1 + mean_return*0.7/100):,.0f}")
    
    # Save results
    output_file = PROJECT_ROOT / "results" / f"backtest_threshold_080_108windows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    window_results_df.to_csv(output_file, index=False)
    print(f"\n✅ Window results saved: {output_file}")
    
    # Save all trades
    trades_file = PROJECT_ROOT / "results" / f"backtest_threshold_080_all_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_all_trades.to_csv(trades_file, index=False)
    print(f"✅ All trades saved: {trades_file}")

else:
    print("\n⚠️  No windows executed")

print("\n" + "=" * 80)