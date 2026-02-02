"""
Backtest: Reduced Feature Models (90 features vs 107 original)
================================================================

Compare reduced feature models against production baseline:
- Feature Reduction: 107 → 90 features (-15.9%)
- Expected: Maintained or improved performance
- Benefit: Reduced overfitting, faster inference

Models:
  LONG Entry: 37 features (was 44)
  SHORT Entry: 30 features (was 38)
  Exit: 23 features (was 25)

Configuration:
  Leverage: 4x
  Position Sizing: Dynamic (20-95%)
  LONG Threshold: 0.65
  SHORT Threshold: 0.70
  Gate Threshold: 0.001
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import reduced feature calculator
import importlib.util
spec = importlib.util.spec_from_file_location(
    "calculate_reduced_features",
    PROJECT_ROOT / "scripts" / "experiments" / "calculate_reduced_features.py"
)
calc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calc_module)
calculate_reduced_features = calc_module.calculate_reduced_features

from scripts.production.dynamic_position_sizing import DynamicPositionSizer

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("BACKTEST: REDUCED FEATURE MODELS (90 features)")
print("="*80)
print(f"\nComparing vs Original (107 features)")
print(f"Feature Reduction: -15.9%")
print(f"Expected: Maintained/Improved performance\n")

# Strategy Parameters
LEVERAGE = 4
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001

# Exit Parameters (FULLY OPTIMIZED 2025-10-22)
ML_EXIT_THRESHOLD_LONG = 0.75  # Optimized
ML_EXIT_THRESHOLD_SHORT = 0.75  # Optimized
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours (optimized)
EMERGENCY_STOP_LOSS = -0.03  # -3% (optimized)

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Initial capital
INITIAL_CAPITAL = 10000.0

# Trading Fees
TAKER_FEE = 0.0005  # 0.05%

# ============================================================================
# Load REDUCED Feature Models
# ============================================================================
print("Loading REDUCED Feature Models...")
timestamp = "20251023_050635"

# LONG Entry Model (37 features)
long_model_path = MODELS_DIR / f"xgboost_long_entry_reduced_{timestamp}.pkl"
long_scaler_path = MODELS_DIR / f"scaler_long_entry_reduced_{timestamp}.pkl"
long_features_path = MODELS_DIR / f"xgboost_long_entry_reduced_{timestamp}_features.txt"

with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)
long_scaler = joblib.load(long_scaler_path)
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

print(f"✅ LONG Entry: {len(long_feature_columns)} features")

# SHORT Entry Model (30 features)
short_model_path = MODELS_DIR / f"xgboost_short_entry_reduced_{timestamp}.pkl"
short_scaler_path = MODELS_DIR / f"scaler_short_entry_reduced_{timestamp}.pkl"
short_features_path = MODELS_DIR / f"xgboost_short_entry_reduced_{timestamp}_features.txt"

with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)
short_scaler = joblib.load(short_scaler_path)
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print(f"✅ SHORT Entry: {len(short_feature_columns)} features")

# Exit Models (23 features)
long_exit_model_path = MODELS_DIR / f"xgboost_long_exit_reduced_{timestamp}.pkl"
long_exit_scaler_path = MODELS_DIR / f"scaler_long_exit_reduced_{timestamp}.pkl"
long_exit_features_path = MODELS_DIR / f"xgboost_long_exit_reduced_{timestamp}_features.txt"

with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)
long_exit_scaler = joblib.load(long_exit_scaler_path)
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines()]

short_exit_model_path = MODELS_DIR / f"xgboost_short_exit_reduced_{timestamp}.pkl"
short_exit_scaler_path = MODELS_DIR / f"scaler_short_exit_reduced_{timestamp}.pkl"
short_exit_features_path = MODELS_DIR / f"xgboost_short_exit_reduced_{timestamp}_features.txt"

with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)
short_exit_scaler = joblib.load(short_exit_scaler_path)
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines()]

print(f"✅ Exit Models: {len(long_exit_feature_columns)} features")
print(f"\nTotal Features: {len(long_feature_columns)} + {len(short_feature_columns)} + {len(long_exit_feature_columns)} = {len(long_feature_columns) + len(short_feature_columns) + len(long_exit_feature_columns)}")

# ============================================================================
# Load and Prepare Data
# ============================================================================
print(f"\n{'='*80}")
print("Loading and Preparing Data")
print(f"{'='*80}")

data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"\n✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Calculate REDUCED features
print("\nCalculating REDUCED features...")
start_time = time.time()
df = calculate_reduced_features(df)
calc_time = time.time() - start_time
print(f"✅ Features calculated in {calc_time:.2f}s ({len(df.columns)} columns)")

# Remove NaN
df_clean = df.dropna()
print(f"✅ Clean data: {len(df_clean):,} candles (removed {len(df)-len(df_clean):,} NaN)")

# ============================================================================
# Backtest Configuration
# ============================================================================
print(f"\n{'='*80}")
print("Backtest Configuration")
print(f"{'='*80}")
print(f"""
Strategy: Opportunity Gating with 4x Leverage
  - LONG Threshold: {LONG_THRESHOLD}
  - SHORT Threshold: {SHORT_THRESHOLD}
  - Gate Threshold: {GATE_THRESHOLD}
  - Leverage: {LEVERAGE}x

Position Sizing: Dynamic (20-95%)
  - Signal-based sizing
  - Higher signal = larger position

Exit Rules:
  - ML Exit: {ML_EXIT_THRESHOLD_LONG} (LONG), {ML_EXIT_THRESHOLD_SHORT} (SHORT)
  - Max Hold: {EMERGENCY_MAX_HOLD_TIME} candles ({EMERGENCY_MAX_HOLD_TIME*5/60:.1f} hours)
  - Stop Loss: {EMERGENCY_STOP_LOSS*100:.1f}% of balance

Capital: ${INITIAL_CAPITAL:,.2f}
Fees: {TAKER_FEE*100:.2f}% per trade
""")

# ============================================================================
# Initialize Position Sizer
# ============================================================================
position_sizer = DynamicPositionSizer(
    min_position_pct=0.20,
    max_position_pct=0.95,
    base_position_pct=0.50
)

# ============================================================================
# Backtest Loop
# ============================================================================
print(f"\n{'='*80}")
print("Running Backtest...")
print(f"{'='*80}\n")

# Trading state
balance = INITIAL_CAPITAL
position = None  # {'direction': 'LONG/SHORT', 'entry_price': float, 'size_pct': float, 'entry_idx': int}
trades = []

# Stats tracking
entry_signals = {'LONG': 0, 'SHORT': 0, 'GATED': 0}
exit_reasons = {'ML': 0, 'MAX_HOLD': 0, 'STOP_LOSS': 0}

for idx in range(len(df_clean)):
    current_price = df_clean['close'].iloc[idx]

    # ====================
    # POSITION MANAGEMENT
    # ====================
    if position is not None:
        direction = position['direction']
        entry_price = position['entry_price']
        size_pct = position['size_pct']
        entry_idx = position['entry_idx']
        candles_held = idx - entry_idx

        # Calculate leveraged P&L
        if direction == 'LONG':
            price_change_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            price_change_pct = (entry_price - current_price) / entry_price

        leveraged_pnl_pct = price_change_pct * LEVERAGE
        balance_pnl_pct = leveraged_pnl_pct * size_pct

        # Exit conditions
        should_exit = False
        exit_reason = None

        # 1. Emergency Stop Loss (-3% of total balance)
        if balance_pnl_pct <= EMERGENCY_STOP_LOSS:
            should_exit = True
            exit_reason = 'STOP_LOSS'
            exit_reasons['STOP_LOSS'] += 1

        # 2. Emergency Max Hold Time (120 candles = 10 hours)
        elif candles_held >= EMERGENCY_MAX_HOLD_TIME:
            should_exit = True
            exit_reason = 'MAX_HOLD'
            exit_reasons['MAX_HOLD'] += 1

        # 3. ML Exit Signal
        else:
            exit_features = df_clean[long_exit_feature_columns if direction == 'LONG' else short_exit_feature_columns].iloc[idx].values.reshape(1, -1)
            exit_scaler_obj = long_exit_scaler if direction == 'LONG' else short_exit_scaler
            exit_model_obj = long_exit_model if direction == 'LONG' else short_exit_model
            threshold = ML_EXIT_THRESHOLD_LONG if direction == 'LONG' else ML_EXIT_THRESHOLD_SHORT

            exit_features_scaled = exit_scaler_obj.transform(exit_features)
            exit_prob = exit_model_obj.predict_proba(exit_features_scaled)[0][1]

            if exit_prob >= threshold:
                should_exit = True
                exit_reason = 'ML'
                exit_reasons['ML'] += 1

        # Execute Exit
        if should_exit:
            # Calculate final P&L with fees
            position_value = balance * size_pct
            leveraged_position = position_value * LEVERAGE

            entry_fee = leveraged_position * TAKER_FEE
            exit_fee = leveraged_position * TAKER_FEE
            total_fees = entry_fee + exit_fee

            gross_pnl = balance * balance_pnl_pct
            net_pnl = gross_pnl - total_fees

            # Update balance
            balance += net_pnl

            # Record trade
            trades.append({
                'entry_time': df_clean['timestamp'].iloc[entry_idx],
                'exit_time': df_clean['timestamp'].iloc[idx],
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': current_price,
                'size_pct': size_pct,
                'leveraged_pnl_pct': leveraged_pnl_pct,
                'balance_pnl_pct': balance_pnl_pct,
                'gross_pnl': gross_pnl,
                'fees': total_fees,
                'net_pnl': net_pnl,
                'candles_held': candles_held,
                'exit_reason': exit_reason,
                'balance_after': balance
            })

            position = None

    # ====================
    # ENTRY SIGNALS
    # ====================
    if position is None:
        # Get LONG probability
        long_features = df_clean[long_feature_columns].iloc[idx].values.reshape(1, -1)
        long_features_scaled = long_scaler.transform(long_features)
        long_prob = long_model.predict_proba(long_features_scaled)[0][1]

        # Get SHORT probability
        short_features = df_clean[short_feature_columns].iloc[idx].values.reshape(1, -1)
        short_features_scaled = short_scaler.transform(short_features)
        short_prob = short_model.predict_proba(short_features_scaled)[0][1]

        # LONG Entry
        if long_prob >= LONG_THRESHOLD:
            sizing_result = position_sizer.get_position_size_simple(
                capital=balance,
                signal_strength=long_prob,
                leverage=LEVERAGE
            )
            position = {
                'direction': 'LONG',
                'entry_price': current_price,
                'size_pct': sizing_result['position_size_pct'],
                'position_value': sizing_result['position_value'],
                'leveraged_value': sizing_result['leveraged_value'],
                'entry_idx': idx
            }
            entry_signals['LONG'] += 1

        # SHORT Entry (with Opportunity Gating)
        elif short_prob >= SHORT_THRESHOLD:
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN
            opportunity_cost = short_ev - long_ev

            if opportunity_cost > GATE_THRESHOLD:
                sizing_result = position_sizer.get_position_size_simple(
                    capital=balance,
                    signal_strength=short_prob,
                    leverage=LEVERAGE
                )
                position = {
                    'direction': 'SHORT',
                    'entry_price': current_price,
                    'size_pct': sizing_result['position_size_pct'],
                    'position_value': sizing_result['position_value'],
                    'leveraged_value': sizing_result['leveraged_value'],
                    'entry_idx': idx
                }
                entry_signals['SHORT'] += 1
            else:
                entry_signals['GATED'] += 1

# ============================================================================
# Results Analysis
# ============================================================================
print(f"\n{'='*80}")
print("BACKTEST RESULTS: REDUCED FEATURE MODELS")
print(f"{'='*80}\n")

trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    # Overall Performance
    total_return = (balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
    win_rate = winning_trades / total_trades * 100

    avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() if (total_trades - winning_trades) > 0 else 0

    # Calculate max drawdown
    cumulative = trades_df['balance_after'].values
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    # Sharpe Ratio (annualized)
    returns = trades_df['balance_pnl_pct'].values
    avg_return = returns.mean()
    std_return = returns.std()
    sharpe = (avg_return / std_return) * np.sqrt(252 * 24 * 12) if std_return > 0 else 0  # 5-min candles

    print(f"Overall Performance:")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Balance:   ${balance:,.2f}")
    print(f"  Total Return:    {total_return:+.2f}%")
    print(f"  Max Drawdown:    {max_drawdown:.2f}%")
    print(f"  Sharpe Ratio:    {sharpe:.3f}")

    print(f"\nTrade Statistics:")
    print(f"  Total Trades:    {total_trades}")
    print(f"  Winning Trades:  {winning_trades}")
    print(f"  Win Rate:        {win_rate:.2f}%")
    print(f"  Average Win:     ${avg_win:,.2f}")
    print(f"  Average Loss:    ${avg_loss:,.2f}")
    print(f"  Profit Factor:   {abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades))):.2f}" if avg_loss != 0 else "N/A")

    print(f"\nEntry Signals:")
    print(f"  LONG Entries:    {entry_signals['LONG']}")
    print(f"  SHORT Entries:   {entry_signals['SHORT']}")
    print(f"  Gated SHORTs:    {entry_signals['GATED']}")
    print(f"  LONG %:          {entry_signals['LONG']/(entry_signals['LONG']+entry_signals['SHORT'])*100:.1f}%")
    print(f"  SHORT %:         {entry_signals['SHORT']/(entry_signals['LONG']+entry_signals['SHORT'])*100:.1f}%")

    print(f"\nExit Reasons:")
    print(f"  ML Exit:         {exit_reasons['ML']} ({exit_reasons['ML']/total_trades*100:.1f}%)")
    print(f"  Max Hold:        {exit_reasons['MAX_HOLD']} ({exit_reasons['MAX_HOLD']/total_trades*100:.1f}%)")
    print(f"  Stop Loss:       {exit_reasons['STOP_LOSS']} ({exit_reasons['STOP_LOSS']/total_trades*100:.1f}%)")

    # Direction Performance
    long_trades = trades_df[trades_df['direction'] == 'LONG']
    short_trades = trades_df[trades_df['direction'] == 'SHORT']

    print(f"\nPerformance by Direction:")
    print(f"  LONG:  {len(long_trades)} trades, {len(long_trades[long_trades['net_pnl']>0])/len(long_trades)*100:.1f}% WR, ${long_trades['net_pnl'].sum():,.2f}")
    print(f"  SHORT: {len(short_trades)} trades, {len(short_trades[short_trades['net_pnl']>0])/len(short_trades)*100:.1f}% WR, ${short_trades['net_pnl'].sum():,.2f}")

    # Save results
    results_path = RESULTS_DIR / f"backtest_reduced_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    trades_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path.name}")

    # Comparison with Original (from docs)
    print(f"\n{'='*80}")
    print("COMPARISON: Reduced vs Original Models")
    print(f"{'='*80}")
    print(f"""
REDUCED (90 features):
  Return:       {total_return:+.2f}%
  Win Rate:     {win_rate:.1f}%
  Max DD:       {max_drawdown:.2f}%
  Sharpe:       {sharpe:.3f}
  Trades:       {total_trades}

ORIGINAL (107 features) - Latest 30-day backtest:
  Return:       +75.58%
  Win Rate:     63.6%
  Max DD:       -12.2%
  Sharpe:       0.336
  Trades:       ~55

Performance Ratio:
  Return:       {total_return/75.58*100:.1f}%
  Win Rate:     {win_rate/63.6*100:.1f}%
  Risk (DD):    {abs(max_drawdown)/12.2*100:.1f}%
  Sharpe:       {sharpe/0.336*100:.1f}%
""")

    # Decision Criteria
    performance_ratio = total_return / 75.58
    if performance_ratio >= 0.95:
        print("\n✅ DECISION: DEPLOY REDUCED MODELS")
        print("   Performance maintained (>=95% of original)")
        print("   Benefits: -15.9% features, faster inference, reduced overfitting")
    elif performance_ratio >= 0.85:
        print("\n⚠️ DECISION: FURTHER TESTING NEEDED")
        print("   Performance acceptable but not optimal (85-95%)")
        print("   Recommend: Testnet validation before deployment")
    else:
        print("\n❌ DECISION: DO NOT DEPLOY")
        print("   Performance degraded significantly (<85%)")
        print("   Action: Analyze removed features, consider selective restoration")

else:
    print("⚠️ No trades executed!")

print(f"\n{'='*80}")
print("BACKTEST COMPLETE")
print(f"{'='*80}\n")
