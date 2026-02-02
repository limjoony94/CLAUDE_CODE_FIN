"""
Threshold Grid Search for Enhanced Models (20251024)
======================================================

Test different Entry and Exit threshold combinations to find optimal settings
for currently deployed Enhanced models.

Grid:
- Entry: [0.65, 0.70, 0.75, 0.80, 0.85]
- Exit:  [0.70, 0.75, 0.80, 0.85]
- Total: 20 combinations
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import joblib
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS_DIR = PROJECT_ROOT / "models"
LEVERAGE = 4
INITIAL_CAPITAL = 10000
TAKER_FEE = 0.0005  # 0.05%

# Emergency Safety
EMERGENCY_STOP_LOSS = 0.03  # -3%
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours

# Opportunity Gating
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047
GATE_THRESHOLD = 0.001

# Grid Search Parameters
ENTRY_THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85]
EXIT_THRESHOLDS = [0.70, 0.75, 0.80, 0.85]

print("="*80)
print("THRESHOLD GRID SEARCH - Enhanced Models (20251024)")
print("="*80)
print(f"\nGrid Configuration:")
print(f"  Entry Thresholds: {ENTRY_THRESHOLDS}")
print(f"  Exit Thresholds: {EXIT_THRESHOLDS}")
print(f"  Total Combinations: {len(ENTRY_THRESHOLDS) * len(EXIT_THRESHOLDS)}")
print("="*80)

# =============================================================================
# LOAD MODELS (Enhanced 20251024)
# =============================================================================

print("\nLoading Enhanced models (20251024)...")

# LONG Entry
long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt") as f:
    long_entry_features = [line.strip() for line in f.readlines()]

# SHORT Entry
short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt") as f:
    short_entry_features = [line.strip() for line in f.readlines()]

# LONG Exit (Latest - 20251027)
long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251027_023257.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251027_023257_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251027_023257_features.txt") as f:
    long_exit_features = [line.strip() for line in f.readlines()]

# SHORT Exit (Latest - 20251027)
short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251027_023919.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251027_023919_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251027_023919_features.txt") as f:
    short_exit_features = [line.strip() for line in f.readlines()]

print("✅ All models loaded")
print(f"  LONG Entry: {len(long_entry_features)} features")
print(f"  SHORT Entry: {len(short_entry_features)} features")
print(f"  LONG Exit: {len(long_exit_features)} features")
print(f"  SHORT Exit: {len(short_exit_features)} features")

# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading feature data...")
features_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_features.csv"
df = pd.read_csv(features_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Prepare exit features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
df = prepare_exit_features(df)

print(f"✅ Data loaded: {len(df):,} candles")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# =============================================================================
# BACKTEST FUNCTIONS
# =============================================================================

def dynamic_position_size(prob, min_size=0.20, max_size=0.95):
    """Dynamic position sizing"""
    if prob < 0.65:
        return min_size
    elif prob >= 0.85:
        return max_size
    else:
        return min_size + (max_size - min_size) * ((prob - 0.65) / (0.85 - 0.65))

def run_single_window(df_window, long_threshold, short_threshold, ml_exit_long, ml_exit_short):
    """Run backtest on single 5-day window"""

    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df_window)):
        row = df_window.iloc[i]
        price = row['close']

        # Position Management
        if position is not None:
            side = position['side']
            entry_price = position['entry_price']
            position_size_pct = position['position_size_pct']

            # Calculate P&L
            if side == 'LONG':
                price_change_pct = (price - entry_price) / entry_price
            else:
                price_change_pct = (entry_price - price) / entry_price

            leveraged_pnl_pct = price_change_pct * LEVERAGE
            leveraged_value = INITIAL_CAPITAL * position_size_pct * LEVERAGE
            pnl_usd = leveraged_value * price_change_pct

            should_exit = False
            exit_reason = None

            # 1. Stop Loss
            if leveraged_pnl_pct <= -EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = "STOP_LOSS"

            # 2. Max Hold
            elif i - position['entry_index'] >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = "MAX_HOLD"

            # 3. ML Exit
            else:
                if side == 'LONG':
                    X_exit = row[long_exit_features].values.reshape(1, -1)
                    X_exit_scaled = long_exit_scaler.transform(X_exit)
                    exit_prob = long_exit_model.predict_proba(X_exit_scaled)[0][1]

                    if exit_prob >= ml_exit_long:
                        should_exit = True
                        exit_reason = "ML_EXIT"
                else:
                    X_exit = row[short_exit_features].values.reshape(1, -1)
                    X_exit_scaled = short_exit_scaler.transform(X_exit)
                    exit_prob = short_exit_model.predict_proba(X_exit_scaled)[0][1]

                    if exit_prob >= ml_exit_short:
                        should_exit = True
                        exit_reason = "ML_EXIT"

            # Execute Exit
            if should_exit:
                entry_commission = leveraged_value * TAKER_FEE
                exit_commission = position['quantity'] * price * TAKER_FEE
                total_commission = entry_commission + exit_commission
                net_pnl_usd = pnl_usd - total_commission

                capital += net_pnl_usd

                trades.append({
                    'side': side,
                    'net_pnl_usd': net_pnl_usd,
                    'exit_reason': exit_reason,
                    'hold_time': i - position['entry_index']
                })

                position = None

        # Entry Signals
        if position is None:
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
            if long_prob >= long_threshold:
                side = 'LONG'
                entry_prob = long_prob
            elif short_prob >= short_threshold and opportunity_cost > GATE_THRESHOLD:
                side = 'SHORT'
                entry_prob = short_prob
            else:
                continue

            # Dynamic sizing
            position_size_pct = dynamic_position_size(entry_prob)
            leveraged_value = INITIAL_CAPITAL * position_size_pct * LEVERAGE
            quantity = leveraged_value / price

            position = {
                'side': side,
                'entry_price': price,
                'entry_index': i,
                'position_size_pct': position_size_pct,
                'leveraged_value': leveraged_value,
                'quantity': quantity
            }

    return trades, capital

def run_108_windows(long_threshold, short_threshold, ml_exit_long, ml_exit_short):
    """Run 108-window backtest"""

    window_size = 1440
    step_size = 288
    num_windows = (len(df) - window_size) // step_size

    window_results = []
    all_trades = []

    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        if end_idx > len(df):
            break

        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        trades, final_capital = run_single_window(
            window_df, long_threshold, short_threshold, ml_exit_long, ml_exit_short
        )

        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            all_trades.extend(trades)

            total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

            window_results.append({
                'return_pct': total_return,
                'total_trades': len(trades),
                'long_trades': len(trades_df[trades_df['side'] == 'LONG']),
                'short_trades': len(trades_df[trades_df['side'] == 'SHORT']),
                'win_rate': (trades_df['net_pnl_usd'] > 0).mean() * 100,
                'ml_exit_rate': (trades_df['exit_reason'] == 'ML_EXIT').mean() * 100
            })

    return pd.DataFrame(window_results), all_trades

# =============================================================================
# GRID SEARCH
# =============================================================================

print("\nStarting grid search...")
start_time = time.time()

results = []

for entry_threshold in ENTRY_THRESHOLDS:
    for exit_threshold in EXIT_THRESHOLDS:
        combo_start = time.time()

        # Run 108-window backtest
        window_df, all_trades = run_108_windows(
            long_threshold=entry_threshold,
            short_threshold=entry_threshold,  # Same for both
            ml_exit_long=exit_threshold,
            ml_exit_short=exit_threshold
        )

        if len(window_df) > 0:
            trades_df = pd.DataFrame(all_trades)

            # Calculate metrics
            mean_return = window_df['return_pct'].mean()
            std_return = window_df['return_pct'].std()
            sharpe = (mean_return / std_return) * 8.544 if std_return > 0 else 0

            # Composite score (return * win_rate * sharpe / std)
            win_rate = (trades_df['net_pnl_usd'] > 0).mean() * 100
            composite = (mean_return * win_rate * sharpe) / (std_return + 1e-6)

            results.append({
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'mean_return': mean_return,
                'std_return': std_return,
                'sharpe_ratio': sharpe,
                'win_rate': win_rate,
                'total_trades': len(all_trades),
                'avg_trades_per_window': window_df['total_trades'].mean(),
                'ml_exit_rate': window_df['ml_exit_rate'].mean(),
                'positive_windows': (window_df['return_pct'] > 0).sum(),
                'total_windows': len(window_df),
                'composite_score': composite
            })

        combo_time = time.time() - combo_start
        print(f"  Entry={entry_threshold:.2f}, Exit={exit_threshold:.2f}: "
              f"Return={mean_return:+.2f}%, WR={win_rate:.1f}%, "
              f"Sharpe={sharpe:.3f} ({combo_time:.1f}s)")

total_time = time.time() - start_time
print(f"\n✅ Grid search complete in {total_time/60:.1f} minutes")

# =============================================================================
# ANALYZE RESULTS
# =============================================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('composite_score', ascending=False)

print("\n" + "="*80)
print("GRID SEARCH RESULTS - TOP 5")
print("="*80)

for idx, row in results_df.head(5).iterrows():
    rank = list(results_df.index).index(idx) + 1
    print(f"\nRank {rank}: Entry {row['entry_threshold']:.2f} / Exit {row['exit_threshold']:.2f}")
    print(f"  Return: {row['mean_return']:+.2f}% ± {row['std_return']:.2f}%")
    print(f"  Sharpe: {row['sharpe_ratio']:.3f}")
    print(f"  Win Rate: {row['win_rate']:.1f}%")
    print(f"  Trades/Window: {row['avg_trades_per_window']:.1f}")
    print(f"  ML Exit: {row['ml_exit_rate']:.1f}%")
    print(f"  Positive Windows: {row['positive_windows']}/{row['total_windows']} ({row['positive_windows']/row['total_windows']*100:.1f}%)")
    print(f"  Composite Score: {row['composite_score']:.2f}")

# Save results
output_file = PROJECT_ROOT / "results" / f"enhanced_models_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved: {output_file}")

print("\n" + "="*80)
