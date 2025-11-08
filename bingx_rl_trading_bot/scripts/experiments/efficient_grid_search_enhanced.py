"""
Efficient Grid Search for Enhanced Models
==========================================

Tests threshold combinations efficiently by:
1. Loading data and features once
2. Pre-calculating all signals once
3. Running window backtests with different thresholds

Much faster than re-running entire backtest for each combination.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
import time
from datetime import datetime

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
LEVERAGE = 4
INITIAL_CAPITAL = 10000
TAKER_FEE = 0.0005
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD_TIME = 120
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047
GATE_THRESHOLD = 0.001

# Grid parameters
ENTRY_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]
EXIT_THRESHOLDS = [0.70, 0.75, 0.80, 0.85]

print("="*80)
print("EFFICIENT GRID SEARCH - Enhanced Models (20251024)")
print("="*80)
print(f"\nGrid: {len(ENTRY_THRESHOLDS)} × {len(EXIT_THRESHOLDS)} = {len(ENTRY_THRESHOLDS)*len(EXIT_THRESHOLDS)} combinations")
print(f"  Entry: {ENTRY_THRESHOLDS}")
print(f"  Exit: {EXIT_THRESHOLDS}")
print("="*80)

# =============================================================================
# LOAD MODELS
# =============================================================================

print("\nLoading models...")

# Entry Models
long_model = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl")
long_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt") as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl")
short_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt") as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

# Exit Models (Latest)
long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251027_023257.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251027_023257_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251027_023257_features.txt") as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines()]

short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251027_023919.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251027_023919_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251027_023919_features.txt") as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines()]

print("✅ Models loaded")

# Position sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# =============================================================================
# LOAD DATA & CALCULATE FEATURES (ONCE!)
# =============================================================================

print("\nLoading data...")
df_full = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
print(f"✅ Data loaded: {len(df_full):,} candles")

print("\nCalculating features...")
df = calculate_all_features_enhanced_v2(df_full, phase='phase1')
df = prepare_exit_features(df)
print(f"✅ Features ready: {len(df):,} rows")

# Pre-calculate entry signals (ONCE!)
print("\nPre-calculating entry signals...")
long_feat = df[long_feature_columns].values
long_feat_scaled = long_scaler.transform(long_feat)
df['long_prob'] = long_model.predict_proba(long_feat_scaled)[:, 1]

short_feat = df[short_feature_columns].values
short_feat_scaled = short_scaler.transform(short_feat)
df['short_prob'] = short_model.predict_proba(short_feat_scaled)[:, 1]

print("✅ Entry signals pre-calculated")

# =============================================================================
# BACKTEST FUNCTION (with parameterized thresholds)
# =============================================================================

def backtest_window(window_df, long_threshold, short_threshold, ml_exit_long, ml_exit_short):
    """Run backtest on single window with given thresholds"""

    trades = []
    position = None
    capital = INITIAL_CAPITAL
    first_exit_signal_received = False

    for i in range(len(window_df) - 1):
        current_price = window_df['close'].iloc[i]

        # Check for first exit signal
        if not first_exit_signal_received and position is None:
            try:
                exit_feat_long = window_df[long_exit_feature_columns].iloc[i:i+1].values
                long_exit_prob = long_exit_model.predict_proba(long_exit_scaler.transform(exit_feat_long))[0][1]

                exit_feat_short = window_df[short_exit_feature_columns].iloc[i:i+1].values
                short_exit_prob = short_exit_model.predict_proba(short_exit_scaler.transform(exit_feat_short))[0][1]

                if long_exit_prob >= ml_exit_long or short_exit_prob >= ml_exit_short:
                    first_exit_signal_received = True
            except:
                pass

        # Entry logic
        if first_exit_signal_received and position is None:
            long_prob = window_df['long_prob'].iloc[i]
            short_prob = window_df['short_prob'].iloc[i]

            # LONG entry
            if long_prob >= long_threshold:
                sizing_result = sizer.get_position_size_simple(capital, long_prob, LEVERAGE)
                position = {
                    'side': 'LONG',
                    'entry_idx': i,
                    'entry_price': current_price,
                    'position_size_pct': sizing_result['position_size_pct'],
                    'leveraged_value': sizing_result['leveraged_value'],
                    'quantity': sizing_result['leveraged_value'] / current_price
                }

            # SHORT entry (gated)
            elif short_prob >= short_threshold:
                long_ev = long_prob * LONG_AVG_RETURN
                short_ev = short_prob * SHORT_AVG_RETURN

                if (short_ev - long_ev) > GATE_THRESHOLD:
                    sizing_result = sizer.get_position_size_simple(capital, short_prob, LEVERAGE)
                    position = {
                        'side': 'SHORT',
                        'entry_idx': i,
                        'entry_price': current_price,
                        'position_size_pct': sizing_result['position_size_pct'],
                        'leveraged_value': sizing_result['leveraged_value'],
                        'quantity': sizing_result['leveraged_value'] / current_price
                    }

        # Exit logic
        if position is not None:
            time_in_pos = i - position['entry_idx']

            # Calculate P&L
            entry_notional = position['quantity'] * position['entry_price']
            current_notional = position['quantity'] * current_price

            if position['side'] == 'LONG':
                pnl_usd = current_notional - entry_notional
                price_change_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl_usd = entry_notional - current_notional
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

            balance_loss_pct = pnl_usd / capital

            should_exit = False
            exit_reason = None

            # 1. ML Exit
            try:
                if position['side'] == 'LONG':
                    exit_feat = window_df[long_exit_feature_columns].iloc[i:i+1].values
                    exit_prob = long_exit_model.predict_proba(long_exit_scaler.transform(exit_feat))[0][1]
                    threshold = ml_exit_long
                else:
                    exit_feat = window_df[short_exit_feature_columns].iloc[i:i+1].values
                    exit_prob = short_exit_model.predict_proba(short_exit_scaler.transform(exit_feat))[0][1]
                    threshold = ml_exit_short

                if exit_prob >= threshold:
                    should_exit = True
                    exit_reason = 'ML_EXIT'
            except:
                pass

            # 2. Stop Loss
            if not should_exit and balance_loss_pct <= EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = 'STOP_LOSS'

            # 3. Max Hold
            if not should_exit and time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = 'MAX_HOLD'

            # Execute exit
            if should_exit:
                entry_commission = position['leveraged_value'] * TAKER_FEE
                exit_commission = position['quantity'] * current_price * TAKER_FEE
                net_pnl_usd = pnl_usd - entry_commission - exit_commission

                capital += net_pnl_usd

                trades.append({
                    'side': position['side'],
                    'net_pnl_usd': net_pnl_usd,
                    'exit_reason': exit_reason
                })

                position = None

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
        trades, final_capital = backtest_window(
            window_df, long_threshold, short_threshold, ml_exit_long, ml_exit_short
        )

        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            all_trades.extend(trades)

            total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

            window_results.append({
                'return_pct': total_return,
                'total_trades': len(trades),
                'win_rate': (trades_df['net_pnl_usd'] > 0).mean() * 100,
                'ml_exit_rate': (trades_df['exit_reason'] == 'ML_EXIT').mean() * 100
            })

    return pd.DataFrame(window_results), all_trades

# =============================================================================
# GRID SEARCH
# =============================================================================

print("\nStarting grid search...")
grid_start = time.time()

results = []

for entry_threshold in ENTRY_THRESHOLDS:
    for exit_threshold in EXIT_THRESHOLDS:
        combo_start = time.time()

        # Run 108-window backtest
        window_df, all_trades = run_108_windows(
            long_threshold=entry_threshold,
            short_threshold=entry_threshold,
            ml_exit_long=exit_threshold,
            ml_exit_short=exit_threshold
        )

        if len(window_df) > 0:
            trades_df = pd.DataFrame(all_trades)

            mean_return = window_df['return_pct'].mean()
            std_return = window_df['return_pct'].std()
            sharpe = (mean_return / std_return) * 8.544 if std_return > 0 else 0
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
        print(f"  [{len(results)}/{len(ENTRY_THRESHOLDS)*len(EXIT_THRESHOLDS)}] Entry={entry_threshold:.2f}, Exit={exit_threshold:.2f}: "
              f"Return={mean_return:+.2f}%, WR={win_rate:.1f}%, Sharpe={sharpe:.3f} ({combo_time:.1f}s)")

grid_time = time.time() - grid_start
print(f"\n✅ Grid search complete in {grid_time/60:.1f} minutes")

# =============================================================================
# RESULTS
# =============================================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('composite_score', ascending=False)

print("\n" + "="*80)
print("TOP 5 CONFIGURATIONS")
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

# Save
output_file = PROJECT_ROOT / "results" / f"efficient_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved: {output_file}")

print("\n" + "="*80)
