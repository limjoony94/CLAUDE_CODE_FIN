"""
Backtest XGBoost Phase 4 with TUNED Lag Features

Critical Question:
"Does hyperparameter tuning fix the lag features performance problem?"

Comparison:
- Phase 4 Base: F1=0.089, Returns=7.68%/5d
- Phase 4 Lag (Untuned): F1=0.046, Returns=2.38%/5d (69% worse!)
- Phase 4 Lag (Tuned): F1=0.075, Returns=? (need to validate)

Hypothesis: Proper hyperparameters unlock lag features' potential
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from loguru import logger

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.production.lag_features import LagFeatureGenerator

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
STOP_LOSS = 0.01
TAKE_PROFIT = 0.03
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002


def classify_market_regime(df_window):
    """Classify market regime"""
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100
    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"


def backtest_strategy(df, model, feature_columns, entry_threshold):
    """Backtest trading strategy"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            pnl_pct = (current_price - entry_price) / entry_price

            exit_reason = None
            if pnl_pct <= -STOP_LOSS:
                exit_reason = "Stop Loss"
            elif pnl_pct >= TAKE_PROFIT:
                exit_reason = "Take Profit"
            elif hours_held >= MAX_HOLDING_HOURS:
                exit_reason = "Max Holding"

            if exit_reason:
                quantity = position['quantity']
                pnl_usd = pnl_pct * (entry_price * quantity)

                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                total_cost = entry_cost + exit_cost
                pnl_usd -= total_cost

                capital += pnl_usd

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': quantity,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'probability': position['probability']
                })

                position = None

        # Look for entry
        if position is None and i < len(df) - 1:
            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            probability = model.predict_proba(features)[0][1]

            should_enter = (probability > entry_threshold)

            if should_enter:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'probability': probability
                }

    # Calculate metrics
    if len(trades) == 0:
        return trades, {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_pnl_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100
    avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades]) * 100

    # Drawdown
    cumulative_returns = []
    running_capital = INITIAL_CAPITAL
    for trade in trades:
        running_capital += trade['pnl_usd']
        cumulative_returns.append(running_capital)

    if len(cumulative_returns) > 0:
        peak = cumulative_returns[0]
        max_dd = 0
        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd
    else:
        max_dd = 0.0

    # Sharpe
    if len(trades) > 1:
        returns = [t['pnl_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0

    metrics = {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_pnl_pct': avg_pnl_pct,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }

    return trades, metrics


def rolling_window_backtest(df, model, feature_columns, entry_threshold):
    """Rolling window backtest"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        trades, metrics = backtest_strategy(window_df, model, feature_columns, entry_threshold)

        # Buy & Hold
        bh_start = window_df['close'].iloc[0]
        bh_end = window_df['close'].iloc[-1]
        bh_return = ((bh_end - bh_start) / bh_start) * 100
        bh_cost = 2 * TRANSACTION_COST * 100
        bh_return -= bh_cost

        windows.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'regime': regime,
            'return': metrics['total_return_pct'],
            'bh_return': bh_return,
            'difference': metrics['total_return_pct'] - bh_return,
            'num_trades': metrics['num_trades'],
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe_ratio'],
            'max_dd': metrics['max_drawdown']
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)


logger.info("=" * 80)
logger.info("XGBoost Phase 4 TUNED Lag Features Backtest")
logger.info("="* 80)

# Load TUNED lag-feature model
model_file = MODELS_DIR / "xgboost_v4_phase4_lag_tuned_lookahead3_thresh3.pkl"
with open(model_file, 'rb') as f:
    model = pickle.load(f)
logger.success(f"✅ Tuned lag model loaded: {model_file}")

# Load feature columns
feature_file = MODELS_DIR / "xgboost_v4_phase4_lag_tuned_lookahead3_thresh3_features.txt"
with open(feature_file, 'r') as f:
    temporal_feature_columns = [line.strip() for line in f.readlines()]
logger.success(f"✅ Features loaded: {len(temporal_feature_columns)} temporal features")

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
logger.success(f"✅ Data loaded: {len(df)} candles")

# Calculate features
logger.info("Calculating baseline features...")
df = calculate_features(df)

logger.info("Calculating advanced technical features...")
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)

# Create lag features
logger.info("Creating lag features...")
base_feature_columns = [
    'close_change_1', 'close_change_3', 'volume_ma_ratio',
    'rsi', 'macd', 'macd_signal', 'macd_diff',
    'bb_high', 'bb_mid', 'bb_low'
] + adv_features.get_feature_names()

lag_gen = LagFeatureGenerator(lag_periods=[1, 2])
df_temporal, _ = lag_gen.create_all_temporal_features(df, base_feature_columns, include_momentum=True)

logger.success(f"✅ Temporal features created: {len(df_temporal)} rows")

# Handle NaN
df_temporal = df_temporal.ffill()
df_temporal = df_temporal.dropna()
logger.info(f"After dropna: {len(df_temporal)} rows")

# Test thresholds
thresholds = [0.3, 0.5, 0.7]

all_results = {}

for threshold in thresholds:
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Testing TUNED Lag Model with threshold {threshold:.2f}")
    logger.info(f"{'=' * 80}")

    results = rolling_window_backtest(df_temporal, model, temporal_feature_columns, threshold)
    all_results[threshold] = results

    # Summary
    logger.info(f"\nResults ({len(results)} windows):")
    logger.info(f"  Tuned Lag Return: {results['return'].mean():.2f}% ± {results['return'].std():.2f}%")
    logger.info(f"  Buy & Hold Return: {results['bh_return'].mean():.2f}% ± {results['bh_return'].std():.2f}%")
    logger.info(f"  Difference: {results['difference'].mean():.2f}% ± {results['difference'].std():.2f}%")
    logger.info(f"  🎯 Avg Trades: {results['num_trades'].mean():.1f}")
    logger.info(f"  Avg Win Rate: {results['win_rate'].mean():.1f}%")
    logger.info(f"  Avg Sharpe: {results['sharpe'].mean():.3f}")
    logger.info(f"  Avg Max DD: {results['max_dd'].mean():.2f}%")

    # Save
    output_file = RESULTS_DIR / f"backtest_phase4_lag_tuned_thresh{int(threshold*10)}.csv"
    results.to_csv(output_file, index=False)
    logger.success(f"\n✅ Results saved: {output_file}")

# Best threshold
best_threshold = max(thresholds, key=lambda t: all_results[t]['difference'].mean())
best_results = all_results[best_threshold]

logger.info(f"\n{'=' * 80}")
logger.info("CRITICAL COMPARISON")
logger.info(f"{'=' * 80}")

logger.info(f"\n📊 Phase 4 Base (37 features, no lags):")
logger.info(f"    - F1: 0.089")
logger.info(f"    - Returns: +7.68% per 5 days")
logger.info(f"    - Win Rate: 69.1%")
logger.info(f"    - Threshold: 0.7")

logger.info(f"\n⚠️ Phase 4 Lag UNTUNED (185 features, wrong params):")
logger.info(f"    - F1: 0.046 (-48%)")
logger.info(f"    - Returns: +2.38% per 5 days (-69%)")
logger.info(f"    - Win Rate: 75.3%")
logger.info(f"    - Threshold: 0.7")

logger.info(f"\n✅ Phase 4 Lag TUNED (185 features, optimized params):")
logger.info(f"    - F1: 0.075 (+63% vs untuned)")
logger.info(f"    - Best Threshold: {best_threshold:.1f}")
logger.info(f"    - Returns: {best_results['difference'].mean():+.2f}% per 5 days")
logger.info(f"    - Win Rate: {best_results['win_rate'].mean():.1f}%")
logger.info(f"    - Trades: {best_results['num_trades'].mean():.1f}")

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_rel(best_results['return'], best_results['bh_return'])
logger.info(f"\n  Statistical Significance:")
logger.info(f"    - t-statistic: {t_stat:.4f}")
logger.info(f"    - p-value: {p_value:.4f}")
logger.info(f"    - Significant: {'✅ Yes' if p_value < 0.05 else '❌ No'}")

# Final decision
improvement_vs_untuned = ((best_results['difference'].mean() - 2.38) / 2.38) * 100
improvement_vs_base = ((best_results['difference'].mean() - 7.68) / 7.68) * 100

logger.info(f"\n{'=' * 80}")
logger.info("FINAL VERDICT")
logger.info(f"{'=' * 80}")

if best_results['difference'].mean() > 7.68:
    logger.success(f"\n🎉 TUNED LAG FEATURES WIN!")
    logger.success(f"  Improvement vs Base: {improvement_vs_base:+.1f}%")
    logger.success(f"  Returns: {best_results['difference'].mean():.2f}% vs 7.68% (Base)")
    logger.success(f"  F1 improvement: 0.075 vs 0.089 (Base)")
    logger.success(f"\n  → DEPLOY TUNED LAG MODEL TO TESTNET!")
elif best_results['difference'].mean() > 2.38:
    logger.warning(f"\n⚡ TUNING HELPED BUT NOT ENOUGH")
    logger.warning(f"  Improvement vs Untuned: {improvement_vs_untuned:+.1f}%")
    logger.warning(f"  Still worse than Base: {improvement_vs_base:.1f}%")
    logger.warning(f"  Returns: {best_results['difference'].mean():.2f}% vs 7.68% (Base)")
    logger.warning(f"\n  Conclusion:")
    logger.warning(f"    - Hyperparameter tuning DID improve F1 (0.046 → 0.075)")
    logger.warning(f"    - But trading returns still worse than base model")
    logger.warning(f"    - Problem: Lag features may not capture useful patterns")
    logger.warning(f"\n  → KEEP BASE MODEL, TRY DIFFERENT APPROACH")
else:
    logger.error(f"\n❌ TUNING DIDN'T HELP")
    logger.error(f"  Change vs Untuned: {improvement_vs_untuned:+.1f}%")
    logger.error(f"  Returns: {best_results['difference'].mean():.2f}% vs 2.38% (Untuned)")
    logger.error(f"\n  → KEEP BASE MODEL, LAG FEATURES NOT THE SOLUTION")
