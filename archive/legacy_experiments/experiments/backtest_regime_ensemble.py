"""
Backtest Regime-Based Ensemble Strategy

비판적 실험:
"Regime-specific 모델이 general 모델보다 나은가?"

Hypothesis:
각 regime에 특화된 모델을 사용하면 general 모델보다 우수할 것

Test:
1. General XGBoost (Phase 4 Advanced)
2. Regime-Based Ensemble (Bull/Bear/Sideways models)

Expected:
Ensemble이 10-20% 더 나은 성능
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.production.regime_detector import RegimeDetector, simple_regime_detection

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
LEVERAGE = 2.0
STOP_LOSS = 0.005  # 0.5%
TAKE_PROFIT = 0.03  # 3%
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002  # 0.02%

def backtest_strategy(df, models, feature_columns, threshold, use_regime_ensemble=False):
    """
    Backtest with general or regime-specific models

    Args:
        models: dict with 'general' or regime-specific keys
        use_regime_ensemble: If True, use regime-specific models
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    detector = RegimeDetector(lookback_window=240) if use_regime_ensemble else None

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # Calculate P&L
            price_change_pct = (current_price - entry_price) / entry_price
            leveraged_pnl_pct = price_change_pct * LEVERAGE
            leveraged_pnl_usd = leveraged_pnl_pct * position['base_value']

            exit_reason = None
            if leveraged_pnl_pct <= -STOP_LOSS:
                exit_reason = "Stop Loss"
            elif leveraged_pnl_pct >= TAKE_PROFIT:
                exit_reason = "Take Profit"
            elif hours_held >= MAX_HOLDING_HOURS:
                exit_reason = "Max Holding"

            if exit_reason:
                # Transaction costs
                entry_cost = position['leveraged_value'] * TRANSACTION_COST
                exit_cost = (current_price / entry_price) * position['leveraged_value'] * TRANSACTION_COST
                total_cost = entry_cost + exit_cost

                net_pnl_usd = leveraged_pnl_usd - total_cost

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'base_value': position['base_value'],
                    'leveraged_value': position['leveraged_value'],
                    'position_size_pct': 0.95,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'leveraged_pnl_usd': leveraged_pnl_usd,
                    'pnl_usd_net': net_pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'probability': position['probability'],
                    'regime': position['regime'],
                    'model_used': position.get('model_used', 'general')
                })

                capital += net_pnl_usd
                position = None

        # Look for entry
        if position is None and i < len(df) - 1:
            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            # Determine regime
            if use_regime_ensemble:
                regime = detector.detect_regime(df, i)
                model_key = regime  # Bull, Bear, or Sideways
            else:
                regime = simple_regime_detection(df.iloc[max(0, i-20):i+1])
                model_key = 'general'

            # Get appropriate model
            if model_key in models:
                model = models[model_key]
            else:
                # Fallback to general if regime model not available
                model = models.get('general', models.get('Bull'))

            probability = model.predict_proba(features)[0][1]

            if probability <= threshold:
                continue

            # Calculate position
            position_size_pct = 0.95
            base_value = capital * position_size_pct
            leveraged_value = base_value * LEVERAGE

            position = {
                'entry_idx': i,
                'entry_price': current_price,
                'base_value': base_value,
                'leveraged_value': leveraged_value,
                'probability': probability,
                'regime': regime,
                'model_used': model_key
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
    winning_trades = [t for t in trades if t['pnl_usd_net'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100
    avg_pnl_pct = np.mean([t['leveraged_pnl_pct'] for t in trades]) * 100

    # Drawdown
    cumulative_returns = []
    running_capital = INITIAL_CAPITAL
    for trade in trades:
        running_capital += trade['pnl_usd_net']
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
        returns = [t['leveraged_pnl_pct'] for t in trades]
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

def rolling_window_backtest(df, models, feature_columns, threshold, use_regime_ensemble=False):
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = simple_regime_detection(window_df)

        trades, metrics = backtest_strategy(
            window_df, models, feature_columns, threshold, use_regime_ensemble
        )

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


print("=" * 80)
print("Regime-Based Ensemble Backtest")
print("=" * 80)
print()
print("🎯 Critical Test: Regime-specific models vs General model")
print()

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Data loaded: {len(df)} candles")

# Calculate features
print("Calculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna()
print(f"✅ Features calculated: {len(df)} rows")

# Load models
print("\nLoading models...")

# General model (Phase 4 Advanced)
general_model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(general_model_file, 'rb') as f:
    general_model = pickle.load(f)
print(f"✅ General model loaded")

# Regime-specific models
regime_models = {}
for regime in ['bull', 'bear', 'sideways']:
    model_file = MODELS_DIR / f"xgboost_regime_{regime}.pkl"
    if model_file.exists():
        with open(model_file, 'rb') as f:
            regime_models[regime.capitalize()] = pickle.load(f)
        print(f"✅ {regime.capitalize()} model loaded")

# Feature columns
feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(feature_file, 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]
print(f"✅ Features: {len(feature_columns)}")

threshold = 0.7  # Best from previous backtest

print(f"\n{'=' * 80}")
print(f"Backtesting with Threshold {threshold:.1f}")
print(f"{'=' * 80}")

# Test 1: General model
print(f"\n1. BASELINE: General XGBoost (Phase 4 Advanced)")
general_results = rolling_window_backtest(
    df,
    {'general': general_model},
    feature_columns,
    threshold,
    use_regime_ensemble=False
)

print(f"\nResults ({len(general_results)} windows):")
print(f"  Avg Return: {general_results['return'].mean():.2f}%")
print(f"  vs B&H: {general_results['difference'].mean():.2f}%")
print(f"  Win Rate: {general_results['win_rate'].mean():.1f}%")
print(f"  Sharpe: {general_results['sharpe'].mean():.3f}")
print(f"  Max DD: {general_results['max_dd'].mean():.2f}%")

# Test 2: Regime-based ensemble
print(f"\n2. ADVANCED: Regime-Based Ensemble")
ensemble_results = rolling_window_backtest(
    df,
    regime_models,
    feature_columns,
    threshold,
    use_regime_ensemble=True
)

print(f"\nResults ({len(ensemble_results)} windows):")
print(f"  Avg Return: {ensemble_results['return'].mean():.2f}%")
print(f"  vs B&H: {ensemble_results['difference'].mean():.2f}%")
print(f"  Win Rate: {ensemble_results['win_rate'].mean():.1f}%")
print(f"  Sharpe: {ensemble_results['sharpe'].mean():.3f}")
print(f"  Max DD: {ensemble_results['max_dd'].mean():.2f}%")

# Comparison
print(f"\n{'=' * 80}")
print("COMPARISON: Ensemble vs General")
print(f"{'=' * 80}")

return_diff = ensemble_results['return'].mean() - general_results['return'].mean()
sharpe_diff = ensemble_results['sharpe'].mean() - general_results['sharpe'].mean()
dd_diff = general_results['max_dd'].mean() - ensemble_results['max_dd'].mean()
wr_diff = ensemble_results['win_rate'].mean() - general_results['win_rate'].mean()

print(f"\nReturn Difference: {return_diff:+.2f}%")
if return_diff > 0:
    print(f"  ✅ Ensemble better (+{(return_diff/general_results['return'].mean())*100:.1f}% improvement)")
else:
    print(f"  ⚠️ General better")

print(f"\nSharpe Ratio Difference: {sharpe_diff:+.3f}")
print(f"  {'✅ Ensemble better' if sharpe_diff > 0 else '⚠️ General better'}")

print(f"\nDrawdown Reduction: {dd_diff:+.2f}%")
print(f"  {'✅ Ensemble reduces DD' if dd_diff > 0 else '⚠️ General has lower DD'}")

print(f"\nWin Rate Difference: {wr_diff:+.1f}%p")
print(f"  {'✅ Ensemble better' if wr_diff > 0 else '⚠️ General better'}")

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_rel(ensemble_results['return'], general_results['return'])
print(f"\nStatistical Test (Return):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant: {'✅ Yes' if p_value < 0.05 else '❌ No'}")

# Save results
output_file_general = RESULTS_DIR / "backtest_general_xgboost.csv"
general_results.to_csv(output_file_general, index=False)
print(f"\n✅ General results saved: {output_file_general}")

output_file_ensemble = RESULTS_DIR / "backtest_regime_ensemble.csv"
ensemble_results.to_csv(output_file_ensemble, index=False)
print(f"✅ Ensemble results saved: {output_file_ensemble}")

print(f"\n{'=' * 80}")
print("Backtest Complete!")
print(f"{'=' * 80}")

print(f"\n🎯 비판적 결론:")
print(f"\n  General XGBoost:")
print(f"    - Avg Return: {general_results['return'].mean():.2f}%")
print(f"    - Sharpe: {general_results['sharpe'].mean():.3f}")
print(f"    - Max DD: {general_results['max_dd'].mean():.2f}%")

print(f"\n  Regime-Based Ensemble:")
print(f"    - Avg Return: {ensemble_results['return'].mean():.2f}%")
print(f"    - Sharpe: {ensemble_results['sharpe'].mean():.3f}")
print(f"    - Max DD: {ensemble_results['max_dd'].mean():.2f}%")

if return_diff > 0 and sharpe_diff > 0:
    improvement = (return_diff / general_results['return'].mean()) * 100
    print(f"\n  ✅ Regime-Based Ensemble 성공!")
    print(f"     Return: +{return_diff:.2f}%p ({improvement:+.1f}% improvement)")
    print(f"     Sharpe: +{sharpe_diff:.3f}")
    print(f"     → Regime-specific models가 더 나음!")
elif return_diff < -0.5:
    print(f"\n  ⚠️ Ensemble이 general보다 나쁨")
    print(f"     Return diff: {return_diff:.2f}%")
    print(f"     → General model이 더 robust함")
else:
    print(f"\n  ⚖️ 성능 차이 미미")
    print(f"     Return diff: {return_diff:.2f}%")
    print(f"     → Ensemble의 추가 복잡도가 가치 없음")
