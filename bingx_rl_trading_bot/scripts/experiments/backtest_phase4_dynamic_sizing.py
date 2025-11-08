"""
Backtest Phase 4 Advanced with Dynamic Position Sizing

비판적 사고:
"고정 95% vs 동적 조절 - 실제로 성과가 개선되는가?"

Tests:
1. Fixed 95% position size (baseline)
2. Dynamic position sizing (signal + volatility + regime + streak)

Expected:
- Dynamic sizing should reduce drawdowns
- Better risk-adjusted returns (Sharpe ratio)
- Lower maximum loss in poor conditions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

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
TRANSACTION_COST = 0.0002  # 0.02% maker fee

def classify_market_regime(df_window):
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100
    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"

def backtest_with_dynamic_sizing(df, model, feature_columns, threshold, use_dynamic=True):
    """
    Backtest with dynamic or fixed position sizing

    Args:
        use_dynamic: True = dynamic sizing, False = fixed 95%
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    # Initialize dynamic position sizer
    if use_dynamic:
        sizer = DynamicPositionSizer(
            base_position_pct=0.50,
            max_position_pct=0.95,
            min_position_pct=0.20,
            signal_weight=0.4,
            volatility_weight=0.3,
            regime_weight=0.2,
            streak_weight=0.1
        )

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # Calculate P&L (with leverage)
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
                    'position_size_pct': position['position_size_pct'],
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'leveraged_pnl_usd': leveraged_pnl_usd,
                    'pnl_usd_net': net_pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'probability': position['probability'],
                    'regime': position['regime']
                })

                capital += net_pnl_usd
                position = None

        # Look for entry
        if position is None and i < len(df) - 1:
            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            probability = model.predict_proba(features)[0][1]

            if probability <= threshold:
                continue

            # Calculate position size
            if use_dynamic:
                # Dynamic sizing
                current_regime = classify_market_regime(df.iloc[max(0, i-20):i+1])
                current_volatility = df['atr_pct'].iloc[i] if 'atr_pct' in df.columns else 0.01
                avg_volatility = df['atr_pct'].iloc[max(0, i-50):i].mean() if 'atr_pct' in df.columns else 0.01

                sizing_result = sizer.calculate_position_size(
                    capital=capital,
                    signal_strength=probability,
                    current_volatility=current_volatility,
                    avg_volatility=avg_volatility,
                    market_regime=current_regime,
                    recent_trades=trades[-10:] if len(trades) > 0 else [],
                    leverage=LEVERAGE
                )

                position_size_pct = sizing_result['position_size_pct']
                base_value = sizing_result['position_value']
                leveraged_value = sizing_result['leveraged_value']
            else:
                # Fixed 95%
                position_size_pct = 0.95
                base_value = capital * position_size_pct
                leveraged_value = base_value * LEVERAGE
                current_regime = classify_market_regime(df.iloc[max(0, i-20):i+1])

            position = {
                'entry_idx': i,
                'entry_price': current_price,
                'base_value': base_value,
                'leveraged_value': leveraged_value,
                'position_size_pct': position_size_pct,
                'probability': probability,
                'regime': current_regime
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

def rolling_window_backtest(df, model, feature_columns, threshold, use_dynamic=True):
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        trades, metrics = backtest_with_dynamic_sizing(
            window_df, model, feature_columns, threshold, use_dynamic=use_dynamic
        )

        # Buy & Hold
        bh_start = window_df['close'].iloc[0]
        bh_end = window_df['close'].iloc[-1]
        bh_return = ((bh_end - bh_start) / bh_start) * 100
        bh_cost = 2 * TRANSACTION_COST * 100
        bh_return -= bh_cost

        # Calculate average position size
        avg_position_size = np.mean([t['position_size_pct'] for t in trades]) * 100 if len(trades) > 0 else 0

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
            'max_dd': metrics['max_drawdown'],
            'avg_position_size': avg_position_size
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)


print("=" * 80)
print("Phase 4 Advanced: Fixed vs Dynamic Position Sizing Backtest")
print("=" * 80)

# Load model
model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(model_file, 'rb') as f:
    model = pickle.load(f)
print(f"✅ Model loaded: {model_file}")

# Load feature columns
feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(feature_file, 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]
print(f"✅ Features loaded: {len(feature_columns)} features")

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Data loaded: {len(df)} candles")

# Calculate baseline features (Phase 2)
print("Calculating baseline features...")
df = calculate_features(df)

# Calculate advanced features
print("Calculating advanced technical features...")
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)

# Handle NaN
df = df.ffill()
df = df.dropna()
print(f"✅ Features calculated: {len(df)} rows after dropna")

threshold = 0.7  # Best from previous backtest

print(f"\n{'=' * 80}")
print(f"Testing with Threshold {threshold:.1f}")
print(f"{'=' * 80}")

# Test 1: Fixed 95% position size
print(f"\n1. BASELINE: Fixed 95% Position Size")
results_fixed = rolling_window_backtest(df, model, feature_columns, threshold, use_dynamic=False)

print(f"\nResults ({len(results_fixed)} windows):")
print(f"  Avg Return: {results_fixed['return'].mean():.2f}%")
print(f"  vs B&H: {results_fixed['difference'].mean():.2f}%")
print(f"  Win Rate: {results_fixed['win_rate'].mean():.1f}%")
print(f"  Sharpe: {results_fixed['sharpe'].mean():.3f}")
print(f"  Max DD: {results_fixed['max_dd'].mean():.2f}%")
print(f"  Avg Position: {results_fixed['avg_position_size'].mean():.1f}%")

# Test 2: Dynamic position sizing
print(f"\n2. IMPROVED: Dynamic Position Sizing")
results_dynamic = rolling_window_backtest(df, model, feature_columns, threshold, use_dynamic=True)

print(f"\nResults ({len(results_dynamic)} windows):")
print(f"  Avg Return: {results_dynamic['return'].mean():.2f}%")
print(f"  vs B&H: {results_dynamic['difference'].mean():.2f}%")
print(f"  Win Rate: {results_dynamic['win_rate'].mean():.1f}%")
print(f"  Sharpe: {results_dynamic['sharpe'].mean():.3f}")
print(f"  Max DD: {results_dynamic['max_dd'].mean():.2f}%")
print(f"  Avg Position: {results_dynamic['avg_position_size'].mean():.1f}%")

# Comparison
print(f"\n{'=' * 80}")
print("COMPARISON: Dynamic vs Fixed")
print(f"{'=' * 80}")

return_diff = results_dynamic['return'].mean() - results_fixed['return'].mean()
sharpe_diff = results_dynamic['sharpe'].mean() - results_fixed['sharpe'].mean()
dd_diff = results_fixed['max_dd'].mean() - results_dynamic['max_dd'].mean()  # Positive = improvement
wr_diff = results_dynamic['win_rate'].mean() - results_fixed['win_rate'].mean()

print(f"\nReturn Difference: {return_diff:+.2f}%")
print(f"  {'✅ Dynamic better' if return_diff > 0 else '⚠️ Fixed better'}")

print(f"\nSharpe Ratio Difference: {sharpe_diff:+.3f}")
print(f"  {'✅ Dynamic better' if sharpe_diff > 0 else '⚠️ Fixed better'}")

print(f"\nDrawdown Reduction: {dd_diff:+.2f}%")
print(f"  {'✅ Dynamic reduces DD' if dd_diff > 0 else '⚠️ Fixed has lower DD'}")

print(f"\nWin Rate Difference: {wr_diff:+.1f}%p")
print(f"  {'✅ Dynamic better' if wr_diff > 0 else '⚠️ Fixed better'}")

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_rel(results_dynamic['return'], results_fixed['return'])
print(f"\nStatistical Test (Return):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant: {'✅ Yes' if p_value < 0.05 else '❌ No'}")

# Save results
output_file_fixed = RESULTS_DIR / "backtest_phase4_fixed95_sizing.csv"
results_fixed.to_csv(output_file_fixed, index=False)
print(f"\n✅ Fixed sizing results saved: {output_file_fixed}")

output_file_dynamic = RESULTS_DIR / "backtest_phase4_dynamic_sizing.csv"
results_dynamic.to_csv(output_file_dynamic, index=False)
print(f"✅ Dynamic sizing results saved: {output_file_dynamic}")

print(f"\n{'=' * 80}")
print("Backtest Complete!")
print(f"{'=' * 80}")

print(f"\n🎯 비판적 분석:")
print(f"\n  Fixed 95%:")
print(f"    - Avg Return: {results_fixed['return'].mean():.2f}%")
print(f"    - Sharpe: {results_fixed['sharpe'].mean():.3f}")
print(f"    - Max DD: {results_fixed['max_dd'].mean():.2f}%")

print(f"\n  Dynamic (20-95%):")
print(f"    - Avg Return: {results_dynamic['return'].mean():.2f}%")
print(f"    - Sharpe: {results_dynamic['sharpe'].mean():.3f}")
print(f"    - Max DD: {results_dynamic['max_dd'].mean():.2f}%")

if sharpe_diff > 0 and dd_diff > 0:
    print(f"\n  ✅ Dynamic sizing 성공: Better risk-adjusted returns!")
    print(f"     Sharpe: {sharpe_diff:+.3f}, DD reduction: {dd_diff:+.2f}%")
elif return_diff > 0.5:
    print(f"\n  ✅ Dynamic sizing 부분 성공: Higher returns (+{return_diff:.2f}%)")
else:
    print(f"\n  ⚠️ Dynamic sizing 효과 제한적")
    print(f"     고정 95%와 큰 차이 없음 (return diff: {return_diff:+.2f}%)")
