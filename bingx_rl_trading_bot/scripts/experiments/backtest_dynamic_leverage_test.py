"""
비판적 사고: Dynamic Sizing + Leverage 조합 테스트

가설:
- Dynamic Sizing (평균 56.3%)이 수익 기회를 놓침
- 레버리지를 높이면 수익률을 회복하면서도 리스크 관리 유지 가능

테스트:
1. Dynamic @ 2x (baseline)
2. Dynamic @ 3x
3. Dynamic @ 4x
4. Dynamic @ 5x

비교 지표:
- 수익률 vs Fixed 95% @ 2x (7.68%)
- Max Drawdown (청산 리스크)
- Sharpe Ratio (위험 대비 수익)
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
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
STOP_LOSS = 0.01  # 1%
TAKE_PROFIT = 0.03  # 3%
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002

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

def backtest_with_leverage(df, model, feature_columns, threshold, leverage):
    """
    Dynamic Position Sizing with variable leverage
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []
    liquidations = 0

    # Dynamic sizer
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

        # Manage position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # P&L with leverage
            price_change_pct = (current_price - entry_price) / entry_price
            leveraged_pnl_pct = price_change_pct * leverage
            leveraged_pnl_usd = leveraged_pnl_pct * position['base_value']

            # Liquidation check (leverage-dependent)
            liquidation_threshold = -0.95 / leverage  # e.g., 2x = -47.5%, 5x = -19%
            if leveraged_pnl_pct <= liquidation_threshold:
                # LIQUIDATION!
                liquidations += 1
                exit_reason = "LIQUIDATION"

                # Lose entire position
                leveraged_pnl_usd = -position['base_value']
                net_pnl_usd = leveraged_pnl_usd

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
                continue

            # Normal exits
            exit_reason = None
            if leveraged_pnl_pct <= -STOP_LOSS:
                exit_reason = "Stop Loss"
            elif leveraged_pnl_pct >= TAKE_PROFIT:
                exit_reason = "Take Profit"
            elif hours_held >= MAX_HOLDING_HOURS:
                exit_reason = "Max Holding"

            if exit_reason:
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

        # Entry logic
        if position is None and i < len(df) - 1:
            if capital <= 0:
                break  # Account blown

            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            probability = model.predict_proba(features)[0][1]

            if probability <= threshold:
                continue

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
                leverage=leverage
            )

            position_size_pct = sizing_result['position_size_pct']
            base_value = sizing_result['position_value']
            leveraged_value = sizing_result['leveraged_value']

            position = {
                'entry_idx': i,
                'entry_price': current_price,
                'base_value': base_value,
                'leveraged_value': leveraged_value,
                'position_size_pct': position_size_pct,
                'probability': probability,
                'regime': current_regime
            }

    # Metrics
    if len(trades) == 0:
        return trades, {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'liquidations': 0,
            'final_capital': capital
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd_net'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100

    # Drawdown
    cumulative_returns = []
    running_capital = INITIAL_CAPITAL
    for trade in trades:
        running_capital += trade['pnl_usd_net']
        cumulative_returns.append(running_capital)

    peak = cumulative_returns[0]
    max_dd = 0
    for value in cumulative_returns:
        if value > peak:
            peak = value
        dd = ((peak - value) / peak) * 100
        if dd > max_dd:
            max_dd = dd

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
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'liquidations': liquidations,
        'final_capital': capital
    }

    return trades, metrics

def rolling_window_backtest(df, model, feature_columns, threshold, leverage):
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        trades, metrics = backtest_with_leverage(
            window_df, model, feature_columns, threshold, leverage
        )

        # Buy & Hold
        bh_start = window_df['close'].iloc[0]
        bh_end = window_df['close'].iloc[-1]
        bh_return = ((bh_end - bh_start) / bh_start) * 100
        bh_cost = 2 * TRANSACTION_COST * 100
        bh_return -= bh_cost

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
            'liquidations': metrics['liquidations'],
            'final_capital': metrics['final_capital'],
            'avg_position_size': avg_position_size
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)


print("=" * 80)
print("Dynamic Position Sizing: Leverage Optimization Test")
print("=" * 80)

# Load model
model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(model_file, 'rb') as f:
    model = pickle.load(f)
print(f"✅ Model loaded")

# Load features
feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(feature_file, 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]
print(f"✅ Features loaded: {len(feature_columns)}")

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Data loaded: {len(df)} candles")

# Features
print("Calculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna()
print(f"✅ Features ready: {len(df)} rows")

threshold = 0.7

# Test different leverage levels
leverage_levels = [2, 3, 4, 5]
results_all = {}

print(f"\n{'=' * 80}")
print("TESTING LEVERAGE LEVELS")
print(f"{'=' * 80}")

for lev in leverage_levels:
    print(f"\n{'=' * 80}")
    print(f"Testing Leverage: {lev}x")
    print(f"{'=' * 80}")

    results = rolling_window_backtest(df, model, feature_columns, threshold, lev)
    results_all[lev] = results

    total_liquidations = results['liquidations'].sum()
    blown_accounts = (results['final_capital'] <= 0).sum()

    print(f"\nResults ({len(results)} windows):")
    print(f"  Avg Return: {results['return'].mean():.2f}%")
    print(f"  vs B&H: {results['difference'].mean():.2f}%")
    print(f"  Win Rate: {results['win_rate'].mean():.1f}%")
    print(f"  Sharpe: {results['sharpe'].mean():.3f}")
    print(f"  Max DD: {results['max_dd'].mean():.2f}%")
    print(f"  Avg Position: {results['avg_position_size'].mean():.1f}%")
    print(f"  Liquidations: {total_liquidations} {'🚨' if total_liquidations > 0 else '✅'}")
    print(f"  Blown Accounts: {blown_accounts} {'🚨' if blown_accounts > 0 else '✅'}")

    # Save
    output_file = RESULTS_DIR / f"backtest_dynamic_leverage_{lev}x.csv"
    results.to_csv(output_file, index=False)
    print(f"  Saved: {output_file.name}")

# Comparison
print(f"\n{'=' * 80}")
print("LEVERAGE COMPARISON")
print(f"{'=' * 80}")

print(f"\n{'Leverage':<10} {'Return':<10} {'vs B&H':<10} {'Sharpe':<10} {'Max DD':<10} {'Liquidations':<15}")
print("-" * 80)

for lev in leverage_levels:
    r = results_all[lev]
    liq_total = r['liquidations'].sum()
    print(f"{lev}x{'':<8} {r['return'].mean():>7.2f}%  {r['difference'].mean():>7.2f}%  {r['sharpe'].mean():>8.3f}  {r['max_dd'].mean():>7.2f}%  {liq_total:>5} {'🚨' if liq_total > 0 else '✅'}")

# Baseline comparison
print(f"\n{'=' * 80}")
print("COMPARISON vs FIXED 95% @ 2x (7.68%)")
print(f"{'=' * 80}")

baseline_return = 7.68

for lev in leverage_levels:
    r = results_all[lev]
    avg_return = r['return'].mean()
    diff = avg_return - baseline_return
    liq_total = r['liquidations'].sum()

    status = "✅" if diff >= 0 and liq_total == 0 else "⚠️" if diff >= -1 and liq_total == 0 else "🚨"

    print(f"\nDynamic @ {lev}x: {avg_return:.2f}% (diff: {diff:+.2f}%) {status}")
    print(f"  Liquidations: {liq_total}")
    print(f"  Max DD: {r['max_dd'].mean():.2f}%")

# Recommendation
print(f"\n{'=' * 80}")
print("🎯 비판적 분석 및 권장사항")
print(f"{'=' * 80}")

best_leverage = None
best_score = -999

for lev in leverage_levels:
    r = results_all[lev]
    avg_return = r['return'].mean()
    max_dd = r['max_dd'].mean()
    liq_total = r['liquidations'].sum()

    # Scoring: return - DD penalty - liquidation penalty
    score = avg_return - (max_dd * 0.5) - (liq_total * 10)

    if liq_total == 0 and score > best_score:
        best_score = score
        best_leverage = lev

if best_leverage:
    r = results_all[best_leverage]
    print(f"\n✅ 최적 레버리지: {best_leverage}x")
    print(f"   평균 수익률: {r['return'].mean():.2f}%")
    print(f"   vs Baseline: {r['return'].mean() - baseline_return:+.2f}%")
    print(f"   Max DD: {r['max_dd'].mean():.2f}%")
    print(f"   Liquidations: {r['liquidations'].sum()}")
    print(f"   Sharpe: {r['sharpe'].mean():.3f}")

    if r['return'].mean() >= baseline_return:
        print(f"\n   ✅ Dynamic + {best_leverage}x가 Fixed 95% @ 2x보다 우수!")
    elif r['return'].mean() >= baseline_return * 0.9:
        print(f"\n   ⚠️ Dynamic + {best_leverage}x가 근접 (90%+ of baseline)")
    else:
        print(f"\n   ⚠️ Fixed 95% @ 2x가 여전히 더 나음")
else:
    print(f"\n🚨 모든 레버리지에서 청산 발생! Dynamic + Leverage 조합 위험")

print(f"\n{'=' * 80}")
print("테스트 완료!")
print(f"{'=' * 80}")
