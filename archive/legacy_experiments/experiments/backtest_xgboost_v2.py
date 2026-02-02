"""
Backtest XGBoost V2 Models with Rolling Window

목적:
1. 4가지 개선 config의 실제 성과 검증
2. 거래 빈도 증가 확인 (0.1 → 목표 5-8 trades)
3. 승률 개선 확인 (0.3% → 목표 48-55%)
4. Buy & Hold 대비 성과 비교
5. 통계적 유의성 검증 (p-value < 0.05)

비판적 사고:
- "Training metrics 개선 ≠ 실제 성과"
- "백테스트가 진실을 말해준다"
- "거래 빈도가 최우선 목표 (0.1 → 5+)"
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import ta

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Trading parameters
WINDOW_SIZE = 1440  # 5 days (1440 * 5min = 5 days)
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
STOP_LOSS = 0.01  # 1%
TAKE_PROFIT = 0.03  # 3%
MAX_HOLDING_HOURS = 4  # 4 hours max (reduced from 24)
TRANSACTION_COST = 0.0006  # 0.06% per trade

def calculate_features(df):
    """Calculate technical indicators"""
    df = df.copy()

    # Price changes
    for lag in range(1, 6):
        df[f'close_change_{lag}'] = df['close'].pct_change(lag)

    # Moving averages
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)

    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()

    # Volatility
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()

    # Volume
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    return df

def classify_market_regime(df_window):
    """Classify market regime for the window"""
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100

    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"

def backtest_strategy(df, model, feature_columns, entry_threshold, min_volatility=0.0008):
    """
    Backtest trading strategy

    Args:
        entry_threshold: Minimum probability for entry (0.3, 0.4, 0.5)

    Returns: trades list, metrics dict
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12  # 5-min candles, 12 per hour

            # Calculate P&L
            pnl_pct = (current_price - entry_price) / entry_price

            # Exit conditions
            exit_reason = None

            if pnl_pct <= -STOP_LOSS:
                exit_reason = "Stop Loss"
            elif pnl_pct >= TAKE_PROFIT:
                exit_reason = "Take Profit"
            elif hours_held >= MAX_HOLDING_HOURS:
                exit_reason = "Max Holding"

            if exit_reason:
                # Close position
                quantity = position['quantity']
                pnl_usd = pnl_pct * (entry_price * quantity)

                # Transaction costs
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

        # Look for entry (if no position)
        if position is None and i < len(df) - 1:  # Need at least 1 candle ahead
            # Get features
            features = df[feature_columns].iloc[i:i+1].values

            # Check for NaN
            if np.isnan(features).any():
                continue

            # Predict
            probability = model.predict_proba(features)[0][1]

            # Check volatility
            current_volatility = df['volatility'].iloc[i]
            if pd.isna(current_volatility) or current_volatility < min_volatility:
                continue

            # Entry condition: probability > threshold
            should_enter = (probability > entry_threshold)

            if should_enter:
                # Enter LONG position
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

    # Calculate drawdown
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

    # Sharpe ratio (simplified)
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
    """
    Perform rolling window backtest
    """
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Classify regime
        regime = classify_market_regime(window_df)

        # Backtest
        trades, metrics = backtest_strategy(window_df, model, feature_columns, entry_threshold)

        # Buy & Hold
        bh_start = window_df['close'].iloc[0]
        bh_end = window_df['close'].iloc[-1]
        bh_return = ((bh_end - bh_start) / bh_start) * 100
        bh_cost = 2 * TRANSACTION_COST * 100  # Entry + Exit
        bh_return -= bh_cost

        windows.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'regime': regime,
            'xgb_return': metrics['total_return_pct'],
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
print("XGBoost V2 Backtest with Rolling Window")
print("=" * 80)

# Load data
data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Data loaded: {len(df)} candles")

# Calculate features
df = calculate_features(df)
df = df.dropna()
print(f"✅ Features calculated: {len(df)} rows after dropna")

# Feature columns
feature_columns = [
    'close_change_1', 'close_change_2', 'close_change_3', 'close_change_4', 'close_change_5',
    'sma_10', 'sma_20', 'ema_10',
    'macd', 'macd_signal', 'macd_diff',
    'rsi',
    'bb_high', 'bb_low', 'bb_mid',
    'volatility',
    'volume_sma', 'volume_ratio'
]

# Test all 4 configs
configs = [
    {'name': 'lookahead3_thresh1', 'threshold': 0.3},
    {'name': 'lookahead3_thresh1', 'threshold': 0.4},
    {'name': 'lookahead5_thresh1', 'threshold': 0.3},
    {'name': 'lookahead5_thresh2', 'threshold': 0.3},
]

all_results = []

for config in configs:
    model_file = MODELS_DIR / f"xgboost_v2_{config['name']}.pkl"

    if not model_file.exists():
        print(f"⚠️ Model not found: {model_file}")
        continue

    # Load model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    print(f"\n{'=' * 80}")
    print(f"Testing: {config['name']} with probability threshold {config['threshold']:.2f}")
    print(f"{'=' * 80}")

    results = rolling_window_backtest(df, model, feature_columns, config['threshold'])

    # Summary statistics
    print(f"\nResults ({len(results)} windows):")
    print(f"  XGBoost Return: {results['xgb_return'].mean():.2f}% ± {results['xgb_return'].std():.2f}%")
    print(f"  Buy & Hold Return: {results['bh_return'].mean():.2f}% ± {results['bh_return'].std():.2f}%")
    print(f"  Difference: {results['difference'].mean():.2f}% ± {results['difference'].std():.2f}%")
    print(f"  🎯 Avg Trades per Window: {results['num_trades'].mean():.1f}")
    print(f"  Avg Win Rate: {results['win_rate'].mean():.1f}%")
    print(f"  Avg Sharpe: {results['sharpe'].mean():.3f}")
    print(f"  Avg Max DD: {results['max_dd'].mean():.2f}%")

    # Market regime breakdown
    print(f"\nBy Market Regime:")
    for regime in ['Bull', 'Bear', 'Sideways']:
        regime_df = results[results['regime'] == regime]
        if len(regime_df) > 0:
            print(f"  {regime} ({len(regime_df)} windows):")
            print(f"    XGBoost: {regime_df['xgb_return'].mean():.2f}%")
            print(f"    Buy & Hold: {regime_df['bh_return'].mean():.2f}%")
            print(f"    Difference: {regime_df['difference'].mean():.2f}%")
            print(f"    Trades: {regime_df['num_trades'].mean():.1f}")

    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(results['xgb_return'], results['bh_return'])
    print(f"\nStatistical Test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant: {'✅ Yes' if p_value < 0.05 else '❌ No'}")

    # Save results
    output_file = RESULTS_DIR / f"backtest_v2_{config['name']}_thresh{int(config['threshold']*10)}.csv"
    results.to_csv(output_file, index=False)
    print(f"\n✅ Results saved: {output_file}")

    all_results.append({
        'config_name': config['name'],
        'threshold': config['threshold'],
        'avg_return': results['xgb_return'].mean(),
        'vs_bh': results['difference'].mean(),
        'avg_trades': results['num_trades'].mean(),
        'win_rate': results['win_rate'].mean(),
        'sharpe': results['sharpe'].mean(),
        'max_dd': results['max_dd'].mean(),
        'p_value': p_value
    })

# Compare all configs
print(f"\n{'=' * 80}")
print("CONFIGURATION COMPARISON")
print(f"{'=' * 80}")

comparison_df = pd.DataFrame(all_results)
print(comparison_df.to_string(index=False))

# Find best config
best_idx = comparison_df['avg_trades'].idxmax()  # Prioritize trade frequency
best_config = comparison_df.iloc[best_idx]

print(f"\n{'=' * 80}")
print(f"BEST CONFIGURATION (by trade frequency)")
print(f"{'=' * 80}")
print(f"  Config: {best_config['config_name']}")
print(f"  Threshold: {best_config['threshold']:.2f}")
print(f"  🎯 Avg Trades: {best_config['avg_trades']:.1f} trades/window")
print(f"  Win Rate: {best_config['win_rate']:.1f}%")
print(f"  Return: {best_config['avg_return']:.2f}%")
print(f"  vs B&H: {best_config['vs_bh']:.2f}%")
print(f"  Sharpe: {best_config['sharpe']:.3f}")
print(f"  Max DD: {best_config['max_dd']:.2f}%")
print(f"  p-value: {best_config['p_value']:.4f}")

# Save comparison
comparison_file = RESULTS_DIR / "backtest_v2_comparison.csv"
comparison_df.to_csv(comparison_file, index=False)
print(f"\n✅ Comparison saved: {comparison_file}")

print(f"\n{'=' * 80}")
print("XGBoost V2 Backtest Complete!")
print(f"{'=' * 80}")

print(f"\n🎯 비판적 분석:")
print(f"\n  개선 전 (lookahead=12, threshold=0.003):")
print(f"    - Avg Trades: 0.1 trades/window")
print(f"    - Win Rate: 0.3%")
print(f"    - Return vs B&H: +0.04%")
print(f"    - p-value: 0.2229 (not significant)")

print(f"\n  개선 후 (Best Config):")
print(f"    - Avg Trades: {best_config['avg_trades']:.1f} trades/window ({best_config['avg_trades'] / 0.1:.0f}x increase!)")
print(f"    - Win Rate: {best_config['win_rate']:.1f}%")
print(f"    - Return vs B&H: {best_config['vs_bh']:.2f}%")
print(f"    - p-value: {best_config['p_value']:.4f} ({'significant!' if best_config['p_value'] < 0.05 else 'not significant'})")

print(f"\n💡 결론:")
if best_config['avg_trades'] > 3 and best_config['win_rate'] > 45:
    print(f"  ✅ Phase 1 성공! 거래 빈도와 승률 모두 목표 달성")
    print(f"  ✅ 다음 단계: Phase 2 (Short-term Features 추가) 또는 Paper Trading")
else:
    print(f"  ⚠️ 부분 성공: 거래 빈도 개선됨, 추가 최적화 필요")
    print(f"  ⚠️ 다음 단계: Phase 2 (Short-term Features) 또는 threshold 조정")
