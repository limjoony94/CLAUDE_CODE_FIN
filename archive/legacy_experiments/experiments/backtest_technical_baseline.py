"""
Technical Analysis Baseline Strategy

비판적 질문: "ML이 정말 필요한가?"

Simple strategy:
- EMA 9/21 crossover (trend)
- RSI 50-70 range (momentum)
- Volume confirmation (1.2x average)

If this beats XGBoost → ML has serious problems!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters (same as XGBoost for fair comparison)
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
LEVERAGE = 2.0
STOP_LOSS = 0.005  # 0.5%
TAKE_PROFIT = 0.03  # 3%
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002  # 0.02%

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

def technical_analysis_strategy(df, position_size_pct=0.95):
    """
    Pure Technical Analysis Strategy

    Entry Rules:
    1. EMA 9 > EMA 21 (uptrend)
    2. RSI between 50 and 70 (momentum, not overbought)
    3. Current volume > 1.2 × 20-period average volume (confirmation)

    Exit Rules:
    1. Stop loss: -0.5%
    2. Take profit: +3%
    3. EMA 9 < EMA 21 (trend reversal)
    4. Max holding: 4 hours
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
            price_change_pct = (current_price - entry_price) / entry_price
            leveraged_pnl_pct = price_change_pct * LEVERAGE
            leveraged_pnl_usd = leveraged_pnl_pct * position['base_value']

            exit_reason = None

            # Exit conditions
            if leveraged_pnl_pct <= -STOP_LOSS:
                exit_reason = "Stop Loss"
            elif leveraged_pnl_pct >= TAKE_PROFIT:
                exit_reason = "Take Profit"
            elif hours_held >= MAX_HOLDING_HOURS:
                exit_reason = "Max Holding"
            elif i < len(df) - 1:  # Check trend reversal
                ema9 = df['ema_9'].iloc[i]
                ema21 = df['ema_21'].iloc[i]
                if ema9 < ema21:
                    exit_reason = "Trend Reversal"

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
                    'position_size_pct': position_size_pct,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'leveraged_pnl_usd': leveraged_pnl_usd,
                    'pnl_usd_net': net_pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'signal': position['signal'],
                    'regime': position['regime']
                })

                capital += net_pnl_usd
                position = None

        # Look for entry
        if position is None and i < len(df) - 1:
            # Check if we have required indicators
            if i < 50:  # Need enough history for indicators
                continue

            # Get indicator values
            ema9 = df['ema_9'].iloc[i]
            ema21 = df['ema_21'].iloc[i]
            rsi = df['rsi'].iloc[i]
            current_volume = df['volume'].iloc[i]
            avg_volume = df['volume'].iloc[max(0, i-20):i].mean()

            # Check for NaN
            if pd.isna(ema9) or pd.isna(ema21) or pd.isna(rsi) or pd.isna(avg_volume):
                continue

            # Entry conditions
            uptrend = ema9 > ema21
            momentum = 50 < rsi < 70
            volume_confirmation = current_volume > 1.2 * avg_volume

            if uptrend and momentum and volume_confirmation:
                # Calculate regime
                current_regime = classify_market_regime(df.iloc[max(0, i-20):i+1])

                # Calculate position
                base_value = capital * position_size_pct
                leveraged_value = base_value * LEVERAGE

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'base_value': base_value,
                    'leveraged_value': leveraged_value,
                    'signal': {
                        'ema9': ema9,
                        'ema21': ema21,
                        'rsi': rsi,
                        'volume_ratio': current_volume / avg_volume
                    },
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

def rolling_window_backtest(df, position_size_pct=0.95):
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        trades, metrics = technical_analysis_strategy(window_df, position_size_pct)

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
print("Technical Analysis Baseline Strategy")
print("=" * 80)
print()
print("🎯 Critical Test: Does ML add value?")
print()
print("Strategy:")
print("  Entry:")
print("    - EMA 9 > EMA 21 (uptrend)")
print("    - RSI 50-70 (momentum, not overbought)")
print("    - Volume > 1.2 × average (confirmation)")
print()
print("  Exit:")
print("    - Stop loss: -0.5%")
print("    - Take profit: +3%")
print("    - Trend reversal: EMA 9 < EMA 21")
print("    - Max holding: 4 hours")
print()
print("=" * 80)

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Data loaded: {len(df)} candles")

# Calculate technical indicators
print("Calculating technical indicators...")

# EMA 9 and 21
df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()

# RSI (using existing RSI from calculate_features)
df = calculate_features(df)

# Forward fill and drop NaN
df = df.ffill()
df = df.dropna()
print(f"✅ Features calculated: {len(df)} rows after dropna")

# Backtest
print(f"\n{'=' * 80}")
print(f"Running Backtest (Rolling 5-day windows)")
print(f"{'=' * 80}")

results = rolling_window_backtest(df, position_size_pct=0.95)

print(f"\nResults ({len(results)} windows):")
print(f"  Avg Return: {results['return'].mean():.2f}%")
print(f"  vs B&H: {results['difference'].mean():.2f}%")
print(f"  Win Rate: {results['win_rate'].mean():.1f}%")
print(f"  Sharpe: {results['sharpe'].mean():.3f}")
print(f"  Max DD: {results['max_dd'].mean():.2f}%")
print(f"  Avg Trades: {results['num_trades'].mean():.1f}")

# Breakdown by regime
print(f"\n{'=' * 80}")
print("Performance by Market Regime")
print(f"{'=' * 80}")

for regime in ['Bull', 'Bear', 'Sideways']:
    regime_data = results[results['regime'] == regime]
    if len(regime_data) > 0:
        print(f"\n{regime} Market ({len(regime_data)} windows):")
        print(f"  Avg Return: {regime_data['return'].mean():.2f}%")
        print(f"  vs B&H: {regime_data['difference'].mean():.2f}%")
        print(f"  Win Rate: {regime_data['win_rate'].mean():.1f}%")
        print(f"  Sharpe: {regime_data['sharpe'].mean():.3f}")

# Save results
output_file = RESULTS_DIR / "backtest_technical_baseline.csv"
results.to_csv(output_file, index=False)
print(f"\n✅ Results saved: {output_file}")

print(f"\n{'=' * 80}")
print("Backtest Complete!")
print(f"{'=' * 80}")

print(f"\n🎯 비판적 분석:")
print(f"\nTechnical Analysis (No ML):")
print(f"  - Avg Return: {results['return'].mean():.2f}%")
print(f"  - Sharpe: {results['sharpe'].mean():.3f}")
print(f"  - Max DD: {results['max_dd'].mean():.2f}%")

print(f"\n⏳ Next: Compare vs XGBoost to see if ML adds value!")
print(f"   If Technical beats XGBoost → ML has serious problems!")
print(f"   If XGBoost beats Technical → ML is worthwhile!")
