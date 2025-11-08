"""
Random Baseline Validation Experiment

Purpose: Empirically test if ML models add value beyond random signals

Methodology:
1. Generate random LONG/SHORT signals with same rate as trained models
2. Apply identical trading strategy (TP 3%, SL 1%, Max Hold 4h)
3. Compare performance: Random vs Model
4. Decision criteria:
   - Model > Random by 15%+ win rate → Model adds significant value
   - Model > Random by 10-15% → Model adds moderate value
   - Model > Random by <10% → Focus on strategy optimization instead

This addresses critical question: "Does the model actually help?"
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters (identical to dual-model backtest)
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
STOP_LOSS = 0.01  # 1%
TAKE_PROFIT = 0.03  # 3%
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002  # 0.02% maker fee

# Random seed for reproducibility
RANDOM_SEED = 42


def classify_market_regime(df_window):
    """Classify market regime based on price movement"""
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100
    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"


def generate_random_signals(n_samples, long_signal_rate, short_signal_rate, seed=RANDOM_SEED):
    """
    Generate random LONG/SHORT signals matching model signal rates

    Args:
        n_samples: Number of samples
        long_signal_rate: Percentage of LONG signals (e.g., 0.015 = 1.5%)
        short_signal_rate: Percentage of SHORT signals (e.g., 0.005 = 0.5%)
        seed: Random seed for reproducibility

    Returns:
        long_signals: Boolean array of LONG signals
        short_signals: Boolean array of SHORT signals
    """
    np.random.seed(seed)

    # Generate random signals with same rates as models
    long_signals = np.random.rand(n_samples) < long_signal_rate
    short_signals = np.random.rand(n_samples) < short_signal_rate

    # Ensure no conflicts (if both signal, randomly choose one)
    conflicts = long_signals & short_signals
    if conflicts.any():
        conflict_indices = np.where(conflicts)[0]
        for idx in conflict_indices:
            if np.random.rand() > 0.5:
                short_signals[idx] = False
            else:
                long_signals[idx] = False

    return long_signals, short_signals


def backtest_random_strategy(df, long_signals, short_signals):
    """
    Backtest with random signals (same strategy as dual-model)

    Args:
        df: DataFrame with price data
        long_signals: Boolean array for LONG entries
        short_signals: Boolean array for SHORT entries
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            side = position['side']
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # Calculate P&L based on position side
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price

            # Check exit conditions
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

                # Transaction costs
                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                pnl_usd -= (entry_cost + exit_cost)

                capital += pnl_usd

                trades.append({
                    'side': side,
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'regime': position['regime']
                })

                position = None

        # Look for entry (LONG or SHORT)
        if position is None and i < len(df) - 1:
            side = None

            if long_signals[i] and not short_signals[i]:
                side = 'LONG'
            elif short_signals[i] and not long_signals[i]:
                side = 'SHORT'
            elif long_signals[i] and short_signals[i]:
                # Both signal (shouldn't happen due to conflict resolution, but handle anyway)
                side = 'LONG' if np.random.rand() > 0.5 else 'SHORT'

            if side is not None:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price
                current_regime = classify_market_regime(df.iloc[max(0, i-20):i+1])

                position = {
                    'side': side,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'regime': current_regime
                }

    # Calculate metrics
    if len(trades) == 0:
        return trades, {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'num_long': 0,
            'num_short': 0,
            'win_rate': 0.0,
            'win_rate_long': 0.0,
            'win_rate_short': 0.0,
            'avg_pnl_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    # Overall metrics
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100
    avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades]) * 100

    # LONG vs SHORT breakdown
    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']

    win_rate_long = (len([t for t in long_trades if t['pnl_usd'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0.0
    win_rate_short = (len([t for t in short_trades if t['pnl_usd'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0.0

    # Drawdown
    cumulative_returns = []
    running_capital = INITIAL_CAPITAL
    for trade in trades:
        running_capital += trade['pnl_usd']
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
        returns = [t['pnl_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0

    metrics = {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'num_long': len(long_trades),
        'num_short': len(short_trades),
        'win_rate': win_rate,
        'win_rate_long': win_rate_long,
        'win_rate_short': win_rate_short,
        'avg_pnl_pct': avg_pnl_pct,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }

    return trades, metrics


def rolling_window_backtest_random(df, long_signal_rate, short_signal_rate):
    """Rolling window backtest with random signals"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        # Generate random signals for this window
        long_signals, short_signals = generate_random_signals(
            len(window_df),
            long_signal_rate,
            short_signal_rate,
            seed=RANDOM_SEED + start_idx  # Unique seed per window
        )

        trades, metrics = backtest_random_strategy(
            window_df, long_signals, short_signals
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
            'random_return': metrics['total_return_pct'],
            'bh_return': bh_return,
            'difference': metrics['total_return_pct'] - bh_return,
            'num_trades': metrics['num_trades'],
            'num_long': metrics['num_long'],
            'num_short': metrics['num_short'],
            'win_rate': metrics['win_rate'],
            'win_rate_long': metrics['win_rate_long'],
            'win_rate_short': metrics['win_rate_short'],
            'sharpe': metrics['sharpe_ratio'],
            'max_dd': metrics['max_drawdown']
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)


def main():
    print("=" * 80)
    print("RANDOM BASELINE VALIDATION EXPERIMENT")
    print("=" * 80)
    print("\n🎯 Purpose: Empirically test if ML models add value beyond random signals")
    print("\n⚠️ Critical Question: Does the model actually help?")

    # Load models to get signal rates
    long_model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
    long_scaler_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
    short_model_file = MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3.pkl"
    short_scaler_file = MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3_scaler.pkl"

    with open(long_model_file, 'rb') as f:
        long_model = pickle.load(f)
    with open(long_scaler_file, 'rb') as f:
        long_scaler = pickle.load(f)
    with open(short_model_file, 'rb') as f:
        short_model = pickle.load(f)
    with open(short_scaler_file, 'rb') as f:
        short_scaler = pickle.load(f)
    print(f"✅ Models loaded")

    # Load feature columns
    feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
    with open(feature_file, 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    print(f"✅ Features loaded: {len(feature_columns)} features")

    # Load data
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Data loaded: {len(df)} candles")

    # Calculate features
    print("\nCalculating features...")
    df = calculate_features(df)
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)
    df = df.ffill().dropna()
    print(f"✅ Features calculated: {len(df)} rows")

    # Get model signal rates (threshold 0.7)
    print("\n" + "=" * 80)
    print("Analyzing Model Signal Rates (threshold 0.7)")
    print("=" * 80)

    features_scaled_long = long_scaler.transform(df[feature_columns].values)
    long_probs = long_model.predict_proba(features_scaled_long)[:, 1]
    long_signal_rate = np.sum(long_probs >= 0.7) / len(long_probs)

    features_scaled_short = short_scaler.transform(df[feature_columns].values)
    short_probs = short_model.predict_proba(features_scaled_short)[:, 1]
    short_signal_rate = np.sum(short_probs >= 0.7) / len(short_probs)

    print(f"\nLONG Model Signal Rate: {long_signal_rate * 100:.3f}% ({np.sum(long_probs >= 0.7)} signals)")
    print(f"SHORT Model Signal Rate: {short_signal_rate * 100:.3f}% ({np.sum(short_probs >= 0.7)} signals)")

    # Generate random baseline
    print("\n" + "=" * 80)
    print("Generating Random Baseline (same signal rates)")
    print("=" * 80)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Strategy: TP {TAKE_PROFIT*100}%, SL {STOP_LOSS*100}%, Max Hold {MAX_HOLDING_HOURS}h")

    results_random = rolling_window_backtest_random(
        df=df,
        long_signal_rate=long_signal_rate,
        short_signal_rate=short_signal_rate
    )

    # Load model results for comparison
    model_results_file = RESULTS_DIR / "backtest_dual_model_mainnet.csv"
    if model_results_file.exists():
        results_model = pd.read_csv(model_results_file)
        has_model_comparison = True
    else:
        print(f"\n⚠️ Model results not found: {model_results_file}")
        print("   Run backtest_dual_model_mainnet.py first for comparison")
        has_model_comparison = False

    # Summary - Random Baseline
    print("\n" + "=" * 80)
    print(f"📊 RANDOM BASELINE RESULTS ({len(results_random)} windows)")
    print("=" * 80)
    print(f"\n  Returns:")
    print(f"    Random Strategy: {results_random['random_return'].mean():.2f}% ± {results_random['random_return'].std():.2f}%")
    print(f"    Buy & Hold: {results_random['bh_return'].mean():.2f}% ± {results_random['bh_return'].std():.2f}%")
    print(f"    Difference: {results_random['difference'].mean():.2f}% ± {results_random['difference'].std():.2f}%")

    print(f"\n  Trade Breakdown:")
    print(f"    Total Trades: {results_random['num_trades'].mean():.1f}")
    print(f"    LONG Trades: {results_random['num_long'].mean():.1f} ({results_random['num_long'].sum() / results_random['num_trades'].sum() * 100:.1f}%)")
    print(f"    SHORT Trades: {results_random['num_short'].mean():.1f} ({results_random['num_short'].sum() / results_random['num_trades'].sum() * 100:.1f}%)")

    print(f"\n  Win Rates:")
    print(f"    Overall: {results_random['win_rate'].mean():.1f}%")
    print(f"    LONG: {results_random['win_rate_long'].mean():.1f}%")
    print(f"    SHORT: {results_random['win_rate_short'].mean():.1f}%")

    print(f"\n  Risk Metrics:")
    print(f"    Sharpe: {results_random['sharpe'].mean():.3f}")
    print(f"    Max DD: {results_random['max_dd'].mean():.2f}%")

    # Comparison with Model
    if has_model_comparison:
        print("\n" + "=" * 80)
        print("🔬 MODEL vs RANDOM COMPARISON")
        print("=" * 80)

        model_return = results_model['xgb_return'].mean()
        random_return = results_random['random_return'].mean()
        model_wr = results_model['win_rate'].mean()
        random_wr = results_random['win_rate'].mean()
        model_sharpe = results_model['sharpe'].mean()
        random_sharpe = results_random['sharpe'].mean()

        print(f"\n  Returns:")
        print(f"    Model:  {model_return:+.2f}%")
        print(f"    Random: {random_return:+.2f}%")
        print(f"    Δ:      {model_return - random_return:+.2f}% ({(model_return - random_return) / abs(random_return) * 100 if random_return != 0 else 0:+.1f}%)")

        print(f"\n  Win Rate:")
        print(f"    Model:  {model_wr:.1f}%")
        print(f"    Random: {random_wr:.1f}%")
        print(f"    Δ:      {model_wr - random_wr:+.1f}%p")

        print(f"\n  Sharpe Ratio:")
        print(f"    Model:  {model_sharpe:.3f}")
        print(f"    Random: {random_sharpe:.3f}")
        print(f"    Δ:      {model_sharpe - random_sharpe:+.3f}")

        # Decision criteria
        print("\n" + "=" * 80)
        print("🎯 DECISION ANALYSIS")
        print("=" * 80)

        wr_improvement = model_wr - random_wr

        print(f"\n  Win Rate Improvement: {wr_improvement:+.1f}%p")

        if wr_improvement >= 15.0:
            print("\n  ✅ MODEL ADDS SIGNIFICANT VALUE (≥15%p improvement)")
            print("     Recommendation: Proceed with model improvements")
        elif wr_improvement >= 10.0:
            print("\n  ✅ MODEL ADDS MODERATE VALUE (10-15%p improvement)")
            print("     Recommendation: Model improvements worthwhile")
        elif wr_improvement >= 5.0:
            print("\n  ⚠️ MODEL ADDS MARGINAL VALUE (5-10%p improvement)")
            print("     Recommendation: Consider strategy optimization first")
        else:
            print("\n  ❌ MODEL ADDS MINIMAL VALUE (<5%p improvement)")
            print("     Recommendation: Focus on strategy parameters, not model")
            print("     Action: Run threshold sensitivity & parameter optimization")

        # Statistical test
        from scipy import stats
        if len(results_model) == len(results_random):
            t_stat, p_value = stats.ttest_rel(results_model['xgb_return'], results_random['random_return'])
            print(f"\n  Statistical Test:")
            print(f"    t-statistic: {t_stat:.4f}")
            print(f"    p-value: {p_value:.4f}")
            print(f"    Significant: {'✅ Yes (p<0.05)' if p_value < 0.05 else '❌ No (p≥0.05)'}")

    # Save results
    output_file = RESULTS_DIR / "backtest_random_baseline.csv"
    results_random.to_csv(output_file, index=False)
    print(f"\n✅ Results saved: {output_file}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print("\n📝 Next Steps:")
    print("   1. Review win rate improvement (Model vs Random)")
    print("   2. If <10%p improvement → Run threshold sensitivity analysis")
    print("   3. If <10%p improvement → Run strategy parameter optimization")
    print("   4. Only proceed with model retraining if improvement ≥10%p")


if __name__ == "__main__":
    main()
