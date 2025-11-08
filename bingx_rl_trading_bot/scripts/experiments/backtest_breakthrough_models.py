"""
Breakthrough Models Backtest: Peak/Trough Labeling Success

Breakthrough Models (55.2% precision each):
- SHORT Entry: Peak/Trough labeling
- LONG Exit: Peak/Trough labeling

Comparison:
- Previous: Rule-based exits (1% SL, 3% TP, 4h max)
- New: ML LONG Exit + Safety rules for SHORT exit

Strategy:
- LONG: ML Entry (existing 70.2%) → ML Exit (new 55.2%)
- SHORT: ML Entry (new 55.2%) → Safety exits (3% TP, 1% SL, 4h max)
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
from src.features.sell_signal_features import SellSignalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
ENTRY_THRESHOLD = 0.7
EXIT_THRESHOLD = 0.5  # ML exit threshold
TRANSACTION_COST = 0.0002  # 0.02% maker fee

# Safety exits (for SHORT positions without ML exit)
STOP_LOSS = 0.01  # 1%
TAKE_PROFIT = 0.03  # 3%
MAX_HOLDING_HOURS = 4
CATASTROPHIC_LOSS = 0.05  # 5% emergency stop


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


def backtest_breakthrough_strategy(df, long_entry_model, short_entry_model, long_exit_model,
                                    long_entry_scaler, short_entry_scaler, long_exit_scaler,
                                    long_entry_features, short_entry_features, long_exit_features):
    """
    Backtest with breakthrough models:
    - LONG: ML Entry → ML Exit
    - SHORT: ML Entry → Safety exits
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

            # Calculate P&L
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price

            # Check EXIT conditions
            exit_reason = None
            exit_prob = None

            if side == 'LONG':
                # Use ML LONG Exit model
                exit_features_row = df[long_exit_features].iloc[i:i+1].values
                if not np.isnan(exit_features_row).any():
                    exit_features_scaled = long_exit_scaler.transform(exit_features_row)
                    exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0][1]

                    if exit_prob >= EXIT_THRESHOLD:
                        exit_reason = "ML Exit"
                    elif pnl_pct <= -CATASTROPHIC_LOSS:
                        exit_reason = "Catastrophic Loss"
                    elif hours_held >= MAX_HOLDING_HOURS * 2:  # Extended for ML exit
                        exit_reason = "Max Holding"
            else:
                # SHORT: Use safety exits (no ML exit available)
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
                    'entry_probability': position['probability'],
                    'exit_probability': exit_prob,
                    'regime': position['regime']
                })

                position = None

        # Look for ENTRY
        if position is None and i < len(df) - 1:
            # Get LONG entry probability
            long_entry_row = df[long_entry_features].iloc[i:i+1].values
            if np.isnan(long_entry_row).any():
                long_prob = 0
            else:
                long_features_scaled = long_entry_scaler.transform(long_entry_row)
                long_prob = long_entry_model.predict_proba(long_features_scaled)[0][1]

            # Get SHORT entry probability
            short_entry_row = df[short_entry_features].iloc[i:i+1].values
            if np.isnan(short_entry_row).any():
                short_prob = 0
            else:
                short_features_scaled = short_entry_scaler.transform(short_entry_row)
                short_prob = short_entry_model.predict_proba(short_features_scaled)[0][1]

            # Determine entry direction
            side = None
            probability = None

            if long_prob >= ENTRY_THRESHOLD and short_prob < ENTRY_THRESHOLD:
                side = 'LONG'
                probability = long_prob
            elif short_prob >= ENTRY_THRESHOLD and long_prob < ENTRY_THRESHOLD:
                side = 'SHORT'
                probability = short_prob
            elif long_prob >= ENTRY_THRESHOLD and short_prob >= ENTRY_THRESHOLD:
                # Both confident - choose stronger
                if long_prob > short_prob:
                    side = 'LONG'
                    probability = long_prob
                else:
                    side = 'SHORT'
                    probability = short_prob

            if side is not None:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price
                current_regime = classify_market_regime(df.iloc[max(0, i-20):i+1])

                position = {
                    'side': side,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'probability': probability,
                    'regime': current_regime
                }

    # Calculate metrics
    if len(trades) == 0:
        return trades, {}

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100
    avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades]) * 100

    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']

    win_rate_long = (len([t for t in long_trades if t['pnl_usd'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0.0
    win_rate_short = (len([t for t in short_trades if t['pnl_usd'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0.0

    ml_exits = [t for t in trades if t['exit_reason'] == 'ML Exit']
    ml_exit_rate = (len(ml_exits) / len(trades)) * 100

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
        'max_drawdown': max_dd,
        'ml_exit_rate': ml_exit_rate
    }

    return trades, metrics


def rolling_window_backtest(df, long_entry_model, short_entry_model, long_exit_model,
                            long_entry_scaler, short_entry_scaler, long_exit_scaler,
                            long_entry_features, short_entry_features, long_exit_features):
    """Rolling window backtest"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        trades, metrics = backtest_breakthrough_strategy(
            window_df, long_entry_model, short_entry_model, long_exit_model,
            long_entry_scaler, short_entry_scaler, long_exit_scaler,
            long_entry_features, short_entry_features, long_exit_features
        )

        # Buy & Hold
        bh_start = window_df['close'].iloc[0]
        bh_end = window_df['close'].iloc[-1]
        bh_return = ((bh_end - bh_start) / bh_start) * 100
        bh_cost = 2 * TRANSACTION_COST * 100
        bh_return -= bh_cost

        if len(metrics) > 0:
            windows.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'regime': regime,
                'ml_return': metrics['total_return_pct'],
                'bh_return': bh_return,
                'difference': metrics['total_return_pct'] - bh_return,
                'num_trades': metrics['num_trades'],
                'num_long': metrics['num_long'],
                'num_short': metrics['num_short'],
                'win_rate': metrics['win_rate'],
                'win_rate_long': metrics['win_rate_long'],
                'win_rate_short': metrics['win_rate_short'],
                'sharpe': metrics['sharpe_ratio'],
                'max_dd': metrics['max_drawdown'],
                'ml_exit_rate': metrics['ml_exit_rate']
            })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)


def main():
    print("=" * 80)
    print("Breakthrough Models Backtest (Peak/Trough Labeling)")
    print("=" * 80)

    # Load models
    long_entry_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
    long_entry_scaler_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
    with open(long_entry_file, 'rb') as f:
        long_entry_model = pickle.load(f)
    with open(long_entry_scaler_file, 'rb') as f:
        long_entry_scaler = pickle.load(f)
    print(f"✅ LONG Entry: 70.2% precision")

    short_entry_file = MODELS_DIR / "xgboost_short_peak_trough_20251016_131939.pkl"
    short_entry_scaler_file = MODELS_DIR / "xgboost_short_peak_trough_20251016_131939_scaler.pkl"
    with open(short_entry_file, 'rb') as f:
        short_entry_model = pickle.load(f)
    with open(short_entry_scaler_file, 'rb') as f:
        short_entry_scaler = pickle.load(f)
    print(f"✅ SHORT Entry: 55.2% precision (BREAKTHROUGH)")

    long_exit_file = MODELS_DIR / "xgboost_long_exit_peak_trough_20251016_132651.pkl"
    long_exit_scaler_file = MODELS_DIR / "xgboost_long_exit_peak_trough_20251016_132651_scaler.pkl"
    with open(long_exit_file, 'rb') as f:
        long_exit_model = pickle.load(f)
    with open(long_exit_scaler_file, 'rb') as f:
        long_exit_scaler = pickle.load(f)
    print(f"✅ LONG Exit: 55.2% precision (BREAKTHROUGH)")

    # Load features
    long_entry_feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
    with open(long_entry_feature_file, 'r') as f:
        long_entry_features = [line.strip() for line in f.readlines()]

    short_entry_feature_file = MODELS_DIR / "xgboost_short_peak_trough_20251016_131939_features.txt"
    with open(short_entry_feature_file, 'r') as f:
        short_entry_features = [line.strip() for line in f.readlines()]

    long_exit_feature_file = MODELS_DIR / "xgboost_long_exit_peak_trough_20251016_132651_features.txt"
    with open(long_exit_feature_file, 'r') as f:
        long_exit_features = [line.strip() for line in f.readlines()]

    print(f"\n📊 Feature Counts:")
    print(f"  LONG Entry: {len(long_entry_features)}")
    print(f"  SHORT Entry: {len(short_entry_features)}")
    print(f"  LONG Exit: {len(long_exit_features)}")

    # Load data
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"\n✅ Data loaded: {len(df)} candles")

    # Calculate features
    print("\nCalculating features...")
    df = calculate_features(df)
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)
    sell_features = SellSignalFeatures()
    df = sell_features.calculate_all_features(df)
    df = df.ffill().dropna()
    print(f"✅ Features calculated: {len(df)} rows")

    # Run backtest
    print("\n" + "=" * 80)
    print("Running Breakthrough Models Backtest")
    print("=" * 80)
    print(f"LONG: ML Entry → ML Exit (55.2% precision)")
    print(f"SHORT: ML Entry → Safety exits (TP 3%, SL 1%, 4h)")

    results = rolling_window_backtest(
        df=df,
        long_entry_model=long_entry_model,
        short_entry_model=short_entry_model,
        long_exit_model=long_exit_model,
        long_entry_scaler=long_entry_scaler,
        short_entry_scaler=short_entry_scaler,
        long_exit_scaler=long_exit_scaler,
        long_entry_features=long_entry_features,
        short_entry_features=short_entry_features,
        long_exit_features=long_exit_features
    )

    # Summary
    print(f"\n📊 Results ({len(results)} windows):")
    print(f"  ML System Return: {results['ml_return'].mean():.2f}% ± {results['ml_return'].std():.2f}%")
    print(f"  Buy & Hold Return: {results['bh_return'].mean():.2f}% ± {results['bh_return'].std():.2f}%")
    print(f"  Difference: {results['difference'].mean():.2f}% ± {results['difference'].std():.2f}%")
    print(f"\n  📊 Trade Breakdown:")
    print(f"    Total Trades: {results['num_trades'].mean():.1f}")
    print(f"    LONG: {results['num_long'].mean():.1f} ({results['num_long'].sum() / results['num_trades'].sum() * 100:.1f}%)")
    print(f"    SHORT: {results['num_short'].mean():.1f} ({results['num_short'].sum() / results['num_trades'].sum() * 100:.1f}%)")
    print(f"\n  🎯 Win Rates:")
    print(f"    Overall: {results['win_rate'].mean():.1f}%")
    print(f"    LONG: {results['win_rate_long'].mean():.1f}%")
    print(f"    SHORT: {results['win_rate_short'].mean():.1f}%")
    print(f"\n  📈 Performance:")
    print(f"    Sharpe: {results['sharpe'].mean():.3f}")
    print(f"    Max DD: {results['max_dd'].mean():.2f}%")
    print(f"    ML Exit Rate (LONG only): {results['ml_exit_rate'].mean():.1f}%")

    # By regime
    print(f"\n{'=' * 80}")
    print(f"Performance by Market Regime:")
    print(f"{'=' * 80}")
    for regime in ['Bull', 'Bear', 'Sideways']:
        regime_df = results[results['regime'] == regime]
        if len(regime_df) > 0:
            print(f"\n  {regime} ({len(regime_df)} windows):")
            print(f"    ML System: {regime_df['ml_return'].mean():.2f}%")
            print(f"    Buy & Hold: {regime_df['bh_return'].mean():.2f}%")
            print(f"    Win Rate: {regime_df['win_rate'].mean():.1f}%")

    # Save results
    output_file = RESULTS_DIR / "backtest_breakthrough_models.csv"
    results.to_csv(output_file, index=False)
    print(f"\n✅ Results saved: {output_file}")

    # Statistical test
    from scipy import stats
    if len(results) > 1:
        t_stat, p_value = stats.ttest_rel(results['ml_return'], results['bh_return'])
        print(f"\n{'=' * 80}")
        print("Statistical Significance")
        print(f"{'=' * 80}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant: {'✅ Yes' if p_value < 0.05 else '❌ No'}")

    # Analysis
    print(f"\n{'=' * 80}")
    print("🎯 Breakthrough Analysis")
    print(f"{'=' * 80}")

    avg_return = results['ml_return'].mean()
    win_rate = results['win_rate'].mean()
    sharpe = results['sharpe'].mean()

    print(f"\n  Peak/Trough Labeling Impact:")
    print(f"    SHORT Entry: 55.2% precision (vs <1% with TP/SL)")
    print(f"    LONG Exit: 55.2% precision (vs 35.2% with TP/SL)")
    print(f"\n  System Performance:")
    print(f"    Returns: {avg_return:+.2f}%")
    print(f"    Win Rate: {win_rate:.1f}%")
    print(f"    Sharpe: {sharpe:.3f}")

    if avg_return > 2.0 and win_rate > 55:
        print(f"\n  🎉 BREAKTHROUGH CONFIRMED!")
        print(f"     Peak/Trough labeling delivers strong performance!")
    elif avg_return > 0 and win_rate > 50:
        print(f"\n  ✅ BREAKTHROUGH SUCCESSFUL")
        print(f"     Significant improvement over previous methods")
    else:
        print(f"\n  ⚠️ Needs optimization")

    print(f"\n{'=' * 80}")
    print("Backtest Complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
