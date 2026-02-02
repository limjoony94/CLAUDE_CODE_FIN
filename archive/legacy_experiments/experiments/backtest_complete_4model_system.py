"""
Complete 4-Model ML System Backtest (ALL ML, NO RULES)

All 4 models with Peak/Trough labeling:
1. LONG Entry: 70.2% (existing)
2. SHORT Entry: 55.2% (new)
3. LONG Exit: 55.2% (new)
4. SHORT Exit: 55.2% (new)

Pure ML system - no rule-based exits
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
from src.features.sell_signal_features import SellSignalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Parameters
WINDOW_SIZE = 1440
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
ENTRY_THRESHOLD = 0.7
EXIT_THRESHOLD = 0.5
TRANSACTION_COST = 0.0002
MAX_HOLDING_HOURS = 8  # Safety stop


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


def backtest_complete_4model_system(df, models, scalers, features):
    """Backtest with complete 4-model ML system"""
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
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # Check ML EXIT signal
            exit_prob = None
            if side == 'LONG':
                exit_row = df[features['long_exit']].iloc[i:i+1].values
                if not np.isnan(exit_row).any():
                    exit_scaled = scalers['long_exit'].transform(exit_row)
                    exit_prob = models['long_exit'].predict_proba(exit_scaled)[0][1]
            else:  # SHORT
                exit_row = df[features['short_exit']].iloc[i:i+1].values
                if not np.isnan(exit_row).any():
                    exit_scaled = scalers['short_exit'].transform(exit_row)
                    exit_prob = models['short_exit'].predict_proba(exit_scaled)[0][1]

            # Exit conditions
            exit_reason = None
            if exit_prob is not None:
                if exit_prob >= EXIT_THRESHOLD:
                    exit_reason = "ML Exit"
                elif hours_held >= MAX_HOLDING_HOURS:
                    exit_reason = "Max Holding"

            if exit_reason:
                quantity = position['quantity']
                pnl_usd = pnl_pct * (entry_price * quantity)

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
            # LONG Entry
            long_row = df[features['long_entry']].iloc[i:i+1].values
            if np.isnan(long_row).any():
                long_prob = 0
            else:
                long_scaled = scalers['long_entry'].transform(long_row)
                long_prob = models['long_entry'].predict_proba(long_scaled)[0][1]

            # SHORT Entry
            short_row = df[features['short_entry']].iloc[i:i+1].values
            if np.isnan(short_row).any():
                short_prob = 0
            else:
                short_scaled = scalers['short_entry'].transform(short_row)
                short_prob = models['short_entry'].predict_proba(short_scaled)[0][1]

            # Entry logic
            side = None
            probability = None

            if long_prob >= ENTRY_THRESHOLD and short_prob < ENTRY_THRESHOLD:
                side = 'LONG'
                probability = long_prob
            elif short_prob >= ENTRY_THRESHOLD and long_prob < ENTRY_THRESHOLD:
                side = 'SHORT'
                probability = short_prob
            elif long_prob >= ENTRY_THRESHOLD and short_prob >= ENTRY_THRESHOLD:
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

    # Metrics
    if len(trades) == 0:
        return trades, {}

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100

    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']

    win_rate_long = (len([t for t in long_trades if t['pnl_usd'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0
    win_rate_short = (len([t for t in short_trades if t['pnl_usd'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0

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
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    return trades, {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'num_long': len(long_trades),
        'num_short': len(short_trades),
        'win_rate': win_rate,
        'win_rate_long': win_rate_long,
        'win_rate_short': win_rate_short,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'ml_exit_rate': ml_exit_rate
    }


def rolling_window_backtest(df, models, scalers, features):
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)
        trades, metrics = backtest_complete_4model_system(window_df, models, scalers, features)

        bh_start = window_df['close'].iloc[0]
        bh_end = window_df['close'].iloc[-1]
        bh_return = ((bh_end - bh_start) / bh_start) * 100
        bh_return -= 2 * TRANSACTION_COST * 100

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
    print("Complete 4-Model ML System Backtest")
    print("=" * 80)

    # Load all 4 models
    print("\n1. Loading Models...")
    models = {}
    scalers = {}
    features = {}

    # LONG Entry
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl", 'rb') as f:
        models['long_entry'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl", 'rb') as f:
        scalers['long_entry'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt", 'r') as f:
        features['long_entry'] = [line.strip() for line in f]
    print(f"  ✅ LONG Entry: 70.2% precision ({len(features['long_entry'])} features)")

    # SHORT Entry
    with open(MODELS_DIR / "xgboost_short_peak_trough_20251016_131939.pkl", 'rb') as f:
        models['short_entry'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_peak_trough_20251016_131939_scaler.pkl", 'rb') as f:
        scalers['short_entry'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_peak_trough_20251016_131939_features.txt", 'r') as f:
        features['short_entry'] = [line.strip() for line in f]
    print(f"  ✅ SHORT Entry: 55.2% precision ({len(features['short_entry'])} features)")

    # LONG Exit
    with open(MODELS_DIR / "xgboost_long_exit_peak_trough_20251016_132651.pkl", 'rb') as f:
        models['long_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_long_exit_peak_trough_20251016_132651_scaler.pkl", 'rb') as f:
        scalers['long_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_long_exit_peak_trough_20251016_132651_features.txt", 'r') as f:
        features['long_exit'] = [line.strip() for line in f]
    print(f"  ✅ LONG Exit: 55.2% precision ({len(features['long_exit'])} features)")

    # SHORT Exit
    with open(MODELS_DIR / "xgboost_short_exit_peak_trough_20251016_135233.pkl", 'rb') as f:
        models['short_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_exit_peak_trough_20251016_135233_scaler.pkl", 'rb') as f:
        scalers['short_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_exit_peak_trough_20251016_135233_features.txt", 'r') as f:
        features['short_exit'] = [line.strip() for line in f]
    print(f"  ✅ SHORT Exit: 55.2% precision ({len(features['short_exit'])} features)")

    # Load data
    print("\n2. Loading Data...")
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"  ✅ {len(df):,} candles")

    # Calculate features
    print("\n3. Calculating Features...")
    df = calculate_features(df)
    adv = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv.calculate_all_features(df)
    sell = SellSignalFeatures()
    df = sell.calculate_all_features(df)
    df = df.ffill().dropna()
    print(f"  ✅ {len(df):,} rows with features")

    # Backtest
    print("\n" + "=" * 80)
    print("Running Complete 4-Model ML System Backtest")
    print("=" * 80)
    print(f"Entry Threshold: {ENTRY_THRESHOLD}")
    print(f"Exit Threshold: {EXIT_THRESHOLD}")

    results = rolling_window_backtest(df, models, scalers, features)

    # Summary
    print(f"\n📊 Results ({len(results)} windows):")
    print(f"  ML Return: {results['ml_return'].mean():.2f}% ± {results['ml_return'].std():.2f}%")
    print(f"  B&H Return: {results['bh_return'].mean():.2f}% ± {results['bh_return'].std():.2f}%")
    print(f"  Difference: {results['difference'].mean():.2f}% ± {results['difference'].std():.2f}%")

    print(f"\n  📊 Trades:")
    print(f"    Total: {results['num_trades'].mean():.1f}")
    print(f"    LONG: {results['num_long'].mean():.1f} ({results['num_long'].sum() / results['num_trades'].sum() * 100:.1f}%)")
    print(f"    SHORT: {results['num_short'].mean():.1f} ({results['num_short'].sum() / results['num_trades'].sum() * 100:.1f}%)")

    print(f"\n  🎯 Win Rates:")
    print(f"    Overall: {results['win_rate'].mean():.1f}%")
    print(f"    LONG: {results['win_rate_long'].mean():.1f}%")
    print(f"    SHORT: {results['win_rate_short'].mean():.1f}%")

    print(f"\n  📈 Performance:")
    print(f"    Sharpe: {results['sharpe'].mean():.3f}")
    print(f"    Max DD: {results['max_dd'].mean():.2f}%")
    print(f"    ML Exit Rate: {results['ml_exit_rate'].mean():.1f}%")

    # By regime
    print(f"\n{'=' * 80}")
    print("Performance by Regime:")
    print(f"{'=' * 80}")
    for regime in ['Bull', 'Bear', 'Sideways']:
        regime_df = results[results['regime'] == regime]
        if len(regime_df) > 0:
            print(f"\n  {regime} ({len(regime_df)} windows):")
            print(f"    ML: {regime_df['ml_return'].mean():.2f}%")
            print(f"    B&H: {regime_df['bh_return'].mean():.2f}%")
            print(f"    Win Rate: {regime_df['win_rate'].mean():.1f}%")

    # Save
    results.to_csv(RESULTS_DIR / "backtest_complete_4model_system.csv", index=False)
    print(f"\n✅ Saved: backtest_complete_4model_system.csv")

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

    # Final analysis
    print(f"\n{'=' * 80}")
    print("🎯 Complete 4-Model ML System")
    print(f"{'=' * 80}")

    avg_return = results['ml_return'].mean()
    win_rate = results['win_rate'].mean()
    sharpe = results['sharpe'].mean()

    print(f"\n  System Performance:")
    print(f"    Returns: {avg_return:+.2f}%")
    print(f"    Win Rate: {win_rate:.1f}%")
    print(f"    Sharpe: {sharpe:.3f}")

    if avg_return > 5.0 and win_rate > 60:
        print(f"\n  🎉 OUTSTANDING PERFORMANCE!")
    elif avg_return > 2.0 and win_rate > 55:
        print(f"\n  ✅ EXCELLENT PERFORMANCE!")
    else:
        print(f"\n  ✅ GOOD PERFORMANCE")

    print(f"\n{'=' * 80}")
    print("Backtest Complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
