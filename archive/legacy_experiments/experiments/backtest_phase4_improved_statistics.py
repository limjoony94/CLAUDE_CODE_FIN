"""
Improved Backtest with Robust Statistical Methodology

Critical Improvements:
1. Smaller windows (2 days) for n≥30 samples
2. Bootstrap confidence intervals
3. Bonferroni correction for multiple comparisons
4. Effect size (Cohen's d) calculation
5. Statistical power analysis

Goal: Statistically robust validation of Base model
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from scipy import stats
from loguru import logger

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters
WINDOW_SIZE = 576  # 2 days (was 5 days = 1440) → n≥30 windows ✅
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
    """Rolling window backtest with 2-day windows"""
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


def bootstrap_confidence_interval(data, n_bootstrap=10000, confidence=0.95):
    """
    Calculate bootstrap confidence intervals

    Args:
        data: Array of values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 = 95%)

    Returns:
        (lower_bound, upper_bound)
    """
    bootstrapped_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrapped_means, alpha/2 * 100)
    upper = np.percentile(bootstrapped_means, (1 - alpha/2) * 100)

    return lower, upper


def cohen_d(group1, group2):
    """
    Calculate Cohen's d effect size

    d = (mean1 - mean2) / pooled_std

    Interpretation:
    - |d| < 0.2: Small effect
    - 0.2 ≤ |d| < 0.5: Medium effect
    - 0.5 ≤ |d| < 0.8: Large effect
    - |d| ≥ 0.8: Very large effect
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    d = (mean1 - mean2) / pooled_std
    return d


def calculate_statistical_power(effect_size, n, alpha=0.05):
    """
    Approximate statistical power for paired t-test

    Based on non-centrality parameter
    """
    from scipy.stats import nct

    df = n - 1
    t_critical = stats.t.ppf(1 - alpha/2, df)
    ncp = effect_size * np.sqrt(n)

    # Power = P(reject H0 | H1 is true)
    power = 1 - nct.cdf(t_critical, df, ncp) + nct.cdf(-t_critical, df, ncp)

    return power


logger.info("=" * 80)
logger.info("Improved Backtest with Robust Statistics")
logger.info("=" * 80)

# Load Base model
model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(model_file, 'rb') as f:
    model = pickle.load(f)
logger.success(f"✅ Model loaded: {model_file}")

# Load feature columns
feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(feature_file, 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]
logger.success(f"✅ Features loaded: {len(feature_columns)} features")

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

# Handle NaN
df = df.ffill()
df = df.dropna()
logger.info(f"After dropna: {len(df)} rows")

# Backtest with best threshold (0.7)
threshold = 0.7

logger.info(f"\n{'=' * 80}")
logger.info(f"Testing Base Model (2-day windows for n≥30)")
logger.info(f"Threshold: {threshold:.2f}")
logger.info(f"{'=' * 80}")

results = rolling_window_backtest(df, model, feature_columns, threshold)

logger.info(f"\n📊 Sample Size:")
logger.info(f"  Windows: {len(results)} (2-day windows)")
logger.info(f"  Status: {'✅ n≥30 (robust)' if len(results) >= 30 else '⚠️ n<30 (limited)'}")

# Summary statistics
logger.info(f"\n📈 Performance Metrics:")
logger.info(f"  Base Model Return: {results['return'].mean():.2f}% ± {results['return'].std():.2f}%")
logger.info(f"  Buy & Hold Return: {results['bh_return'].mean():.2f}% ± {results['bh_return'].std():.2f}%")
logger.info(f"  Difference: {results['difference'].mean():.2f}% ± {results['difference'].std():.2f}%")
logger.info(f"  Avg Trades per Window: {results['num_trades'].mean():.1f}")
logger.info(f"  Avg Win Rate: {results['win_rate'].mean():.1f}%")
logger.info(f"  Avg Sharpe: {results['sharpe'].mean():.3f}")
logger.info(f"  Avg Max DD: {results['max_dd'].mean():.2f}%")

# Bootstrap confidence intervals
logger.info(f"\n🔬 Statistical Analysis:")

ci_low, ci_high = bootstrap_confidence_interval(results['difference'].values, n_bootstrap=10000)
logger.info(f"  Bootstrap 95% CI: [{ci_low:.2f}%, {ci_high:.2f}%]")

# Effect size
effect_size = cohen_d(results['return'].values, results['bh_return'].values)
logger.info(f"  Effect Size (Cohen's d): {effect_size:.3f}")
if abs(effect_size) >= 0.8:
    logger.success(f"    → Very large effect ✅")
elif abs(effect_size) >= 0.5:
    logger.info(f"    → Large effect")
elif abs(effect_size) >= 0.2:
    logger.info(f"    → Medium effect")
else:
    logger.warning(f"    → Small effect")

# Statistical test (paired t-test)
t_stat, p_value = stats.ttest_rel(results['return'], results['bh_return'])

# Bonferroni correction (tested 3 thresholds × 3 models = 9 comparisons)
num_comparisons = 9
alpha_corrected = 0.05 / num_comparisons
logger.info(f"  t-statistic: {t_stat:.4f}")
logger.info(f"  p-value: {p_value:.4f}")
logger.info(f"  Bonferroni α: {alpha_corrected:.4f} (9 comparisons)")
if p_value < alpha_corrected:
    logger.success(f"    → Significant after correction ✅")
else:
    logger.warning(f"    → Not significant after correction ⚠️")

# Statistical power
power = calculate_statistical_power(effect_size, len(results), alpha=0.05)
logger.info(f"  Statistical Power: {power:.3f}")
if power >= 0.80:
    logger.success(f"    → Excellent power (≥0.80) ✅")
else:
    logger.warning(f"    → Low power (<0.80) ⚠️")

# By regime
logger.info(f"\n📊 By Market Regime:")
for regime in ['Bull', 'Bear', 'Sideways']:
    regime_df = results[results['regime'] == regime]
    if len(regime_df) > 0:
        logger.info(f"  {regime} (n={len(regime_df)} windows):")
        logger.info(f"    Base Return: {regime_df['return'].mean():.2f}% ± {regime_df['return'].std():.2f}%")
        logger.info(f"    Buy & Hold: {regime_df['bh_return'].mean():.2f}% ± {regime_df['bh_return'].std():.2f}%")
        logger.info(f"    Difference: {regime_df['difference'].mean():.2f}%")

# Save results
output_file = RESULTS_DIR / f"backtest_phase4_improved_stats_2day_windows.csv"
results.to_csv(output_file, index=False)
logger.success(f"\n✅ Results saved: {output_file}")

# Final assessment
logger.info(f"\n{'=' * 80}")
logger.info("STATISTICAL VALIDITY ASSESSMENT")
logger.info(f"{'=' * 80}")

validity_checks = {
    'Sample size (n≥30)': len(results) >= 30,
    'Effect size (|d|≥0.8)': abs(effect_size) >= 0.8,
    'Statistical power (≥0.80)': power >= 0.80,
    'Bonferroni-corrected p<α': p_value < alpha_corrected,
    'CI excludes zero': ci_low > 0 or ci_high < 0
}

logger.info("\nValidity Checks:")
for check, passed in validity_checks.items():
    status = "✅" if passed else "❌"
    logger.info(f"  {status} {check}: {passed}")

passing_checks = sum(validity_checks.values())
total_checks = len(validity_checks)

logger.info(f"\nOverall Validity: {passing_checks}/{total_checks} checks passed")

if passing_checks == total_checks:
    logger.success(f"\n🎉 HIGHLY CONFIDENT: All statistical checks passed!")
    logger.success(f"Base model significantly outperforms Buy & Hold")
    logger.success(f"Results: {results['difference'].mean():.2f}% improvement per 2 days")
elif passing_checks >= 3:
    logger.info(f"\n✅ CONFIDENT: Most statistical checks passed")
    logger.info(f"Base model likely outperforms Buy & Hold")
    logger.info(f"Results: {results['difference'].mean():.2f}% improvement per 2 days")
else:
    logger.warning(f"\n⚠️ LIMITED CONFIDENCE: Few statistical checks passed")
    logger.warning(f"Results are suggestive but not definitive")
    logger.warning(f"Need more data or different methodology")

logger.info(f"\n{'=' * 80}")
logger.success("✅ Improved Statistical Backtest Complete!")
logger.info(f"{'=' * 80}")
