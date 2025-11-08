"""
Full Period Backtest: Opportunity Gating Strategy
==================================================

Test Opportunity Gating with FULL historical data (Aug-Oct 2025)
- Multiple threshold combinations
- Transaction cost analysis
- Detailed performance metrics
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("FULL PERIOD BACKTEST: OPPORTUNITY GATING")
print("="*80)
print(f"\nTesting Winner Strategy with FULL historical data\n")

# Load models
print("Loading models...")
long_model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
with open(long_scaler_path, 'rb') as f:
    long_scaler = pickle.load(f)

long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl"
with open(short_scaler_path, 'rb') as f:
    short_scaler = pickle.load(f)

short_features_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print(f"  ✅ Models loaded\n")

# Load FULL data (no sampling!)
print("Loading FULL historical data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"  ✅ Full data loaded: {len(df_full):,} candles (~{len(df_full)//288:.0f} days)\n")

# Calculate features
print("Calculating ALL features (LONG + SHORT)...")
start_time = time.time()
df = calculate_all_features(df_full)
feature_time = time.time() - start_time
print(f"  ✅ All features calculated ({feature_time:.1f}s)\n")

# Pre-calculate signals
print("Pre-calculating signals...")
long_probs = []
short_probs = []

for i in range(len(df)):
    try:
        long_feat = df[long_feature_columns].iloc[i:i+1].values
        long_feat_scaled = long_scaler.transform(long_feat)
        long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
    except:
        long_prob = 0.0
    long_probs.append(long_prob)

    try:
        short_feat = df[short_feature_columns].iloc[i:i+1].values
        short_feat_scaled = short_scaler.transform(short_feat)
        short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
    except:
        short_prob = 0.0
    short_probs.append(short_prob)

df['long_prob'] = long_probs
df['short_prob'] = short_probs
print(f"  ✅ Signals pre-calculated\n")


def backtest_opportunity_gating(window_df, long_thresh, short_thresh, gate_thresh):
    """
    Opportunity Gating Strategy

    Only enter SHORT when expected value exceeds LONG by gate_threshold
    """
    trades = []
    position = None

    for i in range(len(window_df) - 1):
        if position is None:
            # LONG entry (standard)
            if window_df['long_prob'].iloc[i] >= long_thresh:
                position = {
                    'side': 'LONG',
                    'entry_idx': i,
                    'entry_price': window_df['close'].iloc[i]
                }

            # SHORT entry (gated)
            elif window_df['short_prob'].iloc[i] >= short_thresh:
                # Calculate expected values
                long_ev = window_df['long_prob'].iloc[i] * 0.0041  # LONG avg return
                short_ev = window_df['short_prob'].iloc[i] * 0.0047  # SHORT avg return

                # Gate: Only if SHORT clearly better
                if (short_ev - long_ev) > gate_thresh:
                    position = {
                        'side': 'SHORT',
                        'entry_idx': i,
                        'entry_price': window_df['close'].iloc[i]
                    }

        if position is not None:
            time_in_pos = i - position['entry_idx']

            if position['side'] == 'LONG':
                pnl_pct = (window_df['close'].iloc[i] - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - window_df['close'].iloc[i]) / position['entry_price']

            # Exit conditions
            if time_in_pos >= 240 or pnl_pct >= 0.03 or pnl_pct <= -0.015:
                trades.append({
                    'side': position['side'],
                    'pnl_pct': pnl_pct,
                    'hold_time': time_in_pos
                })
                position = None

    return trades


def run_window_backtest(df, long_thresh, short_thresh, gate_thresh):
    """Run backtest across all windows"""
    window_size = 1440
    step_size = 288
    num_windows = (len(df) - window_size) // step_size

    results = []

    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        if end_idx > len(df):
            break

        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Run strategy
        trades = backtest_opportunity_gating(window_df, long_thresh, short_thresh, gate_thresh)

        # Calculate metrics
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)

            avg_return = trades_df['pnl_pct'].mean() * 100
            total_return = trades_df['pnl_pct'].sum() * 100

            long_trades = trades_df[trades_df['side'] == 'LONG']
            short_trades = trades_df[trades_df['side'] == 'SHORT']

            results.append({
                'window': window_idx,
                'total_trades': len(trades),
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'avg_return': avg_return,
                'total_return': total_return,
                'win_rate': (trades_df['pnl_pct'] > 0).sum() / len(trades) * 100,
            })

    return pd.DataFrame(results)


# Test multiple configurations
print("="*80)
print("TESTING MULTIPLE CONFIGURATIONS")
print("="*80)
print()

configs = [
    # (long_thresh, short_thresh, gate_thresh)
    (0.65, 0.70, 0.0015),  # Original (샘플 테스트 최고 성능)
    (0.60, 0.70, 0.0015),  # Lower LONG thresh
    (0.65, 0.65, 0.0015),  # Lower SHORT thresh
    (0.65, 0.70, 0.0010),  # Lower gate (more SHORT)
    (0.65, 0.70, 0.0020),  # Higher gate (less SHORT)
]

all_results = []

for config_idx, (long_t, short_t, gate_t) in enumerate(configs, 1):
    print(f"\nConfig {config_idx}/5: LONG={long_t}, SHORT={short_t}, Gate={gate_t}")
    print("-" * 80)

    start_time = time.time()
    results_df = run_window_backtest(df, long_t, short_t, gate_t)
    test_time = time.time() - start_time

    if len(results_df) > 0:
        avg_return = results_df['total_return'].mean()
        avg_trades = results_df['total_trades'].mean()
        avg_long = results_df['long_trades'].mean()
        avg_short = results_df['short_trades'].mean()
        avg_wr = results_df['win_rate'].mean()

        print(f"  Return: {avg_return:.2f}% per window")
        print(f"  Trades: {avg_trades:.1f} (LONG {avg_long:.1f} + SHORT {avg_short:.1f})")
        print(f"  Win Rate: {avg_wr:.1f}%")
        print(f"  Windows: {len(results_df)}")
        print(f"  Time: {test_time:.1f}s")

        all_results.append({
            'Config': f"L{long_t}_S{short_t}_G{gate_t}",
            'LONG Thresh': long_t,
            'SHORT Thresh': short_t,
            'Gate Thresh': gate_t,
            'Avg Return (%)': avg_return,
            'Avg Trades': avg_trades,
            'Avg LONG': avg_long,
            'Avg SHORT': avg_short,
            'Win Rate (%)': avg_wr,
            'Windows': len(results_df)
        })

# Final comparison
print("\n" + "="*80)
print("FULL PERIOD BACKTEST RESULTS")
print("="*80)
print()

comparison_df = pd.DataFrame(all_results)
comparison_df = comparison_df.sort_values('Avg Return (%)', ascending=False)

print(comparison_df.to_string(index=False))

# Best configuration
best = comparison_df.iloc[0]
print(f"\n{'='*80}")
print("🏆 BEST CONFIGURATION")
print(f"{'='*80}\n")
print(f"Config: {best['Config']}")
print(f"  LONG Threshold: {best['LONG Thresh']}")
print(f"  SHORT Threshold: {best['SHORT Thresh']}")
print(f"  Gate Threshold: {best['Gate Thresh']}")
print(f"\nPerformance:")
print(f"  Return: {best['Avg Return (%)']:.2f}% per window")
print(f"  Trades: {best['Avg Trades']:.1f} (LONG {best['Avg LONG']:.1f} + SHORT {best['Avg SHORT']:.1f})")
print(f"  Win Rate: {best['Win Rate (%)']:.1f}%")
print(f"  Windows: {int(best['Windows'])}")

# Annualized return estimate
days_per_window = 5
windows_per_year = 365 / days_per_window
annualized_return = (1 + best['Avg Return (%)'] / 100) ** windows_per_year - 1
print(f"\n📊 Annualized Return Estimate: {annualized_return * 100:.1f}%")

# Transaction cost analysis
transaction_cost_pct = 0.05  # 0.05% per trade
slippage_pct = 0.02  # 0.02% per trade
total_cost_per_trade = transaction_cost_pct + slippage_pct

cost_per_window = best['Avg Trades'] * (total_cost_per_trade / 100)
net_return = best['Avg Return (%)'] - (cost_per_window * 100)
net_annualized = (1 + net_return / 100) ** windows_per_year - 1

print(f"\n💰 After Transaction Costs:")
print(f"  Cost per trade: {total_cost_per_trade}%")
print(f"  Cost per window: {cost_per_window * 100:.2f}%")
print(f"  Net return: {net_return:.2f}% per window")
print(f"  Net annualized: {net_annualized * 100:.1f}%")

# Save results
output_file = RESULTS_DIR / f"full_backtest_opportunity_gating_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
comparison_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")

print(f"\n{'='*80}")
print("FULL BACKTEST COMPLETE")
print(f"{'='*80}\n")
