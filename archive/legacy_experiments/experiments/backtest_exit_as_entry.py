"""
Exit-as-Entry Strategy Backtest: Exit Models as Entry Signals
==============================================================

TEST HYPOTHESIS:
  LONG Exit 신호 → SHORT 진입 (LONG 위험 = 하락 예상 = SHORT 기회)
  SHORT Exit 신호 → LONG 진입 (SHORT 위험 = 상승 예상 = LONG 기회)

Configuration:
  Entry: EXIT MODELS (REVERSE LOGIC)
    - LONG Entry: SHORT Exit model (threshold 0.75)
    - SHORT Entry: LONG Exit model (threshold 0.75)

  Exit: EMERGENCY ONLY (no ML exit)
    - Stop Loss: -3% balance
    - Max Hold: 10 hours

  Leverage: 4x
  Position Sizing: Dynamic (20-95%)
  Period: Full historical data (108 windows, ~540 days)
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

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("EXIT-AS-ENTRY STRATEGY BACKTEST")
print("="*80)
print(f"\n💡 Hypothesis: Use Exit signals for OPPOSITE direction Entry\n")
print(f"  LONG Exit signal → SHORT Entry (LONG risk = SHORT opportunity)")
print(f"  SHORT Exit signal → LONG Entry (SHORT risk = LONG opportunity)\n")

# Strategy Parameters
LEVERAGE = 4
ENTRY_THRESHOLD = 0.75  # Same threshold for both (using exit models)
GATE_THRESHOLD = 0.001

# Exit Parameters (EMERGENCY ONLY)
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours
EMERGENCY_STOP_LOSS = -0.03  # -3% balance

# Expected Values (not used in this test)
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Initial capital
INITIAL_CAPITAL = 10000.0

# Trading Fees
TAKER_FEE = 0.0005  # 0.05%

# Load Exit Models (Will be used as ENTRY signals!)
print("Loading Exit models (Will be used as ENTRY signals - 20251024)...")
import joblib

long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ Exit models loaded (using as ENTRY):")
print(f"     LONG Exit → SHORT Entry ({len(long_exit_feature_columns)} features)")
print(f"     SHORT Exit → LONG Entry ({len(short_exit_feature_columns)} features)\n")

# Initialize Dynamic Position Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

# Load FULL data
print("Loading FULL historical data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"  ✅ Full data loaded: {len(df_full):,} candles (~{len(df_full)//288:.0f} days)\n")

# Calculate features
print("Calculating ALL features (Enhanced v2 + EXIT)...")
start_time = time.time()
df = calculate_all_features_enhanced_v2(df_full, phase='phase1')
df = prepare_exit_features(df)  # Need exit features!
feature_time = time.time() - start_time
print(f"  ✅ All features calculated ({feature_time:.1f}s)\n")

# Pre-calculate Entry signals (using EXIT models!)
print("Pre-calculating Entry signals (using EXIT models)...")
signal_start = time.time()
try:
    # LONG Entry = SHORT Exit model
    long_entry_feat = df[short_exit_feature_columns].values
    long_entry_feat_scaled = short_exit_scaler.transform(long_entry_feat)
    long_entry_probs = short_exit_model.predict_proba(long_entry_feat_scaled)[:, 1]
    df['long_entry_prob'] = long_entry_probs

    # SHORT Entry = LONG Exit model
    short_entry_feat = df[long_exit_feature_columns].values
    short_entry_feat_scaled = long_exit_scaler.transform(short_entry_feat)
    short_entry_probs = long_exit_model.predict_proba(short_entry_feat_scaled)[:, 1]
    df['short_entry_prob'] = short_entry_probs

    signal_time = time.time() - signal_start
    print(f"  ✅ Entry signals pre-calculated ({signal_time:.1f}s)")
    print(f"     LONG Entry (via SHORT Exit): {(df['long_entry_prob'] >= ENTRY_THRESHOLD).sum()} signals")
    print(f"     SHORT Entry (via LONG Exit): {(df['short_entry_prob'] >= ENTRY_THRESHOLD).sum()} signals\n")
except Exception as e:
    print(f"  ❌ Error pre-calculating signals: {e}")
    df['long_entry_prob'] = 0.0
    df['short_entry_prob'] = 0.0
    print(f"  ⚠️ Using zero probabilities\n")


def backtest_exit_as_entry(window_df):
    """
    EXIT-AS-ENTRY STRATEGY:
      - LONG Entry: When SHORT Exit model triggers (SHORT risk = LONG opportunity)
      - SHORT Entry: When LONG Exit model triggers (LONG risk = SHORT opportunity)
      - Exit: EMERGENCY ONLY (Stop Loss + Max Hold, NO ML Exit)

    Each window tested independently with $10,000 starting capital.
    """
    trades = []
    position = None
    capital = INITIAL_CAPITAL

    for i in range(len(window_df) - 1):
        current_price = window_df['close'].iloc[i]

        # Entry logic (using EXIT models as ENTRY signals)
        if position is None:
            long_entry_prob = window_df['long_entry_prob'].iloc[i]
            short_entry_prob = window_df['short_entry_prob'].iloc[i]

            # LONG entry (via SHORT Exit model)
            if long_entry_prob >= ENTRY_THRESHOLD:
                sizing_result = sizer.get_position_size_simple(
                    capital=capital,
                    signal_strength=long_entry_prob,
                    leverage=LEVERAGE
                )

                position = {
                    'side': 'LONG',
                    'entry_idx': i,
                    'entry_price': current_price,
                    'position_size_pct': sizing_result['position_size_pct'],
                    'position_value': sizing_result['position_value'],
                    'leveraged_value': sizing_result['leveraged_value'],
                    'quantity': sizing_result['leveraged_value'] / current_price,
                    'entry_prob': long_entry_prob
                }

            # SHORT entry (via LONG Exit model) - NO GATE (testing raw signal)
            elif short_entry_prob >= ENTRY_THRESHOLD:
                sizing_result = sizer.get_position_size_simple(
                    capital=capital,
                    signal_strength=short_entry_prob,
                    leverage=LEVERAGE
                )

                position = {
                    'side': 'SHORT',
                    'entry_idx': i,
                    'entry_price': current_price,
                    'position_size_pct': sizing_result['position_size_pct'],
                    'position_value': sizing_result['position_value'],
                    'leveraged_value': sizing_result['leveraged_value'],
                    'quantity': sizing_result['leveraged_value'] / current_price,
                    'entry_prob': short_entry_prob
                }

        # Exit logic (EMERGENCY ONLY - no ML exit!)
        if position is not None:
            time_in_pos = i - position['entry_idx']
            current_price = window_df['close'].iloc[i]

            # Calculate P&L
            entry_notional = position['quantity'] * position['entry_price']
            current_notional = position['quantity'] * current_price

            if position['side'] == 'LONG':
                pnl_usd = current_notional - entry_notional
                price_change_pct = (current_price - position['entry_price']) / position['entry_price']
            else:  # SHORT
                pnl_usd = entry_notional - current_notional
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl_pct = price_change_pct * LEVERAGE
            balance_loss_pct = pnl_usd / capital

            # Exit conditions (EMERGENCY ONLY)
            should_exit = False
            exit_reason = None

            # 1. Emergency Stop Loss
            if balance_loss_pct <= EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = 'emergency_stop_loss'

            # 2. Emergency Max Hold
            elif time_in_pos >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = 'emergency_max_hold'

            if should_exit:
                # Calculate commissions
                entry_commission = position['leveraged_value'] * TAKER_FEE
                exit_commission = position['quantity'] * current_price * TAKER_FEE
                total_commission = entry_commission + exit_commission

                net_pnl_usd = pnl_usd - total_commission
                capital += net_pnl_usd

                # Record trade
                trade = {
                    'side': position['side'],
                    'pnl_pct': price_change_pct,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd': pnl_usd,
                    'commission_usd': total_commission,
                    'net_pnl_usd': net_pnl_usd,
                    'hold_time': time_in_pos,
                    'exit_reason': exit_reason,
                    'position_size_pct': position['position_size_pct'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_prob': position['entry_prob']
                }

                trades.append(trade)
                position = None

    return trades, capital


def run_window_backtest(df):
    """Run backtest across all windows"""
    window_size = 1440
    step_size = 288
    num_windows = (len(df) - window_size) // step_size

    results = []
    all_trades = []

    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        if end_idx > len(df):
            break

        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        trades, final_capital = backtest_exit_as_entry(window_df)

        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            all_trades.extend(trades)

            total_return_pct = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

            long_trades = trades_df[trades_df['side'] == 'LONG']
            short_trades = trades_df[trades_df['side'] == 'SHORT']

            results.append({
                'window': window_idx,
                'total_trades': len(trades),
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'total_return_pct': total_return_pct,
                'win_rate': (trades_df['leveraged_pnl_pct'] > 0).sum() / len(trades) * 100,
                'final_capital': final_capital,
                'avg_position_size': trades_df['position_size_pct'].mean() * 100
            })

    return pd.DataFrame(results), all_trades


# Run backtest
print("="*80)
print("RUNNING EXIT-AS-ENTRY BACKTEST")
print("="*80)
print()

start_time = time.time()
results_df, all_trades = run_window_backtest(df)
test_time = time.time() - start_time

if len(results_df) > 0:
    # Calculate metrics
    avg_window_return = results_df['total_return_pct'].mean()
    avg_trades = results_df['total_trades'].mean()
    avg_long = results_df['long_trades'].mean()
    avg_short = results_df['short_trades'].mean()
    avg_wr = results_df['win_rate'].mean()
    avg_pos_size = results_df['avg_position_size'].mean()

    print(f"\n{'='*80}")
    print("EXIT-AS-ENTRY RESULTS (108 Windows)")
    print(f"{'='*80}\n")

    print(f"Performance (Each window starts with $10,000):")
    print(f"  Avg Return: {avg_window_return:.2f}% per 5-day window")
    print(f"  Win Rate: {avg_wr:.1f}%")
    print(f"  Trades: {avg_trades:.1f} per window")
    print(f"    LONG: {avg_long:.1f} ({avg_long/avg_trades*100:.1f}%)")
    print(f"    SHORT: {avg_short:.1f} ({avg_short/avg_trades*100:.1f}%)")
    print(f"  Avg Position Size: {avg_pos_size:.1f}%")
    print(f"  Test Time: {test_time:.1f}s")

    # Exit Reason Analysis
    if len(all_trades) > 0:
        all_trades_df = pd.DataFrame(all_trades)

        print(f"\n🚪 EXIT REASON ANALYSIS:")
        print(f"  Total Trades: {len(all_trades)}")

        exit_counts = all_trades_df['exit_reason'].value_counts()
        for reason, count in exit_counts.items():
            pct = count / len(all_trades) * 100
            reason_name = "Emergency Stop Loss" if reason == 'emergency_stop_loss' else \
                         "Emergency Max Hold" if reason == 'emergency_max_hold' else reason
            print(f"    {reason_name:<30s}: {count:>4d} ({pct:>5.1f}%)")

    # Calculate Sharpe ratio
    if len(results_df) > 1:
        returns = results_df['total_return_pct'].values
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(73)  # Annualized
        print(f"\n📈 Risk Metrics:")
        print(f"  Sharpe Ratio: {sharpe:.3f} (annualized)")
        print(f"  Max Drawdown: {results_df['total_return_pct'].min():.2f}%")
        print(f"  Std Dev: {np.std(returns):.2f}%")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = RESULTS_DIR / f"backtest_exit_as_entry_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")

    print(f"\n{'='*80}")
    print("BACKTEST COMPLETE")
    print(f"{'='*80}\n")

else:
    print("❌ No trades executed during backtest")
