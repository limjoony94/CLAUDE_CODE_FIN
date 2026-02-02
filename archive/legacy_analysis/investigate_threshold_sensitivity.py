"""
Investigate Threshold Sensitivity Issue
========================================

TEST 6 revealed high sensitivity:
- Entry 0.80: +1.67%
- Entry 0.81: -10.24% (!!!)

Why does 1% change cause -11.91% swing?
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib

from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

LEVERAGE = 4
STOP_LOSS = -0.03
MAX_HOLD = 120
FEE_RATE = 0.0005
ML_EXIT_THRESHOLD = 0.80

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("THRESHOLD SENSITIVITY INVESTIGATION")
print("="*80)
print()

# Load data
csv_file = sorted(DATA_DIR.glob("BTCUSDT_5m_raw_35days_*.csv"), reverse=True)[0]
df_raw = pd.read_csv(csv_file)
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df_raw[col] = df_raw[col].astype(float)

df = calculate_all_features_enhanced_v2(df_raw.copy(), phase='phase1')
df = prepare_exit_features(df)

split_date = '2025-10-28'
df_val = df[df['timestamp'] >= split_date].copy().reset_index(drop=True)

# Load models
short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines()]

long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines()]

short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines()]

# Calculate probabilities
X_long_entry = df_val[long_entry_features].values
X_long_entry_scaled = long_entry_scaler.transform(X_long_entry)
long_entry_probs = long_entry_model.predict_proba(X_long_entry_scaled)[:, 1]

X_short_entry = df_val[short_entry_features].values
X_short_entry_scaled = short_entry_scaler.transform(X_short_entry)
short_entry_probs = short_entry_model.predict_proba(X_short_entry_scaled)[:, 1]

X_long_exit = df_val[long_exit_features].values
X_long_exit_scaled = long_exit_scaler.transform(X_long_exit)
long_exit_probs = long_exit_model.predict_proba(X_long_exit_scaled)[:, 1]

X_short_exit = df_val[short_exit_features].values
X_short_exit_scaled = short_exit_scaler.transform(X_short_exit)
short_exit_probs = short_exit_model.predict_proba(X_short_exit_scaled)[:, 1]

print("✅ Data loaded")
print()

# ==============================================================================
# Analyze Entry Probability Distribution
# ==============================================================================

print("-"*80)
print("ENTRY PROBABILITY DISTRIBUTION")
print("-"*80)
print()

print("LONG entry probability distribution:")
print(f"  0.70-0.80: {((long_entry_probs >= 0.70) & (long_entry_probs < 0.80)).sum()} candles")
print(f"  0.80-0.81: {((long_entry_probs >= 0.80) & (long_entry_probs < 0.81)).sum()} candles ← 0.80 cutoff")
print(f"  0.81-0.90: {((long_entry_probs >= 0.81) & (long_entry_probs < 0.90)).sum()} candles ← 0.81 cutoff")
print(f"  0.90-1.00: {(long_entry_probs >= 0.90).sum()} candles")
print()

print("SHORT entry probability distribution:")
print(f"  0.70-0.80: {((short_entry_probs >= 0.70) & (short_entry_probs < 0.80)).sum()} candles")
print(f"  0.80-0.81: {((short_entry_probs >= 0.80) & (short_entry_probs < 0.81)).sum()} candles ← 0.80 cutoff")
print(f"  0.81-0.90: {((short_entry_probs >= 0.81) & (short_entry_probs < 0.90)).sum()} candles ← 0.81 cutoff")
print(f"  0.90-1.00: {(short_entry_probs >= 0.90).sum()} candles")
print()

# Count signal changes
signals_0_80 = ((long_entry_probs >= 0.80) | (short_entry_probs >= 0.80)).sum()
signals_0_81 = ((long_entry_probs >= 0.81) | (short_entry_probs >= 0.81)).sum()
lost_signals = signals_0_80 - signals_0_81

print(f"Signal count change:")
print(f"  Entry ≥0.80: {signals_0_80} candles")
print(f"  Entry ≥0.81: {signals_0_81} candles")
print(f"  Lost signals: {lost_signals} ({lost_signals/signals_0_80*100:.1f}%)")
print()

# ==============================================================================
# Run Detailed Backtest Comparison
# ==============================================================================

print("-"*80)
print("DETAILED BACKTEST COMPARISON")
print("-"*80)
print()

def run_detailed_backtest(entry_thresh):
    """Run backtest and track all trades"""
    balance = 100.0
    position = None
    trades = []

    for i in range(len(df_val)):
        current_price = df_val.iloc[i]['close']
        current_time = df_val.iloc[i]['timestamp']

        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            side = position['side']
            hold_time = i - entry_idx

            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                exit_prob = long_exit_probs[i]
            else:
                pnl_pct = (entry_price - current_price) / entry_price
                exit_prob = short_exit_probs[i]

            leveraged_pnl_pct = pnl_pct * LEVERAGE

            exit_reason = None
            if leveraged_pnl_pct <= STOP_LOSS:
                exit_reason = "Stop Loss"
            elif exit_prob >= ML_EXIT_THRESHOLD:
                exit_reason = "ML Exit"
            elif hold_time >= MAX_HOLD:
                exit_reason = "Max Hold"

            if exit_reason:
                pnl_usd = balance * leveraged_pnl_pct
                entry_fee = balance * FEE_RATE
                exit_fee = (balance + pnl_usd) * FEE_RATE
                pnl_net = pnl_usd - (entry_fee + exit_fee)

                balance += pnl_net

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'entry_prob': position['entry_prob'],
                    'pnl_net': pnl_net,
                    'exit_reason': exit_reason
                })

                position = None

        if position is None:
            long_signal = long_entry_probs[i] >= entry_thresh
            short_signal = short_entry_probs[i] >= entry_thresh

            if long_signal or short_signal:
                long_ev = long_entry_probs[i] if long_signal else 0
                short_ev = short_entry_probs[i] if short_signal else 0

                if short_ev > long_ev + 0.001:
                    chosen_side = 'SHORT'
                    chosen_prob = short_ev
                elif long_signal:
                    chosen_side = 'LONG'
                    chosen_prob = long_ev
                else:
                    chosen_side = None

                if chosen_side:
                    position = {
                        'side': chosen_side,
                        'entry_idx': i,
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'entry_prob': chosen_prob
                    }

    return balance, pd.DataFrame(trades) if trades else pd.DataFrame()

# Run both configurations
print("Running Entry=0.80...")
balance_0_80, trades_0_80 = run_detailed_backtest(0.80)
print(f"✅ Entry=0.80: ${balance_0_80:.2f}, {len(trades_0_80)} trades")

print("Running Entry=0.81...")
balance_0_81, trades_0_81 = run_detailed_backtest(0.81)
print(f"✅ Entry=0.81: ${balance_0_81:.2f}, {len(trades_0_81)} trades")
print()

return_0_80 = (balance_0_80 - 100) / 100 * 100
return_0_81 = (balance_0_81 - 100) / 100 * 100

print(f"Return comparison:")
print(f"  Entry=0.80: {return_0_80:+.2f}%")
print(f"  Entry=0.81: {return_0_81:+.2f}%")
print(f"  Difference: {return_0_81 - return_0_80:+.2f}%")
print()

# ==============================================================================
# Identify Lost Trades
# ==============================================================================

print("-"*80)
print("IDENTIFYING LOST TRADES (0.80 → 0.81)")
print("-"*80)
print()

# Find trades that exist in 0.80 but not in 0.81
if len(trades_0_80) > 0 and len(trades_0_81) > 0:
    # Merge to find missing trades
    trades_0_80['config'] = '0.80'
    trades_0_81['config'] = '0.81'

    # Trades in 0.80 that match 0.81
    merged = pd.merge(
        trades_0_80[['entry_time', 'side', 'pnl_net']],
        trades_0_81[['entry_time', 'side']],
        on=['entry_time', 'side'],
        how='left',
        indicator=True
    )

    lost_trades = merged[merged['_merge'] == 'left_only']

    if len(lost_trades) > 0:
        print(f"Lost {len(lost_trades)} trades when threshold increased to 0.81:")
        print()

        # Analyze lost trades
        lost_long = lost_trades[lost_trades['side'] == 'LONG']
        lost_short = lost_trades[lost_trades['side'] == 'SHORT']

        print(f"Lost LONG trades: {len(lost_long)} (P&L: ${lost_long['pnl_net'].sum():+.2f})")
        if len(lost_long) > 0:
            print(f"  Avg P&L: ${lost_long['pnl_net'].mean():+.2f}")
            print(f"  Wins: {(lost_long['pnl_net'] > 0).sum()}")
            print(f"  Losses: {(lost_long['pnl_net'] <= 0).sum()}")
        print()

        print(f"Lost SHORT trades: {len(lost_short)} (P&L: ${lost_short['pnl_net'].sum():+.2f})")
        if len(lost_short) > 0:
            print(f"  Avg P&L: ${lost_short['pnl_net'].mean():+.2f}")
            print(f"  Wins: {(lost_short['pnl_net'] > 0).sum()}")
            print(f"  Losses: {(lost_short['pnl_net'] <= 0).sum()}")
        print()

        # Show top profitable lost trades
        top_lost = lost_trades.nlargest(5, 'pnl_net')
        print("Top 5 profitable trades LOST at 0.81:")
        for i, (idx, trade) in enumerate(top_lost.iterrows(), 1):
            # Get entry probability from original backtest
            entry_time = trade['entry_time']
            side = trade['side']

            # Find the original entry
            candle_idx = df_val[df_val['timestamp'] == entry_time].index[0]
            if side == 'LONG':
                entry_prob = long_entry_probs[candle_idx]
            else:
                entry_prob = short_entry_probs[candle_idx]

            print(f"{i}. {side} @ {entry_time}: P&L ${trade['pnl_net']:+.2f}, Prob {entry_prob:.4f}")
        print()

        # Calculate impact
        total_lost_pnl = lost_trades['pnl_net'].sum()
        print(f"Total P&L from lost trades: ${total_lost_pnl:+.2f}")
        print(f"This explains {abs(total_lost_pnl / (return_0_81 - return_0_80) * 100):.1f}% of the performance difference")
        print()

# ==============================================================================
# Entry Probability Analysis of Lost Trades
# ==============================================================================

print("-"*80)
print("ENTRY PROBABILITY ANALYSIS OF LOST TRADES")
print("-"*80)
print()

# Identify candles with entry probs in 0.80-0.81 range
long_80_81_range = ((long_entry_probs >= 0.80) & (long_entry_probs < 0.81))
short_80_81_range = ((short_entry_probs >= 0.80) & (short_entry_probs < 0.81))

print(f"Candles with entry probability 0.80-0.81:")
print(f"  LONG: {long_80_81_range.sum()} candles")
print(f"  SHORT: {short_80_81_range.sum()} candles")
print(f"  Total: {(long_80_81_range | short_80_81_range).sum()} candles")
print()

print(f"LONG probability distribution in 0.80-0.81 range:")
if long_80_81_range.sum() > 0:
    range_probs = long_entry_probs[long_80_81_range]
    print(f"  Mean: {range_probs.mean():.4f}")
    print(f"  Min: {range_probs.min():.4f}")
    print(f"  Max: {range_probs.max():.4f}")
else:
    print("  (None)")
print()

print(f"SHORT probability distribution in 0.80-0.81 range:")
if short_80_81_range.sum() > 0:
    range_probs = short_entry_probs[short_80_81_range]
    print(f"  Mean: {range_probs.mean():.4f}")
    print(f"  Min: {range_probs.min():.4f}")
    print(f"  Max: {range_probs.max():.4f}")
else:
    print("  (None)")
print()

# ==============================================================================
# Conclusion
# ==============================================================================

print("="*80)
print("SENSITIVITY ANALYSIS CONCLUSION")
print("="*80)
print()

if len(lost_trades) > 0:
    print(f"🔍 ROOT CAUSE IDENTIFIED:")
    print(f"   Increasing threshold 0.80 → 0.81 loses {len(lost_trades)} trades")
    print(f"   Lost trades contribute ${lost_trades['pnl_net'].sum():+.2f} to P&L")
    print(f"   Most lost trades have entry probability 0.80-0.81 (marginal signals)")
    print()

    if lost_trades['pnl_net'].sum() > 0:
        print("⚠️  RISK: The 0.80-0.81 range contains PROFITABLE trades")
        print("   Recommendation: Entry=0.80 is optimal despite sensitivity")
        print("   Alternative: Test Entry=0.79 for more robustness")
    else:
        print("✅ FINDING: The 0.80-0.81 range contains UNPROFITABLE trades")
        print("   Recommendation: Entry=0.81 may be more robust")
        print("   But: Overall return is worse (-10.24% vs +1.67%)")

    print()

    avg_lost_pnl = lost_trades['pnl_net'].mean()
    if avg_lost_pnl > 0:
        print(f"💡 KEY INSIGHT:")
        print(f"   Average lost trade P&L: ${avg_lost_pnl:+.2f}")
        print(f"   Conclusion: Entry=0.80 captures valuable marginal signals")
        print(f"   Risk: Performance depends on these marginal trades")
    else:
        print(f"💡 KEY INSIGHT:")
        print(f"   Average lost trade P&L: ${avg_lost_pnl:+.2f}")
        print(f"   Paradox: Lost trades are unprofitable, but overall return worse at 0.81")
        print(f"   Likely: Trade timing/sequence effects matter")

print()
print("="*80)
