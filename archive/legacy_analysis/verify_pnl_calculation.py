"""
Verify P&L Calculation Logic in simulate_trade()
=================================================

Extract sample trades from backtest and manually verify:
1. Price change calculation
2. Leverage application
3. Stop Loss triggering
4. Fee calculation
5. Balance update logic

Created: 2025-10-30
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib

# Configuration (from backtest script)
ENTRY_THRESHOLD = 0.75
ML_EXIT_THRESHOLD = 0.75
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
INITIAL_BALANCE = 10000
MIN_POSITION_PCT = 0.20
MAX_POSITION_PCT = 0.95
TAKER_FEE = 0.0005

DATA_DIR = PROJECT_ROOT / "data" / "features"
CSV_FILE = DATA_DIR / "BTCUSDT_5m_features_exit_ready.csv"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("P&L CALCULATION VERIFICATION")
print("="*80)
print()

# Load data
print("Loading data...")
df = pd.read_csv(CSV_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"✅ Loaded {len(df):,} candles")
print()

# Load Exit models (for ML Exit verification)
print("Loading Exit models...")
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Exit model loaded ({len(long_exit_features)} features)")
print()

# Manual trade simulation function
def simulate_trade_manual(df, entry_idx, side, exit_model, exit_scaler, exit_features, verbose=True):
    """Manually simulate a trade with detailed logging"""

    entry_price = df['close'].iloc[entry_idx]
    entry_time = df['timestamp'].iloc[entry_idx]

    if verbose:
        print(f"\n{'='*80}")
        print(f"TRADE SIMULATION: {side}")
        print(f"{'='*80}")
        print(f"Entry Index: {entry_idx}")
        print(f"Entry Time: {entry_time}")
        print(f"Entry Price: ${entry_price:,.2f}")
        print()

    max_hold_end = min(entry_idx + EMERGENCY_MAX_HOLD + 1, len(df))
    prices = df['close'].iloc[entry_idx+1:max_hold_end].values

    if verbose:
        print(f"Max Hold: {EMERGENCY_MAX_HOLD} candles")
        print(f"Price range: {len(prices)} candles available")
        print()

    # Iterate through each candle
    for i, current_price in enumerate(prices):
        current_idx = entry_idx + 1 + i
        hold_time = i + 1

        # Calculate P&L
        if side == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        leveraged_pnl = pnl_pct * LEVERAGE

        # Check Stop Loss
        if leveraged_pnl <= EMERGENCY_STOP_LOSS:
            fee_total = 2 * TAKER_FEE * LEVERAGE
            final_pnl = leveraged_pnl - fee_total

            if verbose:
                print(f"🛑 STOP LOSS TRIGGERED (Candle {hold_time})")
                print(f"  Exit Price: ${current_price:,.2f}")
                print(f"  Price Change: {pnl_pct*100:.4f}%")
                print(f"  Leveraged P&L: {leveraged_pnl*100:.4f}%")
                print(f"  Fee (0.05% × 2 × 4x): {fee_total*100:.4f}%")
                print(f"  Final P&L: {final_pnl*100:.4f}%")
                print()

                # Verify calculation
                print(f"  Verification:")
                print(f"    Entry: ${entry_price:,.2f}")
                print(f"    Exit: ${current_price:,.2f}")
                print(f"    Price Δ: {pnl_pct*100:.4f}%")
                print(f"    Leverage 4x: {pnl_pct*100:.4f}% × 4 = {leveraged_pnl*100:.4f}%")
                print(f"    SL Threshold: {EMERGENCY_STOP_LOSS*100:.2f}%")
                print(f"    Triggered: {leveraged_pnl:.6f} <= {EMERGENCY_STOP_LOSS:.6f} ✅")

            return final_pnl, hold_time, 'stop_loss', current_idx

        # Check ML Exit
        try:
            exit_feat = df[exit_features].iloc[current_idx:current_idx+1].values
            if not np.isnan(exit_feat).any():
                exit_scaled = exit_scaler.transform(exit_feat)
                exit_prob = exit_model.predict_proba(exit_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    fee_total = 2 * TAKER_FEE * LEVERAGE
                    final_pnl = leveraged_pnl - fee_total

                    if verbose:
                        print(f"🎯 ML EXIT (Candle {hold_time})")
                        print(f"  Exit Probability: {exit_prob:.4f}")
                        print(f"  Exit Price: ${current_price:,.2f}")
                        print(f"  Price Change: {pnl_pct*100:.4f}%")
                        print(f"  Leveraged P&L: {leveraged_pnl*100:.4f}%")
                        print(f"  Fee: {fee_total*100:.4f}%")
                        print(f"  Final P&L: {final_pnl*100:.4f}%")

                    return final_pnl, hold_time, 'ml_exit', current_idx
        except Exception as e:
            if verbose and i == 0:
                print(f"  ⚠️ ML Exit check failed: {e}")
            pass

        # Check Max Hold
        if hold_time >= EMERGENCY_MAX_HOLD:
            fee_total = 2 * TAKER_FEE * LEVERAGE
            final_pnl = leveraged_pnl - fee_total

            if verbose:
                print(f"⏰ MAX HOLD (Candle {hold_time})")
                print(f"  Exit Price: ${current_price:,.2f}")
                print(f"  Price Change: {pnl_pct*100:.4f}%")
                print(f"  Leveraged P&L: {leveraged_pnl*100:.4f}%")
                print(f"  Fee: {fee_total*100:.4f}%")
                print(f"  Final P&L: {final_pnl*100:.4f}%")

            return final_pnl, hold_time, 'max_hold', current_idx

    # Fallback
    final_pnl = pnl_series[-1] if len(pnl_series) > 0 else 0
    fee_total = 2 * TAKER_FEE * LEVERAGE

    if verbose:
        print(f"📍 DATA END")
        print(f"  Final P&L: {final_pnl*100:.4f}%")

    return final_pnl * LEVERAGE - fee_total, len(prices), 'data_end', entry_idx + len(prices)

# Sample trades to verify
print("="*80)
print("SAMPLING TRADES FOR VERIFICATION")
print("="*80)
print()

# Sample 1: Early in dataset (index 1000)
print("Sample 1: LONG at index 1000")
pnl1, hold1, reason1, exit_idx1 = simulate_trade_manual(
    df, 1000, 'LONG', long_exit_model, long_exit_scaler, long_exit_features, verbose=True
)

# Sample 2: Mid dataset (index 15000)
print("\n" + "="*80)
print("Sample 2: SHORT at index 15000")
pnl2, hold2, reason2, exit_idx2 = simulate_trade_manual(
    df, 15000, 'SHORT', long_exit_model, long_exit_scaler, long_exit_features, verbose=True
)

# Sample 3: Late dataset (index 25000)
print("\n" + "="*80)
print("Sample 3: LONG at index 25000")
pnl3, hold3, reason3, exit_idx3 = simulate_trade_manual(
    df, 25000, 'LONG', long_exit_model, long_exit_scaler, long_exit_features, verbose=True
)

# Summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)
print()
print(f"Sample 1: P&L {pnl1*100:.4f}%, Hold {hold1}, Reason: {reason1}")
print(f"Sample 2: P&L {pnl2*100:.4f}%, Hold {hold2}, Reason: {reason2}")
print(f"Sample 3: P&L {pnl3*100:.4f}%, Hold {hold3}, Reason: {reason3}")
print()

# Balance impact simulation
print("="*80)
print("BALANCE IMPACT SIMULATION (3 trades)")
print("="*80)
print()

balance = 10000
position_sizes = [0.5, 0.7, 0.6]  # Example position sizes

for i, (pnl, pos_size) in enumerate(zip([pnl1, pnl2, pnl3], position_sizes), 1):
    position_value = balance * pos_size
    pnl_dollars = position_value * pnl
    new_balance = balance + pnl_dollars

    print(f"Trade {i}:")
    print(f"  Starting Balance: ${balance:,.2f}")
    print(f"  Position Size: {pos_size*100:.1f}% (${position_value:,.2f})")
    print(f"  P&L %: {pnl*100:.4f}%")
    print(f"  P&L $: ${pnl_dollars:,.2f}")
    print(f"  Ending Balance: ${new_balance:,.2f}")
    print(f"  Cumulative Return: {(new_balance/10000 - 1)*100:.2f}%")
    print()

    balance = new_balance

print("="*80)
print("VERIFICATION COMPLETE")
print("="*80)
