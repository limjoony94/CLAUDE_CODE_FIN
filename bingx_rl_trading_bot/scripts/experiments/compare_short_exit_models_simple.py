"""
Simple SHORT Exit Model Comparison
===================================

Compare two SHORT exit models on simulated SHORT trades:
- Current: xgboost_short_exit_oppgating_improved_20251017_152440.pkl (25 features)
- New: xgboost_short_exit_specialized_20251018_053307.pkl (32 features + reversal)

Focus: Exit timing quality only (not full system simulation)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.reversal_detection_features import add_all_reversal_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

# Configuration
DATA_FILE = project_root / "data/historical/BTCUSDT_5m_max.csv"
MODELS_DIR = project_root / "models"

# Models to compare
CURRENT_MODEL = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
CURRENT_SCALER = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl"
CURRENT_FEATURES = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt"

NEW_MODEL = MODELS_DIR / "xgboost_short_exit_specialized_20251018_053307.pkl"
NEW_SCALER = MODELS_DIR / "xgboost_short_exit_specialized_20251018_053307_scaler.pkl"
NEW_FEATURES = MODELS_DIR / "xgboost_short_exit_specialized_20251018_053307_features.txt"

# SHORT Entry Model (to find entry points)
SHORT_ENTRY_MODEL = MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl"
SHORT_ENTRY_SCALER = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl"
SHORT_ENTRY_FEATURES = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"

# Parameters
SHORT_THRESHOLD = 0.70
EXIT_THRESHOLD = 0.72
MAX_HOLD_CANDLES = 240  # 20 hours
LEVERAGE = 4

print("="*80)
print("Simple SHORT Exit Model Comparison")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"✅ {len(df):,} candles loaded")

# Calculate features
print("\nCalculating features...")
df = calculate_all_features(df.copy())
df = prepare_exit_features(df)  # Add exit-specific features (25 features)
df = add_all_reversal_features(df)  # Add reversal features (8 features)
df = df.dropna().reset_index(drop=True)
print(f"✅ Features calculated: {len(df):,} candles")

# Load models
print("\nLoading models...")
with open(SHORT_ENTRY_MODEL, 'rb') as f:
    entry_model = pickle.load(f)
with open(SHORT_ENTRY_SCALER, 'rb') as f:
    entry_scaler = pickle.load(f)
with open(SHORT_ENTRY_FEATURES, 'r') as f:
    entry_features = [line.strip() for line in f]

with open(CURRENT_MODEL, 'rb') as f:
    current_exit_model = pickle.load(f)
with open(CURRENT_SCALER, 'rb') as f:
    current_exit_scaler = pickle.load(f)
with open(CURRENT_FEATURES, 'r') as f:
    current_exit_features = [line.strip() for line in f]

with open(NEW_MODEL, 'rb') as f:
    new_exit_model = pickle.load(f)
with open(NEW_SCALER, 'rb') as f:
    new_exit_scaler = pickle.load(f)
with open(NEW_FEATURES, 'r') as f:
    new_exit_features = [line.strip() for line in f]

print(f"✅ Models loaded")
print(f"   SHORT Entry: {len(entry_features)} features")
print(f"   Current Exit: {len(current_exit_features)} features")
print(f"   New Exit: {len(new_exit_features)} features")

# Find SHORT entry points
print("\n" + "="*80)
print("Finding SHORT Entry Points (threshold: 0.70)")
print("="*80)

entries = []
for i in range(100, len(df)):
    # Get SHORT entry probability
    latest = df.iloc[i:i+1]
    short_feat = latest[entry_features].values
    short_feat_scaled = entry_scaler.transform(short_feat)
    short_prob = entry_model.predict_proba(short_feat_scaled)[0][1]

    if short_prob >= SHORT_THRESHOLD:
        entries.append({
            'idx': i,
            'time': df.iloc[i]['timestamp'],
            'price': df.iloc[i]['close'],
            'prob': short_prob
        })

print(f"✅ Found {len(entries)} SHORT entry signals")

# Simulate trades with both exit models
def test_exit_model(entries, df, exit_model, exit_scaler, exit_features, model_name):
    """Test an exit model on SHORT trades"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}")

    trades = []

    for entry in entries:
        entry_idx = entry['idx']
        entry_price = entry['price']
        entry_time = entry['time']
        entry_prob = entry['prob']

        # Simulate holding until exit signal or max hold
        exit_idx = None
        exit_reason = None
        exit_prob = None

        for i in range(entry_idx + 1, min(entry_idx + MAX_HOLD_CANDLES, len(df))):
            current_price = df.iloc[i]['close']
            hold_candles = i - entry_idx

            # Calculate P&L
            price_change = (entry_price - current_price) / entry_price
            leveraged_pnl = price_change * LEVERAGE

            # Emergency exits
            if leveraged_pnl <= -0.04:  # Stop loss
                exit_idx = i
                exit_reason = "Stop Loss"
                break
            elif leveraged_pnl >= 0.03:  # Take profit
                exit_idx = i
                exit_reason = "Take Profit"
                break
            elif hold_candles >= MAX_HOLD_CANDLES:  # Max hold
                exit_idx = i
                exit_reason = "Max Hold"
                break

            # ML exit signal
            latest = df.iloc[i:i+1]
            try:
                exit_feat = latest[exit_features].values
                exit_feat_scaled = exit_scaler.transform(exit_feat)
                exit_prob = exit_model.predict_proba(exit_feat_scaled)[0][1]

                if exit_prob >= EXIT_THRESHOLD:
                    exit_idx = i
                    exit_reason = f"ML Exit"
                    break
            except Exception as e:
                print(f"   Warning: Exit model error at {i}: {e}")
                continue

        # If no exit triggered, use last candle
        if exit_idx is None:
            exit_idx = min(entry_idx + MAX_HOLD_CANDLES - 1, len(df) - 1)
            exit_reason = "End of Data"

        # Calculate final P&L
        exit_price = df.iloc[exit_idx]['close']
        exit_time = df.iloc[exit_idx]['timestamp']
        price_change = (entry_price - exit_price) / entry_price
        leveraged_pnl = price_change * LEVERAGE
        hold_candles = exit_idx - entry_idx

        # Find best possible exit in next 240 candles
        future_window = df.iloc[entry_idx:min(entry_idx + 240, len(df))]
        best_price = future_window['low'].min()
        best_pnl = (entry_price - best_price) / entry_price * LEVERAGE
        opportunity_cost = best_pnl - leveraged_pnl

        # Classify exit timing
        if best_pnl < 0:
            timing = "GOOD (avoided worse)"
        elif leveraged_pnl > 0 and opportunity_cost < 0.005:
            timing = "GOOD (near peak)"
        elif leveraged_pnl > 0 and opportunity_cost >= 0.005:
            timing = "LATE (gave back profit)"
        elif leveraged_pnl < 0:
            timing = "EARLY (missed profit)"
        else:
            timing = "NEUTRAL"

        trades.append({
            'entry_idx': entry_idx,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'entry_prob': entry_prob,
            'exit_idx': exit_idx,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'exit_prob': exit_prob,
            'hold_candles': hold_candles,
            'pnl_pct': leveraged_pnl,
            'best_pnl': best_pnl,
            'opportunity_cost': opportunity_cost,
            'timing': timing
        })

    # Analyze results
    df_trades = pd.DataFrame(trades)

    print(f"\nResults ({len(df_trades)} trades):")
    print(f"  Win Rate: {(df_trades['pnl_pct'] > 0).mean():.1%}")
    print(f"  Avg P&L: {df_trades['pnl_pct'].mean():.2%}")
    print(f"  Avg Hold: {df_trades['hold_candles'].mean():.1f} candles")

    print(f"\nExit Timing Quality:")
    timing_counts = df_trades['timing'].value_counts()
    for timing, count in timing_counts.items():
        pct = count / len(df_trades) * 100
        print(f"  {timing}: {count} ({pct:.1f}%)")

    print(f"\nOpportunity Cost:")
    print(f"  Mean: {df_trades['opportunity_cost'].mean():.2%}")
    print(f"  Median: {df_trades['opportunity_cost'].median():.2%}")

    late_exits = df_trades[df_trades['timing'] == 'LATE (gave back profit)']
    if len(late_exits) > 0:
        print(f"  Late Exits: {len(late_exits)} ({len(late_exits)/len(df_trades)*100:.1f}%)")
        print(f"  Late Exit Avg Cost: {late_exits['opportunity_cost'].mean():.2%}")

    print(f"\nExit Reasons:")
    reason_counts = df_trades['exit_reason'].value_counts()
    for reason, count in reason_counts.items():
        pct = count / len(df_trades) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")

    return df_trades

# Test both models
current_results = test_exit_model(entries, df, current_exit_model, current_exit_scaler, current_exit_features, "Current Model (25 features)")
new_results = test_exit_model(entries, df, new_exit_model, new_exit_scaler, new_exit_features, "New Model (32 features + reversal)")

# Comparison
print("\n" + "="*80)
print("COMPARISON: New vs Current")
print("="*80)

print(f"\nWin Rate:")
print(f"  Current: {(current_results['pnl_pct'] > 0).mean():.1%}")
print(f"  New: {(new_results['pnl_pct'] > 0).mean():.1%}")
print(f"  Change: {((new_results['pnl_pct'] > 0).mean() - (current_results['pnl_pct'] > 0).mean()) * 100:+.1f} pp")

print(f"\nAvg P&L:")
print(f"  Current: {current_results['pnl_pct'].mean():.2%}")
print(f"  New: {new_results['pnl_pct'].mean():.2%}")
print(f"  Improvement: {(new_results['pnl_pct'].mean() - current_results['pnl_pct'].mean()):.2%}")

print(f"\nOpportunity Cost:")
print(f"  Current: {current_results['opportunity_cost'].mean():.2%}")
print(f"  New: {new_results['opportunity_cost'].mean():.2%}")
print(f"  Improvement: {(current_results['opportunity_cost'].mean() - new_results['opportunity_cost'].mean()):.2%}")

current_late = (current_results['timing'] == 'LATE (gave back profit)').sum()
new_late = (new_results['timing'] == 'LATE (gave back profit)').sum()
print(f"\nLate Exits:")
print(f"  Current: {current_late} ({current_late/len(current_results)*100:.1f}%)")
print(f"  New: {new_late} ({new_late/len(new_results)*100:.1f}%)")
print(f"  Reduction: {current_late - new_late} ({(current_late - new_late)/len(current_results)*100:.1f} pp)")

# Decision
improvements = 0

if (new_results['pnl_pct'] > 0).mean() > (current_results['pnl_pct'] > 0).mean():
    improvements += 1
    print("\n✅ Win rate improved")
else:
    print("\n❌ Win rate declined")

if new_results['pnl_pct'].mean() > current_results['pnl_pct'].mean():
    improvements += 1
    print("✅ Avg P&L improved")
else:
    print("❌ Avg P&L declined")

if new_results['opportunity_cost'].mean() < current_results['opportunity_cost'].mean():
    improvements += 1
    print("✅ Opportunity cost improved")
else:
    print("❌ Opportunity cost worsened")

if new_late < current_late:
    improvements += 1
    print("✅ Late exits reduced")
else:
    print("❌ Late exits increased")

print(f"\n{'='*80}")
print(f"SCORE: {improvements}/4 metrics improved")
print(f"{'='*80}")

if improvements >= 3:
    print("\n🎉 RECOMMENDATION: Deploy New Model")
    print("   - Strong improvement across metrics")
    print("   - Ready for testnet validation")
elif improvements == 2:
    print("\n⚠️ RECOMMENDATION: Borderline - Consider deployment if key metrics improved")
else:
    print("\n❌ RECOMMENDATION: Keep Current Model")
    print("   - New model did not show sufficient improvement")

print("\n" + "="*80)
