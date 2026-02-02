"""
Signal Frequency Analysis
=========================

Analyze LONG and SHORT signal frequencies with production models.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("SIGNAL FREQUENCY ANALYSIS")
print("="*80)
print()

# =============================================================================
# CONFIGURATION
# =============================================================================

LEVERAGE = 4
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001

LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# =============================================================================
# LOAD PRODUCTION MODELS
# =============================================================================

print("Loading PRODUCTION Entry models...")
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print(f"  ✅ Models loaded\n")

# =============================================================================
# ANALYZE MULTIPLE TIME PERIODS
# =============================================================================

print("Loading historical data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"  ✅ Data loaded: {len(df_full):,} candles\n")

# Analyze different time periods
periods = [
    ("7 days", 7 * 288),
    ("14 days", 14 * 288),
    ("30 days", 30 * 288),
    ("Full dataset", len(df_full) - 500)  # Leave buffer for features
]

results = []

for period_name, candles in periods:
    print(f"{'='*80}")
    print(f"ANALYZING: {period_name.upper()} ({candles:,} candles)")
    print(f"{'='*80}")

    # Take recent data + buffer
    buffer = 500
    df_period = df_full.tail(candles + buffer).reset_index(drop=True)

    print(f"  Calculating features...")
    df = calculate_all_features(df_period)
    df = prepare_exit_features(df)
    df = df.dropna().reset_index(drop=True)

    # Take exact period after feature calculation
    df = df.tail(candles).reset_index(drop=True)

    print(f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"  Total candles: {len(df):,}\n")

    # Calculate signals
    print(f"  Calculating signals...")
    long_feat = df[long_feature_columns].values
    long_feat_scaled = long_scaler.transform(long_feat)
    long_probs = long_model.predict_proba(long_feat_scaled)[:, 1]

    short_feat = df[short_feature_columns].values
    short_feat_scaled = short_scaler.transform(short_feat)
    short_probs = short_model.predict_proba(short_feat_scaled)[:, 1]

    # Analyze LONG signals
    long_signals = long_probs >= LONG_THRESHOLD
    long_signal_count = np.sum(long_signals)
    long_signal_pct = (long_signal_count / len(df)) * 100

    # Analyze SHORT signals (before gating)
    short_signals_raw = short_probs >= SHORT_THRESHOLD
    short_signal_count_raw = np.sum(short_signals_raw)
    short_signal_pct_raw = (short_signal_count_raw / len(df)) * 100

    # Analyze SHORT signals (after gating)
    long_ev = long_probs * LONG_AVG_RETURN
    short_ev = short_probs * SHORT_AVG_RETURN
    opportunity_cost = short_ev - long_ev

    short_signals_gated = (short_probs >= SHORT_THRESHOLD) & (opportunity_cost > GATE_THRESHOLD)
    short_signal_count_gated = np.sum(short_signals_gated)
    short_signal_pct_gated = (short_signal_count_gated / len(df)) * 100

    # Calculate probability distributions
    long_prob_stats = {
        'mean': np.mean(long_probs),
        'median': np.median(long_probs),
        'std': np.std(long_probs),
        'max': np.max(long_probs),
        'min': np.min(long_probs)
    }

    short_prob_stats = {
        'mean': np.mean(short_probs),
        'median': np.median(short_probs),
        'std': np.std(short_probs),
        'max': np.max(short_probs),
        'min': np.min(short_probs)
    }

    # Store results
    result = {
        'period': period_name,
        'candles': len(df),
        'long_signal_count': long_signal_count,
        'long_signal_pct': long_signal_pct,
        'short_signal_count_raw': short_signal_count_raw,
        'short_signal_pct_raw': short_signal_pct_raw,
        'short_signal_count_gated': short_signal_count_gated,
        'short_signal_pct_gated': short_signal_pct_gated,
        'gate_filtered_pct': ((short_signal_count_raw - short_signal_count_gated) / short_signal_count_raw * 100) if short_signal_count_raw > 0 else 0,
        'long_prob_stats': long_prob_stats,
        'short_prob_stats': short_prob_stats
    }
    results.append(result)

    # Print results for this period
    print(f"\n  📊 LONG Signals:")
    print(f"     Count: {long_signal_count:,} ({long_signal_pct:.2f}%)")
    print(f"     Frequency: 1 signal per {len(df)/long_signal_count:.1f} candles (~{(len(df)/long_signal_count)*5/60:.1f} hours)" if long_signal_count > 0 else "     Frequency: No signals")
    print(f"     Probability stats:")
    print(f"       Mean: {long_prob_stats['mean']:.3f}")
    print(f"       Median: {long_prob_stats['median']:.3f}")
    print(f"       Range: {long_prob_stats['min']:.3f} - {long_prob_stats['max']:.3f}")

    print(f"\n  📊 SHORT Signals (Raw - before gating):")
    print(f"     Count: {short_signal_count_raw:,} ({short_signal_pct_raw:.2f}%)")
    print(f"     Frequency: 1 signal per {len(df)/short_signal_count_raw:.1f} candles (~{(len(df)/short_signal_count_raw)*5/60:.1f} hours)" if short_signal_count_raw > 0 else "     Frequency: No signals")
    print(f"     Probability stats:")
    print(f"       Mean: {short_prob_stats['mean']:.3f}")
    print(f"       Median: {short_prob_stats['median']:.3f}")
    print(f"       Range: {short_prob_stats['min']:.3f} - {short_prob_stats['max']:.3f}")

    print(f"\n  📊 SHORT Signals (After Opportunity Gating):")
    print(f"     Count: {short_signal_count_gated:,} ({short_signal_pct_gated:.2f}%)")
    print(f"     Frequency: 1 signal per {len(df)/short_signal_count_gated:.1f} candles (~{(len(df)/short_signal_count_gated)*5/60:.1f} hours)" if short_signal_count_gated > 0 else "     Frequency: No signals")
    print(f"     Gate filtered: {short_signal_count_raw - short_signal_count_gated:,} signals ({result['gate_filtered_pct']:.1f}%)")

    print(f"\n  ⚖️ Signal Ratio:")
    total_signals = long_signal_count + short_signal_count_gated
    if total_signals > 0:
        long_ratio = (long_signal_count / total_signals) * 100
        short_ratio = (short_signal_count_gated / total_signals) * 100
        print(f"     LONG: {long_ratio:.1f}%")
        print(f"     SHORT: {short_ratio:.1f}%")
    else:
        print(f"     No signals in this period")

    print()

# =============================================================================
# SUMMARY COMPARISON
# =============================================================================

print("="*80)
print("SUMMARY COMPARISON")
print("="*80)
print()

comparison_df = pd.DataFrame([
    {
        'Period': r['period'],
        'Candles': f"{r['candles']:,}",
        'LONG %': f"{r['long_signal_pct']:.2f}%",
        'SHORT Raw %': f"{r['short_signal_pct_raw']:.2f}%",
        'SHORT Gated %': f"{r['short_signal_pct_gated']:.2f}%",
        'Gate Filter': f"{r['gate_filtered_pct']:.1f}%"
    }
    for r in results
])

print(comparison_df.to_string(index=False))
print()

# Calculate average frequencies
print("="*80)
print("AVERAGE SIGNAL FREQUENCIES")
print("="*80)
print()

for r in results:
    print(f"{r['period']}:")
    if r['long_signal_count'] > 0:
        long_freq_hours = (r['candles'] / r['long_signal_count']) * 5 / 60
        print(f"  LONG: 1 signal every ~{long_freq_hours:.1f} hours")
    else:
        print(f"  LONG: No signals")

    if r['short_signal_count_gated'] > 0:
        short_freq_hours = (r['candles'] / r['short_signal_count_gated']) * 5 / 60
        print(f"  SHORT: 1 signal every ~{short_freq_hours:.1f} hours (after gating)")
    else:
        print(f"  SHORT: No signals (after gating)")
    print()

print("="*80)
print("SIGNAL FREQUENCY ANALYSIS COMPLETE")
print("="*80)
print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
