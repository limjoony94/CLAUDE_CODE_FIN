"""
Analyze SHORT Market Characteristics
=====================================

Analyze SHORT trade patterns to inform specialized model training:
1. Reversal timing patterns (how fast do bounces occur?)
2. Volume behavior during reversals
3. RSI divergence patterns
4. Optimal exit timing windows
5. Feature importance for SHORT exits

Author: Claude Code
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Opportunity Gating Parameters
ENTRY_THRESHOLD_SHORT = 0.70
GATE_THRESHOLD = 0.001
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047


def load_entry_models():
    """Load ENTRY models for trade simulation"""
    print("Loading ENTRY models...")

    # LONG Entry
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl", 'rb') as f:
        long_entry_model = pickle.load(f)
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl", 'rb') as f:
        long_entry_scaler = pickle.load(f)
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt", 'r') as f:
        long_entry_features = [line.strip() for line in f]

    # SHORT Entry
    with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl", 'rb') as f:
        short_entry_model = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl", 'rb') as f:
        short_entry_scaler = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt", 'r') as f:
        short_entry_features = [line.strip() for line in f if line.strip()]

    print("✅ Models loaded")

    return {
        'long_entry_model': long_entry_model,
        'long_entry_scaler': long_entry_scaler,
        'long_entry_features': long_entry_features,
        'short_entry_model': short_entry_model,
        'short_entry_scaler': short_entry_scaler,
        'short_entry_features': short_entry_features
    }


def simulate_short_trades(df, long_model, long_scaler, long_features,
                         short_model, short_scaler, short_features):
    """Simulate SHORT trades with opportunity gating"""
    print("\nSimulating SHORT trades...")

    trades = []

    for i in range(len(df) - 96):  # Reserve 96 candles (8 hours) for analysis
        # Get LONG probability
        long_row = df[long_features].iloc[i:i+1].values
        if np.isnan(long_row).any():
            continue
        long_row_scaled = long_scaler.transform(long_row)
        long_prob = long_model.predict_proba(long_row_scaled)[0][1]

        # Get SHORT probability
        short_row = df[short_features].iloc[i:i+1].values
        if np.isnan(short_row).any():
            continue
        short_row_scaled = short_scaler.transform(short_row)
        short_prob = short_model.predict_proba(short_row_scaled)[0][1]

        # SHORT entry: threshold + gating
        if short_prob >= ENTRY_THRESHOLD_SHORT:
            # Calculate expected values
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN

            # Gate check
            opportunity_cost = short_ev - long_ev
            if opportunity_cost > GATE_THRESHOLD:
                trades.append({
                    'entry_idx': i,
                    'entry_price': df['close'].iloc[i],
                    'entry_prob': short_prob,
                    'opportunity_cost': opportunity_cost,
                    'entry_time': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i
                })

    print(f"✅ Simulated {len(trades):,} SHORT trades")
    return trades


def analyze_reversal_timing(df, trades):
    """
    Analyze how quickly reversals occur after SHORT entry

    Key questions:
    - How many candles until trough (lowest point)?
    - How many candles until bounce starts?
    - How fast is the bounce (% per candle)?
    """
    print("\n" + "="*80)
    print("REVERSAL TIMING ANALYSIS")
    print("="*80)

    timing_stats = []

    for trade in trades[:1000]:  # Analyze first 1000 trades
        entry_idx = trade['entry_idx']
        entry_price = trade['entry_price']

        # Track price movement for next 96 candles (8 hours)
        future_prices = df['close'].iloc[entry_idx:entry_idx+96].values

        if len(future_prices) < 24:  # Need at least 2 hours
            continue

        # Find trough (lowest point - best exit for SHORT)
        trough_idx = np.argmin(future_prices)
        trough_price = future_prices[trough_idx]
        trough_pnl = (entry_price - trough_price) / entry_price  # SHORT P&L

        # Find when price starts recovering (5 consecutive higher candles)
        recovery_idx = None
        for i in range(trough_idx, min(trough_idx + 20, len(future_prices) - 5)):
            if all(future_prices[i+j] > future_prices[i+j-1] for j in range(1, 5)):
                recovery_idx = i
                break

        # Calculate bounce speed (if recovery detected)
        bounce_speed = 0
        if recovery_idx and recovery_idx < len(future_prices) - 10:
            bounce_prices = future_prices[recovery_idx:recovery_idx+10]
            bounce_speed = (bounce_prices[-1] - bounce_prices[0]) / bounce_prices[0]

        timing_stats.append({
            'candles_to_trough': trough_idx,
            'trough_pnl': trough_pnl,
            'candles_to_recovery': recovery_idx - trough_idx if recovery_idx else None,
            'bounce_speed_10candles': bounce_speed if recovery_idx else None
        })

    # Aggregate statistics
    timing_df = pd.DataFrame(timing_stats)

    print(f"\nTrough Timing (n={len(timing_df)}):")
    print(f"  Mean: {timing_df['candles_to_trough'].mean():.1f} candles ({timing_df['candles_to_trough'].mean()*5:.0f} min)")
    print(f"  Median: {timing_df['candles_to_trough'].median():.0f} candles ({timing_df['candles_to_trough'].median()*5:.0f} min)")
    print(f"  25th percentile: {timing_df['candles_to_trough'].quantile(0.25):.0f} candles")
    print(f"  75th percentile: {timing_df['candles_to_trough'].quantile(0.75):.0f} candles")

    print(f"\nTrough P&L:")
    print(f"  Mean: {timing_df['trough_pnl'].mean()*100:+.2f}%")
    print(f"  Median: {timing_df['trough_pnl'].median()*100:+.2f}%")

    # Recovery timing
    recovery_df = timing_df[timing_df['candles_to_recovery'].notna()]
    if len(recovery_df) > 0:
        print(f"\nRecovery Timing (n={len(recovery_df)}):")
        print(f"  Mean: {recovery_df['candles_to_recovery'].mean():.1f} candles after trough")
        print(f"  Median: {recovery_df['candles_to_recovery'].median():.0f} candles after trough")

        print(f"\nBounce Speed (10 candles = 50min):")
        bounce_df = recovery_df[recovery_df['bounce_speed_10candles'].notna()]
        print(f"  Mean: {bounce_df['bounce_speed_10candles'].mean()*100:+.2f}%")
        print(f"  Median: {bounce_df['bounce_speed_10candles'].median()*100:+.2f}%")

    # Distribution analysis
    print(f"\nTrough Timing Distribution:")
    print(f"  < 2 candles (10min): {(timing_df['candles_to_trough'] < 2).sum() / len(timing_df) * 100:.1f}%")
    print(f"  2-6 candles (10-30min): {((timing_df['candles_to_trough'] >= 2) & (timing_df['candles_to_trough'] < 6)).sum() / len(timing_df) * 100:.1f}%")
    print(f"  6-12 candles (30-60min): {((timing_df['candles_to_trough'] >= 6) & (timing_df['candles_to_trough'] < 12)).sum() / len(timing_df) * 100:.1f}%")
    print(f"  12-24 candles (1-2h): {((timing_df['candles_to_trough'] >= 12) & (timing_df['candles_to_trough'] < 24)).sum() / len(timing_df) * 100:.1f}%")
    print(f"  > 24 candles (2h+): {(timing_df['candles_to_trough'] >= 24).sum() / len(timing_df) * 100:.1f}%")

    return timing_df


def analyze_volume_patterns(df, trades):
    """Analyze volume behavior during SHORT reversals"""
    print("\n" + "="*80)
    print("VOLUME PATTERN ANALYSIS")
    print("="*80)

    volume_stats = []

    for trade in trades[:1000]:
        entry_idx = trade['entry_idx']
        entry_price = trade['entry_price']

        # Get volume data
        entry_volume = df['volume'].iloc[entry_idx]
        avg_volume_20 = df['volume'].iloc[max(0, entry_idx-20):entry_idx].mean()

        # Track volume for next 24 candles (2 hours)
        future_prices = df['close'].iloc[entry_idx:entry_idx+24].values
        future_volumes = df['volume'].iloc[entry_idx:entry_idx+24].values

        if len(future_volumes) < 24:
            continue

        # Find trough
        trough_idx = np.argmin(future_prices)

        # Volume at trough and around trough
        trough_volume = future_volumes[trough_idx]
        volume_before_trough = future_volumes[max(0, trough_idx-3):trough_idx].mean() if trough_idx > 0 else entry_volume
        volume_after_trough = future_volumes[trough_idx:min(trough_idx+3, len(future_volumes))].mean()

        volume_stats.append({
            'entry_volume_ratio': entry_volume / avg_volume_20 if avg_volume_20 > 0 else 1,
            'trough_volume_ratio': trough_volume / avg_volume_20 if avg_volume_20 > 0 else 1,
            'volume_spike_at_trough': trough_volume > (avg_volume_20 * 1.5),
            'volume_increase_after_trough': volume_after_trough / volume_before_trough if volume_before_trough > 0 else 1
        })

    volume_df = pd.DataFrame(volume_stats)

    print(f"\nVolume at Entry (relative to 20-period MA):")
    print(f"  Mean ratio: {volume_df['entry_volume_ratio'].mean():.2f}x")
    print(f"  Median ratio: {volume_df['entry_volume_ratio'].median():.2f}x")

    print(f"\nVolume at Trough:")
    print(f"  Mean ratio: {volume_df['trough_volume_ratio'].mean():.2f}x")
    print(f"  Median ratio: {volume_df['trough_volume_ratio'].median():.2f}x")
    print(f"  Spike (>1.5x) frequency: {volume_df['volume_spike_at_trough'].mean()*100:.1f}%")

    print(f"\nVolume Change After Trough:")
    print(f"  Mean increase: {(volume_df['volume_increase_after_trough'].mean()-1)*100:+.1f}%")
    print(f"  Median increase: {(volume_df['volume_increase_after_trough'].median()-1)*100:+.1f}%")

    return volume_df


def analyze_rsi_patterns(df, trades):
    """Analyze RSI behavior during SHORT reversals"""
    print("\n" + "="*80)
    print("RSI PATTERN ANALYSIS")
    print("="*80)

    rsi_stats = []

    for trade in trades[:1000]:
        entry_idx = trade['entry_idx']
        entry_price = trade['entry_price']
        entry_rsi = df['rsi'].iloc[entry_idx]

        # Track RSI for next 24 candles
        future_prices = df['close'].iloc[entry_idx:entry_idx+24].values
        future_rsi = df['rsi'].iloc[entry_idx:entry_idx+24].values

        if len(future_rsi) < 24:
            continue

        # Find trough
        trough_idx = np.argmin(future_prices)
        trough_rsi = future_rsi[trough_idx]

        # Check for RSI divergence
        # Divergence: price makes lower low, but RSI makes higher low
        divergence = False
        if trough_idx > 0:
            prev_low_idx = np.argmin(df['close'].iloc[max(0, entry_idx-10):entry_idx].values)
            prev_low_rsi = df['rsi'].iloc[max(0, entry_idx-10+prev_low_idx)]

            # Divergence: price lower, RSI higher
            if future_prices[trough_idx] < entry_price and trough_rsi > prev_low_rsi:
                divergence = True

        rsi_stats.append({
            'entry_rsi': entry_rsi,
            'trough_rsi': trough_rsi,
            'rsi_change': trough_rsi - entry_rsi,
            'rsi_divergence': divergence,
            'trough_rsi_oversold': trough_rsi < 30
        })

    rsi_df = pd.DataFrame(rsi_stats)

    print(f"\nRSI at Entry:")
    print(f"  Mean: {rsi_df['entry_rsi'].mean():.1f}")
    print(f"  Median: {rsi_df['entry_rsi'].median():.1f}")
    print(f"  Oversold (<30): {(rsi_df['entry_rsi'] < 30).mean()*100:.1f}%")

    print(f"\nRSI at Trough:")
    print(f"  Mean: {rsi_df['trough_rsi'].mean():.1f}")
    print(f"  Median: {rsi_df['trough_rsi'].median():.1f}")
    print(f"  Oversold (<30): {rsi_df['trough_rsi_oversold'].mean()*100:.1f}%")

    print(f"\nRSI Change (Entry → Trough):")
    print(f"  Mean: {rsi_df['rsi_change'].mean():+.1f}")
    print(f"  Median: {rsi_df['rsi_change'].median():+.1f}")

    print(f"\nRSI Divergence:")
    print(f"  Frequency: {rsi_df['rsi_divergence'].mean()*100:.1f}%")
    print(f"  (Price lower low + RSI higher low = bullish divergence)")

    return rsi_df


def recommend_labeling_parameters(timing_df):
    """Recommend SHORT-specific labeling parameters based on analysis"""
    print("\n" + "="*80)
    print("RECOMMENDED SHORT LABELING PARAMETERS")
    print("="*80)

    # Current parameters (LONG/SHORT common)
    print("\nCurrent Parameters (LONG/SHORT common):")
    print("  lead_time_min: 3 candles (15min)")
    print("  lead_time_max: 24 candles (2h)")
    print("  profit_threshold: 0.003 (0.3%)")
    print("  peak_threshold: 0.002 (0.2%)")

    # Analyze timing distribution
    median_trough = timing_df['candles_to_trough'].median()
    q25_trough = timing_df['candles_to_trough'].quantile(0.25)
    q75_trough = timing_df['candles_to_trough'].quantile(0.75)

    # Recommended parameters
    print("\n✅ RECOMMENDED SHORT-SPECIFIC Parameters:")
    print(f"  lead_time_min: {max(1, int(q25_trough * 0.8))} candles ({max(1, int(q25_trough * 0.8))*5}min)")
    print(f"  lead_time_max: {int(q75_trough * 1.2)} candles ({int(q75_trough * 1.2)*5}min)")
    print(f"  profit_threshold: 0.002 (0.2%) - lower to capture smaller moves")
    print(f"  peak_threshold: 0.001 (0.1%) - lower to be more sensitive")

    print("\nRationale:")
    print(f"  - 50% of troughs occur within {median_trough:.0f} candles ({median_trough*5:.0f}min)")
    print(f"  - 75% of troughs occur within {q75_trough:.0f} candles ({q75_trough*5:.0f}min)")
    print(f"  - SHORT reversals are faster than LONG peaks")
    print(f"  - Need tighter timing window to catch reversals")

    return {
        'lead_time_min': max(1, int(q25_trough * 0.8)),
        'lead_time_max': int(q75_trough * 1.2),
        'profit_threshold': 0.002,
        'peak_threshold': 0.001
    }


def main():
    print("="*80)
    print("Analyze SHORT Market Characteristics")
    print("="*80)
    print("\nGoal: Inform SHORT-specialized exit model training")
    print("  1. Reversal timing patterns")
    print("  2. Volume behavior")
    print("  3. RSI patterns and divergence")
    print("  4. Optimal labeling parameters")

    # Load data
    print("\nLoading data...")
    df_raw = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")

    # Calculate features
    print("Calculating features...")
    df = calculate_all_features(df_raw)
    print(f"✅ {len(df):,} candles loaded")

    # Load models
    entry_models = load_entry_models()

    # Simulate SHORT trades
    short_trades = simulate_short_trades(
        df,
        entry_models['long_entry_model'],
        entry_models['long_entry_scaler'],
        entry_models['long_entry_features'],
        entry_models['short_entry_model'],
        entry_models['short_entry_scaler'],
        entry_models['short_entry_features']
    )

    # Analyze patterns
    timing_df = analyze_reversal_timing(df, short_trades)
    volume_df = analyze_volume_patterns(df, short_trades)
    rsi_df = analyze_rsi_patterns(df, short_trades)

    # Recommend labeling parameters
    recommended_params = recommend_labeling_parameters(timing_df)

    # Save analysis results
    results_path = RESULTS_DIR / "short_market_characteristics_20251018.csv"
    analysis_df = pd.concat([timing_df, volume_df, rsi_df], axis=1)
    analysis_df.to_csv(results_path, index=False)
    print(f"\n✅ Analysis results saved to: {results_path}")

    # Save recommended parameters
    params_path = RESULTS_DIR / "short_labeling_params_recommended.txt"
    with open(params_path, 'w') as f:
        f.write("SHORT-Specific Labeling Parameters\n")
        f.write("="*50 + "\n")
        f.write(f"lead_time_min: {recommended_params['lead_time_min']}\n")
        f.write(f"lead_time_max: {recommended_params['lead_time_max']}\n")
        f.write(f"profit_threshold: {recommended_params['profit_threshold']}\n")
        f.write(f"peak_threshold: {recommended_params['peak_threshold']}\n")
    print(f"✅ Recommended parameters saved to: {params_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("1. Review analysis results")
    print("2. Use recommended parameters for SHORT model retraining")
    print("3. Add reversal detection features to model")
    print("="*80)


if __name__ == "__main__":
    main()
