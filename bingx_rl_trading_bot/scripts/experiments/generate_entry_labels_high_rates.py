"""
Generate Entry Labels with Higher Entry Rates (30%, 50%)

Purpose: Increase trade frequency from 0.37/day to 2-10/day target

Current: 15% entry rate → too few entries
Target: 30%, 50% entry rates → test which achieves 2-10/day

Methodology: Same Trade Outcome Triple Barrier
- 1:1 R/R ATR barriers
- P&L-weighted scoring
- Only percentile changes (15→30→50)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# CONFIGURATIONS TO TEST
# ====================================================================

ENTRY_RATES = [30, 50]  # Percentiles for entry rate control

# Barrier Configuration (Same as Trade Outcome)
ATR_STOP_MULTIPLIER = 1.0
ATR_PROFIT_MULTIPLIER = 1.0
TIME_LIMIT = 60  # 5 hours
SCORING_METHOD = 'pnl_weighted'

# ====================================================================
# HELPER FUNCTIONS (SAME AS TRADE OUTCOME)
# ====================================================================

def check_barriers(entry_price, future_prices, profit_barrier, stop_barrier, side='LONG'):
    """Check which barrier hits first"""
    for i, price in enumerate(future_prices):
        if side == 'LONG':
            if price >= profit_barrier:
                pnl_pct = (price - entry_price) / entry_price
                return 'profit', i, pnl_pct
            elif price <= stop_barrier:
                pnl_pct = (price - entry_price) / entry_price
                return 'stop', i, pnl_pct
        else:  # SHORT
            if price <= profit_barrier:
                pnl_pct = (entry_price - price) / entry_price
                return 'profit', i, pnl_pct
            elif price >= stop_barrier:
                pnl_pct = (entry_price - price) / entry_price
                return 'stop', i, pnl_pct

    # Timeout
    final_price = future_prices[-1] if len(future_prices) > 0 else entry_price
    if side == 'LONG':
        pnl_pct = (final_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - final_price) / entry_price

    return 'timeout', -1, pnl_pct


def calculate_quality_score_pnl_weighted(outcome, pnl_pct):
    """
    Calculate quality score based on P&L magnitude
    Higher score = better entry opportunity
    """
    if outcome == 'profit':
        # Larger profit = higher quality
        profit_magnitude = abs(pnl_pct) * 100
        score = 10.0 + min(profit_magnitude * 2, 10.0)
        return score
    elif outcome == 'timeout':
        return 5.0
    else:  # stop
        # Losses get low scores
        return 0.0


def generate_entry_labels_for_side(df, side, entry_percentile):
    """Generate entry labels for one side with specified entry rate"""
    print(f"\n{'='*80}")
    print(f"GENERATING ENTRY LABELS: {side} ({entry_percentile}% ENTRY RATE)")
    print(f"{'='*80}\n")

    quality_scores = []
    outcomes = []
    pnl_list = []

    valid_range = len(df) - TIME_LIMIT

    print(f"📊 Processing {valid_range:,} candles...")

    for i in range(valid_range):
        current_price = df.iloc[i]['close']
        current_atr = df.iloc[i].get('atr', current_price * 0.01)

        # Calculate barriers
        if side == 'LONG':
            stop_barrier = current_price - (current_atr * ATR_STOP_MULTIPLIER)
            profit_barrier = current_price + (current_atr * ATR_PROFIT_MULTIPLIER)
        else:
            stop_barrier = current_price + (current_atr * ATR_STOP_MULTIPLIER)
            profit_barrier = current_price - (current_atr * ATR_PROFIT_MULTIPLIER)

        # Get future prices
        future_prices = df.iloc[i+1:i+1+TIME_LIMIT]['close'].values

        # Check barriers
        outcome, hit_index, pnl_pct = check_barriers(
            current_price, future_prices,
            profit_barrier, stop_barrier, side
        )

        # Calculate quality score
        quality_score = calculate_quality_score_pnl_weighted(outcome, pnl_pct)

        quality_scores.append(quality_score)
        outcomes.append(outcome)
        pnl_list.append(pnl_pct)

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1:,}/{valid_range:,} candles ({(i+1)/valid_range*100:.1f}%)")

    # Pad remaining
    quality_scores.extend([0.0] * TIME_LIMIT)

    # Apply percentile-based filtering
    # Select top ENTRY_PERCENTILE% of opportunities (highest quality scores)
    quality_threshold = np.percentile(quality_scores[:valid_range], 100 - entry_percentile)

    # Generate binary labels
    entry_labels = [1 if score >= quality_threshold else 0 for score in quality_scores]

    # Statistics
    entry_rate = (sum(entry_labels[:valid_range]) / valid_range) * 100
    profit_hit_pct = (outcomes.count('profit') / len(outcomes)) * 100
    stop_hit_pct = (outcomes.count('stop') / len(outcomes)) * 100
    timeout_pct = (outcomes.count('timeout') / len(outcomes)) * 100
    avg_pnl = np.mean(pnl_list) * 100

    statistics = {
        'entry_percentile': entry_percentile,
        'entry_rate': entry_rate,
        'profit_hit_pct': profit_hit_pct,
        'stop_hit_pct': stop_hit_pct,
        'timeout_pct': timeout_pct,
        'avg_pnl': avg_pnl,
        'quality_threshold': quality_threshold
    }

    print(f"\n✅ {side} Label Generation Complete!")
    print(f"\n📊 ENTRY LABELS:")
    print(f"   Entry Rate:   {entry_rate:6.2f}% (target: {entry_percentile}%)")
    print(f"   Entry Labels: {sum(entry_labels):,}/{len(entry_labels):,}")
    print(f"   Deviation:    {abs(entry_rate - entry_percentile):.2f}%")

    return entry_labels, quality_scores, outcomes, statistics


# ====================================================================
# MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":
    print("="*80)
    print("GENERATE ENTRY LABELS - HIGH ENTRY RATES (30%, 50%)")
    print("="*80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load features
    print(f"\n{'='*80}")
    print("LOADING FEATURES")
    print(f"{'='*80}\n")

    features_file = "data/features/BTCUSDT_5m_features_90days_complete_20251106_164542.csv"
    print(f"📂 File: {features_file}")

    df = pd.read_csv(features_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"   Rows: {len(df):,}")
    print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Generate labels for each entry rate
    for entry_rate in ENTRY_RATES:
        print(f"\n{'='*80}")
        print(f"PROCESSING ENTRY RATE: {entry_rate}%")
        print(f"{'='*80}")

        # Generate for both sides
        long_labels, long_scores, long_outcomes, long_stats = generate_entry_labels_for_side(
            df.copy(), 'LONG', entry_rate
        )
        short_labels, short_scores, short_outcomes, short_stats = generate_entry_labels_for_side(
            df.copy(), 'SHORT', entry_rate
        )

        # Create output dataframe
        df_output = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_output['entry_label_long'] = long_labels
        df_output['entry_label_short'] = short_labels
        df_output['quality_score_long'] = long_scores
        df_output['quality_score_short'] = short_scores

        # Save
        output_file = f"data/labels/entry_labels_rate{entry_rate}pct_{timestamp}.csv"
        df_output.to_csv(output_file, index=False)
        print(f"\n💾 Saved: {output_file}")

        # Validation
        print(f"\n📊 VALIDATION (Entry Rate {entry_rate}%):")
        print(f"   LONG: {long_stats['entry_rate']:.2f}% (deviation: {abs(long_stats['entry_rate'] - entry_rate):.2f}%)")
        print(f"   SHORT: {short_stats['entry_rate']:.2f}% (deviation: {abs(short_stats['entry_rate'] - entry_rate):.2f}%)")

        if abs(long_stats['entry_rate'] - entry_rate) < 0.1 and \
           abs(short_stats['entry_rate'] - entry_rate) < 0.1:
            print(f"   Status: ✅ PASS")
        else:
            print(f"   Status: ⚠️ CHECK")

    # Summary
    print(f"\n{'='*80}")
    print("LABEL GENERATION COMPLETE")
    print(f"{'='*80}\n")

    print(f"✅ Generated labels for {len(ENTRY_RATES)} entry rates:")
    for rate in ENTRY_RATES:
        print(f"   - {rate}% entry rate: entry_labels_rate{rate}pct_{timestamp}.csv")

    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Review generated labels")
    print(f"   2. Retrain Entry models with 171 features")
    print(f"   3. Backtest all configurations")
    print(f"   4. Select optimal entry rate for 2-10 trades/day")
