"""
Generate Exit Labels with Higher Exit Rates (30%, 50%, 70%)

Purpose: Increase trade frequency from 0.37/day to 2-10/day target

Current: 15% exit rate → 0.37 trades/day (too low)
Target: 30%, 50%, 70% exit rates → test which achieves 2-10/day

Methodology: Same Optimal Triple Barrier
- 1:1 R/R ATR barriers
- P&L-weighted scoring
- Only percentile changes (15→30→50→70)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# CONFIGURATIONS TO TEST
# ====================================================================

EXIT_RATES = [30, 50, 70]  # Percentiles for exit rate control

# Barrier Configuration (Same as Optimal)
ATR_STOP_MULTIPLIER = 1.0
ATR_PROFIT_MULTIPLIER = 1.0
TIME_LIMIT = 60  # 5 hours
SCORING_METHOD = 'pnl_weighted'

# ====================================================================
# HELPER FUNCTIONS (SAME AS OPTIMAL)
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


def calculate_risk_score_pnl_weighted(outcome, pnl_pct):
    """Calculate risk score based on P&L magnitude"""
    if outcome == 'profit':
        return 0.0
    elif outcome == 'timeout':
        return 5.0
    else:  # stop
        loss_magnitude = abs(pnl_pct) * 100
        score = 5.0 + min(loss_magnitude * 2, 10.0)
        return score


def generate_exit_labels_for_side(df, side, exit_percentile):
    """Generate exit labels for one side with specified exit rate"""
    print(f"\n{'='*80}")
    print(f"GENERATING EXIT LABELS: {side} ({exit_percentile}% EXIT RATE)")
    print(f"{'='*80}\n")

    risk_scores = []
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

        # Calculate risk score
        risk_score = calculate_risk_score_pnl_weighted(outcome, pnl_pct)

        risk_scores.append(risk_score)
        outcomes.append(outcome)
        pnl_list.append(pnl_pct)

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1:,}/{valid_range:,} candles ({(i+1)/valid_range*100:.1f}%)")

    # Pad remaining
    risk_scores.extend([0.0] * TIME_LIMIT)

    # Apply percentile-based filtering
    risk_threshold = np.percentile(risk_scores[:valid_range], 100 - exit_percentile)

    # Generate binary labels
    exit_labels = [1 if score >= risk_threshold else 0 for score in risk_scores]

    # Statistics
    exit_rate = (sum(exit_labels[:valid_range]) / valid_range) * 100
    profit_hit_pct = (outcomes.count('profit') / len(outcomes)) * 100
    stop_hit_pct = (outcomes.count('stop') / len(outcomes)) * 100
    timeout_pct = (outcomes.count('timeout') / len(outcomes)) * 100
    avg_pnl = np.mean(pnl_list) * 100

    statistics = {
        'exit_percentile': exit_percentile,
        'exit_rate': exit_rate,
        'profit_hit_pct': profit_hit_pct,
        'stop_hit_pct': stop_hit_pct,
        'timeout_pct': timeout_pct,
        'avg_pnl': avg_pnl,
        'risk_threshold': risk_threshold
    }

    print(f"\n✅ {side} Label Generation Complete!")
    print(f"\n📊 EXIT LABELS:")
    print(f"   Exit Rate:   {exit_rate:6.2f}% (target: {exit_percentile}%)")
    print(f"   Exit Labels: {sum(exit_labels):,}/{len(exit_labels):,}")
    print(f"   Deviation:   {abs(exit_rate - exit_percentile):.2f}%")

    return exit_labels, risk_scores, outcomes, statistics


# ====================================================================
# MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":
    print("="*80)
    print("GENERATE EXIT LABELS - HIGH EXIT RATES (30%, 50%, 70%)")
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

    # Generate labels for each exit rate
    for exit_rate in EXIT_RATES:
        print(f"\n{'='*80}")
        print(f"PROCESSING EXIT RATE: {exit_rate}%")
        print(f"{'='*80}")

        # Generate for both sides
        long_labels, long_scores, long_outcomes, long_stats = generate_exit_labels_for_side(
            df.copy(), 'LONG', exit_rate
        )
        short_labels, short_scores, short_outcomes, short_stats = generate_exit_labels_for_side(
            df.copy(), 'SHORT', exit_rate
        )

        # Create output dataframe
        df_output = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_output['exit_label_long'] = long_labels
        df_output['exit_label_short'] = short_labels
        df_output['risk_score_long'] = long_scores
        df_output['risk_score_short'] = short_scores

        # Save
        output_file = f"data/labels/exit_labels_rate{exit_rate}pct_{timestamp}.csv"
        df_output.to_csv(output_file, index=False)
        print(f"\n💾 Saved: {output_file}")

        # Validation
        print(f"\n📊 VALIDATION (Exit Rate {exit_rate}%):")
        print(f"   LONG: {long_stats['exit_rate']:.2f}% (deviation: {abs(long_stats['exit_rate'] - exit_rate):.2f}%)")
        print(f"   SHORT: {short_stats['exit_rate']:.2f}% (deviation: {abs(short_stats['exit_rate'] - exit_rate):.2f}%)")

        if abs(long_stats['exit_rate'] - exit_rate) < 0.1 and \
           abs(short_stats['exit_rate'] - exit_rate) < 0.1:
            print(f"   Status: ✅ PASS")
        else:
            print(f"   Status: ⚠️ CHECK")

    # Summary
    print(f"\n{'='*80}")
    print("LABEL GENERATION COMPLETE")
    print(f"{'='*80}\n")

    print(f"✅ Generated labels for {len(EXIT_RATES)} exit rates:")
    for rate in EXIT_RATES:
        print(f"   - {rate}% exit rate: exit_labels_rate{rate}pct_{timestamp}.csv")

    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Review generated labels")
    print(f"   2. Retrain Exit models with 171 features")
    print(f"   3. Backtest all configurations")
    print(f"   4. Select optimal exit rate for 2-10 trades/day")
