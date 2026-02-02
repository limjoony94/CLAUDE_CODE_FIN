"""
Generate Exit Labels with Optimal Triple Barrier Configuration

Optimal Configuration (from comprehensive comparison):
- Barrier: 1:1 R/R (1.0 ATR stop, 1.0 ATR profit)
- Scoring: pnl_weighted
- Percentile: 15th (worst 15% labeled as exits)
- Time Limit: 60 candles (5 hours)

Expected Results:
- LONG Exit Rate: 15.00% (exact)
- LONG Outcomes: Profit 48.9%, Stop 50.8%, Timeout 0.3%
- SHORT Exit Rate: 15.00% (exact)
- SHORT Outcomes: Profit 50.8%, Stop 48.9%, Timeout 0.3%

Reference: triple_barrier_comparison_20251106_221211.csv
Config #8 (LONG) and Config #53 (SHORT)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# OPTIMAL CONFIGURATION (Data-Driven Selection)
# ====================================================================

# Barrier Configuration
ATR_STOP_MULTIPLIER = 1.0      # 1 ATR (~1% for BTC)
ATR_PROFIT_MULTIPLIER = 1.0    # 1 ATR (~1% for BTC)
RISK_REWARD_RATIO = "1:1"      # Equal risk/reward

# Scoring Method
SCORING_METHOD = 'pnl_weighted'  # Continuous scoring based on P&L magnitude

# Label Sparsity Control
EXIT_PERCENTILE = 15            # Worst 15% labeled as exits
TIME_LIMIT = 60                 # 60 candles = 5 hours (5 min × 60)

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def check_barriers(entry_price, future_prices, profit_barrier, stop_barrier, side='LONG'):
    """
    Simulate forward to find which barrier hits first

    Args:
        entry_price: Entry price
        future_prices: Array of future prices
        profit_barrier: Profit target price
        stop_barrier: Stop loss price
        side: 'LONG' or 'SHORT'

    Returns:
        outcome: 'profit', 'stop', or 'timeout'
        hit_index: Index where barrier was hit (or -1 if timeout)
        pnl_pct: P&L percentage at outcome
    """
    for i, price in enumerate(future_prices):
        if side == 'LONG':
            # LONG: profit if price goes up, stop if price goes down
            if price >= profit_barrier:
                pnl_pct = (price - entry_price) / entry_price
                return 'profit', i, pnl_pct
            elif price <= stop_barrier:
                pnl_pct = (price - entry_price) / entry_price
                return 'stop', i, pnl_pct
        else:  # SHORT
            # SHORT: profit if price goes down, stop if price goes up
            if price <= profit_barrier:
                pnl_pct = (entry_price - price) / entry_price
                return 'profit', i, pnl_pct
            elif price >= stop_barrier:
                pnl_pct = (entry_price - price) / entry_price
                return 'stop', i, pnl_pct

    # Timeout: neither barrier hit within time limit
    final_price = future_prices[-1] if len(future_prices) > 0 else entry_price
    if side == 'LONG':
        pnl_pct = (final_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - final_price) / entry_price

    return 'timeout', -1, pnl_pct


def calculate_risk_score_pnl_weighted(outcome, pnl_pct):
    """
    Calculate risk score based on P&L magnitude

    Scoring Logic:
    - Profit outcomes: 0.0 (no risk)
    - Timeout: 5.0 (neutral risk)
    - Stop losses: 5.0 + (|loss%| × 100 × 2), capped at 15.0

    This creates continuous distribution instead of discrete plateaus

    Args:
        outcome: 'profit', 'stop', or 'timeout'
        pnl_pct: P&L percentage (e.g., -0.01 = -1% loss)

    Returns:
        risk_score: Float between 0 and 15
    """
    if outcome == 'profit':
        return 0.0
    elif outcome == 'timeout':
        return 5.0
    else:  # stop
        # Larger loss = higher risk
        # Scale: 5.0 baseline + up to 10.0 based on loss magnitude
        loss_magnitude = abs(pnl_pct) * 100  # Convert to percentage
        score = 5.0 + min(loss_magnitude * 2, 10.0)
        return score


def generate_exit_labels_for_side(df, side):
    """
    Generate exit labels for one side (LONG or SHORT)

    Process:
    1. For each candle, calculate dynamic ATR-based barriers
    2. Simulate forward TIME_LIMIT candles to find outcome
    3. Calculate continuous risk score based on P&L
    4. Apply percentile-based filtering (worst 15%)
    5. Label worst outcomes as exit=1

    Args:
        df: DataFrame with features and prices
        side: 'LONG' or 'SHORT'

    Returns:
        exit_labels: Array of 0/1 labels
        risk_scores: Array of continuous risk scores
        outcomes: List of barrier outcomes
        statistics: Dictionary with analysis stats
    """
    print(f"\n{'='*80}")
    print(f"GENERATING EXIT LABELS: {side}")
    print(f"{'='*80}\n")

    risk_scores = []
    outcomes = []
    pnl_list = []
    hit_indices = []

    # Cannot label last TIME_LIMIT candles (no future data)
    valid_range = len(df) - TIME_LIMIT

    print(f"📊 Processing {valid_range:,} candles...")

    for i in range(valid_range):
        # Current candle data
        current_price = df.iloc[i]['close']
        current_atr = df.iloc[i].get('atr', current_price * 0.01)  # Fallback to 1%

        # Calculate dynamic barriers based on ATR
        if side == 'LONG':
            stop_barrier = current_price - (current_atr * ATR_STOP_MULTIPLIER)
            profit_barrier = current_price + (current_atr * ATR_PROFIT_MULTIPLIER)
        else:  # SHORT
            stop_barrier = current_price + (current_atr * ATR_STOP_MULTIPLIER)
            profit_barrier = current_price - (current_atr * ATR_PROFIT_MULTIPLIER)

        # Get future prices
        future_prices = df.iloc[i+1:i+1+TIME_LIMIT]['close'].values

        # Check which barrier hits first
        outcome, hit_index, pnl_pct = check_barriers(
            current_price, future_prices,
            profit_barrier, stop_barrier, side
        )

        # Calculate continuous risk score
        risk_score = calculate_risk_score_pnl_weighted(outcome, pnl_pct)

        # Store results
        risk_scores.append(risk_score)
        outcomes.append(outcome)
        pnl_list.append(pnl_pct)
        hit_indices.append(hit_index)

        # Progress indicator
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1:,}/{valid_range:,} candles ({(i+1)/valid_range*100:.1f}%)")

    # Pad remaining candles with 0 (cannot label without future data)
    risk_scores.extend([0.0] * TIME_LIMIT)

    # Apply percentile-based filtering
    # Select worst EXIT_PERCENTILE% of outcomes
    risk_threshold = np.percentile(risk_scores[:valid_range], 100 - EXIT_PERCENTILE)

    # Generate binary labels
    exit_labels = []
    for score in risk_scores:
        if score >= risk_threshold:
            exit_labels.append(1)  # Exit
        else:
            exit_labels.append(0)  # Hold

    # Calculate statistics
    exit_rate = (sum(exit_labels[:valid_range]) / valid_range) * 100
    profit_hit_pct = (outcomes.count('profit') / len(outcomes)) * 100
    stop_hit_pct = (outcomes.count('stop') / len(outcomes)) * 100
    timeout_pct = (outcomes.count('timeout') / len(outcomes)) * 100
    avg_pnl = np.mean(pnl_list) * 100  # Convert to percentage

    statistics = {
        'valid_candles': valid_range,
        'exit_rate': exit_rate,
        'profit_hit_pct': profit_hit_pct,
        'stop_hit_pct': stop_hit_pct,
        'timeout_pct': timeout_pct,
        'avg_pnl': avg_pnl,
        'risk_threshold': risk_threshold,
        'risk_scores_q25': np.percentile(risk_scores[:valid_range], 25),
        'risk_scores_median': np.percentile(risk_scores[:valid_range], 50),
        'risk_scores_q75': np.percentile(risk_scores[:valid_range], 75),
    }

    # Print results
    print(f"\n✅ {side} Label Generation Complete!")
    print(f"\n📊 BARRIER OUTCOMES:")
    print(f"   Profit Hit:  {profit_hit_pct:6.2f}%")
    print(f"   Stop Hit:    {stop_hit_pct:6.2f}%")
    print(f"   Timeout:     {timeout_pct:6.2f}%")

    print(f"\n📊 RISK SCORE DISTRIBUTION:")
    print(f"   25th percentile: {statistics['risk_scores_q25']:.2f}")
    print(f"   50th percentile: {statistics['risk_scores_median']:.2f}")
    print(f"   75th percentile: {statistics['risk_scores_q75']:.2f}")
    print(f"   Threshold ({100-EXIT_PERCENTILE}th): {risk_threshold:.2f}")

    print(f"\n📊 EXIT LABELS:")
    print(f"   Exit Rate:   {exit_rate:6.2f}% (target: {EXIT_PERCENTILE}%)")
    print(f"   Exit Labels: {sum(exit_labels):,}/{len(exit_labels):,}")
    print(f"   Avg P&L:     {avg_pnl:+.2f}%")

    return exit_labels, risk_scores, outcomes, statistics


# ====================================================================
# MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":
    print("="*80)
    print("GENERATE EXIT LABELS - OPTIMAL TRIPLE BARRIER CONFIGURATION")
    print("="*80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n⚙️ CONFIGURATION:")
    print(f"   Barrier: {RISK_REWARD_RATIO} R/R")
    print(f"   Stop:    {ATR_STOP_MULTIPLIER} × ATR")
    print(f"   Profit:  {ATR_PROFIT_MULTIPLIER} × ATR")
    print(f"   Scoring: {SCORING_METHOD}")
    print(f"   Target:  {EXIT_PERCENTILE}% exit rate")
    print(f"   Horizon: {TIME_LIMIT} candles ({TIME_LIMIT * 5} min)")

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
    print(f"   Features: {len(df.columns)}")

    # Generate labels for both sides
    long_labels, long_scores, long_outcomes, long_stats = generate_exit_labels_for_side(df, 'LONG')
    short_labels, short_scores, short_outcomes, short_stats = generate_exit_labels_for_side(df, 'SHORT')

    # Add labels to dataframe
    df['exit_label_long'] = long_labels
    df['exit_label_short'] = short_labels
    df['risk_score_long'] = long_scores
    df['risk_score_short'] = short_scores

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")

    output_file = f"data/labels/exit_labels_optimal_triple_barrier_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved: {output_file}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    print(f"📊 LONG EXIT LABELS:")
    print(f"   Exit Rate: {long_stats['exit_rate']:.2f}% (target: {EXIT_PERCENTILE}%)")
    print(f"   Deviation: {abs(long_stats['exit_rate'] - EXIT_PERCENTILE):.2f}%")

    print(f"\n📊 SHORT EXIT LABELS:")
    print(f"   Exit Rate: {short_stats['exit_rate']:.2f}% (target: {EXIT_PERCENTILE}%)")
    print(f"   Deviation: {abs(short_stats['exit_rate'] - EXIT_PERCENTILE):.2f}%")

    print(f"\n📊 VALIDATION:")
    if abs(long_stats['exit_rate'] - EXIT_PERCENTILE) < 0.1 and \
       abs(short_stats['exit_rate'] - EXIT_PERCENTILE) < 0.1:
        print(f"   Status: ✅ PASS - Both within 0.1% of target")
    else:
        print(f"   Status: ⚠️ CHECK - Review deviation from target")

    print(f"\n{'='*80}")
    print("EXIT LABEL GENERATION COMPLETE")
    print(f"{'='*80}\n")

    print(f"📋 NEXT STEPS:")
    print(f"   1. Review {output_file}")
    print(f"   2. Verify exit rates match target (15%)")
    print(f"   3. Proceed to Exit model retraining (171 features)")
    print(f"   4. Backtest improved models")
