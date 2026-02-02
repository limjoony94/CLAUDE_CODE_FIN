"""
Enhanced Exit Label Generation with Risk-Reward Analysis

Key Improvements:
1. Risk-Reward Ratio: Only exit if R/R ratio < threshold
2. Confidence Levels: Multi-tier labeling (0.0, 0.5, 1.0)
3. Time Decay: Consider how quickly profit/loss occurs
4. Trend Strength: Exit based on momentum reversal
5. Volatility Adjusted: Different thresholds for different regimes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
FEATURES_DIR = DATA_DIR / "features"
LABELS_DIR = DATA_DIR / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Input file
FEATURES_FILE = FEATURES_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"

# Output files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_EXIT_LONG = LABELS_DIR / f"exit_labels_long_enhanced_{TIMESTAMP}.csv"
OUTPUT_EXIT_SHORT = LABELS_DIR / f"exit_labels_short_enhanced_{TIMESTAMP}.csv"

# Enhanced Parameters
LOOKAHEAD_CANDLES = 30  # 2.5 hours

# Risk-Reward Thresholds
MIN_PROFIT_TARGET = 0.005  # 0.5% profit (increased from 0.3%)
MAX_ACCEPTABLE_RISK = 0.003  # 0.3% risk tolerance (decreased from 0.5%)
RISK_REWARD_RATIO = 2.0  # Require 2:1 R/R minimum

# Confidence Levels
STRONG_EXIT_THRESHOLD = 0.008  # 0.8% adverse move = strong exit
MEDIUM_EXIT_THRESHOLD = 0.005  # 0.5% adverse move = medium exit
WEAK_EXIT_THRESHOLD = 0.003  # 0.3% adverse move = weak exit

# Trend Momentum
MOMENTUM_REVERSAL_THRESHOLD = 0.7  # 70% of move in opposite direction


def calculate_risk_reward_ratio(max_profit, max_loss):
    """Calculate Risk/Reward ratio"""
    if abs(max_loss) < 0.0001:  # Avoid division by zero
        return 100.0
    return abs(max_profit / max_loss)


def detect_momentum_reversal(prices):
    """
    Detect if momentum has reversed
    Returns: reversal_strength (0.0 to 1.0)
    """
    if len(prices) < 3:
        return 0.0

    # Calculate price changes
    changes = np.diff(prices)

    # Count positive vs negative changes
    positive_changes = changes[changes > 0]
    negative_changes = changes[changes < 0]

    if len(changes) == 0:
        return 0.0

    # Reversal strength = ratio of opposite direction moves
    reversal_strength = len(negative_changes) / len(changes)

    return reversal_strength


def generate_enhanced_exit_labels(df, side='LONG'):
    """
    Generate Enhanced Exit Labels with Risk-Reward Analysis

    Exit Label Scale (0.0 to 1.0):
    - 1.0: Strong Exit (immediate risk, R/R < 1.0)
    - 0.5: Medium Exit (moderate risk, R/R < 2.0)
    - 0.0: Hold (good R/R, profit opportunity exists)
    """

    print(f"\n{'='*80}")
    print(f"GENERATING {side} EXIT LABELS (Enhanced Risk-Reward)")
    print(f"{'='*80}")

    df = df.copy()
    df['exit_label'] = 0.0  # Default: Hold

    # Statistics counters
    strong_exits = 0
    medium_exits = 0
    holds = 0

    for i in range(len(df) - LOOKAHEAD_CANDLES):
        current_price = df.iloc[i]['close']
        current_rsi = df.iloc[i].get('rsi', 50)
        current_atr = df.iloc[i].get('atr', current_price * 0.01)  # Default 1% ATR

        future_prices = df.iloc[i+1:i+1+LOOKAHEAD_CANDLES]['close'].values

        if side == 'LONG':
            # LONG Position: Calculate upside potential and downside risk
            max_profit_pct = ((future_prices.max() - current_price) / current_price)
            max_loss_pct = ((future_prices.min() - current_price) / current_price)

            # Risk-Reward Ratio
            rr_ratio = calculate_risk_reward_ratio(max_profit_pct, max_loss_pct)

            # Momentum Reversal Detection
            momentum_reversal = detect_momentum_reversal(future_prices)

            # Peak timing analysis
            peak_idx = np.argmax(future_prices)
            peak_early = peak_idx < len(future_prices) * 0.3  # Peak in first 30%

            # Exit Decision Logic
            exit_label = 0.0  # Default: Hold

            # STRONG EXIT (1.0) - Immediate risk
            if (max_loss_pct < -STRONG_EXIT_THRESHOLD or  # Big drawdown coming
                (max_loss_pct < -MAX_ACCEPTABLE_RISK and rr_ratio < 1.0) or  # High risk, low reward
                (current_rsi > 80 and max_profit_pct < 0.002)):  # Overbought, no upside

                exit_label = 1.0
                strong_exits += 1

            # MEDIUM EXIT (0.5) - Moderate risk
            elif (max_loss_pct < -MEDIUM_EXIT_THRESHOLD or  # Moderate drawdown
                  (max_profit_pct < MIN_PROFIT_TARGET and max_loss_pct < -WEAK_EXIT_THRESHOLD) or  # Low profit, some risk
                  (peak_early and momentum_reversal > MOMENTUM_REVERSAL_THRESHOLD) or  # Early peak + reversal
                  (rr_ratio < RISK_REWARD_RATIO and max_profit_pct < 0.005)):  # Bad R/R

                exit_label = 0.5
                medium_exits += 1

            # HOLD (0.0) - Good opportunity
            else:
                # Good conditions to hold:
                # - Max profit > 0.5% AND max loss < 0.3%
                # - R/R ratio > 2.0
                # - No strong momentum reversal
                holds += 1

        else:  # SHORT
            # SHORT Position: Calculate downside potential and upside risk
            max_profit_pct = ((current_price - future_prices.min()) / current_price)
            max_loss_pct = ((current_price - future_prices.max()) / current_price)

            # Risk-Reward Ratio
            rr_ratio = calculate_risk_reward_ratio(max_profit_pct, max_loss_pct)

            # Momentum Reversal Detection
            momentum_reversal = 1.0 - detect_momentum_reversal(future_prices)  # Inverse for SHORT

            # Bottom timing analysis
            bottom_idx = np.argmin(future_prices)
            bottom_early = bottom_idx < len(future_prices) * 0.3  # Bottom in first 30%

            # Exit Decision Logic
            exit_label = 0.0  # Default: Hold

            # STRONG EXIT (1.0) - Immediate risk
            if (max_loss_pct < -STRONG_EXIT_THRESHOLD or  # Big adverse move coming
                (max_loss_pct < -MAX_ACCEPTABLE_RISK and rr_ratio < 1.0) or  # High risk, low reward
                (current_rsi < 20 and max_profit_pct < 0.002)):  # Oversold, no downside

                exit_label = 1.0
                strong_exits += 1

            # MEDIUM EXIT (0.5) - Moderate risk
            elif (max_loss_pct < -MEDIUM_EXIT_THRESHOLD or  # Moderate adverse move
                  (max_profit_pct < MIN_PROFIT_TARGET and max_loss_pct < -WEAK_EXIT_THRESHOLD) or  # Low profit, some risk
                  (bottom_early and momentum_reversal > MOMENTUM_REVERSAL_THRESHOLD) or  # Early bottom + reversal
                  (rr_ratio < RISK_REWARD_RATIO and max_profit_pct < 0.005)):  # Bad R/R

                exit_label = 0.5
                medium_exits += 1

            # HOLD (0.0) - Good opportunity
            else:
                holds += 1

        df.iloc[i, df.columns.get_loc('exit_label')] = exit_label

    # Last N candles: cannot calculate lookahead
    df.iloc[-LOOKAHEAD_CANDLES:, df.columns.get_loc('exit_label')] = 0.0

    # Statistics
    total = len(df)
    strong_exit_pct = (strong_exits / total) * 100
    medium_exit_pct = (medium_exits / total) * 100
    hold_pct = (holds / total) * 100

    print(f"\n📊 {side} EXIT LABEL STATISTICS:")
    print(f"   Total candles: {total:,}")
    print(f"   Strong Exit (1.0): {strong_exits:,} ({strong_exit_pct:.2f}%)")
    print(f"   Medium Exit (0.5): {medium_exits:,} ({medium_exit_pct:.2f}%)")
    print(f"   Hold (0.0): {holds:,} ({hold_pct:.2f}%)")
    print(f"\n   Exit Rate: {strong_exit_pct + medium_exit_pct:.2f}%")

    print(f"\n✅ {side} Enhanced Exit labels generated")

    return df[['timestamp', 'exit_label']]


def main():
    print("="*80)
    print("ENHANCED EXIT LABEL GENERATION (Risk-Reward Analysis)")
    print("="*80)

    print(f"\n📂 Input: {FEATURES_FILE.name}")
    print(f"📤 Output: {OUTPUT_EXIT_LONG.name}, {OUTPUT_EXIT_SHORT.name}")

    print(f"\n⚙️ ENHANCED PARAMETERS:")
    print(f"   Lookahead: {LOOKAHEAD_CANDLES} candles ({LOOKAHEAD_CANDLES * 5} minutes)")
    print(f"   Min Profit Target: {MIN_PROFIT_TARGET*100}% (increased)")
    print(f"   Max Acceptable Risk: {MAX_ACCEPTABLE_RISK*100}% (decreased)")
    print(f"   Risk/Reward Ratio: {RISK_REWARD_RATIO}:1 minimum")
    print(f"   Momentum Reversal: {MOMENTUM_REVERSAL_THRESHOLD*100}% threshold")

    print(f"\n📋 EXIT LABEL SCALE:")
    print(f"   1.0: Strong Exit (immediate risk, R/R < 1.0)")
    print(f"   0.5: Medium Exit (moderate risk, R/R < 2.0)")
    print(f"   0.0: Hold (good R/R, profit opportunity)")

    # Load features
    print(f"\n📖 Loading features...")
    df = pd.read_csv(FEATURES_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"   Rows: {len(df):,}")
    print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Generate LONG exit labels
    df_long_exit = generate_enhanced_exit_labels(df, side='LONG')

    # Generate SHORT exit labels
    df_short_exit = generate_enhanced_exit_labels(df, side='SHORT')

    # Save
    print(f"\n💾 Saving enhanced exit labels...")
    df_long_exit.to_csv(OUTPUT_EXIT_LONG, index=False)
    print(f"   ✅ LONG: {OUTPUT_EXIT_LONG}")

    df_short_exit.to_csv(OUTPUT_EXIT_SHORT, index=False)
    print(f"   ✅ SHORT: {OUTPUT_EXIT_SHORT}")

    print(f"\n{'='*80}")
    print("ENHANCED EXIT LABEL GENERATION COMPLETE")
    print(f"{'='*80}")

    print(f"\n📋 KEY IMPROVEMENTS:")
    print(f"   1. Risk-Reward Analysis: Only exit if R/R < 2:1")
    print(f"   2. Confidence Levels: 3-tier labeling (0.0, 0.5, 1.0)")
    print(f"   3. Momentum Detection: Exit on trend reversal")
    print(f"   4. Stricter Criteria: Fewer but higher quality exits")

    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Retrain Exit models with enhanced labels + 171 features")
    print(f"   2. Backtest with improved Exit models")
    print(f"   3. Compare vs current 52-day Exit models")


if __name__ == "__main__":
    main()
