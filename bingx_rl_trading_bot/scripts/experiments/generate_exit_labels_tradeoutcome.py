"""
Generate Trade Outcome Exit Labels for 90-Day Dataset

Exit Logic (More sophisticated than entry):
- LONG Exit = 1 if: Should close position (profit take or risk avoid)
- SHORT Exit = 1 if: Should close position (profit take or risk avoid)

Trade Outcome Exit Criteria:
- Look ahead N candles (default: 30 = 2.5 hours)
- Compare max profit potential vs risk exposure
- Label = 1 if profit opportunity < risk OR good profit reached
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

# Input file (90-day features)
FEATURES_FILE = FEATURES_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"

# Output files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_EXIT_LONG = LABELS_DIR / f"exit_labels_long_tradeoutcome_90days_{TIMESTAMP}.csv"
OUTPUT_EXIT_SHORT = LABELS_DIR / f"exit_labels_short_tradeoutcome_90days_{TIMESTAMP}.csv"

# Exit Label Parameters
LOOKAHEAD_CANDLES = 30  # 2.5 hours (5-min candles)
MIN_PROFIT_TARGET = 0.003  # 0.3% minimum profit to justify exit
RISK_THRESHOLD = 0.005  # 0.5% risk = immediate exit
PROFIT_STAGNATION = 0.002  # 0.2% = no more profit opportunity

def generate_exit_labels(df, side='LONG'):
    """
    Generate Trade Outcome Exit Labels

    Exit = 1 (should close) if:
    1. Good profit reached (>0.3% and declining)
    2. Risk exposure high (drawdown >0.5%)
    3. Profit stagnation (next N candles < 0.2% upside)

    Exit = 0 (keep holding) if:
    - More profit opportunity exists (>0.3% upside)
    - Low risk (drawdown < 0.5%)
    """

    print(f"\n{'='*80}")
    print(f"GENERATING {side} EXIT LABELS (Trade Outcome)")
    print(f"{'='*80}")

    # Make copy
    df = df.copy()

    # Initialize exit labels
    df['exit_label'] = 0

    # Calculate future price movements
    for i in range(len(df) - LOOKAHEAD_CANDLES):
        current_price = df.iloc[i]['close']
        future_prices = df.iloc[i+1:i+1+LOOKAHEAD_CANDLES]['close'].values

        if side == 'LONG':
            # LONG Exit logic
            # Calculate max profit and max drawdown
            max_profit_pct = ((future_prices.max() - current_price) / current_price)
            max_drawdown_pct = ((future_prices.min() - current_price) / current_price)

            # Exit conditions
            exit_signal = False

            # Condition 1: High risk (drawdown > 0.5%)
            if max_drawdown_pct < -RISK_THRESHOLD:
                exit_signal = True

            # Condition 2: Profit stagnation (upside < 0.2%)
            elif max_profit_pct < PROFIT_STAGNATION:
                exit_signal = True

            # Condition 3: Good profit reached (>0.3%) but declining
            elif max_profit_pct > MIN_PROFIT_TARGET:
                # Check if price peaks early then declines
                peak_idx = np.argmax(future_prices)
                if peak_idx < len(future_prices) * 0.5:  # Peak in first half
                    exit_signal = True

        else:  # SHORT
            # SHORT Exit logic (inverse of LONG)
            max_profit_pct = ((current_price - future_prices.min()) / current_price)
            max_drawdown_pct = ((current_price - future_prices.max()) / current_price)

            # Exit conditions
            exit_signal = False

            # Condition 1: High risk (adverse move > 0.5%)
            if max_drawdown_pct < -RISK_THRESHOLD:
                exit_signal = True

            # Condition 2: Profit stagnation (downside < 0.2%)
            elif max_profit_pct < PROFIT_STAGNATION:
                exit_signal = True

            # Condition 3: Good profit reached (>0.3%) but reversing
            elif max_profit_pct > MIN_PROFIT_TARGET:
                # Check if price bottoms early then reverses
                bottom_idx = np.argmin(future_prices)
                if bottom_idx < len(future_prices) * 0.5:  # Bottom in first half
                    exit_signal = True

        df.iloc[i, df.columns.get_loc('exit_label')] = 1 if exit_signal else 0

    # Last N candles: cannot calculate lookahead → default to 0
    df.iloc[-LOOKAHEAD_CANDLES:, df.columns.get_loc('exit_label')] = 0

    # Statistics
    total = len(df)
    exits = df['exit_label'].sum()
    exit_pct = (exits / total) * 100

    print(f"\n📊 {side} EXIT LABEL STATISTICS:")
    print(f"   Total candles: {total:,}")
    print(f"   Exit signals: {exits:,} ({exit_pct:.2f}%)")
    print(f"   Hold signals: {total - exits:,} ({100-exit_pct:.2f}%)")

    print(f"\n✅ {side} Exit labels generated")

    return df[['timestamp', 'exit_label']]


def main():
    print("="*80)
    print("TRADE OUTCOME EXIT LABEL GENERATION")
    print("="*80)
    print(f"\n📂 Input: {FEATURES_FILE.name}")
    print(f"📤 Output: {OUTPUT_EXIT_LONG.name}, {OUTPUT_EXIT_SHORT.name}")

    print(f"\n⚙️ PARAMETERS:")
    print(f"   Lookahead: {LOOKAHEAD_CANDLES} candles ({LOOKAHEAD_CANDLES * 5} minutes)")
    print(f"   Min Profit Target: {MIN_PROFIT_TARGET*100}%")
    print(f"   Risk Threshold: {RISK_THRESHOLD*100}%")
    print(f"   Profit Stagnation: {PROFIT_STAGNATION*100}%")

    # Load features
    print(f"\n📖 Loading features...")
    df = pd.read_csv(FEATURES_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"   Rows: {len(df):,}")
    print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Generate LONG exit labels
    df_long_exit = generate_exit_labels(df, side='LONG')

    # Generate SHORT exit labels
    df_short_exit = generate_exit_labels(df, side='SHORT')

    # Save
    print(f"\n💾 Saving exit labels...")
    df_long_exit.to_csv(OUTPUT_EXIT_LONG, index=False)
    print(f"   ✅ LONG: {OUTPUT_EXIT_LONG}")

    df_short_exit.to_csv(OUTPUT_EXIT_SHORT, index=False)
    print(f"   ✅ SHORT: {OUTPUT_EXIT_SHORT}")

    print(f"\n{'='*80}")
    print("EXIT LABEL GENERATION COMPLETE")
    print(f"{'='*80}")

    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Retrain Exit models with these labels + 171 features")
    print(f"   2. Use same 90-day training period as Entry models")
    print(f"   3. Backtest with improved Exit models")
    print(f"   4. Compare vs current 52-day Exit models")


if __name__ == "__main__":
    main()
