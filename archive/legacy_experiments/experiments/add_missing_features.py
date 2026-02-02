"""
Add Missing Features to Dataset

Adds features required by Exit models but missing from current dataset:
- volatility_20: 20-period volatility
- sma_50: 50-period simple moving average
- ema_26: 26-period exponential moving average
"""

import pandas as pd
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "features"

def calculate_volatility(df, period=20):
    """Calculate rolling volatility"""
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=period).std()
    return volatility


def calculate_sma(df, period=50):
    """Calculate simple moving average"""
    return df['close'].rolling(window=period).mean()


def calculate_ema(df, period=26):
    """Calculate exponential moving average"""
    return df['close'].ewm(span=period, adjust=False).mean()


def add_missing_features():
    """Add missing features to dataset"""
    # Load current features
    features_path = DATA_DIR / "BTCUSDT_5m_features.csv"
    print(f"Loading: {features_path}")

    df = pd.read_csv(features_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"\nOriginal shape: {df.shape}")
    print(f"Original columns: {len(df.columns)}")

    # Add missing features
    print("\nAdding missing features...")

    # 1. volatility_20
    if 'volatility_20' not in df.columns:
        print("  Adding volatility_20...")
        df['volatility_20'] = calculate_volatility(df, period=20)
    else:
        print("  volatility_20 already exists")

    # 2. sma_50
    if 'sma_50' not in df.columns:
        print("  Adding sma_50...")
        df['sma_50'] = calculate_sma(df, period=50)
    else:
        print("  sma_50 already exists")

    # 3. ema_26
    if 'ema_26' not in df.columns:
        print("  Adding ema_26...")
        df['ema_26'] = calculate_ema(df, period=26)
    else:
        print("  ema_26 already exists")

    # 4. ma_20 (used by Exit models)
    if 'ma_20' not in df.columns:
        print("  Adding ma_20 (alias for sma_20)...")
        if 'sma_20' in df.columns:
            df['ma_20'] = df['sma_20']
        else:
            df['ma_20'] = calculate_sma(df, period=20)
    else:
        print("  ma_20 already exists")

    # Clean NaN values (forward fill then backward fill)
    df = df.ffill().bfill()

    print(f"\nUpdated shape: {df.shape}")
    print(f"Updated columns: {len(df.columns)}")

    # Verify new features
    print("\nVerifying new features:")
    for feature in ['volatility_20', 'sma_50', 'ema_26', 'ma_20']:
        if feature in df.columns:
            print(f"  ✅ {feature}: {df[feature].notna().sum()} / {len(df)} values")
        else:
            print(f"  ❌ {feature}: MISSING")

    # Save updated dataset
    output_path = DATA_DIR / "BTCUSDT_5m_features_enhanced.csv"
    print(f"\nSaving enhanced dataset: {output_path}")
    df.to_csv(output_path, index=False)

    print(f"\n✅ Enhanced dataset saved!")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


if __name__ == "__main__":
    print("=" * 80)
    print("ADDING MISSING FEATURES")
    print("=" * 80)

    df = add_missing_features()

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
