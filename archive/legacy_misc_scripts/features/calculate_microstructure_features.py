"""
Microstructure Feature Calculator
==================================

Calculates 20 microstructure features:
- Order Flow Proxies (8 features): Buy/sell imbalance, trade aggression
- Volume Microstructure (6 features): Tick rule, volume at price levels
- Trade Intensity (6 features): Trade velocity, clustering

Created: 2025-10-29
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

def calculate_order_flow_features(df):
    """
    Calculate 8 order flow proxy features.

    Returns: DataFrame with microstructure_* features
    """
    print("  Calculating Order Flow Proxy features...")

    # 1. Buy/Sell Imbalance (close position in candle as proxy)
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['microstructure_buy_sell_imbalance'] = (df['close_position'] - 0.5) * 2  # -1 to +1

    # 2. Trade Aggression (close vs typical price)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['microstructure_trade_aggression'] = (df['close'] - df['typical_price']) / df['typical_price']

    # 3. Volume-Weighted Buy Pressure
    df['microstructure_vwap_buy_pressure'] = (
        df['close_position'] * df['volume']
    ).rolling(20).mean()

    # 4. Volume-Weighted Sell Pressure
    df['microstructure_vwap_sell_pressure'] = (
        (1 - df['close_position']) * df['volume']
    ).rolling(20).mean()

    # 5. Order Flow Momentum (change in imbalance)
    df['microstructure_order_flow_momentum'] = (
        df['microstructure_buy_sell_imbalance'].diff(5)
    )

    # 6. Cumulative Delta (cumulative buy/sell imbalance)
    df['delta'] = df['microstructure_buy_sell_imbalance'] * df['volume']
    df['microstructure_cumulative_delta'] = df['delta'].rolling(20).sum()

    # 7. Delta Divergence (price vs cumulative delta)
    price_change = df['close'].diff(20)
    delta_change = df['microstructure_cumulative_delta'].diff(20)
    df['microstructure_delta_divergence'] = np.where(
        (price_change > 0) & (delta_change < 0), -1,  # Bearish divergence
        np.where((price_change < 0) & (delta_change > 0), 1, 0)  # Bullish divergence
    )

    # 8. Order Flow Consistency (std of imbalance)
    df['microstructure_order_flow_consistency'] = (
        1 - df['microstructure_buy_sell_imbalance'].rolling(20).std()
    )

    # Select microstructure features
    microstructure_cols = [col for col in df.columns if col.startswith('microstructure_') and any(x in col for x in ['imbalance', 'aggression', 'pressure', 'momentum', 'delta', 'consistency'])]
    return df[microstructure_cols]

def calculate_volume_microstructure_features(df):
    """
    Calculate 6 volume microstructure features.

    Returns: DataFrame with microstructure_* features
    """
    print("  Calculating Volume Microstructure features...")

    # 1. Tick Rule (price change direction weighted by volume)
    price_change = df['close'].diff()
    df['tick_direction'] = np.sign(price_change)
    df['microstructure_tick_rule'] = (
        df['tick_direction'] * df['volume']
    ).rolling(20).mean()

    # 2. Volume at Ask/Bid (proxy using close position)
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['microstructure_volume_at_ask'] = (
        df['close_position'] * df['volume']
    ).rolling(20).sum()
    df['microstructure_volume_at_bid'] = (
        (1 - df['close_position']) * df['volume']
    ).rolling(20).sum()

    # 3. Volume Imbalance Ratio
    df['microstructure_volume_imbalance_ratio'] = (
        df['microstructure_volume_at_ask'] /
        df['microstructure_volume_at_bid'].replace(0, np.nan)
    ).fillna(1.0)

    # 4. Volume Concentration (HHI - Herfindahl index approximation)
    # Measure how concentrated volume is (high = few large trades)
    df['microstructure_volume_concentration'] = df['volume'].rolling(20).apply(
        lambda x: (x ** 2).sum() / (x.sum() ** 2) if x.sum() > 0 else 0.05
    )

    # 5. Large Trade Frequency (volume > 2x median)
    volume_median = df['volume'].rolling(50).median()
    df['microstructure_large_trade_frequency'] = (
        df['volume'] > 2 * volume_median
    ).rolling(20).sum()

    # Select microstructure features
    microstructure_cols = [col for col in df.columns if col.startswith('microstructure_') and any(x in col for x in ['tick_rule', 'volume_at', 'volume_imbalance', 'concentration', 'large_trade'])]
    return df[microstructure_cols]

def calculate_trade_intensity_features(df):
    """
    Calculate 6 trade intensity features.

    Returns: DataFrame with microstructure_* features
    """
    print("  Calculating Trade Intensity features...")

    # 1. Price Velocity (rate of price change)
    df['microstructure_price_velocity'] = df['close'].diff(5) / 5

    # 2. Volume Velocity (rate of volume change)
    df['microstructure_volume_velocity'] = df['volume'].diff(5) / 5

    # 3. Trade Clustering (volume spikes)
    volume_mean = df['volume'].rolling(50).mean()
    volume_std = df['volume'].rolling(50).std()
    df['microstructure_trade_clustering'] = (
        (df['volume'] - volume_mean) / volume_std.replace(0, np.nan)
    ).fillna(0)

    # 4. Volatility Clustering (volatility persistence)
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(5).std()
    df['microstructure_volatility_clustering'] = (
        df['volatility'].rolling(20).mean()
    )

    # 5. Trade Arrival Rate (approximation using volume frequency)
    df['microstructure_trade_arrival_rate'] = (
        df['volume'] > df['volume'].rolling(20).mean()
    ).rolling(10).sum()

    # 6. Order Flow Toxicity (proxy: price impact per unit volume)
    # High toxicity = informed trading
    price_impact = abs(df['close'].diff())
    volume_normalized = df['volume'] / df['volume'].rolling(50).mean()
    df['microstructure_order_flow_toxicity'] = (
        price_impact / volume_normalized.replace(0, np.nan)
    ).rolling(20).mean()

    # Select microstructure features
    microstructure_cols = [col for col in df.columns if col.startswith('microstructure_') and any(x in col for x in ['velocity', 'clustering', 'arrival', 'toxicity'])]
    return df[microstructure_cols]

def calculate_all_microstructure_features(df):
    """
    Main function: Calculate all 20 microstructure features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with original data + 20 microstructure features
    """
    print("Calculating Microstructure Features...")
    print(f"Input: {len(df):,} candles")

    # Calculate order flow features (8)
    order_flow_features = calculate_order_flow_features(df.copy())
    df = pd.concat([df, order_flow_features], axis=1)

    # Calculate volume microstructure features (6)
    volume_features = calculate_volume_microstructure_features(df.copy())
    df = pd.concat([df, volume_features], axis=1)

    # Calculate trade intensity features (6)
    intensity_features = calculate_trade_intensity_features(df.copy())
    df = pd.concat([df, intensity_features], axis=1)

    # Verify feature count
    microstructure_features = [col for col in df.columns if col.startswith('microstructure_')]
    print(f"\n✅ Generated {len(microstructure_features)} microstructure features")

    # List categories
    print("\nFeature Categories:")
    order_flow_cols = ['imbalance', 'aggression', 'pressure', 'delta', 'consistency']
    volume_cols = ['tick_rule', 'volume_at', 'volume_imbalance', 'concentration', 'large_trade']
    intensity_cols = ['velocity', 'clustering', 'arrival', 'toxicity']

    print(f"  Order Flow: {len([c for c in microstructure_features if any(x in c for x in order_flow_cols)])}")
    print(f"  Volume Microstructure: {len([c for c in microstructure_features if any(x in c for x in volume_cols)])}")
    print(f"  Trade Intensity: {len([c for c in microstructure_features if any(x in c for x in intensity_cols)])}")

    return df

if __name__ == '__main__':
    print("="*80)
    print("MICROSTRUCTURE FEATURE CALCULATOR")
    print("="*80)
    print()

    # Load data with momentum features
    data_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_momentum_features.csv"
    print(f"Loading: {data_file.name}")

    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} candles with {len(df.columns)} columns")
    print()

    # Calculate microstructure features
    df_with_microstructure = calculate_all_microstructure_features(df)

    # Save
    output_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_microstructure_features.csv"
    df_with_microstructure.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file.name}")
    print(f"   Columns: {len(df_with_microstructure.columns)}")
    print(f"   Rows: {len(df_with_microstructure):,}")
    print()

    # Sample output
    print("Sample Microstructure Features (last row):")
    microstructure_cols = [col for col in df_with_microstructure.columns if col.startswith('microstructure_')]
    print(df_with_microstructure[microstructure_cols].iloc[-1].to_string())
    print()
