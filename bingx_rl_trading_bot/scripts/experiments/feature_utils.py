"""
Optimized Feature Calculation Utilities
Fixes DataFrame fragmentation by using pd.concat instead of repeated column assignments
"""

import pandas as pd
import numpy as np
import talib

def calculate_short_features_optimized(df):
    """
    Calculate all SHORT-specific features efficiently (no fragmentation)
    Returns: DataFrame with all original + new features
    """
    features = {}

    # Symmetric features (keep intermediates separate)
    rsi_raw = talib.RSI(df['close'], timeperiod=14)
    features['rsi_deviation'] = np.abs(rsi_raw - 50)
    features['rsi_direction'] = np.sign(rsi_raw - 50)
    features['rsi_extreme'] = ((rsi_raw > 70) | (rsi_raw < 30)).astype(float)

    macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    features['macd_strength'] = np.abs(macd_hist)
    features['macd_direction'] = np.sign(macd_hist)
    features['macd_divergence_abs'] = np.abs(macd - macd_signal)

    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    features['price_distance_ma20'] = np.abs(df['close'] - ma_20) / ma_20
    features['price_direction_ma20'] = np.sign(df['close'] - ma_20)
    features['price_distance_ma50'] = np.abs(df['close'] - ma_50) / ma_50
    features['price_direction_ma50'] = np.sign(df['close'] - ma_50)

    # Note: 'volatility' already exists from base features, don't recreate
    features['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    features['atr_pct'] = features['atr'] / df['close']

    # Inverse features
    features['negative_momentum'] = -df['close'].pct_change(5).clip(upper=0)
    features['negative_acceleration'] = -df['close'].diff().diff().clip(upper=0)

    features['down_candle'] = (df['close'] < df['open']).astype(float)
    features['down_candle_ratio'] = features['down_candle'].rolling(10).mean()
    features['down_candle_body'] = (df['open'] - df['close']).clip(lower=0)

    features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    features['lower_low_streak'] = features['lower_low'].rolling(5).sum()

    resistance = df['high'].rolling(20).max()
    features['near_resistance'] = (df['high'] > resistance.shift(1) * 0.99).astype(float)
    features['rejection_from_resistance'] = ((features['near_resistance'] == 1) & (df['close'] < df['open'])).astype(float)
    features['resistance_rejection_count'] = features['rejection_from_resistance'].rolling(10).sum()

    price_change_5 = df['close'].diff(5)
    rsi_change_5 = rsi_raw.diff(5)
    features['bearish_divergence'] = ((price_change_5 > 0) & (rsi_change_5 < 0)).astype(float)

    price_up = (df['close'] > df['close'].shift(1)).astype(float)
    features['volume_on_decline'] = df['volume'] * (1 - price_up)
    features['volume_on_advance'] = df['volume'] * price_up
    features['volume_decline_ratio'] = features['volume_on_decline'].rolling(10).sum() / (features['volume_on_advance'].rolling(10).sum() + 1e-10)

    price_range_20 = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
    volume_surge = df['volume'] / df['volume'].rolling(20).mean()
    features['distribution_signal'] = ((price_range_20 < 0.05) & (volume_surge > 1.5)).astype(float)

    # Opportunity cost features
    returns_20 = df['close'].pct_change(20)
    features['bear_market_strength'] = (-returns_20).clip(lower=0)

    features['ema_12'] = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    features['trend_strength'] = (features['ema_12'] - ema_26) / df['close']
    features['downtrend_confirmed'] = (features['trend_strength'] < -0.01).astype(float)

    returns = df['close'].pct_change()
    upside_vol = returns[returns > 0].rolling(20).std()
    downside_vol = returns[returns < 0].rolling(20).std()
    features['downside_volatility'] = downside_vol.ffill()
    features['upside_volatility'] = upside_vol.ffill()
    features['volatility_asymmetry'] = features['downside_volatility'] / (features['upside_volatility'] + 1e-10)

    support = df['low'].rolling(50).min()
    features['below_support'] = (df['close'] < support.shift(1) * 1.01).astype(float)
    features['support_breakdown'] = ((df['close'].shift(1) >= support.shift(1)) & (df['close'] < support.shift(1))).astype(float)

    features['panic_selling'] = ((features['down_candle'] == 1) & (df['volume'] > df['volume'].rolling(10).mean() * 1.5)).astype(float)

    # Single concat operation - NO FRAGMENTATION!
    feature_df = pd.DataFrame(features, index=df.index)
    result = pd.concat([df, feature_df], axis=1)

    return result.ffill().bfill().fillna(0)
