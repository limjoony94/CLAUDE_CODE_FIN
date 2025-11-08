#!/usr/bin/env python
"""
백테스트 데이터의 LONG/SHORT 확률 분포 분석

최근 실제 운영과 비교하여 모델이 정상적으로 작동하는지 확인
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load models
long_model = joblib.load('models/xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl')
short_model = joblib.load('models/xgboost_short_redesigned_20251016_233322.pkl')

# Load scalers
long_scaler = joblib.load('models/xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl')
short_scaler = joblib.load('models/xgboost_short_redesigned_20251016_233322_scaler.pkl')

# Load validation/test data (if available)
print("=" * 70)
print("백테스트 vs 실제 운영 LONG/SHORT 확률 분포 비교")
print("=" * 70)
print()

# Try to load test data from backtest
try:
    # Load historical data
    from scripts.experiments.calculate_all_features import calculate_all_features
    from src.api.bingx_client import BingXClient
    import yaml

    # Load API keys
    with open('config/api_keys.yaml', 'r') as f:
        config = yaml.safe_load(f)

    api_key = config['bingx']['mainnet']['api_key']
    api_secret = config['bingx']['mainnet']['secret_key']

    # Initialize client
    client = BingXClient(api_key, api_secret, testnet=False)

    print("1️⃣ 최근 1000개 캔들 데이터로 백테스트 확률 계산...")

    # Get historical data (last 1000 candles)
    klines = client.get_klines('BTC-USDT', '5m', limit=1000)

    # Convert to DataFrame
    df = pd.DataFrame(klines)
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    print(f"   데이터 범위: {df.iloc[0]['timestamp']} ~ {df.iloc[-1]['timestamp']}")
    print(f"   총 캔들 수: {len(df)}")

    # Calculate features
    print("\n2️⃣ Feature 계산 중...")
    df_features = calculate_all_features(df.copy())

    # LONG features
    LONG_FEATURES = [
        'returns', 'volatility', 'rsi', 'macd', 'bb_width', 'volume_change',
        'price_vs_ma20', 'price_vs_ma50', 'ma20_slope', 'ma50_slope',
        'high_low_range', 'close_vs_open', 'upper_shadow', 'lower_shadow',
        'trend_strength', 'volume_trend', 'price_momentum', 'volatility_regime',
        'sr_resistance_dist', 'sr_support_dist', 'trend_quality',
        'log_returns', 'returns_std', 'volume_ma_ratio',
        'price_range', 'body_ratio', 'trend_consistency',
        'ema_distance_20', 'ema_distance_50',
        'volume_spike', 'consecutive_direction',
        'atr', 'adx', 'plus_di', 'minus_di',
        'stoch_k', 'stoch_d', 'cci', 'willr', 'roc', 'mfi',
        'obv_change', 'cmf', 'vwap_distance', 'pivot_distance'
    ]

    # SHORT features
    SHORT_FEATURES = [
        'rsi_deviation', 'rsi_direction', 'rsi_extreme',
        'macd_strength', 'macd_direction', 'macd_divergence_abs',
        'price_distance_ma20', 'price_direction_ma20',
        'price_distance_ma50', 'price_direction_ma50',
        'volatility', 'atr_pct', 'atr',
        'negative_momentum', 'negative_acceleration',
        'down_candle_ratio', 'down_candle_body',
        'lower_low_streak', 'resistance_rejection_count',
        'bearish_divergence', 'breakdown_momentum',
        'lower_shadow_dominance', 'wick_rejection_ratio',
        'bear_market_strength', 'trend_strength',
        'downtrend_confirmed', 'downside_volatility',
        'upside_volatility', 'volatility_asymmetry',
        'below_support', 'support_breakdown', 'panic_selling'
    ]

    # Calculate probabilities for all rows
    print("\n3️⃣ LONG/SHORT 확률 계산 중...")

    # Ensure all features exist
    for feat in LONG_FEATURES:
        if feat not in df_features.columns:
            print(f"   ⚠️ Missing LONG feature: {feat}")

    for feat in SHORT_FEATURES:
        if feat not in df_features.columns:
            print(f"   ⚠️ Missing SHORT feature: {feat}")

    # Drop rows with NaN
    df_clean = df_features.dropna(subset=LONG_FEATURES + SHORT_FEATURES)
    print(f"   유효한 행 수: {len(df_clean)} (NaN 제거 후)")

    if len(df_clean) == 0:
        print("   ❌ 유효한 데이터가 없습니다!")
        sys.exit(1)

    # Prepare features
    X_long = df_clean[LONG_FEATURES].values
    X_short = df_clean[SHORT_FEATURES].values

    # Scale features
    X_long_scaled = long_scaler.transform(X_long)
    X_short_scaled = short_scaler.transform(X_short)

    # Predict probabilities
    long_probs = long_model.predict_proba(X_long_scaled)[:, 1]
    short_probs = short_model.predict_proba(X_short_scaled)[:, 1]

    print("\n" + "=" * 70)
    print("📊 백테스트 데이터 확률 분포 (최근 1000개 캔들)")
    print("=" * 70)
    print()

    print("LONG Probability 통계:")
    print(f"  평균: {long_probs.mean():.4f}")
    print(f"  중앙값: {np.median(long_probs):.4f}")
    print(f"  최소: {long_probs.min():.4f}")
    print(f"  최대: {long_probs.max():.4f}")
    print()
    print("LONG Probability 분포:")
    print(f"  < 0.3: {(long_probs < 0.3).sum()} ({(long_probs < 0.3).sum()/len(long_probs)*100:.1f}%)")
    print(f"  0.3-0.5: {((long_probs >= 0.3) & (long_probs < 0.5)).sum()} ({((long_probs >= 0.3) & (long_probs < 0.5)).sum()/len(long_probs)*100:.1f}%)")
    print(f"  0.5-0.65: {((long_probs >= 0.5) & (long_probs < 0.65)).sum()} ({((long_probs >= 0.5) & (long_probs < 0.65)).sum()/len(long_probs)*100:.1f}%)")
    print(f"  0.65-0.8: {((long_probs >= 0.65) & (long_probs < 0.8)).sum()} ({((long_probs >= 0.65) & (long_probs < 0.8)).sum()/len(long_probs)*100:.1f}%)")
    print(f"  >= 0.8: {(long_probs >= 0.8).sum()} ({(long_probs >= 0.8).sum()/len(long_probs)*100:.1f}%)")
    print()

    print("SHORT Probability 통계:")
    print(f"  평균: {short_probs.mean():.4f}")
    print(f"  중앙값: {np.median(short_probs):.4f}")
    print(f"  최소: {short_probs.min():.4f}")
    print(f"  최대: {short_probs.max():.4f}")
    print()
    print("SHORT Probability 분포:")
    print(f"  < 0.1: {(short_probs < 0.1).sum()} ({(short_probs < 0.1).sum()/len(short_probs)*100:.1f}%)")
    print(f"  0.1-0.3: {((short_probs >= 0.1) & (short_probs < 0.3)).sum()} ({((short_probs >= 0.1) & (short_probs < 0.3)).sum()/len(short_probs)*100:.1f}%)")
    print(f"  0.3-0.5: {((short_probs >= 0.3) & (short_probs < 0.5)).sum()} ({((short_probs >= 0.3) & (short_probs < 0.5)).sum()/len(short_probs)*100:.1f}%)")
    print(f"  0.5-0.7: {((short_probs >= 0.5) & (short_probs < 0.7)).sum()} ({((short_probs >= 0.5) & (short_probs < 0.7)).sum()/len(short_probs)*100:.1f}%)")
    print(f"  >= 0.7: {(short_probs >= 0.7).sum()} ({(short_probs >= 0.7).sum()/len(short_probs)*100:.1f}%)")
    print()

    print("=== Threshold 초과 비율 ===")
    print(f"LONG >= 0.65: {(long_probs >= 0.65).sum()} ({(long_probs >= 0.65).sum()/len(long_probs)*100:.1f}%)")
    print(f"SHORT >= 0.70: {(short_probs >= 0.70).sum()} ({(short_probs >= 0.70).sum()/len(short_probs)*100:.1f}%)")
    print()
    print(f"총 샘플 수: {len(long_probs)}")

    print("\n" + "=" * 70)
    print("🔍 분석 결론")
    print("=" * 70)

    # Compare with live data (from previous analysis)
    live_long_mean = 0.8147
    live_long_above_065 = 88.9

    backtest_long_mean = long_probs.mean()
    backtest_long_above_065 = (long_probs >= 0.65).sum() / len(long_probs) * 100

    print(f"\nLONG 확률 평균:")
    print(f"  실제 운영: {live_long_mean:.4f}")
    print(f"  백테스트: {backtest_long_mean:.4f}")
    print(f"  차이: {abs(live_long_mean - backtest_long_mean):.4f}")

    print(f"\nLONG threshold (0.65) 초과 비율:")
    print(f"  실제 운영: {live_long_above_065:.1f}%")
    print(f"  백테스트: {backtest_long_above_065:.1f}%")
    print(f"  차이: {abs(live_long_above_065 - backtest_long_above_065):.1f}%p")

    if abs(live_long_mean - backtest_long_mean) < 0.1:
        print("\n✅ 실제 운영과 백테스트의 확률 분포가 유사합니다.")
        print("   모델이 정상적으로 작동하고 있습니다.")
    else:
        print("\n⚠️  실제 운영과 백테스트의 확률 분포에 차이가 있습니다!")
        print("   시장 상황이 변했거나 데이터 문제일 수 있습니다.")

except Exception as e:
    print(f"❌ 에러 발생: {e}")
    import traceback
    traceback.print_exc()
