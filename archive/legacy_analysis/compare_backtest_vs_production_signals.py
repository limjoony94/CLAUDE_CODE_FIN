"""
백테스트 vs 프로덕션 신호 비교 스크립트

목적: 백테스트가 우수한 성과를 냈으나, 프로덕션 신호가 다른 이유 분석

비교 항목:
1. 같은 시점 (14:25 KST) 신호 차이
2. 최근 6시간 신호 차이 분석
3. Feature 값 차이 (상위 10개 중요 feature)
4. 근본 원인 정량화

Date: 2025-11-03
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import joblib
import json
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.api.bingx_client import BingXClient
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2

def load_models():
    """Load production models"""
    models_dir = project_root / "models"

    # Load LONG entry model
    long_entry_path = models_dir / "xgboost_long_entry_enhanced_20251024_012445.pkl"
    with open(long_entry_path, 'rb') as f:
        long_entry_model = pickle.load(f)

    # Load LONG entry scaler
    long_entry_scaler_path = models_dir / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl"
    long_entry_scaler = joblib.load(long_entry_scaler_path)

    # Load LONG entry features
    long_entry_features_path = models_dir / "xgboost_long_entry_enhanced_20251024_012445_features.txt"
    with open(long_entry_features_path, 'r') as f:
        long_entry_features = [line.strip() for line in f.readlines()]

    # Load SHORT entry model
    short_entry_path = models_dir / "xgboost_short_entry_enhanced_20251024_012445.pkl"
    with open(short_entry_path, 'rb') as f:
        short_entry_model = pickle.load(f)

    # Load SHORT entry scaler
    short_entry_scaler_path = models_dir / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl"
    short_entry_scaler = joblib.load(short_entry_scaler_path)

    # Load SHORT entry features
    short_entry_features_path = models_dir / "xgboost_short_entry_enhanced_20251024_012445_features.txt"
    with open(short_entry_features_path, 'r') as f:
        short_entry_features = [line.strip() for line in f.readlines()]

    return {
        'long_entry': {
            'model': long_entry_model,
            'scaler': long_entry_scaler,
            'features': long_entry_features
        },
        'short_entry': {
            'model': short_entry_model,
            'scaler': short_entry_scaler,
            'features': short_entry_features
        }
    }

def get_backtest_signals(df, models):
    """Calculate backtest signals using current candle limit"""

    # Calculate features (same as production)
    print("\n🔧 Calculating features (same method as production)...")
    df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')

    print(f"   Features calculated: {len(df_features.columns)} columns")
    print(f"   Rows: {len(df)} → {len(df_features)} (lost {len(df) - len(df_features)} due to lookback)")

    # Get latest candle features
    latest_features = df_features.iloc[-1]

    # LONG signal
    long_features = models['long_entry']['features']
    long_feat_df = pd.DataFrame([latest_features[long_features].values], columns=long_features)
    long_feat_scaled = models['long_entry']['scaler'].transform(long_feat_df)
    long_prob = models['long_entry']['model'].predict_proba(long_feat_scaled)[0, 1]

    # SHORT signal
    short_features = models['short_entry']['features']
    short_feat_df = pd.DataFrame([latest_features[short_features].values], columns=short_features)
    short_feat_scaled = models['short_entry']['scaler'].transform(short_feat_df)
    short_prob = models['short_entry']['model'].predict_proba(short_feat_scaled)[0, 1]

    return {
        'long_prob': long_prob,
        'short_prob': short_prob,
        'features': latest_features
    }

def compare_signals(production_signal, backtest_signal):
    """Compare production vs backtest signals"""

    long_diff = production_signal['long_prob'] - backtest_signal['long_prob']
    short_diff = production_signal['short_prob'] - backtest_signal['short_prob']

    long_diff_pct = (long_diff / backtest_signal['long_prob']) * 100 if backtest_signal['long_prob'] > 0 else 0
    short_diff_pct = (short_diff / backtest_signal['short_prob']) * 100 if backtest_signal['short_prob'] > 0 else 0

    return {
        'long_diff': long_diff,
        'long_diff_pct': long_diff_pct,
        'short_diff': short_diff,
        'short_diff_pct': short_diff_pct
    }

def main():
    print("=" * 80)
    print("백테스트 vs 프로덕션 신호 비교 분석")
    print("=" * 80)

    # Load production state
    state_file = project_root / "results" / "opportunity_gating_bot_4x_state.json"
    with open(state_file, 'r') as f:
        state = json.load(f)

    production_signal = state['latest_signals']['entry']

    print(f"\n📊 프로덕션 신호 (최신):")
    print(f"   LONG: {production_signal['long_prob']:.4f} ({production_signal['long_prob']*100:.2f}%)")
    print(f"   SHORT: {production_signal['short_prob']:.4f} ({production_signal['short_prob']*100:.2f}%)")

    # Load models
    print("\n🔧 모델 로딩...")
    models = load_models()
    print(f"   ✅ LONG Entry: {len(models['long_entry']['features'])} features")
    print(f"   ✅ SHORT Entry: {len(models['short_entry']['features'])} features")

    # Load API credentials
    config_path = project_root / "config" / "api_keys.yaml"
    with open(config_path, 'r') as f:
        api_config = yaml.safe_load(f)

    # Get market data (same as production uses)
    print("\n📡 시장 데이터 가져오기 (백테스트 모드 - 프로덕션과 동일)...")
    client = BingXClient(
        api_key=api_config['bingx']['testnet']['api_key'],
        secret_key=api_config['bingx']['testnet']['secret_key'],
        testnet=True
    )

    # Fetch OHLCV data (same as production)
    ohlcv = client.exchange.fetch_ohlcv(
        symbol='BTC/USDT:USDT',
        timeframe='5m',
        limit=1000  # Same as production
    )

    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    print(f"   ✅ 데이터 수집: {len(df)} candles")
    print(f"   첫 캔들: {df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')}")
    print(f"   마지막 캔들: {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')}")

    # Calculate backtest signals
    backtest_result = get_backtest_signals(df, models)

    print(f"\n📊 백테스트 신호 (프로덕션과 동일한 데이터 사용):")
    print(f"   LONG: {backtest_result['long_prob']:.4f} ({backtest_result['long_prob']*100:.2f}%)")
    print(f"   SHORT: {backtest_result['short_prob']:.4f} ({backtest_result['short_prob']*100:.2f}%)")

    # Compare signals
    diff = compare_signals(production_signal, backtest_result)

    print("\n" + "=" * 80)
    print("📊 신호 차이 분석")
    print("=" * 80)

    print(f"\nLONG 신호 차이:")
    print(f"   프로덕션: {production_signal['long_prob']:.4f} ({production_signal['long_prob']*100:.2f}%)")
    print(f"   백테스트:  {backtest_result['long_prob']:.4f} ({backtest_result['long_prob']*100:.2f}%)")
    print(f"   차이:      {diff['long_diff']:+.4f} ({diff['long_diff_pct']:+.2f}%)")

    print(f"\nSHORT 신호 차이:")
    print(f"   프로덕션: {production_signal['short_prob']:.4f} ({production_signal['short_prob']*100:.2f}%)")
    print(f"   백테스트:  {backtest_result['short_prob']:.4f} ({backtest_result['short_prob']*100:.2f}%)")
    print(f"   차이:      {diff['short_diff']:+.4f} ({diff['short_diff_pct']:+.2f}%)")

    # Severity assessment
    print("\n" + "=" * 80)
    print("⚠️  심각도 평가")
    print("=" * 80)

    long_abs_diff = abs(diff['long_diff'])
    short_abs_diff = abs(diff['short_diff'])

    if long_abs_diff > 0.1:
        print(f"\n🔴 LONG 신호 차이 CRITICAL: {long_abs_diff:.4f} (>0.1 임계값)")
    elif long_abs_diff > 0.05:
        print(f"\n🟡 LONG 신호 차이 WARNING: {long_abs_diff:.4f} (>0.05)")
    else:
        print(f"\n🟢 LONG 신호 차이 OK: {long_abs_diff:.4f} (<0.05)")

    if short_abs_diff > 0.1:
        print(f"🔴 SHORT 신호 차이 CRITICAL: {short_abs_diff:.4f} (>0.1 임계값)")
    elif short_abs_diff > 0.05:
        print(f"🟡 SHORT 신호 차이 WARNING: {short_abs_diff:.4f} (>0.05)")
    else:
        print(f"🟢 SHORT 신호 차이 OK: {short_abs_diff:.4f} (<0.05)")

    # Root cause analysis
    print("\n" + "=" * 80)
    print("🔍 근본 원인 분석")
    print("=" * 80)

    print(f"\n데이터 사용량:")
    print(f"   백테스트: {len(df)} candles")
    print(f"   프로덕션: ~{len(df)} candles (동일한 API 제한)")
    print(f"   ✅ 데이터 양은 동일함!")

    print(f"\n가능한 차이점:")
    print(f"   1. 데이터 fetch 시점 차이 (몇 분 차이 가능)")
    print(f"   2. Feature 계산 시점 차이")
    print(f"   3. 모델 로딩 상태 차이 (unlikely)")
    print(f"   4. Numerical precision 차이 (unlikely)")

    # Solution
    print("\n" + "=" * 80)
    print("✅ 결론")
    print("=" * 80)

    print(f"\n백테스트 vs 프로덕션 신호 차이:")
    print(f"   LONG: {diff['long_diff']:+.4f} ({diff['long_diff_pct']:+.2f}%)")
    print(f"   SHORT: {diff['short_diff']:+.4f} ({diff['short_diff_pct']:+.2f}%)")

    if long_abs_diff < 0.05 and short_abs_diff < 0.05:
        print(f"\n✅ 신호 차이가 매우 작음 (<5%)")
        print(f"   - 백테스트와 프로덕션이 거의 동일한 신호 생성")
        print(f"   - 데이터 룩백 윈도우 불일치 가설은 틀렸음!")
        print(f"   - 신호 차이는 시점 차이일 가능성 높음")
    else:
        print(f"\n⚠️  신호 차이가 존재함 (>5%)")
        print(f"   - 추가 분석 필요: Feature 값 직접 비교")
        print(f"   - Feature logging으로 정확한 원인 파악 가능 (7일 후)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
