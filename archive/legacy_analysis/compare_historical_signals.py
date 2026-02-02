"""
프로덕션 로그 vs 백테스트 신호 비교 (Historical)

목적: 일주일 전 프로덕션 로그의 신호를 백테스트로 재현하여 비교

Date: 2025-11-03
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import joblib
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
    """Calculate backtest signals for historical data"""

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
        'features': latest_features,
        'timestamp': df_features.iloc[-1]['timestamp'] if 'timestamp' in df_features.columns else df.iloc[-1]['timestamp']
    }

def parse_production_log(log_file, target_time_kst):
    """Parse production log to extract signals at specific time"""

    print(f"\n📋 프로덕션 로그 파싱 중...")
    print(f"   파일: {log_file.name}")
    print(f"   목표 시간 (KST): {target_time_kst.strftime('%Y-%m-%d %H:%M')}")

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the line matching target time
    target_str = target_time_kst.strftime('%H:%M:00 KST')

    for line in lines:
        if target_str in line and 'LONG:' in line and 'SHORT:' in line:
            # Parse: [Candle 21:00:00 KST] Price: $114,514.7 | Balance: $4,376.87 | LONG: 0.5876 | SHORT: 0.5968
            parts = line.split('|')

            # Extract price
            price_str = [p for p in parts if 'Price:' in p][0]
            price = float(price_str.split('$')[1].replace(',', '').strip())

            # Extract LONG
            long_str = [p for p in parts if 'LONG:' in p][0]
            long_prob = float(long_str.split(':')[1].strip())

            # Extract SHORT
            short_str = [p for p in parts if 'SHORT:' in p][0]
            short_prob = float(short_str.split(':')[1].strip())

            print(f"   ✅ 로그 발견!")
            print(f"      Price: ${price:,.1f}")
            print(f"      LONG: {long_prob:.4f} ({long_prob*100:.2f}%)")
            print(f"      SHORT: {short_prob:.4f} ({short_prob*100:.2f}%)")

            return {
                'timestamp': target_time_kst,
                'price': price,
                'long_prob': long_prob,
                'short_prob': short_prob
            }

    print(f"   ❌ 해당 시간 로그를 찾을 수 없음")
    return None

def main():
    print("=" * 80)
    print("Historical 백테스트 vs 프로덕션 신호 비교 분석")
    print("=" * 80)

    # Target time: Oct 28, 21:00 KST (12:00 UTC)
    target_time_kst = datetime(2025, 10, 28, 21, 0, 0)
    target_time_utc = target_time_kst - timedelta(hours=9)

    print(f"\n🎯 비교 시점:")
    print(f"   KST: {target_time_kst.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   UTC: {target_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    # Parse production log
    log_file = project_root / "logs" / "opportunity_gating_bot_4x_20251028.log"
    production_signal = parse_production_log(log_file, target_time_kst)

    if production_signal is None:
        print("\n❌ 프로덕션 로그에서 해당 시간 신호를 찾을 수 없습니다.")
        return

    # Load models
    print("\n🔧 모델 로딩...")
    models = load_models()
    print(f"   ✅ LONG Entry: {len(models['long_entry']['features'])} features")
    print(f"   ✅ SHORT Entry: {len(models['short_entry']['features'])} features")

    # Load API credentials
    config_path = project_root / "config" / "api_keys.yaml"
    with open(config_path, 'r') as f:
        api_config = yaml.safe_load(f)

    # Get historical market data
    print("\n📡 Historical 시장 데이터 가져오기...")
    print(f"   목표: {target_time_utc.strftime('%Y-%m-%d %H:%M')} UTC 까지의 1000 캔들")
    print(f"   ⚠️  Using MAINNET API (production was on mainnet)")

    client = BingXClient(
        api_key=api_config['bingx']['mainnet']['api_key'],
        secret_key=api_config['bingx']['mainnet']['secret_key'],
        testnet=False  # Use mainnet to match production environment
    )

    # Calculate the timestamp for "since" parameter
    # We want 1000 candles ending at target_time_utc
    # The last candle timestamp is 5 minutes before target (candle that completes at target)
    # For 1000 candles: since = target - 5 min - (999 × 5 min) = target - 5000 min
    # But we fetch 1440 to be safe, then filter to last 1000
    # 1440 candles × 5 min = 7200 minutes

    # Calculate since time: we want candles ending at target, so:
    # Last candle at (target - 5min), first candle 999 candles before that
    last_candle_time = target_time_utc - timedelta(minutes=5)
    since_time = last_candle_time - timedelta(minutes=999 * 5)  # 4995 minutes
    since_ms = int(since_time.timestamp() * 1000)

    # Fetch OHLCV data (request max 1440 allowed by BingX API)
    ohlcv = client.exchange.fetch_ohlcv(
        symbol='BTC/USDT:USDT',
        timeframe='5m',
        since=since_ms,
        limit=1440
    )

    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Filter to get data up to target time
    df = df[df['timestamp'] <= target_time_utc].copy()

    # Keep only the last 1000 candles (to match production's limit=1000)
    if len(df) > 1000:
        df = df.iloc[-1000:].copy()

    print(f"   ✅ 데이터 수집: {len(df)} candles")
    print(f"   첫 캔들: {df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"   마지막 캔들: {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')} UTC")

    # Verify we got the right ending time
    expected_end = target_time_utc - timedelta(minutes=5)  # Completed candle is 5 min before
    actual_end = df['timestamp'].iloc[-1]

    if actual_end != expected_end:
        print(f"   ⚠️  Warning: Expected end {expected_end}, got {actual_end}")

    # Calculate backtest signals
    backtest_result = get_backtest_signals(df, models)

    print(f"\n📊 백테스트 신호 (Historical data {target_time_kst.strftime('%Y-%m-%d %H:%M KST')}):")
    print(f"   LONG: {backtest_result['long_prob']:.4f} ({backtest_result['long_prob']*100:.2f}%)")
    print(f"   SHORT: {backtest_result['short_prob']:.4f} ({backtest_result['short_prob']*100:.2f}%)")

    # Compare signals
    long_diff = production_signal['long_prob'] - backtest_result['long_prob']
    short_diff = production_signal['short_prob'] - backtest_result['short_prob']

    long_diff_pct = (abs(long_diff) / production_signal['long_prob']) * 100 if production_signal['long_prob'] > 0 else 0
    short_diff_pct = (abs(short_diff) / production_signal['short_prob']) * 100 if production_signal['short_prob'] > 0 else 0

    print("\n" + "=" * 80)
    print("📊 신호 차이 분석")
    print("=" * 80)

    print(f"\nLONG 신호 차이:")
    print(f"   프로덕션: {production_signal['long_prob']:.4f} ({production_signal['long_prob']*100:.2f}%)")
    print(f"   백테스트:  {backtest_result['long_prob']:.4f} ({backtest_result['long_prob']*100:.2f}%)")
    print(f"   차이:      {long_diff:+.4f} ({long_diff_pct:+.2f}%)")

    print(f"\nSHORT 신호 차이:")
    print(f"   프로덕션: {production_signal['short_prob']:.4f} ({production_signal['short_prob']*100:.2f}%)")
    print(f"   백테스트:  {backtest_result['short_prob']:.4f} ({backtest_result['short_prob']*100:.2f}%)")
    print(f"   차이:      {short_diff:+.4f} ({short_diff_pct:+.2f}%)")

    # Severity assessment
    print("\n" + "=" * 80)
    print("⚠️  심각도 평가")
    print("=" * 80)

    long_abs_diff = abs(long_diff)
    short_abs_diff = abs(short_diff)

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

    # Conclusion
    print("\n" + "=" * 80)
    print("✅ 결론")
    print("=" * 80)

    print(f"\n일주일 전 (10월 28일) 백테스트 vs 프로덕션 신호 차이:")
    print(f"   LONG: {long_diff:+.4f} ({long_diff_pct:+.2f}%)")
    print(f"   SHORT: {short_diff:+.4f} ({short_diff_pct:+.2f}%)")

    if long_abs_diff < 0.05 and short_abs_diff < 0.05:
        print(f"\n✅ 신호 차이가 매우 작음 (<5%)")
        print(f"   - 일주일 전에도 백테스트와 프로덕션이 거의 동일한 신호 생성")
        print(f"   - 데이터 룩백 윈도우 불일치 가설은 틀렸음! (historical 데이터에서도 동일)")
        print(f"   - 백테스트 수익과 프로덕션 손실의 차이는 시장 regime 변화 때문")
    else:
        print(f"\n⚠️  신호 차이가 존재함 (>5%)")
        print(f"   - Historical 데이터에서도 백테스트 ≠ 프로덕션")
        print(f"   - Feature 계산이나 모델 로딩에 차이 있을 수 있음")
        print(f"   - 추가 분석 필요: Feature 값 직접 비교")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
