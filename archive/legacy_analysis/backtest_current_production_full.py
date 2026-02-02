"""
전체 기간 백테스트 - 현재 프로덕션 설정

목적: 현재 프로덕션에 적용된 모델, 피처 계산방법, 정규화 방법을 사용하여
      전체 역사 기간에 대한 백테스트를 수행하고 최근 손실을 예측했는지 검증

Current Production Configuration:
- Entry Models: Enhanced 5-Fold CV (20251024_012445)
  - LONG Entry: 0.80 threshold, 85 features
  - SHORT Entry: 0.80 threshold, 79 features
- Exit Models: oppgating_improved (20251024_043527/044510)
  - LONG Exit: 0.70 threshold, 27 features
  - SHORT Exit: 0.70 threshold, 27 features
- Risk: Stop Loss -3% balance, Max Hold 120 candles (10h)
- Feature Calculation: production_features_v1.py (Nov 3 09:42)

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
from scripts.production.production_exit_features_v1 import prepare_exit_features

# EXACT PRODUCTION CONFIGURATION
LONG_ENTRY_THRESHOLD = 0.80
SHORT_ENTRY_THRESHOLD = 0.80
ML_EXIT_THRESHOLD_LONG = 0.70
ML_EXIT_THRESHOLD_SHORT = 0.70
EMERGENCY_STOP_LOSS = 0.03  # -3% balance
EMERGENCY_MAX_HOLD = 120  # candles (10 hours)
LEVERAGE = 4

def load_production_models():
    """Load EXACT production models"""
    models_dir = project_root / "models"

    # LONG Entry Model (Enhanced 5-Fold CV)
    long_entry_path = models_dir / "xgboost_long_entry_enhanced_20251024_012445.pkl"
    with open(long_entry_path, 'rb') as f:
        long_entry_model = pickle.load(f)

    long_entry_scaler_path = models_dir / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl"
    long_entry_scaler = joblib.load(long_entry_scaler_path)

    long_entry_features_path = models_dir / "xgboost_long_entry_enhanced_20251024_012445_features.txt"
    with open(long_entry_features_path, 'r') as f:
        long_entry_features = [line.strip() for line in f.readlines()]

    # SHORT Entry Model (Enhanced 5-Fold CV)
    short_entry_path = models_dir / "xgboost_short_entry_enhanced_20251024_012445.pkl"
    with open(short_entry_path, 'rb') as f:
        short_entry_model = pickle.load(f)

    short_entry_scaler_path = models_dir / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl"
    short_entry_scaler = joblib.load(short_entry_scaler_path)

    short_entry_features_path = models_dir / "xgboost_short_entry_enhanced_20251024_012445_features.txt"
    with open(short_entry_features_path, 'r') as f:
        short_entry_features = [line.strip() for line in f.readlines()]

    # LONG Exit Model (oppgating_improved)
    long_exit_path = models_dir / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl"
    with open(long_exit_path, 'rb') as f:
        long_exit_model = pickle.load(f)

    long_exit_scaler_path = models_dir / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl"
    long_exit_scaler = joblib.load(long_exit_scaler_path)

    long_exit_features_path = models_dir / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt"
    with open(long_exit_features_path, 'r') as f:
        long_exit_features = [line.strip() for line in f.readlines()]

    # SHORT Exit Model (oppgating_improved)
    short_exit_path = models_dir / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl"
    with open(short_exit_path, 'rb') as f:
        short_exit_model = pickle.load(f)

    short_exit_scaler_path = models_dir / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl"
    short_exit_scaler = joblib.load(short_exit_scaler_path)

    short_exit_features_path = models_dir / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt"
    with open(short_exit_features_path, 'r') as f:
        short_exit_features = [line.strip() for line in f.readlines()]

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
        },
        'long_exit': {
            'model': long_exit_model,
            'scaler': long_exit_scaler,
            'features': long_exit_features
        },
        'short_exit': {
            'model': short_exit_model,
            'scaler': short_exit_scaler,
            'features': short_exit_features
        }
    }

def run_backtest(df, models):
    """Run backtest with EXACT production logic"""

    print("\n🔧 Calculating features (production method)...")
    df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')

    print(f"   Features calculated: {len(df_features.columns)} columns")
    print(f"   Rows: {len(df)} → {len(df_features)} (lost {len(df) - len(df_features)} due to lookback)")

    # Pre-calculate ALL exit features (OPTIMIZATION: once instead of 29,841 times)
    print(f"\n🔧 Pre-calculating exit features...")
    df_exit_features = prepare_exit_features(df_features.copy())
    print(f"   Exit features calculated: {len(df_exit_features.columns)} columns")
    print(f"   Rows: {len(df_features)} → {len(df_exit_features)}")

    # Initialize backtest
    initial_balance = 10000
    balance = initial_balance
    position = None
    trades = []

    print(f"\n📊 Running backtest...")
    print(f"   Period: {df_features['timestamp'].iloc[0]} to {df_features['timestamp'].iloc[-1]}")
    print(f"   Candles: {len(df_features)}")

    for i in range(len(df_features)):
        row = df_features.iloc[i]
        timestamp = row['timestamp']
        close_price = row['close']

        # Entry signals
        if position is None:
            # Calculate LONG signal (use .values to avoid sklearn warning)
            long_features = models['long_entry']['features']
            long_feat_values = row[long_features].values.reshape(1, -1)
            long_feat_scaled = models['long_entry']['scaler'].transform(long_feat_values)
            long_prob = models['long_entry']['model'].predict_proba(long_feat_scaled)[0, 1]

            # Calculate SHORT signal (use .values to avoid sklearn warning)
            short_features = models['short_entry']['features']
            short_feat_values = row[short_features].values.reshape(1, -1)
            short_feat_scaled = models['short_entry']['scaler'].transform(short_feat_values)
            short_prob = models['short_entry']['model'].predict_proba(short_feat_scaled)[0, 1]

            # Entry logic (EXACT production)
            if long_prob >= LONG_ENTRY_THRESHOLD:
                # LONG entry
                position_size_pct = 0.95  # Using max for simplicity (production uses dynamic)
                position_size_usd = balance * position_size_pct

                # Calculate balance-based stop loss (EXACT production)
                price_sl_pct = abs(EMERGENCY_STOP_LOSS) / position_size_pct
                stop_loss_price = close_price * (1 - price_sl_pct)

                position = {
                    'side': 'LONG',
                    'entry_price': close_price,
                    'entry_time': timestamp,
                    'entry_idx': i,
                    'position_size_pct': position_size_pct,
                    'position_size_usd': position_size_usd,
                    'stop_loss_price': stop_loss_price,
                    'entry_long_prob': long_prob,
                    'entry_short_prob': short_prob
                }

            elif short_prob >= SHORT_ENTRY_THRESHOLD:
                # Opportunity gating (simplified - using basic comparison)
                long_ev = long_prob * 0.0041
                short_ev = short_prob * 0.0047
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > 0.001:  # Gate threshold
                    # SHORT entry
                    position_size_pct = 0.95
                    position_size_usd = balance * position_size_pct

                    # Calculate balance-based stop loss
                    price_sl_pct = abs(EMERGENCY_STOP_LOSS) / position_size_pct
                    stop_loss_price = close_price * (1 + price_sl_pct)

                    position = {
                        'side': 'SHORT',
                        'entry_price': close_price,
                        'entry_time': timestamp,
                        'entry_idx': i,
                        'position_size_pct': position_size_pct,
                        'position_size_usd': position_size_usd,
                        'stop_loss_price': stop_loss_price,
                        'entry_long_prob': long_prob,
                        'entry_short_prob': short_prob
                    }

        # Exit logic
        if position is not None:
            hold_candles = i - position['entry_idx']

            # Use pre-calculated exit features (OPTIMIZED)
            if i < len(df_exit_features):
                exit_row = df_exit_features.iloc[i]

                # Calculate ML Exit signal
                if position['side'] == 'LONG':
                    exit_features = models['long_exit']['features']
                    exit_threshold = ML_EXIT_THRESHOLD_LONG
                elif position['side'] == 'SHORT':
                    exit_features = models['short_exit']['features']
                    exit_threshold = ML_EXIT_THRESHOLD_SHORT

                # Use .values to avoid sklearn warning
                exit_feat_values = exit_row[exit_features].values.reshape(1, -1)
                exit_feat_scaled = models[f"{position['side'].lower()}_exit"]['scaler'].transform(exit_feat_values)
                exit_prob = models[f"{position['side'].lower()}_exit"]['model'].predict_proba(exit_feat_scaled)[0, 1]
            else:
                exit_prob = 0.0

            # Exit conditions (EXACT production)
            exit_reason = None

            # 1. ML Exit
            if exit_prob >= exit_threshold:
                exit_reason = 'ML Exit'

            # 2. Emergency Max Hold
            elif hold_candles >= EMERGENCY_MAX_HOLD:
                exit_reason = 'Emergency Max Hold'

            # 3. Stop Loss
            elif position['side'] == 'LONG' and close_price <= position['stop_loss_price']:
                exit_reason = 'Stop Loss'
            elif position['side'] == 'SHORT' and close_price >= position['stop_loss_price']:
                exit_reason = 'Stop Loss'

            # Execute exit
            if exit_reason is not None:
                # Calculate P&L (4x leverage)
                if position['side'] == 'LONG':
                    price_change_pct = (close_price - position['entry_price']) / position['entry_price']
                else:  # SHORT
                    price_change_pct = (position['entry_price'] - close_price) / position['entry_price']

                leveraged_pnl_pct = price_change_pct * LEVERAGE
                pnl_usd = balance * leveraged_pnl_pct
                balance += pnl_usd

                trade = {
                    'side': position['side'],
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'entry_long_prob': position['entry_long_prob'],
                    'entry_short_prob': position['entry_short_prob'],
                    'exit_time': timestamp,
                    'exit_price': close_price,
                    'exit_prob': exit_prob,
                    'exit_reason': exit_reason,
                    'hold_candles': hold_candles,
                    'price_change_pct': price_change_pct * 100,
                    'leveraged_pnl_pct': leveraged_pnl_pct * 100,
                    'pnl_usd': pnl_usd,
                    'balance_after': balance
                }

                trades.append(trade)
                position = None

    # Convert to DataFrame
    df_trades = pd.DataFrame(trades)

    return df_trades, balance

def main():
    print("=" * 80)
    print("전체 기간 백테스트 - 현재 프로덕션 설정")
    print("=" * 80)

    print("\n📋 Configuration:")
    print(f"   Entry LONG: {LONG_ENTRY_THRESHOLD} ({LONG_ENTRY_THRESHOLD*100:.0f}%)")
    print(f"   Entry SHORT: {SHORT_ENTRY_THRESHOLD} ({SHORT_ENTRY_THRESHOLD*100:.0f}%)")
    print(f"   Exit LONG: {ML_EXIT_THRESHOLD_LONG} ({ML_EXIT_THRESHOLD_LONG*100:.0f}%)")
    print(f"   Exit SHORT: {ML_EXIT_THRESHOLD_SHORT} ({ML_EXIT_THRESHOLD_SHORT*100:.0f}%)")
    print(f"   Stop Loss: -{EMERGENCY_STOP_LOSS*100:.1f}% balance")
    print(f"   Max Hold: {EMERGENCY_MAX_HOLD} candles ({EMERGENCY_MAX_HOLD*5/60:.1f}h)")
    print(f"   Leverage: {LEVERAGE}x")

    # Load models
    print("\n🔧 모델 로딩...")
    models = load_production_models()
    print(f"   ✅ LONG Entry: {len(models['long_entry']['features'])} features")
    print(f"   ✅ SHORT Entry: {len(models['short_entry']['features'])} features")
    print(f"   ✅ LONG Exit: {len(models['long_exit']['features'])} features")
    print(f"   ✅ SHORT Exit: {len(models['short_exit']['features'])} features")

    # Load API credentials
    config_path = project_root / "config" / "api_keys.yaml"
    with open(config_path, 'r') as f:
        api_config = yaml.safe_load(f)

    # Get market data (full period)
    print("\n📡 시장 데이터 가져오기 (전체 기간)...")
    client = BingXClient(
        api_key=api_config['bingx']['testnet']['api_key'],
        secret_key=api_config['bingx']['testnet']['secret_key'],
        testnet=True
    )

    # Fetch maximum data (1440 candles max per API)
    # We'll fetch multiple batches to cover full period
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=120)  # ~4 months

    print(f"   Period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")

    all_data = []
    current_end = end_time

    while current_end > start_time:
        since_ms = int((current_end - timedelta(minutes=1440 * 5)).timestamp() * 1000)

        ohlcv = client.exchange.fetch_ohlcv(
            symbol='BTC/USDT:USDT',
            timeframe='5m',
            since=since_ms,
            limit=1440
        )

        if not ohlcv:
            break

        all_data.extend(ohlcv)
        current_end = datetime.fromtimestamp(ohlcv[0][0] / 1000)

        if len(all_data) % 5000 == 0:
            print(f"   Fetched {len(all_data)} candles...")

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    print(f"   ✅ 데이터 수집: {len(df)} candles")
    print(f"   첫 캔들: {df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"   마지막 캔들: {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')} UTC")

    # Run backtest
    df_trades, final_balance = run_backtest(df, models)

    # Calculate metrics
    initial_balance = 10000
    total_return = ((final_balance - initial_balance) / initial_balance) * 100

    if len(df_trades) > 0:
        wins = len(df_trades[df_trades['pnl_usd'] > 0])
        losses = len(df_trades[df_trades['pnl_usd'] <= 0])
        win_rate = (wins / len(df_trades)) * 100

        avg_win = df_trades[df_trades['pnl_usd'] > 0]['pnl_usd'].mean() if wins > 0 else 0
        avg_loss = df_trades[df_trades['pnl_usd'] <= 0]['pnl_usd'].mean() if losses > 0 else 0

        # Exit distribution
        exit_counts = df_trades['exit_reason'].value_counts()

        # Side distribution
        side_counts = df_trades['side'].value_counts()

        print("\n" + "=" * 80)
        print("📊 백테스트 결과")
        print("=" * 80)

        print(f"\n전체 성과:")
        print(f"   초기 자본: ${initial_balance:,.2f}")
        print(f"   최종 자본: ${final_balance:,.2f}")
        print(f"   총 수익률: {total_return:+.2f}%")

        print(f"\n거래 통계:")
        print(f"   총 거래: {len(df_trades)}")
        print(f"   승리: {wins} ({win_rate:.1f}%)")
        print(f"   패배: {losses} ({100-win_rate:.1f}%)")
        print(f"   평균 승리: ${avg_win:+.2f}")
        print(f"   평균 패배: ${avg_loss:+.2f}")

        print(f"\nExit 분포:")
        for reason, count in exit_counts.items():
            pct = (count / len(df_trades)) * 100
            print(f"   {reason}: {count} ({pct:.1f}%)")

        print(f"\nSide 분포:")
        for side, count in side_counts.items():
            pct = (count / len(df_trades)) * 100
            print(f"   {side}: {count} ({pct:.1f}%)")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = project_root / "results" / f"backtest_current_production_full_{timestamp}.csv"
        df_trades.to_csv(output_file, index=False)
        print(f"\n💾 Results saved: {output_file.name}")

        # Recent period analysis (last 7 days)
        recent_cutoff = df['timestamp'].max() - timedelta(days=7)
        df_recent = df_trades[df_trades['entry_time'] >= recent_cutoff]

        if len(df_recent) > 0:
            recent_return = df_recent['pnl_usd'].sum()
            recent_wins = len(df_recent[df_recent['pnl_usd'] > 0])
            recent_losses = len(df_recent[df_recent['pnl_usd'] <= 0])
            recent_wr = (recent_wins / len(df_recent)) * 100

            print("\n" + "=" * 80)
            print("📊 최근 7일 성과 (프로덕션 손실 기간)")
            print("=" * 80)
            print(f"   거래: {len(df_recent)}")
            print(f"   수익: ${recent_return:+.2f}")
            print(f"   승률: {recent_wr:.1f}%")
            print(f"   승리/패배: {recent_wins}W / {recent_losses}L")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
