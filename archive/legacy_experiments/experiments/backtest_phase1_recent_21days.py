"""
Phase 1 Recent 21-Day Validation Backtest
==========================================
Tests Phase 1 reduced-feature models (LONG 80, SHORT 79) on most recent 21 days.
Period: Oct 6 - Oct 26, 2025 (within Holdout period)

This extends the 14-day validation for better statistical reliability.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.production.dynamic_position_sizing import DynamicPositionSizer

# Configuration
TEST_DAYS = 21
INITIAL_BALANCE = 10000
LEVERAGE = 4
ENTRY_THRESHOLD = 0.75
ML_EXIT_THRESHOLD = 0.75
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120

DATA_DIR = project_root / "data" / "features"
MODELS_DIR = project_root / "models"
RESULTS_DIR = project_root / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Phase 1 Recent 21-Day Validation Backtest")
print("=" * 80)
print(f"Period: Last {TEST_DAYS} days (within Holdout)")
print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"Leverage: {LEVERAGE}x")
print(f"Entry Threshold: {ENTRY_THRESHOLD}")
print(f"ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"Stop Loss: {EMERGENCY_STOP_LOSS*100:.1f}% balance")
print(f"Max Hold: {EMERGENCY_MAX_HOLD} candles")
print()

# Initialize position sizer
position_sizer = DynamicPositionSizer()

# Load Full Features Dataset
print("-" * 80)
print("Loading Data and Selecting Test Period")
print("-" * 80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Select ONLY last 21 days (within Holdout period)
test_candles = TEST_DAYS * 24 * 12
test_start_idx = len(df) - test_candles

df = df.iloc[test_start_idx:].copy().reset_index(drop=True)
print(f"\n✅ Test Period Selected (Last {TEST_DAYS} days):")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"   Candles: {len(df):,} ({TEST_DAYS} days)")
print()

# Prepare Exit Features
def prepare_exit_features(df):
    """Prepare EXIT features with enhanced market context"""
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    if 'ma_50' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['ma_50']) / df['ma_50']
    else:
        ma_20 = df['close'].rolling(20).mean()
        df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20

    if 'ma_200' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['ma_200']) / df['ma_200']
    else:
        ma_50 = df['close'].rolling(50).mean()
        df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50

    df['volatility_20'] = df['close'].pct_change().rolling(20).std()

    df['rsi_slope'] = df['rsi'].diff()
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)

    rsi_high = df['rsi'].rolling(20).max()
    rsi_low = df['rsi'].rolling(20).min()
    df['rsi_divergence'] = np.where(
        (df['close'] > df['close'].shift(1)) & (df['rsi'] < df['rsi'].shift(1)),
        1.0,
        np.where(
            (df['close'] < df['close'].shift(1)) & (df['rsi'] > df['rsi'].shift(1)),
            -1.0,
            0.0
        )
    )

    df['macd_histogram_slope'] = df['macd_histogram'].diff()
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)

    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        bb_width = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / bb_width
    else:
        df['bb_position'] = 0.5

    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)

    support = df['low'].rolling(20).min()
    df['near_support'] = ((df['close'] - support) / support < 0.01).astype(float)

    return df

df = prepare_exit_features(df)
print(f"✅ Exit features prepared")
print()

# Load Phase 1 models
print("Loading Phase 1 models (timestamp: 20251029_050448)...")
long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_walkforward_reduced_phase1_20251029_050448.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_walkforward_reduced_phase1_20251029_050448_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_walkforward_reduced_phase1_20251029_050448_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f if line.strip()]

short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_entry_walkforward_reduced_phase1_20251029_050448.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_walkforward_reduced_phase1_20251029_050448_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_walkforward_reduced_phase1_20251029_050448_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f if line.strip()]

long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251029_055201.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251029_055201_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_threshold_075_20251029_055201_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f if line.strip()]

short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251029_055201.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251029_055201_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251029_055201_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f if line.strip()]

print(f"✅ LONG Entry: {len(long_entry_features)} features")
print(f"✅ SHORT Entry: {len(short_entry_features)} features")
print(f"✅ LONG Exit: {len(long_exit_features)} features")
print(f"✅ SHORT Exit: {len(short_exit_features)} features")
print()

df = calculate_all_features(df)
print(f"✅ Calculated features: {len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} features")
print()

# Backtest simulation
balance = INITIAL_BALANCE
position = None
trades = []
candle_count = 0

print("Running backtest simulation...")
print("-" * 80)

for idx in range(len(df)):
    candle = df.iloc[idx]
    candle_count += 1

    # Check exit if in position
    if position:
        position['hold_time'] += 1
        current_price = candle['close']

        # Calculate P&L
        if position['side'] == 'LONG':
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

        leveraged_pnl_pct = pnl_pct * LEVERAGE

        # Check emergency exit
        exit_reason = None
        if leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
            exit_reason = 'STOP_LOSS'
        elif position['hold_time'] >= EMERGENCY_MAX_HOLD:
            exit_reason = 'MAX_HOLD'
        else:
            # Check ML exit
            exit_features = short_exit_features if position['side'] == 'SHORT' else long_exit_features
            exit_model = short_exit_model if position['side'] == 'SHORT' else long_exit_model
            exit_scaler = short_exit_scaler if position['side'] == 'SHORT' else long_exit_scaler

            if all(f in df.columns for f in exit_features):
                exit_feat = candle[exit_features].values.reshape(1, -1)
                exit_feat_scaled = exit_scaler.transform(exit_feat)
                exit_prob = exit_model.predict_proba(exit_feat_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    exit_reason = 'ML_EXIT'

        # Execute exit
        if exit_reason:
            pnl = balance * position['position_size_pct'] * leveraged_pnl_pct
            balance += pnl

            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': candle['timestamp'],
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'position_size_pct': position['position_size_pct'],
                'hold_time': position['hold_time'],
                'pnl': pnl,
                'pnl_pct': leveraged_pnl_pct * 100,
                'exit_reason': exit_reason,
                'balance': balance
            })

            position = None

    # Check entry if no position
    if not position:
        # Check features exist
        long_ready = all(f in df.columns for f in long_entry_features)
        short_ready = all(f in df.columns for f in short_entry_features)

        if long_ready and short_ready:
            # Calculate entry signals
            long_feat = candle[long_entry_features].values.reshape(1, -1)
            long_feat_scaled = long_entry_scaler.transform(long_feat)
            long_prob = long_entry_model.predict_proba(long_feat_scaled)[0][1]

            short_feat = candle[short_entry_features].values.reshape(1, -1)
            short_feat_scaled = short_entry_scaler.transform(short_feat)
            short_prob = short_entry_model.predict_proba(short_feat_scaled)[0][1]

            # Check LONG entry
            if long_prob >= ENTRY_THRESHOLD:
                position_size_pct = position_sizer.calculate_position_size(
                    long_prob, long_prob, 0.20, 0.95, 0.65, 0.85
                )

                position = {
                    'entry_time': candle['timestamp'],
                    'side': 'LONG',
                    'entry_price': candle['close'],
                    'position_size_pct': position_size_pct,
                    'hold_time': 0
                }

            # Check SHORT entry (Opportunity Gating)
            elif short_prob >= ENTRY_THRESHOLD:
                long_ev = long_prob * 0.0041
                short_ev = short_prob * 0.0047
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > 0.001:
                    position_size_pct = position_sizer.calculate_position_size(
                        short_prob, short_prob, 0.20, 0.95, 0.65, 0.85
                    )

                    position = {
                        'entry_time': candle['timestamp'],
                        'side': 'SHORT',
                        'entry_price': candle['close'],
                        'position_size_pct': position_size_pct,
                        'hold_time': 0
                    }

# Force close final position
if position:
    current_price = df.iloc[-1]['close']
    if position['side'] == 'LONG':
        pnl_pct = (current_price - position['entry_price']) / position['entry_price']
    else:
        pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

    leveraged_pnl_pct = pnl_pct * LEVERAGE
    pnl = balance * position['position_size_pct'] * leveraged_pnl_pct
    balance += pnl

    trades.append({
        'entry_time': position['entry_time'],
        'exit_time': df.iloc[-1]['timestamp'],
        'side': position['side'],
        'entry_price': position['entry_price'],
        'exit_price': current_price,
        'position_size_pct': position['position_size_pct'],
        'hold_time': position['hold_time'],
        'pnl': pnl,
        'pnl_pct': leveraged_pnl_pct * 100,
        'exit_reason': 'FORCE_CLOSE',
        'balance': balance
    })

# Results
print("=" * 80)
print("BACKTEST RESULTS - Phase 1 Recent 21-Day Validation")
print("=" * 80)

total_return = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
num_trades = len(trades)
test_days_actual = (df['timestamp'].max() - df['timestamp'].min()).days + 1
trade_freq = num_trades / test_days_actual if test_days_actual > 0 else 0

print(f"\n📊 Performance Summary:")
print(f"   Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"   Final Balance: ${balance:,.2f}")
print(f"   Total Return: {total_return:+.2f}%")
print(f"   Total Trades: {num_trades}")
print(f"   Test Period: {test_days_actual} days")
print(f"   Trade Frequency: {trade_freq:.2f} trades/day")

if trades:
    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    win_rate = (len(wins) / len(trades_df)) * 100
    avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0

    long_trades = trades_df[trades_df['side'] == 'LONG']
    short_trades = trades_df[trades_df['side'] == 'SHORT']
    long_pct = (len(long_trades) / len(trades_df)) * 100
    short_pct = (len(short_trades) / len(trades_df)) * 100

    ml_exits = len(trades_df[trades_df['exit_reason'] == 'ML_EXIT'])
    sl_exits = len(trades_df[trades_df['exit_reason'] == 'STOP_LOSS'])
    mh_exits = len(trades_df[trades_df['exit_reason'] == 'MAX_HOLD'])

    print(f"\n📈 Trade Statistics:")
    print(f"   Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"   Average Win: {avg_win:+.2f}%")
    print(f"   Average Loss: {avg_loss:+.2f}%")
    print(f"   Average Hold: {trades_df['hold_time'].mean():.1f} candles")

    print(f"\n🎯 Direction Split:")
    print(f"   LONG: {long_pct:.1f}% ({len(long_trades)} trades)")
    print(f"   SHORT: {short_pct:.1f}% ({len(short_trades)} trades)")

    print(f"\n🚪 Exit Distribution:")
    print(f"   ML Exit: {ml_exits} ({ml_exits/len(trades_df)*100:.1f}%)")
    print(f"   Stop Loss: {sl_exits} ({sl_exits/len(trades_df)*100:.1f}%)")
    print(f"   Max Hold: {mh_exits} ({mh_exits/len(trades_df)*100:.1f}%)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"backtest_phase1_recent_21days_{timestamp}.csv"
    trades_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved: {output_file.name}")

print("\n" + "=" * 80)
print("VALIDATION CRITERIA ASSESSMENT")
print("=" * 80)

criteria_passed = 0
criteria_total = 4

print("\n✓/✗ Success Criteria (Phase 1):")
print(f"   1. Return > 0%: {total_return:+.2f}% {'✅ PASS' if total_return > 0 else '❌ FAIL'}")
if total_return > 0:
    criteria_passed += 1

print(f"   2. Win Rate > 60%: {win_rate:.1f}% {'✅ PASS' if win_rate > 60 else '❌ FAIL'}")
if win_rate > 60:
    criteria_passed += 1

print(f"   3. Trade Frequency ≥ 1.5/day: {trade_freq:.2f}/day {'✅ PASS' if trade_freq >= 1.5 else '⚠️ FAIL'}")
if trade_freq >= 1.5:
    criteria_passed += 1

print(f"   4. Total Trades ≥ 20: {num_trades} {'✅ PASS' if num_trades >= 20 else '⚠️ FAIL'}")
if num_trades >= 20:
    criteria_passed += 1

print(f"\n📊 Criteria Passed: {criteria_passed}/{criteria_total}")

if criteria_passed == 4:
    print("   Status: ✅ EXCELLENT - All criteria passed")
elif criteria_passed >= 3:
    print("   Status: ✅ GOOD - Deploy with monitoring")
elif criteria_passed >= 2:
    print("   Status: ⚠️ BORDERLINE - Deploy with strict monitoring")
else:
    print("   Status: ❌ FAILED - Investigate before deployment")

print("\n" + "=" * 80)
print("MULTI-PERIOD COMPARISON")
print("=" * 80)

print("\n30-Day Holdout (Sep 26 - Oct 26):")
print("  Return: +40.24%")
print("  Win Rate: 79.7%")
print("  Trade Frequency: 1.97/day")
print("  Total Trades: 59")

print(f"\n21-Day Recent (Oct 6 - Oct 26):")
print(f"  Return: {total_return:+.2f}%")
print(f"  Win Rate: {win_rate:.1f}%")
print(f"  Trade Frequency: {trade_freq:.2f}/day")
print(f"  Total Trades: {num_trades}")

print("\n14-Day Recent (Oct 13 - Oct 26):")
print("  Return: +4.84%")
print("  Win Rate: 71.4%")
print("  Trade Frequency: 1.00/day")
print("  Total Trades: 14")

print("\n7-Day Recent (Oct 19 - Oct 26):")
print("  Return: -0.28%")
print("  Win Rate: 0.0%")
print("  Trade Frequency: 0.14/day")
print("  Total Trades: 1")

print("\n" + "=" * 80)
print("✅ 21-Day Validation Complete!")
print("=" * 80)
