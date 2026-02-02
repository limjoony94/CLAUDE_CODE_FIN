"""
SHORT Model - LONG Model Inversion (Approach #12)

최종 비판적 통찰:
- Approach #1 (label inverse): 46% ← 가장 성공적!
- Approach #11 (threshold inverse): 24.4% ← 실패

Why Approach #1 worked better?
- LONG 모델을 inversion 했음
- LONG 모델이 69.1%로 검증되어 있으므로
- 그 inverse도 신뢰할 수 있음

New Strategy:
- 검증된 Phase 4 LONG 모델 사용 (69.1% win rate)
- LONG probability inversion:
  * LONG prob >= 0.7 → LONG 진입 (original)
  * LONG prob <= 0.3 → SHORT 진입 (inverse)
- 예상: 45-55% win rate

This is the FINAL attempt based on best performing logic.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.market_regime_filter import MarketRegimeFilter

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"


def load_long_model():
    """Load the verified Phase 4 LONG model (69.1% win rate)"""
    print("="*80)
    print("Loading Verified LONG Model (Phase 4)")
    print("="*80)

    # Find Phase 4 model
    model_files = list(MODELS_DIR.glob("xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"))

    if not model_files:
        print("❌ Phase 4 model not found!")
        print("Available models:")
        for f in MODELS_DIR.glob("*.pkl"):
            print(f"  - {f.name}")
        return None, None

    model_path = model_files[0]
    print(f"✅ Found model: {model_path.name}")

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"✅ Model loaded successfully")
    print(f"✅ This model achieved 69.1% LONG win rate")

    # Get feature names
    feature_names = [
        'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
        'ema_21', 'ema_50', 'atr', 'volume_ratio', 'price_distance_ema',
        # Advanced features from Phase 4
        'atr_ratio', 'bb_width', 'true_range', 'high_low_range',
        'stochrsi', 'willr', 'cci', 'cmo', 'uo', 'roc', 'mfi', 'tsi', 'kst',
        'adx', 'di_plus', 'di_minus', 'aroon_up', 'aroon_down', 'vi',
        'obv', 'cmf',
        'macd_histogram', 'bb_position', 'price_momentum'
    ]

    return model, feature_names


def create_short_labels(df, lookahead=3, threshold=0.0):
    """
    Create labels for SHORT validation

    Label 1: Price will fall (SHORT opportunity)
    Label 0: Price will not fall (NO SHORT)
    """
    labels = []

    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(0)
            continue

        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+lookahead]
        min_future = future_prices.min()
        decrease_pct = (current_price - min_future) / current_price

        if decrease_pct >= 0.002:  # 0.2% decrease = SHORT success
            labels.append(1)
        else:
            labels.append(0)

    return np.array(labels)


def load_and_prepare_data(feature_names):
    """Load data and calculate Phase 4 features"""
    print("\n" + "="*80)
    print("Loading Data with Phase 4 Features")
    print("="*80)

    # Load candles
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"Loaded {len(df)} candles")

    # Generate ALL Phase 4 features
    import ta

    # Baseline features (must create first)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()

    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)

    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    df['price_distance_ema'] = (df['close'] - df['ema_21']) / df['ema_21']

    # Advanced Phase 4 features
    df['atr_ratio'] = df['atr'] / df['close']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    df['true_range'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['high_low_range'] = (df['high'] - df['low']) / df['low']

    # Momentum indicators
    df['stochrsi'] = ta.momentum.stochrsi(df['close'], window=14)
    df['willr'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
    df['cmo'] = ta.momentum.roc(df['close'], window=14)
    df['uo'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'])
    df['roc'] = ta.momentum.roc(df['close'], window=12)
    df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
    df['tsi'] = ta.momentum.tsi(df['close'])
    df['kst'] = ta.trend.kst(df['close'])

    # Trend indicators
    adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_indicator.adx()
    df['di_plus'] = adx_indicator.adx_pos()
    df['di_minus'] = adx_indicator.adx_neg()

    aroon = ta.trend.AroonIndicator(df['high'], df['low'], window=25)
    df['aroon_up'] = aroon.aroon_up()
    df['aroon_down'] = aroon.aroon_down()

    vortex = ta.trend.VortexIndicator(df['high'], df['low'], df['close'], window=14)
    df['vi'] = vortex.vortex_indicator_pos()

    # Volume indicators
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])

    # Additional features
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['price_momentum'] = df['close'].pct_change(5)

    # Regime filter
    regime_filter = MarketRegimeFilter()
    df = regime_filter.get_regime_features(df)

    # Remove NaN
    df = df.ffill().dropna()
    print(f"After processing: {len(df)} candles")

    # Verify all features exist
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")

    available_features = [f for f in feature_names if f in df.columns]
    print(f"\nFeatures available: {len(available_features)} / {len(feature_names)}")

    # Create SHORT labels for validation
    labels = create_short_labels(df, lookahead=3, threshold=0.002)

    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nSHORT Label Distribution:")
    print(f"  NO SHORT (0): {counts[0]} ({counts[0]/len(labels)*100:.1f}%)")
    print(f"  SHORT (1): {counts[1]} ({counts[1]/len(labels)*100:.1f}%)")

    X = df[available_features].values
    y = labels

    return X, y, available_features, df


def backtest_short_inverse(model, X, y, df):
    """Backtest SHORT using LONG model inversion"""
    print("\n" + "="*80)
    print("Backtesting SHORT with LONG Model Inversion")
    print("="*80)
    print("Strategy: LONG prob <= 0.3 → SHORT entry (model says 'NOT going up')")

    tscv = TimeSeriesSplit(n_splits=5)

    all_trades = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_val = X[val_idx]
        y_val = y[val_idx]

        # Get LONG probabilities
        y_prob_long = model.predict_proba(X_val)[:, 1]

        # INVERSE threshold: LOW LONG prob = SHORT signal
        short_threshold = 0.3

        for i, idx in enumerate(val_idx):
            # Inverse logic: LONG prob <= 0.3 → Enter SHORT
            long_prob = y_prob_long[i]
            model_signal = long_prob <= short_threshold
            regime_allowed = df['short_allowed'].iloc[idx] == 1

            if model_signal and regime_allowed:
                actual_label = y_val[i]

                trade = {
                    'fold': fold,
                    'index': idx,
                    'long_probability': long_prob,  # LOW = SHORT signal
                    'regime': df['regime_trend'].iloc[idx],
                    'predicted': 1,  # We predict SHORT will succeed
                    'actual': actual_label,
                    'correct': (1 == actual_label)
                }
                all_trades.append(trade)

        # Fold results
        fold_trades = [t for t in all_trades if t['fold'] == fold]
        if len(fold_trades) > 0:
            fold_correct = sum(t['correct'] for t in fold_trades)
            fold_total = len(fold_trades)
            fold_win_rate = fold_correct / fold_total
            print(f"Fold {fold}: {fold_total} trades, {fold_correct} correct ({fold_win_rate*100:.1f}%)")

    # Overall results
    print(f"\n{'='*80}")
    print("SHORT Inverse Results (from LONG Model)")
    print(f"{'='*80}")

    if len(all_trades) == 0:
        print("\n❌ No trades generated")
        return 0.0, all_trades

    total_trades = len(all_trades)
    correct_trades = sum(t['correct'] for t in all_trades)
    overall_win_rate = correct_trades / total_trades

    print(f"\nOverall Results:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Correct: {correct_trades}")
    print(f"  Incorrect: {total_trades - correct_trades}")
    print(f"  Win Rate: {overall_win_rate*100:.1f}%")

    # LONG probability distribution for SHORT signals
    long_probs = [t['long_probability'] for t in all_trades]
    print(f"\nLONG Probability Distribution (for SHORT signals):")
    print(f"  Mean: {np.mean(long_probs):.3f}")
    print(f"  Std: {np.std(long_probs):.3f}")
    print(f"  Min: {np.min(long_probs):.3f}")
    print(f"  Max: {np.max(long_probs):.3f}")

    # Regime analysis
    trades_df = pd.DataFrame(all_trades)
    print(f"\nTrades by Regime:")
    for regime in trades_df['regime'].unique():
        regime_trades = trades_df[trades_df['regime'] == regime]
        regime_correct = regime_trades['correct'].sum()
        regime_total = len(regime_trades)
        regime_wr = regime_correct / regime_total * 100 if regime_total > 0 else 0
        print(f"  {regime}: {regime_total} trades, {regime_correct} correct ({regime_wr:.1f}%)")

    return overall_win_rate, all_trades


def main():
    """Main pipeline"""
    print("="*80)
    print("SHORT Model - LONG Model Inversion (Approach #12)")
    print("="*80)
    print("Final Strategy Based on Best Logic:")
    print("  - Use verified Phase 4 LONG model (69.1% win rate)")
    print("  - Invert predictions: LOW LONG prob = SHORT signal")
    print("  - Approach #1 (label inverse) achieved 46%")
    print("  - This should perform similarly or better")
    print("="*80)

    # Load LONG model
    model, feature_names = load_long_model()
    if model is None:
        return 0.0, []

    # Load data
    X, y, feature_columns, df = load_and_prepare_data(feature_names)

    # Backtest SHORT inverse
    win_rate, trades = backtest_short_inverse(model, X, y, df)

    # Decision
    print("\n" + "="*80)
    print("Final Decision - LONG Model Inversion")
    print("="*80)

    if win_rate >= 0.60:
        print(f"✅ SUCCESS! SHORT win rate {win_rate*100:.1f}% >= 60%")
        print(f"✅ LONG model inversion WORKS!")

    elif win_rate >= 0.45:
        print(f"🔄 STRONG IMPROVEMENT: SHORT win rate {win_rate*100:.1f}% (45-60%)")
        print(f"🔄 Close to target! Consider threshold tuning")

    elif win_rate >= 0.30:
        print(f"⚠️ MODERATE IMPROVEMENT: SHORT win rate {win_rate*100:.1f}% (30-45%)")
        print(f"⚠️ Better than baseline but insufficient")

    else:
        print(f"❌ INSUFFICIENT: SHORT win rate {win_rate*100:.1f}%")

    print(f"\nComplete Progress Summary (All 12 Approaches):")
    print(f"  #1  2-Class Inverse: 46.0% ✅ Best standalone")
    print(f"  #2  3-Class Unbalanced: 0.0%")
    print(f"  #3  3-Class Balanced: 36.4%")
    print(f"  #4  Optuna (100 trials): 22-25%")
    print(f"  #5  V2 Baseline: 26.0%")
    print(f"  #6  V3 Strict: 9.7%")
    print(f"  #7  V4 Ensemble: 20.3%")
    print(f"  #8  V5 SMOTE: 0.0%")
    print(f"  #9  LSTM: 17.3%")
    print(f"  #10 Funding Rate: 22.4%")
    print(f"  #11 Inverse Threshold: 24.4%")
    print(f"  #12 LONG Model Inverse: {win_rate*100:.1f}%")

    return win_rate, trades


if __name__ == "__main__":
    win_rate, trades = main()
