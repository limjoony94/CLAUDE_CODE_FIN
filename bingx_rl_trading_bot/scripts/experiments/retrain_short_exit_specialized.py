"""
Retrain SHORT Exit Model with Specialization
===========================================

SHORT 특화 exit 모델 재학습:
1. SHORT 특화 labeling parameters (분석 기반)
2. Reversal detection features 추가
3. SHORT 데이터로만 학습

Expected Improvement:
- Current: 61.9% late exits, -2.27% opportunity cost
- Target: <30% late exits, <-0.5% opportunity cost

Author: Claude Code
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.reversal_detection_features import add_all_reversal_features
from src.labeling.improved_exit_labeling import ImprovedExitLabeling

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Opportunity Gating Parameters
ENTRY_THRESHOLD_LONG = 0.65
ENTRY_THRESHOLD_SHORT = 0.70
GATE_THRESHOLD = 0.001
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# SHORT-Specific Labeling Parameters (from analysis)
SHORT_LEAD_TIME_MIN = 5     # 25 min (vs 3 for LONG)
SHORT_LEAD_TIME_MAX = 78    # 6.5 hours (vs 24 for LONG)
SHORT_PROFIT_THRESHOLD = 0.002  # 0.2% (vs 0.3% for LONG)
SHORT_PEAK_THRESHOLD = 0.001    # 0.1% (vs 0.2% for LONG)


def load_entry_models():
    """Load ENTRY models for trade simulation"""
    print("Loading ENTRY models...")

    # LONG Entry
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl", 'rb') as f:
        long_entry_model = pickle.load(f)
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl", 'rb') as f:
        long_entry_scaler = pickle.load(f)
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt", 'r') as f:
        long_entry_features = [line.strip() for line in f]

    # SHORT Entry
    with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl", 'rb') as f:
        short_entry_model = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl", 'rb') as f:
        short_entry_scaler = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt", 'r') as f:
        short_entry_features = [line.strip() for line in f if line.strip()]

    print("✅ Models loaded")

    return {
        'long_entry_model': long_entry_model,
        'long_entry_scaler': long_entry_scaler,
        'long_entry_features': long_entry_features,
        'short_entry_model': short_entry_model,
        'short_entry_scaler': short_entry_scaler,
        'short_entry_features': short_entry_features
    }


def simulate_short_trades(df, long_model, long_scaler, long_features,
                         short_model, short_scaler, short_features):
    """Simulate SHORT trades with opportunity gating"""
    print("\nSimulating SHORT trades for labeling...")

    trades = []

    for i in range(len(df) - 96):
        # Get LONG probability
        long_row = df[long_features].iloc[i:i+1].values
        if np.isnan(long_row).any():
            continue
        long_row_scaled = long_scaler.transform(long_row)
        long_prob = long_model.predict_proba(long_row_scaled)[0][1]

        # Get SHORT probability
        short_row = df[short_features].iloc[i:i+1].values
        if np.isnan(short_row).any():
            continue
        short_row_scaled = short_scaler.transform(short_row)
        short_prob = short_model.predict_proba(short_row_scaled)[0][1]

        # SHORT entry: threshold + gating
        if short_prob >= ENTRY_THRESHOLD_SHORT:
            # Calculate expected values
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN

            # Gate check
            opportunity_cost = short_ev - long_ev
            if opportunity_cost > GATE_THRESHOLD:
                trades.append({
                    'entry_idx': i,
                    'entry_price': df['close'].iloc[i],
                    'entry_prob': short_prob,
                    'opportunity_cost': opportunity_cost
                })

    print(f"✅ Simulated {len(trades):,} SHORT trades")
    return trades


def prepare_exit_features(df):
    """
    Prepare EXIT features with enhanced market context
    (Reuse from retrain_exit_models_opportunity_gating.py)
    """
    print("\nCalculating enhanced market context features...")

    # Volume features
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    # Price momentum features
    if 'sma_50' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['sma_50']) / df['sma_50']
    else:
        ma_20 = df['close'].rolling(20).mean()
        df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20

    if 'sma_200' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['sma_200']) / df['sma_200']
    else:
        ma_50 = df['close'].rolling(50).mean()
        df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50

    # Volatility features
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(100).median()).astype(float)

    # RSI dynamics
    df['rsi_slope'] = df['rsi'].diff(3) / 3
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)

    # MACD dynamics
    df['macd_histogram_slope'] = df['macd_hist'].diff(3) / 3 if 'macd_hist' in df.columns else 0
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)

    # Price patterns
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    df['price_acceleration'] = df['close'].diff().diff()

    # Support/Resistance proximity
    if 'support_level' in df.columns and 'resistance_level' in df.columns:
        df['near_resistance'] = (df['close'] > df['resistance_level'] * 0.98).astype(float)
        df['near_support'] = (df['close'] < df['support_level'] * 1.02).astype(float)
    else:
        df['near_resistance'] = 0
        df['near_support'] = 0

    # Bollinger Band position
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    else:
        df['bb_position'] = 0.5

    # Clean NaN
    df = df.ffill().bfill()

    print(f"✅ Enhanced features calculated")

    return df


def train_short_exit_model(df, labels, model_name="xgboost_short_exit_specialized"):
    """
    Train SHORT-specialized EXIT model

    Uses:
    - All standard technical features
    - Enhanced market context features
    - NEW: Reversal detection features
    """
    print(f"\n{'='*80}")
    print(f"Training SHORT-Specialized EXIT Model")
    print(f"{'='*80}")

    # Define features
    standard_features = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'atr', 'volume_sma_ratio',
        'ema_12', 'ema_26', 'sma_50', 'sma_200',
        'stoch_k', 'stoch_d',
        'adx', 'plus_di', 'minus_di',
        'cci', 'roc', 'williams_r',
        'obv', 'cmf', 'mfi',
        'vwap', 'typical_price',
        'recent_high', 'recent_low',
        'support_level', 'resistance_level',
        'trend_strength', 'volatility_regime',
        'volume_surge', 'price_acceleration',
        'momentum_quality', 'breakout_signal',
        'sell_signal_strength', 'sell_momentum_score'
    ]

    # Enhanced market context features
    context_features = [
        'volume_ratio', 'price_vs_ma20', 'price_vs_ma50',
        'volatility_20', 'rsi_slope', 'rsi_overbought', 'rsi_oversold',
        'macd_histogram_slope', 'macd_crossover', 'macd_crossunder',
        'higher_high', 'lower_low',
        'near_resistance', 'near_support', 'bb_position'
    ]

    # NEW: Reversal detection features
    reversal_features = [
        'price_momentum_reversal',
        'volume_spike_on_bounce',
        'rsi_divergence_real',
        'support_bounce',
        'consecutive_green_candles',
        'buy_pressure',
        'short_liquidation_cascade',
        'reversal_composite'
    ]

    all_features = standard_features + context_features + reversal_features

    # Filter features that exist in df
    available_features = [f for f in all_features if f in df.columns]

    print(f"\nUsing {len(available_features)} features:")
    print(f"  Standard technical: {len([f for f in available_features if f in standard_features])}")
    print(f"  Market context: {len([f for f in available_features if f in context_features])}")
    print(f"  Reversal detection: {len([f for f in available_features if f in reversal_features])}")

    # Prepare X, y
    X = df[available_features].values
    y = labels

    # Remove NaN rows
    valid_idx = ~np.isnan(X).any(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]

    print(f"\nDataset:")
    print(f"  Total samples: {len(y):,}")
    print(f"  Positive (exit): {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")
    print(f"  Negative (hold): {(len(y)-y.sum()):,} ({(len(y)-y.sum())/len(y)*100:.2f}%)")

    # Check positive rate
    positive_rate = y.sum() / len(y)
    if positive_rate < 0.05:
        print(f"⚠️ WARNING: Positive rate very low ({positive_rate*100:.2f}%)")
    elif positive_rate > 0.30:
        print(f"⚠️ WARNING: Positive rate high ({positive_rate*100:.2f}%)")
    else:
        print(f"✅ Positive rate reasonable ({positive_rate*100:.2f}%)")

    # Scale features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    # Calculate class weight
    neg_count = len(y) - y.sum()
    pos_count = y.sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    print(f"\nClass weight: {scale_pos_weight:.2f}")

    # Train XGBoost model
    print(f"\nTraining XGBoost model...")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    # Time series cross-validation
    print(f"Running 5-fold time series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='precision', n_jobs=-1)

    print(f"Cross-validation precision: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Folds: {', '.join([f'{s:.4f}' for s in cv_scores])}")

    # Train final model
    print(f"\nTraining final model on all data...")
    model.fit(X_scaled, y)

    # Evaluate
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\n{'='*80}")
    print(f"Training Metrics:")
    print(f"{'='*80}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Probability distribution
    print(f"\nProbability Distribution:")
    print(f"  Mean: {y_proba.mean():.4f}")
    print(f"  Median: {np.median(y_proba):.4f}")
    print(f"  Std: {y_proba.std():.4f}")

    # Feature importance (top 20)
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print(f"\nTop 20 Feature Importances:")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    # Check reversal features in top 20
    reversal_in_top20 = importance_df.head(20)['feature'].isin(reversal_features).sum()
    print(f"\n  Reversal features in top 20: {reversal_in_top20}/8")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"{model_name}_{timestamp}.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_{timestamp}_scaler.pkl"
    features_path = MODELS_DIR / f"{model_name}_{timestamp}_features.txt"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(features_path, 'w') as f:
        f.write('\n'.join(available_features))

    print(f"\n✅ Model saved:")
    print(f"   {model_path}")
    print(f"   {scaler_path}")
    print(f"   {features_path}")

    return model, scaler, available_features, {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_precision_mean': cv_scores.mean(),
        'cv_precision_std': cv_scores.std(),
        'prob_mean': y_proba.mean(),
        'positive_rate': positive_rate,
        'feature_importance': importance_df
    }


def main():
    print("="*80)
    print("Retrain SHORT Exit Model with Specialization")
    print("="*80)

    print(f"\nSHORT-Specific Configuration:")
    print(f"  Entry Threshold: {ENTRY_THRESHOLD_SHORT}")
    print(f"  Gate Threshold: {GATE_THRESHOLD}")
    print(f"\nLabeling Parameters (vs LONG):")
    print(f"  Lead time min: {SHORT_LEAD_TIME_MIN} (vs 3)")
    print(f"  Lead time max: {SHORT_LEAD_TIME_MAX} (vs 24)")
    print(f"  Profit threshold: {SHORT_PROFIT_THRESHOLD} (vs 0.003)")
    print(f"  Peak threshold: {SHORT_PEAK_THRESHOLD} (vs 0.002)")

    # Load data
    print("\nLoading data...")
    df_raw = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")

    # Calculate ALL features
    print("Calculating ALL features...")
    df = calculate_all_features(df_raw)
    print(f"✅ {len(df):,} candles loaded")

    # Add enhanced exit features
    df = prepare_exit_features(df)

    # Add reversal detection features (NEW!)
    df = add_all_reversal_features(df)

    # Load ENTRY models
    entry_models = load_entry_models()

    # Simulate SHORT trades
    short_trades = simulate_short_trades(
        df,
        entry_models['long_entry_model'],
        entry_models['long_entry_scaler'],
        entry_models['long_entry_features'],
        entry_models['short_entry_model'],
        entry_models['short_entry_scaler'],
        entry_models['short_entry_features']
    )

    # Create SHORT-specialized labels
    print(f"\n{'='*80}")
    print("Creating SHORT-Specialized Labels")
    print(f"{'='*80}")

    labeler = ImprovedExitLabeling(
        lead_time_min=SHORT_LEAD_TIME_MIN,
        lead_time_max=SHORT_LEAD_TIME_MAX,
        profit_threshold=SHORT_PROFIT_THRESHOLD,
        peak_threshold=SHORT_PEAK_THRESHOLD,
        momentum_rsi_high=55.0,
        momentum_rsi_low=45.0,
        relative_tolerance=0.001
    )

    print(f"\nLabeling with SHORT-specific parameters...")
    short_exit_labels = labeler.create_short_exit_labels(df, short_trades)

    # Validate labels
    short_stats = labeler.validate_labels(short_exit_labels, df)
    print(f"\nSHORT EXIT Label Statistics:")
    print(f"  Total candles: {short_stats['total_candles']:,}")
    print(f"  Positive labels: {short_stats['positive_labels']:,}")
    print(f"  Positive rate: {short_stats['positive_rate']*100:.2f}%")
    print(f"  Average spacing: {short_stats['avg_spacing']:.1f} candles")

    # Train SHORT-specialized model
    model, scaler, features, metrics = train_short_exit_model(df, short_exit_labels)

    # Summary
    print(f"\n" + "="*80)
    print("RETRAINING COMPLETE")
    print("="*80)

    print(f"\nSHORT-Specialized Exit Model:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  CV Precision: {metrics['cv_precision_mean']:.4f} ± {metrics['cv_precision_std']:.4f}")
    print(f"  Prob Mean: {metrics['prob_mean']:.4f}")
    print(f"  Positive Rate: {metrics['positive_rate']*100:.2f}%")
    print(f"  Features: {len(features)} (including 8 reversal features)")

    print(f"\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print(f"1. Backtest new model vs current model (threshold 0.72)")
    print(f"2. Compare metrics:")
    print(f"   - Opportunity cost: target <-0.5% (vs -2.27% original)")
    print(f"   - Late exits: target <30% (vs 61.9% original)")
    print(f"   - SHORT win rate: target >79% (vs 79.3% with 0.72)")
    print(f"3. If better: Deploy to production")
    print(f"4. If not: Keep threshold 0.72, iterate on features")
    print(f"\n" + "="*80)


if __name__ == "__main__":
    main()
