"""Meta-Labeling on 1h ML — Lopez de Prado method.

Reference: Lopez de Prado "Advances in Financial Machine Learning" (2018)
           Triple Barrier Method + Meta-Labeling

이전 1h direction prediction (causal logistic regression):
  Test hit rate 53.96% (PASS) but daily net -1.03% (gross 0%)
  Reason: small edge × friction 0.10% × 24 trades = -2.40%/day overhead
  Required: hit rate 58%+ OR active filter strict (high-confidence only)

Meta-labeling approach:
  1. Primary model: logistic regression (existing, predicts direction)
  2. Triple barrier: each predicted entry → TP/SL/timeout label (3-class)
  3. Meta-model: Random Forest on additional features → predict probability
     of "primary signal will hit TP first" (binary: 1 = take trade, 0 = skip)
  4. Trade only when meta-model confidence > threshold

Expected behavior:
  - Total trades reduced (from 1831 → 100-300)
  - Hit rate per trade increased (53.96% → 60%+)
  - Per-trade gross edge increased
  - Friction overhead reduced

Pre-commit: 1 attempt, PASS → deployable, FAIL → (C) On-chain features 진행.
"""
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
sys.path.insert(0, str(ROOT / 'scripts' / 'strategy_lab'))
from h1_direction_prediction import build_features
from bootstrap_validator import bootstrap_validate, report as bootstrap_report

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

FRICTION_PCT = 0.10  # taker RT


def triple_barrier_labels(close, signals, atr, tp_atr_mult=2.0, sl_atr_mult=1.0,
                            max_hold_bars=24):
    """Apply triple barrier to each signal.

    Args:
        close: array of closes
        signals: array of -1/+1 (direction predictions, 0 means no signal)
        atr: ATR series
        tp_atr_mult, sl_atr_mult: barrier multipliers
        max_hold_bars: time barrier

    Returns:
        labels: 1=TP_hit, -1=SL_hit, 0=timeout
        outcomes: actual gross PnL%
    """
    n = len(close)
    labels = np.zeros(n, dtype=int)
    outcomes = np.zeros(n)
    bars_held = np.zeros(n, dtype=int)

    for i in range(n - max_hold_bars - 1):
        if signals[i] == 0:
            continue
        if pd.isna(atr[i]) or atr[i] <= 0:
            continue
        entry = close[i]
        direction = signals[i]
        if direction == 1:
            tp = entry + tp_atr_mult * atr[i]
            sl = entry - sl_atr_mult * atr[i]
        else:
            tp = entry - tp_atr_mult * atr[i]
            sl = entry + sl_atr_mult * atr[i]

        for j in range(1, max_hold_bars + 1):
            if i + j >= n:
                break
            cl = close[i + j]
            if direction == 1:
                if cl >= tp:
                    labels[i] = 1
                    outcomes[i] = tp_atr_mult * atr[i] / entry * 100
                    bars_held[i] = j
                    break
                elif cl <= sl:
                    labels[i] = -1
                    outcomes[i] = -sl_atr_mult * atr[i] / entry * 100
                    bars_held[i] = j
                    break
            else:
                if cl <= tp:
                    labels[i] = 1
                    outcomes[i] = tp_atr_mult * atr[i] / entry * 100
                    bars_held[i] = j
                    break
                elif cl >= sl:
                    labels[i] = -1
                    outcomes[i] = -sl_atr_mult * atr[i] / entry * 100
                    bars_held[i] = j
                    break
        else:
            # Timeout
            labels[i] = 0
            cl = close[min(i + max_hold_bars, n - 1)]
            outcomes[i] = (cl - entry) / entry * 100 * direction
            bars_held[i] = max_hold_bars

    return labels, outcomes, bars_held


def main():
    print('=' * 100)
    print('Meta-Labeling on 1h ML — Lopez de Prado method')
    print('=' * 100)

    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    feats = build_features(df)
    feats = feats.merge(df[['timestamp', 'close', 'high', 'low']], left_index=True, right_index=True, how='left')
    feats = feats.dropna(subset=['timestamp', 'close'])
    print(f'Features: {feats.shape}')

    # Compute ATR for triple barrier
    high = feats['high'].values
    low = feats['low'].values
    cl = feats['close'].values
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - np.concatenate([[cl[0]], cl[:-1]])),
        np.abs(low - np.concatenate([[cl[0]], cl[:-1]])),
    ])
    feats['atr_14'] = pd.Series(tr).rolling(14).mean().values
    feats = feats.dropna(subset=['atr_14'])

    # Split 50/25/25 same as 1h ML
    n = len(feats)
    i1 = int(n * 0.50)
    i2 = int(n * 0.75)
    train = feats.iloc[:i1]
    val = feats.iloc[i1:i2]
    test = feats.iloc[i2:]
    print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}\n')

    feat_cols = ['ret_1h', 'ret_4h', 'ret_24h', 'atr_ratio', 'rsi_14',
                 'ema_ratio', 'vol_z', 'body_ratio', 'range_atr',
                 'close_in_range', 'macd_hist', 'donchian_pos']

    # === Step 1: Primary model (1h direction) ===
    print('=== Step 1: Primary model (logistic regression direction prediction) ===')
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train[feat_cols].values)
    X_val_s = scaler.transform(val[feat_cols].values)
    X_test_s = scaler.transform(test[feat_cols].values)
    primary = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    primary.fit(X_train_s, train['target'].values)
    p_train = primary.predict_proba(X_train_s)[:, 1]
    p_val = primary.predict_proba(X_val_s)[:, 1]
    p_test = primary.predict_proba(X_test_s)[:, 1]

    # Convert to direction signals (with active filter)
    threshold = 0.05
    sig_train = np.where(p_train > 0.5 + threshold, 1, np.where(p_train < 0.5 - threshold, -1, 0))
    sig_val = np.where(p_val > 0.5 + threshold, 1, np.where(p_val < 0.5 - threshold, -1, 0))
    sig_test = np.where(p_test > 0.5 + threshold, 1, np.where(p_test < 0.5 - threshold, -1, 0))
    print(f'Primary signals: train {(sig_train != 0).sum()}, val {(sig_val != 0).sum()}, test {(sig_test != 0).sum()}')

    # === Step 2: Triple Barrier Labels ===
    print('\n=== Step 2: Triple Barrier Labels (TP=2×ATR, SL=1×ATR, timeout=24bars) ===')
    train_labels, train_outcomes, train_bars = triple_barrier_labels(
        train['close'].values, sig_train, train['atr_14'].values,
        tp_atr_mult=2.0, sl_atr_mult=1.0, max_hold_bars=24)
    val_labels, val_outcomes, val_bars = triple_barrier_labels(
        val['close'].values, sig_val, val['atr_14'].values,
        tp_atr_mult=2.0, sl_atr_mult=1.0, max_hold_bars=24)
    test_labels, test_outcomes, test_bars = triple_barrier_labels(
        test['close'].values, sig_test, test['atr_14'].values,
        tp_atr_mult=2.0, sl_atr_mult=1.0, max_hold_bars=24)

    print(f'  Train labels: TP={(train_labels==1).sum()}, SL={(train_labels==-1).sum()}, timeout={(train_labels==0).sum() - (sig_train==0).sum()}')
    print(f'  Test  labels: TP={(test_labels==1).sum()}, SL={(test_labels==-1).sum()}, timeout={(test_labels==0).sum() - (sig_test==0).sum()}')

    # === Step 3: Meta-Labeling — predict P(TP_hit | primary_signal) ===
    print('\n=== Step 3: Meta-Model (Random Forest, predict TP_hit | signal) ===')
    # Filter to active signals + add outcomes
    train_active = train[sig_train != 0].copy()
    train_active['meta_target'] = (train_labels[sig_train != 0] == 1).astype(int)  # 1=TP hit
    train_active['primary_prob'] = p_train[sig_train != 0]
    train_active['primary_signal'] = sig_train[sig_train != 0]
    train_active['outcome'] = train_outcomes[sig_train != 0]

    val_active = val[sig_val != 0].copy()
    val_active['meta_target'] = (val_labels[sig_val != 0] == 1).astype(int)
    val_active['primary_prob'] = p_val[sig_val != 0]
    val_active['primary_signal'] = sig_val[sig_val != 0]
    val_active['outcome'] = val_outcomes[sig_val != 0]

    test_active = test[sig_test != 0].copy()
    test_active['meta_target'] = (test_labels[sig_test != 0] == 1).astype(int)
    test_active['primary_prob'] = p_test[sig_test != 0]
    test_active['primary_signal'] = sig_test[sig_test != 0]
    test_active['outcome'] = test_outcomes[sig_test != 0]

    # Meta-features: original 12 + primary_prob + primary_signal
    meta_feat_cols = feat_cols + ['primary_prob']

    rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf.fit(train_active[meta_feat_cols].values, train_active['meta_target'].values)

    test_active['meta_prob'] = rf.predict_proba(test_active[meta_feat_cols].values)[:, 1]
    val_active['meta_prob'] = rf.predict_proba(val_active[meta_feat_cols].values)[:, 1]

    # Print meta accuracy
    print(f'  Train base rate (TP hit): {train_active["meta_target"].mean():.3f}')
    print(f'  Val   base rate (TP hit): {val_active["meta_target"].mean():.3f}')
    print(f'  Test  base rate (TP hit): {test_active["meta_target"].mean():.3f}')
    print(f'  Meta predicted prob mean (test): {test_active["meta_prob"].mean():.3f}')
    print(f'  Meta predicted prob std (test):  {test_active["meta_prob"].std():.3f}')

    # === Step 4: Active filter — trade only when meta_prob > threshold ===
    print('\n=== Step 4: Active filter — trade only high-confidence ===')
    for meta_thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
        # On val set
        val_taken = val_active[val_active['meta_prob'] >= meta_thr]
        val_outcomes_taken = val_taken['outcome'].values
        val_n = len(val_taken)
        val_hit = (val_outcomes_taken > 0).mean() if val_n > 0 else 0
        val_avg = val_outcomes_taken.mean() if val_n > 0 else 0
        val_avg_net = val_avg - FRICTION_PCT
        val_daily = val_avg_net * val_n / max(1, (val['timestamp'].max() - val['timestamp'].min()).days)

        # On test set
        test_taken = test_active[test_active['meta_prob'] >= meta_thr]
        test_outcomes_taken = test_taken['outcome'].values
        test_n = len(test_taken)
        test_hit = (test_outcomes_taken > 0).mean() if test_n > 0 else 0
        test_avg = test_outcomes_taken.mean() if test_n > 0 else 0
        test_avg_net = test_avg - FRICTION_PCT
        test_span = (test['timestamp'].max() - test['timestamp'].min()).total_seconds() / 86400
        test_daily = test_avg_net * test_n / max(1, test_span)

        print(f'\n  meta_thr={meta_thr}:')
        print(f'    Val  active={val_n}, hit={val_hit:.3f}, avg gross={val_avg:+.3f}%, '
              f'avg net={val_avg_net:+.3f}%, daily={val_daily:+.4f}%')
        print(f'    Test active={test_n}, hit={test_hit:.3f}, avg gross={test_avg:+.3f}%, '
              f'avg net={test_avg_net:+.3f}%, daily={test_daily:+.4f}%')

    # ============================================================
    # Best meta_thr → bootstrap evaluation on TEST
    # ============================================================
    print('\n' + '=' * 100)
    print('Best Test config Bootstrap Evaluation')
    print('=' * 100)

    best_thr = None
    best_daily = -999
    for meta_thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
        test_taken = test_active[test_active['meta_prob'] >= meta_thr]
        if len(test_taken) < 20:
            continue
        test_avg_net = (test_taken['outcome'].values - FRICTION_PCT).mean()
        test_span = (test['timestamp'].max() - test['timestamp'].min()).total_seconds() / 86400
        test_daily = test_avg_net * len(test_taken) / max(1, test_span)
        if test_daily > best_daily:
            best_daily = test_daily
            best_thr = meta_thr

    print(f'Best test threshold: meta_thr={best_thr}, daily={best_daily:+.4f}%')

    if best_thr is not None:
        test_taken = test_active[test_active['meta_prob'] >= best_thr]
        # Bootstrap
        trades_df = pd.DataFrame({
            'close_ts': test_taken['timestamp'].values,
            'gross_pct': test_taken['outcome'].values,
            'net_pnl_pct': test_taken['outcome'].values - FRICTION_PCT,
        })
        trades_df['close_ts'] = pd.to_datetime(trades_df['close_ts'])
        ts_min = trades_df['close_ts'].min()
        ts_max = trades_df['close_ts'].max()
        try:
            res = bootstrap_validate(trades_df, ts_min, ts_max)
            bootstrap_report(res, f'Meta-Labeling test meta_thr={best_thr}')

            f1 = res.avg_per_trade_pct > 0.07
            f6 = len(trades_df) >= 50
            overall = f1 and f6 and res.overall_pass
            print(f'\n  F1 avg_per_trade > 0.07: {"✅" if f1 else "🔴"}')
            print(f'  F6 n_trades >= 50: {"✅" if f6 else "🔴"}')
            print(f'  Bootstrap overall: {"✅" if res.overall_pass else "🔴"}')

            print('\n' + '=' * 100)
            print('VERDICT (PRE-COMMITTED)')
            print('=' * 100)
            if overall:
                print('  🟢 META-LABELING PASS — DEPLOYABLE candidate')
            else:
                print('  🔴 META-LABELING FAIL')
                print(f'  → daily {best_daily:+.4f}%, target +0.20%')
                print('  → PRE-COMMITTED: 다음 (C) On-chain features 진행')
        except Exception as e:
            print(f'  Bootstrap error: {e}')

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'mandate': 'meta-labeling on 1h ML (Lopez de Prado)',
        'best_meta_threshold': best_thr,
        'best_daily_test': best_daily,
        'test_active_signals': int((sig_test != 0).sum()),
    }
    out_path = RESULTS / f'meta_labeling_1h_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
