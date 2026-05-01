"""1h Direction Prediction — single attempt (pre-committed).

Pre-commit: memory/h1_direction_prediction_precommit_20260501.md (frozen).

Approach:
  - Simple logistic regression on BTC 1h next-bar direction
  - 8-12 causal features (returns, vol, momentum, volume)
  - 50/25/25 train/val/fresh OOS
  - Active filter: only trade when |prob - 0.5| > threshold
  - Friction 0.10% RT per trade

Stopping: 1 attempt. PASS → deployable, FAIL → closure 강제.
Hit rate > 60% → lookahead 의심.
"""
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'strategy_lab'))
from bootstrap_validator import bootstrap_validate, report as bootstrap_report

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

FRICTION_PCT = 0.10  # taker RT per trade
ACTIVE_THRESHOLD = 0.05  # |prob - 0.5| > 0.05 (filter low-confidence)


def build_features(df):
    """Causal features from BTC 1h OHLCV."""
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    cl = df['close'].values
    hi = df['high'].values
    lo = df['low'].values
    op = df['open'].values
    vol = df['volume'].values
    n = len(df)

    feats = pd.DataFrame(index=df.index)
    # 1. 1h return
    feats['ret_1h'] = pd.Series(cl).pct_change()
    # 2. 4h return
    feats['ret_4h'] = pd.Series(cl).pct_change(4)
    # 3. 24h return
    feats['ret_24h'] = pd.Series(cl).pct_change(24)
    # 4. ATR ratio (current ATR / 200-bar avg)
    tr = np.maximum.reduce([
        hi - lo,
        np.abs(hi - np.concatenate([[cl[0]], cl[:-1]])),
        np.abs(lo - np.concatenate([[cl[0]], cl[:-1]])),
    ])
    atr = pd.Series(tr).rolling(14).mean()
    atr_ma200 = atr.rolling(200).mean()
    feats['atr_ratio'] = atr / atr_ma200
    # 5. RSI 14
    delta = np.diff(cl, prepend=cl[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_l = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, 1e-10)
    feats['rsi_14'] = 100 - 100 / (1 + rs)
    # 6. EMA9/EMA21 ratio
    ema9 = pd.Series(cl).ewm(span=9, adjust=False).mean()
    ema21 = pd.Series(cl).ewm(span=21, adjust=False).mean()
    feats['ema_ratio'] = ema9 / ema21 - 1
    # 7. Volume z-score (50-bar)
    vol_ma = pd.Series(vol).rolling(50).mean()
    vol_std = pd.Series(vol).rolling(50).std()
    feats['vol_z'] = (vol - vol_ma) / vol_std.replace(0, 1)
    # 8. Body ratio
    rng = hi - lo
    body = np.abs(cl - op)
    feats['body_ratio'] = np.where(rng > 0, body / rng, 0)
    # 9. High-low range (relative to ATR)
    feats['range_atr'] = rng / atr.replace(0, 1)
    # 10. Position in range (close vs high-low)
    feats['close_in_range'] = np.where(rng > 0, (cl - lo) / rng, 0.5)
    # 11. MACD signal
    ema12 = pd.Series(cl).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(cl).ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    feats['macd_hist'] = (macd_line - macd_sig) / cl  # normalize
    # 12. Donchian position
    high_24 = pd.Series(hi).rolling(24).max()
    low_24 = pd.Series(lo).rolling(24).min()
    feats['donchian_pos'] = np.where(high_24 - low_24 > 0,
                                       (cl - low_24) / (high_24 - low_24), 0.5)

    # Target: next-bar direction
    next_ret = pd.Series(cl).pct_change().shift(-1)  # next bar return
    target = (next_ret > 0).astype(int)

    # Forward return for PnL
    feats['next_ret_pct'] = next_ret * 100
    feats['target'] = target

    # Drop NaN
    feats = feats.dropna()
    return feats


def evaluate_predictions(probs, df_test, threshold=ACTIVE_THRESHOLD):
    """Compute trade-like PnL from predictions with active filter.

    Args:
        probs: P(next return > 0)
        df_test: DataFrame with 'next_ret_pct', 'timestamp'
        threshold: |prob - 0.5| > threshold for active

    Returns:
        trades_df, stats dict
    """
    n = len(probs)
    direction = np.zeros(n, dtype=int)
    direction[probs > 0.5 + threshold] = 1
    direction[probs < 0.5 - threshold] = -1

    trades = []
    next_ret = df_test['next_ret_pct'].values
    ts = df_test['timestamp'].values if 'timestamp' in df_test.columns else df_test.index
    for i in range(n):
        if direction[i] == 0:
            continue
        ret = next_ret[i]
        if pd.isna(ret):
            continue
        gross = ret * direction[i]  # LONG: ret, SHORT: -ret
        net = gross - FRICTION_PCT
        trades.append({
            'close_ts': ts[i],
            'gross_pct': gross,
            'net_pnl_pct': net,
            'direction': direction[i],
            'prob': probs[i],
        })

    trades_df = pd.DataFrame(trades)

    # Stats
    if len(trades_df) > 0:
        hits = ((trades_df['gross_pct'] > 0)).sum()
        hit_rate = hits / len(trades_df)
    else:
        hit_rate = 0

    stats = {
        'n_trades': int(len(trades_df)),
        'n_active_pct': float(len(trades_df) / n) if n > 0 else 0,
        'hit_rate': float(hit_rate),
        'cum_gross': float(trades_df['gross_pct'].sum()) if len(trades_df) > 0 else 0,
        'cum_net': float(trades_df['net_pnl_pct'].sum()) if len(trades_df) > 0 else 0,
        'avg_gross': float(trades_df['gross_pct'].mean()) if len(trades_df) > 0 else 0,
        'avg_net': float(trades_df['net_pnl_pct'].mean()) if len(trades_df) > 0 else 0,
    }
    return trades_df, stats


def main():
    print('=' * 100)
    print('1h Direction Prediction — single attempt (pre-committed)')
    print('=' * 100)
    print(f'Friction: {FRICTION_PCT}% RT per trade')
    print(f'Active threshold: |prob - 0.5| > {ACTIVE_THRESHOLD}')
    print(f'Pre-commit: memory/h1_direction_prediction_precommit_20260501.md')
    print()

    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} 1h bars')

    feats = build_features(df)
    print(f'Features (post-NaN): {feats.shape}')

    # Reattach timestamp (for trade close_ts)
    feats = feats.merge(df[['timestamp']], left_index=True, right_index=True, how='left')
    feats = feats.dropna(subset=['timestamp'])

    # Split 50/25/25
    n = len(feats)
    i1 = int(n * 0.50)
    i2 = int(n * 0.75)
    train = feats.iloc[:i1]
    val = feats.iloc[i1:i2]
    test = feats.iloc[i2:]
    print(f'Train: {len(train)} bars, Val: {len(val)}, Test: {len(test)}')

    feat_cols = ['ret_1h', 'ret_4h', 'ret_24h', 'atr_ratio', 'rsi_14',
                 'ema_ratio', 'vol_z', 'body_ratio', 'range_atr',
                 'close_in_range', 'macd_hist', 'donchian_pos']

    X_train = train[feat_cols].values
    y_train = train['target'].values
    X_val = val[feat_cols].values
    y_val = val['target'].values
    X_test = test[feat_cols].values
    y_test = test['target'].values

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Train logistic regression
    print('\n=== Training Logistic Regression (L2 reg) ===')
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    # Predict probabilities
    p_train = model.predict_proba(X_train_s)[:, 1]
    p_val = model.predict_proba(X_val_s)[:, 1]
    p_test = model.predict_proba(X_test_s)[:, 1]

    # Naive accuracy
    print(f'\n=== Naive accuracy (no active filter) ===')
    print(f'  Train: {(model.predict(X_train_s) == y_train).mean():.3f}')
    print(f'  Val:   {(model.predict(X_val_s) == y_val).mean():.3f}')
    print(f'  Test:  {(model.predict(X_test_s) == y_test).mean():.3f}')

    # Probability distributions
    print(f'\n  Train prob distribution: mean={p_train.mean():.3f}, std={p_train.std():.3f}')
    print(f'  Test  prob distribution: mean={p_test.mean():.3f}, std={p_test.std():.3f}')
    print(f'  Test  |prob-0.5|>{ACTIVE_THRESHOLD}: {(np.abs(p_test - 0.5) > ACTIVE_THRESHOLD).mean()*100:.1f}%')

    # ============================================================
    # Per-stage trade evaluation
    # ============================================================
    print('\n' + '=' * 100)
    print('Trade Evaluation per stage')
    print('=' * 100)

    for stage_name, stage_df, probs in [('TRAIN (in-sample)', train, p_train),
                                          ('VAL', val, p_val),
                                          ('TEST (fresh OOS)', test, p_test)]:
        print(f'\n--- {stage_name} ---')
        trades, stats = evaluate_predictions(probs, stage_df)
        span_days = (stage_df['timestamp'].max() - stage_df['timestamp'].min()).total_seconds() / 86400
        print(f'  span: {span_days:.0f}d, total bars: {len(stage_df)}')
        print(f'  Active trades: {stats["n_trades"]} ({stats["n_active_pct"]*100:.1f}%)')
        print(f'  Hit rate: {stats["hit_rate"]:.4f}')
        print(f'  Cum gross: {stats["cum_gross"]:+.2f}%, Cum net: {stats["cum_net"]:+.2f}%')
        print(f'  Avg per trade: gross {stats["avg_gross"]:+.4f}%, net {stats["avg_net"]:+.4f}%')
        if stats['n_trades'] > 0:
            print(f'  Daily net: {stats["cum_net"] / span_days:+.4f}%')

    # ============================================================
    # Bootstrap on TEST (fresh OOS)
    # ============================================================
    print('\n' + '=' * 100)
    print('Bootstrap evaluation on FRESH OOS (TEST)')
    print('=' * 100)

    test_trades, test_stats = evaluate_predictions(p_test, test)
    if len(test_trades) > 0:
        test_trades['close_ts'] = pd.to_datetime(test_trades['close_ts'])
        span_min = test_trades['close_ts'].min()
        span_max = test_trades['close_ts'].max()
        res = bootstrap_validate(test_trades, span_min, span_max)
        bootstrap_report(res, '1h Direction Prediction — Fresh OOS')

        f1 = res.avg_per_trade_pct > 0.07
        f6 = len(test_trades) >= 50
        overall = f1 and f6 and res.overall_pass
        hit_rate_53 = test_stats['hit_rate'] >= 0.53

        print(f'\n  F1 avg_per_trade > 0.07: {"✅" if f1 else "🔴"}')
        print(f'  F6 n_trades >= 50: {"✅" if f6 else "🔴"}')
        print(f'  Bootstrap overall: {"✅" if res.overall_pass else "🔴"}')
        print(f'  Hit rate >= 0.53: {"✅" if hit_rate_53 else "🔴"} ({test_stats["hit_rate"]:.4f})')

        # Lookahead suspicion
        if test_stats['hit_rate'] > 0.60:
            print(f'\n⚠️  LOOKAHEAD SUSPICION: hit rate {test_stats["hit_rate"]:.4f} > 0.60')
            print(f'   Code audit 필요: target shift, train/test split 재확인')

        # Save
        out = {
            'date': datetime.now(timezone.utc).isoformat(),
            'mandate': '1h direction prediction (causal, mechanism-free)',
            'pre_commit': 'memory/h1_direction_prediction_precommit_20260501.md',
            'train_size': len(train),
            'val_size': len(val),
            'test_size': len(test),
            'features': feat_cols,
            'naive_accuracy': {
                'train': float((model.predict(X_train_s) == y_train).mean()),
                'val': float((model.predict(X_val_s) == y_val).mean()),
                'test': float((model.predict(X_test_s) == y_test).mean()),
            },
            'test_stats': test_stats,
            'test_bootstrap': {
                'mean_daily': float(res.mean_daily_pct),
                'pos_rate': float(res.pos_rate),
                'p5_daily': float(res.p5_daily_pct),
                'avg_per_trade': float(res.avg_per_trade_pct),
                'pass_criteria': {k: bool(v) for k, v in res.pass_criteria.items()},
                'overall_pass': bool(res.overall_pass),
            },
            'F1_avg_gross_pass': bool(f1),
            'F6_full_n_pass': bool(f6),
            'hit_rate_53_pass': bool(hit_rate_53),
            'overall_pass': bool(overall and hit_rate_53),
            'lookahead_suspicion': bool(test_stats['hit_rate'] > 0.60),
        }
        out_path = RESULTS / f'h1_direction_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f'\nSaved: {out_path}')

        # VERDICT
        print('\n' + '=' * 100)
        print('VERDICT (PRE-COMMITTED)')
        print('=' * 100)
        if overall and hit_rate_53:
            print('  🟢 1h DIRECTION PREDICTION PASS — DEPLOYABLE candidate')
            print('  → Lookahead audit + advisor reconcile + regime test')
        else:
            print('  🔴 1h DIRECTION PREDICTION FAIL')
            print('  → PRE-COMMITTED: closure 강제. Deep learning/RNN/feature iteration silent pivot 금지.')
            print(f'  → Hit rate {test_stats["hit_rate"]:.4f} (need ≥0.53)')
            print(f'  → 6-criteria: avg_per_trade {"✅" if f1 else "🔴"}, '
                  f'n_trades {"✅" if f6 else "🔴"}, '
                  f'bootstrap overall {"✅" if res.overall_pass else "🔴"}')


if __name__ == '__main__':
    main()
