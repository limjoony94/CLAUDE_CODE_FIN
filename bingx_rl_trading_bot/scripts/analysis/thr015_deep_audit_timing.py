"""thr=0.15 Deep Audit — Feature Timing Lookahead (Audit 2).

사용자 critique: F&G publish time, Coinmetrics daily snapshot timing.
On-chain features가 그 day의 close 후 publish될 가능성 → timezone-shifted lookahead.

Conservative test: features t-1 day shift → re-evaluate.
  PASS: t-1 lag에서 hit rate 60%+ 유지 (signal 진짜)
  FAIL: t-1 lag에서 hit rate ~50% → timing lookahead 발견

Pre-commit: memory/thr015_deep_audit_precommit_20260501.md
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
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
sys.path.insert(0, str(ROOT / 'scripts' / 'strategy_lab'))
from h1_direction_prediction import build_features
from h1_ml_with_onchain_features import load_onchain_daily

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

THRESHOLD = 0.15


def evaluate(probs, df_segment, threshold, friction_pct):
    n = len(probs)
    direction = np.where(probs > 0.5 + threshold, 1,
                          np.where(probs < 0.5 - threshold, -1, 0))
    next_ret = df_segment['next_ret_pct'].values
    trades = []
    for i in range(n):
        if direction[i] == 0 or pd.isna(next_ret[i]):
            continue
        gross = next_ret[i] * direction[i]
        net = gross - friction_pct
        trades.append({'gross': gross, 'net': net})
    if not trades:
        return {'n_trades': 0, 'hit_rate': 0, 'daily_net': 0}
    df_t = pd.DataFrame(trades)
    span_days = (df_segment['timestamp'].max() - df_segment['timestamp'].min()).total_seconds() / 86400
    return {
        'n_trades': int(len(df_t)),
        'hit_rate': float((df_t['gross'] > 0).mean()),
        'daily_net': float(df_t['net'].sum() / max(1, span_days)),
    }


def build_features_with_lag(lag_days):
    """Build features with on-chain lag_days extra delay."""
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    feats = build_features(df)
    feats = feats.merge(df[['timestamp']], left_index=True, right_index=True, how='left')
    feats = feats.dropna(subset=['timestamp'])

    onchain = load_onchain_daily()
    # Apply additional lag — shift on-chain dates forward by lag_days
    onchain = onchain.copy()
    onchain['date'] = onchain['date'] + pd.Timedelta(days=lag_days)

    feats['date'] = pd.to_datetime(feats['timestamp']).dt.normalize()
    if feats['date'].dt.tz is not None:
        feats['date'] = feats['date'].dt.tz_localize(None)
    if onchain['date'].dt.tz is not None:
        onchain['date'] = onchain['date'].dt.tz_localize(None)
    feats = pd.merge_asof(
        feats.sort_values('date'),
        onchain.sort_values('date'),
        on='date', direction='backward',
    )
    feats = feats.dropna(subset=['fng_value', 'mvrv'])
    feats = feats.sort_values('timestamp').reset_index(drop=True)
    return feats


def run_wf_at_lag(feats, lag_days, friction):
    tech_cols = ['ret_1h', 'ret_4h', 'ret_24h', 'atr_ratio', 'rsi_14',
                 'ema_ratio', 'vol_z', 'body_ratio', 'range_atr',
                 'close_in_range', 'macd_hist', 'donchian_pos']
    onchain_cols = ['fng_value', 'fng_z90', 'mvrv', 'mvrv_z90',
                     'adr_chg_30d', 'hashrate_ratio', 'flow_imbalance']
    all_cols = tech_cols + onchain_cols
    n = len(feats)

    fold_results = []
    for k in range(5):
        train_end = 0.30 + 0.14 * k
        test_start = train_end
        test_end = min(train_end + 0.14, 1.0)
        i_tr_end = int(n * train_end)
        i_te_start = int(n * test_start)
        i_te_end = int(n * test_end)

        train = feats.iloc[:i_tr_end]
        test = feats.iloc[i_te_start:i_te_end]
        if len(train) < 100 or len(test) < 50:
            continue

        X_tr = train[all_cols].values
        y_tr = train['target'].values
        X_te = test[all_cols].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_tr_s, y_tr)
        p_te = model.predict_proba(X_te_s)[:, 1]

        r = evaluate(p_te, test, THRESHOLD, friction)
        fold_results.append({'fold': k+1, **r})

    avg_daily = np.mean([r['daily_net'] for r in fold_results]) if fold_results else 0
    avg_hit = np.mean([r['hit_rate'] for r in fold_results]) if fold_results else 0
    pos_folds = sum(1 for r in fold_results if r['daily_net'] > 0)

    return {
        'lag_days': lag_days,
        'avg_daily_net': float(avg_daily),
        'avg_hit_rate': float(avg_hit),
        'positive_folds': int(pos_folds),
        'fold_results': fold_results,
    }


def main():
    print('=' * 100)
    print('Audit 2: Feature Timing Lookahead (thr=0.15)')
    print('=' * 100)
    print(f'Conservative test: shift on-chain features by N days extra delay')
    print(f'PASS: t-1, t-2, t-3 lag에서도 hit rate 60%+ 유지')
    print(f'Friction: 0.10% (baseline)\n')

    LAGS = [0, 1, 2, 3, 7]  # 0=current, 1-3=conservative, 7=very conservative
    print(f'{"Lag (days)":<12} {"WF avg daily":>15} {"Avg hit":>10} {"Pos folds":>12}')
    print('-' * 60)

    results = {}
    for lag in LAGS:
        feats = build_features_with_lag(lag)
        r = run_wf_at_lag(feats, lag, friction=0.10)
        results[lag] = r
        marker = ''
        if lag > 0 and r['avg_hit_rate'] < 0.55:
            marker = ' 🔴 (timing lookahead 의심)'
        elif lag > 0 and r['avg_hit_rate'] >= 0.60:
            marker = ' ✅ (robust to lag)'
        print(f'{lag:<12d} {r["avg_daily_net"]:>+12.4f}% {r["avg_hit_rate"]:>10.4f} {r["positive_folds"]:>3d}/5{marker}')

    # ============================================================
    # VERDICT
    # ============================================================
    print('\n' + '=' * 100)
    print('VERDICT — Feature Timing Lookahead')
    print('=' * 100)

    base = results[0]
    lag1 = results[1]
    lag3 = results[3]

    print(f'\n  Lag 0 (current):  hit {base["avg_hit_rate"]:.4f}, daily {base["avg_daily_net"]:+.4f}%')
    print(f'  Lag 1 (t-1):      hit {lag1["avg_hit_rate"]:.4f}, daily {lag1["avg_daily_net"]:+.4f}%')
    print(f'  Lag 3 (t-3):      hit {lag3["avg_hit_rate"]:.4f}, daily {lag3["avg_daily_net"]:+.4f}%')

    hit_drop_1 = base['avg_hit_rate'] - lag1['avg_hit_rate']
    hit_drop_3 = base['avg_hit_rate'] - lag3['avg_hit_rate']

    pass_lag = lag1['avg_hit_rate'] >= 0.60 and lag3['avg_hit_rate'] >= 0.55

    print('\n  Pre-committed criteria:')
    print(f'  PASS lag 1 (hit ≥0.60):  {"✅" if lag1["avg_hit_rate"] >= 0.60 else "🔴"} ({lag1["avg_hit_rate"]:.4f})')
    print(f'  PASS lag 3 (hit ≥0.55):  {"✅" if lag3["avg_hit_rate"] >= 0.55 else "🔴"} ({lag3["avg_hit_rate"]:.4f})')

    if pass_lag:
        print('\n  🟢 TIMING LOOKAHEAD AUDIT PASS — On-chain features robust to time lag')
    else:
        print('\n  🔴 TIMING LOOKAHEAD AUDIT FAIL — Hit rate drops significantly with lag')
        print('  → On-chain features may be using future-publish info')

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'mandate': 'thr=0.15 deep audit (Audit 2 timing lookahead)',
        'threshold': THRESHOLD,
        'results_by_lag': results,
        'hit_drop_lag1': float(hit_drop_1),
        'hit_drop_lag3': float(hit_drop_3),
        'audit2_pass': bool(pass_lag),
    }
    out_path = RESULTS / f'thr015_audit2_timing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
