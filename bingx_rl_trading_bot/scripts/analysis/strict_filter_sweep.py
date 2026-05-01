"""Strict Active Filter Sweep on 1h ML + on-chain features.

Pre-commit: memory/strict_filter_sweep_precommit_20260501.md (frozen).

Hypothesis:
  (C) on-chain ML hit 56.24% at |prob-0.5|>0.05 (43% bars active, 2093 trades).
  Friction overhead 2093 × 0.10% = -1.18%/day destroys edge.
  Strict filter → fewer trades + higher hit rate → friction overcome가능?

Sweep: |prob - 0.5| > {0.05, 0.08, 0.10, 0.12, 0.15, 0.20}
Each threshold: trade count, hit rate, avg gross/net, daily PnL, bootstrap user 6-criteria

Bonferroni: 6 thresholds → effective α = 0.008 (val로 best 선택, test 1회 평가)
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
from bootstrap_validator import bootstrap_validate, report as bootstrap_report

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
FRICTION_PCT = 0.10
THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]


def evaluate_at_threshold(probs, df_segment, threshold):
    n = len(probs)
    direction = np.where(probs > 0.5 + threshold, 1,
                          np.where(probs < 0.5 - threshold, -1, 0))
    next_ret = df_segment['next_ret_pct'].values
    ts = df_segment['timestamp'].values
    trades = []
    for i in range(n):
        if direction[i] == 0 or pd.isna(next_ret[i]):
            continue
        gross = next_ret[i] * direction[i]
        net = gross - FRICTION_PCT
        trades.append({
            'close_ts': ts[i],
            'gross_pct': gross,
            'net_pnl_pct': net,
            'direction': direction[i],
            'prob': probs[i],
        })
    return pd.DataFrame(trades)


def stats_from_trades(trades_df, span_days):
    if len(trades_df) == 0:
        return {'n_trades': 0, 'hit_rate': 0, 'daily_net': 0,
                'avg_gross': 0, 'avg_net': 0, 'cum_net': 0}
    return {
        'n_trades': int(len(trades_df)),
        'hit_rate': float((trades_df['gross_pct'] > 0).mean()),
        'avg_gross': float(trades_df['gross_pct'].mean()),
        'avg_net': float(trades_df['net_pnl_pct'].mean()),
        'cum_gross': float(trades_df['gross_pct'].sum()),
        'cum_net': float(trades_df['net_pnl_pct'].sum()),
        'daily_net': float(trades_df['net_pnl_pct'].sum() / max(1, span_days)),
    }


def main():
    print('=' * 100)
    print('Strict Active Filter Sweep on 1h ML + On-chain Features')
    print('=' * 100)
    print(f'Thresholds: {THRESHOLDS}')
    print(f'Friction: {FRICTION_PCT}% RT per trade')
    print(f'Pre-commit: memory/strict_filter_sweep_precommit_20260501.md\n')

    # ============================================================
    # Build features (reuse (C))
    # ============================================================
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    feats = build_features(df)
    feats = feats.merge(df[['timestamp']], left_index=True, right_index=True, how='left')
    feats = feats.dropna(subset=['timestamp'])

    onchain = load_onchain_daily()
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
    print(f'Combined features: {feats.shape}\n')

    tech_cols = ['ret_1h', 'ret_4h', 'ret_24h', 'atr_ratio', 'rsi_14',
                 'ema_ratio', 'vol_z', 'body_ratio', 'range_atr',
                 'close_in_range', 'macd_hist', 'donchian_pos']
    onchain_cols = ['fng_value', 'fng_z90', 'mvrv', 'mvrv_z90',
                     'adr_chg_30d', 'hashrate_ratio', 'flow_imbalance']
    all_cols = tech_cols + onchain_cols

    # Split
    n = len(feats)
    i1 = int(n * 0.50)
    i2 = int(n * 0.75)
    train = feats.iloc[:i1]
    val = feats.iloc[i1:i2]
    test = feats.iloc[i2:]
    print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')

    X_train = train[all_cols].values
    y_train = train['target'].values
    X_val = val[all_cols].values
    X_test = test[all_cols].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)
    p_val = model.predict_proba(X_val_s)[:, 1]
    p_test = model.predict_proba(X_test_s)[:, 1]

    val_span = (val['timestamp'].max() - val['timestamp'].min()).total_seconds() / 86400
    test_span = (test['timestamp'].max() - test['timestamp'].min()).total_seconds() / 86400

    print(f'\nVal span: {val_span:.0f}d, Test span: {test_span:.0f}d')

    # ============================================================
    # Sweep on VAL — best threshold selection
    # ============================================================
    print('\n' + '=' * 100)
    print('VAL sweep — threshold selection (Bonferroni α=0.008)')
    print('=' * 100)
    print(f'{"thr":<6} {"n":>6} {"hit":>6} {"avg_gross":>10} {"avg_net":>10} {"daily_net":>10}')
    print('-' * 60)

    val_results = []
    for thr in THRESHOLDS:
        trades = evaluate_at_threshold(p_val, val, thr)
        s = stats_from_trades(trades, val_span)
        s['threshold'] = thr
        val_results.append(s)
        print(f'{thr:<6.2f} {s["n_trades"]:>6} {s["hit_rate"]:>6.4f} {s["avg_gross"]:>+9.4f}% {s["avg_net"]:>+9.4f}% {s["daily_net"]:>+9.4f}%')

    # Best by val daily_net (assuming sufficient n)
    val_eligible = [r for r in val_results if r['n_trades'] >= 30]
    if val_eligible:
        best_val = max(val_eligible, key=lambda r: r['daily_net'])
        best_thr = best_val['threshold']
        print(f'\nBest val threshold: {best_thr} (daily_net {best_val["daily_net"]:+.4f}%, n={best_val["n_trades"]})')
    else:
        best_thr = THRESHOLDS[0]
        print(f'\nNo threshold has n≥30 on val. Using lowest: {best_thr}')

    # ============================================================
    # Test all thresholds (for spectrum) + bootstrap on best
    # ============================================================
    print('\n' + '=' * 100)
    print('TEST sweep — full spectrum + bootstrap on val-best')
    print('=' * 100)
    print(f'{"thr":<6} {"n":>6} {"hit":>6} {"avg_gross":>10} {"avg_net":>10} {"daily_net":>10}')
    print('-' * 60)

    test_results = []
    for thr in THRESHOLDS:
        trades = evaluate_at_threshold(p_test, test, thr)
        s = stats_from_trades(trades, test_span)
        s['threshold'] = thr
        test_results.append(s)
        marker = ' ← val-best' if abs(thr - best_thr) < 1e-9 else ''
        print(f'{thr:<6.2f} {s["n_trades"]:>6} {s["hit_rate"]:>6.4f} {s["avg_gross"]:>+9.4f}% {s["avg_net"]:>+9.4f}% {s["daily_net"]:>+9.4f}%{marker}')

    # Bootstrap on val-best threshold
    print('\n' + '=' * 100)
    print(f'TEST bootstrap on val-best threshold {best_thr}')
    print('=' * 100)
    test_trades = evaluate_at_threshold(p_test, test, best_thr)
    overall_pass = False
    bs_dict = None
    if len(test_trades) > 0:
        test_trades['close_ts'] = pd.to_datetime(test_trades['close_ts'])
        ts_min = test_trades['close_ts'].min()
        ts_max = test_trades['close_ts'].max()
        try:
            res = bootstrap_validate(test_trades, ts_min, ts_max)
            bootstrap_report(res, f'Test thr={best_thr}')

            f1 = res.avg_per_trade_pct > 0.07
            f6 = len(test_trades) >= 50
            test_hit = float((test_trades['gross_pct'] > 0).mean())
            hit_53 = test_hit >= 0.53
            overall_pass = f1 and f6 and res.overall_pass and hit_53

            print(f'\n  F1 avg_per_trade > 0.07:    {"✅" if f1 else "🔴"}')
            print(f'  F6 n_trades >= 50:           {"✅" if f6 else "🔴"}')
            print(f'  Bootstrap overall:           {"✅" if res.overall_pass else "🔴"}')
            print(f'  Hit rate >= 0.53:            {"✅" if hit_53 else "🔴"} ({test_hit:.4f})')

            bs_dict = {
                'mean_daily': float(res.mean_daily_pct),
                'p5_daily': float(res.p5_daily_pct),
                'pos_rate': float(res.pos_rate),
                'avg_per_trade': float(res.avg_per_trade_pct),
                'pass_count_6': sum(res.pass_criteria.values()),
                'overall_pass': bool(res.overall_pass),
            }
        except Exception as e:
            print(f'Bootstrap error: {e}')

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'mandate': 'strict active filter sweep on 1h ML + on-chain',
        'pre_commit': 'memory/strict_filter_sweep_precommit_20260501.md',
        'thresholds': THRESHOLDS,
        'val_results': val_results,
        'test_results': test_results,
        'val_best_threshold': best_thr,
        'test_bootstrap': bs_dict,
        'overall_pass': bool(overall_pass),
    }
    out_path = RESULTS / f'strict_filter_sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')

    # VERDICT
    print('\n' + '=' * 100)
    print('VERDICT (PRE-COMMITTED)')
    print('=' * 100)
    if overall_pass:
        print('  🟢 STRICT FILTER SWEEP PASS — DEPLOYABLE candidate')
        print('  → Lookahead audit + advisor reconcile + regime test')
    else:
        print('  🔴 STRICT FILTER SWEEP FAIL')
        print('  → PRE-COMMITTED: 자율 mandate 안 final closure.')
        print('  → 새 path는 사용자 explicit instruction 필요.')

        # Diagnostic: any threshold deployable in test?
        test_pos = [r for r in test_results if r['daily_net'] > 0 and r['n_trades'] >= 50]
        if test_pos:
            print(f'\n  Diagnostic: test에서 daily_net > 0 + n≥50 thresholds:')
            for r in test_pos:
                print(f'    thr={r["threshold"]}: n={r["n_trades"]}, hit={r["hit_rate"]:.4f}, daily={r["daily_net"]:+.4f}%')


if __name__ == '__main__':
    main()
