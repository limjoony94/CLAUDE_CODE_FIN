"""thr=0.15 Robustness Audit — Walk-forward + Regime split.

Pre-commit: memory/thr015_robustness_audit_precommit_20260501.md (frozen).

Test thr=0.15 (val-best from strict filter sweep) for:
  1. Walk-forward 5-fold expanding window — temporal robustness
  2. Regime split (high-vol vs low-vol) — regime robustness

Goal: distinguish in-sample fitting (val→test) vs robust edge.

Pre-committed criteria:
  - WF ≥ 3/5 positive daily AND
  - 2/2 regimes positive AND
  - Avg hit rate ≥ 58%
  → LIVE deploy candidate
  Else → closure 강제.
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
FRICTION_PCT = 0.10
THRESHOLD = 0.15  # FROZEN per pre-commit


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
        'cum_net': float(trades_df['net_pnl_pct'].sum()),
        'daily_net': float(trades_df['net_pnl_pct'].sum() / max(1, span_days)),
    }


def build_combined_features():
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

    # ATR ratio for regime
    cl = feats['close'].values if 'close' in feats.columns else None
    return feats


def main():
    print('=' * 100)
    print('thr=0.15 Robustness Audit — Walk-forward + Regime')
    print('=' * 100)
    print(f'Threshold: {THRESHOLD} (FROZEN per pre-commit)')
    print(f'Friction: {FRICTION_PCT}% RT per trade')
    print(f'Pre-commit: memory/thr015_robustness_audit_precommit_20260501.md\n')

    feats = build_combined_features()
    print(f'Combined features: {feats.shape}')

    tech_cols = ['ret_1h', 'ret_4h', 'ret_24h', 'atr_ratio', 'rsi_14',
                 'ema_ratio', 'vol_z', 'body_ratio', 'range_atr',
                 'close_in_range', 'macd_hist', 'donchian_pos']
    onchain_cols = ['fng_value', 'fng_z90', 'mvrv', 'mvrv_z90',
                     'adr_chg_30d', 'hashrate_ratio', 'flow_imbalance']
    all_cols = tech_cols + onchain_cols

    n = len(feats)
    print(f'Total bars: {n}')

    # ============================================================
    # Walk-forward 5-fold expanding window
    # ============================================================
    print('\n' + '=' * 100)
    print('Walk-forward 5-fold (expanding window)')
    print('=' * 100)

    # Fold splits: train always starts at 0, test windows are non-overlapping after first 30%
    # Fold k: train [0, 0.30 + 0.14*k), test [0.30 + 0.14*k, 0.30 + 0.14*(k+1))
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
            print(f'  Fold {k+1}: skipped (insufficient data)')
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

        trades = evaluate_at_threshold(p_te, test, THRESHOLD)
        span = (test['timestamp'].max() - test['timestamp'].min()).total_seconds() / 86400
        s = stats_from_trades(trades, span)
        s['fold'] = k + 1
        s['train_pct'] = train_end
        s['test_pct'] = (test_start, test_end)
        s['span_days'] = float(span)
        fold_results.append(s)

        print(f'  Fold {k+1} train [0, {train_end:.2f}) test [{test_start:.2f}, {test_end:.2f}): '
              f'n={s["n_trades"]:>4}, hit={s["hit_rate"]:.4f}, '
              f'avg_net={s["avg_net"]:>+7.4f}%, daily={s["daily_net"]:>+7.4f}%')

    # WF aggregate
    wf_positive = sum(1 for r in fold_results if r['daily_net'] > 0)
    wf_total = len(fold_results)
    avg_hit = np.mean([r['hit_rate'] for r in fold_results if r['n_trades'] > 0])
    avg_daily = np.mean([r['daily_net'] for r in fold_results])
    total_n = sum(r['n_trades'] for r in fold_results)

    print(f'\n  WF Aggregate: {wf_positive}/{wf_total} folds positive')
    print(f'  Avg hit rate: {avg_hit:.4f}')
    print(f'  Avg daily: {avg_daily:+.4f}%')
    print(f'  Total trades: {total_n}')

    # ============================================================
    # Regime split (high-vol vs low-vol)
    # Use train-final model (50%) on rest 50%
    # ============================================================
    print('\n' + '=' * 100)
    print('Regime split (high-vol vs low-vol on test 50%)')
    print('=' * 100)

    i_split = int(n * 0.50)
    train = feats.iloc[:i_split]
    test = feats.iloc[i_split:]

    X_tr = train[all_cols].values
    y_tr = train['target'].values
    X_te = test[all_cols].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_tr_s, y_tr)
    p_te = model.predict_proba(X_te_s)[:, 1]

    trades = evaluate_at_threshold(p_te, test, THRESHOLD)

    # ATR ratio at trade time → regime label
    test = test.reset_index(drop=True)
    high_vol_mask = test['atr_ratio'] > 1.0
    print(f'  High-vol bars: {high_vol_mask.sum()} ({high_vol_mask.mean()*100:.1f}%)')
    print(f'  Low-vol bars:  {(~high_vol_mask).sum()} ({(~high_vol_mask).mean()*100:.1f}%)')

    # Match trades to bars by timestamp
    if len(trades) > 0:
        trades['close_ts'] = pd.to_datetime(trades['close_ts'])
        # Find regime per trade
        test_with_regime = test[['timestamp', 'atr_ratio']].copy()
        test_with_regime['ts'] = pd.to_datetime(test_with_regime['timestamp'])
        trades['ts'] = trades['close_ts']
        trades_with_regime = trades.merge(test_with_regime[['ts', 'atr_ratio']], on='ts', how='left')

        high_vol_trades = trades_with_regime[trades_with_regime['atr_ratio'] > 1.0]
        low_vol_trades = trades_with_regime[trades_with_regime['atr_ratio'] <= 1.0]

        # Per-regime stats
        test_span = (test['timestamp'].max() - test['timestamp'].min()).total_seconds() / 86400
        # Approximate per-regime span
        hv_span = test_span * high_vol_mask.mean()
        lv_span = test_span * (~high_vol_mask).mean()

        hv_stats = stats_from_trades(high_vol_trades[['close_ts', 'gross_pct', 'net_pnl_pct']], hv_span)
        lv_stats = stats_from_trades(low_vol_trades[['close_ts', 'gross_pct', 'net_pnl_pct']], lv_span)

        print(f'\n  High-vol regime ({hv_span:.0f}d):')
        print(f'    n={hv_stats["n_trades"]}, hit={hv_stats["hit_rate"]:.4f}, '
              f'avg_net={hv_stats["avg_net"]:+.4f}%, daily={hv_stats["daily_net"]:+.4f}%')
        print(f'  Low-vol regime ({lv_span:.0f}d):')
        print(f'    n={lv_stats["n_trades"]}, hit={lv_stats["hit_rate"]:.4f}, '
              f'avg_net={lv_stats["avg_net"]:+.4f}%, daily={lv_stats["daily_net"]:+.4f}%')

        regimes_positive = (hv_stats['daily_net'] > 0) + (lv_stats['daily_net'] > 0)
    else:
        hv_stats = None
        lv_stats = None
        regimes_positive = 0

    # ============================================================
    # Pre-committed evaluation
    # ============================================================
    print('\n' + '=' * 100)
    print('Pre-Committed Evaluation')
    print('=' * 100)

    cond1 = wf_positive >= 3
    cond2 = regimes_positive >= 2
    cond3 = avg_hit >= 0.58
    overall = cond1 and cond2 and cond3

    print(f'  C1: WF ≥ 3/5 positive — {wf_positive}/{wf_total} {"✅" if cond1 else "🔴"}')
    print(f'  C2: 2/2 regimes positive — {regimes_positive}/2 {"✅" if cond2 else "🔴"}')
    print(f'  C3: Avg hit ≥ 0.58 — {avg_hit:.4f} {"✅" if cond3 else "🔴"}')

    print('\n' + '=' * 100)
    if overall:
        print('  🟢 ROBUSTNESS PASS — LIVE DEPLOY CANDIDATE')
        print('  → 별도 mandate 필요 (LIVE deploy execution은 사용자 explicit instruction)')
    else:
        print('  🔴 ROBUSTNESS FAIL — IN-SAMPLE FITTING CONFIRMED')
        print('  → PRE-COMMITTED: closure 강제. 사용자 explicit instruction 필요.')

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'mandate': 'thr=0.15 robustness audit (WF + regime)',
        'pre_commit': 'memory/thr015_robustness_audit_precommit_20260501.md',
        'threshold': THRESHOLD,
        'wf_results': fold_results,
        'wf_positive_count': wf_positive,
        'wf_total': wf_total,
        'wf_avg_hit': float(avg_hit) if avg_hit else 0,
        'wf_avg_daily': float(avg_daily) if avg_daily else 0,
        'wf_total_trades': int(total_n),
        'regime_high_vol': hv_stats,
        'regime_low_vol': lv_stats,
        'regimes_positive': int(regimes_positive),
        'pre_commit_eval': {
            'C1_wf_3of5': bool(cond1),
            'C2_2of2_regimes': bool(cond2),
            'C3_avg_hit_58': bool(cond3),
            'overall_pass': bool(overall),
        },
    }
    out_path = RESULTS / f'thr015_robustness_audit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
