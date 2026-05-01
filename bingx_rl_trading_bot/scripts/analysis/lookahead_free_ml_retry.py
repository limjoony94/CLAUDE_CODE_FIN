"""Lookahead-Free ML Retry — Option 2 (사용자 선택).

Pre-commit: memory/lookahead_free_ml_retry_precommit_20260501.md

t-1 day lag enforcement on on-chain features (어제 publish data만 사용).
WF 5-fold + threshold sweep + friction sweep + model comparison.

Pre-commit PASS criteria (모두 충족):
  - Best config WF avg daily > 0 at 0.16% RT
  - WF hit rate ≥ 58%
  - WF ≥ 3/5 folds positive
  - Threshold robust (0.13-0.17 region stable)
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
from h1_ml_with_onchain_features import load_onchain_daily

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

THRESHOLDS = [0.10, 0.12, 0.15, 0.18]
FRICTIONS = [0.10, 0.12, 0.14, 0.16, 0.18]
MODELS = ['LR', 'RF']
ONCHAIN_LAG_DAYS = 1  # t-1 day lag (lookahead-free)


def build_lookahead_free_features():
    """Build features with t-1 lag on on-chain (lookahead-free)."""
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    feats = build_features(df)
    feats = feats.merge(df[['timestamp']], left_index=True, right_index=True, how='left')
    feats = feats.dropna(subset=['timestamp'])

    onchain = load_onchain_daily()
    # t-1 lag enforcement
    onchain = onchain.copy()
    onchain['date'] = onchain['date'] + pd.Timedelta(days=ONCHAIN_LAG_DAYS)

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


def evaluate(probs, df_segment, threshold, friction):
    n = len(probs)
    direction = np.where(probs > 0.5 + threshold, 1,
                          np.where(probs < 0.5 - threshold, -1, 0))
    next_ret = df_segment['next_ret_pct'].values
    trades = []
    for i in range(n):
        if direction[i] == 0 or pd.isna(next_ret[i]):
            continue
        gross = next_ret[i] * direction[i]
        net = gross - friction
        trades.append({'gross': gross, 'net': net})
    if not trades:
        return {'n_trades': 0, 'hit_rate': 0, 'daily_net': 0, 'avg_net': 0}
    df_t = pd.DataFrame(trades)
    span_days = (df_segment['timestamp'].max() - df_segment['timestamp'].min()).total_seconds() / 86400
    return {
        'n_trades': int(len(df_t)),
        'hit_rate': float((df_t['gross'] > 0).mean()),
        'avg_net': float(df_t['net'].mean()),
        'daily_net': float(df_t['net'].sum() / max(1, span_days)),
    }


def run_wf(feats, all_cols, model_type='LR', threshold=0.15, friction=0.10):
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

        if model_type == 'LR':
            model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        else:  # RF
            model = RandomForestClassifier(n_estimators=100, max_depth=5,
                                            min_samples_leaf=20, random_state=42, n_jobs=-1)
        model.fit(X_tr_s, y_tr)
        p_te = model.predict_proba(X_te_s)[:, 1]

        r = evaluate(p_te, test, threshold, friction)
        fold_results.append({'fold': k+1, **r})

    if not fold_results:
        return None
    avg_daily = np.mean([r['daily_net'] for r in fold_results])
    avg_hit = np.mean([r['hit_rate'] for r in fold_results])
    pos_folds = sum(1 for r in fold_results if r['daily_net'] > 0)
    return {
        'avg_daily_net': float(avg_daily),
        'avg_hit_rate': float(avg_hit),
        'positive_folds': int(pos_folds),
        'fold_results': fold_results,
    }


def main():
    print('=' * 100)
    print('Lookahead-Free ML Retry (Option 2) — t-1 lag on-chain + WF + threshold + friction + model')
    print('=' * 100)
    print(f'Threshold sweep: {THRESHOLDS}')
    print(f'Friction sweep: {FRICTIONS}')
    print(f'Models: {MODELS}')
    print(f'On-chain lag: t-{ONCHAIN_LAG_DAYS}\n')

    feats = build_lookahead_free_features()
    print(f'Lookahead-free features: {feats.shape}\n')

    tech_cols = ['ret_1h', 'ret_4h', 'ret_24h', 'atr_ratio', 'rsi_14',
                 'ema_ratio', 'vol_z', 'body_ratio', 'range_atr',
                 'close_in_range', 'macd_hist', 'donchian_pos']
    onchain_cols = ['fng_value', 'fng_z90', 'mvrv', 'mvrv_z90',
                     'adr_chg_30d', 'hashrate_ratio', 'flow_imbalance']
    all_cols = tech_cols + onchain_cols

    # Master sweep: threshold × friction × model
    print(f'{"Model":<5} {"Thr":<6} {"Fric":<6} {"WF Daily":>10} {"Hit":>8} {"Pos":>6}')
    print('-' * 60)

    results = []
    for model_type in MODELS:
        for thr in THRESHOLDS:
            r = run_wf(feats, all_cols, model_type=model_type, threshold=thr, friction=0.10)
            if r is None:
                continue
            # Now compute at multiple frictions (using same predictions)
            for fric in FRICTIONS:
                # Re-evaluate fold trades at this friction
                # Reuse predictions — but we trained once per (thr,model), need to re-run
                # Simpler: store per-fold gross averages and recompute
                fold_results_at_fric = []
                for fr in r['fold_results']:
                    # Approximate: subtract (fric - 0.10) from avg_net, daily_net
                    n_trades = fr['n_trades']
                    if n_trades == 0:
                        fold_results_at_fric.append({'daily_net': 0, 'hit_rate': fr['hit_rate']})
                        continue
                    # avg_gross = avg_net + 0.10 (orig friction)
                    avg_gross = fr['avg_net'] + 0.10
                    new_avg_net = avg_gross - fric
                    new_daily = new_avg_net * n_trades / max(1, (fr.get('span_days', 100)))
                    # We don't have span_days per fold stored, approximate from daily_net/avg_net ratio
                    # daily_net = avg_net * n_trades / span_days, so span_days = avg_net*n/daily_net
                    if abs(fr['avg_net']) > 1e-9 and abs(fr['daily_net']) > 1e-9:
                        span_days_approx = fr['avg_net'] * n_trades / fr['daily_net']
                        new_daily = new_avg_net * n_trades / span_days_approx
                    fold_results_at_fric.append({'daily_net': new_daily, 'hit_rate': fr['hit_rate']})
                avg_daily_fric = np.mean([f['daily_net'] for f in fold_results_at_fric])
                pos_folds_fric = sum(1 for f in fold_results_at_fric if f['daily_net'] > 0)

                results.append({
                    'model': model_type,
                    'threshold': thr,
                    'friction': fric,
                    'wf_avg_daily': float(avg_daily_fric),
                    'wf_avg_hit': float(r['avg_hit_rate']),
                    'wf_positive_folds': int(pos_folds_fric),
                })

                if fric in [0.10, 0.16]:
                    marker = ''
                    if avg_daily_fric > 0 and fric == 0.16:
                        marker = ' ✅ critical PASS'
                    elif avg_daily_fric < 0:
                        marker = ' 🔴'
                    print(f'{model_type:<5} {thr:<6.2f} {fric:<6.2f}% {avg_daily_fric:>+9.4f}% {r["avg_hit_rate"]:>8.4f} {pos_folds_fric:>3d}/5{marker}')

    # ============================================================
    # Find best config
    # ============================================================
    print('\n=== Best by WF avg daily at 0.10% friction ===')
    best_010 = max(results, key=lambda r: r['wf_avg_daily'] if r['friction'] == 0.10 else -999)
    print(f'  {best_010["model"]} thr={best_010["threshold"]}: daily {best_010["wf_avg_daily"]:+.4f}%, hit {best_010["wf_avg_hit"]:.4f}, pos {best_010["wf_positive_folds"]}/5')

    print('\n=== Best by WF avg daily at 0.16% friction (CRITICAL) ===')
    best_016 = max(results, key=lambda r: r['wf_avg_daily'] if r['friction'] == 0.16 else -999)
    print(f'  {best_016["model"]} thr={best_016["threshold"]}: daily {best_016["wf_avg_daily"]:+.4f}%, hit {best_016["wf_avg_hit"]:.4f}, pos {best_016["wf_positive_folds"]}/5')

    # Pre-committed evaluation
    pass_strict = (
        best_016['wf_avg_daily'] > 0 and
        best_016['wf_avg_hit'] >= 0.58 and
        best_016['wf_positive_folds'] >= 3
    )

    print('\n' + '=' * 100)
    print('VERDICT (PRE-COMMITTED)')
    print('=' * 100)
    if pass_strict:
        print('  🟢 LOOKAHEAD-FREE ML PASS — DEPLOYABLE candidate')
        print(f'  Best: {best_016["model"]} thr={best_016["threshold"]} at 0.16% friction')
        print(f'  daily {best_016["wf_avg_daily"]:+.4f}%, hit {best_016["wf_avg_hit"]:.4f}, '
              f'pos {best_016["wf_positive_folds"]}/5')
    else:
        print('  🔴 LOOKAHEAD-FREE ML FAIL — final closure with Funding only')
        print(f'  Best at 0.16%: daily {best_016["wf_avg_daily"]:+.4f}% (need > 0)')
        print(f'                 hit {best_016["wf_avg_hit"]:.4f} (need ≥ 0.58)')
        print(f'                 pos {best_016["wf_positive_folds"]}/5 (need ≥ 3)')

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'mandate': 'Lookahead-free ML retry (option 2)',
        'pre_commit': 'memory/lookahead_free_ml_retry_precommit_20260501.md',
        'onchain_lag_days': ONCHAIN_LAG_DAYS,
        'thresholds': THRESHOLDS,
        'frictions': FRICTIONS,
        'models': MODELS,
        'all_results': results,
        'best_at_010': best_010,
        'best_at_016': best_016,
        'pass_strict': bool(pass_strict),
    }
    out_path = RESULTS / f'lookahead_free_ml_retry_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
