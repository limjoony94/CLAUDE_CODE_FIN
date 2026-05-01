"""thr=0.15 Deep Audit — Friction Sensitivity (Audit 1, priority 1).

사용자 critique: 0.10% RT는 lenient. BingX taker 실제 0.045-0.05%/side + slippage
0.02-0.05%/side = 0.12-0.20% RT realistic.

Sweep friction RT [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
  각 friction에서 thr=0.15 WF 5-fold daily PnL
  PASS: 0.16% RT에서 WF avg daily > 0
  FAIL: 0.12% 또는 lower에서 negative → 즉시 retract

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
FRICTION_LEVELS = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]


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
    return feats


def evaluate_at_friction(probs, df_segment, threshold, friction_pct):
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
        return {'n_trades': 0, 'hit_rate': 0, 'avg_gross': 0, 'avg_net': 0, 'cum_net': 0, 'daily_net': 0}
    df_t = pd.DataFrame(trades)
    span_days = (df_segment['timestamp'].max() - df_segment['timestamp'].min()).total_seconds() / 86400
    return {
        'n_trades': int(len(df_t)),
        'hit_rate': float((df_t['gross'] > 0).mean()),
        'avg_gross': float(df_t['gross'].mean()),
        'avg_net': float(df_t['net'].mean()),
        'cum_net': float(df_t['net'].sum()),
        'daily_net': float(df_t['net'].sum() / max(1, span_days)),
    }


def main():
    print('=' * 100)
    print('Audit 1: Friction Sensitivity (thr=0.15)')
    print('=' * 100)
    print(f'Friction levels: {FRICTION_LEVELS}')
    print(f'Pre-commit: memory/thr015_deep_audit_precommit_20260501.md\n')

    feats = build_combined_features()
    print(f'Combined features: {feats.shape}\n')

    tech_cols = ['ret_1h', 'ret_4h', 'ret_24h', 'atr_ratio', 'rsi_14',
                 'ema_ratio', 'vol_z', 'body_ratio', 'range_atr',
                 'close_in_range', 'macd_hist', 'donchian_pos']
    onchain_cols = ['fng_value', 'fng_z90', 'mvrv', 'mvrv_z90',
                     'adr_chg_30d', 'hashrate_ratio', 'flow_imbalance']
    all_cols = tech_cols + onchain_cols

    n = len(feats)

    # WF 5-fold per friction level
    print(f'{"Friction":<10} {"WF avg daily":>15} {"Min daily":>12} {"Max daily":>12} {"Avg hit":>10} {"Pos folds":>12}')
    print('-' * 80)

    results_by_friction = {}
    for fric in FRICTION_LEVELS:
        fold_dailies = []
        fold_hits = []
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

            r = evaluate_at_friction(p_te, test, THRESHOLD, fric)
            fold_dailies.append(r['daily_net'])
            fold_hits.append(r['hit_rate'])
            fold_results.append({'fold': k+1, **r})

        avg_daily = np.mean(fold_dailies) if fold_dailies else 0
        min_daily = np.min(fold_dailies) if fold_dailies else 0
        max_daily = np.max(fold_dailies) if fold_dailies else 0
        avg_hit = np.mean(fold_hits) if fold_hits else 0
        pos_folds = sum(1 for d in fold_dailies if d > 0)
        n_folds = len(fold_dailies)

        marker = ''
        if avg_daily <= 0:
            marker = ' 🔴'
        elif fric == 0.16 and avg_daily > 0:
            marker = ' ✅ (PASS critical 0.16%)'

        print(f'{fric:<10.2f}% {avg_daily:>+12.4f}% {min_daily:>+9.4f}% {max_daily:>+9.4f}% {avg_hit:>10.4f} {pos_folds:>3d}/{n_folds:<2d}{marker}')

        results_by_friction[fric] = {
            'fold_results': fold_results,
            'avg_daily_net': float(avg_daily),
            'min_daily_net': float(min_daily),
            'max_daily_net': float(max_daily),
            'avg_hit_rate': float(avg_hit),
            'positive_folds': int(pos_folds),
            'n_folds': int(n_folds),
        }

    # ============================================================
    # VERDICT
    # ============================================================
    print('\n' + '=' * 100)
    print('VERDICT — Friction Sensitivity')
    print('=' * 100)

    fric_010 = results_by_friction[0.10]
    fric_012 = results_by_friction[0.12]
    fric_014 = results_by_friction[0.14]
    fric_016 = results_by_friction[0.16]
    fric_020 = results_by_friction[0.20]

    print(f'\n  RT 0.10% (lenient, current): WF avg daily {fric_010["avg_daily_net"]:+.4f}%')
    print(f'  RT 0.12% (BingX basic):      WF avg daily {fric_012["avg_daily_net"]:+.4f}%')
    print(f'  RT 0.14% (mid):              WF avg daily {fric_014["avg_daily_net"]:+.4f}%')
    print(f'  RT 0.16% (CRITICAL):         WF avg daily {fric_016["avg_daily_net"]:+.4f}%')
    print(f'  RT 0.20% (worst-case):       WF avg daily {fric_020["avg_daily_net"]:+.4f}%')

    # Pre-committed evaluation
    pass_016 = fric_016['avg_daily_net'] > 0 and fric_016['positive_folds'] >= 3
    pass_012 = fric_012['avg_daily_net'] > 0
    pass_020 = fric_020['avg_daily_net'] > 0

    print('\n  Pre-committed criteria:')
    print(f'  PASS 0.12%: {"✅" if pass_012 else "🔴"} (need daily > 0)')
    print(f'  PASS 0.16%: {"✅" if pass_016 else "🔴"} (CRITICAL — need daily > 0 + 3/5 folds positive)')
    print(f'  PASS 0.20%: {"✅" if pass_020 else "🔴"} (BONUS robust)')

    print('\n' + '=' * 100)
    if pass_016:
        print('  🟢 FRICTION AUDIT PASS — Realistic friction (0.16%) 도 deployable')
        print('  → Audit 2 (feature timing lookahead) 진행')
    else:
        print('  🔴 FRICTION AUDIT FAIL — Realistic friction에서 fragile')
        print('  → DEPLOYABLE 주장 RETRACT (per pre-commit)')
        print(f'  → RT 0.16% 결과: avg daily {fric_016["avg_daily_net"]:+.4f}%, '
              f'pos folds {fric_016["positive_folds"]}/5')

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'mandate': 'thr=0.15 deep audit (Audit 1 friction sensitivity)',
        'pre_commit': 'memory/thr015_deep_audit_precommit_20260501.md',
        'threshold': THRESHOLD,
        'friction_levels': FRICTION_LEVELS,
        'results_by_friction': results_by_friction,
        'pass_012': bool(pass_012),
        'pass_016_critical': bool(pass_016),
        'pass_020': bool(pass_020),
        'audit1_pass': bool(pass_016),
    }
    out_path = RESULTS / f'thr015_audit1_friction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
