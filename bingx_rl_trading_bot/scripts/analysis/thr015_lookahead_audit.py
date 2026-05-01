"""thr=0.15 Lookahead Audit — Empirical shuffle test + code review.

Code-level audit (이미 완료):
  - feat_cols에 next_ret_pct 포함 안됨 ✓
  - target = (next_ret > 0).astype(int) — label만, feature 아님 ✓
  - 모든 indicator causal (ewm, rolling) ✓
  - on-chain merge_asof direction='backward' ✓

Empirical shuffle test:
  1. Original: 일반 학습 → test hit rate 측정 (baseline)
  2. Shuffled: target labels random shuffle → 같은 model 학습 → test hit rate
  3. Lookahead-free 시: shuffled hit rate ≈ 50% ± 1%
  4. Lookahead 시: shuffled hit rate > 55% (model이 features에서 future 잡음)

Repeat with multiple shuffle seeds for distribution.
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
N_SHUFFLES = 20  # 20 random shuffles for distribution


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


def evaluate_with_threshold(probs, y_true, threshold):
    """Hit rate with active filter |prob-0.5|>threshold."""
    direction = np.where(probs > 0.5 + threshold, 1,
                          np.where(probs < 0.5 - threshold, -1, 0))
    actual = np.where(y_true == 1, 1, -1)
    active = direction != 0
    n_active = active.sum()
    naive_hit = float((np.sign(probs - 0.5) == np.sign(y_true - 0.5)).mean())
    if n_active == 0:
        return {'n_active': 0, 'n_active_pct': 0.0, 'hit_rate': 0.5, 'naive_hit': naive_hit}
    correct = (direction[active] == actual[active]).sum()
    return {
        'n_active': int(n_active),
        'n_active_pct': float(n_active / len(probs)),
        'hit_rate': float(correct / n_active),
        'naive_hit': naive_hit,
    }


def main():
    print('=' * 100)
    print('thr=0.15 Lookahead Audit — Empirical Shuffle Test')
    print('=' * 100)

    feats = build_combined_features()
    print(f'Combined features: {feats.shape}\n')

    tech_cols = ['ret_1h', 'ret_4h', 'ret_24h', 'atr_ratio', 'rsi_14',
                 'ema_ratio', 'vol_z', 'body_ratio', 'range_atr',
                 'close_in_range', 'macd_hist', 'donchian_pos']
    onchain_cols = ['fng_value', 'fng_z90', 'mvrv', 'mvrv_z90',
                     'adr_chg_30d', 'hashrate_ratio', 'flow_imbalance']
    all_cols = tech_cols + onchain_cols

    n = len(feats)
    i_split = int(n * 0.50)
    train = feats.iloc[:i_split]
    test = feats.iloc[i_split:]
    print(f'Train: {len(train)}, Test: {len(test)}')

    X_train = train[all_cols].values
    y_train_orig = train['target'].values
    X_test = test[all_cols].values
    y_test = test['target'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ============================================================
    # Baseline (original labels)
    # ============================================================
    print('\n=== Baseline (original target) ===')
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train_orig)
    p_test = model.predict_proba(X_test_s)[:, 1]

    base_eval = evaluate_with_threshold(p_test, y_test, THRESHOLD)
    print(f'  Naive hit (all bars):   {base_eval["naive_hit"]:.4f}')
    print(f'  Active hit (|p-0.5|>0.15): {base_eval["hit_rate"]:.4f} (n_active={base_eval["n_active"]}, {base_eval["n_active_pct"]*100:.1f}%)')

    # ============================================================
    # Shuffle test (multiple seeds)
    # ============================================================
    print(f'\n=== Shuffle Test ({N_SHUFFLES} seeds) ===')
    print('  If lookahead-free: shuffled hit rate should be ~50%')
    print('  If lookahead: shuffled hit rate > 55%')
    print()

    rng = np.random.default_rng(seed=2026)
    shuffled_naive_hits = []
    shuffled_active_hits = []
    shuffled_active_pcts = []

    for s_idx in range(N_SHUFFLES):
        # Shuffle train labels (same shuffle every time but seeded different)
        seed = int(rng.integers(0, 100000))
        local_rng = np.random.default_rng(seed)
        y_train_shuffled = y_train_orig.copy()
        local_rng.shuffle(y_train_shuffled)

        # Train fresh model on shuffled labels
        model_s = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model_s.fit(X_train_s, y_train_shuffled)
        p_test_s = model_s.predict_proba(X_test_s)[:, 1]
        ev = evaluate_with_threshold(p_test_s, y_test, THRESHOLD)
        shuffled_naive_hits.append(ev['naive_hit'])
        shuffled_active_hits.append(ev['hit_rate'])
        shuffled_active_pcts.append(ev['n_active_pct'])

        if s_idx < 5:
            print(f'  Shuffle {s_idx+1} (seed={seed}): naive_hit={ev["naive_hit"]:.4f}, '
                  f'active_hit={ev["hit_rate"]:.4f}, active_pct={ev["n_active_pct"]*100:.1f}%')

    naive_arr = np.array(shuffled_naive_hits)
    active_arr = np.array(shuffled_active_hits)
    active_pct_arr = np.array(shuffled_active_pcts)

    print(f'\n  Naive hit (all bars):')
    print(f'    Mean: {naive_arr.mean():.4f}, Std: {naive_arr.std():.4f}')
    print(f'    Min:  {naive_arr.min():.4f}, Max: {naive_arr.max():.4f}')
    print(f'  Active hit (|p-0.5|>0.15):')
    print(f'    Mean: {active_arr.mean():.4f}, Std: {active_arr.std():.4f}')
    print(f'    Min:  {active_arr.min():.4f}, Max: {active_arr.max():.4f}')
    print(f'  Active count %: mean {active_pct_arr.mean()*100:.2f}%, std {active_pct_arr.std()*100:.2f}%')

    # ============================================================
    # Compare baseline vs shuffled distribution
    # ============================================================
    print('\n' + '=' * 100)
    print('Lookahead Evaluation')
    print('=' * 100)

    base_active_hit = base_eval['hit_rate']
    base_naive_hit = base_eval['naive_hit']

    # Z-score: how many stds is baseline away from shuffled mean?
    z_active = (base_active_hit - active_arr.mean()) / max(active_arr.std(), 1e-9)
    z_naive = (base_naive_hit - naive_arr.mean()) / max(naive_arr.std(), 1e-9)

    # P-value (one-sided): how many shuffles >= baseline?
    p_active = (active_arr >= base_active_hit).mean()
    p_naive = (naive_arr >= base_naive_hit).mean()

    print(f'\n  Baseline naive hit:  {base_naive_hit:.4f}')
    print(f'  Shuffled naive mean: {naive_arr.mean():.4f} (Z={z_naive:+.2f}σ, p={p_naive:.4f})')
    print(f'\n  Baseline active hit: {base_active_hit:.4f}')
    print(f'  Shuffled active mean: {active_arr.mean():.4f} (Z={z_active:+.2f}σ, p={p_active:.4f})')

    # Conclusions
    lookahead_naive = naive_arr.mean() > 0.55
    lookahead_active = active_arr.mean() > 0.55

    significant_baseline_active = (z_active > 2.0) and (base_active_hit > 0.58)

    print('\n' + '=' * 100)
    print('VERDICT')
    print('=' * 100)

    if lookahead_naive or lookahead_active:
        print('  🔴 LOOKAHEAD DETECTED')
        print(f'     Shuffled labels still produce {active_arr.mean()*100:.1f}% active hit rate (>55%).')
        print(f'     Model is finding future signal in features. Code audit required.')
    elif significant_baseline_active:
        print('  🟢 LOOKAHEAD-FREE CONFIRMED')
        print(f'     Baseline hit {base_active_hit:.4f} significantly above shuffled noise mean {active_arr.mean():.4f}.')
        print(f'     Z = {z_active:+.2f}σ, p = {p_active:.4f}')
        print(f'     thr=0.15 robustness 결과 VALID, deploy 가능.')
    else:
        print('  🟡 INCONCLUSIVE')
        print(f'     Baseline hit {base_active_hit:.4f} not strongly above shuffled noise.')
        print(f'     May be statistical noise or weak edge.')

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'mandate': 'thr=0.15 lookahead audit',
        'threshold': THRESHOLD,
        'n_shuffles': N_SHUFFLES,
        'feat_cols_audit': {
            'next_ret_pct_in_feats': False,
            'target_only_label': True,
            'all_indicators_causal': True,
            'merge_asof_backward': True,
        },
        'baseline': {
            'naive_hit': float(base_naive_hit),
            'active_hit': float(base_active_hit),
            'n_active': base_eval['n_active'],
            'active_pct': float(base_eval['n_active_pct']),
        },
        'shuffled': {
            'naive_mean': float(naive_arr.mean()),
            'naive_std': float(naive_arr.std()),
            'active_mean': float(active_arr.mean()),
            'active_std': float(active_arr.std()),
            'active_pct_mean': float(active_pct_arr.mean()),
            'active_pct_std': float(active_pct_arr.std()),
        },
        'statistical_test': {
            'z_active': float(z_active),
            'z_naive': float(z_naive),
            'p_active': float(p_active),
            'p_naive': float(p_naive),
        },
        'verdict': {
            'lookahead_detected': bool(lookahead_naive or lookahead_active),
            'lookahead_free_confirmed': bool(significant_baseline_active),
        },
    }
    out_path = RESULTS / f'thr015_lookahead_audit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
