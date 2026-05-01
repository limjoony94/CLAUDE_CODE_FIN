"""1h ML with On-Chain Features extension (C — final autonomous attempt).

(C) per (D) sequential mandate. Last attempt before final closure.

Features added to original 12:
  - F&G Index value (Alternative.me, daily forward-fill)
  - Coinmetrics: MVRV ratio, active addresses change, hashrate, exchange flows
  - Composite: F&G z-score, MVRV z-score, address change rate

Total features: 12 (1h technical) + 8 (on-chain daily) = 20
Same logistic regression + L2 reg, 50/25/25 split, |prob-0.5|>0.05 active filter.

Pre-commit: 1 attempt. PASS → deployable, FAIL → final closure (자율 mandate 종료).
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
from bootstrap_validator import bootstrap_validate, report as bootstrap_report

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
FRICTION_PCT = 0.10


def load_onchain_daily():
    # F&G Index
    fng = pd.read_csv(DATA / 'fng_index_history.csv', parse_dates=['date'])
    fng['date'] = fng['date'].dt.tz_localize(None)
    fng = fng.sort_values('date').reset_index(drop=True)
    # Z-score over 90d
    fng['fng_z90'] = (fng['fng_value'] - fng['fng_value'].rolling(90).mean()) / fng['fng_value'].rolling(90).std()

    # Coinmetrics
    cm = pd.read_csv(DATA / 'coinmetrics_btc.csv', parse_dates=['time'])
    cm['date'] = cm['time'].dt.tz_localize(None) if cm['time'].dt.tz is not None else cm['time']
    cm = cm.sort_values('date').reset_index(drop=True)
    # Select features
    cm['mvrv'] = cm['CapMVRVCur']
    cm['mvrv_z90'] = (cm['mvrv'] - cm['mvrv'].rolling(90).mean()) / cm['mvrv'].rolling(90).std()
    cm['adr_chg_30d'] = cm['AdrActCnt'].pct_change(30)
    cm['hashrate_ratio'] = cm['HashRate'] / cm['HashRate'].rolling(30).mean()
    # Exchange flow imbalance
    cm['flow_imbalance'] = (cm['FlowInExUSD'] - cm['FlowOutExUSD']) / (cm['FlowInExUSD'] + cm['FlowOutExUSD'] + 1)

    cm_feat = cm[['date', 'mvrv', 'mvrv_z90', 'adr_chg_30d', 'hashrate_ratio', 'flow_imbalance']]

    # Merge F&G + Coinmetrics
    onchain = pd.merge(fng[['date', 'fng_value', 'fng_z90']], cm_feat, on='date', how='outer')
    onchain = onchain.sort_values('date').reset_index(drop=True)
    onchain = onchain.fillna(method='ffill').dropna()
    return onchain


def main():
    print('=' * 100)
    print('1h ML with On-Chain Features — (C) final autonomous attempt')
    print('=' * 100)

    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    feats = build_features(df)
    feats = feats.merge(df[['timestamp']], left_index=True, right_index=True, how='left')
    feats = feats.dropna(subset=['timestamp'])
    print(f'1h technical features: {feats.shape}')

    # Add on-chain (daily, forward-fill to 1h)
    onchain = load_onchain_daily()
    print(f'On-chain daily features: {onchain.shape}')

    # Forward-fill on-chain to 1h timestamps
    feats['date'] = pd.to_datetime(feats['timestamp']).dt.normalize()
    if feats['date'].dt.tz is not None:
        feats['date'] = feats['date'].dt.tz_localize(None)
    if onchain['date'].dt.tz is not None:
        onchain['date'] = onchain['date'].dt.tz_localize(None)
    feats = pd.merge_asof(
        feats.sort_values('date'),
        onchain.sort_values('date'),
        on='date',
        direction='backward',
    )
    feats = feats.dropna(subset=['fng_value', 'mvrv'])
    feats = feats.sort_values('timestamp').reset_index(drop=True)
    print(f'Combined features: {feats.shape}\n')

    # Feature columns
    tech_cols = ['ret_1h', 'ret_4h', 'ret_24h', 'atr_ratio', 'rsi_14',
                 'ema_ratio', 'vol_z', 'body_ratio', 'range_atr',
                 'close_in_range', 'macd_hist', 'donchian_pos']
    onchain_cols = ['fng_value', 'fng_z90', 'mvrv', 'mvrv_z90',
                     'adr_chg_30d', 'hashrate_ratio', 'flow_imbalance']
    all_cols = tech_cols + onchain_cols
    print(f'Total features: {len(all_cols)} ({len(tech_cols)} tech + {len(onchain_cols)} on-chain)')

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
    y_val = val['target'].values
    X_test = test[all_cols].values
    y_test = test['target'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Train
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)
    p_train = model.predict_proba(X_train_s)[:, 1]
    p_val = model.predict_proba(X_val_s)[:, 1]
    p_test = model.predict_proba(X_test_s)[:, 1]

    print('\n=== Naive accuracy ===')
    print(f'  Train: {(model.predict(X_train_s) == y_train).mean():.3f}')
    print(f'  Val:   {(model.predict(X_val_s) == y_val).mean():.3f}')
    print(f'  Test:  {(model.predict(X_test_s) == y_test).mean():.3f}')

    # Active filter |prob - 0.5| > 0.05
    threshold = 0.05

    def evaluate_predictions(probs, df_segment, name):
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
        trades_df = pd.DataFrame(trades)
        if len(trades_df) == 0:
            return None, None
        hit = (trades_df['gross_pct'] > 0).mean()
        span_days = (df_segment['timestamp'].max() - df_segment['timestamp'].min()).total_seconds() / 86400
        stats = {
            'name': name,
            'n_trades': int(len(trades_df)),
            'hit_rate': float(hit),
            'avg_gross': float(trades_df['gross_pct'].mean()),
            'avg_net': float(trades_df['net_pnl_pct'].mean()),
            'cum_gross': float(trades_df['gross_pct'].sum()),
            'cum_net': float(trades_df['net_pnl_pct'].sum()),
            'daily_net': float(trades_df['net_pnl_pct'].sum() / span_days),
        }
        return trades_df, stats

    print('\n=== Trade evaluation per stage ===')
    for name, seg, probs in [('TRAIN', train, p_train),
                              ('VAL', val, p_val),
                              ('TEST (fresh OOS)', test, p_test)]:
        trades, stats = evaluate_predictions(probs, seg, name)
        if stats:
            print(f'\n  {name}:')
            print(f'    n={stats["n_trades"]}, hit={stats["hit_rate"]:.4f}, '
                  f'avg_gross={stats["avg_gross"]:+.4f}%, avg_net={stats["avg_net"]:+.4f}%, '
                  f'daily_net={stats["daily_net"]:+.4f}%')

    # Test bootstrap
    test_trades, test_stats = evaluate_predictions(p_test, test, 'Test')
    if test_trades is not None and len(test_trades) > 0:
        test_trades['close_ts'] = pd.to_datetime(test_trades['close_ts'])
        ts_min = test_trades['close_ts'].min()
        ts_max = test_trades['close_ts'].max()
        res = bootstrap_validate(test_trades, ts_min, ts_max)
        bootstrap_report(res, 'Test')

        f1 = res.avg_per_trade_pct > 0.07
        f6 = len(test_trades) >= 50
        hit_53 = test_stats['hit_rate'] >= 0.53
        overall = f1 and f6 and res.overall_pass and hit_53

        print(f'\n  F1 avg_per_trade > 0.07: {"✅" if f1 else "🔴"}')
        print(f'  F6 n_trades >= 50: {"✅" if f6 else "🔴"}')
        print(f'  Bootstrap overall: {"✅" if res.overall_pass else "🔴"}')
        print(f'  Hit rate >= 0.53: {"✅" if hit_53 else "🔴"} ({test_stats["hit_rate"]:.4f})')

        # Save
        out = {
            'date': datetime.now(timezone.utc).isoformat(),
            'mandate': '(C) 1h ML with on-chain features (final attempt)',
            'features': all_cols,
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
                'pass_count_6': sum(res.pass_criteria.values()),
                'overall_pass': bool(res.overall_pass),
            },
            'overall_pass': bool(overall),
        }
        out_path = RESULTS / f'h1_ml_onchain_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f'\nSaved: {out_path}')

        # Final verdict
        print('\n' + '=' * 100)
        print('FINAL VERDICT — (C) On-chain features (자율 mandate 마지막 attempt)')
        print('=' * 100)
        if overall:
            print('  🟢 PASS — DEPLOYABLE candidate (on-chain features 추가 → ML edge)')
        else:
            print('  🔴 FAIL — autonomous mandate 종료, final closure')
            print(f'    Hit rate: {test_stats["hit_rate"]:.4f}')
            print(f'    Daily: {test_stats["daily_net"]:+.4f}% (target +0.20%)')
            print(f'    avg_per_trade: {res.avg_per_trade_pct:+.4f}% (need >0.07)')
            print(f'\n  사용자 explicit instruction 필요:')
            print(f'    - Deep learning (LSTM/Transformer)')
            print(f'    - Paid on-chain APIs (Glassnode $200/mo)')
            print(f'    - Different paradigm (market making, options)')
            print(f'    - Capital scale change')
            print(f'    - Honest closure 수용 (Portfolio_EW = +21%/yr Sharpe 1.7)')


if __name__ == '__main__':
    main()
