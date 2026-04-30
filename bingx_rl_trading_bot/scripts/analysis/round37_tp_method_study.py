"""Round 37 — TP method study (4 methods, single round).

Pre-reg: claudedocs/round37_tp_method_study_prereg.md (commit 11b1026)
"""
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

DATA_FILE = DATA / 'btc_1h_720days.csv'

LOCKED = {
    'capital_usd': 1500,
    'grid_spacing_pct': 0.30,
    'grid_levels_each_side': 5,
    'trend_exit_distance_pct': 1.5,
    'atr_period_for_TP': 14,
    'atr_period_for_ranging': 20,
    'atr_pct_median_lookback_bars': 720,
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,
    'max_grid_lifetime_bars': 168,
    'tp_cap_pct': 0.50,
    'train_test_split': 0.60,
}

CRITERIA = {
    'stability_gate_bs_pos_min': 0.85,
    'switch_test_bs_pos_min': 0.85,
    'switch_daily_improvement_pct': 0.02,
    'switch_retention_min': 0.60,
    'wf_min_pos_folds': 4,
    'wf_min_bs_pos_per_fold': 0.80,
    'wf_total_folds': 5,
}

TP_METHODS = ['M1', 'M2', 'M3', 'M4']
BASELINE_METHOD = 'M1'


def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def compute_atr(df, period):
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_ranging_filter(df):
    atr = compute_atr(df, LOCKED['atr_period_for_ranging'])
    atr_pct = atr / df['close']
    median = atr_pct.rolling(LOCKED['atr_pct_median_lookback_bars'], min_periods=240).median()
    return (atr_pct < median).fillna(False)


def tp_distance_pct(method, atr_pct_at_fill):
    """LOCKED formulas per pre-reg."""
    cap = LOCKED['tp_cap_pct']
    if method == 'M1':
        return 0.30
    elif method == 'M2':
        return min(cap, 0.5 * atr_pct_at_fill)
    elif method == 'M3':
        return min(cap, 1.0 * atr_pct_at_fill)
    elif method == 'M4':
        return max(0.30, min(cap, 0.5 * atr_pct_at_fill))
    raise ValueError(f"Unknown TP method: {method}")


def simulate_grid(df, tp_method, start_idx=0, end_idx=None):
    n = len(df)
    if end_idx is None:
        end_idx = n
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    is_ranging = compute_ranging_filter(df).values
    atr_tp = compute_atr(df, LOCKED['atr_period_for_TP']).values
    atr_pct_tp = atr_tp / close * 100  # ATR as % of price

    spacing = LOCKED['grid_spacing_pct'] / 100
    levels = LOCKED['grid_levels_each_side']
    per_level_usd = LOCKED['capital_usd'] / (2 * levels)
    capital = LOCKED['capital_usd']
    maker_fric = LOCKED['maker_friction_per_side_pct'] / 100
    taker_fric = LOCKED['taker_friction_per_side_pct'] / 100
    trend_exit_dist = LOCKED['trend_exit_distance_pct'] / 100
    max_lifetime = LOCKED['max_grid_lifetime_bars']

    active_grid = None
    cum_harvest = 0.0
    cum_drift = 0.0
    n_cycles = 0
    n_grid_setups = 0
    n_trend_exits = 0
    daily_returns = {}
    timestamps = df['timestamp'].values
    tp_distances_used = []  # for diagnostic

    start = max(LOCKED['atr_period_for_ranging'] + 50,
                  LOCKED['atr_pct_median_lookback_bars'])
    sim_start = max(start, start_idx)

    for i in range(sim_start, end_idx):
        if active_grid is None and is_ranging[i]:
            init_mid = close[i]
            buy_levels = [init_mid * (1 - spacing * (k + 1)) for k in range(levels)]
            sell_levels = [init_mid * (1 + spacing * (k + 1)) for k in range(levels)]
            active_grid = {
                'init_mid': init_mid, 'init_idx': i,
                'buy_levels': buy_levels, 'sell_levels': sell_levels,
                'buy_filled': [False] * levels, 'sell_filled': [False] * levels,
                'open_positions': [],
            }
            n_grid_setups += 1

        if active_grid is None:
            continue

        elapsed = i - active_grid['init_idx']
        price_dist = abs(close[i] - active_grid['init_mid']) / active_grid['init_mid']
        force_exit = False
        if elapsed >= max_lifetime:
            force_exit = True
        elif price_dist > trend_exit_dist and not is_ranging[i]:
            force_exit = True
            n_trend_exits += 1

        if force_exit:
            for p in active_grid['open_positions']:
                if p['side'] == 'LONG':
                    pnl_pct = (close[i] - p['entry_price']) / p['entry_price'] * 100
                else:
                    pnl_pct = (p['entry_price'] - close[i]) / p['entry_price'] * 100
                fric_pct = taker_fric * 100
                contrib = (pnl_pct - fric_pct) * (per_level_usd / capital)
                cum_drift += contrib
                d = pd.to_datetime(timestamps[i]).floor('D')
                daily_returns[d] = daily_returns.get(d, 0) + contrib
            active_grid = None
            continue

        # Check fills + place TP per method
        for k in range(levels):
            if not active_grid['buy_filled'][k] and low[i] <= active_grid['buy_levels'][k]:
                buy_price = active_grid['buy_levels'][k]
                # Compute TP distance per method (use ATR at fill time = bar i)
                atr_pct_now = atr_pct_tp[i] if not np.isnan(atr_pct_tp[i]) else 0.30
                tp_dist_pct = tp_distance_pct(tp_method, atr_pct_now)
                tp_price = buy_price * (1 + tp_dist_pct / 100)
                tp_distances_used.append(tp_dist_pct)
                active_grid['open_positions'].append({
                    'side': 'LONG', 'entry_price': buy_price, 'tp_price': tp_price,
                    'tp_dist_pct': tp_dist_pct, 'open_idx': i,
                })
                active_grid['buy_filled'][k] = True
            if not active_grid['sell_filled'][k] and high[i] >= active_grid['sell_levels'][k]:
                sell_price = active_grid['sell_levels'][k]
                atr_pct_now = atr_pct_tp[i] if not np.isnan(atr_pct_tp[i]) else 0.30
                tp_dist_pct = tp_distance_pct(tp_method, atr_pct_now)
                tp_price = sell_price * (1 - tp_dist_pct / 100)
                tp_distances_used.append(tp_dist_pct)
                active_grid['open_positions'].append({
                    'side': 'SHORT', 'entry_price': sell_price, 'tp_price': tp_price,
                    'tp_dist_pct': tp_dist_pct, 'open_idx': i,
                })
                active_grid['sell_filled'][k] = True

        # Check TP fills
        new_open = []
        for p in active_grid['open_positions']:
            tp_hit = False
            if p['side'] == 'LONG' and high[i] >= p['tp_price']:
                tp_hit = True
                exit_price = p['tp_price']
            elif p['side'] == 'SHORT' and low[i] <= p['tp_price']:
                tp_hit = True
                exit_price = p['tp_price']
            if tp_hit:
                if p['side'] == 'LONG':
                    pnl_pct = (exit_price - p['entry_price']) / p['entry_price'] * 100
                else:
                    pnl_pct = (p['entry_price'] - exit_price) / p['entry_price'] * 100
                fric_pct = maker_fric * 100
                contrib = (pnl_pct - fric_pct) * (per_level_usd / capital)
                cum_harvest += contrib
                n_cycles += 1
                d = pd.to_datetime(timestamps[i]).floor('D')
                daily_returns[d] = daily_returns.get(d, 0) + contrib
            else:
                new_open.append(p)
        active_grid['open_positions'] = new_open

    cum_net = cum_harvest + cum_drift
    n_days = (end_idx - sim_start) / 24
    daily_pct = cum_net / n_days if n_days > 0 else 0

    daily_arr = np.array(list(daily_returns.values()))
    if len(daily_arr) > 4:
        random.seed(42)
        n_iter = min(1000, len(daily_arr) - 3)
        starts = random.sample(range(len(daily_arr) - 3), n_iter)
        cums = [daily_arr[s:s + 3].sum() for s in starts]
        bs_pos = float((np.array(cums) > 0).mean())
    else:
        bs_pos = 0

    avg_tp_dist = float(np.mean(tp_distances_used)) if tp_distances_used else 0
    return {
        'tp_method': tp_method,
        'cum_net_pct': cum_net,
        'cum_harvest_pct': cum_harvest,
        'cum_drift_pct': cum_drift,
        'daily_pct': daily_pct,
        'n_cycles': n_cycles,
        'n_grid_setups': n_grid_setups,
        'n_trend_exits': n_trend_exits,
        'bs_pos_rate': bs_pos,
        'avg_tp_dist_pct': avg_tp_dist,
    }


def main():
    print('=' * 100)
    print('Round 37 — TP Method Study (4 methods)')
    print('=' * 100)
    print('Pre-reg: 11b1026\n')

    df = load_data()
    n = len(df)
    train_end_idx = int(n * LOCKED['train_test_split'])
    train_days = (df['timestamp'].iloc[train_end_idx] -
                   df['timestamp'].iloc[0]).total_seconds() / 86400
    test_days = ((df['timestamp'].iloc[-1] - df['timestamp'].iloc[train_end_idx]).total_seconds() / 86400)
    print(f'Train: {train_days:.1f}d | Test: {test_days:.1f}d\n')

    print('Running 4 TP methods on TRAIN...')
    train_results = []
    for method in TP_METHODS:
        r = simulate_grid(df, method, 0, train_end_idx)
        train_results.append(r)
        marker = ' *' if method == BASELINE_METHOD else '  '
        print(f'  {method}: avg_tp={r["avg_tp_dist_pct"]:.4f}%, '
              f'daily={r["daily_pct"]:+.4f}%, BS_pos={r["bs_pos_rate"]:.4f}, '
              f'cycles={r["n_cycles"]}{marker}')
    print(f'  * = M1 baseline\n')

    # Stability gate
    stability_gated = [r for r in train_results
                        if r['bs_pos_rate'] >= CRITERIA['stability_gate_bs_pos_min']]
    print(f'Stability gate (BS_pos ≥ 0.85): {len(stability_gated)}/{len(train_results)}\n')

    if not stability_gated:
        print('NO method passed stability gate. M1 baseline retained.')
        return

    stability_gated.sort(key=lambda r: r['daily_pct'], reverse=True)
    winner = stability_gated[0]
    baseline = next(r for r in train_results if r['tp_method'] == BASELINE_METHOD)

    print(f'TRAIN WINNER: {winner["tp_method"]}')
    print(f'  daily={winner["daily_pct"]:+.4f}%, BS_pos={winner["bs_pos_rate"]:.4f}, '
          f'avg_tp={winner["avg_tp_dist_pct"]:.4f}%\n')
    print(f'M1 BASELINE: daily={baseline["daily_pct"]:+.4f}%, '
          f'BS_pos={baseline["bs_pos_rate"]:.4f}\n')

    # Test
    winner_test = simulate_grid(df, winner['tp_method'], train_end_idx, n)
    baseline_test = simulate_grid(df, BASELINE_METHOD, train_end_idx, n)
    print(f'TEST validation:')
    print(f'  Winner ({winner["tp_method"]}) test: daily={winner_test["daily_pct"]:+.4f}%, '
          f'BS_pos={winner_test["bs_pos_rate"]:.4f}')
    print(f'  M1 baseline test: daily={baseline_test["daily_pct"]:+.4f}%, '
          f'BS_pos={baseline_test["bs_pos_rate"]:.4f}\n')

    test_bs_ok = winner_test['bs_pos_rate'] >= CRITERIA['switch_test_bs_pos_min']
    daily_imp = winner_test['daily_pct'] - baseline_test['daily_pct']
    daily_imp_ok = daily_imp >= CRITERIA['switch_daily_improvement_pct']
    if winner['daily_pct'] > 0:
        retention = winner_test['daily_pct'] / winner['daily_pct']
    else:
        retention = 0
    retention_ok = retention >= CRITERIA['switch_retention_min']
    print(f'Switch criterion:')
    print(f'  Test BS_pos ≥ 0.85: {winner_test["bs_pos_rate"]:.4f} → {"PASS" if test_bs_ok else "FAIL"}')
    print(f'  Daily imp ≥ 0.02%: +{daily_imp:.4f}% → {"PASS" if daily_imp_ok else "FAIL"}')
    print(f'  Retention ≥ 60%: {retention:.4f} → {"PASS" if retention_ok else "FAIL"}')

    # WF
    folds = CRITERIA['wf_total_folds']
    fold_size = n // (folds + 1)
    fold_results = []
    print(f'\nWF 5-fold (winner {winner["tp_method"]}):')
    for f in range(folds):
        s = (f + 1) * fold_size
        e = min(s + fold_size, n)
        r = simulate_grid(df, winner['tp_method'], s, e)
        fold_results.append(r)
        print(f'  Fold {f+1}: daily={r["daily_pct"]:+.4f}%, BS_pos={r["bs_pos_rate"]:.4f}')

    n_pos = sum(1 for r in fold_results if r['daily_pct'] > 0)
    n_bs_ok = sum(1 for r in fold_results
                    if r['bs_pos_rate'] >= CRITERIA['wf_min_bs_pos_per_fold'])
    wf_pos_ok = n_pos >= CRITERIA['wf_min_pos_folds']
    wf_bs_ok = n_bs_ok >= CRITERIA['wf_min_pos_folds']
    print(f'  Folds positive: {n_pos}/5 → {"PASS" if wf_pos_ok else "FAIL"}')
    print(f'  Folds BS_pos ≥ 0.80: {n_bs_ok}/5 → {"PASS" if wf_bs_ok else "FAIL"}\n')

    all_pass = test_bs_ok and daily_imp_ok and retention_ok and wf_pos_ok and wf_bs_ok

    print('=' * 100)
    if all_pass:
        print(f'VERDICT: SWITCH RECOMMENDED — TP method = {winner["tp_method"]}')
    else:
        print('VERDICT: KEEP M1 BASELINE (TP formula not binding constraint)')
    print('=' * 100)

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '11b1026',
        'locked': LOCKED, 'criteria': CRITERIA,
        'tp_methods': TP_METHODS,
        'all_train': train_results,
        'baseline_train': baseline,
        'baseline_test': baseline_test,
        'winner_train': winner,
        'winner_test': winner_test,
        'wf_folds': fold_results,
        'switch_recommended': bool(all_pass),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'round37_tp_method_study_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
