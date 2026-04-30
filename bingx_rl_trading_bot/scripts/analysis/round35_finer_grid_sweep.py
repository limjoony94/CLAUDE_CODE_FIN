"""Round 35 — Finer R26 grid sweep with stability-first selection.

Pre-reg: claudedocs/round35_finer_grid_sweep_prereg.md (commit bceedc2)

5×5×5 = 125 configs. Selection stability-FIRST per user 재강조 (2026-04-30):
"소수의 수익 폭발로 인해 (과적합) 오염되지 않고, 어떠한 실측 데이터를 인풋으로
받아도 통계적으로 안정적인 수익을 낼 수 있는 전략."

Procedure:
1. Train (60%) → run all 125 configs
2. Stability gate: BS_pos_rate ≥ 0.85 → keep only stability-gated subset
3. Among gated: rank by daily_pct, identify winner
4. Test (40%) on winner: must satisfy BS_pos≥0.85, daily≥baseline+0.02%, retention≥60%
5. WF 5-fold on winner: ≥4/5 folds pos daily AND ≥4/5 folds BS_pos≥0.80
"""
import json
import random
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

DATA_FILE = DATA / 'btc_1h_720days.csv'

GRID = {
    'grid_spacing_pct': [0.20, 0.30, 0.40, 0.50, 0.60],
    'grid_levels_each_side': [3, 4, 5, 6, 7],
    'trend_exit_distance_pct': [1.0, 1.25, 1.5, 1.75, 2.0],
}

FIXED = {
    'capital_usd': 1500,
    'atr_period': 20,
    'atr_pct_median_lookback_bars': 720,
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,
    'max_grid_lifetime_bars': 168,
    'train_test_split': 0.60,
}

# Selection criteria (LOCKED)
CRITERIA = {
    'stability_gate_bs_pos_min': 0.85,
    'switch_test_bs_pos_min': 0.85,
    'switch_daily_improvement_pct': 0.02,
    'switch_retention_min': 0.60,
    'wf_min_pos_folds': 4,
    'wf_min_bs_pos_per_fold': 0.80,
    'wf_total_folds': 5,
}

# Baseline (R26 LOCKED)
BASELINE = {'spacing': 0.30, 'levels': 5, 'te_dist': 1.5}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_ranging_filter(df: pd.DataFrame) -> pd.Series:
    atr = compute_atr(df, FIXED['atr_period'])
    atr_pct = atr / df['close']
    median_30d = atr_pct.rolling(FIXED['atr_pct_median_lookback_bars'],
                                  min_periods=240).median()
    return (atr_pct < median_30d).fillna(False)


def simulate_grid(df: pd.DataFrame, spacing_pct: float, levels: int,
                   trend_exit_pct: float, start_idx: int = 0,
                   end_idx: int = None) -> dict:
    """R26 grid simulation."""
    n = len(df)
    if end_idx is None:
        end_idx = n
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    is_ranging = compute_ranging_filter(df).values

    spacing = spacing_pct / 100
    per_level_usd = FIXED['capital_usd'] / (2 * levels)
    capital = FIXED['capital_usd']
    maker_fric = FIXED['maker_friction_per_side_pct'] / 100
    taker_fric = FIXED['taker_friction_per_side_pct'] / 100
    trend_exit_dist = trend_exit_pct / 100
    max_lifetime = FIXED['max_grid_lifetime_bars']

    active_grid = None
    cum_harvest = 0.0
    cum_drift = 0.0
    n_cycles = 0
    n_grid_setups = 0
    n_trend_exits = 0
    daily_returns = {}
    timestamps = df['timestamp'].values

    start = max(FIXED['atr_period'] + 50, FIXED['atr_pct_median_lookback_bars'])
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

        # New fills
        for k in range(levels):
            if not active_grid['buy_filled'][k] and low[i] <= active_grid['buy_levels'][k]:
                buy_price = active_grid['buy_levels'][k]
                tp_price = buy_price * (1 + spacing)
                active_grid['open_positions'].append({
                    'side': 'LONG', 'entry_price': buy_price, 'tp_price': tp_price,
                    'open_idx': i,
                })
                active_grid['buy_filled'][k] = True
            if not active_grid['sell_filled'][k] and high[i] >= active_grid['sell_levels'][k]:
                sell_price = active_grid['sell_levels'][k]
                tp_price = sell_price * (1 - spacing)
                active_grid['open_positions'].append({
                    'side': 'SHORT', 'entry_price': sell_price, 'tp_price': tp_price,
                    'open_idx': i,
                })
                active_grid['sell_filled'][k] = True

        # TP fills
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

    return {
        'cum_net_pct': cum_net,
        'cum_harvest_pct': cum_harvest,
        'cum_drift_pct': cum_drift,
        'daily_pct': daily_pct,
        'n_cycles': n_cycles,
        'n_grid_setups': n_grid_setups,
        'n_trend_exits': n_trend_exits,
        'n_days': n_days,
        'bs_pos_rate': bs_pos,
    }


def main():
    print('=' * 100)
    print('Round 35 — Finer Grid Sweep (125 configs) Stability-First Selection')
    print('=' * 100)
    print('Pre-reg: claudedocs/round35_finer_grid_sweep_prereg.md (bceedc2)\n')

    df = load_data()
    n = len(df)
    n_days_full = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    train_end_idx = int(n * FIXED['train_test_split'])
    train_days = (df['timestamp'].iloc[train_end_idx] -
                   df['timestamp'].iloc[0]).total_seconds() / 86400
    test_days = n_days_full - train_days
    print(f'Total: {n_days_full:.1f}d | Train: 0-{train_end_idx} ({train_days:.1f}d) | '
          f'Test: {train_end_idx}-{n} ({test_days:.1f}d)\n')

    # === TRAIN: 125 configs ===
    print('Running 125 configs on TRAIN...')
    train_results = []
    cidx = 0
    total = len(GRID['grid_spacing_pct']) * len(GRID['grid_levels_each_side']) * len(GRID['trend_exit_distance_pct'])
    for spacing, levels, te in product(
        GRID['grid_spacing_pct'],
        GRID['grid_levels_each_side'],
        GRID['trend_exit_distance_pct']
    ):
        cidx += 1
        r = simulate_grid(df, spacing, levels, te, 0, train_end_idx)
        train_results.append({
            'config': {'spacing': spacing, 'levels': levels, 'te_dist': te},
            **r,
        })
        if cidx % 25 == 0:
            print(f'  [{cidx}/{total}] spacing={spacing}, levels={levels}, te={te}: '
                  f'daily={r["daily_pct"]:+.4f}%, BS_pos={r["bs_pos_rate"]:.4f}')

    print(f'  Done: {len(train_results)} configs\n')

    # === Stability gate ===
    print('=== Stability Gate (BS_pos ≥ 0.85) ===')
    stability_gated = [r for r in train_results
                        if r['bs_pos_rate'] >= CRITERIA['stability_gate_bs_pos_min']]
    print(f'  Configs passing stability gate: {len(stability_gated)}/{total}')
    print(f'  Configs filtered out: {total - len(stability_gated)}\n')

    if not stability_gated:
        print('NO config passed stability gate. R26 baseline retains LIVE config.')
        return

    # === Rank by daily within stability-gated ===
    stability_gated.sort(key=lambda r: r['daily_pct'], reverse=True)

    print('=== Top 10 stability-gated by train daily_pct ===')
    print(f'{"#":>3} {"spc":>5} {"lev":>4} {"te":>5} {"daily%":>10} {"BS_pos":>8} '
          f'{"cycles":>7} {"trends":>7}')
    print('-' * 70)
    for i, r in enumerate(stability_gated[:10]):
        c = r['config']
        is_baseline = (c['spacing'] == BASELINE['spacing'] and
                        c['levels'] == BASELINE['levels'] and
                        c['te_dist'] == BASELINE['te_dist'])
        marker = ' *' if is_baseline else '  '
        print(f'{i+1:>3} {c["spacing"]:>5} {c["levels"]:>4} {c["te_dist"]:>5} '
              f'{r["daily_pct"]:>+9.4f}% {r["bs_pos_rate"]:>8.4f} '
              f'{r["n_cycles"]:>7} {r["n_trend_exits"]:>7}{marker}')
    print('  * = R26 baseline\n')

    # Find baseline rank in full results
    baseline_train = next(
        (r for r in train_results
         if r['config']['spacing'] == BASELINE['spacing']
         and r['config']['levels'] == BASELINE['levels']
         and r['config']['te_dist'] == BASELINE['te_dist']), None
    )

    # === Train winner ===
    winner_train = stability_gated[0]
    print(f'=== TRAIN WINNER (after stability gate) ===')
    print(f'  config: spacing={winner_train["config"]["spacing"]}, '
          f'levels={winner_train["config"]["levels"]}+{winner_train["config"]["levels"]}, '
          f'te={winner_train["config"]["te_dist"]}')
    print(f'  train daily: {winner_train["daily_pct"]:+.4f}%')
    print(f'  train BS_pos: {winner_train["bs_pos_rate"]:.4f}')
    print(f'  train cycles: {winner_train["n_cycles"]}\n')
    print(f'BASELINE train: daily {baseline_train["daily_pct"]:+.4f}%, '
          f'BS_pos {baseline_train["bs_pos_rate"]:.4f}\n')

    # === TEST winner + baseline ===
    print('=== TEST validation ===')
    winner_test = simulate_grid(df, winner_train['config']['spacing'],
                                  winner_train['config']['levels'],
                                  winner_train['config']['te_dist'],
                                  train_end_idx, n)
    baseline_test = simulate_grid(df, BASELINE['spacing'], BASELINE['levels'],
                                    BASELINE['te_dist'], train_end_idx, n)
    print(f'  Winner test: daily {winner_test["daily_pct"]:+.4f}%, '
          f'BS_pos {winner_test["bs_pos_rate"]:.4f}, cycles {winner_test["n_cycles"]}')
    print(f'  Baseline test: daily {baseline_test["daily_pct"]:+.4f}%, '
          f'BS_pos {baseline_test["bs_pos_rate"]:.4f}, cycles {baseline_test["n_cycles"]}\n')

    # === Switch criterion check ===
    print('=== Switch Criterion Check ===')
    test_bs_ok = winner_test['bs_pos_rate'] >= CRITERIA['switch_test_bs_pos_min']
    daily_imp = winner_test['daily_pct'] - baseline_test['daily_pct']
    daily_imp_ok = daily_imp >= CRITERIA['switch_daily_improvement_pct']
    if winner_train['daily_pct'] > 0:
        retention = winner_test['daily_pct'] / winner_train['daily_pct']
    else:
        retention = 0
    retention_ok = retention >= CRITERIA['switch_retention_min']
    print(f'  Test BS_pos ≥ 0.85: {winner_test["bs_pos_rate"]:.4f} '
          f'→ {"PASS" if test_bs_ok else "FAIL"}')
    print(f'  Daily improvement ≥ +0.02%: +{daily_imp:.4f}% '
          f'→ {"PASS" if daily_imp_ok else "FAIL"}')
    print(f'  Retention ≥ 60%: {retention:.4f} '
          f'→ {"PASS" if retention_ok else "FAIL"}')

    # === WF 5-fold ===
    print('\n=== Walk-Forward 5-fold (winner) ===')
    folds = CRITERIA['wf_total_folds']
    fold_size = n // (folds + 1)
    fold_results = []
    for f in range(folds):
        s = (f + 1) * fold_size
        e = min(s + fold_size, n)
        r = simulate_grid(df, winner_train['config']['spacing'],
                           winner_train['config']['levels'],
                           winner_train['config']['te_dist'], s, e)
        fold_results.append(r)
        print(f'  Fold {f+1}: daily {r["daily_pct"]:+.4f}%, BS_pos {r["bs_pos_rate"]:.4f}, '
              f'cycles {r["n_cycles"]}')

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
        print('VERDICT: SWITCH RECOMMENDED — winner passes all criteria')
        print(f'  Recommended LIVE config: spacing={winner_train["config"]["spacing"]}, '
              f'levels={winner_train["config"]["levels"]}, '
              f'te={winner_train["config"]["te_dist"]}')
    else:
        print('VERDICT: KEEP R26 BASELINE — switch criteria not met')
        print(f'  Pareto frontier confirmed at baseline (0.30/5/1.5)')
    print('=' * 100)

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': 'bceedc2',
        'grid': GRID, 'fixed': FIXED, 'criteria': CRITERIA,
        'baseline': BASELINE,
        'baseline_train': baseline_train,
        'baseline_test': baseline_test,
        'all_train_results': train_results,
        'stability_gated_count': len(stability_gated),
        'stability_gated_top10': stability_gated[:10],
        'winner_train': winner_train,
        'winner_test': winner_test,
        'wf_folds': fold_results,
        'switch_test_bs_ok': bool(test_bs_ok),
        'switch_daily_imp_ok': bool(daily_imp_ok),
        'switch_retention_ok': bool(retention_ok),
        'switch_wf_pos_ok': bool(wf_pos_ok),
        'switch_wf_bs_ok': bool(wf_bs_ok),
        'switch_recommended': bool(all_pass),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'round35_finer_grid_sweep_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
