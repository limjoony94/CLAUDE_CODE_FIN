#!/usr/bin/env python3
"""
New Pattern Discovery Framework
================================

Systematically explores all 1,728 (12^3) three-candle pattern combinations
to find new tradeable patterns beyond the current 12.

Validation pipeline:
1. Discovery: Grid search TP/SL, minimum trades/WR filter
2. Monte Carlo significance test (sign-flip, N=5000)
3. Walk-Forward validation (5-fold temporal)
4. Period profitability (3 equal periods, require ≥2/3 profitable)
5. Holm-Bonferroni multiple testing correction

References:
- full_270d_revalidation.py (backtest engine)
- constants.py (current patterns, classification thresholds)

Usage:
    python scripts/analysis/discover_new_patterns.py --data data/btc_5m_270days.csv
    python scripts/analysis/discover_new_patterns.py --data data/btc_5m_270days.csv --include-current
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Import production canonical classify_candle
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, os.path.dirname(_PROJECT_ROOT))
from scripts.production.pattern_5m.indicators import classify_candle as _prod_classify

# ============================================================
# CONFIGURATION
# ============================================================

CANDLE_TYPES = ['D', 'DF', 'GS', 'H', 'IH', 'ST', 'MU', 'MD', 'BU', 'BD', 'U', 'DN']

# Classification thresholds (must match production constants.py)
DOJI_BODY_RATIO = 0.10
WICK_DOMINANCE = 0.70
MARUBOZU_WICK_RATIO = 0.15
HAMMER_WICK_TO_BODY = 2.0
HAMMER_OPPOSITE_WICK = 0.3
SPINNING_TOP_NORM = 0.5
SPINNING_TOP_WICK = 0.5
BIG_CANDLE_NORM = 1.5
AVG_BODY_WINDOW = 20

# Trading parameters
LEVERAGE = 3
FEE_PCT = 0.10  # round-trip fee %
MAX_BARS = 500  # max holding period (production standard)

# TP/SL search grid
TP_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SL_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Discovery thresholds
DISC_MIN_TRADES = 15
DISC_MIN_WR = 55.0
DISC_MIN_WF = 3  # out of 5 folds

# Validation thresholds (stricter)
VAL_MIN_TRADES = 15
VAL_MIN_WR = 55.0
VAL_MIN_WF = 4
VAL_MIN_EXCESS_WR = 15.0  # excess over baseline
VAL_MAX_MC_P = 0.05
VAL_MIN_PERIOD_PROFIT = 2  # out of 3 periods

# Monte Carlo
N_MC = 5000

# Current production patterns (to flag as already-known)
CURRENT_LONG = [
    'U-MU-H', 'MD-ST-MD', 'GS-U-BD', 'MD-MD-ST',
    'BU-IH-DN', 'MD-H-MD', 'IH-MD-MD',
]
CURRENT_SHORT = [
    'BD-U-GS', 'DN-GS-H', 'U-DF-BU', 'BD-GS-BD', 'DN-IH-IH',
]


def parse_args():
    parser = argparse.ArgumentParser(description="Discover new tradeable 3-candle patterns")
    parser.add_argument("--data", required=True, help="Path to 5m OHLCV CSV (e.g., data/btc_5m_270days.csv)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: results/pattern_discovery_TIMESTAMP.json)")
    parser.add_argument("--include-current", action="store_true", help="Include currently used patterns in analysis")
    parser.add_argument("--mc-sims", type=int, default=N_MC, help=f"Monte Carlo simulations (default: {N_MC})")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# ============================================================
# DATA PREPARATION
# ============================================================

def classify_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Classify candles using production canonical classify_candle."""
    df = df.copy()
    df['body_abs'] = (df['close'] - df['open']).abs()
    df['avg_body'] = df['body_abs'].rolling(AVG_BODY_WINDOW).mean()

    types = []
    for i, row in df.iterrows():
        avg_b = df.at[i, 'avg_body']
        if pd.isna(avg_b):
            avg_b = 1.0
        types.append(_prod_classify(row, avg_b).value)
    df['candle_type'] = types

    # 3-candle pattern
    df['pattern'] = df['candle_type'].shift(2) + '-' + df['candle_type'].shift(1) + '-' + df['candle_type']

    return df


def build_pattern_index(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Build {pattern: array_of_indices} for fast lookup."""
    index = {}
    patterns = df['pattern'].values
    for i in range(len(df)):
        p = patterns[i]
        if isinstance(p, str):
            index.setdefault(p, []).append(i)
    return {k: np.array(v) for k, v in index.items()}


# ============================================================
# BACKTEST ENGINE
# ============================================================

def fast_backtest(indices: np.ndarray, direction: str, tp_pct: float, sl_pct: float,
                  highs: np.ndarray, lows: np.ndarray, opens: np.ndarray, n_bars: int) -> Dict:
    """Simulate trades with TP/SL on next-bar open entry."""
    pnls = []
    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if direction == 'LONG':
            tp_price = entry * (1 + tp_pct / 100)
            sl_price = entry * (1 - sl_pct / 100)
            for i in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
                if highs[i] >= tp_price:
                    pnls.append(tp_pct * LEVERAGE - FEE_PCT); break
                elif lows[i] <= sl_price:
                    pnls.append(-sl_pct * LEVERAGE - FEE_PCT); break
        else:
            tp_price = entry * (1 - tp_pct / 100)
            sl_price = entry * (1 + sl_pct / 100)
            for i in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
                if lows[i] <= tp_price:
                    pnls.append(tp_pct * LEVERAGE - FEE_PCT); break
                elif highs[i] >= sl_price:
                    pnls.append(-sl_pct * LEVERAGE - FEE_PCT); break
    if not pnls:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'edge': 0, 'pnls': []}
    t = len(pnls)
    w = sum(1 for p in pnls if p > 0)
    return {'trades': t, 'wr': w / t * 100, 'pnl': sum(pnls), 'edge': sum(pnls) / t, 'pnls': pnls}


# ============================================================
# VALIDATION TESTS
# ============================================================

def walk_forward(indices: np.ndarray, direction: str, tp: float, sl: float,
                 highs: np.ndarray, lows: np.ndarray, opens: np.ndarray, n_bars: int,
                 n_folds: int = 5) -> int:
    """Walk-forward: count profitable folds."""
    if len(indices) < 10:
        return 0
    fold_size = len(indices) // n_folds
    profitable = 0
    for i in range(n_folds):
        s = i * fold_size
        e = s + fold_size if i < n_folds - 1 else len(indices)
        r = fast_backtest(indices[s:e], direction, tp, sl, highs, lows, opens, n_bars)
        if r['pnl'] > 0:
            profitable += 1
    return profitable


def period_profitability(indices: np.ndarray, direction: str, tp: float, sl: float,
                         highs: np.ndarray, lows: np.ndarray, opens: np.ndarray, n_bars: int,
                         n_periods: int = 3) -> int:
    """Split data into equal time periods, count profitable ones."""
    if len(indices) < n_periods * 5:
        return 0
    period_size = len(indices) // n_periods
    profitable = 0
    for i in range(n_periods):
        s = i * period_size
        e = s + period_size if i < n_periods - 1 else len(indices)
        r = fast_backtest(indices[s:e], direction, tp, sl, highs, lows, opens, n_bars)
        if r['pnl'] > 0:
            profitable += 1
    return profitable


def mc_test(pnls: list, n_sim: int = N_MC) -> float:
    """Monte Carlo sign-flip test. Returns p-value."""
    if len(pnls) < 5:
        return 1.0
    actual = sum(pnls)
    arr = np.array(pnls)
    count = sum(1 for _ in range(n_sim) if np.sum(arr * np.random.choice([-1, 1], size=len(arr))) >= actual)
    return count / n_sim


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni correction. Returns list of bools (True = significant)."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        if p <= adjusted_alpha:
            significant[orig_idx] = True
        else:
            break  # all remaining are non-significant
    return significant


def compute_baseline_wr(highs, lows, opens, n_bars, direction, tp, sl, n_samples=500) -> float:
    """Random entry baseline WR for excess calculation."""
    rng = np.random.default_rng(42)
    idxs = rng.choice(range(AVG_BODY_WINDOW + 5, n_bars - MAX_BARS - 5), size=min(n_samples, n_bars // 10), replace=False)
    r = fast_backtest(np.sort(idxs), direction, tp, sl, highs, lows, opens, n_bars)
    return r['wr'] if r['trades'] > 0 else 50.0


# ============================================================
# MAIN PIPELINE
# ============================================================

def discover(data_path: str, include_current: bool = False, mc_sims: int = N_MC, verbose: bool = False):
    """Run full discovery pipeline."""
    print('=' * 70)
    print('PATTERN DISCOVERY FRAMEWORK')
    print(f'Started: {datetime.now():%Y-%m-%d %H:%M:%S}')
    print('=' * 70)

    # Load data
    df = pd.read_csv(data_path)
    if 'timestamp' not in df.columns:
        # Try common column names
        for col in ['time', 'date', 'datetime']:
            if col in df.columns:
                df = df.rename(columns={col: 'timestamp'})
                break
    print(f'Loaded: {len(df):,} bars from {data_path}')

    # Classify
    df = classify_candles(df)
    pattern_index = build_pattern_index(df)
    print(f'Unique patterns: {len(pattern_index)}')
    print(f'Type distribution:')
    for t in CANDLE_TYPES:
        count = (df['candle_type'] == t).sum()
        print(f'  {t:3s}: {count:6,} ({count/len(df)*100:.1f}%)')

    # Global arrays for speed
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    n_bars = len(df)

    # Current pattern set
    current_set = set()
    if not include_current:
        for p in CURRENT_LONG:
            current_set.add((p, 'LONG'))
        for p in CURRENT_SHORT:
            current_set.add((p, 'SHORT'))

    # ---- Phase 1: Discovery ----
    print(f'\n{"="*70}')
    print('PHASE 1: DISCOVERY')
    print(f'{"="*70}')

    candidates = []
    checked = 0
    t0 = time.time()

    for c1 in CANDLE_TYPES:
        for c2 in CANDLE_TYPES:
            for c3 in CANDLE_TYPES:
                checked += 1
                pat = f'{c1}-{c2}-{c3}'
                if pat not in pattern_index or len(pattern_index[pat]) < DISC_MIN_TRADES:
                    continue
                idxs = pattern_index[pat]

                for direction in ['LONG', 'SHORT']:
                    if (pat, direction) in current_set:
                        continue

                    # Grid search best TP/SL
                    best_score, best_tp, best_sl, best_r = -999, 1.0, 1.0, None
                    for tp in TP_GRID:
                        for sl in SL_GRID:
                            r = fast_backtest(idxs, direction, tp, sl, highs, lows, opens, n_bars)
                            if r['trades'] < 10 or r['wr'] < 50:
                                continue
                            score = r['wr'] * 0.3 + min(r['edge'], 3.0) * 40 + r['trades'] * 0.1
                            if score > best_score:
                                best_score, best_tp, best_sl, best_r = score, tp, sl, r

                    if best_r and best_r['trades'] >= DISC_MIN_TRADES and best_r['wr'] >= DISC_MIN_WR:
                        wf = walk_forward(idxs, direction, best_tp, best_sl, highs, lows, opens, n_bars)
                        if wf >= DISC_MIN_WF:
                            candidates.append({
                                'pattern': pat, 'direction': direction,
                                'trades': best_r['trades'], 'wr': round(best_r['wr'], 1),
                                'edge': round(best_r['edge'], 3), 'pnl': round(best_r['pnl'], 1),
                                'tp': best_tp, 'sl': best_sl, 'wf': wf,
                                'pnls': best_r['pnls'],
                            })

                if checked % 500 == 0:
                    elapsed = time.time() - t0
                    print(f'  {checked}/1728 ({elapsed:.0f}s) candidates={len(candidates)}')

    print(f'\nPhase 1 complete: {len(candidates)} candidates in {time.time()-t0:.0f}s')

    if not candidates:
        print('No candidates found. Try relaxing thresholds or using more data.')
        return

    # ---- Phase 2: Strict Validation ----
    print(f'\n{"="*70}')
    print('PHASE 2: VALIDATION (MC + WF + Period Profitability)')
    print(f'{"="*70}')

    # Compute baselines
    print('Computing random baselines...')
    baselines = {}
    for tp in set(c['tp'] for c in candidates):
        for sl in set(c['sl'] for c in candidates):
            for d in ['LONG', 'SHORT']:
                key = (d, tp, sl)
                if key not in baselines:
                    baselines[key] = compute_baseline_wr(highs, lows, opens, n_bars, d, tp, sl)

    validated = []
    mc_pvalues = []

    for c in candidates:
        pat, direction = c['pattern'], c['direction']
        tp, sl = c['tp'], c['sl']
        idxs = pattern_index[pat]

        r = fast_backtest(idxs, direction, tp, sl, highs, lows, opens, n_bars)
        if r['trades'] < VAL_MIN_TRADES or r['wr'] < VAL_MIN_WR or r['edge'] <= 0:
            continue

        # Walk-forward
        wf = walk_forward(idxs, direction, tp, sl, highs, lows, opens, n_bars)
        if wf < VAL_MIN_WF:
            continue

        # Period profitability
        pp = period_profitability(idxs, direction, tp, sl, highs, lows, opens, n_bars)
        if pp < VAL_MIN_PERIOD_PROFIT:
            continue

        # Excess WR over baseline
        baseline_wr = baselines.get((direction, tp, sl), 50.0)
        excess_wr = r['wr'] - baseline_wr
        if excess_wr < VAL_MIN_EXCESS_WR:
            continue

        # Monte Carlo
        mc_p = mc_test(r['pnls'], mc_sims)
        mc_pvalues.append(mc_p)

        validated.append({
            'pattern': pat, 'direction': direction,
            'trades': r['trades'], 'wr': round(r['wr'], 1),
            'edge': round(r['edge'], 3), 'pnl': round(r['pnl'], 1),
            'tp': tp, 'sl': sl, 'wf': wf, 'pp': pp,
            'excess_wr': round(excess_wr, 1),
            'mc_p': round(mc_p, 4),
            'baseline_wr': round(baseline_wr, 1),
        })

    # Holm-Bonferroni correction
    if mc_pvalues:
        holm_results = holm_bonferroni(mc_pvalues)
        for i, v in enumerate(validated):
            v['holm_pass'] = holm_results[i]
    else:
        for v in validated:
            v['holm_pass'] = False

    # Filter: only Holm-significant
    final = [v for v in validated if v['holm_pass'] and v['mc_p'] < VAL_MAX_MC_P]
    final.sort(key=lambda x: (-x['wf'], -x['wr'], -x['excess_wr']))

    # ---- Results ----
    print(f'\n{"="*70}')
    print('RESULTS')
    print(f'{"="*70}')
    print(f'Candidates: {len(candidates)}')
    print(f'Validated (pre-Holm): {len(validated)}')
    print(f'Final (Holm-significant): {len(final)}')

    longs = [f for f in final if f['direction'] == 'LONG']
    shorts = [f for f in final if f['direction'] == 'SHORT']

    print(f'\n--- NEW LONG PATTERNS ({len(longs)}) ---')
    for f in longs:
        print(f"  {f['pattern']:12s} | {f['trades']:3d}t | WR {f['wr']:5.1f}% (excess +{f['excess_wr']:.1f}%) | "
              f"E {f['edge']:+.3f} | WF {f['wf']}/5 | PP {f['pp']}/3 | MC={f['mc_p']:.4f} | {f['tp']}/{f['sl']}")

    print(f'\n--- NEW SHORT PATTERNS ({len(shorts)}) ---')
    for f in shorts:
        print(f"  {f['pattern']:12s} | {f['trades']:3d}t | WR {f['wr']:5.1f}% (excess +{f['excess_wr']:.1f}%) | "
              f"E {f['edge']:+.3f} | WF {f['wf']}/5 | PP {f['pp']}/3 | MC={f['mc_p']:.4f} | {f['tp']}/{f['sl']}")

    # Also show rejected-by-Holm for reference
    rejected_holm = [v for v in validated if not v['holm_pass']]
    if rejected_holm and verbose:
        print(f'\n--- REJECTED BY HOLM ({len(rejected_holm)}) ---')
        for r in rejected_holm[:10]:
            print(f"  {r['pattern']:12s} ({r['direction']:5s}) | MC={r['mc_p']:.4f} | WR {r['wr']:.1f}%")

    # Config output
    if final:
        print(f'\n--- SUGGESTED ADDITIONS TO constants.py ---')
        print('# New patterns from discovery:')
        for f in longs:
            print(f"    \"{f['pattern']}\",  # WR {f['wr']}%, {f['trades']}t, excess +{f['excess_wr']}%, MC={f['mc_p']}, WF {f['wf']}/5")
        for f in shorts:
            print(f"    \"{f['pattern']}\",  # WR {f['wr']}%, {f['trades']}t, excess +{f['excess_wr']}%, MC={f['mc_p']}, WF {f['wf']}/5")
        print()
        print('PATTERN_OPTIMAL_TPSL additions:')
        for f in final:
            print(f"    '{f['pattern']}': ({f['tp']}, {f['sl']}),  # {f['direction']}, WR {f['wr']}%, {f['trades']}t")

    # Save
    output_path = args.output
    if not output_path:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        os.makedirs(os.path.join(project_root, 'results'), exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(project_root, 'results', f'pattern_discovery_{ts}.json')

    save_data = {
        'timestamp': datetime.now().isoformat(),
        'data_path': data_path,
        'data_bars': len(df),
        'config': {
            'disc_min_trades': DISC_MIN_TRADES, 'disc_min_wr': DISC_MIN_WR,
            'val_min_trades': VAL_MIN_TRADES, 'val_min_wr': VAL_MIN_WR,
            'val_min_wf': VAL_MIN_WF, 'val_min_excess_wr': VAL_MIN_EXCESS_WR,
            'val_max_mc_p': VAL_MAX_MC_P, 'mc_sims': mc_sims,
        },
        'current_patterns': {'long': CURRENT_LONG, 'short': CURRENT_SHORT},
        'candidates_count': len(candidates),
        'validated_count': len(validated),
        'final': [{k: v for k, v in f.items()} for f in final],
        'rejected_holm': [{k: v for k, v in r.items()} for r in rejected_holm],
    }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f'\nSaved: {output_path}')
    print(f'Completed: {datetime.now():%Y-%m-%d %H:%M:%S}')


if __name__ == '__main__':
    args = parse_args()
    discover(args.data, args.include_current, args.mc_sims, args.verbose)
