"""
v1.25.0 Moderate-B-20 Portfolio Validation (Fixed - Discovery-Consistent)
Using identical backtest logic as unified_pattern_discovery.py

Created: 2026-02-04
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys

# Import production classify_candle
sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle

# ============================================================
# CONFIGURATION (matching discovery)
# ============================================================

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "btc_5m_270days.csv"
RESULTS_PATH = Path(__file__).parent.parent.parent / "results"

MAX_BARS = 500
FEE_PCT = 0.10  # Single fee (matching discovery)
LEVERAGE = 3

# v1.25.0 Moderate-B-20 Portfolio (from unified_pattern_discovery.json)
# Using FULL NAMES to match discovery script output
PATTERN_CONFIG = {
    # LONG patterns (10)
    "MARUBOZU_DOWN-BIG_UP-MED_UP": {"direction": "LONG", "tp": 0.5, "sl": 2.0},
    "MARUBOZU_UP-MARUBOZU_UP-MED_UP": {"direction": "LONG", "tp": 0.7, "sl": 1.5},
    "MARUBOZU_UP-MED_UP-MARUBOZU_UP": {"direction": "LONG", "tp": 1.5, "sl": 2.0},
    "BIG_UP-BIG_UP-BIG_DOWN": {"direction": "LONG", "tp": 0.7, "sl": 1.5},
    "SPINNING_TOP-HAMMER-MED_DOWN": {"direction": "LONG", "tp": 0.3, "sl": 0.7},
    "SPINNING_TOP-MARUBOZU_UP-MED_UP": {"direction": "LONG", "tp": 0.5, "sl": 1.0},
    "MED_DOWN-INV_HAMMER-SPINNING_TOP": {"direction": "LONG", "tp": 0.5, "sl": 0.7},
    "INV_HAMMER-MED_DOWN-MED_DOWN": {"direction": "LONG", "tp": 0.7, "sl": 1.0},
    "MARUBOZU_DOWN-MED_DOWN-MARUBOZU_UP": {"direction": "LONG", "tp": 1.0, "sl": 1.0},
    "BIG_DOWN-SPINNING_TOP-MED_UP": {"direction": "LONG", "tp": 1.5, "sl": 1.5},
    # SHORT patterns (10)
    "MARUBOZU_DOWN-SPINNING_TOP-SPINNING_TOP": {"direction": "SHORT", "tp": 0.5, "sl": 2.0},
    "MED_UP-MARUBOZU_UP-BIG_UP": {"direction": "SHORT", "tp": 0.5, "sl": 2.0},
    "MARUBOZU_UP-BIG_UP-MED_DOWN": {"direction": "SHORT", "tp": 0.7, "sl": 2.0},
    "SPINNING_TOP-HAMMER-MED_UP": {"direction": "SHORT", "tp": 0.5, "sl": 2.0},
    "SPINNING_TOP-MED_DOWN-HAMMER": {"direction": "SHORT", "tp": 0.7, "sl": 2.0},
    "MARUBOZU_DOWN-MARUBOZU_UP-MED_UP": {"direction": "SHORT", "tp": 1.0, "sl": 2.0},
    "BIG_UP-MED_UP-SPINNING_TOP": {"direction": "SHORT", "tp": 0.5, "sl": 1.5},
    "HAMMER-MED_DOWN-SPINNING_TOP": {"direction": "SHORT", "tp": 0.5, "sl": 1.5},
    "MED_DOWN-BIG_DOWN-BIG_UP": {"direction": "SHORT", "tp": 0.7, "sl": 1.0},
    "MED_DOWN-BIG_UP-MED_UP": {"direction": "SHORT", "tp": 1.0, "sl": 1.0},
}


def classify_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Classify candles using production-identical logic."""
    df = df.copy()
    body_abs = np.abs(df['close'] - df['open'])
    avg_body_20 = body_abs.rolling(20).mean().values

    types = []
    for i in range(len(df)):
        row = df.iloc[i]
        avg_b = avg_body_20[i] if not pd.isna(avg_body_20[i]) else 1.0
        ct = classify_candle(row, avg_b)
        types.append(ct.name)  # Use full name (MED_UP, etc.) - matching discovery

    df['candle_type'] = types
    return df


def bt_fixed(indices: np.ndarray, direction: str, tp_pct: float, sl_pct: float,
             opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, n_bars: int,
             slippage_pct: float = 0.0, extra_fee: float = 0.0) -> List[float]:
    """
    Backtest function identical to unified_pattern_discovery.py
    Returns list of per-trade PnL percentages (leveraged).
    """
    pnls = []
    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue

        # Apply slippage to entry
        if slippage_pct > 0:
            if direction == 'LONG':
                entry *= (1 + slippage_pct / 100)
            else:
                entry *= (1 - slippage_pct / 100)

        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)

        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)

            if ht and hs:
                # Both hit in same bar - use distance-based decision
                if abs(tpp - entry) <= abs(slp - entry):
                    pnl = tp_pct * LEVERAGE - FEE_PCT - extra_fee
                else:
                    pnl = -sl_pct * LEVERAGE - FEE_PCT - extra_fee
                pnls.append(pnl)
                break
            elif ht:
                pnl = tp_pct * LEVERAGE - FEE_PCT - extra_fee
                pnls.append(pnl)
                break
            elif hs:
                pnl = -sl_pct * LEVERAGE - FEE_PCT - extra_fee
                pnls.append(pnl)
                break

    return pnls


def backtest_portfolio(df: pd.DataFrame, slippage_pct: float = 0.0, extra_fee: float = 0.0) -> Dict:
    """Backtest entire portfolio with discovery-identical logic."""
    # Build arrays
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)
    types = df['candle_type'].values

    # Build pattern indices
    pattern_indices = {}
    for i in range(2, n_bars):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in pattern_indices:
            pattern_indices[pat] = []
        pattern_indices[pat].append(i)

    # Convert to numpy arrays
    for k in pattern_indices:
        pattern_indices[k] = np.array(pattern_indices[k])

    # Backtest each pattern
    results = {'trades': [], 'by_pattern': {}}

    for pattern, config in PATTERN_CONFIG.items():
        if pattern not in pattern_indices:
            continue

        indices = pattern_indices[pattern]
        direction = config['direction']
        tp = config['tp']
        sl = config['sl']

        pnls = bt_fixed(indices, direction, tp, sl, opens, highs, lows, n_bars, slippage_pct, extra_fee)

        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100 if pnls else 0
        total_pnl = sum(pnls)

        results['by_pattern'][pattern] = {
            'direction': direction,
            'trades': len(pnls),
            'wins': wins,
            'wr': wr,
            'total_pnl': total_pnl,
            'pnls': pnls
        }

        for p in pnls:
            results['trades'].append({
                'pattern': pattern,
                'direction': direction,
                'pnl': p,
                'win': p > 0
            })

    return results


def calculate_metrics(results: Dict) -> Dict:
    """Calculate portfolio-level metrics."""
    trades = results['trades']
    if not trades:
        return {'wr': 0, 'total_pnl': 0, 'trades': 0, 'pf': 0}

    wins = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]

    total_pnl = sum(t['pnl'] for t in trades)
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.001

    return {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'wr': len(wins) / len(trades) * 100,
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / len(trades),
        'pf': gross_profit / gross_loss if gross_loss > 0 else 999
    }


def bootstrap_ci(trades: List[Dict], n_bootstrap: int = 10000) -> Dict:
    """Bootstrap confidence intervals."""
    n_trades = len(trades)
    trade_array = np.array([(1 if t['win'] else 0, t['pnl']) for t in trades])

    wr_samples, pnl_samples = [], []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_trades, size=n_trades, replace=True)
        sample = trade_array[idx]
        wr_samples.append(sample[:, 0].mean() * 100)
        pnl_samples.append(sample[:, 1].sum())

    return {
        'wr_ci': (np.percentile(wr_samples, 2.5), np.percentile(wr_samples, 97.5)),
        'pnl_ci': (np.percentile(pnl_samples, 2.5), np.percentile(pnl_samples, 97.5)),
    }


def walk_forward_validation(df: pd.DataFrame, n_folds: int = 5) -> Dict:
    """Walk-forward validation."""
    n_bars = len(df)
    fold_size = n_bars // n_folds
    results = []

    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else n_bars
        fold_df = df.iloc[start_idx:end_idx].copy()

        res = backtest_portfolio(fold_df)
        metrics = calculate_metrics(res)
        results.append({
            'fold': fold + 1,
            'wr': metrics['wr'],
            'pnl': metrics['total_pnl'],
            'trades': metrics['trades']
        })

    profitable_folds = sum(1 for r in results if r['pnl'] > 0)
    return {
        'folds': results,
        'profitable_folds': profitable_folds,
        'wf_score': f"{profitable_folds}/{n_folds}"
    }


def main():
    print("=" * 70)
    print("v1.25.0 MODERATE-B-20 VALIDATION (Discovery-Consistent)")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Total bars: {len(df):,}")

    # Classify using production logic
    print("\nClassifying candles (production-identical)...")
    df = classify_dataset(df)
    print("Classification complete.")

    # Baseline backtest
    print("\n" + "-" * 70)
    print("BASELINE BACKTEST (Leverage=3, Fee=0.10%)")
    print("-" * 70)

    results = backtest_portfolio(df)
    metrics = calculate_metrics(results)

    print(f"\nPortfolio Results:")
    print(f"  Total Trades: {metrics['trades']}")
    print(f"  Win Rate: {metrics['wr']:.1f}%")
    print(f"  Total PnL: {metrics['total_pnl']:.1f}%")
    print(f"  Profit Factor: {metrics['pf']:.2f}")

    # Per-pattern breakdown
    print("\n--- Per-Pattern Results ---")
    print(f"{'Pattern':<12} {'Dir':<6} {'Trades':>6} {'WR':>7} {'PnL':>9}")
    print("-" * 50)
    for pattern, data in sorted(results['by_pattern'].items(), key=lambda x: -x[1]['total_pnl']):
        print(f"{pattern:<12} {data['direction']:<6} {data['trades']:>6} {data['wr']:>6.1f}% {data['total_pnl']:>8.1f}%")

    # Bootstrap CI
    print("\n" + "-" * 70)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("-" * 70)

    ci = bootstrap_ci(results['trades'])
    print(f"  Win Rate 95% CI: [{ci['wr_ci'][0]:.1f}%, {ci['wr_ci'][1]:.1f}%]")
    print(f"  Total PnL 95% CI: [{ci['pnl_ci'][0]:.1f}%, {ci['pnl_ci'][1]:.1f}%]")

    bootstrap_pass = ci['wr_ci'][0] > 50 and ci['pnl_ci'][0] > 0
    print(f"\n  Bootstrap Validation: {'✅ PASS' if bootstrap_pass else '❌ FAIL'}")

    # Walk-forward
    print("\n" + "-" * 70)
    print("WALK-FORWARD VALIDATION (5-Fold)")
    print("-" * 70)

    wf = walk_forward_validation(df)
    print(f"\n{'Fold':>6} {'WR':>8} {'PnL':>10} {'Trades':>8}")
    print("-" * 40)
    for f in wf['folds']:
        print(f"{f['fold']:>6} {f['wr']:>7.1f}% {f['pnl']:>9.1f}% {f['trades']:>8}")
    print(f"\n  WF Score: {wf['wf_score']}")
    wf_pass = wf['profitable_folds'] >= 4
    print(f"  WF Validation: {'✅ PASS' if wf_pass else '❌ FAIL'}")

    # Stress test (slippage)
    print("\n" + "-" * 70)
    print("SLIPPAGE STRESS TEST")
    print("-" * 70)

    slippage_levels = [0.0, 0.05, 0.10, 0.15]
    stress_results = []

    print(f"\n{'Slippage':>10} {'Trades':>8} {'WR':>8} {'PnL':>10} {'PF':>8}")
    print("-" * 50)

    for slip in slippage_levels:
        res = backtest_portfolio(df, slippage_pct=slip)
        m = calculate_metrics(res)
        stress_results.append({'slip': slip, **m})
        status = '✅' if m['wr'] > 50 and m['total_pnl'] > 0 else '❌'
        print(f"{slip*100:>9.0f}bp {m['trades']:>8} {m['wr']:>7.1f}% {m['total_pnl']:>9.1f}% {m['pf']:>7.2f}  {status}")

    stress_pass = sum(1 for r in stress_results if r['wr'] > 50 and r['total_pnl'] > 0) >= 3

    # OOS validation (last 60 days)
    print("\n" + "-" * 70)
    print("OUT-OF-SAMPLE VALIDATION (Last 60 Days)")
    print("-" * 70)

    test_bars = 60 * 24 * 12  # 60 days
    train_df = df.iloc[:-test_bars].copy()
    test_df = df.iloc[-test_bars:].copy()

    train_res = backtest_portfolio(train_df)
    test_res = backtest_portfolio(test_df)
    train_m = calculate_metrics(train_res)
    test_m = calculate_metrics(test_res)

    print(f"\n{'Period':<12} {'Trades':>8} {'WR':>8} {'PnL':>10} {'PF':>8}")
    print("-" * 50)
    print(f"{'Train':<12} {train_m['trades']:>8} {train_m['wr']:>7.1f}% {train_m['total_pnl']:>9.1f}% {train_m['pf']:>7.2f}")
    print(f"{'Test (OOS)':<12} {test_m['trades']:>8} {test_m['wr']:>7.1f}% {test_m['total_pnl']:>9.1f}% {test_m['pf']:>7.2f}")

    wr_degradation = test_m['wr'] - train_m['wr']
    print(f"\n  WR Degradation: {wr_degradation:+.1f}%")

    oos_pass = test_m['wr'] > 55 and test_m['total_pnl'] > 0

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_pass = bootstrap_pass and wf_pass and stress_pass and oos_pass

    print(f"\n1. Bootstrap CI:      {'✅ PASS' if bootstrap_pass else '❌ FAIL'}")
    print(f"2. Walk-Forward:      {'✅ PASS' if wf_pass else '❌ FAIL'} ({wf['wf_score']})")
    print(f"3. Stress Test:       {'✅ PASS' if stress_pass else '❌ FAIL'}")
    print(f"4. Out-of-Sample:     {'✅ PASS' if oos_pass else '❌ FAIL'}")

    print("\n" + "=" * 70)
    print(f"FINAL VERDICT: {'✅ ALL TESTS PASSED' if all_pass else '⚠️ SOME TESTS FAILED'}")
    print("=" * 70)

    # Save results
    output = {
        'version': 'v1.25.0',
        'portfolio': 'Moderate-B-20',
        'baseline': {
            'trades': int(metrics['trades']),
            'wr': float(metrics['wr']),
            'total_pnl': float(metrics['total_pnl']),
            'pf': float(metrics['pf'])
        },
        'bootstrap': {
            'wr_ci': [float(ci['wr_ci'][0]), float(ci['wr_ci'][1])],
            'pnl_ci': [float(ci['pnl_ci'][0]), float(ci['pnl_ci'][1])],
            'passed': bool(bootstrap_pass)
        },
        'walk_forward': {
            'score': wf['wf_score'],
            'passed': bool(wf_pass)
        },
        'stress_test': {
            'passed': bool(stress_pass)
        },
        'oos': {
            'test_wr': float(test_m['wr']),
            'test_pnl': float(test_m['total_pnl']),
            'degradation': float(wr_degradation),
            'passed': bool(oos_pass)
        },
        'all_passed': bool(all_pass)
    }

    output_path = RESULTS_PATH / 'v125_validation_fixed.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()
