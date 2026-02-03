"""
Portfolio Size Optimization - Finding Optimal Balance
Backtest different portfolio sizes to find the best quality vs quantity trade-off

Created: 2026-02-04
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Type mapping for short codes
TYPE_MAP = {
    'DOJI': 'D', 'DRAGONFLY': 'DF', 'GRAVESTONE': 'GS',
    'HAMMER': 'H', 'INV_HAMMER': 'IH', 'SPINNING_TOP': 'ST',
    'MARUBOZU_UP': 'MU', 'MARUBOZU_DOWN': 'MD',
    'BIG_UP': 'BU', 'BIG_DOWN': 'BD',
    'MED_UP': 'U', 'MED_DOWN': 'DN'
}

def to_short_code(pattern_name: str) -> str:
    """Convert full pattern name to short code."""
    parts = pattern_name.split('-')
    return '-'.join(TYPE_MAP.get(p, p) for p in parts)


def load_discovery_results() -> pd.DataFrame:
    """Load and prepare pattern discovery results."""
    results_path = Path(__file__).parent.parent.parent / "results" / "unified_pattern_discovery.json"
    with open(results_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data['per_pattern_optimal'])
    df['short_code'] = df['pattern'].apply(to_short_code)
    df['pnl'] = df['total_pnl']
    return df


def load_market_data() -> pd.DataFrame:
    """Load reclassified dataset."""
    data_path = Path(__file__).parent.parent.parent / "data" / "btc_5m_270days_reclassified.csv"
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def select_patterns(discovery_df: pd.DataFrame, mc_th: float, wf_th: int,
                   wr_l_th: float, wr_s_th: float, trades_th: int,
                   n_long: int, n_short: int) -> Tuple[List[Dict], List[Dict]]:
    """Select top N patterns based on criteria and score."""

    # Filter LONG
    long_df = discovery_df[
        (discovery_df['direction'] == 'LONG') &
        (discovery_df['mc'] < mc_th) &
        (discovery_df['wf'] >= wf_th) &
        (discovery_df['wr'] >= wr_l_th) &
        (discovery_df['trades'] >= trades_th)
    ].copy()
    long_df['score'] = long_df['wr'] * np.sqrt(long_df['trades']) * (1 - long_df['mc'])

    # Filter SHORT
    short_df = discovery_df[
        (discovery_df['direction'] == 'SHORT') &
        (discovery_df['mc'] < mc_th) &
        (discovery_df['wf'] >= wf_th) &
        (discovery_df['wr'] >= wr_s_th) &
        (discovery_df['trades'] >= trades_th)
    ].copy()
    short_df['score'] = short_df['wr'] * np.sqrt(short_df['trades']) * (1 - short_df['mc'])

    # Select top N
    long_selected = long_df.nlargest(n_long, 'score').to_dict('records')
    short_selected = short_df.nlargest(n_short, 'score').to_dict('records')

    return long_selected, short_selected


def backtest_portfolio(market_df: pd.DataFrame,
                       long_patterns: List[Dict],
                       short_patterns: List[Dict],
                       commission: float = 0.001) -> Dict:
    """Backtest a portfolio of patterns."""

    results = {
        'total_trades': 0,
        'wins': 0,
        'total_pnl': 0,
        'long_trades': 0,
        'short_trades': 0,
        'long_pnl': 0,
        'short_pnl': 0,
        'pattern_results': []
    }

    # Create pattern -> config mapping
    pattern_config = {}
    for p in long_patterns:
        pattern_config[p['short_code']] = {
            'direction': 'LONG',
            'tp': p['tp'],
            'sl': p['sl']
        }
    for p in short_patterns:
        pattern_config[p['short_code']] = {
            'direction': 'SHORT',
            'tp': p['tp'],
            'sl': p['sl']
        }

    # Backtest each pattern
    for pattern_code, config in pattern_config.items():
        signals = market_df[market_df['pattern_3'] == pattern_code].copy()

        if len(signals) == 0:
            continue

        direction = config['direction']
        tp_pct = config['tp']
        sl_pct = config['sl']

        trades = []
        for idx, row in signals.iterrows():
            entry_idx = market_df.index.get_loc(idx)
            if entry_idx + 1 >= len(market_df):
                continue

            entry_price = market_df.iloc[entry_idx + 1]['open']

            if direction == 'LONG':
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)
            else:
                tp_price = entry_price * (1 - tp_pct / 100)
                sl_price = entry_price * (1 + sl_pct / 100)

            # Simulate trade
            result = None
            for i in range(entry_idx + 1, min(entry_idx + 100, len(market_df))):
                bar = market_df.iloc[i]
                high, low = bar['high'], bar['low']

                if direction == 'LONG':
                    if low <= sl_price:
                        result = {'exit_pct': -sl_pct, 'win': False}
                        break
                    elif high >= tp_price:
                        result = {'exit_pct': tp_pct, 'win': True}
                        break
                else:
                    if high >= sl_price:
                        result = {'exit_pct': -sl_pct, 'win': False}
                        break
                    elif low <= tp_price:
                        result = {'exit_pct': tp_pct, 'win': True}
                        break

            if result:
                pnl = result['exit_pct'] - commission * 100
                trades.append({'pnl': pnl, 'win': result['win']})

        if trades:
            wins = sum(1 for t in trades if t['win'])
            total_pnl = sum(t['pnl'] for t in trades)

            results['total_trades'] += len(trades)
            results['wins'] += wins
            results['total_pnl'] += total_pnl

            if direction == 'LONG':
                results['long_trades'] += len(trades)
                results['long_pnl'] += total_pnl
            else:
                results['short_trades'] += len(trades)
                results['short_pnl'] += total_pnl

            results['pattern_results'].append({
                'pattern': pattern_code,
                'direction': direction,
                'trades': len(trades),
                'wins': wins,
                'wr': wins / len(trades) * 100,
                'pnl': total_pnl
            })

    # Calculate portfolio metrics
    if results['total_trades'] > 0:
        results['portfolio_wr'] = results['wins'] / results['total_trades'] * 100
    else:
        results['portfolio_wr'] = 0

    return results


def walk_forward_validation(market_df: pd.DataFrame,
                           long_patterns: List[Dict],
                           short_patterns: List[Dict],
                           n_folds: int = 5) -> Dict:
    """Perform walk-forward validation."""

    n_bars = len(market_df)
    fold_size = n_bars // n_folds

    fold_results = []
    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else n_bars
        fold_df = market_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

        result = backtest_portfolio(fold_df, long_patterns, short_patterns)
        fold_results.append({
            'fold': fold + 1,
            'pnl': result['total_pnl'],
            'trades': result['total_trades'],
            'wr': result['portfolio_wr']
        })

    profitable_folds = sum(1 for f in fold_results if f['pnl'] > 0)

    return {
        'folds': fold_results,
        'profitable_folds': profitable_folds,
        'wf_score': f"{profitable_folds}/{n_folds}"
    }


def main():
    print("Loading data...")
    discovery_df = load_discovery_results()
    market_df = load_market_data()
    print(f"Loaded {len(discovery_df)} patterns, {len(market_df)} market bars")
    print()

    # Test different portfolio configurations
    configs = [
        # (mc, wf, wr_l, wr_s, trades, n_long, n_short, name)
        (0.01, 5, 60, 55, 50, 5, 5, "Ultra Strict 5+5"),
        (0.01, 4, 60, 55, 50, 5, 5, "Strict 5+5"),
        (0.01, 4, 60, 55, 50, 7, 7, "Strict 7+7"),
        (0.01, 4, 60, 55, 50, 10, 10, "Strict 10+10"),
        (0.02, 4, 55, 50, 40, 7, 7, "Moderate 7+7"),
        (0.02, 4, 55, 50, 40, 10, 10, "Moderate 10+10"),
        (0.02, 4, 55, 50, 40, 12, 12, "Moderate 12+12"),
        (0.02, 4, 55, 50, 30, 15, 15, "Balanced 15+15"),
        (0.03, 4, 55, 50, 30, 10, 10, "Relaxed 10+10"),
        (0.03, 4, 55, 50, 30, 15, 15, "Relaxed 15+15"),
    ]

    print("=" * 100)
    print("PORTFOLIO SIZE OPTIMIZATION - BACKTEST RESULTS")
    print("=" * 100)
    print()
    print(f"{'Config':<20} | {'Pat':>4} | {'Trades':>6} | {'WR':>6} | {'PnL':>8} | {'L PnL':>8} | {'S PnL':>8} | {'WF':>5} | {'AvgMC':>7}")
    print("-" * 100)

    all_results = []

    for mc, wf, wr_l, wr_s, trades, n_l, n_s, name in configs:
        long_sel, short_sel = select_patterns(
            discovery_df, mc, wf, wr_l, wr_s, trades, n_l, n_s
        )

        if len(long_sel) < n_l or len(short_sel) < n_s:
            print(f"{name:<20} | Insufficient patterns (L:{len(long_sel)}, S:{len(short_sel)})")
            continue

        # Backtest
        bt_result = backtest_portfolio(market_df, long_sel, short_sel)

        # Walk-forward
        wf_result = walk_forward_validation(market_df, long_sel, short_sel)

        # Calculate avg MC
        all_patterns = long_sel + short_sel
        avg_mc = np.mean([p['mc'] for p in all_patterns])

        total_patterns = len(long_sel) + len(short_sel)

        print(f"{name:<20} | {total_patterns:>4} | {bt_result['total_trades']:>6} | "
              f"{bt_result['portfolio_wr']:>5.1f}% | {bt_result['total_pnl']:>7.1f}% | "
              f"{bt_result['long_pnl']:>7.1f}% | {bt_result['short_pnl']:>7.1f}% | "
              f"{wf_result['wf_score']:>5} | {avg_mc:>7.4f}")

        all_results.append({
            'name': name,
            'config': (mc, wf, wr_l, wr_s, trades, n_l, n_s),
            'patterns': total_patterns,
            'backtest': bt_result,
            'walk_forward': wf_result,
            'avg_mc': avg_mc,
            'long_patterns': long_sel,
            'short_patterns': short_sel
        })

    print()
    print("=" * 100)
    print("BEST CONFIGURATIONS ANALYSIS")
    print("=" * 100)
    print()

    # Find best by different metrics
    if all_results:
        best_pnl = max(all_results, key=lambda x: x['backtest']['total_pnl'])
        best_wr = max(all_results, key=lambda x: x['backtest']['portfolio_wr'])
        best_wf = max(all_results, key=lambda x: x['walk_forward']['profitable_folds'])

        # Composite score: PnL * WR * WF_ratio
        for r in all_results:
            wf_ratio = r['walk_forward']['profitable_folds'] / 5
            r['composite'] = r['backtest']['total_pnl'] * (r['backtest']['portfolio_wr'] / 100) * wf_ratio

        best_composite = max(all_results, key=lambda x: x['composite'])

        print(f"Best by PnL:       {best_pnl['name']} - PnL {best_pnl['backtest']['total_pnl']:.1f}%")
        print(f"Best by WR:        {best_wr['name']} - WR {best_wr['backtest']['portfolio_wr']:.1f}%")
        print(f"Best by WF:        {best_wf['name']} - WF {best_wf['walk_forward']['wf_score']}")
        print(f"Best Composite:    {best_composite['name']} - Score {best_composite['composite']:.1f}")
        print()

        # Detailed breakdown of best composite
        print("=" * 100)
        print(f"RECOMMENDED PORTFOLIO: {best_composite['name']}")
        print("=" * 100)
        print()

        print("LONG Patterns:")
        print(f"{'Pattern':<15} {'WR':>6} {'MC':>8} {'WF':>3} {'Trades':>6} {'TP':>4} {'SL':>4}")
        print("-" * 55)
        for p in best_composite['long_patterns']:
            print(f"{p['short_code']:<15} {p['wr']:>5.1f}% {p['mc']:>8.4f} {p['wf']:>3} "
                  f"{p['trades']:>6} {p['tp']:>4.1f} {p['sl']:>4.1f}")

        print()
        print("SHORT Patterns:")
        print(f"{'Pattern':<15} {'WR':>6} {'MC':>8} {'WF':>3} {'Trades':>6} {'TP':>4} {'SL':>4}")
        print("-" * 55)
        for p in best_composite['short_patterns']:
            print(f"{p['short_code']:<15} {p['wr']:>5.1f}% {p['mc']:>8.4f} {p['wf']:>3} "
                  f"{p['trades']:>6} {p['tp']:>4.1f} {p['sl']:>4.1f}")

        # Walk-forward detail
        print()
        print("Walk-Forward Detail:")
        for f in best_composite['walk_forward']['folds']:
            status = "✅" if f['pnl'] > 0 else "❌"
            print(f"  Fold {f['fold']}: PnL {f['pnl']:>7.1f}%, Trades {f['trades']:>4}, WR {f['wr']:>5.1f}% {status}")

        # Save recommended portfolio
        output = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'recommendation': best_composite['name'],
            'config': {
                'mc_threshold': best_composite['config'][0],
                'wf_threshold': best_composite['config'][1],
                'wr_long_threshold': best_composite['config'][2],
                'wr_short_threshold': best_composite['config'][3],
                'min_trades': best_composite['config'][4],
            },
            'results': {
                'total_patterns': best_composite['patterns'],
                'total_trades': best_composite['backtest']['total_trades'],
                'portfolio_wr': best_composite['backtest']['portfolio_wr'],
                'total_pnl': best_composite['backtest']['total_pnl'],
                'long_pnl': best_composite['backtest']['long_pnl'],
                'short_pnl': best_composite['backtest']['short_pnl'],
                'walk_forward': best_composite['walk_forward']['wf_score'],
            },
            'long_patterns': [
                {
                    'pattern': p['short_code'],
                    'full_name': p['pattern'],
                    'tp': p['tp'],
                    'sl': p['sl'],
                    'wr': p['wr'],
                    'mc': p['mc'],
                    'wf': p['wf'],
                    'trades': p['trades']
                }
                for p in best_composite['long_patterns']
            ],
            'short_patterns': [
                {
                    'pattern': p['short_code'],
                    'full_name': p['pattern'],
                    'tp': p['tp'],
                    'sl': p['sl'],
                    'wr': p['wr'],
                    'mc': p['mc'],
                    'wf': p['wf'],
                    'trades': p['trades']
                }
                for p in best_composite['short_patterns']
            ],
            'all_configs_tested': [
                {
                    'name': r['name'],
                    'patterns': r['patterns'],
                    'trades': r['backtest']['total_trades'],
                    'wr': r['backtest']['portfolio_wr'],
                    'pnl': r['backtest']['total_pnl'],
                    'wf': r['walk_forward']['wf_score'],
                    'composite_score': r['composite']
                }
                for r in all_results
            ]
        }

        output_path = Path(__file__).parent.parent.parent / "results" / "portfolio_optimization_v124.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print()
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
