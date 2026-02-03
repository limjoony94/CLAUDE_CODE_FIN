"""
Deep Portfolio Analysis - Rigorous Statistical Examination
Multi-dimensional analysis for optimal portfolio selection

Created: 2026-02-04
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from scipy import stats
from collections import defaultdict

# Type mapping
TYPE_MAP = {
    'DOJI': 'D', 'DRAGONFLY': 'DF', 'GRAVESTONE': 'GS',
    'HAMMER': 'H', 'INV_HAMMER': 'IH', 'SPINNING_TOP': 'ST',
    'MARUBOZU_UP': 'MU', 'MARUBOZU_DOWN': 'MD',
    'BIG_UP': 'BU', 'BIG_DOWN': 'BD',
    'MED_UP': 'U', 'MED_DOWN': 'DN'
}

def to_short_code(pattern_name: str) -> str:
    parts = pattern_name.split('-')
    return '-'.join(TYPE_MAP.get(p, p) for p in parts)


def load_data():
    """Load all required data."""
    # Discovery results
    results_path = Path(__file__).parent.parent.parent / "results" / "unified_pattern_discovery.json"
    with open(results_path, 'r') as f:
        discovery = json.load(f)

    patterns_df = pd.DataFrame(discovery['per_pattern_optimal'])
    patterns_df['short_code'] = patterns_df['pattern'].apply(to_short_code)
    patterns_df['pnl'] = patterns_df['total_pnl']

    # Market data
    data_path = Path(__file__).parent.parent.parent / "data" / "btc_5m_270days_reclassified.csv"
    market_df = pd.read_csv(data_path)
    market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])

    return patterns_df, market_df


def select_patterns(df: pd.DataFrame, mc_th: float, wf_th: int,
                   wr_l_th: float, wr_s_th: float, trades_th: int,
                   n_long: int, n_short: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select top patterns based on criteria."""

    long_df = df[
        (df['direction'] == 'LONG') &
        (df['mc'] < mc_th) &
        (df['wf'] >= wf_th) &
        (df['wr'] >= wr_l_th) &
        (df['trades'] >= trades_th)
    ].copy()
    long_df['score'] = long_df['wr'] * np.sqrt(long_df['trades']) * (1 - long_df['mc'])

    short_df = df[
        (df['direction'] == 'SHORT') &
        (df['mc'] < mc_th) &
        (df['wf'] >= wf_th) &
        (df['wr'] >= wr_s_th) &
        (df['trades'] >= trades_th)
    ].copy()
    short_df['score'] = short_df['wr'] * np.sqrt(short_df['trades']) * (1 - short_df['mc'])

    return long_df.nlargest(n_long, 'score'), short_df.nlargest(n_short, 'score')


def simulate_trades(market_df: pd.DataFrame, pattern_code: str,
                   direction: str, tp: float, sl: float) -> List[Dict]:
    """Simulate all trades for a single pattern with detailed info."""
    signals = market_df[market_df['pattern_3'] == pattern_code]
    trades = []

    for idx, row in signals.iterrows():
        entry_idx = market_df.index.get_loc(idx)
        if entry_idx + 1 >= len(market_df):
            continue

        entry_bar = market_df.iloc[entry_idx + 1]
        entry_price = entry_bar['open']
        entry_time = entry_bar['timestamp']

        if direction == 'LONG':
            tp_price = entry_price * (1 + tp / 100)
            sl_price = entry_price * (1 - sl / 100)
        else:
            tp_price = entry_price * (1 - tp / 100)
            sl_price = entry_price * (1 + sl / 100)

        # Simulate
        result = None
        exit_time = None
        bars_held = 0

        for i in range(entry_idx + 1, min(entry_idx + 100, len(market_df))):
            bar = market_df.iloc[i]
            bars_held += 1

            if direction == 'LONG':
                if bar['low'] <= sl_price:
                    result = {'pnl': -sl - 0.1, 'win': False}
                    exit_time = bar['timestamp']
                    break
                elif bar['high'] >= tp_price:
                    result = {'pnl': tp - 0.1, 'win': True}
                    exit_time = bar['timestamp']
                    break
            else:
                if bar['high'] >= sl_price:
                    result = {'pnl': -sl - 0.1, 'win': False}
                    exit_time = bar['timestamp']
                    break
                elif bar['low'] <= tp_price:
                    result = {'pnl': tp - 0.1, 'win': True}
                    exit_time = bar['timestamp']
                    break

        if result:
            trades.append({
                'pattern': pattern_code,
                'direction': direction,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'pnl': result['pnl'],
                'win': result['win'],
                'bars_held': bars_held
            })

    return trades


def analyze_portfolio_deep(market_df: pd.DataFrame,
                          long_patterns: pd.DataFrame,
                          short_patterns: pd.DataFrame,
                          config_name: str) -> Dict:
    """Comprehensive portfolio analysis."""

    all_trades = []
    pattern_trades = {}

    # Collect all trades
    for _, p in long_patterns.iterrows():
        trades = simulate_trades(market_df, p['short_code'], 'LONG', p['tp'], p['sl'])
        pattern_trades[p['short_code']] = trades
        all_trades.extend(trades)

    for _, p in short_patterns.iterrows():
        trades = simulate_trades(market_df, p['short_code'], 'SHORT', p['tp'], p['sl'])
        pattern_trades[p['short_code']] = trades
        all_trades.extend(trades)

    if not all_trades:
        return None

    # Sort by time
    trades_df = pd.DataFrame(all_trades)
    trades_df = trades_df.sort_values('entry_time').reset_index(drop=True)

    # === BASIC METRICS ===
    total_trades = len(trades_df)
    wins = trades_df['win'].sum()
    total_pnl = trades_df['pnl'].sum()
    wr = wins / total_trades * 100

    # === RISK METRICS ===
    # Equity curve
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()

    # Maximum Drawdown
    running_max = trades_df['cumulative_pnl'].cummax()
    drawdown = running_max - trades_df['cumulative_pnl']
    max_drawdown = drawdown.max()

    # Consecutive losses
    trades_df['is_loss'] = ~trades_df['win']
    max_consecutive_losses = 0
    current_streak = 0
    for is_loss in trades_df['is_loss']:
        if is_loss:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0

    # === STATISTICAL ROBUSTNESS ===
    # Profit Factor
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe-like ratio (PnL / StdDev)
    pnl_std = trades_df['pnl'].std()
    avg_pnl = trades_df['pnl'].mean()
    sharpe_like = avg_pnl / pnl_std if pnl_std > 0 else 0

    # === TEMPORAL STABILITY ===
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    monthly_pnl = trades_df.groupby('month')['pnl'].sum()
    monthly_trades = trades_df.groupby('month').size()

    profitable_months = (monthly_pnl > 0).sum()
    total_months = len(monthly_pnl)
    monthly_consistency = profitable_months / total_months * 100

    # Monthly PnL variance
    monthly_pnl_std = monthly_pnl.std()
    monthly_pnl_mean = monthly_pnl.mean()
    monthly_cv = monthly_pnl_std / abs(monthly_pnl_mean) if monthly_pnl_mean != 0 else float('inf')

    # === EDGE DECAY ANALYSIS ===
    # Split into early vs late periods
    mid_point = len(trades_df) // 2
    early_trades = trades_df.iloc[:mid_point]
    late_trades = trades_df.iloc[mid_point:]

    early_wr = early_trades['win'].mean() * 100
    late_wr = late_trades['win'].mean() * 100
    early_pnl = early_trades['pnl'].sum()
    late_pnl = late_trades['pnl'].sum()

    edge_decay_wr = late_wr - early_wr  # Negative = decay
    edge_decay_pnl = late_pnl - early_pnl

    # === PATTERN QUALITY DISTRIBUTION ===
    pattern_metrics = []
    for code, p_trades in pattern_trades.items():
        if p_trades:
            p_df = pd.DataFrame(p_trades)
            p_wr = p_df['win'].mean() * 100
            p_pnl = p_df['pnl'].sum()
            pattern_metrics.append({
                'pattern': code,
                'trades': len(p_trades),
                'wr': p_wr,
                'pnl': p_pnl
            })

    # Patterns with WR > 70%
    high_quality_patterns = sum(1 for p in pattern_metrics if p['wr'] >= 70)

    # === WALK-FORWARD (5-fold) ===
    n_bars = len(market_df)
    fold_size = n_bars // 5
    wf_results = []

    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size if fold < 4 else n_bars
        fold_trades = trades_df[
            (trades_df['entry_time'] >= market_df.iloc[start]['timestamp']) &
            (trades_df['entry_time'] < market_df.iloc[min(end, n_bars-1)]['timestamp'])
        ]

        if len(fold_trades) > 0:
            fold_pnl = fold_trades['pnl'].sum()
            fold_wr = fold_trades['win'].mean() * 100
            wf_results.append({
                'fold': fold + 1,
                'pnl': fold_pnl,
                'trades': len(fold_trades),
                'wr': fold_wr,
                'profitable': fold_pnl > 0
            })

    profitable_folds = sum(1 for f in wf_results if f['profitable'])

    # === COMPOSITE SCORES ===
    # Quality Score: WR × (1 - MC_avg) × WF_ratio
    mc_values = list(long_patterns['mc']) + list(short_patterns['mc'])
    avg_mc = np.mean(mc_values)
    quality_score = wr * (1 - avg_mc) * (profitable_folds / 5)

    # Risk-Adjusted Score: PnL / (1 + MDD)
    risk_adj_score = total_pnl / (1 + max_drawdown) if max_drawdown > 0 else total_pnl

    # Stability Score: Monthly consistency × (1 - CV)
    stability_score = monthly_consistency * max(0, 1 - monthly_cv)

    # Overall Score: Geometric mean of normalized scores
    overall_score = (quality_score * risk_adj_score * stability_score) ** (1/3)

    return {
        'config': config_name,
        'patterns': len(long_patterns) + len(short_patterns),
        'long_count': len(long_patterns),
        'short_count': len(short_patterns),

        # Basic
        'total_trades': total_trades,
        'wins': wins,
        'wr': wr,
        'total_pnl': total_pnl,
        'profit_factor': profit_factor,

        # Risk
        'max_drawdown': max_drawdown,
        'max_consecutive_losses': max_consecutive_losses,
        'sharpe_like': sharpe_like,

        # Temporal
        'profitable_months': profitable_months,
        'total_months': total_months,
        'monthly_consistency': monthly_consistency,
        'monthly_cv': monthly_cv,

        # Edge decay
        'early_wr': early_wr,
        'late_wr': late_wr,
        'edge_decay_wr': edge_decay_wr,
        'early_pnl': early_pnl,
        'late_pnl': late_pnl,

        # Quality
        'avg_mc': avg_mc,
        'high_quality_patterns': high_quality_patterns,

        # Walk-forward
        'wf_results': wf_results,
        'profitable_folds': profitable_folds,

        # Scores
        'quality_score': quality_score,
        'risk_adj_score': risk_adj_score,
        'stability_score': stability_score,
        'overall_score': overall_score,

        # Pattern details
        'pattern_metrics': pattern_metrics,
        'monthly_pnl': monthly_pnl.to_dict(),
    }


def main():
    print("Loading data...")
    patterns_df, market_df = load_data()
    print(f"Patterns: {len(patterns_df)}, Market bars: {len(market_df)}")
    print()

    # Configurations to test
    configs = [
        # (mc, wf, wr_l, wr_s, trades, n_l, n_s, name)
        (0.01, 4, 60, 55, 50, 5, 5, "Strict-10"),
        (0.01, 4, 60, 55, 50, 7, 7, "Strict-14"),
        (0.01, 4, 60, 55, 50, 10, 10, "Strict-20"),
        (0.015, 4, 58, 53, 45, 10, 10, "Moderate-A-20"),
        (0.015, 4, 58, 53, 45, 12, 12, "Moderate-A-24"),
        (0.02, 4, 55, 50, 40, 10, 10, "Moderate-B-20"),
        (0.02, 4, 55, 50, 40, 12, 12, "Moderate-B-24"),
        (0.02, 4, 55, 50, 35, 15, 15, "Balanced-30"),
    ]

    results = []

    for mc, wf, wr_l, wr_s, tr, n_l, n_s, name in configs:
        print(f"Analyzing {name}...")
        long_sel, short_sel = select_patterns(patterns_df, mc, wf, wr_l, wr_s, tr, n_l, n_s)

        if len(long_sel) < n_l or len(short_sel) < n_s:
            print(f"  Skip: Insufficient patterns (L:{len(long_sel)}, S:{len(short_sel)})")
            continue

        analysis = analyze_portfolio_deep(market_df, long_sel, short_sel, name)
        if analysis:
            analysis['long_patterns'] = long_sel.to_dict('records')
            analysis['short_patterns'] = short_sel.to_dict('records')
            results.append(analysis)

    print()
    print("=" * 120)
    print("DEEP ANALYSIS RESULTS - COMPREHENSIVE COMPARISON")
    print("=" * 120)
    print()

    # Summary table
    print(f"{'Config':<15} | {'Pat':>3} | {'Tr':>5} | {'WR':>5} | {'PnL':>7} | {'PF':>5} | {'MDD':>6} | {'MCL':>3} | {'MC':>5} | {'WF':>4} | {'Shrp':>5} | {'Stab':>5} | {'Overall':>7}")
    print("-" * 120)

    for r in results:
        print(f"{r['config']:<15} | {r['patterns']:>3} | {r['total_trades']:>5} | "
              f"{r['wr']:>4.1f}% | {r['total_pnl']:>6.1f}% | {r['profit_factor']:>5.2f} | "
              f"{r['max_drawdown']:>5.1f}% | {r['max_consecutive_losses']:>3} | "
              f"{r['avg_mc']:>5.3f} | {r['profitable_folds']:>1}/5 | "
              f"{r['sharpe_like']:>5.3f} | {r['stability_score']:>5.1f} | {r['overall_score']:>7.1f}")

    print()
    print("Legend: Pat=Patterns, Tr=Trades, PF=Profit Factor, MDD=Max Drawdown, MCL=Max Consecutive Losses")
    print("        MC=Avg Monte Carlo, Shrp=Sharpe-like, Stab=Stability Score")
    print()

    # === DETAILED ANALYSIS ===
    print("=" * 120)
    print("EDGE DECAY ANALYSIS (Early vs Late Period)")
    print("=" * 120)
    print()
    print(f"{'Config':<15} | {'Early WR':>8} | {'Late WR':>8} | {'Decay':>7} | {'Early PnL':>10} | {'Late PnL':>10} | {'Status':>10}")
    print("-" * 90)

    for r in results:
        decay_status = "✅ Stable" if r['edge_decay_wr'] >= -3 else ("⚠️ Minor" if r['edge_decay_wr'] >= -7 else "❌ Decay")
        print(f"{r['config']:<15} | {r['early_wr']:>7.1f}% | {r['late_wr']:>7.1f}% | {r['edge_decay_wr']:>6.1f}% | "
              f"{r['early_pnl']:>9.1f}% | {r['late_pnl']:>9.1f}% | {decay_status:>10}")

    print()
    print("=" * 120)
    print("MONTHLY CONSISTENCY ANALYSIS")
    print("=" * 120)
    print()
    print(f"{'Config':<15} | {'Prof Months':>11} | {'Consistency':>11} | {'Monthly CV':>10} | {'Stability':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r['config']:<15} | {r['profitable_months']:>3}/{r['total_months']:<3} ({r['monthly_consistency']:>4.0f}%) | "
              f"{r['monthly_consistency']:>10.1f}% | {r['monthly_cv']:>10.2f} | {r['stability_score']:>10.1f}")

    print()
    print("=" * 120)
    print("PATTERN QUALITY DISTRIBUTION")
    print("=" * 120)
    print()

    for r in results:
        high_q = r['high_quality_patterns']
        total = r['patterns']
        print(f"{r['config']}: {high_q}/{total} patterns with WR>=70% ({high_q/total*100:.0f}%)")

    print()

    # === FINAL RECOMMENDATION ===
    print("=" * 120)
    print("FINAL RECOMMENDATION")
    print("=" * 120)
    print()

    # Sort by overall score
    sorted_results = sorted(results, key=lambda x: x['overall_score'], reverse=True)

    print("Ranking by Overall Score (Quality × Risk-Adjusted × Stability):")
    print()
    for i, r in enumerate(sorted_results, 1):
        flag = "⭐" if i == 1 else "  "
        print(f"{flag} {i}. {r['config']:<15} - Score: {r['overall_score']:.1f} "
              f"(WR:{r['wr']:.1f}%, PnL:{r['total_pnl']:.1f}%, WF:{r['profitable_folds']}/5, MDD:{r['max_drawdown']:.1f}%)")

    best = sorted_results[0]

    print()
    print("-" * 60)
    print(f"RECOMMENDED: {best['config']}")
    print("-" * 60)
    print()
    print(f"  Patterns: {best['patterns']} ({best['long_count']}L + {best['short_count']}S)")
    print(f"  Total Trades: {best['total_trades']}")
    print(f"  Win Rate: {best['wr']:.1f}%")
    print(f"  Total PnL: {best['total_pnl']:.1f}%")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Max Drawdown: {best['max_drawdown']:.1f}%")
    print(f"  Walk-Forward: {best['profitable_folds']}/5")
    print(f"  Edge Decay: {best['edge_decay_wr']:.1f}% (WR change)")
    print(f"  Monthly Consistency: {best['monthly_consistency']:.0f}%")
    print()

    # Pattern list for best config
    print("LONG Patterns:")
    print(f"{'Pattern':<12} {'WR':>6} {'MC':>7} {'WF':>3} {'Trades':>6} {'TP':>4} {'SL':>4} {'PnL':>7}")
    print("-" * 60)
    for p in sorted(best['pattern_metrics'], key=lambda x: x['pnl'], reverse=True):
        if any(lp['short_code'] == p['pattern'] for lp in best['long_patterns']):
            lp = next(lp for lp in best['long_patterns'] if lp['short_code'] == p['pattern'])
            print(f"{p['pattern']:<12} {p['wr']:>5.1f}% {lp['mc']:>7.4f} {lp['wf']:>3} {p['trades']:>6} {lp['tp']:>4.1f} {lp['sl']:>4.1f} {p['pnl']:>6.1f}%")

    print()
    print("SHORT Patterns:")
    print(f"{'Pattern':<12} {'WR':>6} {'MC':>7} {'WF':>3} {'Trades':>6} {'TP':>4} {'SL':>4} {'PnL':>7}")
    print("-" * 60)
    for p in sorted(best['pattern_metrics'], key=lambda x: x['pnl'], reverse=True):
        if any(sp['short_code'] == p['pattern'] for sp in best['short_patterns']):
            sp = next(sp for sp in best['short_patterns'] if sp['short_code'] == p['pattern'])
            print(f"{p['pattern']:<12} {p['wr']:>5.1f}% {sp['mc']:>7.4f} {sp['wf']:>3} {p['trades']:>6} {sp['tp']:>4.1f} {sp['sl']:>4.1f} {p['pnl']:>6.1f}%")

    # Save results
    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'analysis_type': 'deep_portfolio_analysis',
        'recommendation': best['config'],
        'all_results': [
            {
                'config': r['config'],
                'patterns': r['patterns'],
                'trades': r['total_trades'],
                'wr': r['wr'],
                'pnl': r['total_pnl'],
                'profit_factor': r['profit_factor'],
                'max_drawdown': r['max_drawdown'],
                'sharpe_like': r['sharpe_like'],
                'edge_decay_wr': r['edge_decay_wr'],
                'monthly_consistency': r['monthly_consistency'],
                'wf_score': f"{r['profitable_folds']}/5",
                'overall_score': r['overall_score'],
            }
            for r in results
        ],
        'recommended_portfolio': {
            'long_patterns': [
                {'pattern': p['short_code'], 'tp': p['tp'], 'sl': p['sl'],
                 'wr': p['wr'], 'mc': p['mc'], 'wf': p['wf'], 'trades': p['trades']}
                for p in best['long_patterns']
            ],
            'short_patterns': [
                {'pattern': p['short_code'], 'tp': p['tp'], 'sl': p['sl'],
                 'wr': p['wr'], 'mc': p['mc'], 'wf': p['wf'], 'trades': p['trades']}
                for p in best['short_patterns']
            ]
        }
    }

    output_path = Path(__file__).parent.parent.parent / "results" / "deep_portfolio_analysis_v124.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
