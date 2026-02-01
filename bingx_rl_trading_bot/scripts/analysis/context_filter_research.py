"""
Context Filter Research - Analyze Market Context at Pattern Entry Points
===========================================================================

Goal: Identify profitable context filters (RSI, ATR, trend) that could improve
pattern trading performance by filtering out low-probability setups.

Usage:
    python scripts/analysis/context_filter_research.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys

# Hardcoded constants from production (v1.22.0 - DN-D-BD removed)
PATTERN_CONFIGS_LONG = {
    "U-MU-H": (1.5, 1.5),
    "MD-ST-MD": (2.0, 2.0),
    "GS-U-BD": (1.0, 1.0),
    "MD-MD-ST": (1.5, 2.0),
    "BU-IH-DN": (1.5, 2.0),
    "MD-H-MD": (1.0, 1.0),
    "IH-MD-MD": (1.5, 2.0),
}

PATTERN_CONFIGS_SHORT = {
    "BD-U-GS": (1.5, 2.0),
    "DN-GS-H": (1.0, 1.0),
    "U-DF-BU": (1.0, 1.5),
    "BD-GS-BD": (1.0, 1.0),
    "DN-IH-IH": (1.0, 1.5),
}

LEVERAGE = 3  # Effective leverage for PnL calculations

# ============================================================
# Configuration
# ============================================================

DATA_FILE = Path(__file__).parent / "../../data/btc_5m_270days.csv"
OUTPUT_FILE = Path(__file__).parent / "../../claudedocs/CONTEXT_FILTER_RESEARCH.md"

# Backtest parameters
FEE_PCT = 0.10
MAX_BARS = 500

# Context filter thresholds to test
RSI_BINS = [(0, 30), (30, 40), (40, 60), (60, 70), (70, 100)]
ATR_PERCENTILE_BINS = [(0, 25), (25, 50), (50, 75), (75, 100)]
TREND_CONDITIONS = ['STRONG_UP', 'UP', 'SIDEWAYS', 'DOWN', 'STRONG_DOWN']


# ============================================================
# Indicator Calculation
# ============================================================

def calculate_context_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, ATR, and trend indicators for context analysis."""
    df = df.copy()

    # RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR (14-period)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    df['atr_percentile'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    )

    # EMA-based trend
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()

    # Trend classification
    df['price_vs_ema20'] = (df['close'] / df['ema20'] - 1) * 100
    df['ema20_vs_ema50'] = (df['ema20'] / df['ema50'] - 1) * 100

    conditions = [
        (df['price_vs_ema20'] > 1.0) & (df['ema20_vs_ema50'] > 0.5),  # STRONG_UP
        (df['price_vs_ema20'] > 0) & (df['ema20_vs_ema50'] > 0),      # UP
        (df['price_vs_ema20'].abs() <= 0.5) & (df['ema20_vs_ema50'].abs() <= 0.5),  # SIDEWAYS
        (df['price_vs_ema20'] < 0) & (df['ema20_vs_ema50'] < 0),      # DOWN
        (df['price_vs_ema20'] < -1.0) & (df['ema20_vs_ema50'] < -0.5), # STRONG_DOWN
    ]
    df['trend'] = np.select(conditions, TREND_CONDITIONS, default='SIDEWAYS')

    return df


# ============================================================
# Backtest Functions
# ============================================================

def backtest_pattern(
    df: pd.DataFrame,
    pattern: str,
    direction: str,
    tp_pct: float,
    sl_pct: float,
) -> List[Dict]:
    """
    Backtest a pattern and return trade results with context data.

    Returns list of trade dicts with: outcome, pnl, rsi, atr_pct, trend
    """
    signals = df[df['pattern_3'] == pattern].copy()

    if len(signals) < 5:
        return []

    trades = []
    n_bars = len(df)

    for idx in signals.index:
        pos = df.index.get_loc(idx)
        if pos + 1 >= n_bars:
            continue

        entry_price = df.iloc[pos + 1]['open']
        if entry_price <= 0:
            continue

        # Capture context at entry
        entry_bar = df.iloc[pos + 1]
        context = {
            'rsi': entry_bar.get('rsi', np.nan),
            'atr_pct': entry_bar.get('atr_percentile', np.nan),
            'trend': entry_bar.get('trend', 'SIDEWAYS'),
        }

        # Simulate trade
        for i in range(pos + 2, min(pos + 2 + MAX_BARS, n_bars)):
            bar = df.iloc[i]

            if direction == 'LONG':
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)

                if bar['high'] >= tp_price:
                    pnl = (tp_pct * LEVERAGE) - FEE_PCT
                    trades.append({'outcome': 'WIN', 'pnl': pnl, **context})
                    break
                elif bar['low'] <= sl_price:
                    pnl = -(sl_pct * LEVERAGE) - FEE_PCT
                    trades.append({'outcome': 'LOSS', 'pnl': pnl, **context})
                    break
            else:  # SHORT
                tp_price = entry_price * (1 - tp_pct / 100)
                sl_price = entry_price * (1 + sl_pct / 100)

                if bar['low'] <= tp_price:
                    pnl = (tp_pct * LEVERAGE) - FEE_PCT
                    trades.append({'outcome': 'WIN', 'pnl': pnl, **context})
                    break
                elif bar['high'] >= sl_price:
                    pnl = -(sl_pct * LEVERAGE) - FEE_PCT
                    trades.append({'outcome': 'LOSS', 'pnl': pnl, **context})
                    break

    return trades


# ============================================================
# Analysis Functions
# ============================================================

def analyze_rsi_filter(trades: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Analyze win rate and edge by RSI bins."""
    results = []

    for rsi_min, rsi_max in RSI_BINS:
        mask = (trades['rsi'] >= rsi_min) & (trades['rsi'] < rsi_max)
        subset = trades[mask]

        if len(subset) >= 10:
            wr = (subset['outcome'] == 'WIN').sum() / len(subset) * 100
            edge = subset['pnl'].mean()
            total_pnl = subset['pnl'].sum()

            results.append({
                'rsi_range': f"{rsi_min}-{rsi_max}",
                'trades': len(subset),
                'wr': wr,
                'edge': edge,
                'total_pnl': total_pnl,
            })

    return pd.DataFrame(results)


def analyze_atr_filter(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze win rate and edge by ATR percentile bins."""
    results = []

    for atr_min, atr_max in ATR_PERCENTILE_BINS:
        mask = (trades['atr_pct'] >= atr_min) & (trades['atr_pct'] < atr_max)
        subset = trades[mask]

        if len(subset) >= 10:
            wr = (subset['outcome'] == 'WIN').sum() / len(subset) * 100
            edge = subset['pnl'].mean()
            total_pnl = subset['pnl'].sum()

            results.append({
                'atr_percentile': f"{atr_min}-{atr_max}",
                'trades': len(subset),
                'wr': wr,
                'edge': edge,
                'total_pnl': total_pnl,
            })

    return pd.DataFrame(results)


def analyze_trend_filter(trades: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Analyze win rate and edge by trend condition."""
    results = []

    for trend in TREND_CONDITIONS:
        subset = trades[trades['trend'] == trend]

        if len(subset) >= 10:
            wr = (subset['outcome'] == 'WIN').sum() / len(subset) * 100
            edge = subset['pnl'].mean()
            total_pnl = subset['pnl'].sum()

            results.append({
                'trend': trend,
                'trades': len(subset),
                'wr': wr,
                'edge': edge,
                'total_pnl': total_pnl,
            })

    return pd.DataFrame(results)


# ============================================================
# Main Execution
# ============================================================

def main():
    print("="*70)
    print("Context Filter Research")
    print("="*70)

    # Load data
    print(f"\nLoading data: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} bars")

    # Check for required column
    if 'pattern_3' not in df.columns:
        print("ERROR: Data file missing 'pattern_3' column")
        print("Please run pattern detection on the data first")
        return

    # Calculate context indicators
    print("\nCalculating context indicators...")
    df = calculate_context_indicators(df)

    # Prepare output
    output_lines = [
        "# Context Filter Research",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Data: {len(df)} bars from {DATA_FILE.name}",
        "",
        "## Objective",
        "Analyze market context (RSI, ATR, trend) at pattern entry points to identify",
        "profitable filters that improve win rate and edge.",
        "",
    ]

    # Analyze each pattern
    all_patterns = {**PATTERN_CONFIGS_LONG, **PATTERN_CONFIGS_SHORT}

    for pattern, (tp, sl) in all_patterns.items():
        direction = 'LONG' if pattern in PATTERN_CONFIGS_LONG else 'SHORT'

        print(f"\nAnalyzing {pattern} ({direction})...")

        # Backtest pattern
        trades_list = backtest_pattern(df, pattern, direction, tp, sl)

        if len(trades_list) < 20:
            print(f"  Skipping (only {len(trades_list)} trades)")
            continue

        trades_df = pd.DataFrame(trades_list)

        # Overall stats
        overall_wr = (trades_df['outcome'] == 'WIN').sum() / len(trades_df) * 100
        overall_edge = trades_df['pnl'].mean()
        overall_pnl = trades_df['pnl'].sum()

        output_lines.extend([
            f"## {pattern} ({direction})",
            f"**TP/SL**: {tp}% / {sl}%  |  **Trades**: {len(trades_df)}  |  **Baseline WR**: {overall_wr:.1f}%  |  **Edge**: {overall_edge:+.3f}%  |  **PnL**: {overall_pnl:+.1f}%",
            "",
        ])

        # RSI Analysis
        rsi_results = analyze_rsi_filter(trades_df, direction)
        if not rsi_results.empty:
            output_lines.append("### RSI Filter")
            output_lines.append(f"| RSI Range | Trades | WR | Edge | PnL |")
            output_lines.append(f"|-----------|--------|-----|------|-----|")
            for _, row in rsi_results.iterrows():
                output_lines.append(
                    f"| {row['rsi_range']} | {row['trades']} | {row['wr']:.1f}% | {row['edge']:+.3f}% | {row['total_pnl']:+.1f}% |"
                )
            output_lines.append("")

            # Identify best RSI range
            best_rsi = rsi_results.loc[rsi_results['edge'].idxmax()]
            if best_rsi['edge'] > overall_edge * 1.2:
                output_lines.append(
                    f"**✅ Strong RSI Filter**: {best_rsi['rsi_range']} → "
                    f"WR {best_rsi['wr']:.1f}% (+{best_rsi['wr'] - overall_wr:.1f}pp), "
                    f"Edge {best_rsi['edge']:+.3f}% (baseline {overall_edge:+.3f}%)"
                )
                output_lines.append("")

        # ATR Analysis
        atr_results = analyze_atr_filter(trades_df)
        if not atr_results.empty:
            output_lines.append("### ATR Percentile Filter")
            output_lines.append(f"| ATR Pct | Trades | WR | Edge | PnL |")
            output_lines.append(f"|---------|--------|-----|------|-----|")
            for _, row in atr_results.iterrows():
                output_lines.append(
                    f"| {row['atr_percentile']} | {row['trades']} | {row['wr']:.1f}% | {row['edge']:+.3f}% | {row['total_pnl']:+.1f}% |"
                )
            output_lines.append("")

            # Identify best ATR range
            best_atr = atr_results.loc[atr_results['edge'].idxmax()]
            if best_atr['edge'] > overall_edge * 1.2:
                output_lines.append(
                    f"**✅ Strong ATR Filter**: {best_atr['atr_percentile']} → "
                    f"WR {best_atr['wr']:.1f}% (+{best_atr['wr'] - overall_wr:.1f}pp), "
                    f"Edge {best_atr['edge']:+.3f}% (baseline {overall_edge:+.3f}%)"
                )
                output_lines.append("")

        # Trend Analysis
        trend_results = analyze_trend_filter(trades_df, direction)
        if not trend_results.empty:
            output_lines.append("### Trend Filter")
            output_lines.append(f"| Trend | Trades | WR | Edge | PnL |")
            output_lines.append(f"|-------|--------|-----|------|-----|")
            for _, row in trend_results.iterrows():
                output_lines.append(
                    f"| {row['trend']} | {row['trades']} | {row['wr']:.1f}% | {row['edge']:+.3f}% | {row['total_pnl']:+.1f}% |"
                )
            output_lines.append("")

            # Identify best trend
            best_trend = trend_results.loc[trend_results['edge'].idxmax()]
            if best_trend['edge'] > overall_edge * 1.2:
                output_lines.append(
                    f"**✅ Strong Trend Filter**: {best_trend['trend']} → "
                    f"WR {best_trend['wr']:.1f}% (+{best_trend['wr'] - overall_wr:.1f}pp), "
                    f"Edge {best_trend['edge']:+.3f}% (baseline {overall_edge:+.3f}%)"
                )
                output_lines.append("")

        output_lines.append("---")
        output_lines.append("")

    # Add summary section
    output_lines.extend([
        "## Summary",
        "",
        "### Key Findings",
        "- Filters with >20% edge improvement over baseline are highlighted",
        "- RSI filters may help avoid overbought/oversold extremes",
        "- ATR filters can identify high-volatility opportunities or risk",
        "- Trend filters align pattern direction with broader market momentum",
        "",
        "### Next Steps",
        "1. Validate top filters with walk-forward analysis",
        "2. Test combined filters (e.g., RSI + Trend)",
        "3. Implement as optional config in pattern_5m_config.yaml",
        "4. Monitor performance in paper trading before mainnet",
        "",
    ])

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"\n✅ Results saved to: {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
