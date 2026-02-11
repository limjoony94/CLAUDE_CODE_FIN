"""
Pattern Deep Analysis Research
===============================
v1.27.0 프로덕션 봇의 52개 패턴(32L+20S) 심층 연구.
기존 연구(TP scaling, edge decomposition, MFE/MAE, risk management)와 중복 없이
개별 패턴 프로파일링, 패턴 간 상관관계, 시장 레짐별 성과, 포트폴리오 구성 분석, 안정성 검증을 수행.

Phase 1: Individual Pattern Profiling
Phase 2: Pattern Correlation & Clustering
Phase 3: Pattern Regime Analysis
Phase 4: Portfolio Construction Analysis
Phase 5: Robustness & Stability
Phase 6: Summary & Actionable Insights
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import time
import sys
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.constants import (
    VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
    PATTERN_OPTIMAL_TPSL, PATTERN_STATS, BOT_VERSION,
)

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
BOOTSTRAP_SIMS = 10000
BLOCK_SIZE = 10

print("=" * 70)
print(f"Pattern Deep Analysis (v{BOT_VERSION})")
print("=" * 70)

# ============================================================
# Build portfolio map
# ============================================================
PORTFOLIO = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("LONG", tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("SHORT", tp, sl)

ALL_PATTERNS = sorted(PORTFOLIO.keys())
N_PAT = len(ALL_PATTERNS)
print(f"Patterns: {N_PAT} ({len(VALIDATED_LONG_PATTERNS)}L + {len(VALIDATED_SHORT_PATTERNS)}S)")

# ============================================================
# Load data — use pre-classified type_code/pattern_3 columns
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
print(f"Loaded: {len(df)} bars")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

df['timestamp'] = pd.to_datetime(df['timestamp'])
timestamps = df['timestamp'].values
dates_dt = pd.to_datetime(df['timestamp'])

# Use pre-classified type_code column
types = df['type_code'].values

# Build pattern index from pre-classified pattern_3 column
pattern_indices = defaultdict(list)
for i in range(len(df)):
    pat = df['pattern_3'].iloc[i]
    if pd.notna(pat) and pat in PORTFOLIO:
        pattern_indices[pat].append(i)
for k in pattern_indices:
    pattern_indices[k] = np.array(pattern_indices[k])

print("Using pre-classified type_code/pattern_3 columns.\n")


# ============================================================
# Backtest engines
# ============================================================
def bt_fixed(indices, direction, tp_pct, sl_pct):
    """Type 1 independent backtest — returns list of PnL values."""
    pnls = []
    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
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
                pnls.append((tp_pct if abs(tpp - entry) <= abs(slp - entry) else -sl_pct) * LEVERAGE - FEE_PCT)
                break
            elif ht:
                pnls.append(tp_pct * LEVERAGE - FEE_PCT)
                break
            elif hs:
                pnls.append(-sl_pct * LEVERAGE - FEE_PCT)
                break
    return pnls


def bt_fixed_detailed(indices, direction, tp_pct, sl_pct):
    """Type 1 + entry/exit bar/time, bars_held."""
    trades = []
    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        entry_bar = idx + 1
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
                pnl = (tp_pct if abs(tpp - entry) <= abs(slp - entry) else -sl_pct) * LEVERAGE - FEE_PCT
                trades.append({'signal_bar': idx, 'entry_bar': entry_bar, 'exit_bar': j,
                               'bars_held': j - entry_bar, 'pnl': pnl, 'is_win': pnl > 0})
                break
            elif ht:
                trades.append({'signal_bar': idx, 'entry_bar': entry_bar, 'exit_bar': j,
                               'bars_held': j - entry_bar, 'pnl': tp_pct * LEVERAGE - FEE_PCT, 'is_win': True})
                break
            elif hs:
                trades.append({'signal_bar': idx, 'entry_bar': entry_bar, 'exit_bar': j,
                               'bars_held': j - entry_bar, 'pnl': -sl_pct * LEVERAGE - FEE_PCT, 'is_win': False})
                break
    return trades


def collect_all_raw_trades():
    """Pre-compute raw trades for each pattern (with bar indices)."""
    all_trades = {}
    for pat in ALL_PATTERNS:
        direction, tp, sl = PORTFOLIO[pat]
        indices = pattern_indices.get(pat, np.array([]))
        trades = bt_fixed_detailed(indices, direction, tp, sl)
        for t in trades:
            t['pattern'] = pat
            t['direction'] = direction
        all_trades[pat] = trades
    return all_trades


def collect_trades_1pos(pattern_map):
    """1-position-at-a-time portfolio backtest."""
    raw_trades = []
    for pat, (direction, tp, sl) in pattern_map.items():
        indices = pattern_indices.get(pat, np.array([]))
        for idx in indices:
            if idx + 1 >= n_bars:
                continue
            entry = opens[idx + 1]
            if entry <= 0:
                continue
            entry_bar = idx + 1
            if direction == 'LONG':
                tpp = entry * (1 + tp / 100)
                slp = entry * (1 - sl / 100)
            else:
                tpp = entry * (1 - tp / 100)
                slp = entry * (1 + sl / 100)
            for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
                ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
                hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
                if ht and hs:
                    pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
                    raw_trades.append((entry_bar, j, pnl, pat))
                    break
                elif ht:
                    raw_trades.append((entry_bar, j, tp * LEVERAGE - FEE_PCT, pat))
                    break
                elif hs:
                    raw_trades.append((entry_bar, j, -sl * LEVERAGE - FEE_PCT, pat))
                    break
    raw_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for entry_bar, exit_bar, pnl, pat in raw_trades:
        if entry_bar > last_exit:
            filtered.append((entry_bar, exit_bar, pnl, pat))
            last_exit = exit_bar
    return filtered


def eval_trades_simple(trades):
    """Compute PnL/WR/MDD/PF from trade list."""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_mdd': 0}
    pnl_list = [t[2] if isinstance(t, tuple) else t['pnl'] for t in trades]
    cum_pnl = 0
    peak_pnl = 0
    max_dd = 0
    wins = 0
    win_sum = 0
    loss_sum = 0
    for p in pnl_list:
        cum_pnl += p
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        dd = peak_pnl - cum_pnl
        if dd > max_dd:
            max_dd = dd
        if p > 0:
            wins += 1
            win_sum += p
        else:
            loss_sum += abs(p)
    return {
        'pnl': cum_pnl,
        'trades': len(pnl_list),
        'wr': wins / len(pnl_list) * 100,
        'mdd': max_dd,
        'pf': win_sum / loss_sum if loss_sum > 0 else 999,
        'pnl_mdd': cum_pnl / max_dd if max_dd > 0 else 999,
    }


# ============================================================
# Pre-compute all raw trades
# ============================================================
t0_total = time.time()
print("Pre-computing all raw trades...")
all_raw_trades = collect_all_raw_trades()

# Full portfolio 1-pos trades
portfolio_trades = collect_trades_1pos(PORTFOLIO)
portfolio_stats = eval_trades_simple(portfolio_trades)
print(f"Portfolio: {portfolio_stats['trades']} trades, PnL {portfolio_stats['pnl']:+.1f}%, "
      f"WR {portfolio_stats['wr']:.1f}%, MDD {portfolio_stats['mdd']:.1f}%\n")


# ============================================================
# Phase 1: Individual Pattern Profiling
# ============================================================
print("=" * 70)
print("Phase 1: Individual Pattern Profiling")
print("=" * 70)

phase1 = {}

for pat in ALL_PATTERNS:
    direction, tp, sl = PORTFOLIO[pat]
    trades = all_raw_trades[pat]
    if not trades:
        continue

    pnls = [t['pnl'] for t in trades]
    bars_held = [t['bars_held'] for t in trades]
    n_trades = len(trades)
    wins = sum(1 for p in pnls if p > 0)
    losses = n_trades - wins
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [abs(p) for p in pnls if p <= 0]

    # Consecutive wins/losses
    max_consec_w = 0
    max_consec_l = 0
    cur_w = 0
    cur_l = 0
    for p in pnls:
        if p > 0:
            cur_w += 1
            cur_l = 0
            max_consec_w = max(max_consec_w, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_consec_l = max(max_consec_l, cur_l)

    # Sharpe-like metric
    pnl_arr = np.array(pnls)
    sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr)) if np.std(pnl_arr) > 0 else 0

    # Time distribution (UTC hour, weekday)
    signal_bars = [t['signal_bar'] for t in trades]
    hours = [dates_dt.iloc[b].hour for b in signal_bars if b < len(dates_dt)]
    weekdays = [dates_dt.iloc[b].weekday() for b in signal_bars if b < len(dates_dt)]
    hour_dist = {h: 0 for h in range(24)}
    for h in hours:
        hour_dist[h] += 1
    weekday_dist = {d: 0 for d in range(7)}
    for d in weekdays:
        weekday_dist[d] += 1

    # Monthly performance
    monthly = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0})
    for t in trades:
        ym = dates_dt.iloc[t['signal_bar']].strftime('%Y-%m') if t['signal_bar'] < len(dates_dt) else 'unknown'
        monthly[ym]['trades'] += 1
        monthly[ym]['pnl'] += t['pnl']
        if t['is_win']:
            monthly[ym]['wins'] += 1

    # Frequency trend: 9 blocks of 30 days
    total_days = (dates_dt.iloc[-1] - dates_dt.iloc[0]).days
    block_days = 30
    n_blocks = min(9, total_days // block_days)
    block_counts = []
    start_date = dates_dt.iloc[0]
    for b in range(n_blocks):
        bstart = start_date + pd.Timedelta(days=b * block_days)
        bend = bstart + pd.Timedelta(days=block_days)
        count = sum(1 for t in trades if bstart <= dates_dt.iloc[t['signal_bar']] < bend)
        block_counts.append(count)

    # Linear regression slope for frequency trend
    freq_slope = 0.0
    if len(block_counts) >= 3:
        x = np.arange(len(block_counts))
        coeffs = np.polyfit(x, block_counts, 1)
        freq_slope = float(coeffs[0])

    phase1[pat] = {
        'direction': direction, 'tp': tp, 'sl': sl,
        'trades': n_trades, 'wins': wins, 'losses': losses,
        'wr': wins / n_trades * 100,
        'total_pnl': float(sum(pnls)),
        'avg_win': float(np.mean(win_pnls)) if win_pnls else 0,
        'avg_loss': float(np.mean(loss_pnls)) if loss_pnls else 0,
        'pf': float(sum(win_pnls) / sum(loss_pnls)) if loss_pnls else 999,
        'max_consec_wins': max_consec_w,
        'max_consec_losses': max_consec_l,
        'avg_bars_held': float(np.mean(bars_held)),
        'median_bars_held': float(np.median(bars_held)),
        'sharpe': sharpe,
        'hour_dist': hour_dist,
        'weekday_dist': weekday_dist,
        'monthly': {k: dict(v) for k, v in sorted(monthly.items())},
        'block_counts': block_counts,
        'freq_slope': freq_slope,
    }

# Print summary table
print(f"\n  {'Pattern':<14} {'Dir':<5} {'Tr':>4} {'WR':>6} {'PnL':>8} {'PF':>5} "
      f"{'Sharpe':>6} {'MCW':>3} {'MCL':>3} {'AvgBH':>6} {'FSlope':>7}")
print("  " + "-" * 85)
for pat in ALL_PATTERNS:
    p = phase1[pat]
    print(f"  {pat:<14} {p['direction']:<5} {p['trades']:>4} {p['wr']:>5.1f}% "
          f"{p['total_pnl']:>+7.1f}% {p['pf']:>5.2f} {p['sharpe']:>6.3f} "
          f"{p['max_consec_wins']:>3} {p['max_consec_losses']:>3} "
          f"{p['avg_bars_held']:>5.1f} {p['freq_slope']:>+6.2f}")

total_t1_trades = sum(p['trades'] for p in phase1.values())
print(f"\n  Total Type 1 trades: {total_t1_trades}")
print(f"  Portfolio Type 2 trades: {portfolio_stats['trades']}")


# ============================================================
# Phase 2: Pattern Correlation & Clustering
# ============================================================
print("\n" + "=" * 70)
print("Phase 2: Pattern Correlation & Clustering")
print("=" * 70)

# Build daily PnL series per pattern
total_days_count = (dates_dt.iloc[-1] - dates_dt.iloc[0]).days + 1
date_range = pd.date_range(dates_dt.iloc[0].date(), periods=total_days_count, freq='D')
daily_pnl = pd.DataFrame(0.0, index=date_range, columns=ALL_PATTERNS)

for pat in ALL_PATTERNS:
    for t in all_raw_trades[pat]:
        day = dates_dt.iloc[t['signal_bar']].normalize()
        if day in daily_pnl.index:
            daily_pnl.loc[day, pat] += t['pnl']

# Correlation matrix
corr_matrix = daily_pnl.corr()

# Top/bottom 10 correlated pairs
pairs = []
for i in range(N_PAT):
    for j in range(i + 1, N_PAT):
        pairs.append((ALL_PATTERNS[i], ALL_PATTERNS[j], corr_matrix.iloc[i, j]))

pairs.sort(key=lambda x: x[2], reverse=True)
top10_corr = pairs[:10]
bot10_corr = pairs[-10:]

print("\n  Top 10 most CORRELATED pairs:")
for a, b, c in top10_corr:
    print(f"    {a:<14} ~ {b:<14}  r={c:+.3f}")
print("\n  Top 10 most ANTI-CORRELATED pairs:")
for a, b, c in bot10_corr:
    print(f"    {a:<14} ~ {b:<14}  r={c:+.3f}")

# Hierarchical clustering
corr_dist = 1 - corr_matrix.values
np.fill_diagonal(corr_dist, 0)
corr_dist = np.clip(corr_dist, 0, 2)
corr_dist = (corr_dist + corr_dist.T) / 2
linkage_matrix = linkage(squareform(corr_dist), method='ward')
n_clusters = 6
clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
cluster_map = {}
for i, pat in enumerate(ALL_PATTERNS):
    cluster_map[pat] = int(clusters[i])

print(f"\n  {n_clusters} clusters (Ward linkage):")
for c in range(1, n_clusters + 1):
    members = [p for p in ALL_PATTERNS if cluster_map[p] == c]
    dirs = [PORTFOLIO[p][0][0] for p in members]
    print(f"    Cluster {c} ({len(members)} patterns, {dirs.count('L')}L/{dirs.count('S')}S): "
          f"{', '.join(members[:5])}{'...' if len(members) > 5 else ''}")

# Diversification ratio
individual_stds = daily_pnl[ALL_PATTERNS].std()
portfolio_daily = daily_pnl[ALL_PATTERNS].sum(axis=1)
portfolio_std = portfolio_daily.std()
div_ratio = float(individual_stds.mean() / portfolio_std) if portfolio_std > 0 else 1.0
print(f"\n  Diversification ratio: {div_ratio:.2f} (higher = more diversified)")

# Jaccard similarity (same-day occurrence)
occurrence = (daily_pnl != 0).astype(int)
jaccard_pairs = []
for i in range(N_PAT):
    for j in range(i + 1, N_PAT):
        inter = ((occurrence.iloc[:, i] == 1) & (occurrence.iloc[:, j] == 1)).sum()
        union = ((occurrence.iloc[:, i] == 1) | (occurrence.iloc[:, j] == 1)).sum()
        jacc = inter / union if union > 0 else 0
        jaccard_pairs.append((ALL_PATTERNS[i], ALL_PATTERNS[j], jacc))

jaccard_pairs.sort(key=lambda x: x[2], reverse=True)
print("\n  Top 10 timing overlap (Jaccard similarity):")
for a, b, j in jaccard_pairs[:10]:
    print(f"    {a:<14} & {b:<14}  J={j:.3f}")

phase2 = {
    'top10_corr': [{'a': a, 'b': b, 'r': float(c)} for a, b, c in top10_corr],
    'bot10_corr': [{'a': a, 'b': b, 'r': float(c)} for a, b, c in bot10_corr],
    'clusters': cluster_map,
    'diversification_ratio': div_ratio,
    'top10_jaccard': [{'a': a, 'b': b, 'jaccard': float(j)} for a, b, j in jaccard_pairs[:10]],
    'corr_matrix_summary': {
        'mean': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
        'median': float(np.median(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)])),
        'max': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()),
        'min': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()),
    },
}


# ============================================================
# Phase 3: Pattern Regime Analysis
# ============================================================
print("\n" + "=" * 70)
print("Phase 3: Pattern Regime Analysis")
print("=" * 70)

# Volatility regime: 288-bar rolling ATR% → terciles
atr_period = 288  # ~1 day of 5m bars
tr = np.maximum(highs - lows,
                np.maximum(np.abs(highs - np.roll(closes, 1)),
                           np.abs(lows - np.roll(closes, 1))))
tr[0] = highs[0] - lows[0]
atr_rolling = pd.Series(tr).rolling(atr_period).mean().values
atr_pct = atr_rolling / closes * 100

vol_terciles = np.nanpercentile(atr_pct[atr_period:], [33.33, 66.67])

def classify_vol_regime(bar_idx):
    v = atr_pct[bar_idx]
    if np.isnan(v):
        return 'Unknown'
    if v <= vol_terciles[0]:
        return 'Low'
    elif v <= vol_terciles[1]:
        return 'Med'
    else:
        return 'High'

# Trend regime: efficiency ratio (20-bar)
eff_period = 20
price_change = np.abs(closes - np.roll(closes, eff_period))
sum_bars = pd.Series(np.abs(closes - np.roll(closes, 1))).rolling(eff_period).sum().values
eff_ratio = np.where(sum_bars > 0, price_change / sum_bars, 0)
eff_terciles = np.nanpercentile(eff_ratio[eff_period:], [33.33, 66.67])

def classify_trend_regime(bar_idx):
    e = eff_ratio[bar_idx]
    if np.isnan(e) or bar_idx < eff_period:
        return 'Unknown'
    if e >= eff_terciles[1]:
        return 'Trending'
    elif e >= eff_terciles[0]:
        return 'Mixed'
    else:
        return 'Ranging'

# Per-pattern regime performance
phase3 = {'vol_terciles': [float(v) for v in vol_terciles],
           'eff_terciles': [float(e) for e in eff_terciles],
           'patterns': {}}

print(f"\n  Vol terciles (ATR%): Low<{vol_terciles[0]:.4f}%, Med<{vol_terciles[1]:.4f}%, High above")
print(f"  Eff terciles: Ranging<{eff_terciles[0]:.3f}, Mixed<{eff_terciles[1]:.3f}, Trending above")

print(f"\n  {'Pattern':<14} {'Vol_Regime':>28} {'Trend_Regime':>32}")
print(f"  {'':14} {'Low':>9} {'Med':>9} {'High':>9} {'Rng':>10} {'Mix':>10} {'Trn':>10}")
print("  " + "-" * 85)

for pat in ALL_PATTERNS:
    trades = all_raw_trades[pat]
    vol_stats = {'Low': [], 'Med': [], 'High': []}
    trend_stats = {'Ranging': [], 'Mixed': [], 'Trending': []}

    for t in trades:
        bar = t['signal_bar']
        vr = classify_vol_regime(bar)
        tr_r = classify_trend_regime(bar)
        if vr in vol_stats:
            vol_stats[vr].append(t['pnl'])
        if tr_r in trend_stats:
            trend_stats[tr_r].append(t['pnl'])

    pat_regime = {}
    vol_parts = []
    for regime in ['Low', 'Med', 'High']:
        pnls = vol_stats[regime]
        n = len(pnls)
        wr = sum(1 for p in pnls if p > 0) / n * 100 if n > 0 else 0
        pat_regime[f'vol_{regime}'] = {'n': n, 'wr': round(wr, 1), 'pnl': round(sum(pnls), 1)}
        vol_parts.append(f"{n:>3}|{wr:>4.0f}%")

    trend_parts = []
    for regime in ['Ranging', 'Mixed', 'Trending']:
        pnls = trend_stats[regime]
        n = len(pnls)
        wr = sum(1 for p in pnls if p > 0) / n * 100 if n > 0 else 0
        pat_regime[f'trend_{regime}'] = {'n': n, 'wr': round(wr, 1), 'pnl': round(sum(pnls), 1)}
        trend_parts.append(f"{n:>3}|{wr:>4.0f}%")

    phase3['patterns'][pat] = pat_regime
    print(f"  {pat:<14} {vol_parts[0]:>9} {vol_parts[1]:>9} {vol_parts[2]:>9}"
          f" {trend_parts[0]:>10} {trend_parts[1]:>10} {trend_parts[2]:>10}")

# Drawdown synchronization between pattern pairs (top 10 highest overlap)
# Compute per-pattern cumulative PnL drawdown periods
dd_periods = {}
for pat in ALL_PATTERNS:
    trades = all_raw_trades[pat]
    if not trades:
        dd_periods[pat] = set()
        continue
    cum = 0
    peak = 0
    in_dd_days = set()
    for t in trades:
        cum += t['pnl']
        if cum > peak:
            peak = cum
        if peak - cum > 0:
            day = dates_dt.iloc[t['signal_bar']].normalize()
            in_dd_days.add(day)
    dd_periods[pat] = in_dd_days

dd_overlap_pairs = []
for i in range(N_PAT):
    for j in range(i + 1, N_PAT):
        pa, pb = ALL_PATTERNS[i], ALL_PATTERNS[j]
        inter = len(dd_periods[pa] & dd_periods[pb])
        union = len(dd_periods[pa] | dd_periods[pb])
        overlap = inter / union if union > 0 else 0
        dd_overlap_pairs.append((pa, pb, overlap))

dd_overlap_pairs.sort(key=lambda x: x[2], reverse=True)
print(f"\n  Top 10 drawdown synchronization (Jaccard overlap of DD days):")
for a, b, o in dd_overlap_pairs[:10]:
    print(f"    {a:<14} & {b:<14}  overlap={o:.3f}")

phase3['dd_sync_top10'] = [{'a': a, 'b': b, 'overlap': float(o)} for a, b, o in dd_overlap_pairs[:10]]


# ============================================================
# Phase 4: Portfolio Construction Analysis
# ============================================================
print("\n" + "=" * 70)
print("Phase 4: Portfolio Construction Analysis")
print("=" * 70)

# Marginal Sharpe contribution: remove each pattern, measure Sharpe change
phase4 = {'marginal_sharpe': {}, 'l1o': {}, 'concentration': {}, 'long_short_balance': {}}

print("\n  Leave-one-out analysis:")
print(f"  {'Pattern':<14} {'PnL_full':>8} {'PnL_w/o':>8} {'dPnL':>7} {'MDD_full':>8} {'MDD_w/o':>8} "
      f"{'PnL/MDD_f':>9} {'PnL/MDD_w':>9} {'Verdict':>8}")
print("  " + "-" * 95)

port_sharpe = float(np.mean(portfolio_daily) / portfolio_daily.std()) if portfolio_daily.std() > 0 else 0

for pat in ALL_PATTERNS:
    # Build portfolio without this pattern
    port_wo = {p: PORTFOLIO[p] for p in PORTFOLIO if p != pat}
    trades_wo = collect_trades_1pos(port_wo)
    stats_wo = eval_trades_simple(trades_wo)

    # Daily PnL without pattern
    daily_wo = portfolio_daily - daily_pnl[pat]
    sharpe_wo = float(np.mean(daily_wo) / daily_wo.std()) if daily_wo.std() > 0 else 0
    marginal_sharpe = port_sharpe - sharpe_wo

    phase4['marginal_sharpe'][pat] = round(marginal_sharpe, 4)
    phase4['l1o'][pat] = {
        'pnl_without': round(stats_wo['pnl'], 1),
        'mdd_without': round(stats_wo['mdd'], 1),
        'pnl_mdd_without': round(stats_wo['pnl_mdd'], 1),
        'trades_without': stats_wo['trades'],
    }

    # Verdict: "keep" if removing hurts, "removable" if removing improves
    verdict = "keep" if stats_wo['pnl_mdd'] < portfolio_stats['pnl_mdd'] else "removable"

    print(f"  {pat:<14} {portfolio_stats['pnl']:>+7.1f}% {stats_wo['pnl']:>+7.1f}% "
          f"{stats_wo['pnl'] - portfolio_stats['pnl']:>+6.1f}% "
          f"{portfolio_stats['mdd']:>7.1f}% {stats_wo['mdd']:>7.1f}% "
          f"{portfolio_stats['pnl_mdd']:>8.1f}x {stats_wo['pnl_mdd']:>8.1f}x "
          f"{verdict:>8}")

# Concentration risk
pat_pnl_contrib = {}
for pat in ALL_PATTERNS:
    pat_pnl_contrib[pat] = phase1[pat]['total_pnl']

sorted_by_pnl = sorted(pat_pnl_contrib.items(), key=lambda x: x[1], reverse=True)
total_pnl_abs = sum(abs(v) for v in pat_pnl_contrib.values())
total_pnl_pos = sum(v for v in pat_pnl_contrib.values() if v > 0)

top1_pct = sorted_by_pnl[0][1] / total_pnl_pos * 100 if total_pnl_pos > 0 else 0
top5_pct = sum(v for _, v in sorted_by_pnl[:5]) / total_pnl_pos * 100 if total_pnl_pos > 0 else 0
top10_pct = sum(v for _, v in sorted_by_pnl[:10]) / total_pnl_pos * 100 if total_pnl_pos > 0 else 0
top20_pct = sum(v for _, v in sorted_by_pnl[:20]) / total_pnl_pos * 100 if total_pnl_pos > 0 else 0

# Herfindahl index
shares = np.array([v / total_pnl_pos for v in pat_pnl_contrib.values() if v > 0])
hhi = float(np.sum(shares ** 2))

print(f"\n  Concentration risk (Type 1 PnL contribution):")
print(f"    Top-1:  {sorted_by_pnl[0][0]} → {top1_pct:.1f}%")
print(f"    Top-5:  {top5_pct:.1f}%")
print(f"    Top-10: {top10_pct:.1f}%")
print(f"    Top-20: {top20_pct:.1f}%")
print(f"    Herfindahl index: {hhi:.4f} (lower = more diversified, 1/{N_PAT} = {1/N_PAT:.4f})")

phase4['concentration'] = {
    'top1': {'pattern': sorted_by_pnl[0][0], 'pct': round(top1_pct, 1)},
    'top5_pct': round(top5_pct, 1),
    'top10_pct': round(top10_pct, 1),
    'top20_pct': round(top20_pct, 1),
    'herfindahl': round(hhi, 4),
}

# LONG vs SHORT balance
long_trades = [t for t in portfolio_trades if t[3] in VALIDATED_LONG_PATTERNS]
short_trades = [t for t in portfolio_trades if t[3] in VALIDATED_SHORT_PATTERNS]
long_pnl = sum(t[2] for t in long_trades)
short_pnl = sum(t[2] for t in short_trades)

# Daily LONG/SHORT PnL correlation
long_daily = daily_pnl[[p for p in ALL_PATTERNS if PORTFOLIO[p][0] == 'LONG']].sum(axis=1)
short_daily = daily_pnl[[p for p in ALL_PATTERNS if PORTFOLIO[p][0] == 'SHORT']].sum(axis=1)
ls_corr = float(long_daily.corr(short_daily))

print(f"\n  LONG vs SHORT balance:")
print(f"    LONG:  {len(long_trades)} trades, PnL {long_pnl:+.1f}%")
print(f"    SHORT: {len(short_trades)} trades, PnL {short_pnl:+.1f}%")
print(f"    Daily PnL correlation (L vs S): {ls_corr:.3f}")
print(f"    Balance ratio: {len(long_trades)/(len(long_trades)+len(short_trades))*100:.1f}% LONG")

phase4['long_short_balance'] = {
    'long_trades': len(long_trades), 'short_trades': len(short_trades),
    'long_pnl': round(long_pnl, 1), 'short_pnl': round(short_pnl, 1),
    'daily_corr': round(ls_corr, 3),
    'long_pct': round(len(long_trades) / (len(long_trades) + len(short_trades)) * 100, 1),
}


# ============================================================
# Phase 5: Robustness & Stability
# ============================================================
print("\n" + "=" * 70)
print("Phase 5: Robustness & Stability")
print("=" * 70)

phase5 = {'rolling': {}, 'edge_decay': {}, 'bootstrap': {}, 'worst_periods': {}}

# 60-day rolling window: WR/PnL/trades
print("\n  60-day rolling window stats (portfolio):")
window_days = 60
rolling_results = []
for start_idx in range(0, total_days_count - window_days, 30):
    w_start = date_range[start_idx]
    w_end = w_start + pd.Timedelta(days=window_days)
    w_trades = [t for t in portfolio_trades
                if w_start <= dates_dt.iloc[t[0]].normalize() < w_end]
    if w_trades:
        w_pnls = [t[2] for t in w_trades]
        wr = sum(1 for p in w_pnls if p > 0) / len(w_pnls) * 100
        rolling_results.append({
            'start': str(w_start.date()), 'trades': len(w_trades),
            'wr': round(wr, 1), 'pnl': round(sum(w_pnls), 1)
        })

if rolling_results:
    print(f"    {'Start':<12} {'Trades':>7} {'WR':>7} {'PnL':>8}")
    for r in rolling_results:
        print(f"    {r['start']:<12} {r['trades']:>7} {r['wr']:>6.1f}% {r['pnl']:>+7.1f}%")
phase5['rolling'] = rolling_results

# Edge decay: 3 equal time segments
n_third = len(df) // 3
segments = [
    ('Early', 0, n_third),
    ('Mid', n_third, 2 * n_third),
    ('Late', 2 * n_third, len(df)),
]

print(f"\n  Edge decay analysis (3 temporal segments):")
print(f"  {'Pattern':<14} {'Early_WR':>8} {'Mid_WR':>8} {'Late_WR':>8} {'Slope':>7} {'Status':>10}")
print("  " + "-" * 65)

for pat in ALL_PATTERNS:
    trades = all_raw_trades[pat]
    seg_wrs = []
    for seg_name, seg_start, seg_end in segments:
        seg_trades = [t for t in trades if seg_start <= t['signal_bar'] < seg_end]
        if seg_trades:
            wr = sum(1 for t in seg_trades if t['is_win']) / len(seg_trades) * 100
        else:
            wr = 0
        seg_wrs.append(wr)

    # WR slope across segments
    if all(w > 0 for w in seg_wrs):
        slope = (seg_wrs[2] - seg_wrs[0]) / 2
    else:
        slope = 0

    if slope > 3:
        status = "Improving"
    elif slope < -3:
        status = "Decaying"
    else:
        status = "Stable"

    phase5['edge_decay'][pat] = {
        'early_wr': round(seg_wrs[0], 1), 'mid_wr': round(seg_wrs[1], 1),
        'late_wr': round(seg_wrs[2], 1), 'slope': round(slope, 2), 'status': status,
    }

    print(f"  {pat:<14} {seg_wrs[0]:>7.1f}% {seg_wrs[1]:>7.1f}% {seg_wrs[2]:>7.1f}% "
          f"{slope:>+6.1f} {status:>10}")

# Count statuses
decay_counts = defaultdict(int)
for v in phase5['edge_decay'].values():
    decay_counts[v['status']] += 1
print(f"\n  Status summary: Stable={decay_counts['Stable']}, "
      f"Improving={decay_counts['Improving']}, Decaying={decay_counts['Decaying']}")

# Bootstrap CI for portfolio (block bootstrap, 10-trade blocks)
print(f"\n  Bootstrap CI (portfolio, {BOOTSTRAP_SIMS:,} sims, block size={BLOCK_SIZE})...")
pnl_list = np.array([t[2] for t in portfolio_trades])
n_t = len(pnl_list)
n_blocks = n_t // BLOCK_SIZE

boot_pnl = []
boot_wr = []
boot_mdd = []

for _ in range(BOOTSTRAP_SIMS):
    # Block bootstrap: sample blocks with replacement
    block_indices = np.random.randint(0, n_blocks, size=n_blocks)
    sample = np.concatenate([pnl_list[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] for i in block_indices])

    boot_pnl.append(np.sum(sample))
    boot_wr.append(np.sum(sample > 0) / len(sample) * 100)

    cum = np.cumsum(sample)
    peak = np.maximum.accumulate(cum)
    mdd = np.max(peak - cum)
    boot_mdd.append(mdd)

boot_pnl = np.array(boot_pnl)
boot_wr = np.array(boot_wr)
boot_mdd = np.array(boot_mdd)

ci95_pnl = (float(np.percentile(boot_pnl, 2.5)), float(np.percentile(boot_pnl, 97.5)))
ci95_wr = (float(np.percentile(boot_wr, 2.5)), float(np.percentile(boot_wr, 97.5)))
ci95_mdd = (float(np.percentile(boot_mdd, 2.5)), float(np.percentile(boot_mdd, 97.5)))

print(f"    PnL  95% CI: [{ci95_pnl[0]:+.1f}%, {ci95_pnl[1]:+.1f}%]  (actual: {portfolio_stats['pnl']:+.1f}%)")
print(f"    WR   95% CI: [{ci95_wr[0]:.1f}%, {ci95_wr[1]:.1f}%]  (actual: {portfolio_stats['wr']:.1f}%)")
print(f"    MDD  95% CI: [{ci95_mdd[0]:.1f}%, {ci95_mdd[1]:.1f}%]  (actual: {portfolio_stats['mdd']:.1f}%)")

phase5['bootstrap'] = {
    'pnl_ci95': [round(v, 1) for v in ci95_pnl],
    'wr_ci95': [round(v, 1) for v in ci95_wr],
    'mdd_ci95': [round(v, 1) for v in ci95_mdd],
}

# Worst periods
pnl_series = pd.Series([t[2] for t in portfolio_trades],
                        index=[dates_dt.iloc[t[0]].normalize() for t in portfolio_trades])
daily_sum = pnl_series.groupby(level=0).sum()

worst_1d = float(daily_sum.min()) if len(daily_sum) > 0 else 0
worst_1d_date = str(daily_sum.idxmin().date()) if len(daily_sum) > 0 else 'N/A'

worst_7d = float(daily_sum.rolling(7).sum().min()) if len(daily_sum) >= 7 else 0
worst_30d = float(daily_sum.rolling(30).sum().min()) if len(daily_sum) >= 30 else 0
worst_60d = float(daily_sum.rolling(60).sum().min()) if len(daily_sum) >= 60 else 0
worst_90d = float(daily_sum.rolling(90).sum().min()) if len(daily_sum) >= 90 else 0

print(f"\n  Worst periods:")
print(f"    Worst 1 day:  {worst_1d:+.1f}% ({worst_1d_date})")
print(f"    Worst 7 days: {worst_7d:+.1f}%")
print(f"    Worst 30 days: {worst_30d:+.1f}%")
print(f"    Worst 60 days: {worst_60d:+.1f}%")
print(f"    Worst 90 days: {worst_90d:+.1f}%")

phase5['worst_periods'] = {
    'worst_1d': round(worst_1d, 1), 'worst_1d_date': worst_1d_date,
    'worst_7d': round(worst_7d, 1), 'worst_30d': round(worst_30d, 1),
    'worst_60d': round(worst_60d, 1), 'worst_90d': round(worst_90d, 1),
}


# ============================================================
# Phase 6: Summary & Actionable Insights
# ============================================================
print("\n" + "=" * 70)
print("Phase 6: Summary & Actionable Insights")
print("=" * 70)

# Composite quality score per pattern
# WR(25%) + Sharpe(20%) + 1/(1+max_consec_loss)(15%) + marginal_sharpe(20%) + regime_stability(20%)
phase6 = {'scores': {}, 'weak_patterns': [], 'strong_patterns': [], 'portfolio_suggestions': []}

# Normalize metrics to [0, 1]
wrs = [phase1[p]['wr'] for p in ALL_PATTERNS]
sharpes = [phase1[p]['sharpe'] for p in ALL_PATTERNS]
mcls = [phase1[p]['max_consec_losses'] for p in ALL_PATTERNS]
msharpes = [phase4['marginal_sharpe'][p] for p in ALL_PATTERNS]

# Regime stability: std of WR across vol regimes
regime_stabilities = []
for pat in ALL_PATTERNS:
    pr = phase3['patterns'][pat]
    vol_wrs = [pr[f'vol_{r}']['wr'] for r in ['Low', 'Med', 'High'] if pr[f'vol_{r}']['n'] >= 3]
    if len(vol_wrs) >= 2:
        regime_stabilities.append(100 - np.std(vol_wrs))  # Higher = more stable
    else:
        regime_stabilities.append(50)  # Neutral if not enough data

def normalize(vals):
    arr = np.array(vals, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)

wr_norm = normalize(wrs)
sharpe_norm = normalize(sharpes)
mcl_norm = normalize([1 / (1 + m) for m in mcls])
ms_norm = normalize(msharpes)
rs_norm = normalize(regime_stabilities)

print(f"\n  Composite Quality Score (WR 25% + Sharpe 20% + ConsecLoss 15% + MargSharpe 20% + RegimeStab 20%):")
print(f"  {'Pattern':<14} {'Score':>6} {'WR':>6} {'Sha':>5} {'MCL':>5} {'MSh':>5} {'RS':>5} {'Decay':>10}")
print("  " + "-" * 70)

for i, pat in enumerate(ALL_PATTERNS):
    score = (wr_norm[i] * 0.25 + sharpe_norm[i] * 0.20 + mcl_norm[i] * 0.15 +
             ms_norm[i] * 0.20 + rs_norm[i] * 0.20)
    phase6['scores'][pat] = round(float(score), 3)

scores_sorted = sorted(phase6['scores'].items(), key=lambda x: x[1], reverse=True)
for pat, score in scores_sorted:
    i = ALL_PATTERNS.index(pat)
    decay = phase5['edge_decay'][pat]['status']
    print(f"  {pat:<14} {score:>5.3f} {wr_norm[i]:>5.2f} {sharpe_norm[i]:>5.2f} "
          f"{mcl_norm[i]:>5.2f} {ms_norm[i]:>5.2f} {rs_norm[i]:>5.2f} {decay:>10}")

# Weak patterns: bottom 25% + (marginal_sharpe < 0 OR decaying OR L1O improves)
threshold_25 = np.percentile(list(phase6['scores'].values()), 25)
for pat, score in scores_sorted:
    if score <= threshold_25:
        reasons = []
        if phase4['marginal_sharpe'][pat] < 0:
            reasons.append("neg marginal Sharpe")
        if phase5['edge_decay'][pat]['status'] == 'Decaying':
            reasons.append("edge decaying")
        l1o = phase4['l1o'][pat]
        if l1o['pnl_mdd_without'] > portfolio_stats['pnl_mdd']:
            reasons.append("L1O improves PnL/MDD")
        if reasons:
            phase6['weak_patterns'].append({'pattern': pat, 'score': score, 'reasons': reasons})

# Strong patterns: top 25% + stable/improving + low correlation
threshold_75 = np.percentile(list(phase6['scores'].values()), 75)
for pat, score in scores_sorted:
    if score >= threshold_75:
        decay = phase5['edge_decay'][pat]['status']
        if decay in ('Stable', 'Improving'):
            phase6['strong_patterns'].append({'pattern': pat, 'score': score, 'status': decay})

print(f"\n  Strong patterns (top 25% + stable/improving): {len(phase6['strong_patterns'])}")
for sp in phase6['strong_patterns']:
    print(f"    {sp['pattern']:<14} score={sp['score']:.3f} ({sp['status']})")

print(f"\n  Weak patterns (bottom 25% + issues): {len(phase6['weak_patterns'])}")
for wp in phase6['weak_patterns']:
    print(f"    {wp['pattern']:<14} score={wp['score']:.3f} — {', '.join(wp['reasons'])}")

# Portfolio suggestions
suggestions = []
if phase4['long_short_balance']['long_pct'] > 65:
    suggestions.append(f"LONG bias ({phase4['long_short_balance']['long_pct']:.0f}%): consider adding SHORT patterns")
elif phase4['long_short_balance']['long_pct'] < 35:
    suggestions.append(f"SHORT bias ({100 - phase4['long_short_balance']['long_pct']:.0f}%): consider adding LONG patterns")

if hhi > 2 / N_PAT:
    suggestions.append(f"Concentration risk (HHI={hhi:.4f} > {2/N_PAT:.4f}): profits concentrated in few patterns")

if decay_counts['Decaying'] > N_PAT * 0.25:
    suggestions.append(f"{decay_counts['Decaying']} decaying patterns ({decay_counts['Decaying']/N_PAT*100:.0f}%): monitor for edge loss")

if len(phase6['weak_patterns']) > 0:
    removable = [wp['pattern'] for wp in phase6['weak_patterns']
                 if any('L1O improves' in r for r in wp['reasons'])]
    if removable:
        suggestions.append(f"Removable patterns (L1O improves portfolio): {', '.join(removable)}")

phase6['portfolio_suggestions'] = suggestions

print(f"\n  Portfolio suggestions:")
for s in suggestions:
    print(f"    - {s}")
if not suggestions:
    print(f"    (none — portfolio is well-balanced)")

# Final summary
elapsed = time.time() - t0_total
print(f"\n  Total elapsed: {elapsed:.1f}s")
print(f"\n  Portfolio summary:")
print(f"    PnL: {portfolio_stats['pnl']:+.1f}%, Trades: {portfolio_stats['trades']}, "
      f"WR: {portfolio_stats['wr']:.1f}%, MDD: {portfolio_stats['mdd']:.1f}%, "
      f"PF: {portfolio_stats['pf']:.2f}, PnL/MDD: {portfolio_stats['pnl_mdd']:.1f}x")


# ============================================================
# Save results
# ============================================================
results = {
    'meta': {
        'script': 'pattern_deep_analysis.py',
        'version': BOT_VERSION,
        'timestamp': datetime.now().isoformat(),
        'patterns': N_PAT,
        'elapsed_seconds': round(elapsed, 1),
    },
    'portfolio_summary': {
        'pnl': round(portfolio_stats['pnl'], 1),
        'trades': portfolio_stats['trades'],
        'wr': round(portfolio_stats['wr'], 1),
        'mdd': round(portfolio_stats['mdd'], 1),
        'pf': round(portfolio_stats['pf'], 2),
        'pnl_mdd': round(portfolio_stats['pnl_mdd'], 1),
    },
    'phase1_profiling': phase1,
    'phase2_correlation': phase2,
    'phase3_regime': phase3,
    'phase4_portfolio': phase4,
    'phase5_robustness': phase5,
    'phase6_summary': phase6,
}

# Convert non-serializable types
def make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    return obj

def deep_convert(obj):
    if isinstance(obj, dict):
        return {k: deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_convert(v) for v in obj]
    return make_serializable(obj)

results = deep_convert(results)

out_path = Path(__file__).resolve().parent / '../../results/pattern_deep_analysis.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved: {out_path}")
