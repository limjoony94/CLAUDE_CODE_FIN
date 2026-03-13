#!/usr/bin/env python3
"""
Live WR Gap Root Cause Decomposition
=====================================
Decomposes the gap between IS WR (77.8%) and Live WR (~57%) across
multiple dimensions to identify root causes.

Dimensions:
  1. Time (weekly/daily WR+PnL trends)
  2. Pattern (per-pattern WR/PnL/count)
  3. Direction (LONG vs SHORT)
  4. Exit reason (TP/SL/TIMEOUT/CASCADE_SL/MARKET/etc)
  5. Session (Asia/Europe/US by UTC hour)
  6. Consecutive losses (max streaks, patterns during streaks)
  7. Cascade SL chain analysis
  8. Early vs Late (first half vs second half)

Output: results/live_wr_gap_decomposition.json + console summary
"""

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from statistics import mean, median, stdev

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
METRICS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'pattern_5m_metrics.json')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'live_wr_gap_decomposition.json')
IS_WR = 77.8  # v1.57.0 IS WR with TP x0.5
LEVERAGE = 3
FEE_PCT = 0.10  # percent, both sides

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_ts(s):
    """Parse ISO timestamp string to datetime."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def is_win(trade):
    """A trade is a win if pnl_slot > 0 (slot-level PnL)."""
    return trade.get('pnl_slot', 0) > 0


def safe_wr(wins, total):
    if total == 0:
        return None
    return round(100.0 * wins / total, 2)


def safe_mean(vals):
    if not vals:
        return None
    return round(mean(vals), 4)


def safe_median(vals):
    if not vals:
        return None
    return round(median(vals), 4)


def safe_stdev(vals):
    if len(vals) < 2:
        return None
    return round(stdev(vals), 4)


def week_key(dt):
    """ISO year-week string."""
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def day_key(dt):
    return dt.strftime("%Y-%m-%d")


def session_label(hour_utc):
    """Classify UTC hour into trading session."""
    if 0 <= hour_utc < 8:
        return "Asia"
    elif 8 <= hour_utc < 16:
        return "Europe"
    else:
        return "US"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_trades():
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
    raw = metrics.get('trade_history', [])
    trades = []
    for t in raw:
        pat = t.get('pattern', 'N/A')
        if pat in (None, 'None', '', 'null'):
            pat = 'N/A'
        ts = parse_ts(t.get('timestamp'))
        trades.append({
            'pattern': pat,
            'direction': t.get('direction', '?'),
            'entry_price': t.get('entry_price', 0),
            'exit_price': t.get('exit_price', 0),
            'pnl_slot': t.get('pnl_slot', 0),
            'pnl_portfolio': t.get('pnl_portfolio', 0),
            'tp_price': t.get('tp_price'),
            'sl_price': t.get('sl_price'),
            'exit_reason': t.get('exit_reason', 'UNKNOWN'),
            'hold_minutes': t.get('hold_minutes', 0),
            'timestamp': ts,
            'win': is_win(t),
        })
    # Sort by timestamp
    trades.sort(key=lambda x: x['timestamp'] or datetime.min)
    return trades


# ---------------------------------------------------------------------------
# Dimension 1: Time decomposition (weekly + daily)
# ---------------------------------------------------------------------------

def decompose_time(trades):
    weekly = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl': []})
    daily = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl': []})

    for t in trades:
        if not t['timestamp']:
            continue
        wk = week_key(t['timestamp'])
        dy = day_key(t['timestamp'])
        weekly[wk]['total'] += 1
        daily[dy]['total'] += 1
        weekly[wk]['pnl'].append(t['pnl_portfolio'])
        daily[dy]['pnl'].append(t['pnl_portfolio'])
        if t['win']:
            weekly[wk]['wins'] += 1
            daily[dy]['wins'] += 1

    weekly_out = []
    for wk in sorted(weekly.keys()):
        d = weekly[wk]
        weekly_out.append({
            'week': wk,
            'trades': d['total'],
            'wins': d['wins'],
            'wr': safe_wr(d['wins'], d['total']),
            'total_pnl': round(sum(d['pnl']), 4),
            'avg_pnl': safe_mean(d['pnl']),
        })

    daily_out = []
    for dy in sorted(daily.keys()):
        d = daily[dy]
        daily_out.append({
            'date': dy,
            'trades': d['total'],
            'wins': d['wins'],
            'wr': safe_wr(d['wins'], d['total']),
            'total_pnl': round(sum(d['pnl']), 4),
        })

    return {'weekly': weekly_out, 'daily': daily_out}


# ---------------------------------------------------------------------------
# Dimension 2: Pattern decomposition
# ---------------------------------------------------------------------------

def decompose_pattern(trades):
    pat_data = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl_slot': [], 'pnl_port': [], 'directions': []})
    for t in trades:
        p = t['pattern']
        pat_data[p]['total'] += 1
        pat_data[p]['pnl_slot'].append(t['pnl_slot'])
        pat_data[p]['pnl_port'].append(t['pnl_portfolio'])
        pat_data[p]['directions'].append(t['direction'])
        if t['win']:
            pat_data[p]['wins'] += 1

    results = []
    for p in sorted(pat_data.keys(), key=lambda x: pat_data[x]['total'], reverse=True):
        d = pat_data[p]
        results.append({
            'pattern': p,
            'trades': d['total'],
            'wins': d['wins'],
            'wr': safe_wr(d['wins'], d['total']),
            'wr_gap_vs_is': round(safe_wr(d['wins'], d['total']) - IS_WR, 2) if d['total'] > 0 else None,
            'total_pnl_port': round(sum(d['pnl_port']), 4),
            'avg_pnl_slot': safe_mean(d['pnl_slot']),
            'long_count': d['directions'].count('LONG'),
            'short_count': d['directions'].count('SHORT'),
        })

    # Top losers by total portfolio PnL
    worst = sorted(results, key=lambda x: x['total_pnl_port'])[:10]
    best = sorted(results, key=lambda x: x['total_pnl_port'], reverse=True)[:10]

    return {
        'all_patterns': results,
        'worst_10_by_pnl': worst,
        'best_10_by_pnl': best,
        'unique_patterns': len(results),
        'patterns_below_50wr': len([r for r in results if r['wr'] is not None and r['wr'] < 50]),
        'patterns_above_is_wr': len([r for r in results if r['wr'] is not None and r['wr'] >= IS_WR]),
        'na_pattern_trades': sum(1 for r in results if r['pattern'] == 'N/A'),
    }


# ---------------------------------------------------------------------------
# Dimension 3: Direction decomposition
# ---------------------------------------------------------------------------

def decompose_direction(trades):
    dirs = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl_port': [], 'pnl_slot': []})
    for t in trades:
        d = t['direction']
        dirs[d]['total'] += 1
        dirs[d]['pnl_port'].append(t['pnl_portfolio'])
        dirs[d]['pnl_slot'].append(t['pnl_slot'])
        if t['win']:
            dirs[d]['wins'] += 1

    result = {}
    for d in sorted(dirs.keys()):
        dd = dirs[d]
        result[d] = {
            'trades': dd['total'],
            'wins': dd['wins'],
            'wr': safe_wr(dd['wins'], dd['total']),
            'wr_gap_vs_is': round(safe_wr(dd['wins'], dd['total']) - IS_WR, 2),
            'total_pnl_port': round(sum(dd['pnl_port']), 4),
            'avg_pnl_slot': safe_mean(dd['pnl_slot']),
            'avg_pnl_port': safe_mean(dd['pnl_port']),
        }
    return result


# ---------------------------------------------------------------------------
# Dimension 4: Exit reason decomposition
# ---------------------------------------------------------------------------

def decompose_exit_reason(trades):
    reasons = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl_port': [], 'pnl_slot': [], 'hold_min': []})
    for t in trades:
        r = t['exit_reason']
        reasons[r]['total'] += 1
        reasons[r]['pnl_port'].append(t['pnl_portfolio'])
        reasons[r]['pnl_slot'].append(t['pnl_slot'])
        reasons[r]['hold_min'].append(t['hold_minutes'])
        if t['win']:
            reasons[r]['wins'] += 1

    result = {}
    for r in sorted(reasons.keys()):
        d = reasons[r]
        result[r] = {
            'trades': d['total'],
            'wins': d['wins'],
            'wr': safe_wr(d['wins'], d['total']),
            'pct_of_total': round(100.0 * d['total'] / len(trades), 2),
            'total_pnl_port': round(sum(d['pnl_port']), 4),
            'avg_pnl_slot': safe_mean(d['pnl_slot']),
            'avg_hold_min': safe_mean(d['hold_min']),
            'median_hold_min': safe_median(d['hold_min']),
        }
    return result


# ---------------------------------------------------------------------------
# Dimension 5: Session (Asia/Europe/US) decomposition
# ---------------------------------------------------------------------------

def decompose_session(trades):
    sessions = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl_port': []})
    hourly = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl_port': []})

    for t in trades:
        if not t['timestamp']:
            continue
        h = t['timestamp'].hour
        sess = session_label(h)
        sessions[sess]['total'] += 1
        sessions[sess]['pnl_port'].append(t['pnl_portfolio'])
        hourly[h]['total'] += 1
        hourly[h]['pnl_port'].append(t['pnl_portfolio'])
        if t['win']:
            sessions[sess]['wins'] += 1
            hourly[h]['wins'] += 1

    sess_out = {}
    for s in ['Asia', 'Europe', 'US']:
        d = sessions[s]
        sess_out[s] = {
            'trades': d['total'],
            'wins': d['wins'],
            'wr': safe_wr(d['wins'], d['total']),
            'total_pnl_port': round(sum(d['pnl_port']), 4),
        }

    hourly_out = {}
    for h in range(24):
        d = hourly[h]
        if d['total'] > 0:
            hourly_out[str(h)] = {
                'trades': d['total'],
                'wins': d['wins'],
                'wr': safe_wr(d['wins'], d['total']),
                'total_pnl_port': round(sum(d['pnl_port']), 4),
            }

    return {'sessions': sess_out, 'hourly': hourly_out}


# ---------------------------------------------------------------------------
# Dimension 6: Consecutive losses analysis
# ---------------------------------------------------------------------------

def decompose_consecutive_losses(trades):
    # Find all loss streaks
    streaks = []
    current_streak = []
    for t in trades:
        if not t['win']:
            current_streak.append(t)
        else:
            if len(current_streak) >= 2:
                streaks.append(current_streak)
            current_streak = []
    if len(current_streak) >= 2:
        streaks.append(current_streak)

    if not streaks:
        return {'max_streak': 0, 'streaks': []}

    max_streak = max(streaks, key=len)
    streak_summaries = []
    for s in sorted(streaks, key=len, reverse=True)[:10]:
        pats = [t['pattern'] for t in s]
        dirs = [t['direction'] for t in s]
        reasons = [t['exit_reason'] for t in s]
        total_pnl = sum(t['pnl_portfolio'] for t in s)
        ts_start = s[0]['timestamp'].isoformat() if s[0]['timestamp'] else '?'
        ts_end = s[-1]['timestamp'].isoformat() if s[-1]['timestamp'] else '?'
        streak_summaries.append({
            'length': len(s),
            'start': ts_start,
            'end': ts_end,
            'total_pnl_port': round(total_pnl, 4),
            'patterns': pats,
            'directions': Counter(dirs),
            'exit_reasons': Counter(reasons),
        })

    return {
        'max_streak_length': len(max_streak),
        'total_loss_streaks_ge2': len(streaks),
        'avg_streak_length': round(mean(len(s) for s in streaks), 2),
        'top_10_streaks': streak_summaries,
    }


# ---------------------------------------------------------------------------
# Dimension 7: Cascade SL chain analysis
# ---------------------------------------------------------------------------

def decompose_cascade(trades):
    """Analyze CASCADE_SL trades and their knock-on effects."""
    cascade_trades = [t for t in trades if t['exit_reason'] == 'CASCADE_SL']
    cascade_count = len(cascade_trades)

    if cascade_count == 0:
        return {'cascade_trades': 0, 'chains': []}

    # Look for cascade chains: cascade SL followed by more SLs in short succession
    chains = []
    for i, t in enumerate(trades):
        if t['exit_reason'] != 'CASCADE_SL':
            continue
        chain = [t]
        # Look at next trades within 30 min
        for j in range(i + 1, min(i + 10, len(trades))):
            nxt = trades[j]
            if not t['timestamp'] or not nxt['timestamp']:
                break
            gap_min = (nxt['timestamp'] - t['timestamp']).total_seconds() / 60
            if gap_min <= 30 and nxt['exit_reason'] in ('SL', 'CASCADE_SL'):
                chain.append(nxt)
            elif gap_min > 30:
                break
        if len(chain) > 1:
            chains.append(chain)

    chain_summaries = []
    for ch in chains:
        chain_summaries.append({
            'length': len(ch),
            'total_pnl_port': round(sum(t['pnl_portfolio'] for t in ch), 4),
            'reasons': [t['exit_reason'] for t in ch],
            'directions': [t['direction'] for t in ch],
            'start': ch[0]['timestamp'].isoformat() if ch[0]['timestamp'] else '?',
        })

    # Also look at SL trades near cascade events
    sl_near_cascade = 0
    for i, t in enumerate(trades):
        if t['exit_reason'] != 'CASCADE_SL':
            continue
        # Look +-5 trades
        for j in range(max(0, i - 5), min(len(trades), i + 6)):
            if j == i:
                continue
            if trades[j]['exit_reason'] in ('SL',) and trades[j]['timestamp'] and t['timestamp']:
                gap = abs((trades[j]['timestamp'] - t['timestamp']).total_seconds() / 60)
                if gap < 60:
                    sl_near_cascade += 1

    return {
        'cascade_trades': cascade_count,
        'cascade_total_pnl': round(sum(t['pnl_portfolio'] for t in cascade_trades), 4),
        'cascade_avg_pnl_slot': safe_mean([t['pnl_slot'] for t in cascade_trades]),
        'sl_near_cascade_60min': sl_near_cascade,
        'cascade_chains': chain_summaries,
    }


# ---------------------------------------------------------------------------
# Dimension 8: Early vs Late comparison
# ---------------------------------------------------------------------------

def decompose_early_late(trades):
    mid = len(trades) // 2
    first_half = trades[:mid]
    second_half = trades[mid:]

    def half_stats(half, label):
        wins = sum(1 for t in half if t['win'])
        total = len(half)
        pnl = [t['pnl_portfolio'] for t in half]
        slot_pnl = [t['pnl_slot'] for t in half]
        reasons = Counter(t['exit_reason'] for t in half)
        dirs = Counter(t['direction'] for t in half)
        ts_start = half[0]['timestamp'].isoformat() if half and half[0]['timestamp'] else '?'
        ts_end = half[-1]['timestamp'].isoformat() if half and half[-1]['timestamp'] else '?'
        return {
            'label': label,
            'trades': total,
            'wins': wins,
            'wr': safe_wr(wins, total),
            'total_pnl_port': round(sum(pnl), 4),
            'avg_pnl_port': safe_mean(pnl),
            'avg_pnl_slot': safe_mean(slot_pnl),
            'exit_reasons': dict(reasons),
            'directions': dict(dirs),
            'period': f"{ts_start} to {ts_end}",
        }

    first = half_stats(first_half, f"First {mid} trades")
    second = half_stats(second_half, f"Last {len(second_half)} trades")

    return {
        'first_half': first,
        'second_half': second,
        'wr_change': round((second['wr'] or 0) - (first['wr'] or 0), 2),
    }


# ---------------------------------------------------------------------------
# Summary: WR gap attribution
# ---------------------------------------------------------------------------

def compute_gap_attribution(trades, direction_data, exit_data, session_data):
    """Estimate how much each factor contributes to the WR gap."""
    total = len(trades)
    overall_wr = safe_wr(sum(1 for t in trades if t['win']), total)
    gap = round(overall_wr - IS_WR, 2)

    attribution = {}

    # Direction contribution
    for d in ['LONG', 'SHORT']:
        if d in direction_data:
            dd = direction_data[d]
            weight = dd['trades'] / total
            contrib = weight * (dd['wr'] - IS_WR)
            attribution[f'direction_{d}'] = {
                'weight': round(weight, 4),
                'wr': dd['wr'],
                'gap_contribution_pp': round(contrib, 2),
            }

    # Exit reason contribution (non-TP exits = likely losses)
    non_tp_exits = {k: v for k, v in exit_data.items() if k != 'TP'}
    for reason, d in non_tp_exits.items():
        weight = d['trades'] / total
        # These trades are mostly losses, contributing to gap
        loss_rate = 100.0 - (d['wr'] or 0)
        attribution[f'exit_{reason}'] = {
            'weight': round(weight, 4),
            'wr': d['wr'],
            'loss_rate': round(loss_rate, 2),
            'trades': d['trades'],
            'total_pnl': d['total_pnl_port'],
        }

    # TIMEOUT specifically
    if 'TIMEOUT' in exit_data:
        to = exit_data['TIMEOUT']
        attribution['timeout_impact'] = {
            'trades': to['trades'],
            'pct_of_total': to['pct_of_total'],
            'wr': to['wr'],
            'note': 'IS drops timeout trades entirely; live keeps them and mostly loses',
        }

    # N/A pattern
    na_trades = [t for t in trades if t['pattern'] == 'N/A']
    if na_trades:
        na_wins = sum(1 for t in na_trades if t['win'])
        attribution['na_pattern'] = {
            'trades': len(na_trades),
            'wr': safe_wr(na_wins, len(na_trades)),
            'total_pnl_port': round(sum(t['pnl_portfolio'] for t in na_trades), 4),
            'note': 'N/A patterns from crash recovery — no IS equivalent',
        }

    return {
        'overall_live_wr': overall_wr,
        'is_wr': IS_WR,
        'gap_pp': gap,
        'total_trades': total,
        'attribution': attribution,
    }


# ---------------------------------------------------------------------------
# Rolling WR
# ---------------------------------------------------------------------------

def compute_rolling_wr(trades, window=20):
    """Compute rolling WR over last N trades."""
    results = []
    for i in range(window, len(trades) + 1):
        chunk = trades[i - window:i]
        wins = sum(1 for t in chunk if t['win'])
        results.append({
            'trade_idx': i,
            'wr': safe_wr(wins, window),
            'timestamp': chunk[-1]['timestamp'].isoformat() if chunk[-1]['timestamp'] else '?',
        })
    # Find worst and best rolling windows
    if results:
        worst = min(results, key=lambda x: x['wr'] or 999)
        best = max(results, key=lambda x: x['wr'] or 0)
        return {
            'window_size': window,
            'worst': worst,
            'best': best,
            'final': results[-1],
            'series': results,
        }
    return {}


# ---------------------------------------------------------------------------
# Hold time analysis
# ---------------------------------------------------------------------------

def decompose_hold_time(trades):
    """Analyze hold time vs outcome."""
    win_hold = [t['hold_minutes'] for t in trades if t['win'] and t['hold_minutes']]
    loss_hold = [t['hold_minutes'] for t in trades if not t['win'] and t['hold_minutes']]

    # Bucket by hold time
    buckets = [
        ('0-60min', 0, 60),
        ('1-4h', 60, 240),
        ('4-12h', 240, 720),
        ('12-24h', 720, 1440),
        ('24h+', 1440, 999999),
    ]
    bucket_stats = []
    for label, lo, hi in buckets:
        b_trades = [t for t in trades if lo <= (t['hold_minutes'] or 0) < hi]
        wins = sum(1 for t in b_trades if t['win'])
        bucket_stats.append({
            'bucket': label,
            'trades': len(b_trades),
            'wins': wins,
            'wr': safe_wr(wins, len(b_trades)),
            'avg_pnl_slot': safe_mean([t['pnl_slot'] for t in b_trades]),
        })

    return {
        'win_avg_hold_min': safe_mean(win_hold),
        'win_median_hold_min': safe_median(win_hold),
        'loss_avg_hold_min': safe_mean(loss_hold),
        'loss_median_hold_min': safe_median(loss_hold),
        'buckets': bucket_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    trades = load_trades()
    print(f"Loaded {len(trades)} trades from {METRICS_PATH}")
    print(f"Period: {trades[0]['timestamp']} to {trades[-1]['timestamp']}")
    print(f"IS WR benchmark: {IS_WR}%")
    print()

    # Overall stats
    total = len(trades)
    wins = sum(1 for t in trades if t['win'])
    live_wr = safe_wr(wins, total)
    total_pnl = round(sum(t['pnl_portfolio'] for t in trades), 4)
    print(f"=== OVERALL ===")
    print(f"Trades: {total}, Wins: {wins}, WR: {live_wr}%")
    print(f"Gap vs IS: {round(live_wr - IS_WR, 2)}pp")
    print(f"Total portfolio PnL: {total_pnl}%")
    print()

    # Run all dimensions
    time_data = decompose_time(trades)
    pattern_data = decompose_pattern(trades)
    direction_data = decompose_direction(trades)
    exit_data = decompose_exit_reason(trades)
    session_data = decompose_session(trades)
    consec_data = decompose_consecutive_losses(trades)
    cascade_data = decompose_cascade(trades)
    early_late_data = decompose_early_late(trades)
    rolling_data = compute_rolling_wr(trades, window=20)
    hold_data = decompose_hold_time(trades)
    attribution = compute_gap_attribution(trades, direction_data, exit_data, session_data)

    # Print summaries
    print("=== DIRECTION ===")
    for d, v in direction_data.items():
        print(f"  {d}: {v['trades']}t, WR {v['wr']}%, gap {v['wr_gap_vs_is']}pp, PnL {v['total_pnl_port']}%")

    print("\n=== EXIT REASONS ===")
    for r, v in sorted(exit_data.items(), key=lambda x: -x[1]['trades']):
        print(f"  {r}: {v['trades']}t ({v['pct_of_total']}%), WR {v['wr']}%, PnL {v['total_pnl_port']}%, avg hold {v['avg_hold_min']}min")

    print("\n=== SESSIONS (UTC) ===")
    for s, v in session_data['sessions'].items():
        print(f"  {s}: {v['trades']}t, WR {v['wr']}%, PnL {v['total_pnl_port']}%")

    print("\n=== CONSECUTIVE LOSSES ===")
    print(f"  Max streak: {consec_data['max_streak_length']}")
    print(f"  Total streaks (>=2): {consec_data['total_loss_streaks_ge2']}")
    if consec_data.get('top_10_streaks'):
        worst = consec_data['top_10_streaks'][0]
        print(f"  Worst streak: {worst['length']} losses, PnL {worst['total_pnl_port']}%")
        print(f"    Reasons: {dict(worst['exit_reasons'])}")
        print(f"    Directions: {dict(worst['directions'])}")

    print("\n=== CASCADE SL ===")
    print(f"  Cascade trades: {cascade_data['cascade_trades']}")
    print(f"  Cascade total PnL: {cascade_data.get('cascade_total_pnl', 0)}%")
    print(f"  SL within 60min of cascade: {cascade_data.get('sl_near_cascade_60min', 0)}")

    print("\n=== EARLY vs LATE ===")
    f = early_late_data['first_half']
    s = early_late_data['second_half']
    print(f"  First half:  {f['trades']}t, WR {f['wr']}%, PnL {f['total_pnl_port']}%")
    print(f"  Second half: {s['trades']}t, WR {s['wr']}%, PnL {s['total_pnl_port']}%")
    print(f"  WR change: {early_late_data['wr_change']}pp")

    print("\n=== HOLD TIME ===")
    print(f"  Win avg hold: {hold_data['win_avg_hold_min']}min, median: {hold_data['win_median_hold_min']}min")
    print(f"  Loss avg hold: {hold_data['loss_avg_hold_min']}min, median: {hold_data['loss_median_hold_min']}min")
    for b in hold_data['buckets']:
        print(f"  {b['bucket']}: {b['trades']}t, WR {b['wr']}%, avg slot PnL {b['avg_pnl_slot']}%")

    print("\n=== ROLLING WR (20-trade window) ===")
    if rolling_data:
        print(f"  Worst window: WR {rolling_data['worst']['wr']}% at trade #{rolling_data['worst']['trade_idx']} ({rolling_data['worst']['timestamp']})")
        print(f"  Best window:  WR {rolling_data['best']['wr']}% at trade #{rolling_data['best']['trade_idx']} ({rolling_data['best']['timestamp']})")
        print(f"  Current:      WR {rolling_data['final']['wr']}%")

    print("\n=== WORST 10 PATTERNS (by portfolio PnL) ===")
    for p in pattern_data['worst_10_by_pnl']:
        print(f"  {p['pattern']}: {p['trades']}t, WR {p['wr']}%, PnL {p['total_pnl_port']}%, gap {p['wr_gap_vs_is']}pp")

    print("\n=== WEEKLY TREND ===")
    for w in time_data['weekly']:
        marker = " ***" if (w['wr'] or 0) < 40 else ""
        print(f"  {w['week']}: {w['trades']}t, WR {w['wr']}%, PnL {w['total_pnl']}%{marker}")

    print("\n=== GAP ATTRIBUTION ===")
    print(f"  Overall gap: {attribution['gap_pp']}pp (Live {attribution['overall_live_wr']}% vs IS {IS_WR}%)")
    for k, v in attribution['attribution'].items():
        if 'note' in v:
            print(f"  {k}: {v}")

    # N/A analysis
    print(f"\n=== N/A PATTERNS ===")
    print(f"  N/A trades: {pattern_data['na_pattern_trades']}")
    print(f"  Patterns below 50% WR: {pattern_data['patterns_below_50wr']}")
    print(f"  Patterns >= IS WR ({IS_WR}%): {pattern_data['patterns_above_is_wr']}")

    # Assemble output
    output = {
        'metadata': {
            'script': 'live_wr_gap_decomposition.py',
            'date': datetime.now().isoformat(),
            'total_trades': total,
            'period': f"{trades[0]['timestamp']} to {trades[-1]['timestamp']}",
            'is_wr': IS_WR,
            'live_wr': live_wr,
            'gap_pp': round(live_wr - IS_WR, 2),
        },
        'gap_attribution': attribution,
        'direction': direction_data,
        'exit_reason': exit_data,
        'session': session_data,
        'time': time_data,
        'pattern': pattern_data,
        'consecutive_losses': consec_data,
        'cascade_sl': cascade_data,
        'early_vs_late': early_late_data,
        'rolling_wr_20': {k: v for k, v in rolling_data.items() if k != 'series'},
        'rolling_wr_20_series': rolling_data.get('series', []),
        'hold_time': hold_data,
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
