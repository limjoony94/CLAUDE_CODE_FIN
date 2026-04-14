#!/usr/bin/env python3
"""
Extended Data Pattern Rediscovery
=================================
핵심 가설: E/F 옵션이 실패한 이유가 훈련 기간 부족(54~216일)이라면,
360~720일 훈련 기간에서는 패턴 발굴의 genuine forward edge가 유의미할 수 있다.

검증 방법:
1. Binance에서 BTC/USDT 5m 데이터 720일 다운로드
2. 현재 270일 데이터와 겹치는 구간 검증 (동일성 확인)
3. Expanding-window WF: 훈련 기간 360/450/540일, OOS 90일
4. 각 fold에서 전체 유니버스(3456) 패턴 발굴 → OOS 테스트
5. 훈련 기간별 genuine forward edge 비교

WF 구조 (720일 데이터):
- Fold 1: Train 360d → OOS 90d (day 361-450)
- Fold 2: Train 450d → OOS 90d (day 451-540)
- Fold 3: Train 540d → OOS 90d (day 541-630)
- Fold 4: Train 630d → OOS 90d (day 631-720)

비교 대상:
- 기존 E/F (54~216일 훈련) vs 확장 (360~630일 훈련)
- 훈련 기간 vs OOS edge 상관관계
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import ccxt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
from scripts.production.pattern_5m.indicators import classify_candle as _prod_classify
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── Parameters ────────────────────────────────────────────────
TARGET_DAYS = 720
OOS_DAYS = 90
MIN_TRAIN_DAYS = 360
FEE_PCT = 0.10
MAX_BARS = 500
MC_SIMS = 5000
MC_THRESHOLD = 0.01
EDGE_THRESHOLD = 5.0  # pp
TP_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
MIN_SL = 0.5  # v1.27.2: SL < 0.5% is execution-infeasible

CANDLE_TYPES = ['D', 'DF', 'GS', 'H', 'IH', 'ST', 'MU', 'MD', 'BU', 'BD', 'U', 'DN']


# ── Data download ─────────────────────────────────────────────
def download_binance_data(days: int) -> pd.DataFrame:
    """Download BTC/USDT 5m data from Binance."""
    print(f"\n  Downloading {days} days of BTC/USDT 5m from Binance...")

    ex = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })

    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)

    all_candles = []
    since = start_ts
    batch = 0

    while since < end_ts:
        batch += 1
        try:
            ohlcv = ex.fetch_ohlcv('BTC/USDT:USDT', '5m', since=since, limit=1000)
            if not ohlcv:
                break
            all_candles.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if batch % 50 == 0:
                print(f"    Batch {batch}: {len(all_candles)} candles, "
                      f"up to {ex.iso8601(ohlcv[-1][0])}")
            time.sleep(ex.rateLimit / 1000)
        except Exception as e:
            print(f"    Error at batch {batch}: {e}")
            time.sleep(5)

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    actual_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    print(f"  Downloaded: {len(df)} bars, {actual_days:.0f} days")
    print(f"  Range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    return df


# ── Candle classification — delegates to production canonical ─
def classify_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Classify candles using production classify_candle (canonical source)."""
    print("  Classifying candles...")
    df = df.copy()
    df['body_abs'] = (df['close'] - df['open']).abs()
    df['avg_body_20'] = df['body_abs'].rolling(AVG_BODY_WINDOW).mean()

    codes = []
    for i, row in df.iterrows():
        avg_b = df.at[i, 'avg_body_20']
        if pd.isna(avg_b):
            avg_b = 1.0
        codes.append(_prod_classify(row, avg_b).value)

    df['type_code'] = codes
    df['pattern_3'] = (
        df['type_code'].shift(2).astype(str) + '-' +
        df['type_code'].shift(1).astype(str) + '-' +
        df['type_code'].astype(str)
    )
    df.loc[:1, 'pattern_3'] = None

    print(f"  Classification complete: {len(df)} bars")
    return df


# ── Pattern discovery engine ─────────────────────────────────
def collect_pattern_instances(df: pd.DataFrame):
    """Collect all pattern instances with precomputed price moves."""
    instances = defaultdict(list)  # pattern -> list of (entry_idx, direction)

    for i in range(2, len(df) - 1):
        pat = df['pattern_3'].iloc[i]
        if pd.isna(pat):
            continue
        entry_idx = i + 1  # next bar open
        if entry_idx >= len(df):
            continue
        entry_price = df['open'].iloc[entry_idx]
        if entry_price <= 0:
            continue

        # Precompute max favorable/adverse moves for each future bar
        max_bars = min(MAX_BARS, len(df) - entry_idx)
        fav_long = np.zeros(max_bars)
        adv_long = np.zeros(max_bars)

        for j in range(max_bars):
            bar_idx = entry_idx + j
            h = df['high'].iloc[bar_idx]
            l = df['low'].iloc[bar_idx]
            fav_long[j] = (h - entry_price) / entry_price * 100
            adv_long[j] = (entry_price - l) / entry_price * 100

        for direction in ['LONG', 'SHORT']:
            if direction == 'LONG':
                fav = fav_long
                adv = adv_long
            else:
                fav = adv_long
                adv = fav_long

            key = f"{pat}_{direction}"
            instances[key].append({
                'entry_idx': entry_idx,
                'fav': fav,
                'adv': adv,
            })

    return instances


def evaluate_tpsl(instances: list, tp: float, sl: float) -> dict:
    """Evaluate a TP/SL combination on precomputed instances."""
    pnls = []
    for inst in instances:
        fav = inst['fav']
        adv = inst['adv']
        settled = False
        for j in range(len(fav)):
            tp_hit = fav[j] >= tp
            sl_hit = adv[j] >= sl
            if tp_hit and sl_hit:
                tp_dist = abs(fav[j] - tp)
                sl_dist = abs(adv[j] - sl)
                if tp_dist <= sl_dist:
                    pnls.append(tp - FEE_PCT)
                else:
                    pnls.append(-sl - FEE_PCT)
                settled = True
                break
            elif tp_hit:
                pnls.append(tp - FEE_PCT)
                settled = True
                break
            elif sl_hit:
                pnls.append(-sl - FEE_PCT)
                settled = True
                break
        if not settled:
            # Timeout at last bar
            pnls.append(fav[-1] - FEE_PCT if len(fav) > 0 else -FEE_PCT)

    if not pnls:
        return None

    pnls = np.array(pnls)
    wins = pnls[pnls > 0]
    wr = len(wins) / len(pnls) * 100

    # Random walk baseline
    rw_wr = sl / (tp + sl) * 100
    edge = wr - rw_wr

    return {
        'tp': tp, 'sl': sl,
        'pnl': float(np.sum(pnls)),
        'trades': len(pnls),
        'wr': round(wr, 1),
        'rw_wr': round(rw_wr, 1),
        'edge': round(edge, 1),
        'pnls': pnls,
    }


def mc_test_pnls(pnls: np.ndarray, n_sims: int = MC_SIMS) -> float:
    """MC sign randomization test, return p-value."""
    if len(pnls) < 10:
        return 1.0
    actual = np.sum(pnls)
    abs_pnls = np.abs(pnls)
    rng = np.random.default_rng(42)
    count = sum(1 for _ in range(n_sims)
                if np.sum(rng.choice([-1, 1], len(pnls)) * abs_pnls) >= actual)
    return count / n_sims


def discover_patterns(instances: dict, min_trades: int = 15) -> list:
    """Discover patterns from universe via grid search + MC + edge filter."""
    discovered = []

    for key, insts in instances.items():
        if len(insts) < min_trades:
            continue

        pattern, direction = key.rsplit('_', 1)

        best = None
        best_score = -999

        for tp in TP_GRID:
            for sl in SL_GRID:
                if sl < MIN_SL:
                    continue

                result = evaluate_tpsl(insts, tp, sl)
                if result is None or result['trades'] < min_trades:
                    continue
                if result['edge'] < EDGE_THRESHOLD:
                    continue
                if result['pnl'] <= 0:
                    continue

                # Score: PnL * edge (prioritize both)
                score = result['pnl'] * result['edge']
                if score > best_score:
                    best_score = score
                    best = result
                    best['pattern'] = pattern
                    best['direction'] = direction

        if best is not None:
            # MC test on best TP/SL
            p_val = mc_test_pnls(best['pnls'])
            if p_val < MC_THRESHOLD:
                best['mc_p'] = p_val
                del best['pnls']  # save memory
                discovered.append(best)

    return discovered


# ── Portfolio backtest ────────────────────────────────────────
def backtest_portfolio_1pos(df: pd.DataFrame, patterns: dict, tpsl: dict) -> list:
    """1-position-at-a-time portfolio backtest."""
    trades = []
    in_position = False
    entry_bar = entry_price = tp_pct = sl_pct = None
    direction = pattern_name = None

    for i in range(2, len(df)):
        if in_position:
            bars_held = i - entry_bar
            if bars_held >= MAX_BARS:
                exit_price = df['close'].iloc[i]
                if direction == "LONG":
                    pnl = (exit_price - entry_price) / entry_price * 100 - FEE_PCT
                else:
                    pnl = (entry_price - exit_price) / entry_price * 100 - FEE_PCT
                trades.append({'pattern': pattern_name, 'pnl': pnl, 'exit': 'TIMEOUT'})
                in_position = False
                continue

            h, l = df['high'].iloc[i], df['low'].iloc[i]
            if direction == "LONG":
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)
                tp_hit = h >= tp_price
                sl_hit = l <= sl_price
                if tp_hit and sl_hit:
                    tp_d = abs(h - tp_price)
                    sl_d = abs(l - sl_price)
                    pnl = (tp_pct if tp_d <= sl_d else -sl_pct) - FEE_PCT
                elif tp_hit:
                    pnl = tp_pct - FEE_PCT
                elif sl_hit:
                    pnl = -sl_pct - FEE_PCT
                else:
                    continue
            else:
                tp_price = entry_price * (1 - tp_pct / 100)
                sl_price = entry_price * (1 + sl_pct / 100)
                tp_hit = l <= tp_price
                sl_hit = h >= sl_price
                if tp_hit and sl_hit:
                    tp_d = abs(l - tp_price)
                    sl_d = abs(h - sl_price)
                    pnl = (tp_pct if tp_d <= sl_d else -sl_pct) - FEE_PCT
                elif tp_hit:
                    pnl = tp_pct - FEE_PCT
                elif sl_hit:
                    pnl = -sl_pct - FEE_PCT
                else:
                    continue

            trades.append({'pattern': pattern_name, 'pnl': pnl, 'exit': 'TP' if pnl > 0 else 'SL'})
            in_position = False
        else:
            pat = df['pattern_3'].iloc[i]
            if pd.isna(pat) or pat not in patterns:
                continue
            if i + 1 >= len(df):
                break
            entry_bar = i + 1
            entry_price = df['open'].iloc[i + 1]
            direction = patterns[pat]
            pattern_name = pat
            tp_pct = tpsl[pat]['tp']
            sl_pct = tpsl[pat]['sl']
            in_position = True

    return trades


def eval_trades(trades: list) -> dict:
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    cum = np.cumsum(pnls)
    rm = np.maximum.accumulate(cum)
    mdd = float(np.max(rm - cum)) if len(cum) > 0 else 0
    return {
        'pnl': round(sum(pnls), 2),
        'trades': len(trades),
        'wr': round(len(wins) / len(trades) * 100, 1),
        'mdd': round(mdd, 2),
        'pf': round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 999,
    }


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("Extended Data Pattern Rediscovery")
    print("  핵심 가설: 더 긴 훈련 기간 → 더 나은 genuine forward edge?")
    print("=" * 80)

    # ── Phase 1: Download data ────────────────────────────────
    cache_file = DATA_DIR / "btc_5m_720days_binance.csv"

    if cache_file.exists():
        print(f"\n  Loading cached data: {cache_file}")
        df = pd.read_csv(cache_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'type_code' not in df.columns:
            df = classify_candles(df)
            df.to_csv(cache_file, index=False)
    else:
        df = download_binance_data(TARGET_DAYS)
        df = classify_candles(df)
        df.to_csv(cache_file, index=False)
        print(f"  Saved: {cache_file}")

    total_days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400
    bars_per_day = len(df) / total_days
    print(f"  Total: {len(df)} bars, {total_days:.0f} days, {bars_per_day:.0f} bars/day")

    # ── Phase 2: Validate overlap with BingX data ─────────────
    print(f"\n{'='*80}")
    print("PHASE 2: Data Validation (Binance vs BingX overlap)")
    print(f"{'='*80}")

    bingx_file = DATA_DIR / "btc_5m_270days_reclassified.csv"
    if bingx_file.exists():
        df_bingx = pd.read_csv(bingx_file)
        df_bingx['timestamp'] = pd.to_datetime(df_bingx['timestamp'])

        # Find overlap
        overlap_start = max(df['timestamp'].iloc[0], df_bingx['timestamp'].iloc[0])
        overlap_end = min(df['timestamp'].iloc[-1], df_bingx['timestamp'].iloc[-1])

        binance_overlap = df[(df['timestamp'] >= overlap_start) & (df['timestamp'] <= overlap_end)]
        bingx_overlap = df_bingx[(df_bingx['timestamp'] >= overlap_start) & (df_bingx['timestamp'] <= overlap_end)]

        # Merge on timestamp and compare type_code
        merged = binance_overlap.merge(bingx_overlap[['timestamp', 'type_code']],
                                        on='timestamp', suffixes=('_binance', '_bingx'))
        match_rate = (merged['type_code_binance'] == merged['type_code_bingx']).mean() * 100

        print(f"  Overlap: {overlap_start.date()} ~ {overlap_end.date()}")
        print(f"  Matched bars: {len(merged)}")
        print(f"  Classification match rate: {match_rate:.1f}%")

        if match_rate < 90:
            print(f"  WARNING: Low match rate! Differences may affect pattern discovery.")
    else:
        print("  BingX data not found, skipping validation.")
        match_rate = None

    # ── Phase 3: Expanding-Window WF ──────────────────────────
    print(f"\n{'='*80}")
    print("PHASE 3: Expanding-Window Walk-Forward")
    print(f"  OOS period: {OOS_DAYS} days each fold")
    print(f"{'='*80}")

    oos_bars = OOS_DAYS * int(bars_per_day)
    total_bars = len(df)

    # Calculate fold structure
    folds = []
    # Work backwards from the end to create folds
    fold_end = total_bars
    while True:
        fold_oos_start = fold_end - oos_bars
        train_bars = fold_oos_start
        train_days = train_bars / bars_per_day

        if train_days < MIN_TRAIN_DAYS:
            break

        folds.append({
            'train_end': fold_oos_start,
            'oos_start': fold_oos_start,
            'oos_end': fold_end,
            'train_days': round(train_days),
        })
        fold_end = fold_oos_start

    folds.reverse()

    print(f"  Folds: {len(folds)}")
    for i, f in enumerate(folds):
        print(f"    Fold {i+1}: Train {f['train_days']}d → OOS {OOS_DAYS}d "
              f"(bars {f['oos_start']}-{f['oos_end']})")

    # ── Phase 4: Per-fold discovery + OOS ─────────────────────
    print(f"\n{'='*80}")
    print("PHASE 4: Per-Fold Pattern Discovery + OOS Test")
    print(f"{'='*80}")

    fold_results = []

    # Also test different Top-N selections
    for fi, fold in enumerate(folds):
        t0 = time.time()
        print(f"\n  --- Fold {fi+1} (Train {fold['train_days']}d) ---")

        df_train = df.iloc[:fold['train_end']].copy()
        df_oos = df.iloc[fold['oos_start']:fold['oos_end']].copy()

        # Collect instances from training data
        instances = collect_pattern_instances(df_train)
        print(f"    Universe: {len(instances)} pattern-direction combos")

        # Discover patterns
        discovered = discover_patterns(instances)
        discovered.sort(key=lambda x: x['pnl'], reverse=True)
        print(f"    Discovered: {len(discovered)} patterns (MC<{MC_THRESHOLD}, edge>{EDGE_THRESHOLD}pp)")

        # Test different Top-N selections on OOS
        result = {'fold': fi + 1, 'train_days': fold['train_days'],
                  'discovered': len(discovered), 'options': {}}

        for top_n in [10, 20, 30, 50, len(discovered)]:
            if top_n > len(discovered):
                actual_n = len(discovered)
            else:
                actual_n = top_n

            selected = discovered[:actual_n]

            # Build pattern dict for OOS
            patterns = {}
            tpsl = {}
            for p in selected:
                patterns[p['pattern']] = p['direction']
                tpsl[p['pattern']] = {'tp': p['tp'], 'sl': p['sl']}

            # OOS backtest
            trades = backtest_portfolio_1pos(df_oos, patterns, tpsl)
            stats = eval_trades(trades)

            label = f"Top{top_n}" if top_n <= 50 else "All"
            result['options'][label] = {
                'patterns': actual_n,
                **stats,
            }

            if top_n <= 50 or top_n == len(discovered):
                print(f"    {label:>6} ({actual_n:>3} pats): "
                      f"OOS PnL {stats['pnl']:>+8.1f}%, WR {stats['wr']:>5.1f}%, "
                      f"MDD {stats['mdd']:>5.1f}%, Trades {stats['trades']:>4}")

        elapsed = time.time() - t0
        print(f"    [{elapsed:.1f}s]")
        fold_results.append(result)

    # ── Phase 5: Aggregation & Comparison ─────────────────────
    print(f"\n{'='*80}")
    print("PHASE 5: Aggregation — Training Length vs OOS Performance")
    print(f"{'='*80}")

    # Aggregate by Top-N option across folds
    top_n_options = ['Top10', 'Top20', 'Top30', 'Top50', 'All']

    print(f"\n  {'Option':<10} ", end='')
    for fr in fold_results:
        print(f"{'F'+str(fr['fold'])+'('+str(fr['train_days'])+'d)':>16}", end='')
    print(f"{'TOTAL':>12}")
    print(f"  {'-'*10} ", end='')
    for _ in fold_results:
        print(f"  {'-'*14}", end='')
    print(f"  {'-'*10}")

    summary = {}
    for opt in top_n_options:
        total_pnl = 0
        total_trades = 0
        total_wins = 0
        all_fold_pnls = []
        print(f"  {opt:<10} ", end='')

        for fr in fold_results:
            if opt in fr['options']:
                stats = fr['options'][opt]
                pnl = stats['pnl']
                total_pnl += pnl
                total_trades += stats['trades']
                total_wins += int(stats['wr'] * stats['trades'] / 100)
                all_fold_pnls.append(pnl)
                print(f"  {pnl:>+12.1f}%", end='')
            else:
                print(f"  {'N/A':>13}", end='')

        total_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        print(f"  {total_pnl:>+10.1f}%")

        summary[opt] = {
            'total_pnl': round(total_pnl, 1),
            'total_trades': total_trades,
            'avg_wr': round(total_wr, 1),
            'fold_pnls': all_fold_pnls,
            'folds_positive': sum(1 for p in all_fold_pnls if p > 0),
        }

    # ── Phase 6: Comparison with original short-training results ──
    print(f"\n{'='*80}")
    print("PHASE 6: Extended vs Original Short-Training Comparison")
    print(f"{'='*80}")

    # Original E/F results from strategy_options_evaluation.py
    original = {
        'E_Fresh50_short': {'pnl': 8.6, 'wr': 68.2, 'mdd': 101.7,
                            'train_range': '54-216d', 'trades': 346},
        'F_Fresh20_short': {'pnl': -27.4, 'wr': 63.6, 'mdd': 89.7,
                            'train_range': '54-216d', 'trades': 283},
        'D_Dynamic_short': {'pnl': 80.5, 'wr': 68.5, 'mdd': 31.7,
                            'train_range': '54-216d', 'trades': 89},
    }

    print(f"\n  {'Option':<25} {'Train Range':>12} {'OOS PnL':>10} {'WR':>7} {'Trades':>8} {'Folds+':>7}")
    print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*7} {'-'*8} {'-'*7}")

    for name, data in original.items():
        print(f"  {name:<25} {data['train_range']:>12} {data['pnl']:>+9.1f}% "
              f"{data['wr']:>6.1f}% {data['trades']:>7} {'N/A':>7}")

    print(f"  {'─'*70}")

    train_range = f"{fold_results[0]['train_days']}-{fold_results[-1]['train_days']}d"
    for opt in top_n_options:
        s = summary[opt]
        print(f"  {'Extended_'+opt:<25} {train_range:>12} {s['total_pnl']:>+9.1f}% "
              f"{s['avg_wr']:>6.1f}% {s['total_trades']:>7} "
              f"{s['folds_positive']}/{len(fold_results):>5}")

    # ── Phase 7: Verdict ──────────────────────────────────────
    print(f"\n{'='*80}")
    print("PHASE 7: Verdict")
    print(f"{'='*80}")

    # Find best extended option
    best_opt = max(summary.items(), key=lambda x: x[1]['total_pnl'])

    # Compare with original E (Fresh-50)
    improvement = best_opt[1]['total_pnl'] - original['E_Fresh50_short']['pnl']

    if best_opt[1]['total_pnl'] > 50 and best_opt[1]['folds_positive'] >= len(fold_results) - 1:
        verdict = "HYPOTHESIS CONFIRMED — longer training improves genuine forward edge"
    elif best_opt[1]['total_pnl'] > 0:
        verdict = "PARTIAL — positive but not conclusively better"
    else:
        verdict = "HYPOTHESIS REJECTED — training length is not the bottleneck"

    print(f"\n  Best extended option: {best_opt[0]} (PnL +{best_opt[1]['total_pnl']}%)")
    print(f"  vs Original E (Fresh-50): +{original['E_Fresh50_short']['pnl']}%")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"\n  VERDICT: {verdict}")

    # ── Save results ──────────────────────────────────────────
    results = {
        'meta': {
            'script': 'extended_data_rediscovery.py',
            'timestamp': datetime.now().isoformat(),
            'data_days': round(total_days),
            'data_bars': len(df),
            'data_range': f"{df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}",
            'oos_days': OOS_DAYS,
            'min_train_days': MIN_TRAIN_DAYS,
            'mc_threshold': MC_THRESHOLD,
            'edge_threshold': EDGE_THRESHOLD,
        },
        'data_validation': {
            'overlap_match_rate': match_rate,
        },
        'folds': fold_results,
        'summary': summary,
        'comparison': {
            'original_short_training': original,
            'extended_best': {
                'option': best_opt[0],
                **best_opt[1],
            },
            'improvement_vs_E': improvement,
            'verdict': verdict,
        },
    }

    out_path = RESULTS_DIR / "extended_data_rediscovery.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
