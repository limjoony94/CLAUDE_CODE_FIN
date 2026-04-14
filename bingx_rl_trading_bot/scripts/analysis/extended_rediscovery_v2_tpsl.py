#!/usr/bin/env python3
"""
Extended Rediscovery v2: 5분봉 적합 TP/SL (Optimized)
======================================================
v1 연구에서 3-candle 패턴의 forward edge가 없다고 결론.
그러나 TP/SL 그리드 [0.3~3.0%]가 5분봉에 부적합할 수 있음.

가설: 5분봉 패턴은 단기 방향성을 예측하므로,
      더 타이트한 TP/SL + 짧은 보유 시간이 edge를 포착할 수 있다.

Phase 1: ATR 분석 — 720일간 5분봉 ATR 분포
Phase 2: MFE/MAE 분석 — 패턴 신호 후 실제 가격 움직임 프로파일
Phase 3: 3가지 TP/SL 전략 비교 WF
  A) Original: TP/SL [0.3-3.0%], MAX_BARS 500
  B) Tight: TP/SL [0.1-1.0%], MAX_BARS 100
  C) ATR-adaptive: TP/SL = k * ATR(14), k in [0.5-3.0], MAX_BARS 100

Optimized: numpy vectorization, no .iloc in hot loops
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
sys.path.insert(0, str(PROJECT_ROOT.parent))
from scripts.production.pattern_5m.indicators import classify_candle as _prod_classify
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ── Parameters ────────────────────────────────────────────────
FEE_PCT = 0.10
MC_SIMS = 3000
MC_THRESHOLD = 0.01
EDGE_THRESHOLD = 5.0
OOS_DAYS = 90
MIN_TRAIN_DAYS = 360

# Strategy A: Original wide grid
GRID_A = {
    'name': 'A_Original_Wide',
    'tp': [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0],
    'sl': [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0],
    'max_bars': 500,
}

# Strategy B: Tight grid for 5m
GRID_B = {
    'name': 'B_Tight_5m',
    'tp': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0],
    'sl': [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0],
    'max_bars': 100,
}

# Strategy C: ATR-adaptive
ATR_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
ATR_MAX_BARS = 100


# ── Candle classification — delegates to production canonical ─
def classify_candles(df: pd.DataFrame) -> pd.DataFrame:
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

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14, min_periods=1).mean()
    df['atr_pct'] = df['atr_14'] / df['close'] * 100

    return df


# ── Vectorized instance collection ───────────────────────────
def collect_instances(df: pd.DataFrame, max_bars: int):
    """Collect pattern instances — numpy vectorized, no .iloc in hot loop."""
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    pats = df['pattern_3'].values
    atrs = df['atr_pct'].values
    n = len(df)

    instances = defaultdict(list)

    for i in range(2, n - 1):
        pat = pats[i]
        if pat is None or (isinstance(pat, float) and np.isnan(pat)):
            continue
        entry_idx = i + 1
        if entry_idx >= n:
            continue
        entry_price = opens[entry_idx]
        if entry_price <= 0:
            continue

        end_idx = min(entry_idx + max_bars, n)
        # Vectorized: slice numpy arrays directly
        fav_long = (highs[entry_idx:end_idx] - entry_price) / entry_price * 100
        adv_long = (entry_price - lows[entry_idx:end_idx]) / entry_price * 100

        atr_pct = atrs[i]

        for direction in ['LONG', 'SHORT']:
            if direction == 'LONG':
                fav, adv = fav_long, adv_long
            else:
                fav, adv = adv_long, fav_long
            key = f"{pat}_{direction}"
            instances[key].append((fav.copy(), adv.copy(), atr_pct))

    return instances


# ── Vectorized TP/SL evaluation ──────────────────────────────
def eval_tpsl_batch(instances_list, tp, sl):
    """Evaluate TP/SL for all instances of a pattern. Returns pnl array."""
    pnls = np.empty(len(instances_list))
    for idx, (fav, adv, _) in enumerate(instances_list):
        tp_hits = fav >= tp
        sl_hits = adv >= sl
        either = tp_hits | sl_hits

        if np.any(either):
            first_bar = np.argmax(either)
            tp_hit = tp_hits[first_bar]
            sl_hit = sl_hits[first_bar]
            if tp_hit and sl_hit:
                pnls[idx] = (tp if abs(fav[first_bar]-tp) <= abs(adv[first_bar]-sl) else -sl) - FEE_PCT
            elif tp_hit:
                pnls[idx] = tp - FEE_PCT
            else:
                pnls[idx] = -sl - FEE_PCT
        else:
            pnls[idx] = (fav[-1] if len(fav) > 0 else 0) - FEE_PCT
    return pnls


def eval_atr_tpsl_batch(instances_list, tp_mult, sl_mult):
    """Evaluate ATR-adaptive TP/SL."""
    pnls = []
    for fav, adv, atr in instances_list:
        if atr <= 0:
            continue
        tp = atr * tp_mult
        sl = atr * sl_mult
        if sl < 0.1:
            continue

        tp_hits = fav >= tp
        sl_hits = adv >= sl
        either = tp_hits | sl_hits

        if np.any(either):
            first_bar = np.argmax(either)
            tp_hit = tp_hits[first_bar]
            sl_hit = sl_hits[first_bar]
            if tp_hit and sl_hit:
                pnls.append((tp if abs(fav[first_bar]-tp) <= abs(adv[first_bar]-sl) else -sl) - FEE_PCT)
            elif tp_hit:
                pnls.append(tp - FEE_PCT)
            else:
                pnls.append(-sl - FEE_PCT)
        else:
            pnls.append((fav[-1] if len(fav) > 0 else 0) - FEE_PCT)
    return np.array(pnls) if pnls else np.array([])


# ── MC test ───────────────────────────────────────────────────
def mc_test(pnls, n_sims=MC_SIMS):
    if len(pnls) < 10:
        return 1.0
    actual = np.sum(pnls)
    abs_p = np.abs(pnls)
    rng = np.random.default_rng(42)
    signs = rng.choice([-1, 1], size=(n_sims, len(pnls)))
    sims = np.sum(signs * abs_p, axis=1)  # vectorized MC
    return float(np.mean(sims >= actual))


# ── Discover with fixed grid ─────────────────────────────────
def discover_fixed(instances, grid, min_trades=15):
    discovered = []
    for key, insts in instances.items():
        if len(insts) < min_trades:
            continue
        pat, direction = key.rsplit('_', 1)
        best = None
        best_score = -999

        for tp in grid['tp']:
            for sl in grid['sl']:
                pnls = eval_tpsl_batch(insts, tp, sl)
                if len(pnls) < min_trades:
                    continue
                wr = np.sum(pnls > 0) / len(pnls) * 100
                rw_wr = sl / (tp + sl) * 100
                edge = wr - rw_wr
                total_pnl = np.sum(pnls)

                if edge < EDGE_THRESHOLD or total_pnl <= 0:
                    continue
                score = total_pnl * edge
                if score > best_score:
                    best_score = score
                    best = {
                        'pattern': pat, 'direction': direction,
                        'tp': tp, 'sl': sl,
                        'pnl': float(total_pnl), 'trades': len(pnls),
                        'wr': round(wr, 1), 'edge': round(edge, 1),
                        'pnls': pnls,
                    }

        if best is not None:
            p = mc_test(best['pnls'])
            if p < MC_THRESHOLD:
                best['mc_p'] = p
                del best['pnls']
                discovered.append(best)

    discovered.sort(key=lambda x: x['pnl'], reverse=True)
    return discovered


# ── Discover with ATR-adaptive ────────────────────────────────
def discover_atr(instances, min_trades=15):
    discovered = []
    for key, insts in instances.items():
        if len(insts) < min_trades:
            continue
        pat, direction = key.rsplit('_', 1)
        best = None
        best_score = -999
        med_atr = np.median([inst[2] for inst in insts])

        for tp_m in ATR_MULTIPLIERS:
            for sl_m in ATR_MULTIPLIERS:
                pnls = eval_atr_tpsl_batch(insts, tp_m, sl_m)
                if len(pnls) < min_trades:
                    continue
                wr = np.sum(pnls > 0) / len(pnls) * 100
                tp_approx = tp_m * med_atr
                sl_approx = sl_m * med_atr
                rw_wr = sl_approx / (tp_approx + sl_approx) * 100 if (tp_approx + sl_approx) > 0 else 50
                edge = wr - rw_wr
                total_pnl = np.sum(pnls)

                if edge < EDGE_THRESHOLD or total_pnl <= 0:
                    continue
                score = total_pnl * edge
                if score > best_score:
                    best_score = score
                    best = {
                        'pattern': pat, 'direction': direction,
                        'tp_mult': tp_m, 'sl_mult': sl_m,
                        'pnl': float(total_pnl), 'trades': len(pnls),
                        'wr': round(wr, 1), 'edge': round(edge, 1),
                        'pnls': pnls,
                    }

        if best is not None:
            p = mc_test(best['pnls'])
            if p < MC_THRESHOLD:
                best['mc_p'] = p
                del best['pnls']
                discovered.append(best)

    discovered.sort(key=lambda x: x['pnl'], reverse=True)
    return discovered


# ── Portfolio backtest (numpy-optimized) ─────────────────────
def backtest_fixed(df, patterns, tpsl, max_bars):
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    closes = df['close'].values
    pats = df['pattern_3'].values
    n = len(df)

    trades = []
    in_pos = False
    eb = ep = tp = sl = 0.0
    d = pn = None

    for i in range(2, n):
        if in_pos:
            bh = i - eb
            if bh >= max_bars:
                if d == 'LONG':
                    pnl = (closes[i] - ep) / ep * 100 - FEE_PCT
                else:
                    pnl = (ep - closes[i]) / ep * 100 - FEE_PCT
                trades.append({'pattern': pn, 'pnl': pnl})
                in_pos = False; continue

            h, lo = highs[i], lows[i]
            if d == 'LONG':
                tp_hit = h >= ep * (1 + tp / 100)
                sl_hit = lo <= ep * (1 - sl / 100)
            else:
                tp_hit = lo <= ep * (1 - tp / 100)
                sl_hit = h >= ep * (1 + sl / 100)

            if tp_hit and sl_hit:
                if d == 'LONG':
                    td = abs(h - ep*(1+tp/100))
                    sd = abs(lo - ep*(1-sl/100))
                else:
                    td = abs(lo - ep*(1-tp/100))
                    sd = abs(h - ep*(1+sl/100))
                pnl = (tp if td <= sd else -sl) - FEE_PCT
            elif tp_hit:
                pnl = tp - FEE_PCT
            elif sl_hit:
                pnl = -sl - FEE_PCT
            else:
                continue

            trades.append({'pattern': pn, 'pnl': pnl})
            in_pos = False
        else:
            pat = pats[i]
            if pat is None or (isinstance(pat, float) and np.isnan(pat)):
                continue
            if pat not in patterns:
                continue
            if i + 1 >= n:
                break
            eb = i + 1
            ep = opens[eb]
            d = patterns[pat]
            pn = pat
            tp = tpsl[pat]['tp']
            sl = tpsl[pat]['sl']
            in_pos = True

    return trades


def backtest_atr(df, patterns, tpsl_mults, max_bars):
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    closes = df['close'].values
    pats = df['pattern_3'].values
    atrs = df['atr_pct'].values
    n = len(df)

    trades = []
    in_pos = False
    eb = ep = tp = sl = 0.0
    d = pn = None

    for i in range(2, n):
        if in_pos:
            bh = i - eb
            if bh >= max_bars:
                if d == 'LONG':
                    pnl = (closes[i] - ep) / ep * 100 - FEE_PCT
                else:
                    pnl = (ep - closes[i]) / ep * 100 - FEE_PCT
                trades.append({'pattern': pn, 'pnl': pnl})
                in_pos = False; continue

            h, lo = highs[i], lows[i]
            if d == 'LONG':
                tp_hit = h >= ep * (1 + tp / 100)
                sl_hit = lo <= ep * (1 - sl / 100)
            else:
                tp_hit = lo <= ep * (1 - tp / 100)
                sl_hit = h >= ep * (1 + sl / 100)

            if tp_hit and sl_hit:
                if d == 'LONG':
                    td = abs(h - ep*(1+tp/100))
                    sd = abs(lo - ep*(1-sl/100))
                else:
                    td = abs(lo - ep*(1-tp/100))
                    sd = abs(h - ep*(1+sl/100))
                pnl = (tp if td <= sd else -sl) - FEE_PCT
            elif tp_hit:
                pnl = tp - FEE_PCT
            elif sl_hit:
                pnl = -sl - FEE_PCT
            else:
                continue

            trades.append({'pattern': pn, 'pnl': pnl})
            in_pos = False
        else:
            pat = pats[i]
            if pat is None or (isinstance(pat, float) and np.isnan(pat)):
                continue
            if pat not in patterns:
                continue
            if i + 1 >= n:
                break
            eb = i + 1
            ep = opens[eb]
            d = patterns[pat]
            pn = pat
            atr = atrs[i]
            m = tpsl_mults[pat]
            tp = atr * m['tp_mult']
            sl = atr * m['sl_mult']
            if sl < 0.1:
                continue
            in_pos = True

    return trades


def eval_trades(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
    pnls = np.array([t['pnl'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    cum = np.cumsum(pnls)
    rm = np.maximum.accumulate(cum)
    mdd = float(np.max(rm - cum)) if len(cum) > 0 else 0
    return {
        'pnl': round(float(np.sum(pnls)), 2),
        'trades': len(trades),
        'wr': round(float(len(wins) / len(trades) * 100), 1),
        'mdd': round(mdd, 2),
        'pf': round(float(np.sum(wins) / abs(np.sum(losses))), 2) if len(losses) > 0 and np.sum(losses) != 0 else 999,
    }


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 80, flush=True)
    print("Extended Rediscovery v2: 5분봉 적합 TP/SL (Optimized)", flush=True)
    print("  가설: 더 타이트한 TP/SL + 짧은 보유 시간이 edge를 포착", flush=True)
    print("=" * 80, flush=True)

    # Load data
    cache_file = DATA_DIR / "btc_5m_720days_binance.csv"
    if not cache_file.exists():
        print("ERROR: 720-day data not found. Run extended_data_rediscovery.py first.")
        return

    print("  Loading data...", flush=True)
    df = pd.read_csv(cache_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Always reclassify to ensure consistency
    print("  Classifying candles...", flush=True)
    df = classify_candles(df)

    total_days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400
    bars_per_day = len(df) / total_days
    print(f"  Data: {len(df)} bars, {total_days:.0f} days", flush=True)

    # ── Phase 1: ATR Analysis ─────────────────────────────────
    print(f"\n{'='*80}", flush=True)
    print("PHASE 1: ATR Analysis (5분봉 변동성 프로파일)", flush=True)
    print(f"{'='*80}", flush=True)

    atr_pct = df['atr_pct'].dropna().values
    print(f"  ATR(14) as % of price:", flush=True)
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(atr_pct, p)
        print(f"    P{p:02d}: {val:.4f}%", flush=True)

    med_atr = float(np.median(atr_pct))
    print(f"\n  TP/SL implications (median ATR = {med_atr:.4f}%):", flush=True)
    for mult in [0.5, 1.0, 1.5, 2.0, 3.0]:
        print(f"    {mult}x ATR = {med_atr * mult:.3f}%", flush=True)

    # ── Phase 2: MFE/MAE Analysis ─────────────────────────────
    print(f"\n{'='*80}", flush=True)
    print("PHASE 2: MFE/MAE Analysis (패턴 신호 후 가격 프로파일)", flush=True)
    print(f"{'='*80}", flush=True)

    highs_arr = df['high'].values
    lows_arr = df['low'].values
    opens_arr = df['open'].values
    pats_arr = df['pattern_3'].values

    mfe_by_horizon = defaultdict(list)
    mae_by_horizon = defaultdict(list)
    horizons = [5, 10, 20, 50, 100, 200, 500]
    sample_count = 0
    n_total = len(df)

    for i in range(2, n_total - 500):
        pat = pats_arr[i]
        if pat is None or (isinstance(pat, float) and np.isnan(pat)):
            continue
        entry_idx = i + 1
        entry_price = opens_arr[entry_idx]
        if entry_price <= 0:
            continue

        sample_count += 1
        for h in horizons:
            end = entry_idx + h
            if end > n_total:
                break
            mfe = (np.max(highs_arr[entry_idx:end]) - entry_price) / entry_price * 100
            mae = (entry_price - np.min(lows_arr[entry_idx:end])) / entry_price * 100
            mfe_by_horizon[h].append(mfe)
            mae_by_horizon[h].append(mae)

    print(f"  Signal instances: {sample_count}", flush=True)
    print(f"\n  {'Horizon':>8} {'MFE_P50':>8} {'MFE_P75':>8} {'MFE_P90':>8} | {'MAE_P50':>8} {'MAE_P75':>8} {'MAE_P90':>8}", flush=True)
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8}   {'-'*8} {'-'*8} {'-'*8}", flush=True)

    mfe_mae_data = {}
    for h in horizons:
        mfe = np.array(mfe_by_horizon[h])
        mae = np.array(mae_by_horizon[h])
        print(f"  {h:>7}b {np.percentile(mfe,50):>7.3f}% {np.percentile(mfe,75):>7.3f}% {np.percentile(mfe,90):>7.3f}% |"
              f" {np.percentile(mae,50):>7.3f}% {np.percentile(mae,75):>7.3f}% {np.percentile(mae,90):>7.3f}%", flush=True)
        mfe_mae_data[str(h)] = {
            'mfe_p50': round(float(np.percentile(mfe, 50)), 3),
            'mfe_p75': round(float(np.percentile(mfe, 75)), 3),
            'mfe_p90': round(float(np.percentile(mfe, 90)), 3),
            'mae_p50': round(float(np.percentile(mae, 50)), 3),
            'mae_p75': round(float(np.percentile(mae, 75)), 3),
            'mae_p90': round(float(np.percentile(mae, 90)), 3),
        }

    # ── Phase 3: WF with 3 strategies ─────────────────────────
    print(f"\n{'='*80}", flush=True)
    print("PHASE 3: Walk-Forward — 3 TP/SL Strategies", flush=True)
    print(f"{'='*80}", flush=True)

    oos_bars = OOS_DAYS * int(bars_per_day)
    total_bars = len(df)

    folds = []
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
    print(f"  Folds: {len(folds)}", flush=True)
    for i, f in enumerate(folds):
        print(f"    Fold {i+1}: Train {f['train_days']}d → OOS {OOS_DAYS}d", flush=True)

    all_results = {'A_Original': [], 'B_Tight': [], 'C_ATR': []}

    for fi, fold in enumerate(folds):
        print(f"\n  ━━━ Fold {fi+1} (Train {fold['train_days']}d) ━━━", flush=True)

        df_train = df.iloc[:fold['train_end']].copy()
        df_oos = df.iloc[fold['oos_start']:fold['oos_end']].copy()

        # Strategy A: Original wide grid
        t0 = time.time()
        print(f"    A(Wide): collecting instances (max_bars={GRID_A['max_bars']})...", flush=True)
        instances_a = collect_instances(df_train, GRID_A['max_bars'])
        print(f"    A(Wide): {len(instances_a)} pattern-direction combos, discovering...", flush=True)
        discovered_a = discover_fixed(instances_a, GRID_A)
        top_a = discovered_a[:20]
        if top_a:
            pats_a = {p['pattern']: p['direction'] for p in top_a}
            tpsl_a = {p['pattern']: {'tp': p['tp'], 'sl': p['sl']} for p in top_a}
            trades_a = backtest_fixed(df_oos, pats_a, tpsl_a, GRID_A['max_bars'])
            stats_a = eval_trades(trades_a)
        else:
            stats_a = {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
        ta = time.time() - t0
        print(f"    A(Wide):  {len(discovered_a):>3} disc → Top20 OOS: PnL {stats_a['pnl']:>+8.1f}%, "
              f"WR {stats_a['wr']:>5.1f}%, MDD {stats_a['mdd']:>5.1f}%, Trades {stats_a['trades']:>3} [{ta:.0f}s]", flush=True)
        all_results['A_Original'].append({**stats_a, 'discovered': len(discovered_a)})
        del instances_a  # free memory

        # Strategy B: Tight 5m grid
        t0 = time.time()
        print(f"    B(Tight): collecting instances (max_bars={GRID_B['max_bars']})...", flush=True)
        instances_b = collect_instances(df_train, GRID_B['max_bars'])
        print(f"    B(Tight): {len(instances_b)} pattern-direction combos, discovering...", flush=True)
        discovered_b = discover_fixed(instances_b, GRID_B)
        top_b = discovered_b[:20]
        if top_b:
            pats_b = {p['pattern']: p['direction'] for p in top_b}
            tpsl_b = {p['pattern']: {'tp': p['tp'], 'sl': p['sl']} for p in top_b}
            trades_b = backtest_fixed(df_oos, pats_b, tpsl_b, GRID_B['max_bars'])
            stats_b = eval_trades(trades_b)
        else:
            stats_b = {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
        tb = time.time() - t0
        print(f"    B(Tight): {len(discovered_b):>3} disc → Top20 OOS: PnL {stats_b['pnl']:>+8.1f}%, "
              f"WR {stats_b['wr']:>5.1f}%, MDD {stats_b['mdd']:>5.1f}%, Trades {stats_b['trades']:>3} [{tb:.0f}s]", flush=True)
        all_results['B_Tight'].append({**stats_b, 'discovered': len(discovered_b)})
        del instances_b

        # Strategy C: ATR-adaptive
        t0 = time.time()
        print(f"    C(ATR): collecting instances (max_bars={ATR_MAX_BARS})...", flush=True)
        instances_c = collect_instances(df_train, ATR_MAX_BARS)
        print(f"    C(ATR): {len(instances_c)} pattern-direction combos, discovering...", flush=True)
        discovered_c = discover_atr(instances_c)
        top_c = discovered_c[:20]
        if top_c:
            pats_c = {p['pattern']: p['direction'] for p in top_c}
            tpsl_c = {p['pattern']: {'tp_mult': p['tp_mult'], 'sl_mult': p['sl_mult']} for p in top_c}
            trades_c = backtest_atr(df_oos, pats_c, tpsl_c, ATR_MAX_BARS)
            stats_c = eval_trades(trades_c)
        else:
            stats_c = {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
        tc = time.time() - t0
        print(f"    C(ATR):   {len(discovered_c):>3} disc → Top20 OOS: PnL {stats_c['pnl']:>+8.1f}%, "
              f"WR {stats_c['wr']:>5.1f}%, MDD {stats_c['mdd']:>5.1f}%, Trades {stats_c['trades']:>3} [{tc:.0f}s]", flush=True)
        all_results['C_ATR'].append({**stats_c, 'discovered': len(discovered_c)})
        del instances_c

    # ── Phase 4: Summary ──────────────────────────────────────
    print(f"\n{'='*80}", flush=True)
    print("PHASE 4: Summary Comparison", flush=True)
    print(f"{'='*80}", flush=True)

    print(f"\n  {'Strategy':<15}", end='')
    for fi in range(len(folds)):
        print(f" {'F'+str(fi+1)+'('+str(folds[fi]['train_days'])+'d)':>14}", end='')
    print(f" {'TOTAL':>10} {'Avg WR':>8}", flush=True)
    print(f"  {'-'*15}", end='')
    for _ in folds:
        print(f" {'-'*14}", end='')
    print(f" {'-'*10} {'-'*8}", flush=True)

    summary = {}
    for strat, results in all_results.items():
        total_pnl = sum(r['pnl'] for r in results)
        total_trades = sum(r['trades'] for r in results)
        total_wins = sum(int(r['wr'] * r['trades'] / 100) for r in results)
        avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

        print(f"  {strat:<15}", end='')
        for r in results:
            print(f" {r['pnl']:>+13.1f}%", end='')
        print(f" {total_pnl:>+9.1f}% {avg_wr:>7.1f}%", flush=True)

        summary[strat] = {
            'total_pnl': round(total_pnl, 1),
            'total_trades': total_trades,
            'avg_wr': round(avg_wr, 1),
            'folds_positive': sum(1 for r in results if r['pnl'] > 0),
            'total_folds': len(results),
            'per_fold': results,
        }

    # ── Phase 5: Verdict ──────────────────────────────────────
    print(f"\n{'='*80}", flush=True)
    print("PHASE 5: Verdict", flush=True)
    print(f"{'='*80}", flush=True)

    best = max(summary.items(), key=lambda x: x[1]['total_pnl'])
    print(f"\n  Best strategy: {best[0]} (Total PnL: {best[1]['total_pnl']:+.1f}%)", flush=True)
    print(f"  Folds positive: {best[1]['folds_positive']}/{best[1]['total_folds']}", flush=True)

    if best[1]['total_pnl'] > 30 and best[1]['folds_positive'] >= len(folds) - 1:
        verdict = "TP/SL ADJUSTMENT HELPS — tighter settings capture short-term edge"
    elif best[1]['total_pnl'] > 0:
        verdict = "MARGINAL IMPROVEMENT — some benefit but not conclusive"
    else:
        verdict = "TP/SL NOT THE BOTTLENECK — 3-candle patterns lack forward edge regardless of exit strategy"

    print(f"\n  VERDICT: {verdict}", flush=True)

    print(f"\n  vs v1 (Original Wide only, 450-630d):", flush=True)
    print(f"    v1 Best: -38.7% (Top10)", flush=True)
    print(f"    v2 A(Wide): {summary['A_Original']['total_pnl']:+.1f}%", flush=True)
    print(f"    v2 B(Tight): {summary['B_Tight']['total_pnl']:+.1f}%", flush=True)
    print(f"    v2 C(ATR): {summary['C_ATR']['total_pnl']:+.1f}%", flush=True)

    # Save
    results = {
        'meta': {
            'script': 'extended_rediscovery_v2_tpsl.py',
            'timestamp': datetime.now().isoformat(),
            'data_days': round(total_days),
        },
        'atr_analysis': {
            'percentiles': {f'p{p}': round(float(np.percentile(atr_pct, p)), 4)
                           for p in [10, 25, 50, 75, 90, 95, 99]},
            'median': round(float(np.median(atr_pct)), 4),
        },
        'mfe_mae': mfe_mae_data,
        'strategies': summary,
        'verdict': verdict,
        'grids': {
            'A': {'tp': GRID_A['tp'], 'sl': GRID_A['sl'], 'max_bars': GRID_A['max_bars']},
            'B': {'tp': GRID_B['tp'], 'sl': GRID_B['sl'], 'max_bars': GRID_B['max_bars']},
            'C': {'atr_mults': ATR_MULTIPLIERS, 'max_bars': ATR_MAX_BARS},
        },
    }

    out_path = RESULTS_DIR / "extended_rediscovery_v2_tpsl.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}", flush=True)


if __name__ == '__main__':
    main()
