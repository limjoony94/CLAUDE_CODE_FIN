"""
M2 Round 2 — Pre-BT Map Screening (Gate 5 + Gate 6, single script)
===================================================================
12 NEW variants × Gate 5/6 + 4 Round 1 cited = 16-cell map.

Dimensions:
  D1 Timeframe (1h): V1-V4 from Round 1, applied on 1h timeframe
  D2 NEW signal classes (BTC 15m): vol-regime / range-break+retest / HL pivot / pullback to 1h EMA20
  D3 No trend filter (BTC 15m): Round 1 V1-V4 with trend filter REMOVED

Standard fields (all variants):
  - raw signals + per-day rate
  - isolation gross_sum at 3 horizons (PASS = gross > 0 in ≥2)
  - candidate MFE P50 / MAE P50 / %MFE > friction
  - random baseline avg (5 seeds, same eligible universe per dimension)
  - Δ MFE_P50, Δ %>fr (Gate 6)
  - 신설 MFE+MAE asymmetry sum (대칭이면 ~0)
  - verdict

NO Phase 3 BT.
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from m2_round1_screening import (compute_ema, compute_rsi, compute_bb, rolling_min,
                                  load_ohlcv, resample_to_4h, merge_htf,
                                  apply_n1_sequencing, isolation_test,
                                  measure_mfe_for_signals, percentile, stats_mfe)


def compute_atr(highs, lows, closes, period=14):
    n = len(closes)
    tr = [0.0] * n
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
    atr = [float('nan')] * n
    if n >= period:
        atr[period - 1] = sum(tr[:period]) / period
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return np.array(atr)


def rolling_max(arr, lookback):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        out[i] = np.nanmax(arr[i - lookback + 1:i + 1])
    return out


def rolling_min_arr(arr, lookback):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        out[i] = np.nanmin(arr[i - lookback + 1:i + 1])
    return out


def sma(arr, period):
    """NaN-safe rolling mean (NaN within window → NaN, otherwise mean of window)."""
    return pd.Series(arr).rolling(period, min_periods=period).mean().values


# ---------- Data preparation ----------

def prepare_15m_data():
    """15m + 1h + 4h trend filter on 15m bars."""
    df_15m = load_ohlcv(ROOT / 'data' / 'btc_15m_720days.csv')
    df_1h = load_ohlcv(ROOT / 'data' / 'btc_1h_720days.csv')
    df_4h = resample_to_4h(df_1h)

    closes = df_15m['close'].values
    highs = df_15m['high'].values
    lows = df_15m['low'].values

    df_15m['ema9'] = compute_ema(closes, 9)
    df_15m['ema20'] = compute_ema(closes, 20)
    df_15m['rsi14'] = compute_rsi(closes, 14)
    upper, lower, _, width = compute_bb(closes, 20, 2.0)
    df_15m['bb_upper'] = upper
    df_15m['bb_lower'] = lower
    df_15m['bb_width'] = width
    df_15m['bb_width_min50'] = rolling_min(width, 50)
    df_15m['atr14'] = compute_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)
    df_15m['atr_sma50'] = sma(df_15m['atr14'].values, 50)
    df_15m['high_24'] = rolling_max(highs, 24)
    df_15m['low_24'] = rolling_min_arr(lows, 24)
    df_15m['high_24_prev'] = df_15m['high_24'].shift(1)  # 24-bar high BEFORE current bar
    df_15m['low_24_prev'] = df_15m['low_24'].shift(1)
    df_15m['low_10'] = rolling_min_arr(lows, 10)
    df_15m['low_20'] = rolling_min_arr(lows, 20)
    df_15m['high_10'] = rolling_max(highs, 10)
    df_15m['high_20'] = rolling_max(highs, 20)
    df_15m['body_ratio'] = (df_15m['close'] - df_15m['open']).abs() / \
        (df_15m['high'] - df_15m['low']).replace(0, np.nan)

    df_1h['ema20'] = compute_ema(df_1h['close'].values, 20)
    df_1h['ema50'] = compute_ema(df_1h['close'].values, 50)
    df_1h['htf_long'] = df_1h['ema20'] > df_1h['ema50']

    df_4h['ema50'] = compute_ema(df_4h['close'].values, 50)
    df_4h['htf_long'] = df_4h['close'] > df_4h['ema50']

    df_15m['close_time'] = df_15m['timestamp'] + pd.Timedelta(minutes=15)
    df_15m = merge_htf(df_15m, df_1h.rename(columns={'htf_long': 'h1_long', 'ema20': 'h1_ema20'}),
                        60, ['h1_long', 'h1_ema20'])
    df_15m = merge_htf(df_15m, df_4h.rename(columns={'htf_long': 'h4_long'}), 240, ['h4_long'])
    df_15m = df_15m.sort_values('timestamp').reset_index(drop=True)

    h1_long = df_15m['h1_long'].fillna(False).astype(bool).values
    h4_long = df_15m['h4_long'].fillna(False).astype(bool).values
    valid_with_filter = ((~pd.isna(df_15m['rsi14'])) & (~pd.isna(df_15m['ema9']))
                         & (~pd.isna(df_15m['atr14'])) & (~pd.isna(df_15m['atr_sma50']))
                         & (~pd.isna(df_15m['bb_width_min50'])) & (~pd.isna(df_15m['high_24_prev']))
                         & (~pd.isna(df_15m['low_10'])) & (~pd.isna(df_15m['h1_ema20']))
                         & (~df_15m['h1_long'].isna()) & (~df_15m['h4_long'].isna())).values
    valid_no_filter = ((~pd.isna(df_15m['rsi14'])) & (~pd.isna(df_15m['ema9']))
                       & (~pd.isna(df_15m['bb_width_min50']))).values

    return df_15m, h1_long, h4_long, valid_with_filter, valid_no_filter


def prepare_1h_data():
    """1h + 4h trend filter on 1h bars."""
    df_1h = load_ohlcv(ROOT / 'data' / 'btc_1h_720days.csv')
    df_4h = resample_to_4h(df_1h)

    closes = df_1h['close'].values
    df_1h['ema9'] = compute_ema(closes, 9)
    df_1h['ema20'] = compute_ema(closes, 20)
    df_1h['ema50'] = compute_ema(closes, 50)
    df_1h['rsi14'] = compute_rsi(closes, 14)
    upper, lower, _, width = compute_bb(closes, 20, 2.0)
    df_1h['bb_upper'] = upper
    df_1h['bb_lower'] = lower
    df_1h['bb_width'] = width
    df_1h['bb_width_min50'] = rolling_min(width, 50)
    df_1h['body_ratio'] = (df_1h['close'] - df_1h['open']).abs() / \
        (df_1h['high'] - df_1h['low']).replace(0, np.nan)
    df_1h['htf_long_self'] = df_1h['ema20'] > df_1h['ema50']  # 1h's own trend

    df_4h['ema50'] = compute_ema(df_4h['close'].values, 50)
    df_4h['htf_long'] = df_4h['close'] > df_4h['ema50']

    df_1h['close_time'] = df_1h['timestamp'] + pd.Timedelta(minutes=60)
    df_1h = merge_htf(df_1h, df_4h.rename(columns={'htf_long': 'h4_long'}), 240, ['h4_long'])
    df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)

    h1_long = df_1h['htf_long_self'].fillna(False).astype(bool).values
    h4_long = df_1h['h4_long'].fillna(False).astype(bool).values
    valid_mask = ((~pd.isna(df_1h['rsi14'])) & (~pd.isna(df_1h['ema9']))
                   & (~pd.isna(df_1h['bb_width_min50']))
                   & (~df_1h['htf_long_self'].isna()) & (~df_1h['h4_long'].isna())).values

    return df_1h, h1_long, h4_long, valid_mask


# ---------- Variant signal functions ----------

# D1 (1h timeframe variants) — identical to Round 1 V1-V4 but on 1h df
def signals_v1_mean_rev(df, h1_long, h4_long, valid_mask):
    n = len(df)
    rsi = df['rsi14'].values; op = df['open'].values; cl = df['close'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if h1_long[i] and h4_long[i] and rsi[i] <= 25 and cl[i] > op[i]:
            sigs.append((i, 'LONG'))
        elif (not h1_long[i]) and (not h4_long[i]) and rsi[i] >= 75 and cl[i] < op[i]:
            sigs.append((i, 'SHORT'))
    return sigs


def signals_v2_squeeze(df, h1_long, h4_long, valid_mask):
    n = len(df)
    cl = df['close'].values; width = df['bb_width'].values
    width_min = df['bb_width_min50'].values
    upper = df['bb_upper'].values; lower = df['bb_lower'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if pd.isna(width[i - 1]) or pd.isna(width_min[i - 1]): continue
        if not (width[i - 1] <= width_min[i - 1] * 1.001): continue
        if h1_long[i] and h4_long[i] and cl[i] > upper[i - 1]:
            sigs.append((i, 'LONG'))
        elif (not h1_long[i]) and (not h4_long[i]) and cl[i] < lower[i - 1]:
            sigs.append((i, 'SHORT'))
    return sigs


def signals_v3_momentum(df, h1_long, h4_long, valid_mask):
    n = len(df)
    op = df['open'].values; cl = df['close'].values
    high = df['high'].values; low = df['low'].values
    sigs = []
    for i in range(3, n):
        if not valid_mask[i]: continue
        bull3 = cl[i] > op[i] and cl[i - 1] > op[i - 1] and cl[i - 2] > op[i - 2]
        bear3 = cl[i] < op[i] and cl[i - 1] < op[i - 1] and cl[i - 2] < op[i - 2]
        if bull3 and h1_long[i] and h4_long[i]:
            move_pct = (high[i] - low[i - 2]) / low[i - 2] * 100
            if move_pct >= 0.3: sigs.append((i, 'LONG'))
        elif bear3 and (not h1_long[i]) and (not h4_long[i]):
            move_pct = (high[i - 2] - low[i]) / high[i - 2] * 100
            if move_pct >= 0.3: sigs.append((i, 'SHORT'))
    return sigs


def signals_v4_m1_minus_rsi(df, h1_long, h4_long, valid_mask):
    n = len(df)
    cl = df['close'].values; ema9 = df['ema9'].values
    body_ratio = df['body_ratio'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if pd.isna(body_ratio[i]) or body_ratio[i] <= 0.4: continue
        if h1_long[i] and h4_long[i] and cl[i] > ema9[i]:
            sigs.append((i, 'LONG'))
        elif (not h1_long[i]) and (not h4_long[i]) and cl[i] < ema9[i]:
            sigs.append((i, 'SHORT'))
    return sigs


# D2 NEW signal classes (15m)

def signals_d2_v1_vol_regime(df, h1_long, h4_long, valid_mask):
    """Volatility regime shift: ATR > ATR SMA50 + bullish/bearish bar in trend."""
    n = len(df)
    op = df['open'].values; cl = df['close'].values
    atr = df['atr14'].values; atr_sma = df['atr_sma50'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if pd.isna(atr[i]) or pd.isna(atr_sma[i]): continue
        if not (atr[i] > atr_sma[i]): continue
        if h1_long[i] and h4_long[i] and cl[i] > op[i]:
            sigs.append((i, 'LONG'))
        elif (not h1_long[i]) and (not h4_long[i]) and cl[i] < op[i]:
            sigs.append((i, 'SHORT'))
    return sigs


def signals_d2_v2_range_break_retest(df, h1_long, h4_long, valid_mask):
    """Range break with retest on 24-bar high/low.
    LONG: prev bar's close > 24-bar high (broken), then current bar low < broken level (retest in)
          AND current close > broken level (back above)."""
    n = len(df)
    cl = df['close'].values; lo = df['low'].values; hi = df['high'].values
    high24_prev = df['high_24_prev'].values
    low24_prev = df['low_24_prev'].values
    sigs = []
    for i in range(2, n):
        if not valid_mask[i]: continue
        # Use close at i-1 as "broken level reference at i-2"
        if pd.isna(high24_prev[i - 1]) or pd.isna(low24_prev[i - 1]): continue
        broken_high = high24_prev[i - 1]
        broken_low = low24_prev[i - 1]
        # LONG: prev bar broke above 24-bar high
        if cl[i - 1] > broken_high:
            # Current bar pulls back into range and closes back above level
            if lo[i] < broken_high and cl[i] > broken_high:
                if h1_long[i] and h4_long[i]:
                    sigs.append((i, 'LONG'))
        # SHORT mirror
        if cl[i - 1] < broken_low:
            if hi[i] > broken_low and cl[i] < broken_low:
                if (not h1_long[i]) and (not h4_long[i]):
                    sigs.append((i, 'SHORT'))
    return sigs


def signals_d2_v3_hl_pivot(df, h1_long, h4_long, valid_mask):
    """Higher-low / lower-high structural pivot.
    LONG: 10-bar low > 20-bar low (HL formed) + bullish bar."""
    n = len(df)
    op = df['open'].values; cl = df['close'].values
    low10 = df['low_10'].values; low20 = df['low_20'].values
    high10 = df['high_10'].values; high20 = df['high_20'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if pd.isna(low10[i]) or pd.isna(low20[i]): continue
        if h1_long[i] and h4_long[i] and low10[i] > low20[i] and cl[i] > op[i]:
            sigs.append((i, 'LONG'))
        elif (not h1_long[i]) and (not h4_long[i]) and high10[i] < high20[i] and cl[i] < op[i]:
            sigs.append((i, 'SHORT'))
    return sigs


def signals_d2_v4_pullback_to_1h_ema20(df, h1_long, h4_long, valid_mask):
    """Trend pullback to 1h EMA20.
    LONG: 15m close within 0.3% of 1h EMA20 + close > prev close."""
    n = len(df)
    cl = df['close'].values
    h1_ema20 = df['h1_ema20'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if pd.isna(h1_ema20[i]): continue
        ratio = cl[i] / h1_ema20[i]
        in_band = 0.997 <= ratio <= 1.003
        if not in_band: continue
        bounce = cl[i] > cl[i - 1]
        rejection = cl[i] < cl[i - 1]
        if h1_long[i] and h4_long[i] and bounce:
            sigs.append((i, 'LONG'))
        elif (not h1_long[i]) and (not h4_long[i]) and rejection:
            sigs.append((i, 'SHORT'))
    return sigs


# D3 — Round 1 V1-V4 with NO trend filter
def signals_d3_v1(df, valid_mask):
    n = len(df)
    rsi = df['rsi14'].values; op = df['open'].values; cl = df['close'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if rsi[i] <= 25 and cl[i] > op[i]:
            sigs.append((i, 'LONG'))
        elif rsi[i] >= 75 and cl[i] < op[i]:
            sigs.append((i, 'SHORT'))
    return sigs


def signals_d3_v2(df, valid_mask):
    n = len(df)
    cl = df['close'].values; width = df['bb_width'].values
    width_min = df['bb_width_min50'].values
    upper = df['bb_upper'].values; lower = df['bb_lower'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if pd.isna(width[i - 1]) or pd.isna(width_min[i - 1]): continue
        if not (width[i - 1] <= width_min[i - 1] * 1.001): continue
        if cl[i] > upper[i - 1]:
            sigs.append((i, 'LONG'))
        elif cl[i] < lower[i - 1]:
            sigs.append((i, 'SHORT'))
    return sigs


def signals_d3_v3(df, valid_mask):
    n = len(df)
    op = df['open'].values; cl = df['close'].values
    high = df['high'].values; low = df['low'].values
    sigs = []
    for i in range(3, n):
        if not valid_mask[i]: continue
        bull3 = cl[i] > op[i] and cl[i - 1] > op[i - 1] and cl[i - 2] > op[i - 2]
        bear3 = cl[i] < op[i] and cl[i - 1] < op[i - 1] and cl[i - 2] < op[i - 2]
        if bull3:
            if (high[i] - low[i - 2]) / low[i - 2] * 100 >= 0.3:
                sigs.append((i, 'LONG'))
        elif bear3:
            if (high[i - 2] - low[i]) / high[i - 2] * 100 >= 0.3:
                sigs.append((i, 'SHORT'))
    return sigs


def signals_d3_v4(df, valid_mask):
    n = len(df)
    cl = df['close'].values; ema9 = df['ema9'].values
    body_ratio = df['body_ratio'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if pd.isna(body_ratio[i]) or body_ratio[i] <= 0.4: continue
        if cl[i] > ema9[i]:
            sigs.append((i, 'LONG'))
        elif cl[i] < ema9[i]:
            sigs.append((i, 'SHORT'))
    return sigs


# ---------- Random baseline (per-dimension eligible universe) ----------

def measure_mfe_random_universe(df, eligible_mask, h1_long, h4_long, target_n,
                                  max_bars=8, seed=42, direction_by_trend=True):
    """Random entries within `eligible_mask`. Direction:
       - direction_by_trend=True: LONG if h1+h4 aligned LONG, SHORT if both SHORT.
       - direction_by_trend=False (no-filter case): random LONG/SHORT 50/50.
    """
    random.seed(seed)
    n = len(df)
    op = df['open'].values
    high = df['high'].values
    low = df['low'].values

    eligible_idx = np.where(eligible_mask)[0]
    eligible_idx = eligible_idx[(eligible_idx > 0) & (eligible_idx < n - max_bars - 1)]
    if len(eligible_idx) == 0:
        return []
    pool = eligible_idx.tolist()
    needed = min(target_n * 5, len(pool))
    sampled = sorted(random.sample(pool, needed))

    seq = []
    last_exit = -1
    for idx in sampled:
        if idx > last_exit:
            seq.append(idx)
            last_exit = idx + max_bars + 2
            if len(seq) >= target_n:
                break

    samples = []
    for idx in seq:
        ni = idx + 1
        if ni + max_bars >= n: continue
        if direction_by_trend:
            if h1_long[idx] and h4_long[idx]:
                direction = 'LONG'
            elif (not h1_long[idx]) and (not h4_long[idx]):
                direction = 'SHORT'
            else:
                direction = random.choice(['LONG', 'SHORT'])
        else:
            direction = random.choice(['LONG', 'SHORT'])
        entry = op[ni]
        end_idx = ni + max_bars
        if direction == 'LONG':
            mfe_idx = max(range(ni, end_idx + 1), key=lambda k: high[k])
            mae_idx = min(range(ni, end_idx + 1), key=lambda k: low[k])
            mfe_pct = (high[mfe_idx] / entry - 1) * 100
            mae_pct = (low[mae_idx] / entry - 1) * 100
        else:
            mfe_idx = min(range(ni, end_idx + 1), key=lambda k: low[k])
            mae_idx = max(range(ni, end_idx + 1), key=lambda k: high[k])
            mfe_pct = (1 - low[mfe_idx] / entry) * 100
            mae_pct = (1 - high[mae_idx] / entry) * 100
        samples.append({'mfe': mfe_pct, 'mae': mae_pct})
    return samples


# ---------- Main screening ----------

def screen_variant(df, h1_long, h4_long, eligible_for_random, signals,
                    horizons, label, direction_by_trend=True, friction=0.20):
    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400
    per_day = len(signals) / days if days else 0

    iso = {}
    for h in horizons:
        iso[f'h{h}'] = isolation_test(df, signals, h, friction)

    cand_mfe = measure_mfe_for_signals(df, signals, max_bars=horizons[1])  # mid horizon
    cand_stats = stats_mfe(cand_mfe, friction)

    if cand_stats is None:
        return {
            'variant': label, 'raw_signals': len(signals), 'per_day': round(per_day, 3),
            'isolation': iso, 'candidate_mfe': None, 'random_avg': None,
            'gate5_horizons_pos': 0, 'gate5_pass': False,
            'gate6_diff_mfe_p50_pp': 0.0, 'gate6_diff_pct_above': 0.0,
            'gate6_pass': False, 'asymmetry_sum_mfe_mae_p50': None,
            'verdict': 'NO_SIGNALS',
        }

    rnd_per_seed = []
    for seed in (42, 123, 456, 789, 1234):
        rnd = measure_mfe_random_universe(df, eligible_for_random, h1_long, h4_long,
                                            target_n=cand_stats['n'],
                                            max_bars=horizons[1], seed=seed,
                                            direction_by_trend=direction_by_trend)
        rnd_per_seed.append(stats_mfe(rnd, friction))
    rnd_p50 = sum(r['mfe_p50'] for r in rnd_per_seed if r) / max(1, sum(1 for r in rnd_per_seed if r))
    rnd_pct = sum(r['pct_mfe_gt_friction'] for r in rnd_per_seed if r) / max(1, sum(1 for r in rnd_per_seed if r))
    rnd_mae_p50 = sum(r['mae_p50'] for r in rnd_per_seed if r) / max(1, sum(1 for r in rnd_per_seed if r))

    diff_p50 = cand_stats['mfe_p50'] - rnd_p50
    diff_pct = cand_stats['pct_mfe_gt_friction'] - rnd_pct
    asymm_sum = cand_stats['mfe_p50'] + cand_stats['mae_p50']  # 신설 — 0 = symmetric, > 0 = favorable

    gate5_pos = sum(1 for r in iso.values() if r and r['gross_sum'] > 0)
    gate5_pass = gate5_pos >= 2
    gate6_pass = (diff_p50 >= 0.05) and (diff_pct >= 5.0)
    verdict = ('PASS' if (gate5_pass and gate6_pass) else
               'FAIL_G5_G6' if (not gate5_pass and not gate6_pass) else
               'FAIL_G5' if not gate5_pass else 'FAIL_G6')

    return {
        'variant': label,
        'raw_signals': len(signals),
        'per_day': round(per_day, 3),
        'isolation': iso,
        'candidate_mfe': cand_stats,
        'random_avg': {'mfe_p50': round(rnd_p50, 4), 'mae_p50': round(rnd_mae_p50, 4),
                        'pct_above_friction': round(rnd_pct, 2)},
        'random_per_seed': rnd_per_seed,
        'gate5_horizons_pos': gate5_pos,
        'gate5_pass': bool(gate5_pass),
        'gate6_diff_mfe_p50_pp': round(diff_p50, 4),
        'gate6_diff_pct_above': round(diff_pct, 2),
        'gate6_pass': bool(gate6_pass),
        'asymmetry_sum_mfe_mae_p50': round(asymm_sum, 4),
        'verdict': verdict,
    }


def main():
    print("Loading 15m + 1h data...")
    df15, h1, h4, valid15_filter, valid15_no_filter = prepare_15m_data()
    df1h, h1_1h, h4_1h, valid_1h = prepare_1h_data()
    print(f"  15m: {len(df15):,} bars (with filter: {int(valid15_filter.sum()):,})")
    print(f"  1h : {len(df1h):,} bars (valid: {int(valid_1h.sum()):,})\n")

    # eligible universes for random baseline
    elig_15m_with_filter = (h1 & h4 | (~h1) & (~h4)) & valid15_filter
    elig_15m_no_filter = valid15_no_filter
    elig_1h = (h1_1h & h4_1h | (~h1_1h) & (~h4_1h)) & valid_1h

    H_15M = [4, 8, 16]
    H_1H = [4, 8, 16]  # 1h: 4=4h, 8=8h, 16=16h

    results = []

    # ---------- D1: Timeframe (1h) ----------
    print("=" * 80)
    print("D1 — Timeframe shift (1h execution)")
    print("=" * 80)
    for label, fn in [
        ('D1.V1_mean_rev_1h', signals_v1_mean_rev),
        ('D1.V2_squeeze_1h', signals_v2_squeeze),
        ('D1.V3_momentum_1h', signals_v3_momentum),
        ('D1.V4_m1_minus_rsi_1h', signals_v4_m1_minus_rsi),
    ]:
        sigs = fn(df1h, h1_1h, h4_1h, valid_1h)
        r = screen_variant(df1h, h1_1h, h4_1h, elig_1h, sigs, H_1H, label, direction_by_trend=True)
        results.append(r)
        print(f"  {label}: signals={r['raw_signals']:>5} ({r['per_day']:.2f}/day) "
              f"G5={'P' if r['gate5_pass'] else 'F'}({r['gate5_horizons_pos']}/3) "
              f"Δp50={r['gate6_diff_mfe_p50_pp']:+.4f} Δ%>fr={r['gate6_diff_pct_above']:+.2f} "
              f"asym={r['asymmetry_sum_mfe_mae_p50']:+.4f} → {r['verdict']}")

    # ---------- D2: NEW signal classes (15m) ----------
    print("\n" + "=" * 80)
    print("D2 — NEW signal classes (BTC 15m)")
    print("=" * 80)
    for label, fn in [
        ('D2.V1_vol_regime', signals_d2_v1_vol_regime),
        ('D2.V2_range_break_retest', signals_d2_v2_range_break_retest),
        ('D2.V3_hl_pivot', signals_d2_v3_hl_pivot),
        ('D2.V4_pullback_to_1h_ema20', signals_d2_v4_pullback_to_1h_ema20),
    ]:
        sigs = fn(df15, h1, h4, valid15_filter)
        r = screen_variant(df15, h1, h4, elig_15m_with_filter, sigs, H_15M, label, direction_by_trend=True)
        results.append(r)
        print(f"  {label}: signals={r['raw_signals']:>5} ({r['per_day']:.2f}/day) "
              f"G5={'P' if r['gate5_pass'] else 'F'}({r['gate5_horizons_pos']}/3) "
              f"Δp50={r['gate6_diff_mfe_p50_pp']:+.4f} Δ%>fr={r['gate6_diff_pct_above']:+.2f} "
              f"asym={r['asymmetry_sum_mfe_mae_p50']:+.4f} → {r['verdict']}")

    # ---------- D3: No trend filter (15m) ----------
    print("\n" + "=" * 80)
    print("D3 — No trend filter (BTC 15m, V1-V4 unfiltered)")
    print("=" * 80)
    for label, fn in [
        ('D3.V1_mean_rev_nofilter', signals_d3_v1),
        ('D3.V2_squeeze_nofilter', signals_d3_v2),
        ('D3.V3_momentum_nofilter', signals_d3_v3),
        ('D3.V4_m1_minus_rsi_nofilter', signals_d3_v4),
    ]:
        sigs = fn(df15, valid15_no_filter)
        r = screen_variant(df15, h1, h4, elig_15m_no_filter, sigs, H_15M, label, direction_by_trend=False)
        results.append(r)
        print(f"  {label}: signals={r['raw_signals']:>5} ({r['per_day']:.2f}/day) "
              f"G5={'P' if r['gate5_pass'] else 'F'}({r['gate5_horizons_pos']}/3) "
              f"Δp50={r['gate6_diff_mfe_p50_pp']:+.4f} Δ%>fr={r['gate6_diff_pct_above']:+.2f} "
              f"asym={r['asymmetry_sum_mfe_mae_p50']:+.4f} → {r['verdict']}")

    # Round 1 cited (no rerun)
    round1_path = sorted(Path(ROOT / 'results').glob('m2_round1_screening_*.json'))
    round1_cited = []
    if round1_path:
        with open(round1_path[-1]) as f:
            r1 = json.load(f)
        for r in r1['results']:
            asym = round(r['candidate_mfe']['mfe_p50'] + r['candidate_mfe']['mae_p50'], 4) \
                if r['candidate_mfe'] else None
            round1_cited.append({
                'variant': f"R1.{r['variant']}",
                'raw_signals': r['raw_signals'], 'per_day': r['per_day'],
                'gate5_pass': r['gate5_pass'], 'gate5_horizons_pos': r['gate5_horizons_positive'],
                'gate6_diff_mfe_p50_pp': r['gate6_diff_mfe_p50_pp'],
                'gate6_diff_pct_above': r['gate6_diff_pct_above'],
                'gate6_pass': r['gate6_pass'],
                'asymmetry_sum_mfe_mae_p50': asym,
                'verdict': r['verdict'],
                'cited': True,
            })

    # 16-cell map
    print("\n" + "=" * 80)
    print("M2 ROUND 2 — 16-CELL MAP")
    print("=" * 80)
    print(f"{'cell':<32} {'sigs/day':>8} {'G5':>3} {'Δp50':>9} {'Δ%>fr':>8} {'asym':>9} {'verdict':>12}")
    all_cells = round1_cited + results
    for r in all_cells:
        cited_mark = '*' if r.get('cited') else ' '
        asym = r.get('asymmetry_sum_mfe_mae_p50')
        asym_s = f"{asym:+.4f}" if asym is not None else '   N/A'
        print(f"{cited_mark}{r['variant']:<31} {r['per_day']:>8.2f} "
              f"{'P' if r['gate5_pass'] else 'F':>3} "
              f"{r['gate6_diff_mfe_p50_pp']:>+9.4f} {r['gate6_diff_pct_above']:>+8.2f} "
              f"{asym_s:>9} {r['verdict']:>12}")
    print("(* = Round 1 cited)")

    n_pass_round2 = sum(1 for r in results if r['verdict'] == 'PASS')
    print(f"\nRound 2 PASS: {n_pass_round2}/12")
    print(f"Round 1 PASS (cited): {sum(1 for r in round1_cited if r['verdict'] == 'PASS')}/4")

    if n_pass_round2 == 0:
        print("→ 0 PASS. Map = deliverable. Round 3 (paradigm class shift) 사용자 결정 영역.")
    elif n_pass_round2 <= 3:
        print(f"→ {n_pass_round2} PASS. 모두 보고 → 사용자 picking.")
    else:
        print(f"→ {n_pass_round2} PASS (≥4) — threshold 의심, Δp50 ≥ 0.10pp strict re-run 필요.")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'spec_doc': 'claudedocs/m2_round2_variants.md',
        'frame': '16-cell map (12 NEW + 4 Round 1 cited)',
        'gate6_thresholds': {'mfe_p50_pp': 0.05, 'pct_above_friction_pp': 5.0},
        'horizons_15m': H_15M, 'horizons_1h': H_1H,
        'round1_cited': round1_cited,
        'round2_results': results,
        'round2_n_pass': n_pass_round2,
    }
    p = ROOT / 'results' / f'm2_round2_screening_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
