"""
M2 Round 3 — Pre-BT Screening (3 data families × 3 signal classes)
===================================================================
Family A funding rate / Family B volume / Family C cross-asset
9 cells × Gate 5+6 + asymmetry sum column.

NO Phase 3 BT.
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from m2_round1_screening import (compute_ema, compute_rsi, load_ohlcv,
                                  resample_to_4h, merge_htf, apply_n1_sequencing,
                                  isolation_test, measure_mfe_for_signals,
                                  percentile, stats_mfe)
from m2_round2_screening import (rolling_max, rolling_min_arr, sma,
                                   measure_mfe_random_universe, screen_variant)


# ---------- Data preparation ----------

def prepare_btc_15m_with_filter():
    """BTC 15m + 1h trend + 4h trend."""
    df_15m = load_ohlcv(ROOT / 'data' / 'btc_15m_720days.csv')
    df_1h = load_ohlcv(ROOT / 'data' / 'btc_1h_720days.csv')
    df_4h = resample_to_4h(df_1h)

    closes = df_15m['close'].values
    highs = df_15m['high'].values
    lows = df_15m['low'].values
    df_15m['ema9'] = compute_ema(closes, 9)
    df_15m['rsi14'] = compute_rsi(closes, 14)
    df_15m['high_24'] = rolling_max(highs, 24)
    df_15m['low_24'] = rolling_min_arr(lows, 24)
    df_15m['high_24_prev'] = df_15m['high_24'].shift(1)
    df_15m['low_24_prev'] = df_15m['low_24'].shift(1)
    df_15m['vol_sma20'] = sma(df_15m['volume'].values, 20)
    df_15m['return'] = df_15m['close'].pct_change() * 100

    # rolling VWAP (96 bars = 24h on 15m)
    pv = (df_15m['close'] * df_15m['volume']).values
    v = df_15m['volume'].values
    pv_sum = pd.Series(pv).rolling(96, min_periods=96).sum().values
    v_sum = pd.Series(v).rolling(96, min_periods=96).sum().values
    df_15m['vwap'] = pv_sum / v_sum

    df_1h['ema20'] = compute_ema(df_1h['close'].values, 20)
    df_1h['ema50'] = compute_ema(df_1h['close'].values, 50)
    df_1h['htf_long'] = df_1h['ema20'] > df_1h['ema50']

    df_4h['ema50'] = compute_ema(df_4h['close'].values, 50)
    df_4h['htf_long'] = df_4h['close'] > df_4h['ema50']

    df_15m['close_time'] = df_15m['timestamp'] + pd.Timedelta(minutes=15)
    df_15m = merge_htf(df_15m, df_1h.rename(columns={'htf_long': 'h1_long'}), 60, ['h1_long'])
    df_15m = merge_htf(df_15m, df_4h.rename(columns={'htf_long': 'h4_long'}), 240, ['h4_long'])
    df_15m = df_15m.sort_values('timestamp').reset_index(drop=True)

    h1_long = df_15m['h1_long'].fillna(False).astype(bool).values
    h4_long = df_15m['h4_long'].fillna(False).astype(bool).values
    valid_mask = ((~pd.isna(df_15m['rsi14'])) & (~pd.isna(df_15m['vol_sma20']))
                   & (~pd.isna(df_15m['high_24_prev'])) & (~pd.isna(df_15m['vwap']))
                   & (~df_15m['h1_long'].isna()) & (~df_15m['h4_long'].isna())).values

    return df_15m, h1_long, h4_long, valid_mask


def prepare_funding_aligned(df_15m):
    """Load funding rates, align to 15m timestamps. Returns df_15m with funding columns added.
    Each 15m bar at time t gets funding rate from the most recent 8h funding period.
    """
    with open(ROOT / 'data' / 'bingx_funding_rates_full.json') as f:
        records = json.load(f)
    df_fund = pd.DataFrame(records)
    df_fund['timestamp'] = pd.to_datetime(df_fund['timestamp'], unit='ms', utc=True)
    df_fund = df_fund.sort_values('timestamp').reset_index(drop=True)
    df_fund['funding_pct'] = df_fund['fundingRate'].astype(float) * 100  # to %

    # Backward fill: each 15m bar gets the most recent funding rate
    df_15m = df_15m.copy()
    df_15m_sorted = df_15m.sort_values('timestamp')
    df_fund_sorted = df_fund[['timestamp', 'funding_pct']].sort_values('timestamp')
    merged = pd.merge_asof(df_15m_sorted, df_fund_sorted, on='timestamp', direction='backward')

    # 8 consecutive funding periods sum (for A.3)
    merged['funding_8sum'] = pd.Series(merged['funding_pct'].values).rolling(window=64, min_periods=64).sum().values
    # 보이는 bars × number of funding boundaries 안에 있어야 8 periods.
    # 8 periods = 64 hours = 64*4 = 256 fifteen-min bars. 윈도우 256 better.
    merged['funding_8sum'] = pd.Series(merged['funding_pct'].values).rolling(256, min_periods=256).sum().values
    merged['funding_8avg'] = merged['funding_8sum'] / 8.0  # but funding_pct is per-bar repeated, not per-period

    # Detect funding cross-zero: funding_pct value at a "fresh boundary" — track changes
    merged['funding_prev'] = merged['funding_pct'].shift(1)
    merged['funding_changed'] = (merged['funding_pct'] != merged['funding_prev']).astype(int)
    merged['funding_cross_pos'] = ((merged['funding_pct'] > 0) & (merged['funding_prev'] <= 0) &
                                    merged['funding_changed'].astype(bool)).astype(int)
    merged['funding_cross_neg'] = ((merged['funding_pct'] < 0) & (merged['funding_prev'] >= 0) &
                                    merged['funding_changed'].astype(bool)).astype(int)

    return merged.sort_values('timestamp').reset_index(drop=True)


def prepare_eth_aligned(df_btc_15m):
    """Load ETH 5m, resample to 15m, align with BTC 15m timestamps.
    Returns df with BTC OHLCV (open/high/low/close/volume) + ETH alignment columns."""
    df_eth_5m = load_ohlcv(ROOT / 'data' / 'eth_binance_5m.csv')
    df = df_eth_5m.set_index('timestamp')
    df_eth_15m = df.resample('15min', label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna(subset=['open']).reset_index()

    # Carry BTC OHLCV through (needed for isolation_test, MFE)
    df_btc = df_btc_15m[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df_btc['btc_close'] = df_btc_15m['close'].values
    df_btc['btc_return'] = df_btc_15m['return'].values
    df_btc = df_btc.sort_values('timestamp')
    df_eth_15m = df_eth_15m.rename(columns={'close': 'eth_close', 'open': 'eth_open',
                                              'high': 'eth_high', 'low': 'eth_low'})
    df_eth_15m['eth_return'] = df_eth_15m['eth_close'].pct_change() * 100
    df_eth_15m = df_eth_15m.sort_values('timestamp')

    merged = pd.merge_asof(df_btc, df_eth_15m[['timestamp', 'eth_close', 'eth_return']],
                           on='timestamp', direction='backward', tolerance=pd.Timedelta(minutes=15))
    # log ratio z-score
    merged['log_ratio'] = np.log(merged['btc_close'] / merged['eth_close'])
    merged['ratio_mean50'] = pd.Series(merged['log_ratio'].values).rolling(50, min_periods=50).mean().values
    merged['ratio_std50'] = pd.Series(merged['log_ratio'].values).rolling(50, min_periods=50).std().values
    merged['ratio_z'] = (merged['log_ratio'] - merged['ratio_mean50']) / merged['ratio_std50']

    # rolling correlation (50 bars)
    merged['corr50'] = pd.Series(merged['btc_return'].values).rolling(50, min_periods=50).corr(
        pd.Series(merged['eth_return'].values))

    return merged


# ---------- Family A signals ----------

def signals_a1_extreme_funding_fade(df_15m_funding, h1_long, h4_long, valid_mask):
    """LONG: funding < -0.04% (shorts crowded) AND RSI > 70 → reversion LONG.
    SHORT: funding > +0.04% AND RSI < 30 → reversion SHORT."""
    n = len(df_15m_funding)
    funding = df_15m_funding['funding_pct'].values
    rsi = df_15m_funding['rsi14'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if pd.isna(funding[i]) or pd.isna(rsi[i]): continue
        if funding[i] < -0.04 and rsi[i] > 70:
            sigs.append((i, 'LONG'))  # fade short crowd → LONG
        elif funding[i] > 0.04 and rsi[i] < 30:
            sigs.append((i, 'SHORT'))  # fade long crowd → SHORT
    return sigs


def signals_a2_funding_cross_zero(df, h1_long, h4_long, valid_mask):
    """LONG: funding crosses ≤0 → >0 in latest period AND trend LONG.
    SHORT mirror."""
    n = len(df)
    cross_pos = df['funding_cross_pos'].values
    cross_neg = df['funding_cross_neg'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if pd.isna(cross_pos[i]) or pd.isna(cross_neg[i]): continue
        if cross_pos[i] == 1 and h1_long[i] and h4_long[i]:
            sigs.append((i, 'LONG'))
        elif cross_neg[i] == 1 and (not h1_long[i]) and (not h4_long[i]):
            sigs.append((i, 'SHORT'))
    return sigs


def signals_a3_sustained_extreme(df, h1_long, h4_long, valid_mask):
    """LONG: 8 consecutive funding periods at ≥ +0.03% (overheat longs) → fade on RSI<30.
    Approximated by funding_8sum ≥ 8 × 0.03 = 0.24%.
    SHORT: 8sum ≤ -0.24% AND RSI > 70."""
    n = len(df)
    fsum = df['funding_8sum'].values
    rsi = df['rsi14'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if pd.isna(fsum[i]) or pd.isna(rsi[i]): continue
        if fsum[i] >= 0.24 and rsi[i] < 30:
            sigs.append((i, 'SHORT'))  # fade overheated longs
        elif fsum[i] <= -0.24 and rsi[i] > 70:
            sigs.append((i, 'LONG'))  # fade overheated shorts
    return sigs


# ---------- Family B signals ----------

def signals_b1_volume_spike_break(df, h1_long, h4_long, valid_mask):
    """LONG: close > 24-bar high AND volume > 2× SMA20 of volume + trend filter."""
    n = len(df)
    cl = df['close'].values
    high24_prev = df['high_24_prev'].values
    low24_prev = df['low_24_prev'].values
    vol = df['volume'].values
    vol_sma = df['vol_sma20'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if pd.isna(high24_prev[i]) or pd.isna(vol_sma[i]) or vol_sma[i] <= 0: continue
        is_break_up = cl[i] > high24_prev[i]
        is_break_down = cl[i] < low24_prev[i]
        is_vol_spike = vol[i] > 2.0 * vol_sma[i]
        if not is_vol_spike: continue
        if is_break_up and h1_long[i] and h4_long[i]:
            sigs.append((i, 'LONG'))
        elif is_break_down and (not h1_long[i]) and (not h4_long[i]):
            sigs.append((i, 'SHORT'))
    return sigs


def signals_b2_volume_divergence(df, h1_long, h4_long, valid_mask):
    """LONG: SHORT direction (fade higher-high with weakening volume).
    Wait — divergence is fade-the-trend: price higher-high but volume weakening → SHORT.
    SHORT direction signal."""
    n = len(df)
    cl = df['close'].values
    high = df['high'].values
    low = df['low'].values
    high24_prev = df['high_24_prev'].values
    low24_prev = df['low_24_prev'].values
    vol = df['volume'].values

    # 5-bar volume averages
    vol5_recent = pd.Series(vol).rolling(5, min_periods=5).mean().values
    vol5_prior = pd.Series(vol).rolling(5, min_periods=5).mean().shift(5).values

    sigs = []
    for i in range(10, n):
        if not valid_mask[i]: continue
        if pd.isna(high24_prev[i]) or pd.isna(vol5_recent[i]) or pd.isna(vol5_prior[i]): continue
        # Higher-high formed: high[i] > high24_prev[i]
        hh_formed = high[i] > high24_prev[i]
        ll_formed = low[i] < low24_prev[i]
        vol_weakening = vol5_recent[i] < vol5_prior[i] * 0.8  # 20% lower

        if hh_formed and vol_weakening:
            # Higher-high with weak volume → fade SHORT
            sigs.append((i, 'SHORT'))
        elif ll_formed and vol_weakening:
            # Lower-low with weak volume → fade LONG
            sigs.append((i, 'LONG'))
    return sigs


def signals_b3_vwap_bounce(df, h1_long, h4_long, valid_mask):
    """LONG: close pulls back to within 0.2% of VWAP + close > prev close (bounce) + trend LONG."""
    n = len(df)
    cl = df['close'].values
    vwap = df['vwap'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if pd.isna(vwap[i]): continue
        ratio = cl[i] / vwap[i]
        in_band = 0.998 <= ratio <= 1.002
        if not in_band: continue
        bounce = cl[i] > cl[i - 1]
        rejection = cl[i] < cl[i - 1]
        if h1_long[i] and h4_long[i] and bounce:
            sigs.append((i, 'LONG'))
        elif (not h1_long[i]) and (not h4_long[i]) and rejection:
            sigs.append((i, 'SHORT'))
    return sigs


# ---------- Family C signals ----------

def signals_c1_spread_mean_rev(df_cross, h1_long, h4_long, valid_mask):
    """LONG: BTC-ETH log ratio z < -2 (BTC underpriced) AND BTC trend LONG → BTC LONG.
    SHORT: z > +2 AND BTC trend SHORT → BTC SHORT."""
    n = len(df_cross)
    z = df_cross['ratio_z'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if pd.isna(z[i]): continue
        if z[i] < -2.0 and h1_long[i] and h4_long[i]:
            sigs.append((i, 'LONG'))
        elif z[i] > 2.0 and (not h1_long[i]) and (not h4_long[i]):
            sigs.append((i, 'SHORT'))
    return sigs


def signals_c2_correlation_breakdown(df_cross, h1_long, h4_long, valid_mask):
    """LONG: rolling 50-bar correlation drops < 0.5 + BTC trend LONG → BTC LONG."""
    n = len(df_cross)
    corr = df_cross['corr50'].values
    sigs = []
    for i in range(1, n):
        if not valid_mask[i]: continue
        if pd.isna(corr[i]): continue
        if corr[i] < 0.5:
            if h1_long[i] and h4_long[i]:
                sigs.append((i, 'LONG'))
            elif (not h1_long[i]) and (not h4_long[i]):
                sigs.append((i, 'SHORT'))
    return sigs


def signals_c3_eth_leads_btc(df_cross, h1_long, h4_long, valid_mask):
    """LONG: ETH prev bar return > +0.3% AND BTC prev bar return < +0.1% (BTC lagging) + BTC trend LONG."""
    n = len(df_cross)
    btc_ret = df_cross['btc_return'].values
    eth_ret = df_cross['eth_return'].values
    sigs = []
    for i in range(2, n):
        if not valid_mask[i]: continue
        if pd.isna(btc_ret[i - 1]) or pd.isna(eth_ret[i - 1]): continue
        eth_up = eth_ret[i - 1] > 0.3
        btc_lag_up = btc_ret[i - 1] < 0.1
        eth_down = eth_ret[i - 1] < -0.3
        btc_lag_down = btc_ret[i - 1] > -0.1
        if eth_up and btc_lag_up and h1_long[i] and h4_long[i]:
            sigs.append((i, 'LONG'))
        elif eth_down and btc_lag_down and (not h1_long[i]) and (not h4_long[i]):
            sigs.append((i, 'SHORT'))
    return sigs


# ---------- Main ----------

def main():
    print("Loading BTC 15m + 1h + 4h trend filter...")
    df_btc, h1, h4, valid_btc = prepare_btc_15m_with_filter()
    print(f"  BTC 15m: {len(df_btc):,} bars (valid: {int(valid_btc.sum()):,})")

    print("Loading + aligning funding rate...")
    df_funding = prepare_funding_aligned(df_btc)
    valid_funding = valid_btc & (~pd.isna(df_funding['funding_pct'])).values
    print(f"  Funding aligned: {int((~pd.isna(df_funding['funding_pct'])).sum()):,} bars with funding")

    print("Loading + aligning ETH 5m → 15m...")
    df_cross = prepare_eth_aligned(df_btc)
    # cross requires eth + btc + corr + z
    valid_cross = (valid_btc & (~pd.isna(df_cross['eth_close']))
                    & (~pd.isna(df_cross['ratio_z'])) & (~pd.isna(df_cross['corr50']))).values
    print(f"  Cross aligned: {int((~pd.isna(df_cross['eth_close'])).sum()):,} bars with ETH")

    H = [4, 8, 16]
    elig_btc = (h1 & h4 | (~h1) & (~h4)) & valid_btc
    elig_funding = (h1 & h4 | (~h1) & (~h4)) & valid_funding  # for A.2/A.3 trend-aligned
    elig_funding_no_filter = valid_funding  # for A.1 (no trend filter)
    elig_cross = (h1 & h4 | (~h1) & (~h4)) & valid_cross

    results = []

    # ---------- Family A ----------
    print("\n" + "=" * 80); print("Family A — Funding rate divergence"); print("=" * 80)
    for label, fn, df_use, valid, elig, dir_by_trend in [
        ('A.1_extreme_funding_fade', signals_a1_extreme_funding_fade, df_funding, valid_funding, elig_funding_no_filter, False),
        ('A.2_funding_cross_zero', signals_a2_funding_cross_zero, df_funding, valid_funding, elig_funding, True),
        ('A.3_sustained_extreme', signals_a3_sustained_extreme, df_funding, valid_funding, elig_funding_no_filter, False),
    ]:
        sigs = fn(df_use, h1, h4, valid)
        r = screen_variant(df_use, h1, h4, elig, sigs, H, label, direction_by_trend=dir_by_trend)
        results.append(r)
        print(f"  {label}: signals={r['raw_signals']:>5} ({r['per_day']:.2f}/day) "
              f"G5={'P' if r['gate5_pass'] else 'F'}({r['gate5_horizons_pos']}/3) "
              f"Δp50={r['gate6_diff_mfe_p50_pp']:+.4f} Δ%>fr={r['gate6_diff_pct_above']:+.2f} "
              f"asym={r['asymmetry_sum_mfe_mae_p50'] if r['asymmetry_sum_mfe_mae_p50'] else 'N/A'} → {r['verdict']}")

    # ---------- Family B ----------
    print("\n" + "=" * 80); print("Family B — Volume / volume delta"); print("=" * 80)
    for label, fn in [
        ('B.1_volume_spike_break', signals_b1_volume_spike_break),
        ('B.2_volume_divergence_fade', signals_b2_volume_divergence),
        ('B.3_vwap_bounce', signals_b3_vwap_bounce),
    ]:
        sigs = fn(df_btc, h1, h4, valid_btc)
        r = screen_variant(df_btc, h1, h4, elig_btc, sigs, H, label, direction_by_trend=True)
        results.append(r)
        print(f"  {label}: signals={r['raw_signals']:>5} ({r['per_day']:.2f}/day) "
              f"G5={'P' if r['gate5_pass'] else 'F'}({r['gate5_horizons_pos']}/3) "
              f"Δp50={r['gate6_diff_mfe_p50_pp']:+.4f} Δ%>fr={r['gate6_diff_pct_above']:+.2f} "
              f"asym={r['asymmetry_sum_mfe_mae_p50'] if r['asymmetry_sum_mfe_mae_p50'] else 'N/A'} → {r['verdict']}")

    # ---------- Family C ----------
    print("\n" + "=" * 80); print("Family C — Cross-asset BTC vs ETH"); print("=" * 80)
    for label, fn in [
        ('C.1_spread_mean_rev', signals_c1_spread_mean_rev),
        ('C.2_correlation_breakdown', signals_c2_correlation_breakdown),
        ('C.3_eth_leads_btc', signals_c3_eth_leads_btc),
    ]:
        sigs = fn(df_cross, h1, h4, valid_cross)
        r = screen_variant(df_cross, h1, h4, elig_cross, sigs, H, label, direction_by_trend=True)
        results.append(r)
        print(f"  {label}: signals={r['raw_signals']:>5} ({r['per_day']:.2f}/day) "
              f"G5={'P' if r['gate5_pass'] else 'F'}({r['gate5_horizons_pos']}/3) "
              f"Δp50={r['gate6_diff_mfe_p50_pp']:+.4f} Δ%>fr={r['gate6_diff_pct_above']:+.2f} "
              f"asym={r['asymmetry_sum_mfe_mae_p50'] if r['asymmetry_sum_mfe_mae_p50'] else 'N/A'} → {r['verdict']}")

    # Summary
    print("\n" + "=" * 80)
    print("M2 ROUND 3 — 9-CELL MAP")
    print("=" * 80)
    print(f"{'cell':<32} {'sigs/day':>8} {'G5':>3} {'Δp50':>9} {'Δ%>fr':>8} {'asym':>9} {'verdict':>14}")
    for r in results:
        asym = r.get('asymmetry_sum_mfe_mae_p50')
        asym_s = f"{asym:+.4f}" if asym is not None else "   N/A"
        print(f" {r['variant']:<31} {r['per_day']:>8.2f} {'P' if r['gate5_pass'] else 'F':>3} "
              f"{r['gate6_diff_mfe_p50_pp']:>+9.4f} {r['gate6_diff_pct_above']:>+8.2f} "
              f"{asym_s:>9} {r['verdict']:>14}")

    n_pass = sum(1 for r in results if r['verdict'] == 'PASS')
    print(f"\nRound 3 PASS: {n_pass}/9")
    if n_pass == 0:
        print("→ 0 PASS. Convergent evidence memo 작성 후 paradigm shift / portfolio / pause 사용자 결정.")
    elif n_pass <= 3:
        print(f"→ {n_pass} PASS. 모두 보고 → 사용자 picking.")
    else:
        print(f"→ {n_pass}+ PASS — threshold 의심, strict re-run 필요.")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'spec_doc': 'claudedocs/m2_round3_variants.md',
        'frame': '9-cell map (3 families × 3 signals)',
        'gate6_thresholds': {'mfe_p50_pp': 0.05, 'pct_above_friction_pp': 5.0},
        'data_coverage': {
            'btc_15m_days': int((df_btc['timestamp'].iloc[-1] - df_btc['timestamp'].iloc[0]).days),
            'funding_records': len(df_funding),
            'funding_aligned_bars': int((~pd.isna(df_funding['funding_pct'])).sum()),
            'eth_aligned_bars': int((~pd.isna(df_cross['eth_close'])).sum()),
        },
        'results': results,
        'n_pass': n_pass,
    }
    p = ROOT / 'results' / f'm2_round3_screening_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
