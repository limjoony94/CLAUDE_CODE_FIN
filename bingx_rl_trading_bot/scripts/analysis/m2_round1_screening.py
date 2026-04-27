"""
M2 Round 1 — Pre-BT Variant Screening (Gate 5 + Gate 6 only)
=============================================================
4 variants × BTC 15m × 1h+4h trend filter constant.
- V1 Mean-reversion at extremes (RSI<25/>75 + bullish/bearish bar)
- V2 Volatility squeeze breakout (BB width min in past 50 + breakout)
- V3 Multi-bar momentum continuation (3 consecutive same-dir + 0.3% move)
- V4 M1-A minus RSI (body>0.4 + close vs EMA9)

Gates:
  - Gate 5: entry isolation (fixed-N-bar exit, 3 horizons: 4/8/16 bars)
  - Gate 6: random baseline comparison on same trend-filtered universe

NO Phase 3 BT — pre-BT screening only.
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# ---------- Indicators ----------

def compute_ema(values, period):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period:
        return out
    alpha = 2.0 / (period + 1)
    out[period - 1] = arr[:period].mean()
    for i in range(period, n):
        out[i] = arr[i] * alpha + out[i - 1] * (1 - alpha)
    return out


def compute_rsi(closes, period=14):
    arr = np.asarray(closes, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    diffs = np.diff(arr)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    out[period] = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        out[i] = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    return out


def compute_bb(closes, period=20, mult=2.0):
    arr = np.asarray(closes, dtype=float)
    n = len(arr)
    sma = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    width = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = arr[i - period + 1:i + 1]
        m = window.mean()
        s = window.std(ddof=0)
        sma[i] = m
        upper[i] = m + mult * s
        lower[i] = m - mult * s
        width[i] = (upper[i] - lower[i]) / m * 100  # in %
    return upper, lower, sma, width


def rolling_min(arr, lookback):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        out[i] = np.nanmin(arr[i - lookback + 1:i + 1])
    return out


# ---------- Data ----------

def load_ohlcv(path):
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    return df


def resample_to_4h(df_1h):
    df = df_1h.set_index('timestamp')
    df4 = df.resample('4h', origin='epoch', label='right', closed='right').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna(subset=['open']).reset_index()
    return df4


def merge_htf(df_target, df_htf, htf_minutes, cols):
    df_htf = df_htf.copy()
    df_htf['close_time'] = df_htf['timestamp'] + pd.Timedelta(minutes=htf_minutes)
    df_htf = df_htf[['close_time'] + cols].sort_values('close_time')
    return pd.merge_asof(df_target.sort_values('close_time'), df_htf,
                          on='close_time', direction='backward')


def prepare_data():
    df_15m = load_ohlcv(ROOT / 'data' / 'btc_15m_720days.csv')
    df_1h = load_ohlcv(ROOT / 'data' / 'btc_1h_720days.csv')
    df_4h = resample_to_4h(df_1h)

    # 15m indicators
    closes_15m = df_15m['close'].values
    df_15m['ema9'] = compute_ema(closes_15m, 9)
    df_15m['rsi14'] = compute_rsi(closes_15m, 14)
    upper, lower, _, width = compute_bb(closes_15m, 20, 2.0)
    df_15m['bb_upper'] = upper
    df_15m['bb_lower'] = lower
    df_15m['bb_width'] = width
    df_15m['bb_width_min50'] = rolling_min(df_15m['bb_width'].values, 50)
    df_15m['body_ratio'] = (df_15m['close'] - df_15m['open']).abs() / \
        (df_15m['high'] - df_15m['low']).replace(0, np.nan)

    # 1h trend filter
    df_1h['ema20'] = compute_ema(df_1h['close'].values, 20)
    df_1h['ema50'] = compute_ema(df_1h['close'].values, 50)
    df_1h['htf_long'] = df_1h['ema20'] > df_1h['ema50']

    # 4h trend filter
    df_4h['ema50'] = compute_ema(df_4h['close'].values, 50)
    df_4h['htf_long'] = df_4h['close'] > df_4h['ema50']

    # MTF causal merge
    df_15m['close_time'] = df_15m['timestamp'] + pd.Timedelta(minutes=15)
    df_15m = merge_htf(df_15m, df_1h.rename(columns={'htf_long': 'h1_long'}), 60, ['h1_long'])
    df_15m = merge_htf(df_15m, df_4h.rename(columns={'htf_long': 'h4_long'}), 240, ['h4_long'])
    df_15m = df_15m.sort_values('timestamp').reset_index(drop=True)

    h1_long = df_15m['h1_long'].fillna(False).astype(bool).values
    h4_long = df_15m['h4_long'].fillna(False).astype(bool).values

    valid_mask = ((~pd.isna(df_15m['rsi14'])) & (~pd.isna(df_15m['ema9']))
                   & (~pd.isna(df_15m['bb_width_min50']))
                   & (~df_15m['h1_long'].isna()) & (~df_15m['h4_long'].isna())).values

    return df_15m, h1_long, h4_long, valid_mask


# ---------- Variant entry rules ----------

def signals_v1_mean_rev(df, h1_long, h4_long, valid_mask):
    """V1: RSI <= 25 (LONG) / >= 75 (SHORT) + same-direction bar.
    Trend filter: 1h+4h aligned in trade direction."""
    n = len(df)
    rsi = df['rsi14'].values
    op = df['open'].values
    cl = df['close'].values
    signals = []  # list of (signal_idx, direction)
    for i in range(1, n):
        if not valid_mask[i]:
            continue
        # Use rsi at i (current bar) per spec
        if h1_long[i] and h4_long[i] and rsi[i] <= 25 and cl[i] > op[i]:
            signals.append((i, 'LONG'))
        elif (not h1_long[i]) and (not h4_long[i]) and rsi[i] >= 75 and cl[i] < op[i]:
            signals.append((i, 'SHORT'))
    return signals


def signals_v2_squeeze(df, h1_long, h4_long, valid_mask):
    """V2: BB width at lowest of past 50 (squeeze) + close breakout above/below previous bar's BB band."""
    n = len(df)
    cl = df['close'].values
    width = df['bb_width'].values
    width_min = df['bb_width_min50'].values
    upper = df['bb_upper'].values
    lower = df['bb_lower'].values

    signals = []
    for i in range(1, n):
        if not valid_mask[i]:
            continue
        # Squeeze condition: width[i-1] equals or near min of past 50 ending at i-1
        if pd.isna(width[i - 1]) or pd.isna(width_min[i - 1]):
            continue
        is_squeeze = (width[i - 1] <= width_min[i - 1] * 1.001)  # allow tiny tolerance
        if not is_squeeze:
            continue
        # Breakout
        if h1_long[i] and h4_long[i] and cl[i] > upper[i - 1]:
            signals.append((i, 'LONG'))
        elif (not h1_long[i]) and (not h4_long[i]) and cl[i] < lower[i - 1]:
            signals.append((i, 'SHORT'))
    return signals


def signals_v3_momentum(df, h1_long, h4_long, valid_mask):
    """V3: 3 consecutive same-direction bars + 0.3% total move."""
    n = len(df)
    op = df['open'].values
    cl = df['close'].values
    high = df['high'].values
    low = df['low'].values

    signals = []
    for i in range(3, n):
        if not valid_mask[i]:
            continue
        # 3 consecutive bullish (i-2, i-1, i) — checking previous 3 closed bars
        bull3 = cl[i] > op[i] and cl[i - 1] > op[i - 1] and cl[i - 2] > op[i - 2]
        bear3 = cl[i] < op[i] and cl[i - 1] < op[i - 1] and cl[i - 2] < op[i - 2]
        # 3-bar total move
        # LONG: high[i] - low[i-2] >= 0.3% of low[i-2]
        if bull3 and h1_long[i] and h4_long[i]:
            move_pct = (high[i] - low[i - 2]) / low[i - 2] * 100
            if move_pct >= 0.3:
                signals.append((i, 'LONG'))
        elif bear3 and (not h1_long[i]) and (not h4_long[i]):
            move_pct = (high[i - 2] - low[i]) / high[i - 2] * 100
            if move_pct >= 0.3:
                signals.append((i, 'SHORT'))
    return signals


def signals_v4_m1_minus_rsi(df, h1_long, h4_long, valid_mask):
    """V4: M1-A on 15m, minus RSI cross. body>0.4 + close vs EMA9."""
    n = len(df)
    cl = df['close'].values
    ema9 = df['ema9'].values
    body_ratio = df['body_ratio'].values

    signals = []
    for i in range(1, n):
        if not valid_mask[i]:
            continue
        if pd.isna(body_ratio[i]) or body_ratio[i] <= 0.4:
            continue
        if h1_long[i] and h4_long[i] and cl[i] > ema9[i]:
            signals.append((i, 'LONG'))
        elif (not h1_long[i]) and (not h4_long[i]) and cl[i] < ema9[i]:
            signals.append((i, 'SHORT'))
    return signals


# ---------- Sequencing (N=1, 2-bar cooldown) ----------

def apply_n1_sequencing(signals, max_bars, cooldown_bars=2):
    """Filter to N=1: each entry occupies max_bars + cooldown."""
    seq = []
    last_exit = -1
    for idx, dir_ in signals:
        if idx > last_exit:
            seq.append((idx, dir_))
            last_exit = idx + max_bars + cooldown_bars
    return seq


# ---------- Gate 5: entry isolation (fixed-N exit) ----------

def isolation_test(df, signals, exit_n_bars, friction=0.20):
    """Run signals with fixed-N-bar exit. Return summary."""
    op = df['open'].values
    cl = df['close'].values
    n = len(df)

    seq = apply_n1_sequencing(signals, exit_n_bars, cooldown_bars=2)
    grosses = []
    for idx, direction in seq:
        ni = idx + 1
        if ni + exit_n_bars >= n:
            continue
        entry = op[ni]
        exit_price = cl[ni + exit_n_bars]
        if direction == 'LONG':
            gross = (exit_price / entry - 1) * 100
        else:
            gross = (1 - exit_price / entry) * 100
        grosses.append(gross)

    if not grosses:
        return None
    n_trades = len(grosses)
    s_grosses = sorted(grosses)
    return {
        'n_trades': n_trades,
        'gross_sum': round(sum(grosses), 2),
        'gross_avg': round(sum(grosses) / n_trades, 4),
        'gross_wr_pct': round(100 * sum(1 for x in grosses if x > 0) / n_trades, 2),
        'net_avg': round(sum(grosses) / n_trades - friction, 4),
    }


# ---------- Gate 6: random baseline + MFE ----------

def measure_mfe_for_signals(df, signals, max_bars=8):
    """For each signal, compute MFE/MAE in next max_bars."""
    op = df['open'].values
    high = df['high'].values
    low = df['low'].values
    cl = df['close'].values
    n = len(df)

    seq = apply_n1_sequencing(signals, max_bars, cooldown_bars=2)
    samples = []
    for idx, direction in seq:
        ni = idx + 1
        if ni + max_bars >= n:
            continue
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


def measure_mfe_random(df, h1_long, h4_long, valid_mask, target_n, max_bars=8, seed=42):
    """Random entries on 1h+4h trend-aligned universe, direction = trend, N=1 + 2 cooldown."""
    random.seed(seed)
    n = len(df)
    op = df['open'].values
    high = df['high'].values
    low = df['low'].values

    eligible_long = h1_long & h4_long & valid_mask
    eligible_short = (~h1_long) & (~h4_long) & valid_mask
    eligible_idx = np.where(eligible_long | eligible_short)[0]
    eligible_idx = eligible_idx[(eligible_idx > 0) & (eligible_idx < n - max_bars - 1)]

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
        if ni + max_bars >= n:
            continue
        if h1_long[idx] and h4_long[idx]:
            direction = 'LONG'
        else:
            direction = 'SHORT'
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


def percentile(arr, p):
    s = sorted(arr)
    return s[int(p / 100 * len(s))]


def stats_mfe(samples, friction=0.20):
    if not samples:
        return None
    mfes = [s['mfe'] for s in samples]
    maes = [s['mae'] for s in samples]
    return {
        'n': len(samples),
        'mfe_p25': round(percentile(mfes, 25), 4),
        'mfe_p50': round(percentile(mfes, 50), 4),
        'mfe_p75': round(percentile(mfes, 75), 4),
        'mae_p25': round(percentile(maes, 25), 4),
        'mae_p50': round(percentile(maes, 50), 4),
        'pct_mfe_gt_friction': round(100 * sum(1 for x in mfes if x > friction) / len(mfes), 2),
    }


# ---------- Main ----------

def main():
    print("Loading + indicators (15m + 1h + 4h)...")
    df, h1_long, h4_long, valid_mask = prepare_data()
    print(f"  15m bars: {len(df):,} | valid: {int(valid_mask.sum()):,}\n")

    variants = [
        ('V1_mean_rev', signals_v1_mean_rev),
        ('V2_squeeze_breakout', signals_v2_squeeze),
        ('V3_multi_bar_momentum', signals_v3_momentum),
        ('V4_m1_minus_rsi', signals_v4_m1_minus_rsi),
    ]

    horizons = [4, 8, 16]  # 1h, 2h, 4h equivalent on 15m

    results = []
    for label, sig_func in variants:
        print(f"=== {label} ===")
        signals = sig_func(df, h1_long, h4_long, valid_mask)
        days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400
        per_day = len(signals) / days if days else 0
        print(f"  raw signals: {len(signals)} ({per_day:.2f}/day)")

        # Gate 5: entry isolation at 3 horizons
        iso = {}
        for h in horizons:
            r = isolation_test(df, signals, h, friction=0.20)
            iso[f'h{h}bars'] = r
            if r:
                print(f"  isolation N={h:>2}: n={r['n_trades']:>4} gross_sum={r['gross_sum']:+8.2f}% avg={r['gross_avg']:+.4f}% WR={r['gross_wr_pct']}%")

        # Gate 6: candidate MFE (use mid horizon = 8 bars = 2h)
        cand_mfe = measure_mfe_for_signals(df, signals, max_bars=8)
        cand_stats = stats_mfe(cand_mfe)
        print(f"  candidate MFE (8 bars): n={cand_stats['n']} P50={cand_stats['mfe_p50']:+.4f}% pct>0.20={cand_stats['pct_mfe_gt_friction']}%")

        # Random baseline (5 seeds × n_signals_post_seq)
        n_target = cand_stats['n']
        rnd_per_seed = []
        for seed in (42, 123, 456, 789, 1234):
            rnd_samples = measure_mfe_random(df, h1_long, h4_long, valid_mask,
                                              target_n=n_target, max_bars=8, seed=seed)
            rnd_per_seed.append(stats_mfe(rnd_samples))
        rnd_mfe_p50_avg = sum(r['mfe_p50'] for r in rnd_per_seed) / len(rnd_per_seed)
        rnd_pct_above_avg = sum(r['pct_mfe_gt_friction'] for r in rnd_per_seed) / len(rnd_per_seed)
        rnd_n_avg = sum(r['n'] for r in rnd_per_seed) / len(rnd_per_seed)
        print(f"  random (5 seeds avg): n~{rnd_n_avg:.0f} P50={rnd_mfe_p50_avg:+.4f}% pct>0.20={rnd_pct_above_avg:.2f}%")

        # Pass evaluation (Gate 6)
        diff_p50 = cand_stats['mfe_p50'] - rnd_mfe_p50_avg
        diff_pct = cand_stats['pct_mfe_gt_friction'] - rnd_pct_above_avg
        pass_p50 = diff_p50 >= 0.05
        pass_pct = diff_pct >= 5.0
        gate6_pass = pass_p50 and pass_pct

        # Gate 5 pass: gross_sum > 0 in ≥ 2 horizons
        gate5_horizons_pos = sum(1 for r in iso.values() if r and r['gross_sum'] > 0)
        gate5_pass = gate5_horizons_pos >= 2

        verdict = ('PASS' if (gate5_pass and gate6_pass) else
                   'FAIL_G5' if not gate5_pass else
                   'FAIL_G6' if not gate6_pass else 'FAIL')

        print(f"  Gate 5 (gross>0 in ≥2 horizons): {gate5_horizons_pos}/3 → {'PASS' if gate5_pass else 'FAIL'}")
        print(f"  Gate 6 (Δ MFE_P50 ≥ +0.05pp AND Δ %>0.20 ≥ +5pp): "
              f"Δp50={diff_p50:+.4f} Δpct={diff_pct:+.2f} → {'PASS' if gate6_pass else 'FAIL'}")
        print(f"  VERDICT: {verdict}\n")

        results.append({
            'variant': label,
            'raw_signals': len(signals),
            'per_day': round(per_day, 3),
            'isolation': iso,
            'candidate_mfe': cand_stats,
            'random_avg': {'mfe_p50': round(rnd_mfe_p50_avg, 4),
                           'pct_above_friction': round(rnd_pct_above_avg, 2)},
            'random_per_seed': rnd_per_seed,
            'gate5_horizons_positive': gate5_horizons_pos,
            'gate5_pass': bool(gate5_pass),
            'gate6_diff_mfe_p50_pp': round(diff_p50, 4),
            'gate6_diff_pct_above': round(diff_pct, 2),
            'gate6_pass': bool(gate6_pass),
            'verdict': verdict,
        })

    # Summary
    print("=" * 80)
    print("M2 ROUND 1 SCREENING SUMMARY")
    print("=" * 80)
    print(f"{'variant':<28} {'signals':>8} {'G5':>4} {'Δp50':>9} {'Δ%>fr':>8} {'G6':>4} {'verdict':>10}")
    for r in results:
        print(f"{r['variant']:<28} {r['raw_signals']:>8} {'P' if r['gate5_pass'] else 'F':>4} "
              f"{r['gate6_diff_mfe_p50_pp']:>+9.4f} {r['gate6_diff_pct_above']:>+8.2f} "
              f"{'P' if r['gate6_pass'] else 'F':>4} {r['verdict']:>10}")

    n_pass = sum(1 for r in results if r['verdict'] == 'PASS')
    print(f"\nTotal PASS: {n_pass}/4")
    if n_pass == 0:
        print("→ Zero PASS. Round 2 (timeframe shift or different signal classes) 사용자 결정.")
    elif n_pass == 1:
        winner = next(r for r in results if r['verdict'] == 'PASS')
        print(f"→ Single PASS: {winner['variant']}. 사용자 confirm 후 plan 시작 (Phase 2.5 gates).")
    else:
        print(f"→ Multiple PASS. 모두 보고, 사용자 선택 (assistant 자체 picking 금지).")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'spec_doc': 'claudedocs/m2_round1_variants.md',
        'asset': 'BTC/USDT', 'timeframe': '15m',
        'trend_filter': '1h EMA20>EMA50 AND 4h close>EMA50 (LONG); SHORT mirror',
        'horizons_bars': horizons,
        'friction_per_trade_pct': 0.20,
        'gate6_thresholds': {'mfe_p50_pp': 0.05, 'pct_above_friction_pp': 5.0},
        'results': results,
        'n_pass': n_pass,
    }
    p = ROOT / 'results' / f'm2_round1_screening_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
