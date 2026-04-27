"""
Phase 2.6 Diagnostic: Max Favorable Excursion (MFE) Distribution
=================================================================
M1-A entry 후 24 bars (2h) 동안 maximum favorable excursion 측정.
"Perfect trail" 상한 결정 — exit mechanics 조정의 headroom 측정.

Question:
  - MFE 분포 if entry signals 평균 +X% favorable excursion 도달하는가?
  - Median, P75, P90 MFE는?
  - MFE → final close (TIMEOUT) 비율은? (excursion 후 retrace 정도)

해석:
  - MFE_p50 << friction 0.20% → 어떤 exit도 friction 못 넘음 → M1-A inviable
  - MFE_p50 > 0.5% → exit-tuning 개선 여지 큼 (현재 trail 0.07%만 capture)
  - MFE_p50 0.20-0.50% → marginal, hypothesis 가능
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from m1_bt_framework import M1ABot, prepare_data


def measure_mfe(df_5m, h1_long, h4_long, d3_long, d3_short, valid_mask, max_bars=24):
    """For each entry, measure MFE/MAE within max_bars."""
    n = len(df_5m)
    opens = df_5m['open'].values
    highs = df_5m['high'].values
    lows = df_5m['low'].values
    closes = df_5m['close'].values

    bot = M1ABot()
    in_pos = False
    pdir = None; pentry = None; pstart_idx = None
    cooldown_until = 0
    samples = []  # {direction, entry, mfe_pct, mae_pct, final_close_pct, mfe_bar, mae_bar}

    i = 0
    while i < n:
        if in_pos:
            held = i - pstart_idx
            if held >= max_bars:
                # Compute MFE/MAE/Final
                window_open_idx = pstart_idx
                window_end_idx = min(pstart_idx + max_bars, n - 1)
                if pdir == 'LONG':
                    mfe_bar = max(range(window_open_idx, window_end_idx + 1),
                                   key=lambda k: highs[k])
                    mae_bar = min(range(window_open_idx, window_end_idx + 1),
                                   key=lambda k: lows[k])
                    mfe_pct = (highs[mfe_bar] / pentry - 1) * 100
                    mae_pct = (lows[mae_bar] / pentry - 1) * 100  # negative
                    final_pct = (closes[window_end_idx] / pentry - 1) * 100
                else:
                    mfe_bar = min(range(window_open_idx, window_end_idx + 1),
                                   key=lambda k: lows[k])  # SHORT MFE = lowest low
                    mae_bar = max(range(window_open_idx, window_end_idx + 1),
                                   key=lambda k: highs[k])
                    mfe_pct = (1 - lows[mfe_bar] / pentry) * 100
                    mae_pct = (1 - highs[mae_bar] / pentry) * 100
                    final_pct = (1 - closes[window_end_idx] / pentry) * 100

                samples.append({
                    'direction': pdir,
                    'entry_bar': pstart_idx,
                    'mfe_pct': round(mfe_pct, 4),
                    'mae_pct': round(mae_pct, 4),
                    'final_pct': round(final_pct, 4),
                    'mfe_bar_offset': mfe_bar - pstart_idx,
                    'mae_bar_offset': mae_bar - pstart_idx,
                })
                in_pos = False
                cooldown_until = i + bot.min_bars_between

        if not in_pos and i >= cooldown_until:
            sig = bot.check_entry(i, df_5m, h1_long, h4_long, d3_long, d3_short, valid_mask)
            if sig:
                ni = i + 1
                if ni < n:
                    pentry = opens[ni]
                    pdir = sig['direction']
                    pstart_idx = ni
                    in_pos = True
                    i = ni
                    continue
        i += 1
    return samples


def percentile(arr, pct):
    arr_sorted = sorted(arr)
    return arr_sorted[int(pct/100 * len(arr_sorted))]


def main():
    print("Loading + indicators...")
    df_5m, h1_long, h4_long, d3_long, d3_short, valid_mask = prepare_data(
        ROOT / 'data' / 'btc_5m_720days_binance.csv',
        ROOT / 'data' / 'btc_15m_720days.csv',
        ROOT / 'data' / 'btc_1h_720days.csv',
    )
    print(f"  5m: {len(df_5m):,}, valid: {int(valid_mask.sum()):,}\n")

    print("Measuring MFE distribution (24-bar windows)...")
    samples = measure_mfe(df_5m, h1_long, h4_long, d3_long, d3_short, valid_mask, max_bars=24)
    print(f"  n={len(samples)} samples\n")

    if not samples:
        print("No samples")
        return

    mfes = [s['mfe_pct'] for s in samples]
    maes = [s['mae_pct'] for s in samples]
    finals = [s['final_pct'] for s in samples]

    print("=== MFE distribution (peak favorable %, 24-bar window) ===")
    print(f"  mean : {sum(mfes)/len(mfes):+.4f}%")
    print(f"  P10  : {percentile(mfes, 10):+.4f}%")
    print(f"  P25  : {percentile(mfes, 25):+.4f}%")
    print(f"  P50  : {percentile(mfes, 50):+.4f}%  ← median")
    print(f"  P75  : {percentile(mfes, 75):+.4f}%")
    print(f"  P90  : {percentile(mfes, 90):+.4f}%")
    print(f"  max  : {max(mfes):+.4f}%")

    print("\n=== MAE distribution (worst adverse %, 24-bar window) ===")
    print(f"  mean : {sum(maes)/len(maes):+.4f}%")
    print(f"  P10  : {percentile(maes, 10):+.4f}%")
    print(f"  P25  : {percentile(maes, 25):+.4f}%")
    print(f"  P50  : {percentile(maes, 50):+.4f}%  ← median")
    print(f"  P75  : {percentile(maes, 75):+.4f}%")
    print(f"  P90  : {percentile(maes, 90):+.4f}%")
    print(f"  min  : {min(maes):+.4f}%")

    print("\n=== Final close (24-bar TIMEOUT, what BT random-exit captured) ===")
    print(f"  mean : {sum(finals)/len(finals):+.4f}%")
    print(f"  P50  : {percentile(finals, 50):+.4f}%")

    # Hypothesis support
    pct_mfe_above_friction = 100 * sum(1 for x in mfes if x > 0.20) / len(mfes)
    pct_mfe_above_double_friction = 100 * sum(1 for x in mfes if x > 0.40) / len(mfes)
    print(f"\n=== Hypothesis support ===")
    print(f"  % entries with MFE > friction 0.20%   : {pct_mfe_above_friction:.1f}%")
    print(f"  % entries with MFE > 2×friction 0.40% : {pct_mfe_above_double_friction:.1f}%")

    # MFE bar offset distribution (when does peak happen?)
    mfe_offsets = [s['mfe_bar_offset'] for s in samples]
    print(f"\n=== Peak timing (which bar in 24-bar window?) ===")
    print(f"  mean offset : {sum(mfe_offsets)/len(mfe_offsets):.1f} bars")
    print(f"  P50 offset  : {percentile(mfe_offsets, 50)} bars")

    print(f"\n=== Verdict ===")
    mfe_p50 = percentile(mfes, 50)
    if mfe_p50 < 0.20:
        print(f"  MFE P50 = {mfe_p50:.2f}% < friction 0.20%")
        print(f"  → Half of entries never reach friction even at peak.")
        print(f"  → No exit mechanic can extract sustainable edge.")
        print(f"  → M1-A entry signal STRUCTURALLY INSUFFICIENT.")
    elif mfe_p50 < 0.40:
        print(f"  MFE P50 = {mfe_p50:.2f}% (0.20–0.40%, marginal)")
        print(f"  → Half reach above friction but timing+retracement makes capture hard.")
        print(f"  → Conservative trail with activation might capture ~0.10–0.20% net.")
        print(f"  → Marginal, single hypothesis 가능 but expected gain limited.")
    else:
        print(f"  MFE P50 = {mfe_p50:.2f}% > 2×friction")
        print(f"  → Significant headroom. Exit-tuning has real potential.")
        print(f"  → ONE hypothesis: trail activation at MFE-derived threshold.")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'n_samples': len(samples),
        'mfe': {'mean': sum(mfes)/len(mfes), 'p10': percentile(mfes, 10),
                'p25': percentile(mfes, 25), 'p50': percentile(mfes, 50),
                'p75': percentile(mfes, 75), 'p90': percentile(mfes, 90),
                'max': max(mfes)},
        'mae': {'mean': sum(maes)/len(maes), 'p10': percentile(maes, 10),
                'p25': percentile(maes, 25), 'p50': percentile(maes, 50),
                'p75': percentile(maes, 75), 'p90': percentile(maes, 90),
                'min': min(maes)},
        'final': {'mean': sum(finals)/len(finals), 'p50': percentile(finals, 50)},
        'mfe_offset': {'mean': sum(mfe_offsets)/len(mfe_offsets),
                       'p50': percentile(mfe_offsets, 50)},
        'pct_mfe_gt_friction': pct_mfe_above_friction,
        'pct_mfe_gt_2x_friction': pct_mfe_above_double_friction,
    }
    p = ROOT / 'results' / f'm1_mfe_dist_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
