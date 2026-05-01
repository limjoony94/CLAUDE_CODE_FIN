"""R21b — M3 R21 Pattern Reversal SWEEP retry.

R21 surface result: avg_gross +0.010%/trade (양수, < friction floor 0.07%).
User critique 후 sweep retry — parameter potential 측정.

Pre-registered grid (FROZEN):
  volume_mult:        [1.0, 1.5, 2.0]
  lookback_extreme:   [10, 20]
  swing_lookback:     [5, 10]
  emergency_pct:      [1.0, 1.5]
  timeout_bars:       [12, 24, 48]
  min_bars_between:   [1, 5]
= 3×2×2×2×3×2 = 144 configs

Multi-stage: 50/25/25 IS/VAL/OOS, IS top-5 → VAL → OOS only val-PASS.

Pre-reg: claudedocs/r21b_pattern_reversal_sweep_prereg.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
sys.path.insert(0, str(ROOT / 'scripts' / 'strategy_lab'))

from m3_round21_pattern_structure import (
    detect_engulfing, detect_hammer, add_sma200_1h
)
from m3_round20_dynamic_scalping import prepare_5m_data
from mechanism_sweep_standard import MechanismSweep


DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

FRICTION_TP = 0.04  # maker TP RT
FRICTION_SL = 0.07  # taker SL RT


def find_recent_swing_low(lows, idx, lookback=10):
    start = max(0, idx - lookback)
    return float(np.min(lows[start:idx + 1])) if start <= idx else lows[idx]


def find_recent_swing_high(highs, idx, lookback=10):
    start = max(0, idx - lookback)
    return float(np.max(highs[start:idx + 1])) if start <= idx else highs[idx]


def find_resistance_levels(highs, idx, lookbacks=(20, 50)):
    return [float(np.max(highs[max(0, idx-lb):idx+1])) for lb in lookbacks]


def find_support_levels(lows, idx, lookbacks=(20, 50)):
    return [float(np.min(lows[max(0, idx-lb):idx+1])) for lb in lookbacks]


def entry_psi_prime_param(df, valid, params):
    """R21 ψ' entry with sweepable parameters."""
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    vol = df['volume'].values
    vol_sma = df['volume_sma20'].values
    sma_long = df['sma200_long'].fillna(False).astype(bool).values

    sigs = []
    lookback = params['lookback_extreme']
    vol_mult = params['volume_mult']
    for i in range(lookback + 2, n):
        if not valid[i]:
            continue
        if pd.isna(vol_sma[i]) or pd.isna(vol[i]):
            continue
        if vol[i] < vol_mult * vol_sma[i]:
            continue

        recent_lows = lo[i - lookback:i]
        recent_highs = hi[i - lookback:i]
        recent_min = np.min(recent_lows)
        recent_max = np.max(recent_highs)
        low_touched = (lo[i-1] == recent_min) or (lo[i-2] == recent_min)
        high_touched = (hi[i-1] == recent_max) or (hi[i-2] == recent_max)

        bull_eng, bear_eng = detect_engulfing(op, cl, i)
        hammer, star = detect_hammer(op, cl, hi, lo, i)

        if low_touched and (bull_eng or hammer) and sma_long[i]:
            sigs.append((i, 'LONG'))
        elif high_touched and (bear_eng or star) and (not sma_long[i]):
            sigs.append((i, 'SHORT'))
    return sigs


def run_bt_structural_param(df, sigs, params):
    """Structural exit with sweepable parameters."""
    n = len(df)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    timestamps = df['timestamp'].values
    sig_set = {idx: d for idx, d in sigs}

    swing_lb = params['swing_lookback']
    emerg_pct = params['emergency_pct']
    timeout_bars = params['timeout_bars']
    min_bars_between = params['min_bars_between']

    in_pos = False
    pdir = pentry = psl = ptp1 = ptp2 = pemerg = None
    pstart = None
    tp1_hit = False
    cooldown = 0
    trades = []
    i = 0
    while i < n:
        if in_pos:
            exit_price = None
            exit_reason = None

            if pdir == 'LONG' and lo[i] <= pemerg:
                exit_price, exit_reason = pemerg, 'EMERGENCY'
            elif pdir == 'SHORT' and hi[i] >= pemerg:
                exit_price, exit_reason = pemerg, 'EMERGENCY'

            if exit_price is None:
                if pdir == 'LONG' and lo[i] <= psl:
                    exit_price, exit_reason = psl, 'SL'
                elif pdir == 'SHORT' and hi[i] >= psl:
                    exit_price, exit_reason = psl, 'SL'

            if exit_price is None and not tp1_hit:
                if pdir == 'LONG' and hi[i] >= ptp1:
                    psl = max(psl, pentry * 1.0005)
                    tp1_hit = True
                elif pdir == 'SHORT' and lo[i] <= ptp1:
                    psl = min(psl, pentry * 0.9995)
                    tp1_hit = True

            if exit_price is None:
                if pdir == 'LONG' and hi[i] >= ptp2:
                    exit_price, exit_reason = ptp2, 'TP2'
                elif pdir == 'SHORT' and lo[i] <= ptp2:
                    exit_price, exit_reason = ptp2, 'TP2'

            held = i - pstart
            if exit_price is None and held >= timeout_bars:
                exit_price, exit_reason = cl[i], 'TIMEOUT'

            if exit_price is not None:
                gross = ((exit_price / pentry - 1) * 100) if pdir == 'LONG' else ((1 - exit_price / pentry) * 100)
                if exit_reason in ('TP1', 'TP2'):
                    fric = FRICTION_TP
                elif exit_reason in ('SL', 'EMERGENCY'):
                    fric = FRICTION_SL
                else:
                    fric = FRICTION_SL
                net = gross - fric
                trades.append({
                    'close_ts': timestamps[i],
                    'gross_pct': gross,
                    'net_pnl_pct': net,
                })
                in_pos = False
                cooldown = i + min_bars_between
                tp1_hit = False

        if not in_pos and i >= cooldown and i in sig_set:
            ni = i + 1
            if ni < n:
                pentry = op[ni]
                pdir = sig_set[i]
                if pdir == 'LONG':
                    swing_low = find_recent_swing_low(lo, i, lookback=swing_lb)
                    psl = swing_low * 0.9995
                    res_levels = find_resistance_levels(hi, i)
                    ptp1, ptp2 = res_levels[0], res_levels[1]
                    pemerg = pentry * (1 - emerg_pct / 100)
                    if not (ptp1 > pentry and ptp2 > pentry and psl < pentry):
                        i += 1
                        continue
                else:
                    swing_high = find_recent_swing_high(hi, i, lookback=swing_lb)
                    psl = swing_high * 1.0005
                    sup_levels = find_support_levels(lo, i)
                    ptp1, ptp2 = sup_levels[0], sup_levels[1]
                    pemerg = pentry * (1 + emerg_pct / 100)
                    if not (ptp1 < pentry and ptp2 < pentry and psl > pentry):
                        i += 1
                        continue
                pstart = ni
                in_pos = True
                tp1_hit = False
                i = ni
                continue
        i += 1
    return pd.DataFrame(trades)


# Cache prepared data (expensive)
_PREPARED = None


def get_prepared_data():
    global _PREPARED
    if _PREPARED is None:
        print('Preparing 5m+1h MTF data (one-time)...')
        df, h1, h4, valid = prepare_5m_data()
        df = add_sma200_1h(df)
        # R21 ψ' entry는 ETH cross-asset 미사용 → valid 재정의 (ETH 의존 제거)
        # ETH 데이터가 부분 range (2025-04-06~)이라 R21 sweep 적용 불가하던 것 fix
        valid = (
            (~pd.isna(df['atr14_5m']))
            & (~pd.isna(df['volume_sma20']))
            & (~pd.isna(df['high_20_prev']))
            & (~pd.isna(df['swing_low_10']))
            & (~df['sma200_long'].isna())
        ).values
        _PREPARED = (df, valid)
    return _PREPARED


class R21bSweep(MechanismSweep):
    label = 'r21b_pattern_reversal'
    mechanism_description = 'R21b — M3 R21 Pattern Reversal (parameter sweep)'

    PARAM_GRID = {
        'volume_mult':       [1.0, 1.5, 2.0],
        'lookback_extreme':  [10, 20],
        'swing_lookback':    [5, 10],
        'emergency_pct':     [1.0, 1.5],
        'timeout_bars':      [12, 24, 48],
        'min_bars_between':  [1, 5],
    }

    def build_trades(self, df_segment, config):
        # df_segment is split by timestamp; we need to re-extract from prepared
        df_full, valid_full = get_prepared_data()
        # Filter by ts range
        ts_min = df_segment['timestamp'].min()
        ts_max = df_segment['timestamp'].max()
        mask = (df_full['timestamp'] >= ts_min) & (df_full['timestamp'] <= ts_max)
        df_seg = df_full[mask].reset_index(drop=True)
        valid_seg = valid_full[mask.values]

        sigs = entry_psi_prime_param(df_seg, valid_seg, config)
        if not sigs:
            return pd.DataFrame()
        trades = run_bt_structural_param(df_seg, sigs, config)
        return trades


def main():
    df_full, valid = get_prepared_data()
    print(f'Data: {len(df_full):,} 5m bars, {df_full.timestamp.min()} → {df_full.timestamp.max()}')

    sweep = R21bSweep()
    result = sweep.run_sweep(df_full[['timestamp']].copy(), RESULTS)

    if not result.deployable:
        print('\n→ R21b sweep: 0 OOS-passing configs. Mechanism falsified across grid.')
    else:
        print(f'\n→ R21b sweep: {result.oos_pass_count} OOS-passing configs. PROMISING')


if __name__ == '__main__':
    main()
