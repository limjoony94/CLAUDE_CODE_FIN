#!/usr/bin/env python3
"""
Mechanism Portfolio Study — 10개 활성 포지션 관리 메커니즘 통합 분석

연구 질문: 각 메커니즘의 개별 기여도와 메커니즘 간 시너지/중복 관계는?

10개 활성 메커니즘 (v1.41.0):
  Entry Guards:   G1 DirCap(7), G2 Momentum, G5 AggRisk
  Sizing/Lev:     M1 MDD, M2 Regime(×0.3), M4 AdaptiveLev(1-3x), M5 ATR
  Exit Mgmt:      P1 Timeout(864), P2 EarlyExit(3x,0.3%), P3 CascadeSL(75%)

방법론:
  Phase 1: Full Stack baseline (all 10 ON)
  Phase 2: Individual Ablation (remove 1, others ON) — 개별 가치
  Phase 3: Individual Additive (add 1, others OFF) — 독립 효과
  Phase 4: Pairwise Synergy (~15 pairs) — 상호작용
  Phase 5: WF Validation (3-fold Expanding Window)

Standard Research Protocol:
  - Entry: 신호 다음 봉 Open
  - Exit: Intrabar High/Low (distance-based)
  - Sizing: Compound (복리)
  - Fee: 0.10% × LEVERAGE=3 = 0.30% round-trip
  - Slippage: 0.02% buffer
  - WF: 3-fold Expanding Window
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ============================================================
# Constants
# ============================================================
FEE_PCT = 0.10
LEVERAGE = 3
SLIPPAGE_BUFFER = 0.02
N_SLOTS = 9

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'mechanism_portfolio_study.json')

# Mechanism names (for display)
MECHANISM_NAMES = {
    'dir_cap':           'G1 DirCap(7)',
    'momentum_guard':    'G2 Momentum',
    'agg_risk_cap':      'G5 AggRisk',
    'mdd_sizing':        'M1 MDD',
    'regime_sizing':     'M2 Regime(×0.3)',
    'adaptive_leverage': 'M4 AdaptiveLev',
    'atr_scaling':       'M5 ATR',
    'timeout':           'P1 Timeout(864)',
    'early_exit':        'P2 EarlyExit',
    'cascade_sl':        'P3 CascadeSL',
}

MECHANISM_KEYS = list(MECHANISM_NAMES.keys())

# Full stack config (all ON)
FULL_STACK = {
    'dir_cap': 7,
    'momentum_guard': True,
    'momentum_lookback': 6,
    'momentum_threshold': 1.0,
    'momentum_cooldown': 6,
    'agg_risk_cap': True,
    'agg_risk_counter': 3.0,
    'agg_risk_with': 7.0,
    'mdd_sizing': True,
    'mdd_full_below': 3.0,
    'mdd_min_above': 15.0,
    'mdd_min_scale': 0.25,
    'regime_sizing': True,
    'regime_mult': 0.3,
    'adaptive_leverage': True,
    'adap_lev_window': 12,
    'adap_lev_expected_wr': 0.732,
    'adap_lev_ref_edge': 0.00126,
    'adap_lev_min': 1.0,
    'adap_lev_max': 3.0,
    'atr_scaling': True,
    'atr_clamp_lo': 0.6,
    'atr_clamp_hi': 1.7,
    'timeout': 864,
    'early_exit': True,
    'early_confirm': 3,
    'early_min_profit': 0.3,
    'cascade_sl': True,
    'cascade_tighten_pct': 75,
}

# Minimal config (all OFF)
MINIMAL = {
    'dir_cap': 999,
    'momentum_guard': False,
    'momentum_lookback': 6,
    'momentum_threshold': 1.0,
    'momentum_cooldown': 6,
    'agg_risk_cap': False,
    'agg_risk_counter': 3.0,
    'agg_risk_with': 7.0,
    'mdd_sizing': False,
    'mdd_full_below': 3.0,
    'mdd_min_above': 15.0,
    'mdd_min_scale': 0.25,
    'regime_sizing': False,
    'regime_mult': 1.0,
    'adaptive_leverage': False,
    'adap_lev_window': 12,
    'adap_lev_expected_wr': 0.732,
    'adap_lev_ref_edge': 0.00126,
    'adap_lev_min': 1.0,
    'adap_lev_max': 3.0,
    'atr_scaling': False,
    'atr_clamp_lo': 0.6,
    'atr_clamp_hi': 1.7,
    'timeout': 0,
    'early_exit': False,
    'early_confirm': 3,
    'early_min_profit': 0.3,
    'cascade_sl': False,
    'cascade_tighten_pct': 75,
}


def _ablation_config(key):
    """Full stack with one mechanism disabled."""
    cfg = deepcopy(FULL_STACK)
    if key == 'dir_cap':
        cfg['dir_cap'] = 999
    elif key == 'momentum_guard':
        cfg['momentum_guard'] = False
    elif key == 'agg_risk_cap':
        cfg['agg_risk_cap'] = False
    elif key == 'mdd_sizing':
        cfg['mdd_sizing'] = False
    elif key == 'regime_sizing':
        cfg['regime_sizing'] = False
        cfg['regime_mult'] = 1.0
    elif key == 'adaptive_leverage':
        cfg['adaptive_leverage'] = False
    elif key == 'atr_scaling':
        cfg['atr_scaling'] = False
    elif key == 'timeout':
        cfg['timeout'] = 0
    elif key == 'early_exit':
        cfg['early_exit'] = False
    elif key == 'cascade_sl':
        cfg['cascade_sl'] = False
    return cfg


def _additive_config(key):
    """Minimal + one mechanism enabled."""
    cfg = deepcopy(MINIMAL)
    if key == 'dir_cap':
        cfg['dir_cap'] = 7
    elif key == 'momentum_guard':
        cfg['momentum_guard'] = True
    elif key == 'agg_risk_cap':
        cfg['agg_risk_cap'] = True
    elif key == 'mdd_sizing':
        cfg['mdd_sizing'] = True
    elif key == 'regime_sizing':
        cfg['regime_sizing'] = True
        cfg['regime_mult'] = 0.3
    elif key == 'adaptive_leverage':
        cfg['adaptive_leverage'] = True
    elif key == 'atr_scaling':
        cfg['atr_scaling'] = True
    elif key == 'timeout':
        cfg['timeout'] = 864
    elif key == 'early_exit':
        cfg['early_exit'] = True
    elif key == 'cascade_sl':
        cfg['cascade_sl'] = True
    return cfg


def _pair_ablation_config(key_a, key_b):
    """Full stack with two mechanisms disabled."""
    cfg = _ablation_config(key_a)
    # Apply second ablation
    if key_b == 'dir_cap':
        cfg['dir_cap'] = 999
    elif key_b == 'momentum_guard':
        cfg['momentum_guard'] = False
    elif key_b == 'agg_risk_cap':
        cfg['agg_risk_cap'] = False
    elif key_b == 'mdd_sizing':
        cfg['mdd_sizing'] = False
    elif key_b == 'regime_sizing':
        cfg['regime_sizing'] = False
        cfg['regime_mult'] = 1.0
    elif key_b == 'adaptive_leverage':
        cfg['adaptive_leverage'] = False
    elif key_b == 'atr_scaling':
        cfg['atr_scaling'] = False
    elif key_b == 'timeout':
        cfg['timeout'] = 0
    elif key_b == 'early_exit':
        cfg['early_exit'] = False
    elif key_b == 'cascade_sl':
        cfg['cascade_sl'] = False
    return cfg


# ============================================================
# Data loading
# ============================================================
def load_data():
    df = pd.read_csv(DATA_FILE)
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW).mean()
    types = []
    for i, row in df.iterrows():
        avg_b = df.at[i, 'avg_body']
        if pd.isna(avg_b):
            avg_b = df['body'].iloc[:max(1, i)].mean()
        ct = classify_candle(row, avg_b)
        types.append(ct.value)
    df['candle_type'] = types
    return df, types


def load_patterns():
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    directions = data.get('patterns', {})
    tpsl = data.get('patterns_tpsl', {})
    result = {}
    for direction_key, dir_label in [('long', 'LONG'), ('short', 'SHORT')]:
        for pat_name in directions.get(direction_key, []):
            tp, sl = tpsl.get(pat_name, [1.0, 1.0])
            result[pat_name] = {'direction': dir_label, 'tp_pct': tp, 'sl_pct': sl}
    return result


def build_signal_index(types, n):
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def compute_atr_ratio(highs, lows, closes, atr_period=14, window=576):
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period-1] = tr[:atr_period].mean()
        for i in range(atr_period, n):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    med = pd.Series(atr).rolling(window, min_periods=window).median().values
    ratio = np.full(n, np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio


def compute_ema_slope(closes, period=20, lookback=5):
    n = len(closes)
    ema = pd.Series(closes).ewm(span=period, adjust=False).mean().values
    slope = np.full(n, 0.0)
    for i in range(lookback, n):
        if ema[i - lookback] > 0:
            slope[i] = (ema[i] / ema[i - lookback] - 1) * 100
    return slope


def find_neutral_window(closes, tol_pct=1.0, min_days=90, bars_per_day=288):
    n = len(closes)
    min_bars = min_days * bars_per_day
    best_len, best_start, best_end = 0, 0, 0
    for i in range(0, n - min_bars, bars_per_day):
        sp = closes[i]
        lo, hi = sp * (1 - tol_pct/100), sp * (1 + tol_pct/100)
        for j in range(n-1, i + min_bars - 1, -bars_per_day):
            if lo <= closes[j] <= hi:
                length = j - i
                if length > best_len:
                    best_len, best_start, best_end = length, i, j
                break
    if best_len == 0:
        return None
    return best_start, best_end


# ============================================================
# Core simulator — all 10 mechanisms toggleable
# ============================================================
def portfolio_npos(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar, config,
):
    """N-pos portfolio simulator with all 10 mechanisms toggleable.

    Returns: (trades_list, final_equity, max_dd, stats_dict)
    """
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS

    # Unpack config
    dir_cap = config.get('dir_cap', 999)
    use_momentum = config.get('momentum_guard', False)
    mom_lookback = config.get('momentum_lookback', 6)
    mom_threshold = config.get('momentum_threshold', 1.0)
    mom_cooldown = config.get('momentum_cooldown', 6)
    use_agg_risk = config.get('agg_risk_cap', False)
    agg_counter = config.get('agg_risk_counter', 3.0)
    agg_with = config.get('agg_risk_with', 7.0)
    use_mdd = config.get('mdd_sizing', False)
    mdd_full_below = config.get('mdd_full_below', 3.0)
    mdd_min_above = config.get('mdd_min_above', 15.0)
    mdd_min_scale = config.get('mdd_min_scale', 0.25)
    use_regime = config.get('regime_sizing', False)
    regime_mult = config.get('regime_mult', 0.3)
    use_adaptive_lev = config.get('adaptive_leverage', False)
    adap_window = config.get('adap_lev_window', 12)
    adap_wr = config.get('adap_lev_expected_wr', 0.732)
    adap_edge = config.get('adap_lev_ref_edge', 0.00126)
    adap_min = config.get('adap_lev_min', 1.0)
    adap_max = config.get('adap_lev_max', 3.0)
    use_atr = config.get('atr_scaling', False)
    clamp_lo = config.get('atr_clamp_lo', 0.6)
    clamp_hi = config.get('atr_clamp_hi', 1.7)
    timeout_bars = config.get('timeout', 0)
    use_early = config.get('early_exit', False)
    early_confirm = config.get('early_confirm', 3)
    early_min_profit = config.get('early_min_profit', 0.3)
    use_cascade = config.get('cascade_sl', False)
    cascade_keep = 1.0 - config.get('cascade_tighten_pct', 75) / 100.0

    # State
    positions = []
    trades = []
    recent_trades_pnl = []  # for adaptive leverage
    equity = 100.0
    peak_equity = 100.0
    max_dd = 0.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}
    blocked = {'dir_cap': 0, 'momentum': 0, 'agg_risk': 0, 'dup_pat': 0, 'max_pos': 0}
    early_count = 0
    cascade_count = 0
    timeout_count = 0

    # Index signals by bar for O(1) lookup
    sig_by_bar = {}
    for s_bar, pat, direction, tp, sl in signal_tuples:
        if start_bar <= s_bar < end_bar:
            sig_by_bar.setdefault(s_bar, []).append((pat, direction, tp, sl))

    for bar in range(start_bar, end_bar):
        if bar >= n_bars - 1:
            break

        # ====================================================
        # 1. CHECK EXITS for open positions
        # ====================================================
        closed_slots = set()
        sl_exits = []
        bar_pnl_sum = 0.0

        h_bar = highs[bar] if bar < n_bars else 0
        l_bar = lows[bar] if bar < n_bars else 0
        o_bar = opens[bar] if bar < n_bars else 0

        for pos in positions:
            entry_bar = pos['entry_bar']
            if bar < entry_bar:
                continue
            entry_p = pos['entry_price']
            if entry_p <= 0:
                continue

            direction = pos['direction']
            eff_tp = pos['eff_tp_pct']
            eff_sl = pos['eff_sl_pct']
            lev = pos['leverage']
            sm = pos['size_mult']
            hold = bar - entry_bar

            exit_price = None
            reason = None

            # 1a. Timeout
            if timeout_bars > 0 and hold >= timeout_bars:
                # DROP (no PnL)
                closed_slots.add(pos['slot'])
                timeout_count += 1
                continue

            # 1b. Early exit
            if use_early and hold >= early_confirm:
                exit_types = ['BD'] if direction == 'LONG' else ['BU']
                candle_ok = True
                for k in range(early_confirm):
                    cb = bar - 1 - k  # check completed candles ending at bar-1
                    if cb < 0 or cb >= len(type_codes) or type_codes[cb] not in exit_types:
                        candle_ok = False
                        break
                if candle_ok:
                    cp = closes[bar - 1] if bar - 1 < n_bars else entry_p
                    if direction == 'LONG':
                        unr = (cp / entry_p - 1) * 100 * lev
                    else:
                        unr = (1 - cp / entry_p) * 100 * lev
                    if unr >= early_min_profit:
                        exit_price = cp
                        reason = 'EARLY'

            # 1c. TP/SL intrabar
            if reason is None:
                if direction == 'LONG':
                    tp_p = entry_p * (1 + eff_tp / 100)
                    sl_p = entry_p * (1 - eff_sl / 100)
                    hit_tp = h_bar >= tp_p
                    hit_sl = l_bar <= sl_p
                else:
                    tp_p = entry_p * (1 - eff_tp / 100)
                    sl_p = entry_p * (1 + eff_sl / 100)
                    hit_tp = l_bar <= tp_p
                    hit_sl = h_bar >= sl_p

                if hit_tp and hit_sl:
                    tp_d = abs(tp_p - o_bar)
                    sl_d = abs(sl_p - o_bar)
                    if tp_d <= sl_d:
                        exit_price, reason = tp_p, 'TP'
                    else:
                        exit_price, reason = sl_p, 'SL'
                elif hit_tp:
                    exit_price, reason = tp_p, 'TP'
                elif hit_sl:
                    exit_price, reason = sl_p, 'SL'

            if reason is None:
                continue

            # Calculate PnL
            if direction == 'LONG':
                pnl = (exit_price / entry_p - 1) * 100 * lev
            else:
                pnl = (1 - exit_price / entry_p) * 100 * lev
            pnl -= fee

            pnl_portfolio = pnl * (size_pct / 100) * sm
            trades.append({
                'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
                'reason': reason, 'pattern': pos['pattern'],
                'direction': direction, 'size_mult': sm,
                'pnl_portfolio': pnl_portfolio, 'leverage': lev,
            })
            recent_trades_pnl.append(pnl)
            closed_slots.add(pos['slot'])
            bar_pnl_sum += pnl_portfolio

            if reason == 'EARLY':
                early_count += 1
            if reason == 'SL':
                sl_exits.append(pos)

        # Remove closed positions
        positions = [p for p in positions if p['slot'] not in closed_slots]

        # 1d. Cascade SL tightening
        if use_cascade and sl_exits:
            for sl_pos in sl_exits:
                sl_dir = sl_pos['direction']
                tightened = 0
                for pos in positions:
                    if pos['direction'] == sl_dir:
                        pos['eff_sl_pct'] *= cascade_keep
                        tightened += 1
                if tightened > 0:
                    cascade_count += 1

        # Update equity
        equity *= (1 + bar_pnl_sum / 100)
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

        # ====================================================
        # 2. MOMENTUM GUARD state update
        # ====================================================
        if use_momentum and bar >= mom_lookback:
            pa = closes[bar - mom_lookback]
            if pa > 0:
                pct = (closes[bar] / pa - 1) * 100
                if pct > mom_threshold:
                    mom_pause_until['SHORT'] = max(mom_pause_until['SHORT'], bar + mom_cooldown)
                elif pct < -mom_threshold:
                    mom_pause_until['LONG'] = max(mom_pause_until['LONG'], bar + mom_cooldown)

        # ====================================================
        # 3. PROCESS NEW ENTRIES
        # ====================================================
        if bar not in sig_by_bar:
            continue

        for pat, direction, tp_pct, sl_pct in sig_by_bar[bar]:
            # Max positions
            if len(positions) >= N_SLOTS:
                blocked['max_pos'] += 1
                continue

            # Direction cap
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= dir_cap:
                blocked['dir_cap'] += 1
                continue

            # Duplicate pattern
            if any(p['pattern'] == pat for p in positions):
                blocked['dup_pat'] += 1
                continue

            entry_bar = bar + 1
            if entry_bar >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            # Momentum guard
            if use_momentum and bar < mom_pause_until.get(direction, -1):
                blocked['momentum'] += 1
                continue

            # Regime sizing
            sm = 1.0
            slope = ema_slope[bar] if bar < len(ema_slope) else 0
            is_uptrend = slope > 0
            is_counter = (direction == 'SHORT' and is_uptrend) or (direction == 'LONG' and not is_uptrend)

            if use_regime and is_counter:
                sm = regime_mult

            # MDD sizing
            if use_mdd and peak_equity > 0:
                dd_pct = (peak_equity - equity) / peak_equity * 100
                if dd_pct <= mdd_full_below:
                    mdd_scale = 1.0
                elif dd_pct >= mdd_min_above:
                    mdd_scale = mdd_min_scale
                else:
                    mdd_scale = 1.0 - (1.0 - mdd_min_scale) * (dd_pct - mdd_full_below) / (mdd_min_above - mdd_full_below)
                sm *= mdd_scale

            # ATR scaling
            if use_atr and bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
                r = max(clamp_lo, min(clamp_hi, atr_ratio[bar]))
            else:
                r = 1.0

            eff_tp = tp_pct * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

            # Aggregate risk cap
            if use_agg_risk:
                cap_pct = agg_counter if is_counter else agg_with
                existing = sum(
                    p['eff_sl_pct'] * (1.0 / N_SLOTS) * p['leverage'] * p['size_mult']
                    for p in positions if p['direction'] == direction
                )
                lev_for_check = LEVERAGE  # approximate
                new_exp = eff_sl * (1.0 / N_SLOTS) * lev_for_check * sm
                if existing + new_exp > cap_pct:
                    blocked['agg_risk'] += 1
                    continue

            # Adaptive leverage
            lev = float(LEVERAGE)
            if use_adaptive_lev and len(recent_trades_pnl) >= adap_window:
                recent = recent_trades_pnl[-adap_window:]
                wins = sum(1 for p in recent if p > 0)
                rolling_wr = wins / adap_window
                rolling_edge = np.mean(recent) / 100

                wr_conf = rolling_wr / adap_wr if adap_wr > 0 else 1.0
                edge_conf = rolling_edge / adap_edge if adap_edge > 0 else 1.0
                raw_lev = LEVERAGE * min(wr_conf, edge_conf)
                lev = max(adap_min, min(adap_max, raw_lev))

                # DD reduction
                if peak_equity > 0:
                    dd_now = (peak_equity - equity) / peak_equity * 100
                    if dd_now >= 2.0:
                        lev = max(adap_min, lev * 0.5)

            positions.append({
                'slot': f"{pat}_{bar}",
                'entry_bar': entry_bar,
                'entry_price': entry_price,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'eff_tp_pct': eff_tp,
                'eff_sl_pct': eff_sl,
                'leverage': lev,
                'size_mult': sm,
            })

    # Force-close remaining
    for pos in positions:
        eb = pos['entry_bar']
        if eb >= n_bars:
            continue
        ep = pos['entry_price']
        if ep <= 0:
            continue
        exit_bar = min(end_bar - 1, n_bars - 1)
        exit_price = opens[exit_bar]
        if pos['direction'] == 'LONG':
            pnl = (exit_price / ep - 1) * 100 * pos['leverage']
        else:
            pnl = (1 - exit_price / ep) * 100 * pos['leverage']
        pnl -= fee
        sm = pos['size_mult']
        trades.append({
            'entry_bar': eb, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm,
            'pnl_portfolio': pnl * (size_pct / 100) * sm, 'leverage': pos['leverage'],
        })

    stats = {
        'early_exits': early_count,
        'cascade_triggers': cascade_count,
        'timeouts': timeout_count,
        'blocked': blocked,
    }
    return trades, equity, max_dd, stats


# ============================================================
# Statistics
# ============================================================
def calc_stats(trades, final_equity, max_dd):
    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'mdd': 0, 'pnl_mdd': 0}
    real_trades = [t for t in trades if t['reason'] not in ('END', 'TIMEOUT')]
    n = len(real_trades)
    wins = sum(1 for t in real_trades if t['pnl_slot'] > 0)
    wr = (wins / n * 100) if n > 0 else 0
    pnl = final_equity - 100.0
    pnl_mdd = pnl / max_dd if max_dd > 0 else 0
    return {
        'trades': n,
        'wr': round(wr, 1),
        'pnl': round(pnl, 1),
        'mdd': round(max_dd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
    }


def run_scenario(signal_tuples, opens, highs, lows, closes, type_codes,
                 n_bars, atr_ratio, ema_slope, start_bar, end_bar, config):
    """Run single scenario and return stats dict."""
    trades, equity, max_dd, extra = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar, config,
    )
    stats = calc_stats(trades, equity, max_dd)
    stats.update(extra)
    return stats


def run_wf(signal_tuples, opens, highs, lows, closes, type_codes,
           n_bars, atr_ratio, ema_slope, neutral_start, neutral_end, config,
           n_folds=3):
    """3-fold expanding window WF."""
    neutral_bars = neutral_end - neutral_start
    fold_size = neutral_bars // (n_folds + 1)
    results = []
    for fold in range(n_folds):
        oos_start = neutral_start + fold_size * (fold + 1)
        oos_end = neutral_start + fold_size * (fold + 2) if fold < n_folds - 1 else neutral_end
        trades, equity, max_dd, _ = portfolio_npos(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end, config,
        )
        stats = calc_stats(trades, equity, max_dd)
        stats['fold'] = fold + 1
        results.append(stats)
    return results


# ============================================================
# Pairwise synergy pairs
# ============================================================
SYNERGY_PAIRS = [
    ('adaptive_leverage', 'mdd_sizing',      'M4+M1 Lev×MDD'),
    ('momentum_guard',    'agg_risk_cap',     'G2+G5 Mom×Agg'),
    ('regime_sizing',     'agg_risk_cap',     'M2+G5 Reg×Agg'),
    ('early_exit',        'timeout',          'P2+P1 Early×TO'),
    ('cascade_sl',        'agg_risk_cap',     'P3+G5 Casc×Agg'),
    ('cascade_sl',        'dir_cap',          'P3+G1 Casc×Dir'),
    ('atr_scaling',       'adaptive_leverage', 'M5+M4 ATR×Lev'),
    ('atr_scaling',       'early_exit',       'M5+P2 ATR×Early'),
    ('regime_sizing',     'adaptive_leverage', 'M2+M4 Reg×Lev'),
    ('cascade_sl',        'regime_sizing',    'P3+M2 Casc×Reg'),
    ('momentum_guard',    'cascade_sl',       'G2+P3 Mom×Casc'),
    ('mdd_sizing',        'regime_sizing',    'M1+M2 MDD×Reg'),
    ('timeout',           'atr_scaling',      'P1+M5 TO×ATR'),
    ('dir_cap',           'agg_risk_cap',     'G1+G5 Dir×Agg'),
    ('early_exit',        'cascade_sl',       'P2+P3 Early×Casc'),
]


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    print("=" * 72)
    print("Mechanism Portfolio Study (v1.41.0)")
    print("10 Active Mechanisms — Individual Ablation + Synergy Analysis")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # Phase 0: Load data
    print("\n[Phase 0] Loading data and patterns...")
    df, types = load_data()
    n_bars = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    type_codes = df['candle_type'].values

    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)

    nw = find_neutral_window(closes)
    if nw is None:
        print("ERROR: No neutral window")
        return
    neutral_start, neutral_end = nw
    print(f"  Neutral window: bar {neutral_start} → {neutral_end} ({(neutral_end-neutral_start)/288:.0f}d)")

    patterns = load_patterns()
    print(f"  Patterns: {len(patterns)}")

    sig_idx = build_signal_index(types[neutral_start:neutral_end+1],
                                 neutral_end - neutral_start + 1)
    signal_tuples = []
    for pat_name, info in patterns.items():
        direction = info['direction']
        tp_pct = info['tp_pct']
        sl_pct = info['sl_pct']
        if pat_name in sig_idx:
            for offset in sig_idx[pat_name]:
                abs_bar = neutral_start + offset
                signal_tuples.append((abs_bar, pat_name, direction, tp_pct, sl_pct))
    signal_tuples.sort(key=lambda x: x[0])
    print(f"  Signals: {len(signal_tuples)}")

    output = {'timestamp': datetime.now().isoformat(), 'n_bars': n_bars,
              'neutral_window': [neutral_start, neutral_end],
              'n_patterns': len(patterns), 'n_signals': len(signal_tuples)}

    args = (signal_tuples, opens, highs, lows, closes, type_codes,
            n_bars, atr_ratio, ema_slope, neutral_start, neutral_end)

    # ========================================================
    # Phase 1: Full Stack Baseline
    # ========================================================
    print("\n[Phase 1] Full Stack Baseline (all 10 ON)...")
    full = run_scenario(*args, FULL_STACK)
    print(f"  Trades: {full['trades']:>4} | WR: {full['wr']:>5.1f}% | "
          f"PnL: {full['pnl']:>+8.1f}% | MDD: {full['mdd']:>5.2f}% | "
          f"PnL/MDD: {full['pnl_mdd']:>7.2f}")
    print(f"  Blocked: DirCap={full['blocked']['dir_cap']}, "
          f"Mom={full['blocked']['momentum']}, Agg={full['blocked']['agg_risk']} | "
          f"Early={full['early_exits']} | Cascade={full['cascade_triggers']} | "
          f"Timeout={full['timeouts']}")
    output['phase1_full_stack'] = full

    # ========================================================
    # Phase 2: Individual Ablation
    # ========================================================
    print("\n[Phase 2] Individual Ablation (remove one at a time)...")
    print(f"  {'Mechanism':<22} | {'PnL/MDD':>8} | {'ΔPnL/MDD':>9} | "
          f"{'ΔPnL':>8} | {'ΔMDD':>7} | {'WR':>5} | {'Trades':>6}")
    print("  " + "-" * 82)

    ablation_results = {}
    for key in MECHANISM_KEYS:
        cfg = _ablation_config(key)
        res = run_scenario(*args, cfg)
        delta_pm = full['pnl_mdd'] - res['pnl_mdd']
        delta_pnl = full['pnl'] - res['pnl']
        delta_mdd = res['mdd'] - full['mdd']
        ablation_results[key] = res
        ablation_results[key]['delta_pnl_mdd'] = round(delta_pm, 2)
        ablation_results[key]['delta_pnl'] = round(delta_pnl, 1)
        ablation_results[key]['delta_mdd'] = round(delta_mdd, 2)
        print(f"  {MECHANISM_NAMES[key]:<22} | {res['pnl_mdd']:>8.2f} | "
              f"{delta_pm:>+9.2f} | {delta_pnl:>+8.1f}% | {delta_mdd:>+7.2f}% | "
              f"{res['wr']:>5.1f} | {res['trades']:>6}")

    # Rank by ablation impact
    ranked = sorted(MECHANISM_KEYS, key=lambda k: -ablation_results[k]['delta_pnl_mdd'])
    print("\n  Ablation Ranking (most valuable first):")
    for i, key in enumerate(ranked, 1):
        d = ablation_results[key]
        sign = "+" if d['delta_pnl_mdd'] >= 0 else ""
        print(f"    {i:>2}. {MECHANISM_NAMES[key]:<22} Δ PnL/MDD = {sign}{d['delta_pnl_mdd']:.2f}")
    output['phase2_ablation'] = {k: ablation_results[k] for k in MECHANISM_KEYS}

    # ========================================================
    # Phase 3: Individual Additive
    # ========================================================
    print("\n[Phase 3] Individual Additive (add one to minimal baseline)...")
    minimal = run_scenario(*args, MINIMAL)
    print(f"  MINIMAL (all OFF) | Trades: {minimal['trades']:>4} | WR: {minimal['wr']:>5.1f}% | "
          f"PnL: {minimal['pnl']:>+8.1f}% | MDD: {minimal['mdd']:>5.2f}% | "
          f"PnL/MDD: {minimal['pnl_mdd']:>7.2f}")
    print()
    print(f"  {'Mechanism':<22} | {'PnL/MDD':>8} | {'ΔPnL/MDD':>9} | "
          f"{'ΔPnL':>8} | {'ΔMDD':>7}")
    print("  " + "-" * 65)

    additive_results = {}
    for key in MECHANISM_KEYS:
        cfg = _additive_config(key)
        res = run_scenario(*args, cfg)
        delta_pm = res['pnl_mdd'] - minimal['pnl_mdd']
        delta_pnl = res['pnl'] - minimal['pnl']
        delta_mdd = res['mdd'] - minimal['mdd']
        additive_results[key] = res
        additive_results[key]['delta_pnl_mdd'] = round(delta_pm, 2)
        additive_results[key]['delta_pnl'] = round(delta_pnl, 1)
        additive_results[key]['delta_mdd'] = round(delta_mdd, 2)
        print(f"  {MECHANISM_NAMES[key]:<22} | {res['pnl_mdd']:>8.2f} | "
              f"{delta_pm:>+9.2f} | {delta_pnl:>+8.1f}% | {delta_mdd:>+7.2f}%")

    output['phase3_additive'] = {'minimal': minimal}
    output['phase3_additive'].update({k: additive_results[k] for k in MECHANISM_KEYS})

    # ========================================================
    # Phase 4: Pairwise Synergy
    # ========================================================
    print("\n[Phase 4] Pairwise Synergy Analysis (15 key pairs)...")
    print(f"  {'Pair':<26} | {'Both OFF':>8} | {'Synergy':>8} | {'Type':<12}")
    print("  " + "-" * 66)

    synergy_results = {}
    for key_a, key_b, label in SYNERGY_PAIRS:
        cfg = _pair_ablation_config(key_a, key_b)
        res = run_scenario(*args, cfg)

        # Synergy = (F - AB) - (F - A) - (F - B)
        # where F = full, A = without_a, B = without_b, AB = without_both
        F = full['pnl_mdd']
        A = ablation_results[key_a]['pnl_mdd']
        B = ablation_results[key_b]['pnl_mdd']
        AB = res['pnl_mdd']

        # Individual marginal values
        val_a = F - A  # value of having mechanism a
        val_b = F - B
        combined = F - AB  # value of having both

        synergy = combined - val_a - val_b
        # synergy > 0 → super-additive (true synergy)
        # synergy < 0 → sub-additive (redundancy)

        if abs(synergy) < 0.5:
            stype = 'INDEPENDENT'
        elif synergy > 0:
            stype = 'SYNERGISTIC'
        else:
            stype = 'REDUNDANT'

        synergy_results[f"{key_a}+{key_b}"] = {
            'label': label,
            'both_off_pnl_mdd': round(AB, 2),
            'val_a': round(val_a, 2),
            'val_b': round(val_b, 2),
            'combined': round(combined, 2),
            'synergy': round(synergy, 2),
            'type': stype,
        }
        print(f"  {label:<26} | {AB:>8.2f} | {synergy:>+8.2f} | {stype:<12}")

    output['phase4_synergy'] = synergy_results

    # ========================================================
    # Phase 5: WF Validation
    # ========================================================
    print("\n[Phase 5] Walk-Forward Validation (3-fold Expanding Window)...")

    wf_scenarios = {'full_stack': FULL_STACK, 'minimal': MINIMAL}

    # Also test: full stack minus most harmful, plus any surprising finding
    # Identify the mechanism with worst ablation (removing it helps most)
    worst_key = max(MECHANISM_KEYS, key=lambda k: -ablation_results[k]['delta_pnl_mdd'])
    if ablation_results[worst_key]['delta_pnl_mdd'] < -0.5:
        wf_scenarios[f'without_{worst_key}'] = _ablation_config(worst_key)

    output['phase5_wf'] = {}
    for name, cfg in wf_scenarios.items():
        folds = run_wf(signal_tuples, opens, highs, lows, closes, type_codes,
                       n_bars, atr_ratio, ema_slope, neutral_start, neutral_end, cfg)
        all_pass = all(f['pnl'] > 0 for f in folds)
        total_pnl = sum(f['pnl'] for f in folds)
        fold_strs = [f"+{f['pnl']:.1f}%" if f['pnl'] > 0 else f"{f['pnl']:.1f}%" for f in folds]
        min_fold = min(f['pnl'] for f in folds)
        verdict = "PASS" if all_pass else "FAIL"
        print(f"  {name:<22} | {'✅' if all_pass else '❌'} {verdict} | "
              f"Folds: [{', '.join(fold_strs)}] | Total: {total_pnl:>+.1f}% | Min: {min_fold:>+.1f}%")
        output['phase5_wf'][name] = {
            'folds': [{'fold': f['fold'], 'pnl': f['pnl'], 'wr': f['wr'],
                        'mdd': f['mdd'], 'trades': f['trades']} for f in folds],
            'all_pass': all_pass,
            'total_pnl': round(total_pnl, 1),
            'min_fold': round(min_fold, 1),
        }

    # ========================================================
    # Summary
    # ========================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"\nFull Stack: PnL {full['pnl']:+.1f}% | MDD {full['mdd']:.2f}% | "
          f"PnL/MDD {full['pnl_mdd']:.2f} | WR {full['wr']:.1f}%")
    print(f"Minimal:    PnL {minimal['pnl']:+.1f}% | MDD {minimal['mdd']:.2f}% | "
          f"PnL/MDD {minimal['pnl_mdd']:.2f} | WR {minimal['wr']:.1f}%")
    print(f"\nTotal risk management value: PnL/MDD {full['pnl_mdd']:.2f} vs {minimal['pnl_mdd']:.2f} "
          f"(+{full['pnl_mdd'] - minimal['pnl_mdd']:.2f})")

    print(f"\nTop 3 most valuable mechanisms (ablation):")
    for i, key in enumerate(ranked[:3], 1):
        d = ablation_results[key]
        print(f"  {i}. {MECHANISM_NAMES[key]:<22} Δ PnL/MDD = +{d['delta_pnl_mdd']:.2f}")

    synergistic = [(k, v) for k, v in synergy_results.items() if v['type'] == 'SYNERGISTIC']
    redundant = [(k, v) for k, v in synergy_results.items() if v['type'] == 'REDUNDANT']

    if synergistic:
        print(f"\nSynergistic pairs (super-additive):")
        for k, v in sorted(synergistic, key=lambda x: -x[1]['synergy']):
            print(f"  {v['label']:<26} synergy = +{v['synergy']:.2f}")
    if redundant:
        print(f"\nRedundant pairs (sub-additive):")
        for k, v in sorted(redundant, key=lambda x: x[1]['synergy']):
            print(f"  {v['label']:<26} synergy = {v['synergy']:.2f}")

    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"{'=' * 72}")

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
