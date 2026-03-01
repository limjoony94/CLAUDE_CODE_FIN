#!/usr/bin/env python3
"""
Portfolio Exposure Management Study — Hedge 모드 개선 방안 연구

핵심 문제:
  현행 hedge 모드 (N=9, direction_cap=7, regime_sizing ×0.3, agg_risk_cap)는
  동일 방향 다수 포지션 동시 진입 → 연쇄 SL 피격 → correlated burst loss 발생.
  2026-02 live: 6 bursts = 19 trades, -20.40% portfolio PnL.

연구 질문:
  1. 동일 방향 burst 진입을 제한하면 correlated loss가 줄어드는가?
  2. Net exposure 관리가 per-direction cap보다 효과적인가?
  3. Rolling WR 기반 동적 사이징이 고정 regime_sizing보다 나은가?
  4. 손실 후 방향별 cooldown이 연쇄 손실을 막는가?
  5. 방향별 동적 cap이 고정 cap보다 나은가?

시나리오 (14개):
  H0.  BASELINE — 현행 시스템 (cap7, regime 0.3, agg_risk 3/7)
  H1a. Dynamic direction cap (regime-aware) — uptrend: L7/S3, downtrend: L3/S7
  H1b. Dynamic direction cap (moderate) — uptrend: L7/S5, downtrend: L5/S7
  H1c. Dynamic direction cap (aggressive) — uptrend: L7/S2, downtrend: L2/S7
  H2a. Net exposure cap = 5 — |LONG_count - SHORT_count| ≤ 5
  H2b. Net exposure cap = 3 — |LONG_count - SHORT_count| ≤ 3
  H3a. Entry cooldown (direction) 3 bars — same direction entry ≥ 15min apart
  H3b. Entry cooldown (direction) 6 bars — same direction entry ≥ 30min apart
  H3c. Entry cooldown (direction) 12 bars — same direction entry ≥ 60min apart
  H4a. Rolling WR sizing — LONG size *= min(1.0, rolling_LONG_WR / 80)
  H4b. Rolling WR sizing — both dirs, size *= min(1.0, rolling_dir_WR / 80)
  H5.  Adaptive regime sizing — counter_mult = max(0.1, rolling_counter_WR / 100)
  H6a. Loss burst brake — after 2 same-dir losses in 24h, block that dir for 6h
  H6b. Loss burst brake — after 2 same-dir losses in 24h, block that dir for 12h

평가 지표:
  - PnL, MDD, PnL/MDD
  - Correlated burst count & impact (3+ same-dir losses in 288 bars)
  - Max single-day loss
  - WF 3-fold OOS 검증

Protocol:
  - Production classify_candle() via scanner
  - LEVERAGE = 3, FEE = 0.10% × LEVERAGE
  - ATR-scaled TP/SL (scanner bt_signals_atr)
  - N=9 portfolio, 1/N=11.1% sizing (adjusted by scenario)
  - Timeout 864 bars (DROP)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

import scripts.scanner.pattern_scanner as scanner

RESULT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'portfolio_exposure_study.json')
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
LEVERAGE = scanner.LEVERAGE
FEE_PCT = scanner.FEE_PCT
BARS_PER_DAY = 288
TIMEOUT_BARS = 864
MAX_POSITIONS = 9
BASE_POSITION_SIZE = 1.0 / MAX_POSITIONS  # 11.1%


def load_patterns():
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    patterns = []
    for key, info in data.get('pattern_details', {}).items():
        patterns.append({
            'pattern': info['pattern'],
            'direction': info['direction'],
            'tp': info['tp'],
            'sl': info['sl'],
        })
    return patterns


def generate_all_trades(patterns, signal_index, opens, highs, lows, closes,
                        n_bars, use_atr=True):
    atr_ratio = None
    if use_atr and hasattr(scanner, 'compute_atr_ratio'):
        atr_ratio = scanner.compute_atr_ratio(highs, lows, closes)

    all_trades = []
    for pat in patterns:
        sigs = signal_index.get(pat['pattern'], [])
        if not sigs:
            continue
        direction = pat['direction']
        tp_pct = pat['tp']
        sl_pct = pat['sl']

        if use_atr and atr_ratio is not None:
            raw_trades = scanner.bt_signals_atr(
                sigs, direction, tp_pct, sl_pct,
                opens, highs, lows, n_bars, atr_ratio)
        else:
            raw_trades = scanner.bt_signals(
                sigs, direction, tp_pct, sl_pct,
                opens, highs, lows, n_bars)

        for entry_bar, exit_bar, pnl in raw_trades:
            all_trades.append({
                'bar': entry_bar,
                'exit_bar': exit_bar,
                'direction': direction,
                'pnl': round(pnl, 4),
                'pattern': pat['pattern'],
                'tp': tp_pct,
                'sl': sl_pct,
            })

    all_trades.sort(key=lambda t: t['bar'])
    return all_trades


def compute_ema_regime(closes, period=20, lookback=5):
    """EMA slope regime: +1 uptrend, -1 downtrend, 0 neutral."""
    ema = pd.Series(closes).ewm(span=period, adjust=False).mean().values
    regime = np.zeros(len(closes), dtype=int)
    for i in range(lookback, len(closes)):
        slope = (ema[i] - ema[i - lookback]) / ema[i - lookback] * 100
        if slope > 0.05:
            regime[i] = 1
        elif slope < -0.05:
            regime[i] = -1
    return regime


def compute_rolling_dir_wr(trades, n_bars, direction, window=30):
    """Rolling WR of last N trades for a direction. Returns array indexed by bar."""
    dir_trades = sorted([t for t in trades if t['direction'] == direction],
                        key=lambda t: t['bar'])
    wr_array = np.full(n_bars, 80.0)  # Default 80% (optimistic start)

    for i in range(window, len(dir_trades)):
        recent = dir_trades[i - window:i]
        wr = sum(1 for t in recent if t['pnl'] > 0) / len(recent) * 100
        bar = dir_trades[i]['bar']
        wr_array[bar] = wr

    # Forward fill
    last_val = 80.0
    for i in range(n_bars):
        if wr_array[i] != 80.0 or i == 0:
            last_val = wr_array[i]
        else:
            wr_array[i] = last_val

    return wr_array


# ═══════════════════════════════════════════════════════════════
# PORTFOLIO SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════

def simulate_portfolio(trades, n_bars, closes, regime, long_wr, short_wr,
                       scenario_config):
    """
    Simulate N=9 portfolio with scenario-specific rules.

    scenario_config keys:
      - name: scenario name
      - direction_cap: int or 'dynamic'
      - dynamic_cap_fn: callable(regime_val) -> (long_cap, short_cap)
      - net_exposure_cap: int or None
      - entry_cooldown_bars: int or None (per-direction cooldown)
      - sizing_fn: callable(direction, bar, base_size) -> adjusted_size
      - loss_brake: dict {threshold: int, window_bars: int, block_bars: int} or None
    """
    cfg = scenario_config
    equity = 100.0
    peak_equity = 100.0
    max_dd = 0.0
    max_single_day_loss = 0.0

    active_slots = []  # {direction, exit_bar, entry_bar}
    trade_log = []
    daily_pnl = defaultdict(float)

    # Per-direction tracking
    last_entry_bar = {'LONG': -9999, 'SHORT': -9999}
    recent_dir_losses = {'LONG': [], 'SHORT': []}  # list of bar indices
    dir_blocked_until = {'LONG': -1, 'SHORT': -1}

    for t in trades:
        bar = t['bar']
        direction = t['direction']

        # Remove expired slots
        active_slots = [s for s in active_slots if s['exit_bar'] > bar]

        # ── Max positions ──
        if len(active_slots) >= MAX_POSITIONS:
            continue

        # ── Direction cap ──
        long_count = sum(1 for s in active_slots if s['direction'] == 'LONG')
        short_count = sum(1 for s in active_slots if s['direction'] == 'SHORT')

        if cfg.get('dynamic_cap_fn'):
            reg_val = regime[bar] if bar < len(regime) else 0
            l_cap, s_cap = cfg['dynamic_cap_fn'](reg_val)
        else:
            cap = cfg.get('direction_cap', 7)
            l_cap = s_cap = cap

        if direction == 'LONG' and long_count >= l_cap:
            continue
        if direction == 'SHORT' and short_count >= s_cap:
            continue

        # ── Net exposure cap ──
        net_cap = cfg.get('net_exposure_cap')
        if net_cap is not None:
            if direction == 'LONG':
                new_net = (long_count + 1) - short_count
            else:
                new_net = long_count - (short_count + 1)
            if abs(new_net) > net_cap:
                continue

        # ── Entry cooldown ──
        cooldown = cfg.get('entry_cooldown_bars')
        if cooldown is not None:
            if bar - last_entry_bar[direction] < cooldown:
                continue

        # ── Loss burst brake ──
        brake = cfg.get('loss_brake')
        if brake:
            if bar < dir_blocked_until[direction]:
                continue
            # Clean old losses
            window = brake['window_bars']
            recent_dir_losses[direction] = [
                b for b in recent_dir_losses[direction] if bar - b <= window]

        # ── Position sizing ──
        sizing_fn = cfg.get('sizing_fn')
        if sizing_fn:
            pos_size = sizing_fn(direction, bar, BASE_POSITION_SIZE,
                                 long_wr, short_wr, regime)
        else:
            pos_size = BASE_POSITION_SIZE

        if pos_size <= 0.001:
            continue

        # ── Record trade ──
        pnl_portfolio = t['pnl'] * pos_size
        equity *= (1 + pnl_portfolio / 100)
        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity * 100
        max_dd = min(max_dd, dd)

        active_slots.append({
            'direction': direction,
            'exit_bar': t['exit_bar'],
            'entry_bar': bar,
        })

        last_entry_bar[direction] = bar

        # Track losses for brake
        if t['pnl'] <= 0 and brake:
            recent_dir_losses[direction].append(bar)
            if len(recent_dir_losses[direction]) >= brake['threshold']:
                dir_blocked_until[direction] = bar + brake['block_bars']
                recent_dir_losses[direction] = []

        # Daily PnL tracking
        day_idx = bar // BARS_PER_DAY
        daily_pnl[day_idx] += pnl_portfolio

        trade_log.append({
            'bar': bar,
            'direction': direction,
            'pnl': t['pnl'],
            'pnl_port': round(pnl_portfolio, 4),
            'pos_size': round(pos_size, 4),
            'equity': round(equity, 2),
        })

    total_pnl = equity - 100.0
    wins = sum(1 for t in trade_log if t['pnl'] > 0)

    # Max single day loss
    if daily_pnl:
        max_single_day_loss = min(daily_pnl.values())

    return {
        'pnl': round(total_pnl, 1),
        'mdd': round(abs(max_dd), 1),
        'trades': len(trade_log),
        'wins': wins,
        'wr': round(wins / len(trade_log) * 100, 1) if trade_log else 0,
        'pnl_mdd': round(total_pnl / abs(max_dd), 2) if max_dd != 0 else 0,
        'max_daily_loss': round(max_single_day_loss, 2),
        'trade_log': trade_log,
    }


def count_correlated_bursts(trade_log, window_bars=288, min_losses=3):
    """Count correlated loss bursts (N+ same-direction losses within window)."""
    bursts = []
    for direction in ['LONG', 'SHORT']:
        dir_losses = sorted(
            [t for t in trade_log if t['direction'] == direction and t['pnl'] <= 0],
            key=lambda t: t['bar'])

        i = 0
        while i < len(dir_losses):
            cluster = [dir_losses[i]]
            j = i + 1
            while j < len(dir_losses) and dir_losses[j]['bar'] - cluster[0]['bar'] <= window_bars:
                cluster.append(dir_losses[j])
                j += 1
            if len(cluster) >= min_losses:
                bursts.append({
                    'direction': direction,
                    'start_bar': cluster[0]['bar'],
                    'n_losses': len(cluster),
                    'total_loss': round(sum(t['pnl_port'] for t in cluster), 2),
                })
                i = j
            else:
                i += 1

    return bursts


# ═══════════════════════════════════════════════════════════════
# WF VALIDATION
# ═══════════════════════════════════════════════════════════════

def wf_validate(trades, n_bars, closes, regime, scenario_config, n_folds=3):
    """3-fold expanding window WF."""
    n_segments = n_folds + 1
    fold_size = n_bars // n_segments
    fold_results = []

    for fold in range(n_folds):
        is_end = fold_size * (fold + 1)
        oos_start = is_end
        oos_end = min(is_end + fold_size, n_bars)

        oos_trades = [t for t in trades if oos_start <= t['bar'] < oos_end]
        if not oos_trades:
            fold_results.append({'fold': fold + 1, 'oos_pnl': 0, 'pass': False})
            continue

        # Compute rolling WR from IS trades only
        is_trades = [t for t in trades if t['bar'] < is_end]
        long_wr = compute_rolling_dir_wr(is_trades, n_bars, 'LONG', window=30)
        short_wr = compute_rolling_dir_wr(is_trades, n_bars, 'SHORT', window=30)

        result = simulate_portfolio(
            oos_trades, n_bars, closes, regime, long_wr, short_wr, scenario_config)

        fold_results.append({
            'fold': fold + 1,
            'oos_bars': f"{oos_start}-{oos_end}",
            'oos_days': round((oos_end - oos_start) / BARS_PER_DAY, 0),
            'oos_trades': result['trades'],
            'oos_wr': result['wr'],
            'oos_pnl': result['pnl'],
            'oos_mdd': result['mdd'],
            'pass': result['pnl'] > 0,
        })

    return fold_results


# ═══════════════════════════════════════════════════════════════
# SIZING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def sizing_rolling_wr_long(direction, bar, base_size, long_wr, short_wr, regime):
    """Scale LONG size by rolling WR. SHORT unchanged."""
    if direction == 'LONG':
        wr = long_wr[bar] if bar < len(long_wr) else 80.0
        mult = min(1.0, max(0.1, wr / 80.0))
        return base_size * mult
    return base_size


def sizing_rolling_wr_both(direction, bar, base_size, long_wr, short_wr, regime):
    """Scale both directions by their rolling WR."""
    if direction == 'LONG':
        wr = long_wr[bar] if bar < len(long_wr) else 80.0
    else:
        wr = short_wr[bar] if bar < len(short_wr) else 80.0
    mult = min(1.0, max(0.1, wr / 80.0))
    return base_size * mult


def sizing_adaptive_regime(direction, bar, base_size, long_wr, short_wr, regime):
    """Adaptive counter-regime sizing based on rolling WR instead of fixed 0.3."""
    reg = regime[bar] if bar < len(regime) else 0
    is_counter = (direction == 'LONG' and reg == -1) or (direction == 'SHORT' and reg == 1)
    if not is_counter:
        return base_size
    # Use rolling WR to set counter mult
    if direction == 'LONG':
        wr = long_wr[bar] if bar < len(long_wr) else 80.0
    else:
        wr = short_wr[bar] if bar < len(short_wr) else 80.0
    # Scale: WR 80%→1.0, WR 50%→0.3, WR 30%→0.1
    mult = max(0.1, min(1.0, (wr - 30) / 50))
    return base_size * mult


def sizing_fixed_regime(direction, bar, base_size, long_wr, short_wr, regime):
    """Current production: counter-regime ×0.3."""
    reg = regime[bar] if bar < len(regime) else 0
    is_counter = (direction == 'LONG' and reg == -1) or (direction == 'SHORT' and reg == 1)
    if is_counter:
        return base_size * 0.3
    return base_size


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    results = {'study': 'portfolio_exposure_management', 'sections': {}}

    print("=" * 72)
    print("PORTFOLIO EXPOSURE MANAGEMENT — Hedge 모드 개선 연구")
    print("=" * 72)

    # ── Load ──
    print("\nLoading data...")
    df = scanner.load_and_classify(DATA_FILE)
    types = df['candle_type'].tolist()
    signal_index = scanner.build_signal_index(types, len(df))
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)
    timestamps = pd.to_datetime(df['timestamp'])
    bar_to_date = timestamps.dt.date.values

    patterns = load_patterns()
    long_pats = [p for p in patterns if p['direction'] == 'LONG']
    short_pats = [p for p in patterns if p['direction'] == 'SHORT']

    print(f"  {n_bars} bars ({n_bars/BARS_PER_DAY:.0f} days)")
    print(f"  {len(patterns)} patterns ({len(long_pats)}L + {len(short_pats)}S)")

    # ── Generate trades ──
    print("\nGenerating trades (ATR-scaled)...")
    trades = generate_all_trades(
        patterns, signal_index, opens, highs, lows, closes, n_bars, use_atr=True)
    print(f"  Total signals: {len(trades)} trades")

    # ── Compute regime & rolling WR ──
    regime = compute_ema_regime(closes, period=20, lookback=5)
    long_wr = compute_rolling_dir_wr(trades, n_bars, 'LONG', window=30)
    short_wr = compute_rolling_dir_wr(trades, n_bars, 'SHORT', window=30)

    results['meta'] = {
        'bars': n_bars,
        'days': round(n_bars / BARS_PER_DAY, 1),
        'patterns': len(patterns),
        'total_signals': len(trades),
    }

    # ═══════════════════════════════════════════════════════════
    # SCENARIO DEFINITIONS
    # ═══════════════════════════════════════════════════════════

    scenarios = [
        # H0: Baseline (current production)
        {
            'name': 'H0: Baseline (cap7, regime×0.3)',
            'direction_cap': 7,
            'sizing_fn': sizing_fixed_regime,
        },
        # H1: Dynamic direction cap
        {
            'name': 'H1a: DynCap regime L7/S3↔L3/S7',
            'dynamic_cap_fn': lambda r: (7, 3) if r >= 1 else ((3, 7) if r <= -1 else (5, 5)),
            'sizing_fn': sizing_fixed_regime,
        },
        {
            'name': 'H1b: DynCap moderate L7/S5↔L5/S7',
            'dynamic_cap_fn': lambda r: (7, 5) if r >= 1 else ((5, 7) if r <= -1 else (6, 6)),
            'sizing_fn': sizing_fixed_regime,
        },
        {
            'name': 'H1c: DynCap aggressive L7/S2↔L2/S7',
            'dynamic_cap_fn': lambda r: (7, 2) if r >= 1 else ((2, 7) if r <= -1 else (4, 4)),
            'sizing_fn': sizing_fixed_regime,
        },
        # H2: Net exposure cap
        {
            'name': 'H2a: NetExp cap=5',
            'direction_cap': 7,
            'net_exposure_cap': 5,
            'sizing_fn': sizing_fixed_regime,
        },
        {
            'name': 'H2b: NetExp cap=3',
            'direction_cap': 7,
            'net_exposure_cap': 3,
            'sizing_fn': sizing_fixed_regime,
        },
        # H3: Entry cooldown per direction
        {
            'name': 'H3a: Cooldown 3bars (15min)',
            'direction_cap': 7,
            'entry_cooldown_bars': 3,
            'sizing_fn': sizing_fixed_regime,
        },
        {
            'name': 'H3b: Cooldown 6bars (30min)',
            'direction_cap': 7,
            'entry_cooldown_bars': 6,
            'sizing_fn': sizing_fixed_regime,
        },
        {
            'name': 'H3c: Cooldown 12bars (60min)',
            'direction_cap': 7,
            'entry_cooldown_bars': 12,
            'sizing_fn': sizing_fixed_regime,
        },
        # H4: Rolling WR sizing
        {
            'name': 'H4a: RollingWR sizing (LONG only)',
            'direction_cap': 7,
            'sizing_fn': sizing_rolling_wr_long,
        },
        {
            'name': 'H4b: RollingWR sizing (both dirs)',
            'direction_cap': 7,
            'sizing_fn': sizing_rolling_wr_both,
        },
        # H5: Adaptive regime sizing
        {
            'name': 'H5: Adaptive regime (WR-based mult)',
            'direction_cap': 7,
            'sizing_fn': sizing_adaptive_regime,
        },
        # H6: Loss burst brake
        {
            'name': 'H6a: LossBrake 2in24h→block6h',
            'direction_cap': 7,
            'sizing_fn': sizing_fixed_regime,
            'loss_brake': {'threshold': 2, 'window_bars': 288, 'block_bars': 72},
        },
        {
            'name': 'H6b: LossBrake 2in24h→block12h',
            'direction_cap': 7,
            'sizing_fn': sizing_fixed_regime,
            'loss_brake': {'threshold': 2, 'window_bars': 288, 'block_bars': 144},
        },
    ]

    # ═══════════════════════════════════════════════════════════
    # 1. IN-SAMPLE COMPARISON
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("1. IN-SAMPLE — 14 SCENARIOS (297d, N=9 compound)")
    print(f"{'='*72}")

    print(f"\n  {'Scenario':<40} {'Trades':>6} {'WR%':>6} {'MDD%':>6} "
          f"{'Bursts':>6} {'BurstPnL':>9} {'MaxDayL':>8}")
    print(f"  {'-'*85}")

    is_results = []

    for cfg in scenarios:
        result = simulate_portfolio(
            trades, n_bars, closes, regime, long_wr, short_wr, cfg)
        bursts = count_correlated_bursts(result['trade_log'])
        burst_pnl = sum(b['total_loss'] for b in bursts)

        print(f"  {cfg['name']:<40} {result['trades']:>6} {result['wr']:>5.1f}% "
              f"{result['mdd']:>5.1f}% {len(bursts):>6} {burst_pnl:>+8.1f}% "
              f"{result['max_daily_loss']:>+7.2f}%")

        is_results.append({
            'scenario': cfg['name'],
            'trades': result['trades'],
            'wr': result['wr'],
            'pnl': result['pnl'],
            'mdd': result['mdd'],
            'pnl_mdd': result['pnl_mdd'],
            'burst_count': len(bursts),
            'burst_pnl': round(burst_pnl, 1),
            'max_daily_loss': result['max_daily_loss'],
        })

    results['sections']['is_comparison'] = is_results

    # ── Burst analysis for baseline ──
    baseline_result = simulate_portfolio(
        trades, n_bars, closes, regime, long_wr, short_wr, scenarios[0])
    baseline_bursts = count_correlated_bursts(baseline_result['trade_log'])

    print(f"\n  Baseline burst details ({len(baseline_bursts)} bursts):")
    for b in baseline_bursts[:10]:
        date = str(bar_to_date[min(b['start_bar'], n_bars - 1)])
        print(f"    {b['direction']:<6} bar {b['start_bar']:>6} ({date}): "
              f"{b['n_losses']} losses → {b['total_loss']:+.2f}%")
    if len(baseline_bursts) > 10:
        print(f"    ... and {len(baseline_bursts) - 10} more")

    # ═══════════════════════════════════════════════════════════
    # 2. RELATIVE COMPARISON — Delta from Baseline
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("2. VS BASELINE — Delta Analysis")
    print(f"{'='*72}")

    bl = is_results[0]
    print(f"\n  {'Scenario':<40} {'ΔMDD':>6} {'ΔBursts':>8} {'ΔBurstPnL':>10} {'ΔTrades':>8}")
    print(f"  {'-'*75}")

    for s in is_results[1:]:
        d_mdd = s['mdd'] - bl['mdd']
        d_bursts = s['burst_count'] - bl['burst_count']
        d_burst_pnl = s['burst_pnl'] - bl['burst_pnl']
        d_trades = s['trades'] - bl['trades']
        # Highlight improvements
        mdd_mark = "✓" if d_mdd < -1 else " "
        burst_mark = "✓" if d_bursts < -2 else " "
        print(f"  {s['scenario']:<40} {d_mdd:>+5.1f}{mdd_mark} {d_bursts:>+7}{burst_mark} "
              f"{d_burst_pnl:>+9.1f}% {d_trades:>+7}")

    # ═══════════════════════════════════════════════════════════
    # 3. THIRD-PERIOD FOCUS (most recent ~100d)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("3. RECENT PERIOD FOCUS (last third of data, ~100d)")
    print(f"{'='*72}")

    recent_start = n_bars * 2 // 3
    recent_trades = [t for t in trades if t['bar'] >= recent_start]

    print(f"\n  Recent period: bar {recent_start}→{n_bars} ({(n_bars-recent_start)/BARS_PER_DAY:.0f} days)")
    print(f"  Signals in period: {len(recent_trades)}")

    # Recompute rolling WR for recent period
    past_trades = [t for t in trades if t['bar'] < recent_start]
    long_wr_recent = compute_rolling_dir_wr(past_trades, n_bars, 'LONG', window=30)
    short_wr_recent = compute_rolling_dir_wr(past_trades, n_bars, 'SHORT', window=30)

    print(f"\n  {'Scenario':<40} {'Trades':>6} {'WR%':>6} {'MDD%':>6} "
          f"{'Bursts':>6} {'BurstPnL':>9}")
    print(f"  {'-'*78}")

    recent_results = []
    for cfg in scenarios:
        result = simulate_portfolio(
            recent_trades, n_bars, closes, regime, long_wr_recent, short_wr_recent, cfg)
        bursts = count_correlated_bursts(result['trade_log'])
        burst_pnl = sum(b['total_loss'] for b in bursts)

        print(f"  {cfg['name']:<40} {result['trades']:>6} {result['wr']:>5.1f}% "
              f"{result['mdd']:>5.1f}% {len(bursts):>6} {burst_pnl:>+8.1f}%")

        recent_results.append({
            'scenario': cfg['name'],
            'trades': result['trades'],
            'wr': result['wr'],
            'pnl': result['pnl'],
            'mdd': result['mdd'],
            'burst_count': len(bursts),
            'burst_pnl': round(burst_pnl, 1),
        })

    results['sections']['recent_period'] = recent_results

    # ═══════════════════════════════════════════════════════════
    # 4. WF VALIDATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("4. WALK-FORWARD VALIDATION (3-fold expanding)")
    print(f"{'='*72}")

    print(f"\n  {'Scenario':<40} {'F1':>8} {'F2':>8} {'F3':>8} {'Result':>8} {'OOS Tot':>9}")
    print(f"  {'-'*78}")

    wf_results = []
    for cfg in scenarios:
        folds = wf_validate(trades, n_bars, closes, regime, cfg)
        passes = sum(1 for f in folds if f['pass'])
        total_oos = sum(f['oos_pnl'] for f in folds)
        verdict = f"{passes}/3 {'PASS' if passes == 3 else 'FAIL'}"

        fold_strs = [f"{f['oos_pnl']:+.0f}%" for f in folds]

        print(f"  {cfg['name']:<40} {fold_strs[0]:>8} {fold_strs[1]:>8} "
              f"{fold_strs[2]:>8} {verdict:>8} {total_oos:>+8.0f}%")

        wf_results.append({
            'scenario': cfg['name'],
            'folds': folds,
            'passes': passes,
            'total_oos_pnl': round(total_oos, 1),
            'verdict': 'PASS' if passes == 3 else 'FAIL',
        })

    results['sections']['wf_validation'] = wf_results

    # ═══════════════════════════════════════════════════════════
    # 5. COMBINED BEST — Stack winning strategies
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("5. COMBINATION — Stacking Best Strategies")
    print(f"{'='*72}")

    # Identify WF-PASS scenarios with better burst reduction
    wf_pass = [w['scenario'] for w in wf_results if w['verdict'] == 'PASS']
    print(f"\n  WF 3/3 PASS scenarios: {len(wf_pass)}")
    for s in wf_pass:
        is_data = next(r for r in is_results if r['scenario'] == s)
        wf_data = next(w for w in wf_results if w['scenario'] == s)
        print(f"    {s:<40} MDD {is_data['mdd']:>5.1f}%  Bursts {is_data['burst_count']:>3}  "
              f"OOS {wf_data['total_oos_pnl']:>+8.0f}%")

    # Test combinations of top WF-PASS strategies
    combo_scenarios = [
        # Cooldown + Adaptive sizing
        {
            'name': 'COMBO1: Cool6b + AdaptiveRegime',
            'direction_cap': 7,
            'entry_cooldown_bars': 6,
            'sizing_fn': sizing_adaptive_regime,
        },
        # Cooldown + Loss brake
        {
            'name': 'COMBO2: Cool6b + LossBrake2/24→6h',
            'direction_cap': 7,
            'entry_cooldown_bars': 6,
            'sizing_fn': sizing_fixed_regime,
            'loss_brake': {'threshold': 2, 'window_bars': 288, 'block_bars': 72},
        },
        # Dynamic cap + cooldown
        {
            'name': 'COMBO3: DynCapMod + Cool6b',
            'dynamic_cap_fn': lambda r: (7, 5) if r >= 1 else ((5, 7) if r <= -1 else (6, 6)),
            'entry_cooldown_bars': 6,
            'sizing_fn': sizing_fixed_regime,
        },
        # Rolling WR sizing + Loss brake
        {
            'name': 'COMBO4: RollingWR + LossBrake',
            'direction_cap': 7,
            'sizing_fn': sizing_rolling_wr_both,
            'loss_brake': {'threshold': 2, 'window_bars': 288, 'block_bars': 72},
        },
        # NetExp + cooldown + adaptive
        {
            'name': 'COMBO5: NetExp3 + Cool6b + Adapt',
            'direction_cap': 7,
            'net_exposure_cap': 3,
            'entry_cooldown_bars': 6,
            'sizing_fn': sizing_adaptive_regime,
        },
        # Best individual stack
        {
            'name': 'COMBO6: Cool12b + LossBrake + Adapt',
            'direction_cap': 7,
            'entry_cooldown_bars': 12,
            'sizing_fn': sizing_adaptive_regime,
            'loss_brake': {'threshold': 2, 'window_bars': 288, 'block_bars': 144},
        },
    ]

    print(f"\n  {'Combo':<43} {'Trades':>6} {'WR%':>6} {'MDD%':>6} {'Bursts':>6} "
          f"{'WF':>8} {'OOS Tot':>9}")
    print(f"  {'-'*88}")

    combo_results = []
    for cfg in combo_scenarios:
        # IS
        result = simulate_portfolio(
            trades, n_bars, closes, regime, long_wr, short_wr, cfg)
        bursts = count_correlated_bursts(result['trade_log'])
        burst_pnl = sum(b['total_loss'] for b in bursts)

        # WF
        folds = wf_validate(trades, n_bars, closes, regime, cfg)
        passes = sum(1 for f in folds if f['pass'])
        total_oos = sum(f['oos_pnl'] for f in folds)
        verdict = f"{passes}/3 {'PASS' if passes == 3 else 'FAIL'}"

        print(f"  {cfg['name']:<43} {result['trades']:>6} {result['wr']:>5.1f}% "
              f"{result['mdd']:>5.1f}% {len(bursts):>6} {verdict:>8} {total_oos:>+8.0f}%")

        combo_results.append({
            'scenario': cfg['name'],
            'trades': result['trades'],
            'wr': result['wr'],
            'pnl': result['pnl'],
            'mdd': result['mdd'],
            'pnl_mdd': result['pnl_mdd'],
            'burst_count': len(bursts),
            'burst_pnl': round(burst_pnl, 1),
            'wf_verdict': verdict,
            'total_oos_pnl': round(total_oos, 1),
        })

    results['sections']['combinations'] = combo_results

    # ═══════════════════════════════════════════════════════════
    # 6. CONCLUSION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("6. CONCLUSION & RECOMMENDATION")
    print(f"{'='*72}")

    # Find best: highest OOS PnL among WF PASS with lower burst count than baseline
    bl_bursts = is_results[0]['burst_count']
    bl_mdd = is_results[0]['mdd']

    all_candidates = []
    for s, w in zip(is_results, wf_results):
        if w['verdict'] == 'PASS':
            all_candidates.append({
                'name': s['scenario'],
                'mdd': s['mdd'],
                'burst_count': s['burst_count'],
                'burst_pnl': s['burst_pnl'],
                'total_oos': w['total_oos_pnl'],
                'mdd_improved': s['mdd'] < bl_mdd,
                'burst_improved': s['burst_count'] < bl_bursts,
            })
    for c in combo_results:
        if 'PASS' in c['wf_verdict']:
            all_candidates.append({
                'name': c['scenario'],
                'mdd': c['mdd'],
                'burst_count': c['burst_count'],
                'burst_pnl': c['burst_pnl'],
                'total_oos': c['total_oos_pnl'],
                'mdd_improved': c['mdd'] < bl_mdd,
                'burst_improved': c['burst_count'] < bl_bursts,
            })

    # Sort by burst reduction, then MDD reduction, then OOS PnL
    improved = [c for c in all_candidates if c['burst_improved'] or c['mdd_improved']]
    improved.sort(key=lambda x: (x['burst_count'], x['mdd'], -x['total_oos']))

    print(f"\n  WF-PASS candidates with improvement vs baseline:")
    print(f"  Baseline: MDD {bl_mdd:.1f}%, Bursts {bl_bursts}")
    print(f"\n  {'Candidate':<43} {'MDD%':>6} {'Bursts':>7} {'BurstPnL':>9} {'OOS':>9}")
    print(f"  {'-'*78}")

    for c in improved[:10]:
        mdd_mark = "↓" if c['mdd_improved'] else " "
        burst_mark = "↓" if c['burst_improved'] else " "
        print(f"  {c['name']:<43} {c['mdd']:>5.1f}{mdd_mark} {c['burst_count']:>6}{burst_mark} "
              f"{c['burst_pnl']:>+8.1f}% {c['total_oos']:>+8.0f}%")

    if improved:
        best = improved[0]
        print(f"\n  ★ BEST: {best['name']}")
        print(f"    MDD: {best['mdd']:.1f}% (baseline {bl_mdd:.1f}%)")
        print(f"    Bursts: {best['burst_count']} (baseline {bl_bursts})")
        print(f"    OOS PnL: {best['total_oos']:+.0f}%")
    else:
        print("\n  No scenario improves on baseline.")

    results['sections']['recommendation'] = {
        'improved_candidates': improved[:5] if improved else [],
        'best': improved[0] if improved else None,
    }

    # ── Save ──
    elapsed = time.time() - t0
    results['meta']['elapsed_seconds'] = round(elapsed, 1)
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULT_FILE}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
