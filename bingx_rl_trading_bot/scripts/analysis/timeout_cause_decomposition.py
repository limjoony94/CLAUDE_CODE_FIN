#!/usr/bin/env python3
"""Timeout Cause Decomposition Study.

Research question:
  24h timeout이 PnL을 개선하는 이유가 무엇인가?
  (A) 순수 시간 효과: 오래 들고 있으면 결과가 나빠진다
  (B) 방향 효과: 역방향으로 이동한 포지션은 회복이 어렵다
  (C) 둘 다: 시간과 역방향 이동이 상호작용

Approach:
  Phase 1: 무제한 시뮬레이션 — 각 거래의 시간별 unrealized PnL 경로 추적
  Phase 2: 24h 시점 unrealized PnL vs 최종 결과 상관관계
  Phase 3: 시간 vs adverse excursion 예측력 비교
  Phase 4: 조건부 분석 — "24h에 손실 중인 포지션"의 이후 경로

Protocol: Entry next-bar open, intrabar exit, Fee 0.10%, Slippage 0.02%, Lev 3x.
"""
import os, sys, json, time, logging
import numpy as np, pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'scripts', 'scanner'))

from pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    compute_atr_ratio, compute_ema_slope,
    FEE_PCT, LEVERAGE, MAX_BARS, SLIPPAGE_BUFFER,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    TIMEOUT_BARS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'timeout_cause_decomposition.json')

# Extended timeout for full path tracking
EXTENDED_BARS = 864  # 72h — track what happens if we hold longer


def load_patterns():
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    details = data.get('pattern_details') or {}
    result = {}
    for k, v in details.items():
        result[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp_pct': v['tp'], 'sl_pct': v['sl'],
        }
    return result


def simulate_single_trade(entry_bar, direction, tp_pct, sl_pct,
                          opens, highs, lows, closes, n_bars, atr_ratio, ns, ne):
    """Simulate a single trade with full path tracking.

    Returns dict with:
      - exit info (bar, reason, pnl)
      - path: unrealized PnL at each bar from entry to min(entry+EXTENDED_BARS, ne)
      - mae/mfe at various checkpoints
    """
    eb = entry_bar + 1  # entry at next bar open
    if eb >= ne or eb >= n_bars:
        return None

    entry_price = opens[eb]
    if entry_price <= 0:
        return None

    # ATR scaling
    nr = 1.0
    if atr_ratio is not None and eb < len(atr_ratio):
        r = atr_ratio[eb]
        if not np.isnan(r):
            nr = max(DEFAULT_ATR_CLAMP_LO, min(DEFAULT_ATR_CLAMP_HI, r))

    tp_dist = tp_pct * nr / 100.0
    sl_dist = sl_pct * nr / 100.0

    if direction == 'LONG':
        tp_price = entry_price * (1 + tp_dist)
        sl_price = entry_price * (1 - sl_dist)
    else:
        tp_price = entry_price * (1 - tp_dist)
        sl_price = entry_price * (1 + sl_dist)

    # Track path
    max_bar = min(eb + EXTENDED_BARS, ne, n_bars)
    path = []  # (bars_held, unrealized_pnl_pct)
    exit_bar = None
    exit_reason = None
    exit_pnl = None

    for j in range(eb, max_bar):
        bars_held = j - eb
        h, l, c = highs[j], lows[j], closes[j]

        # Check TP/SL hit (intrabar, distance-based)
        if direction == 'LONG':
            tp_hit = h >= tp_price
            sl_hit = l <= sl_price
        else:
            tp_hit = l <= tp_price
            sl_hit = h >= sl_price

        if tp_hit and sl_hit:
            # Same-bar: distance-based resolution
            tp_d = abs(tp_price - opens[j])
            sl_d = abs(sl_price - opens[j])
            if tp_d <= sl_d:
                tp_hit, sl_hit = True, False
            else:
                tp_hit, sl_hit = False, True

        if tp_hit and exit_bar is None:
            exit_bar = j
            exit_reason = 'TP'
            raw = tp_dist
            exit_pnl = (raw - FEE_PCT / 100 * 2) * LEVERAGE * 100
        elif sl_hit and exit_bar is None:
            exit_bar = j
            exit_reason = 'SL'
            raw = -sl_dist
            exit_pnl = (raw - FEE_PCT / 100 * 2) * LEVERAGE * 100

        # Unrealized PnL at close
        if direction == 'LONG':
            raw_unr = (c - entry_price) / entry_price
        else:
            raw_unr = (entry_price - c) / entry_price
        unr_pnl = (raw_unr - FEE_PCT / 100 * 2) * LEVERAGE * 100

        path.append((bars_held, round(unr_pnl, 3)))

        # If already exited by TP/SL, continue tracking hypothetical path
        # but mark the actual exit point

    # If no TP/SL hit within extended window
    if exit_bar is None:
        exit_bar = max_bar - 1
        exit_reason = 'EXTENDED_TIMEOUT'
        if direction == 'LONG':
            raw = (closes[exit_bar] - entry_price) / entry_price
        else:
            raw = (entry_price - closes[exit_bar]) / entry_price
        exit_pnl = (raw - FEE_PCT / 100 * 2) * LEVERAGE * 100

    # Extract checkpoints
    checkpoints = {}
    for cp_bars in [72, 144, 216, 288, 432, 576, 720, 864]:
        if cp_bars < len(path):
            checkpoints[f'bar_{cp_bars}'] = path[cp_bars][1]

    # MAE/MFE
    pnl_values = [p[1] for p in path[:max(1, (exit_bar - eb + 1) if exit_reason != 'EXTENDED_TIMEOUT' else len(path))]]
    mae = min(pnl_values) if pnl_values else 0
    mfe = max(pnl_values) if pnl_values else 0

    return {
        'entry_bar': eb,
        'exit_bar': exit_bar,
        'exit_reason': exit_reason,
        'exit_pnl': round(exit_pnl, 3),
        'bars_held': exit_bar - eb,
        'direction': direction,
        'tp_pct_scaled': round(tp_pct * nr, 3),
        'sl_pct_scaled': round(sl_pct * nr, 3),
        'checkpoints': checkpoints,
        'mae': round(mae, 3),
        'mfe': round(mfe, 3),
        'path': path,  # full path for deep analysis
    }


def phase1_path_analysis(trades):
    """Phase 1: Analyze unrealized PnL paths at key checkpoints."""
    logger.info("=== Phase 1: Path Analysis ===")

    # Group by exit reason
    by_reason = {}
    for t in trades:
        r = t['exit_reason']
        by_reason.setdefault(r, []).append(t)

    for reason, tlist in sorted(by_reason.items()):
        logger.info(f"  {reason}: {len(tlist)} trades, "
                    f"avg PnL {np.mean([t['exit_pnl'] for t in tlist]):.2f}%, "
                    f"avg hold {np.mean([t['bars_held'] for t in tlist]):.0f} bars")

    # At 288 bars (24h), what's the unrealized PnL distribution?
    unr_at_288 = []
    for t in trades:
        cp = t['checkpoints'].get('bar_288')
        if cp is not None:
            unr_at_288.append({
                'unr_pnl_288': cp,
                'exit_reason': t['exit_reason'],
                'exit_pnl': t['exit_pnl'],
                'bars_held': t['bars_held'],
                'direction': t['direction'],
            })

    # Only trades that lasted >= 288 bars (didn't hit TP/SL before 24h)
    alive_at_288 = [u for u in unr_at_288 if u['bars_held'] >= 288]
    exited_before_288 = [u for u in unr_at_288 if u['bars_held'] < 288]

    logger.info(f"\n  Trades alive at 24h: {len(alive_at_288)}")
    logger.info(f"  Trades exited before 24h: {len(exited_before_288)}")

    if alive_at_288:
        pnls = [u['unr_pnl_288'] for u in alive_at_288]
        in_profit = [u for u in alive_at_288 if u['unr_pnl_288'] > 0]
        in_loss = [u for u in alive_at_288 if u['unr_pnl_288'] <= 0]
        logger.info(f"  At 24h — In profit: {len(in_profit)} ({len(in_profit)/len(alive_at_288)*100:.1f}%), "
                    f"In loss: {len(in_loss)} ({len(in_loss)/len(alive_at_288)*100:.1f}%)")
        logger.info(f"  At 24h — Avg unrealized: {np.mean(pnls):.2f}%, "
                    f"Median: {np.median(pnls):.2f}%")

    result = {
        'total_trades': len(trades),
        'by_exit_reason': {
            r: {'count': len(tl),
                'avg_pnl': round(np.mean([t['exit_pnl'] for t in tl]), 3),
                'avg_hold_bars': round(np.mean([t['bars_held'] for t in tl]), 1)}
            for r, tl in by_reason.items()
        },
        'alive_at_288': len(alive_at_288),
        'exited_before_288': len(exited_before_288),
    }

    if alive_at_288:
        pnls_288 = [u['unr_pnl_288'] for u in alive_at_288]
        result['at_288_stats'] = {
            'in_profit_count': len([p for p in pnls_288 if p > 0]),
            'in_loss_count': len([p for p in pnls_288 if p <= 0]),
            'avg_unrealized': round(np.mean(pnls_288), 3),
            'median_unrealized': round(np.median(pnls_288), 3),
            'pct_in_profit': round(len([p for p in pnls_288 if p > 0]) / len(pnls_288) * 100, 1),
        }

    return result


def phase2_time_vs_direction(trades):
    """Phase 2: Disentangle time effect vs adverse movement effect.

    For trades alive at 24h:
    - Group by unrealized PnL at 24h (profit vs loss, binned)
    - Track what happens next (do they recover? get worse?)
    - Compare: "24h + in profit" vs "24h + in loss" → eventual outcome
    """
    logger.info("=== Phase 2: Time vs Direction Decomposition ===")

    alive_at_288 = [t for t in trades if t['bars_held'] >= 288]
    if not alive_at_288:
        return {'error': 'no trades alive at 288'}

    # Bin by unrealized PnL at bar 288
    bins = {
        'deep_loss': [],      # < -3%
        'moderate_loss': [],   # -3% to -1%
        'slight_loss': [],     # -1% to 0%
        'slight_profit': [],   # 0% to +1%
        'moderate_profit': [], # +1% to +3%
        'deep_profit': [],     # > +3%
    }

    for t in alive_at_288:
        cp = t['checkpoints'].get('bar_288', 0)
        if cp < -3:
            bins['deep_loss'].append(t)
        elif cp < -1:
            bins['moderate_loss'].append(t)
        elif cp < 0:
            bins['slight_loss'].append(t)
        elif cp < 1:
            bins['slight_profit'].append(t)
        elif cp < 3:
            bins['moderate_profit'].append(t)
        else:
            bins['deep_profit'].append(t)

    result = {}
    for bin_name, tlist in bins.items():
        if not tlist:
            result[bin_name] = {'count': 0}
            continue

        # What happens after 24h?
        # Check checkpoints at 432 (36h), 576 (48h), 864 (72h)
        recovery_data = {
            'count': len(tlist),
            'avg_unr_at_288': round(np.mean([t['checkpoints'].get('bar_288', 0) for t in tlist]), 3),
        }

        for cp_name, cp_bars in [('36h', 432), ('48h', 576), ('72h', 864)]:
            cp_pnls = [t['checkpoints'].get(f'bar_{cp_bars}') for t in tlist
                       if t['checkpoints'].get(f'bar_{cp_bars}') is not None]
            if cp_pnls:
                recovery_data[f'avg_unr_{cp_name}'] = round(np.mean(cp_pnls), 3)
                recovery_data[f'improved_{cp_name}'] = round(
                    sum(1 for i, t in enumerate(tlist)
                        if t['checkpoints'].get(f'bar_{cp_bars}') is not None
                        and t['checkpoints'].get(f'bar_{cp_bars}', 0) > t['checkpoints'].get('bar_288', 0))
                    / max(len(cp_pnls), 1) * 100, 1)

        # Final exit PnL (for trades that eventually hit TP/SL)
        tp_exits = [t for t in tlist if t['exit_reason'] == 'TP']
        sl_exits = [t for t in tlist if t['exit_reason'] == 'SL']
        recovery_data['eventual_tp'] = len(tp_exits)
        recovery_data['eventual_sl'] = len(sl_exits)
        recovery_data['eventual_timeout'] = len([t for t in tlist if t['exit_reason'] == 'EXTENDED_TIMEOUT'])
        recovery_data['eventual_tp_pct'] = round(len(tp_exits) / max(len(tlist), 1) * 100, 1)
        recovery_data['avg_final_pnl'] = round(np.mean([t['exit_pnl'] for t in tlist]), 3)

        result[bin_name] = recovery_data

        logger.info(f"  {bin_name}: {len(tlist)}t | at 24h avg {recovery_data['avg_unr_at_288']:+.2f}% "
                    f"| eventual TP {recovery_data['eventual_tp_pct']:.0f}% "
                    f"| final avg {recovery_data['avg_final_pnl']:+.2f}%")

    return result


def phase3_predictive_power(trades):
    """Phase 3: Which is more predictive — time held or adverse excursion?

    Compare correlation of:
    (A) bars_held at exit → exit_pnl
    (B) MAE (worst unrealized PnL) → exit_pnl
    (C) unrealized PnL at 24h → exit_pnl (for trades alive at 24h)
    """
    logger.info("=== Phase 3: Predictive Power — Time vs Adverse Excursion ===")

    # Only non-TP/SL exits for meaningful analysis (TP/SL are deterministic)
    # Use all trades for path analysis
    alive_at_288 = [t for t in trades if t['bars_held'] >= 288]

    # (A) Time vs PnL correlation
    all_holds = np.array([t['bars_held'] for t in trades])
    all_pnls = np.array([t['exit_pnl'] for t in trades])
    corr_time_pnl = np.corrcoef(all_holds, all_pnls)[0, 1] if len(trades) > 2 else 0

    # (B) MAE vs PnL correlation
    all_maes = np.array([t['mae'] for t in trades])
    corr_mae_pnl = np.corrcoef(all_maes, all_pnls)[0, 1] if len(trades) > 2 else 0

    # (C) For trades alive at 24h: unrealized at 24h vs final PnL
    if alive_at_288:
        unr_288 = np.array([t['checkpoints'].get('bar_288', 0) for t in alive_at_288])
        final_pnls = np.array([t['exit_pnl'] for t in alive_at_288])
        corr_unr288_final = np.corrcoef(unr_288, final_pnls)[0, 1] if len(alive_at_288) > 2 else 0

        # MAE for these trades
        maes_288 = np.array([t['mae'] for t in alive_at_288])
        corr_mae288_final = np.corrcoef(maes_288, final_pnls)[0, 1] if len(alive_at_288) > 2 else 0
    else:
        corr_unr288_final = 0
        corr_mae288_final = 0

    logger.info(f"  Correlation (all trades):")
    logger.info(f"    Time held vs PnL: r = {corr_time_pnl:.3f}")
    logger.info(f"    MAE vs PnL:       r = {corr_mae_pnl:.3f}")
    logger.info(f"  Correlation (alive at 24h):")
    logger.info(f"    Unrealized@24h vs Final PnL: r = {corr_unr288_final:.3f}")
    logger.info(f"    MAE vs Final PnL:            r = {corr_mae288_final:.3f}")

    # Determine primary cause
    if abs(corr_unr288_final) > abs(corr_time_pnl) * 1.2:
        primary = 'DIRECTION'
        explanation = '24h 시점의 unrealized PnL(방향)이 시간보다 예측력이 높음'
    elif abs(corr_time_pnl) > abs(corr_unr288_final) * 1.2:
        primary = 'TIME'
        explanation = '보유 시간 자체가 방향보다 예측력이 높음'
    else:
        primary = 'BOTH'
        explanation = '시간과 방향이 비슷한 수준의 예측력'

    logger.info(f"  Primary cause: {primary} — {explanation}")

    return {
        'correlations': {
            'time_vs_pnl': round(corr_time_pnl, 4),
            'mae_vs_pnl': round(corr_mae_pnl, 4),
            'unr_288_vs_final_pnl': round(corr_unr288_final, 4),
            'mae_288_vs_final_pnl': round(corr_mae288_final, 4),
        },
        'primary_cause': primary,
        'explanation': explanation,
    }


def phase4_conditional_analysis(trades):
    """Phase 4: If we only close positions that are IN LOSS at 24h, what happens?

    Compare:
    (A) Timeout ALL at 24h (current)
    (B) Timeout only LOSING positions at 24h, let winners run
    (C) No timeout at all
    """
    logger.info("=== Phase 4: Conditional Timeout Analysis ===")

    alive_at_288 = [t for t in trades if t['bars_held'] >= 288]
    exited_before = [t for t in trades if t['bars_held'] < 288]

    if not alive_at_288:
        return {'error': 'no trades alive at 288'}

    # Split alive-at-288 by unrealized PnL
    in_profit_at_288 = [t for t in alive_at_288 if t['checkpoints'].get('bar_288', 0) > 0]
    in_loss_at_288 = [t for t in alive_at_288 if t['checkpoints'].get('bar_288', 0) <= 0]

    logger.info(f"  Alive at 24h: {len(alive_at_288)} "
                f"(profit: {len(in_profit_at_288)}, loss: {len(in_loss_at_288)})")

    # Scenario A: Current (timeout ALL at 24h)
    # PnL = exited_before exits + timeout at 24h for all alive
    pnl_a_parts = [t['exit_pnl'] for t in exited_before]
    for t in alive_at_288:
        pnl_a_parts.append(t['checkpoints'].get('bar_288', 0))

    # Scenario B: Timeout only LOSING at 24h, let winners continue to TP/SL/72h
    pnl_b_parts = [t['exit_pnl'] for t in exited_before]
    for t in in_loss_at_288:
        pnl_b_parts.append(t['checkpoints'].get('bar_288', 0))  # close at 24h
    for t in in_profit_at_288:
        pnl_b_parts.append(t['exit_pnl'])  # let run to natural exit (TP/SL/72h)

    # Scenario C: No timeout — let all run to TP/SL/72h
    pnl_c_parts = [t['exit_pnl'] for t in trades]  # all trades run to natural exit

    scenarios = {
        'A_timeout_all_24h': {
            'total_pnl': round(sum(pnl_a_parts), 1),
            'avg_pnl': round(np.mean(pnl_a_parts), 3),
            'trades': len(pnl_a_parts),
        },
        'B_timeout_losers_only': {
            'total_pnl': round(sum(pnl_b_parts), 1),
            'avg_pnl': round(np.mean(pnl_b_parts), 3),
            'trades': len(pnl_b_parts),
            'let_run': len(in_profit_at_288),
            'closed_at_24h': len(in_loss_at_288),
        },
        'C_no_timeout': {
            'total_pnl': round(sum(pnl_c_parts), 1),
            'avg_pnl': round(np.mean(pnl_c_parts), 3),
            'trades': len(pnl_c_parts),
        },
    }

    # What happens to "in profit at 24h" if we let them run?
    if in_profit_at_288:
        profit_runners = {
            'count': len(in_profit_at_288),
            'avg_unr_at_24h': round(np.mean([t['checkpoints'].get('bar_288', 0) for t in in_profit_at_288]), 3),
            'avg_final_pnl': round(np.mean([t['exit_pnl'] for t in in_profit_at_288]), 3),
            'eventual_tp': sum(1 for t in in_profit_at_288 if t['exit_reason'] == 'TP'),
            'eventual_sl': sum(1 for t in in_profit_at_288 if t['exit_reason'] == 'SL'),
            'eventual_timeout': sum(1 for t in in_profit_at_288 if t['exit_reason'] == 'EXTENDED_TIMEOUT'),
            'improved_vs_24h': sum(1 for t in in_profit_at_288 if t['exit_pnl'] > t['checkpoints'].get('bar_288', 0)),
        }
        profit_runners['improved_pct'] = round(
            profit_runners['improved_vs_24h'] / max(len(in_profit_at_288), 1) * 100, 1)
        scenarios['profit_runners_detail'] = profit_runners

        logger.info(f"  Winners let run: {len(in_profit_at_288)}t | "
                    f"avg@24h {profit_runners['avg_unr_at_24h']:+.2f}% → "
                    f"final {profit_runners['avg_final_pnl']:+.2f}% | "
                    f"improved {profit_runners['improved_pct']:.0f}%")

    # What happens to "in loss at 24h" if we DON'T close them?
    if in_loss_at_288:
        loss_holders = {
            'count': len(in_loss_at_288),
            'avg_unr_at_24h': round(np.mean([t['checkpoints'].get('bar_288', 0) for t in in_loss_at_288]), 3),
            'avg_final_pnl': round(np.mean([t['exit_pnl'] for t in in_loss_at_288]), 3),
            'eventual_tp': sum(1 for t in in_loss_at_288 if t['exit_reason'] == 'TP'),
            'eventual_sl': sum(1 for t in in_loss_at_288 if t['exit_reason'] == 'SL'),
            'eventual_timeout': sum(1 for t in in_loss_at_288 if t['exit_reason'] == 'EXTENDED_TIMEOUT'),
            'recovered': sum(1 for t in in_loss_at_288 if t['exit_pnl'] > 0),
        }
        loss_holders['recovery_pct'] = round(
            loss_holders['recovered'] / max(len(in_loss_at_288), 1) * 100, 1)
        scenarios['loss_holders_detail'] = loss_holders

        logger.info(f"  Losers if held: {len(in_loss_at_288)}t | "
                    f"avg@24h {loss_holders['avg_unr_at_24h']:+.2f}% → "
                    f"final {loss_holders['avg_final_pnl']:+.2f}% | "
                    f"recovered {loss_holders['recovery_pct']:.0f}%")

    for name, data in scenarios.items():
        if 'total_pnl' in data:
            logger.info(f"  {name}: total PnL {data['total_pnl']:+.1f}%, avg {data['avg_pnl']:+.3f}%")

    return scenarios


def main():
    t0 = time.time()
    logger.info("=== Timeout Cause Decomposition Study ===")

    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)
    types = df['candle_type'].values
    ns, ne = find_neutral_window(closes)
    logger.info(f"Neutral window: {ns}-{ne} ({(ne-ns)/288:.0f}d)")

    atr_ratio = compute_atr_ratio(highs, lows, closes,
                                   atr_period=DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)

    signal_index = build_signal_index(types, n_bars)
    patterns = load_patterns()

    # Build signal tuples
    signal_tuples = []
    for pk, info in patterns.items():
        pn = info.get('pattern') or pk.rsplit('_', 1)[0]
        if pn in signal_index:
            for sb in signal_index[pn]:
                if ns <= sb < ne:
                    signal_tuples.append((sb, pk, info['direction'],
                                          info['tp_pct'], info['sl_pct']))
    signal_tuples.sort(key=lambda x: x[0])
    logger.info(f"{len(patterns)} patterns, {len(signal_tuples)} signals")

    # Simulate all trades individually (1-pos, no portfolio effects)
    # This isolates the pure time/direction effect without slot/cascade/agg interactions
    logger.info("Simulating individual trades (1-pos, no portfolio effects)...")
    trades = []
    for sig_bar, pk, direction, tp_pct, sl_pct in signal_tuples:
        result = simulate_single_trade(
            sig_bar, direction, tp_pct, sl_pct,
            opens, highs, lows, closes, n_bars, atr_ratio, ns, ne)
        if result:
            result['pattern'] = pk
            trades.append(result)

    logger.info(f"Simulated {len(trades)} trades")

    results = {
        'metadata': {
            'script': 'timeout_cause_decomposition.py',
            'date': pd.Timestamp.now().isoformat(),
            'n_patterns': len(patterns),
            'n_signals': len(signal_tuples),
            'n_trades': len(trades),
            'neutral_window': [int(ns), int(ne)],
            'extended_bars': EXTENDED_BARS,
            'note': '1-pos simulation (no portfolio/cascade effects) to isolate time vs direction',
        },
    }

    results['phase1'] = phase1_path_analysis(trades)
    results['phase2'] = phase2_time_vs_direction(trades)
    results['phase3'] = phase3_predictive_power(trades)
    results['phase4'] = phase4_conditional_analysis(trades)

    elapsed = time.time() - t0
    results['metadata']['runtime_sec'] = round(elapsed, 1)

    # Summary verdict
    p3 = results['phase3']
    p4 = results['phase4']
    primary = p3.get('primary_cause', 'UNKNOWN')

    logger.info(f"\n{'='*70}")
    logger.info(f"TIMEOUT CAUSE DECOMPOSITION — VERDICT")
    logger.info(f"{'='*70}")
    logger.info(f"Primary cause: {primary}")
    logger.info(f"  Time vs PnL correlation:    r = {p3['correlations']['time_vs_pnl']:.4f}")
    logger.info(f"  MAE vs PnL correlation:     r = {p3['correlations']['mae_vs_pnl']:.4f}")
    logger.info(f"  Unr@24h vs Final:           r = {p3['correlations']['unr_288_vs_final_pnl']:.4f}")

    if 'A_timeout_all_24h' in p4 and 'B_timeout_losers_only' in p4:
        a_pnl = p4['A_timeout_all_24h']['total_pnl']
        b_pnl = p4['B_timeout_losers_only']['total_pnl']
        c_pnl = p4['C_no_timeout']['total_pnl']
        logger.info(f"\n  Scenario comparison (total PnL):")
        logger.info(f"    A) Timeout ALL @24h:      {a_pnl:+.1f}%")
        logger.info(f"    B) Timeout LOSERS only:    {b_pnl:+.1f}%")
        logger.info(f"    C) No timeout:             {c_pnl:+.1f}%")
        best = max([('A', a_pnl), ('B', b_pnl), ('C', c_pnl)], key=lambda x: x[1])
        logger.info(f"    Best: {best[0]} ({best[1]:+.1f}%)")

    results['verdict'] = {
        'primary_cause': primary,
        'explanation': p3.get('explanation', ''),
    }

    logger.info(f"Runtime: {elapsed:.1f}s")

    # Strip paths from output to keep file size reasonable
    for t in trades:
        del t['path']

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
