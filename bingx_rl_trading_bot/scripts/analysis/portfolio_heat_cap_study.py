#!/usr/bin/env python3
"""
Portfolio Heat Cap Study — Reactive portfolio-level risk management.

Concept:
  Every bar, calculate total unrealized PnL across all open positions.
  When total unrealized loss exceeds `heat_cap` threshold, force-close
  the worst position(s) until heat is below threshold or only 1 pos left.

This is purely REACTIVE — responds to current portfolio state, no prediction.

Sweep: heat_cap = [3, 4, 5, 6, 7, 8, 10, 999 (baseline/disabled)]

Standard Research Protocol:
  - Entry: next bar Open after signal
  - Exit: Intrabar High/Low (distance-based)
  - Sizing: Compound
  - Fee: 0.05% x 2 = 0.10%, Slippage: 0.02%
  - Leverage: 3x
  - WF: 3-fold expanding window
  - MC: 3 seeds (42, 123, 7)
"""

import json
import numpy as np
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scanner"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production" / "pattern_5m"))

from pattern_scanner import (
    load_and_classify, build_signal_index, find_neutral_window,
    calc_stats_compound,
    compute_atr_ratio, compute_ema_slope,
    _check_exit_npos,
    LEVERAGE, FEE_PCT, MAX_BARS, MAX_DAILY_LOSS_PCT,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, TIMEOUT_BARS,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_REGIME_MULT, SLIPPAGE_BUFFER,
    NPOS_EMA_PERIOD, NPOS_EMA_LOOKBACK,
)

DATA_FILE = "data/btc_5m_270days_reclassified.csv"
PATTERNS_FILE = "results/dynamic_patterns.json"
OUTPUT_FILE = "results/portfolio_heat_cap_study.json"

TP_SCALE_FACTOR = 0.72

HEAT_CAPS = [3, 4, 5, 6, 7, 8, 10, 999]  # 999 = disabled (baseline)

MC_SEEDS = [42, 123, 7]
MC_SIMS = 5000

NPOS_DEFAULTS = dict(
    n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
    regime_mult=DEFAULT_REGIME_MULT,
    agg_risk_counter=DEFAULT_AGG_RISK_COUNTER, agg_risk_with=DEFAULT_AGG_RISK_WITH,
    momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK, momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
    momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
    clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
    timeout_bars=TIMEOUT_BARS, cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
)


def load_patterns(filepath=PATTERNS_FILE):
    with open(filepath) as f:
        data = json.load(f)
    details = data.get('pattern_details') or {}
    result = {}
    for k, v in details.items():
        result[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': round(max(0.3, v['tp'] * TP_SCALE_FACTOR), 3),
            'sl': v['sl'],
        }
    return result


def build_signal_tuples(patterns, sig_idx):
    tuples = []
    for k, v in patterns.items():
        pat_name = v.get('pattern') or k.rsplit('_', 1)[0]
        if pat_name in sig_idx:
            for bar in sig_idx[pat_name]:
                tuples.append((bar, k, v['direction'], v['tp'], v['sl']))
    return tuples


def portfolio_npos_heat(signal_tuples, opens, highs, lows, closes, n_bars,
                        atr_ratio, ema_slope, start_bar, end_bar,
                        heat_cap_pct=999.0,
                        n_slots=DEFAULT_N_SLOTS,
                        direction_cap=DEFAULT_DIRECTION_CAP,
                        regime_mult=DEFAULT_REGIME_MULT,
                        agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
                        agg_risk_with=DEFAULT_AGG_RISK_WITH,
                        momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
                        momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
                        momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
                        clamp_lo=DEFAULT_ATR_CLAMP_LO,
                        clamp_hi=DEFAULT_ATR_CLAMP_HI,
                        timeout_bars=TIMEOUT_BARS,
                        cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT):
    """N-position portfolio simulator with heat cap logic.

    Identical to production portfolio_npos except: after exits and before
    entries each bar, checks total unrealized PnL. If total unrealized
    loss exceeds heat_cap_pct, force-closes worst position(s).

    Returns:
        (trades_list, stats_dict)
    """
    size_pct = 100.0 / n_slots
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_dd_mtm = 0.0

    max_corr_loss = 0.0
    max_sim_positions = 0
    total_blocked = {'momentum': 0, 'agg_risk': 0, 'dir_cap': 0,
                     'dup_pat': 0, 'max_pos': 0}
    corr_events = []

    # Heat cap tracking
    heat_exit_count = 0
    heat_exit_pnls = []
    max_bar_dir_loss = 0.0  # max single-bar directional loss

    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    # Filter and sort signals in range
    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        # 1. Check exits (same as production)
        closed_slots = []
        bar_pnl_sum = 0.0
        bar_sl_count = 0

        for pos in positions:
            result = _check_exit_npos(pos, bar, opens, highs, lows, n_bars,
                                      atr_ratio, fee, clamp_lo, clamp_hi,
                                      timeout_bars)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                result['size_mult'] = sm
                pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio
                if result['reason'] == 'SL':
                    bar_sl_count += 1

        # Cascade SL tightening (same as production)
        if cascade_tighten_pct > 0 and bar_sl_count > 0:
            keep_ratio = 1.0 - cascade_tighten_pct / 100.0
            sl_directions = set()
            for t in trades[len(trades) - len(closed_slots):]:
                if t.get('reason') == 'SL':
                    sl_directions.add(t['direction'])
            for sl_dir in sl_directions:
                for pos in positions:
                    if pos['slot'] in closed_slots:
                        continue
                    if pos['direction'] != sl_dir:
                        continue
                    sig = pos['signal_bar']
                    if (atr_ratio is not None and sig < len(atr_ratio)
                            and not np.isnan(atr_ratio[sig])):
                        r = max(clamp_lo, min(clamp_hi, atr_ratio[sig]))
                    else:
                        r = 1.0
                    p_sl = pos['sl_pct']
                    if p_sl > 0:
                        r = min(r, MAX_DAILY_LOSS_PCT / LEVERAGE / p_sl)
                    cur_eff_sl = pos.get('eff_sl_override') or (p_sl * r)
                    pos['eff_sl_override'] = cur_eff_sl * keep_ratio

        positions = [p for p in positions if p['slot'] not in closed_slots]

        if bar_pnl_sum < 0 and bar_sl_count >= 2:
            loss_pct = abs(bar_pnl_sum)
            if loss_pct > max_corr_loss:
                max_corr_loss = loss_pct
            corr_events.append((bar, bar_sl_count, loss_pct))

        equity += bar_pnl_sum

        # Track max single-bar directional loss
        if bar_pnl_sum < 0 and abs(bar_pnl_sum) > max_bar_dir_loss:
            max_bar_dir_loss = abs(bar_pnl_sum)

        # ── HEAT CAP CHECK ──
        # Calculate total unrealized PnL across all open positions
        if heat_cap_pct < 900 and positions and bar < n_bars:
            while True:
                if len(positions) <= 1:
                    break

                # Compute unrealized PnL per position
                unr_list = []
                for pi, pos in enumerate(positions):
                    eb = pos['entry_bar']
                    if eb >= n_bars or bar < eb:
                        unr_list.append((pi, 0.0))
                        continue
                    entry_price = opens[eb]
                    if entry_price <= 0:
                        unr_list.append((pi, 0.0))
                        continue
                    if pos['direction'] == 'LONG':
                        unr = (closes[bar] / entry_price - 1) * 100 * LEVERAGE
                    else:
                        unr = (1 - closes[bar] / entry_price) * 100 * LEVERAGE
                    sm = pos.get('size_mult', 1.0)
                    unr_portfolio = unr * (size_pct / 100) * sm
                    unr_list.append((pi, unr_portfolio))

                total_heat = sum(u for _, u in unr_list)

                if total_heat >= -heat_cap_pct:
                    break  # Portfolio heat within bounds

                # Find worst position (most negative unrealized PnL)
                unr_list.sort(key=lambda x: x[1])
                worst_idx, worst_unr = unr_list[0]
                worst_pos = positions[worst_idx]

                # Force-close at current bar close price
                eb = worst_pos['entry_bar']
                entry_price = opens[eb]
                if entry_price <= 0:
                    # Cannot compute PnL; just remove
                    positions.pop(worst_idx)
                    continue

                exit_price = closes[bar]
                if worst_pos['direction'] == 'LONG':
                    pnl = (exit_price / entry_price - 1) * 100 * LEVERAGE
                else:
                    pnl = (1 - exit_price / entry_price) * 100 * LEVERAGE
                pnl -= fee  # Deduct fee for the close

                sm = worst_pos.get('size_mult', 1.0)
                pnl_portfolio = pnl * (size_pct / 100) * sm

                trades.append({
                    'entry_bar': eb,
                    'exit_bar': bar,
                    'pnl_slot': pnl,
                    'reason': 'HEAT_EXIT',
                    'pattern': worst_pos['pattern'],
                    'direction': worst_pos['direction'],
                    'size_mult': sm,
                    'pnl_portfolio': pnl_portfolio,
                    'drop': False,
                })

                equity += pnl_portfolio
                heat_exit_count += 1
                heat_exit_pnls.append(pnl_portfolio)

                if pnl_portfolio < 0 and abs(pnl_portfolio) > max_bar_dir_loss:
                    max_bar_dir_loss = abs(pnl_portfolio)

                positions.pop(worst_idx)

        if equity > peak_equity:
            peak_equity = equity

        # Momentum guard
        if momentum_lookback > 0 and momentum_threshold > 0 and bar >= momentum_lookback:
            price_now = closes[bar]
            price_ago = closes[bar - momentum_lookback]
            if price_ago > 0:
                pct_change = (price_now / price_ago - 1) * 100
                if pct_change > momentum_threshold:
                    momentum_pause_until['SHORT'] = bar + momentum_cooldown
                elif pct_change < -momentum_threshold:
                    momentum_pause_until['LONG'] = bar + momentum_cooldown

        # 2. Process entries (same as production)
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= n_slots:
                total_blocked['max_pos'] += 1
                continue

            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= direction_cap:
                total_blocked['dir_cap'] += 1
                continue

            if any(p['pattern'] == pat for p in positions):
                total_blocked['dup_pat'] += 1
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue

            if momentum_lookback > 0 and bar < momentum_pause_until.get(direction, -1):
                total_blocked['momentum'] += 1
                continue

            # Regime sizing
            sm = 1.0
            if regime_mult is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = regime_mult
                elif s <= 0 and direction == 'LONG':
                    sm = regime_mult

            # Aggregate risk cap
            if agg_risk_counter > 0 or agg_risk_with > 0:
                is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = ((is_uptrend and direction == 'SHORT') or
                              (not is_uptrend and direction == 'LONG'))
                cap_pct = agg_risk_counter if is_counter else agg_risk_with

                existing_exposure = 0.0
                for p in positions:
                    if p['direction'] == direction:
                        p_sl = p['sl_pct']
                        p_sig = p['signal_bar']
                        if (atr_ratio is not None and p_sig < len(atr_ratio)
                                and not np.isnan(atr_ratio[p_sig])):
                            p_r = max(clamp_lo, min(clamp_hi, atr_ratio[p_sig]))
                        else:
                            p_r = 1.0
                        if p_sl > 0:
                            p_r = min(p_r, MAX_DAILY_LOSS_PCT / LEVERAGE / p_sl)
                        p_eff_sl = p_sl * p_r
                        p_sm = p.get('size_mult', 1.0)
                        existing_exposure += p_eff_sl * (1.0 / n_slots) * LEVERAGE * p_sm

                new_r = 1.0
                if (atr_ratio is not None and sig_bar < len(atr_ratio)
                        and not np.isnan(atr_ratio[sig_bar])):
                    new_r = max(clamp_lo, min(clamp_hi, atr_ratio[sig_bar]))
                if sl_pct > 0:
                    new_r = min(new_r, MAX_DAILY_LOSS_PCT / LEVERAGE / sl_pct)
                new_eff_sl = sl_pct * new_r
                new_exposure = new_eff_sl * (1.0 / n_slots) * LEVERAGE * sm

                if existing_exposure + new_exposure > cap_pct:
                    total_blocked['agg_risk'] += 1
                    continue

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'size_mult': sm,
            })

        if len(positions) > max_sim_positions:
            max_sim_positions = len(positions)

        # Mark-to-market MDD
        if positions and bar < n_bars:
            mtm_equity = equity
            for pos in positions:
                eb = pos['entry_bar']
                if eb >= n_bars or bar < eb:
                    continue
                entry_price = opens[eb]
                if entry_price <= 0:
                    continue
                if pos['direction'] == 'LONG':
                    unr = (closes[bar] / entry_price - 1) * 100 * LEVERAGE
                else:
                    unr = (1 - closes[bar] / entry_price) * 100 * LEVERAGE
                sm = pos.get('size_mult', 1.0)
                mtm_equity += unr * (size_pct / 100) * sm
            if mtm_equity > peak_equity:
                peak_equity = mtm_equity
            dd = (peak_equity - mtm_equity) / peak_equity * 100 if peak_equity > 0 else 0
            if dd > max_dd_mtm:
                max_dd_mtm = dd
        elif not positions:
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            if dd > max_dd_mtm:
                max_dd_mtm = dd

    # Force-close remaining at end
    for pos in positions:
        entry_bar = pos['entry_bar']
        if entry_bar >= n_bars:
            continue
        entry = opens[entry_bar]
        if entry <= 0:
            continue
        exit_bar = min(end_bar - 1, n_bars - 1)
        exit_price = opens[exit_bar]
        if pos['direction'] == 'LONG':
            pnl = (exit_price / entry - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / entry) * 100 * LEVERAGE
        pnl -= fee
        sm = pos.get('size_mult', 1.0)
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'OOS_END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm,
            'pnl_portfolio': pnl * (size_pct / 100) * sm,
        })

    stats = {
        'max_corr_loss': round(max_corr_loss, 2),
        'max_sim_positions': max_sim_positions,
        'corr_events': len(corr_events),
        'blocked': total_blocked,
        'mdd_mtm': round(max_dd_mtm, 2),
        'heat_exits': heat_exit_count,
        'heat_exit_avg_pnl': round(np.mean(heat_exit_pnls), 3) if heat_exit_pnls else 0.0,
        'heat_exit_total_pnl': round(sum(heat_exit_pnls), 3) if heat_exit_pnls else 0.0,
        'max_bar_dir_loss': round(max_bar_dir_loss, 3),
    }
    return trades, stats


def run_heat(signal_tuples, opens, highs, lows, closes, n_bars,
             atr_ratio, ema_slope, start_bar, end_bar,
             heat_cap_pct=999.0, **extra):
    kwargs = {**NPOS_DEFAULTS}
    kwargs.update(extra)
    trades, raw = portfolio_npos_heat(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        heat_cap_pct=heat_cap_pct, **kwargs
    )
    stats = calc_stats_compound(trades)
    if raw.get('mdd_mtm', 0) > 0:
        stats['mdd'] = raw['mdd_mtm']
        stats['pnl_mdd'] = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
    stats.update({k: v for k, v in raw.items() if k not in stats})
    return trades, stats


def run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, ns, ne, heat_cap_pct=999.0,
           n_folds=3, **extra):
    total = ne - ns
    min_train = total // 3
    fold_size = total // (n_folds + 1)
    results = []
    for fold in range(n_folds):
        train_end = ns + min_train + fold_size * fold
        test_start = train_end
        test_end = min(train_end + fold_size, ne)
        if test_start >= ne or test_end <= test_start:
            continue
        _, stats = run_heat(signal_tuples, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, test_start, test_end,
                            heat_cap_pct=heat_cap_pct, **extra)
        results.append({
            'fold': fold + 1, 'pnl': stats.get('pnl', 0),
            'wr': stats.get('wr', 0), 'trades': stats.get('trades', 0),
            'mdd': stats.get('mdd', 0),
            'heat_exits': stats.get('heat_exits', 0),
        })
    oos_pnl = sum(r['pnl'] for r in results)
    all_pass = all(r['pnl'] > 0 for r in results) if results else False
    return results, oos_pnl, all_pass


def run_mc(trades, n_sims=MC_SIMS, seed=42):
    """Monte Carlo sign-randomization test."""
    if len(trades) < 25:
        return 1.0  # Insufficient trades
    pnls = [t['pnl_portfolio'] for t in trades]
    real_pnl = sum(pnls)
    rng = np.random.RandomState(seed)
    pnl_arr = np.array(pnls)
    count_better = 0
    for _ in range(n_sims):
        signs = rng.choice([-1, 1], size=len(pnl_arr))
        sim_pnl = np.sum(pnl_arr * signs)
        if sim_pnl >= real_pnl:
            count_better += 1
    return count_better / n_sims


def compute_rr(trades):
    """Compute R:R ratio (avg win / avg loss absolute)."""
    wins = [t['pnl_portfolio'] for t in trades if t['pnl_portfolio'] > 0]
    losses = [abs(t['pnl_portfolio']) for t in trades if t['pnl_portfolio'] <= 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 1e-9
    return round(avg_win / avg_loss, 3) if avg_loss > 0 else 999.0


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 70)
    print("PORTFOLIO HEAT CAP STUDY")
    print("Reactive force-close worst positions when total unrealized loss")
    print("exceeds threshold.")
    print("=" * 70)

    # Load data
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    n_bars = len(df)
    type_codes = df['candle_type'].values

    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)

    signal_index = build_signal_index(type_codes, n_bars)
    nw = find_neutral_window(closes)
    ns, ne = nw
    print(f"Data: {n_bars} bars, neutral window: {ns}-{ne} ({(ne - ns) // 288:.0f}d)")

    current_patterns = load_patterns()
    print(f"Patterns: {len(current_patterns)} (TP x{TP_SCALE_FACTOR})")

    signal_tuples = build_signal_tuples(current_patterns, signal_index)
    print(f"Signal tuples: {len(signal_tuples)}")

    all_results = {
        "study": "portfolio_heat_cap",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": {
            "fee": f"{FEE_PCT}%", "leverage": LEVERAGE,
            "tp_scale": TP_SCALE_FACTOR,
            "n_slots": DEFAULT_N_SLOTS, "dir_cap": DEFAULT_DIRECTION_CAP,
            "timeout": TIMEOUT_BARS, "cascade": DEFAULT_CASCADE_TIGHTEN_PCT,
            "agg_risk": f"{DEFAULT_AGG_RISK_COUNTER}/{DEFAULT_AGG_RISK_WITH}",
            "data_bars": n_bars, "neutral_window": f"{ns}-{ne}",
        },
        "heat_cap_sweep": {},
    }

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Heat Cap Sweep — IS metrics
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1: IS Sweep — heat_cap = " + str(HEAT_CAPS))
    print("=" * 70)

    baseline_stats = None

    for hc in HEAT_CAPS:
        label = f"{hc}%" if hc < 900 else "BASELINE(disabled)"
        print(f"\n--- heat_cap={label} ---")

        trades, stats = run_heat(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, ns, ne, heat_cap_pct=float(hc)
        )

        rr = compute_rr(trades)

        # Count heat exits
        heat_exits = [t for t in trades if t.get('reason') == 'HEAT_EXIT']
        n_heat = len(heat_exits)
        heat_pct = round(n_heat / stats['trades'] * 100, 1) if stats['trades'] > 0 else 0
        heat_avg_pnl = round(np.mean([t['pnl_portfolio'] for t in heat_exits]), 3) if heat_exits else 0

        # Trades per day
        n_days = (ne - ns) / 288
        tpd = round(stats['trades'] / n_days, 1) if n_days > 0 else 0

        print(f"  WR: {stats['wr']}%  PnL: {stats['pnl']:+.1f}%  "
              f"MDD: {stats['mdd']:.2f}%  PnL/MDD: {stats.get('pnl_mdd', 0):.1f}x  "
              f"R:R: {rr}")
        print(f"  Trades: {stats['trades']} ({tpd}/day)  "
              f"Heat exits: {n_heat} ({heat_pct}%)  "
              f"Heat avg PnL: {heat_avg_pnl:+.3f}%")
        print(f"  Max bar dir loss: {stats.get('max_bar_dir_loss', 0):.3f}%  "
              f"Max corr loss: {stats.get('max_corr_loss', 0):.2f}%")

        if hc >= 900:
            baseline_stats = stats

        entry = {
            "heat_cap": hc,
            "is": {
                "wr": stats['wr'], "pnl": stats['pnl'],
                "mdd": stats['mdd'],
                "pnl_mdd": stats.get('pnl_mdd', 0),
                "trades": stats['trades'], "trades_per_day": tpd,
                "rr": rr,
                "heat_exits": n_heat,
                "heat_exit_pct": heat_pct,
                "heat_exit_avg_pnl": heat_avg_pnl,
                "max_bar_dir_loss": stats.get('max_bar_dir_loss', 0),
                "max_corr_loss": stats.get('max_corr_loss', 0),
            },
        }

        all_results["heat_cap_sweep"][str(hc)] = entry

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: WF Validation for each heat cap
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: Walk-Forward Validation (3-fold expanding window)")
    print("=" * 70)

    for hc in HEAT_CAPS:
        label = f"{hc}%" if hc < 900 else "BASELINE"
        wf_folds, wf_oos, wf_pass = run_wf(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, ns, ne, heat_cap_pct=float(hc)
        )
        tag = "PASS" if wf_pass else "FAIL"
        fold_str = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%"
                             f"(h:{r.get('heat_exits', 0)})" for r in wf_folds)
        print(f"  heat_cap={label:>8s}: OOS {wf_oos:+.1f}% [{tag}]  {fold_str}")

        all_results["heat_cap_sweep"][str(hc)]["wf"] = {
            "folds": wf_folds,
            "oos_pnl": round(wf_oos, 1),
            "all_pass": wf_pass,
        }

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: MC Discrimination Test
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3: Monte Carlo Discrimination (3-seed)")
    print("=" * 70)

    for hc in HEAT_CAPS:
        label = f"{hc}%" if hc < 900 else "BASELINE"
        trades, _ = run_heat(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, ns, ne, heat_cap_pct=float(hc)
        )
        p_values = []
        for seed in MC_SEEDS:
            p = run_mc(trades, n_sims=MC_SIMS, seed=seed)
            p_values.append(p)
        max_p = max(p_values)
        disc = "DISC" if max_p < 0.01 else "NON-DISC"
        print(f"  heat_cap={label:>8s}: p={p_values}  max_p={max_p:.4f}  [{disc}]")

        all_results["heat_cap_sweep"][str(hc)]["mc"] = {
            "p_values": [round(p, 4) for p in p_values],
            "max_p": round(max_p, 4),
            "discriminating": max_p < 0.01,
        }

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Comparison vs Baseline
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 4: Comparison vs Baseline")
    print("=" * 70)

    bl = all_results["heat_cap_sweep"]["999"]
    bl_pnl = bl["is"]["pnl"]
    bl_mdd = bl["is"]["mdd"]
    bl_pnlmdd = bl["is"]["pnl_mdd"]
    bl_oos = bl["wf"]["oos_pnl"]

    print(f"\n  {'heat_cap':>10s} | {'IS PnL':>8s} | {'IS MDD':>7s} | {'PnL/MDD':>8s} | "
          f"{'OOS PnL':>8s} | {'WF':>4s} | {'Heat#':>5s} | {'Verdict':>10s}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*4}-+-{'-'*5}-+-{'-'*10}")

    comparison = []
    for hc in HEAT_CAPS:
        r = all_results["heat_cap_sweep"][str(hc)]
        is_pnl = r["is"]["pnl"]
        is_mdd = r["is"]["mdd"]
        is_pmdd = r["is"]["pnl_mdd"]
        oos = r["wf"]["oos_pnl"]
        wf_tag = "PASS" if r["wf"]["all_pass"] else "FAIL"
        n_heat = r["is"]["heat_exits"]

        # Verdict logic
        pnl_better = is_pnl > bl_pnl * 0.95  # Allow 5% PnL loss
        mdd_better = is_mdd < bl_mdd * 0.90  # MDD must improve by >10%
        oos_better = oos > bl_oos * 0.90      # OOS must not degrade >10%

        if hc >= 900:
            verdict = "BASELINE"
        elif mdd_better and pnl_better and oos_better and wf_tag == "PASS":
            verdict = "GO"
        elif is_pnl < bl_pnl * 0.80:
            verdict = "HARMFUL"
        else:
            verdict = "KEEP_BASE"

        label = f"{hc}%" if hc < 900 else "disabled"
        print(f"  {label:>10s} | {is_pnl:+8.1f} | {is_mdd:7.2f} | {is_pmdd:8.1f} | "
              f"{oos:+8.1f} | {wf_tag:>4s} | {n_heat:5d} | {verdict:>10s}")

        comparison.append({
            "heat_cap": hc,
            "vs_baseline": {
                "pnl_delta": round(is_pnl - bl_pnl, 1),
                "mdd_delta": round(is_mdd - bl_mdd, 2),
                "pnl_mdd_delta": round(is_pmdd - bl_pnlmdd, 1),
                "oos_delta": round(oos - bl_oos, 1),
            },
            "verdict": verdict,
        })

    all_results["comparison"] = comparison

    # Final summary
    elapsed = time.time() - start_time
    all_results["elapsed_seconds"] = round(elapsed, 1)

    # Determine recommendation
    best_go = None
    for c in comparison:
        if c["verdict"] == "GO":
            if best_go is None or c["vs_baseline"]["pnl_mdd_delta"] > best_go["vs_baseline"]["pnl_mdd_delta"]:
                best_go = c

    if best_go:
        rec = f"GO: heat_cap={best_go['heat_cap']}% — MDD improvement with acceptable PnL trade-off"
    else:
        rec = "KEEP_BASELINE: No heat_cap value improves MDD sufficiently without harmful PnL degradation"

    all_results["recommendation"] = rec
    print(f"\n  RECOMMENDATION: {rec}")
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_FILE}")
