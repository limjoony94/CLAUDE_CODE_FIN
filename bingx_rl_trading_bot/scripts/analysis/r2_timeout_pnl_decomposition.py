#!/usr/bin/env python3
"""R2: Timeout Exit PnL Decomposition.

Research questions:
  1. PnL contribution by exit type (TP, SL, TIMEOUT, OOS_END)
  2. TIMEOUT profit/loss fraction and magnitude distribution
  3. Pre-timeout breakeven SL (move SL to entry at bar N) — improve R:R?
  4. WF validation of top breakeven variants

Protocol: Entry next-bar open, intrabar exit, Fee 0.10%, Slippage 0.02%, Lev 3x, compound.
"""
import os, sys, json, logging, time
import numpy as np, pandas as pd
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'scripts', 'scanner'))
from pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    mc_test, compute_atr_ratio, compute_ema_slope,
    portfolio_npos, calc_stats_compound,
    scan_universe_range, _check_exit_npos,
    FEE_PCT, LEVERAGE, MAX_BARS, SLIPPAGE_BUFFER,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_EDGE_THRESHOLD, DEFAULT_MC_THRESHOLD, DEFAULT_MIN_TRADES,
    MC_SEEDS, TIMEOUT_BARS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'r2_timeout_pnl_decomposition.json')
BREAKEVEN_VARIANTS = {'breakeven_144': 144, 'breakeven_192': 192, 'breakeven_240': 240}
REGIME_MULT = 0.3


def load_patterns():
    with open(PATTERNS_FILE, 'r') as f:
        data = json.load(f)
    details = data.get('pattern_details') or {}
    # Normalize keys: dynamic_patterns.json uses 'tp'/'sl', we need 'tp_pct'/'sl_pct'
    for k, v in details.items():
        if 'tp' in v and 'tp_pct' not in v:
            v['tp_pct'] = v['tp']
        if 'sl' in v and 'sl_pct' not in v:
            v['sl_pct'] = v['sl']
    return details


def build_signal_tuples(patterns, signal_index):
    tuples = []
    for pat_key, info in patterns.items():
        pat_name = info.get('pattern') or pat_key.rsplit('_', 1)[0]
        if pat_name in signal_index:
            for sig_bar in signal_index[pat_name]:
                tuples.append((sig_bar, pat_key, info['direction'],
                               info['tp_pct'], info['sl_pct']))
    return tuples


def _atr_r(atr_ratio, bar, clamp_lo, clamp_hi):
    if atr_ratio is not None and bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
        return max(clamp_lo, min(clamp_hi, atr_ratio[bar]))
    return 1.0


def _check_exit_tracked(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee,
                        clamp_lo, clamp_hi, timeout_bars, breakeven_bar=None):
    """Exit check: TIMEOUT gets market-price PnL (not dropped). Breakeven SL support."""
    entry_bar = pos['entry_bar']
    if bar < entry_bar:
        return None
    entry = opens[entry_bar]
    if entry <= 0:
        return None
    direction = pos['direction']
    r = _atr_r(atr_ratio, pos['signal_bar'], clamp_lo, clamp_hi)
    eff_tp = pos['tp_pct'] * r + SLIPPAGE_BUFFER
    eff_sl_override = pos.get('eff_sl_override')
    eff_sl = max(0.1, (eff_sl_override if eff_sl_override is not None
                       else pos['sl_pct'] * r) - SLIPPAGE_BUFFER)
    hold = bar - entry_bar
    if breakeven_bar is not None and hold >= breakeven_bar:
        eff_sl = SLIPPAGE_BUFFER  # SL at entry

    if direction == 'LONG':
        tp_price = entry * (1 + eff_tp / 100)
        sl_price = entry * (1 - eff_sl / 100)
    else:
        tp_price = entry * (1 - eff_tp / 100)
        sl_price = entry * (1 + eff_sl / 100)

    if hold >= timeout_bars:
        exit_price = opens[bar]
        pnl = ((exit_price / entry - 1) if direction == 'LONG'
               else (1 - exit_price / entry)) * 100 * LEVERAGE - fee
        return {'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
                'reason': 'TIMEOUT', 'drop': False, 'hold_bars': hold}

    h, l = highs[bar], lows[bar]
    hit_tp = (h >= tp_price) if direction == 'LONG' else (l <= tp_price)
    hit_sl = (l <= sl_price) if direction == 'LONG' else (h >= sl_price)
    if not hit_tp and not hit_sl:
        return None
    if hit_tp and hit_sl:
        if abs(tp_price - opens[bar]) <= abs(sl_price - opens[bar]):
            exit_price, reason = tp_price, 'TP'
        else:
            exit_price, reason = sl_price, 'SL'
    elif hit_tp:
        exit_price, reason = tp_price, 'TP'
    else:
        exit_price, reason = sl_price, 'SL'
    pnl = ((exit_price / entry - 1) if direction == 'LONG'
           else (1 - exit_price / entry)) * 100 * LEVERAGE - fee
    return {'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
            'reason': reason, 'drop': False, 'hold_bars': hold}


def portfolio_npos_tracked(signal_tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, start_bar, end_bar,
                           breakeven_bar=None, timeout_bars=TIMEOUT_BARS,
                           cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT):
    """N-pos sim with per-trade tracking. TIMEOUT gets market PnL (not dropped)."""
    size_pct = 100.0 / DEFAULT_N_SLOTS
    fee = FEE_PCT * LEVERAGE
    positions, trades = [], []
    equity, peak_equity = 100.0, 100.0
    momentum_pause = {'LONG': -1, 'SHORT': -1}
    cl, ch = DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI

    sigs = sorted([(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                   if start_bar <= s < end_bar], key=lambda x: x[0])
    si = 0

    for bar in range(start_bar, end_bar):
        closed_slots, bar_pnl, bar_sl_count = [], 0.0, 0
        for pos in positions:
            result = _check_exit_tracked(pos, bar, opens, highs, lows, n_bars,
                                         atr_ratio, fee, cl, ch, timeout_bars, breakeven_bar)
            if result is not None:
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                result['size_mult'] = sm
                result['hold_bars'] = bar - pos['entry_bar']
                pp = result['pnl_slot'] * (size_pct / 100) * sm
                result['pnl_portfolio'] = pp
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl += pp
                if result['reason'] == 'SL':
                    bar_sl_count += 1

        if cascade_tighten_pct > 0 and bar_sl_count > 0:
            keep = 1.0 - cascade_tighten_pct / 100.0
            sl_dirs = {t['direction'] for t in trades[len(trades) - len(closed_slots):]
                       if t.get('reason') == 'SL'}
            for sd in sl_dirs:
                for pos in positions:
                    if pos['slot'] in closed_slots or pos['direction'] != sd:
                        continue
                    r = _atr_r(atr_ratio, pos['signal_bar'], cl, ch)
                    cur = pos.get('eff_sl_override') or (pos['sl_pct'] * r)
                    pos['eff_sl_override'] = cur * keep

        positions = [p for p in positions if p['slot'] not in closed_slots]
        equity += bar_pnl
        if equity > peak_equity:
            peak_equity = equity

        ml, mt, mc = DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD, DEFAULT_MOMENTUM_COOLDOWN
        if ml > 0 and mt > 0 and bar >= ml:
            pct = (closes[bar] / closes[bar - ml] - 1) * 100
            if pct > mt:
                momentum_pause['SHORT'] = bar + mc
            elif pct < -mt:
                momentum_pause['LONG'] = bar + mc

        while si < len(sigs) and sigs[si][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = sigs[si]
            si += 1
            if len(positions) >= DEFAULT_N_SLOTS:
                continue
            if sum(1 for p in positions if p['direction'] == direction) >= DEFAULT_DIRECTION_CAP:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            if ml > 0 and bar < momentum_pause.get(direction, -1):
                continue
            sm = 1.0
            if bar < len(ema_slope):
                s = ema_slope[bar]
                if (s > 0 and direction == 'SHORT') or (s <= 0 and direction == 'LONG'):
                    sm = REGIME_MULT
            # Aggregate risk cap
            arc, arw = DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH
            if arc > 0 or arw > 0:
                is_up = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = (is_up and direction == 'SHORT') or (not is_up and direction == 'LONG')
                cap = arc if is_counter else arw
                exp = sum(p['sl_pct'] * _atr_r(atr_ratio, p['signal_bar'], cl, ch)
                          * (1.0 / DEFAULT_N_SLOTS) * LEVERAGE * p.get('size_mult', 1.0)
                          for p in positions if p['direction'] == direction)
                nr = _atr_r(atr_ratio, sig_bar, cl, ch)
                if exp + sl_pct * nr * (1.0 / DEFAULT_N_SLOTS) * LEVERAGE * sm > cap:
                    continue
            positions.append({'slot': f"{pat}_{sig_bar}", 'signal_bar': sig_bar,
                              'entry_bar': entry_bar, 'direction': direction,
                              'pattern': pat, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
                              'size_mult': sm})

    for pos in positions:
        eb = pos['entry_bar']
        if eb >= n_bars:
            continue
        entry = opens[eb]
        if entry <= 0:
            continue
        xb = min(end_bar - 1, n_bars - 1)
        xp = opens[xb]
        pnl = ((xp / entry - 1) if pos['direction'] == 'LONG'
               else (1 - xp / entry)) * 100 * LEVERAGE - fee
        sm = pos.get('size_mult', 1.0)
        trades.append({'entry_bar': eb, 'exit_bar': xb, 'pnl_slot': pnl,
                       'reason': 'OOS_END', 'pattern': pos['pattern'],
                       'direction': pos['direction'], 'size_mult': sm,
                       'pnl_portfolio': pnl * (size_pct / 100) * sm,
                       'hold_bars': xb - eb})
    return trades


def phase1_decomposition(trades):
    by_type, by_td = {}, {}
    to_profit, to_loss = [], []
    for t in trades:
        r, d, pnl = t['reason'], t['direction'], t['pnl_slot']
        by_type.setdefault(r, []).append(pnl)
        by_td.setdefault(f"{r}_{d}", []).append(pnl)
        if r == 'TIMEOUT':
            (to_profit if pnl > 0 else to_loss).append(pnl)

    def _summarize(pnls):
        return {'count': len(pnls), 'avg_pnl': round(np.mean(pnls), 3),
                'median_pnl': round(np.median(pnls), 3),
                'total_pnl': round(np.sum(pnls), 2),
                'wr': round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1)}

    type_sum = {r: _summarize(p) for r, p in sorted(by_type.items())}
    td_sum = {k: {'count': len(p), 'avg_pnl': round(np.mean(p), 3),
                   'wr': round(sum(1 for x in p if x > 0) / len(p) * 100, 1)}
              for k, p in sorted(by_td.items())}
    n_to = max(1, len(to_profit) + len(to_loss))
    timeout_dist = {
        'profit_count': len(to_profit), 'loss_count': len(to_loss),
        'avg_profit': round(np.mean(to_profit), 3) if to_profit else 0,
        'avg_loss': round(np.mean(to_loss), 3) if to_loss else 0,
        'median_profit': round(np.median(to_profit), 3) if to_profit else 0,
        'median_loss': round(np.median(to_loss), 3) if to_loss else 0,
        'profit_fraction': round(len(to_profit) / n_to, 3),
    }
    hold_by = {}
    for t in trades:
        hold_by.setdefault(t['reason'], []).append(t.get('hold_bars', 0))
    hold_sum = {r: {'avg_bars': round(np.mean(h), 1), 'median_bars': round(np.median(h), 1)}
                for r, h in hold_by.items()}
    return {'by_exit_type': type_sum, 'by_exit_direction': td_sum,
            'timeout_distribution': timeout_dist, 'hold_time_by_type': hold_sum,
            'total_trades': len(trades)}


def run_variant(signal_tuples, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, start_bar, end_bar,
                breakeven_bar=None, label='baseline'):
    trades = portfolio_npos_tracked(signal_tuples, opens, highs, lows, closes, n_bars,
                                    atr_ratio, ema_slope, start_bar, end_bar,
                                    breakeven_bar=breakeven_bar)
    stats = calc_stats_compound(trades)
    wins = [t['pnl_slot'] for t in trades if t['pnl_slot'] > 0]
    losses = [abs(t['pnl_slot']) for t in trades if t['pnl_slot'] <= 0]
    avg_w = np.mean(wins) if wins else 0
    avg_l = np.mean(losses) if losses else 1
    exit_counts = {}
    for t in trades:
        exit_counts[t['reason']] = exit_counts.get(t['reason'], 0) + 1
    return {'label': label, 'trades': stats['trades'], 'wr': stats['wr'],
            'pnl': stats['pnl'], 'mdd': stats['mdd'], 'pnl_mdd': stats['pnl_mdd'],
            'rr': round(avg_w / avg_l, 3) if avg_l > 0 else 0,
            'avg_win': round(avg_w, 3), 'avg_loss': round(avg_l, 3),
            'exit_counts': exit_counts}, trades


def phase3_wf(signal_tuples, opens, highs, lows, closes, n_bars, types,
              atr_ratio, ema_slope, ns, ne, breakeven_bar, label, n_folds=3):
    total = ne - ns
    fs = total // (n_folds + 1)
    folds = []
    for fold in range(n_folds):
        is_end = ns + fs * (fold + 1)
        oos_s, oos_e = is_end, is_end + fs
        is_si = build_signal_index(types[ns:is_end], is_end - ns)
        is_si_off = {p: [b + ns for b in bars] for p, bars in is_si.items()}
        is_pats = scan_universe_range(is_si_off, opens, highs, lows, n_bars,
                                      ns, is_end, 'per_pattern',
                                      min_trades=DEFAULT_MIN_TRADES,
                                      edge_threshold=DEFAULT_EDGE_THRESHOLD,
                                      mc_threshold=DEFAULT_MC_THRESHOLD,
                                      atr_ratio=atr_ratio,
                                      clamp_lo=DEFAULT_ATR_CLAMP_LO,
                                      clamp_hi=DEFAULT_ATR_CLAMP_HI)
        full_si = build_signal_index(types, n_bars)
        oos_tups = []
        for pi in is_pats:
            pn, d = pi['pattern'], pi['direction']
            pk = f"{pn}_{d}"
            if pn in full_si:
                oos_tups.extend((s, pk, d, pi.get('tp_pct', pi.get('tp')),
                                pi.get('sl_pct', pi.get('sl')))
                                for s in full_si[pn] if oos_s <= s < oos_e)
        if not oos_tups:
            folds.append({'fold': fold + 1, 'is_bars': is_end - ns, 'oos_bars': fs,
                          'oos_trades': 0, 'oos_pnl': 0, 'oos_wr': 0, 'status': 'NO_TRADES'})
            continue
        oos_r, _ = run_variant(oos_tups, opens, highs, lows, closes, n_bars,
                               atr_ratio, ema_slope, oos_s, oos_e,
                               breakeven_bar=breakeven_bar, label=f'{label}_f{fold+1}')
        folds.append({'fold': fold + 1, 'is_bars': is_end - ns, 'is_patterns': len(is_pats),
                      'oos_bars': fs, 'oos_trades': oos_r['trades'], 'oos_wr': oos_r['wr'],
                      'oos_pnl': oos_r['pnl'], 'oos_mdd': oos_r['mdd'],
                      'status': 'PASS' if oos_r['pnl'] > 0 else 'FAIL'})
    pc = sum(1 for f in folds if f['status'] == 'PASS')
    return {'label': label, 'breakeven_bar': breakeven_bar, 'folds': folds,
            'pass_count': pc, 'total_folds': n_folds,
            'verdict': 'PASS' if pc == n_folds else 'FAIL'}


def main():
    t0 = time.time()
    logger.info("=== R2: Timeout Exit PnL Decomposition ===")

    df = load_and_classify(DATA_FILE)
    opens, highs, lows, closes = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    types = df['candle_type'].values
    n_bars = len(df)

    nw = find_neutral_window(closes)
    if nw is None:
        logger.error("No neutral window found"); return
    ns, ne = nw
    logger.info(f"Neutral window: bars {ns}-{ne} ({(ne - ns) / 288:.0f}d)")

    atr_ratio = compute_atr_ratio(highs, lows, closes,
                                  atr_period=DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)
    ema_slope = compute_ema_slope(closes)

    patterns = load_patterns()
    signal_index = build_signal_index(types, n_bars)
    signal_tuples = build_signal_tuples(patterns, signal_index)
    logger.info(f"Loaded {len(patterns)} patterns, {len(signal_tuples)} signals")

    # === PHASE 1: Exit Type PnL Decomposition ===
    logger.info("--- Phase 1: Exit Type PnL Decomposition ---")
    bl_trades = portfolio_npos_tracked(signal_tuples, opens, highs, lows, closes, n_bars,
                                       atr_ratio, ema_slope, ns, ne)
    p1 = phase1_decomposition(bl_trades)
    logger.info(f"Phase 1: {p1['total_trades']} trades")
    for et, info in p1['by_exit_type'].items():
        logger.info(f"  {et}: n={info['count']}, avg={info['avg_pnl']:.3f}%, "
                     f"WR={info['wr']:.1f}%, total={info['total_pnl']:.1f}%")

    # === PHASE 2: Breakeven Variants ===
    logger.info("--- Phase 2: Pre-Timeout Breakeven Variants ---")
    bl_r, _ = run_variant(signal_tuples, opens, highs, lows, closes, n_bars,
                          atr_ratio, ema_slope, ns, ne, label='baseline')
    logger.info(f"Baseline: t={bl_r['trades']}, WR={bl_r['wr']}%, PnL={bl_r['pnl']:.1f}%, "
                f"MDD={bl_r['mdd']:.2f}%, R:R={bl_r['rr']}")

    var_results = {'baseline': bl_r}
    for name, be in BREAKEVEN_VARIANTS.items():
        r, _ = run_variant(signal_tuples, opens, highs, lows, closes, n_bars,
                           atr_ratio, ema_slope, ns, ne, breakeven_bar=be, label=name)
        var_results[name] = r
        dp, dr = r['pnl'] - bl_r['pnl'], r['rr'] - bl_r['rr']
        logger.info(f"  {name}: t={r['trades']}, WR={r['wr']}%, PnL={r['pnl']:.1f}%"
                     f"({dp:+.1f}%), MDD={r['mdd']:.2f}%, R:R={r['rr']}({dr:+.3f})")

    # === PHASE 3: WF Validation (top 2 + baseline) ===
    logger.info("--- Phase 3: WF Validation (3-fold expanding) ---")
    ranked = sorted([(n, b, var_results[n]['pnl_mdd']) for n, b in BREAKEVEN_VARIANTS.items()],
                    key=lambda x: x[2], reverse=True)

    wf_results = {}
    wf_bl = phase3_wf(signal_tuples, opens, highs, lows, closes, n_bars, types,
                      atr_ratio, ema_slope, ns, ne, None, 'baseline_wf')
    wf_results['baseline'] = wf_bl
    logger.info(f"WF baseline: {wf_bl['verdict']} ({wf_bl['pass_count']}/{wf_bl['total_folds']})")

    for name, be, _ in ranked[:2]:
        wf = phase3_wf(signal_tuples, opens, highs, lows, closes, n_bars, types,
                       atr_ratio, ema_slope, ns, ne, be, name)
        wf_results[name] = wf
        logger.info(f"WF {name}: {wf['verdict']} ({wf['pass_count']}/{wf['total_folds']})")

    # === Verdicts ===
    verdicts = {}
    for name in BREAKEVEN_VARIANTS:
        r = var_results[name]
        dp = round(r['pnl'] - bl_r['pnl'], 2)
        dr = round(r['rr'] - bl_r['rr'], 3)
        wfv = wf_results.get(name, {}).get('verdict', 'NOT_TESTED')
        if dp > 0 and dr > 0 and wfv == 'PASS':
            v = 'GO'
        elif dp > 0 and wfv == 'PASS':
            v = 'MARGINAL'
        else:
            v = 'STOP'
        verdicts[name] = {'delta_pnl': dp, 'delta_rr': dr, 'wf': wfv, 'verdict': v}

    output = {
        'metadata': {
            'script': 'r2_timeout_pnl_decomposition.py',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'data_file': DATA_FILE, 'patterns_file': PATTERNS_FILE,
            'n_patterns': len(patterns),
            'neutral_window': [int(ns), int(ne)],
            'neutral_days': round((ne - ns) / 288, 1),
            'timeout_bars': TIMEOUT_BARS, 'leverage': LEVERAGE, 'fee_pct': FEE_PCT,
            'runtime_sec': round(time.time() - t0, 1),
        },
        'phase1_decomposition': p1,
        'phase2_breakeven_variants': var_results,
        'phase3_wf': wf_results,
        'verdicts': verdicts,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved to {OUTPUT_FILE}")
    logger.info(f"Runtime: {time.time() - t0:.1f}s")
    logger.info("=== SUMMARY ===")
    for name, v in verdicts.items():
        logger.info(f"  {name}: {v['verdict']} (dPnL={v['delta_pnl']:+.1f}%, "
                     f"dR:R={v['delta_rr']:+.3f}, WF={v['wf']})")


if __name__ == '__main__':
    main()
