#!/usr/bin/env python3
"""
Universal TP/SL Fair Comparison Study v2
========================================
TP 1.5/SL 1.0 vs TP 2.1/SL 3.0 — 공정 비교

양쪽 동일 프로토콜:
  1. 3456 패턴 전수 스캔 (12^3 × 2방향)
  2. MC filter (p < 0.01, 10k sims)
  3. 1-pos-at-a-time 포트폴리오
  4. 5-fold Walk-Forward OOS 검증
  5. Compound PnL + MDD

Protocol: production classify_candle, LEVERAGE=3, timeout DROP, 0.10% fee
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(SCRIPT_DIR.parent / 'production'))
from pattern_5m.indicators import classify_candle

# === Constants ===
FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500
MC_SIMS = 10000
MC_THRESHOLD = 0.01
MIN_TRADES = 10

# Two configs to compare
CONFIGS = [
    {'name': 'TP1.5/SL1.0', 'tp': 1.5, 'sl': 1.0},
    {'name': 'TP2.1/SL3.0', 'tp': 2.1, 'sl': 3.0},
]


# ===================================================================
# Core functions
# ===================================================================

def load_and_classify(csv_path):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    o = df['open'].values; h = df['high'].values
    l = df['low'].values; c = df['close'].values
    n = len(df)
    body_abs = np.abs(c - o)
    avg_b20 = pd.Series(body_abs).rolling(20).mean().values
    types = []
    for i in range(n):
        row = pd.Series({'open': o[i], 'high': h[i], 'low': l[i], 'close': c[i]})
        ab = avg_b20[i] if not pd.isna(avg_b20[i]) else 1.0
        types.append(classify_candle(row, ab).value)
    return o, h, l, c, np.array(types), df['timestamp'].values, n


def build_signal_index(types, n):
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def bt_signals(signal_bars, direction, tp_pct, sl_pct, opens, highs, lows, n_bars):
    """Backtest. Timeout trades DROPPED."""
    trades = []
    for sig in signal_bars:
        if sig + 1 >= n_bars:
            continue
        entry = opens[sig + 1]
        if entry <= 0:
            continue
        eb = sig + 1
        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)
        for j in range(sig + 2, min(sig + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                bo = opens[j]
                pnl = (tp_pct if abs(tpp - bo) <= abs(slp - bo) else -sl_pct) * LEVERAGE - FEE_PCT
                trades.append((eb, j, pnl)); break
            elif ht:
                trades.append((eb, j, tp_pct * LEVERAGE - FEE_PCT)); break
            elif hs:
                trades.append((eb, j, -sl_pct * LEVERAGE - FEE_PCT)); break
    return trades


def portfolio_1pos(all_trades):
    if not all_trades:
        return []
    all_trades.sort(key=lambda x: x[0])
    filtered = []; last_exit = -1
    for eb, xb, pnl in all_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl)); last_exit = xb
    return filtered


def calc_stats(trades):
    if not trades:
        return {'pnl': 0, 'compound_pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0,
                'compound_mdd': 0, 'pf': 0, 'pnl_mdd': 0, 'avg_win': 0, 'avg_loss': 0,
                'safety': 0, 'max_consec_loss': 0, 'be_wr': 50}
    pnls = [t[2] for t in trades]
    cum = 0; peak = 0; mdd = 0
    wins = []; losses = []; cl = 0; max_cl = 0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
        if p > 0: wins.append(p); cl = 0
        else: losses.append(p); cl += 1; max_cl = max(max_cl, cl)
    eq = 1.0; cp = 1.0; cmdd = 0
    for p in pnls:
        eq *= (1 + p / 100)
        if eq > cp: cp = eq
        cdd = (cp - eq) / cp * 100
        if cdd > cmdd: cmdd = cdd
    cpnl = (eq - 1) * 100
    aw = float(np.mean(wins)) if wins else 0
    al = float(np.mean(losses)) if losses else 0
    be = abs(al) / (aw + abs(al)) * 100 if (aw + abs(al)) > 0 else 50
    wr = len(wins) / len(pnls) * 100
    ws = sum(wins); ls = sum(abs(x) for x in losses)
    return {
        'pnl': round(cum, 2), 'compound_pnl': round(cpnl, 2),
        'trades': len(pnls), 'wr': round(wr, 1),
        'mdd': round(mdd, 2), 'compound_mdd': round(cmdd, 2),
        'pf': round(ws / ls, 2) if ls > 0 else 999,
        'pnl_mdd': round(cpnl / cmdd, 2) if cmdd > 0 else 999,
        'avg_win': round(aw, 3), 'avg_loss': round(al, 3),
        'safety': round(wr - be, 1), 'max_consec_loss': max_cl,
        'be_wr': round(be, 1),
    }


def mc_test(pnls, n_sims=MC_SIMS):
    if len(pnls) < 5:
        return 1.0
    arr = np.array(pnls)
    actual = np.sum(arr)
    rng = np.random.default_rng(42)
    signs = rng.choice([-1, 1], size=(n_sims, len(arr)))
    return float(np.mean((signs @ arr) >= actual))


# ===================================================================
# Full pipeline for one TP/SL config
# ===================================================================

def full_pipeline(cfg, signal_index, opens, highs, lows, n, timestamps):
    """Full scan → MC filter → portfolio → WF for one TP/SL config."""
    tp = cfg['tp']; sl = cfg['sl']; name = cfg['name']
    be_wr = sl / (tp + sl) * 100
    rr = tp / sl

    print(f"\n{'='*70}")
    print(f"  {name} | R:R={rr:.2f} | BE WR={be_wr:.1f}%")
    print(f"  Win={tp*LEVERAGE - FEE_PCT:+.2f}% | Loss={-sl*LEVERAGE - FEE_PCT:-.2f}%")
    print(f"{'='*70}")

    # === Step 1: Full universe scan ===
    all_results = []
    for pat_name, sig_bars in signal_index.items():
        for direction in ['LONG', 'SHORT']:
            trades = bt_signals(sig_bars, direction, tp, sl, opens, highs, lows, n)
            if len(trades) < MIN_TRADES:
                continue
            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            mc_p = mc_test(pnls)
            all_results.append({
                'pattern': pat_name, 'direction': direction,
                'trades': len(trades), 'wr': round(wr, 1),
                'edge': round(wr - be_wr, 1),
                'pnl': round(sum(pnls), 2),
                'mc_p': round(mc_p, 4),
                'mc_pass': mc_p < MC_THRESHOLD,
            })

    mc_pass = [r for r in all_results if r['mc_pass']]
    mc_pass.sort(key=lambda x: x['pnl'], reverse=True)

    total = len(all_results)
    pos = len([r for r in all_results if r['pnl'] > 0])
    n_long = len([r for r in mc_pass if r['direction'] == 'LONG'])
    n_short = len([r for r in mc_pass if r['direction'] == 'SHORT'])

    print(f"\n  [Scan] Total >= {MIN_TRADES} trades: {total}")
    print(f"  [Scan] Positive PnL: {pos} ({pos/total*100:.1f}%)")
    print(f"  [Scan] MC pass (p<{MC_THRESHOLD}): {len(mc_pass)} ({n_long}L + {n_short}S)")

    # Print MC-pass patterns
    if mc_pass:
        print(f"\n  {'Pattern':<15} {'Dir':<6} {'Trades':>6} {'WR':>6} {'Edge':>7} {'PnL':>8} {'MC':>8}")
        print(f"  {'-'*62}")
        for r in mc_pass:
            print(f"  {r['pattern']:<15} {r['direction']:<6} {r['trades']:>6} "
                  f"{r['wr']:>5.1f}% {r['edge']:>5.1f}pp {r['pnl']:>7.1f}% {r['mc_p']:>7.4f}")

    if not mc_pass:
        print("  NO MC-pass patterns found!")
        return {'config': cfg, 'mc_pass': [], 'portfolio': calc_stats([]),
                'portfolio_mc': 1.0, 'wf': [], 'all_results': all_results}

    # === Step 2: Full-period portfolio ===
    all_trades = []
    for r in mc_pass:
        sig_bars = signal_index[r['pattern']]
        trades = bt_signals(sig_bars, r['direction'], tp, sl, opens, highs, lows, n)
        all_trades.extend(trades)

    portfolio = portfolio_1pos(all_trades)
    port_stats = calc_stats(portfolio)
    port_mc = mc_test([t[2] for t in portfolio])

    print(f"\n  [Portfolio] {len(mc_pass)} patterns, {port_stats['trades']} trades")
    print(f"    WR: {port_stats['wr']:.1f}% | Safety: {port_stats['safety']:.1f}pp")
    print(f"    Compound PnL: {port_stats['compound_pnl']:.1f}%")
    print(f"    Compound MDD: {port_stats['compound_mdd']:.1f}%")
    print(f"    PnL/MDD: {port_stats['pnl_mdd']:.1f}x")
    print(f"    PF: {port_stats['pf']:.2f} | MC: {port_mc:.4f}")
    print(f"    Max Consec Loss: {port_stats['max_consec_loss']}")

    # === Step 3: 5-fold Walk-Forward ===
    print(f"\n  [Walk-Forward] 5-fold:")
    fold_size = n // 5
    wf_results = []

    for fold in range(5):
        oos_start = fold * fold_size
        oos_end = (fold + 1) * fold_size if fold < 4 else n
        is_set = set(range(0, oos_start)) | set(range(oos_end, n))
        oos_set = set(range(oos_start, oos_end))

        # Re-scan IS: check each MC-pass pattern survives in IS
        fold_patterns = []
        for r in mc_pass:
            is_sigs = [s for s in signal_index[r['pattern']] if s in is_set]
            is_trades = bt_signals(is_sigs, r['direction'], tp, sl, opens, highs, lows, n)
            if len(is_trades) < 5:
                continue
            is_pnls = [t[2] for t in is_trades]
            if mc_test(is_pnls) < MC_THRESHOLD and sum(is_pnls) > 0:
                fold_patterns.append(r)

        # OOS test
        oos_trades = []
        for r in fold_patterns:
            oos_sigs = [s for s in signal_index[r['pattern']] if s in oos_set]
            oos_trades.extend(bt_signals(oos_sigs, r['direction'], tp, sl, opens, highs, lows, n))

        oos_port = portfolio_1pos(oos_trades)
        oos_stats = calc_stats(oos_port)

        ts_s = pd.Timestamp(timestamps[oos_start]).strftime('%Y-%m-%d')
        ts_e = pd.Timestamp(timestamps[min(oos_end - 1, n - 1)]).strftime('%Y-%m-%d')

        wf_r = {
            'fold': fold + 1, 'period': f"{ts_s} ~ {ts_e}",
            'is_patterns': len(fold_patterns),
            'oos_trades': oos_stats['trades'], 'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['pnl'], 'oos_compound': oos_stats['compound_pnl'],
            'oos_mdd': oos_stats['compound_mdd'], 'oos_positive': oos_stats['pnl'] > 0,
        }
        wf_results.append(wf_r)
        sign = "+" if wf_r['oos_positive'] else ""
        print(f"    Fold {fold+1}: {wf_r['period']} | {wf_r['is_patterns']:>2} pat | "
              f"{wf_r['oos_trades']:>3} tr | WR {wf_r['oos_wr']:>5.1f}% | "
              f"PnL {sign}{wf_r['oos_pnl']:.1f}% | Cpd {sign}{wf_r['oos_compound']:.1f}%")

    pos_folds = sum(1 for r in wf_results if r['oos_positive'])
    # WF aggregate OOS
    wf_total_pnl = sum(r['oos_pnl'] for r in wf_results)
    wf_total_trades = sum(r['oos_trades'] for r in wf_results)
    print(f"\n    WF Summary: {pos_folds}/5 positive | OOS total PnL: {wf_total_pnl:.1f}% | "
          f"OOS trades: {wf_total_trades}")

    return {
        'config': cfg, 'mc_pass': mc_pass,
        'portfolio': port_stats, 'portfolio_mc': port_mc,
        'wf': wf_results, 'all_results': all_results,
        'wf_positive_folds': pos_folds, 'wf_total_pnl': wf_total_pnl,
        'wf_total_trades': wf_total_trades,
    }


# ===================================================================
# Main
# ===================================================================

def main():
    print("="*70)
    print("Universal TP/SL Fair Comparison Study v2")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)

    csv_path = DATA_DIR / "btc_5m_270days_reclassified.csv"
    print(f"\nLoading: {csv_path}")
    t0 = time.time()
    opens, highs, lows, closes, types, timestamps, n = load_and_classify(str(csv_path))
    signal_index = build_signal_index(types, n)
    print(f"  {n} bars, {len(signal_index)} unique patterns ({time.time()-t0:.1f}s)")
    print(f"  Period: {pd.Timestamp(timestamps[0]).strftime('%Y-%m-%d')} ~ "
          f"{pd.Timestamp(timestamps[-1]).strftime('%Y-%m-%d')}")

    # Run both configs
    results = {}
    for cfg in CONFIGS:
        results[cfg['name']] = full_pipeline(cfg, signal_index, opens, highs, lows, n, timestamps)

    # === FAIR COMPARISON TABLE ===
    print("\n" + "="*70)
    print("FAIR COMPARISON: Both independently scanned + WF validated")
    print("="*70)

    names = [c['name'] for c in CONFIGS]
    r0 = results[names[0]]; r1 = results[names[1]]
    p0 = r0['portfolio']; p1 = r1['portfolio']

    rows = [
        ("R:R",               f"{CONFIGS[0]['tp']/CONFIGS[0]['sl']:.2f}",   f"{CONFIGS[1]['tp']/CONFIGS[1]['sl']:.2f}"),
        ("BE WR",             f"{CONFIGS[0]['sl']/(CONFIGS[0]['tp']+CONFIGS[0]['sl'])*100:.1f}%",
                              f"{CONFIGS[1]['sl']/(CONFIGS[1]['tp']+CONFIGS[1]['sl'])*100:.1f}%"),
        ("MC-pass patterns",  f"{len(r0['mc_pass'])}",                    f"{len(r1['mc_pass'])}"),
        ("  LONG",            f"{sum(1 for r in r0['mc_pass'] if r['direction']=='LONG')}",
                              f"{sum(1 for r in r1['mc_pass'] if r['direction']=='LONG')}"),
        ("  SHORT",           f"{sum(1 for r in r0['mc_pass'] if r['direction']=='SHORT')}",
                              f"{sum(1 for r in r1['mc_pass'] if r['direction']=='SHORT')}"),
        ("Portfolio trades",  f"{p0['trades']}",                          f"{p1['trades']}"),
        ("Win Rate",          f"{p0['wr']:.1f}%",                         f"{p1['wr']:.1f}%"),
        ("Safety (WR-BE)",    f"{p0['safety']:.1f}pp",                    f"{p1['safety']:.1f}pp"),
        ("Compound PnL",      f"{p0['compound_pnl']:.1f}%",              f"{p1['compound_pnl']:.1f}%"),
        ("Compound MDD",      f"{p0['compound_mdd']:.1f}%",              f"{p1['compound_mdd']:.1f}%"),
        ("PnL/MDD",           f"{p0['pnl_mdd']:.1f}x",                   f"{p1['pnl_mdd']:.1f}x"),
        ("Profit Factor",     f"{p0['pf']:.2f}",                         f"{p1['pf']:.2f}"),
        ("Avg Win",           f"{p0['avg_win']:.3f}%",                    f"{p1['avg_win']:.3f}%"),
        ("Avg Loss",          f"{p0['avg_loss']:.3f}%",                   f"{p1['avg_loss']:.3f}%"),
        ("Max Consec Loss",   f"{p0['max_consec_loss']}",                 f"{p1['max_consec_loss']}"),
        ("Portfolio MC",      f"{r0['portfolio_mc']:.4f}",                f"{r1['portfolio_mc']:.4f}"),
        ("WF positive folds", f"{r0.get('wf_positive_folds','N/A')}/5",  f"{r1.get('wf_positive_folds','N/A')}/5"),
        ("WF total OOS PnL",  f"{r0.get('wf_total_pnl',0):.1f}%",       f"{r1.get('wf_total_pnl',0):.1f}%"),
        ("WF total OOS trades", f"{r0.get('wf_total_trades',0)}",        f"{r1.get('wf_total_trades',0)}"),
    ]

    print(f"\n  {'Metric':<22} {names[0]:>16} {names[1]:>16}")
    print(f"  {'-'*56}")
    for label, v0, v1 in rows:
        print(f"  {label:<22} {v0:>16} {v1:>16}")

    # Overlap
    set0 = {(r['pattern'], r['direction']) for r in r0['mc_pass']}
    set1 = {(r['pattern'], r['direction']) for r in r1['mc_pass']}
    overlap = set0 & set1
    only0 = set0 - set1
    only1 = set1 - set0

    print(f"\n  Pattern Overlap:")
    print(f"    Both: {len(overlap)} | {names[0]} only: {len(only0)} | {names[1]} only: {len(only1)}")
    if overlap:
        print(f"    Shared: {', '.join(f'{p}({d[0]})' for p,d in sorted(overlap))}")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'configs': CONFIGS,
        'constants': {'leverage': LEVERAGE, 'fee': FEE_PCT, 'mc_sims': MC_SIMS,
                       'mc_threshold': MC_THRESHOLD, 'min_trades': MIN_TRADES, 'max_bars': MAX_BARS},
    }
    for name in names:
        r = results[name]
        output[name] = {
            'mc_pass': r['mc_pass'],
            'portfolio': r['portfolio'],
            'portfolio_mc': r['portfolio_mc'],
            'wf': r['wf'],
            'wf_positive_folds': r.get('wf_positive_folds'),
            'wf_total_pnl': r.get('wf_total_pnl'),
        }
    output['overlap'] = {
        'both': [list(x) for x in sorted(overlap)],
        'only_' + names[0]: [list(x) for x in sorted(only0)],
        'only_' + names[1]: [list(x) for x in sorted(only1)],
    }

    out_path = RESULTS_DIR / "universal_tp15_sl10_study_v2.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
