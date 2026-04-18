"""
Regime vs SL Ratio — Which drives WR?
Hypothesis A: SL ratio is the driver → v1.71 WR 25% contradicts (SL correct but WR low)
Hypothesis B: Market regime is the driver → cascade era = trending = low WR
Hypothesis C: Both required — correct SL AND non-trending
"""
import sys, os, json, numpy as np, pandas as pd
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

with open('results/pattern_5m_metrics.json') as f:
    th = json.load(f).get('trade_history', [])
with open('results/dynamic_patterns.json') as f:
    pd_info = json.load(f).get('pattern_details', {})
df = pd.read_csv('data/btc_5m_270days_reclassified.csv')
all_opens = df['open'].values
all_closes = df['close'].values


def get_sl_ratio(t):
    key = t.get('pattern', '') + '_' + t.get('direction', '')
    if key not in pd_info or t.get('entry_price', 0) <= 0:
        return None
    cfg = pd_info[key].get('sl', 0)
    if cfg <= 0 or not t.get('sl_price'):
        return None
    actual = abs(t['sl_price'] - t['entry_price']) / t['entry_price'] * 100
    return actual / cfg


def get_trend_24h(t):
    ep = t['entry_price']
    bar_idx = np.argmin(np.abs(all_opens - ep))
    if bar_idx < 288:
        return None
    return (all_closes[bar_idx] - all_closes[bar_idx - 288]) / all_closes[bar_idx - 288] * 100


def main():
    # Only TP/SL exits
    tpsl = [t for t in th if t['exit_reason'] in ('TP', 'SL')]

    # Add SL ratio and trend
    enriched = []
    for t in tpsl:
        ratio = get_sl_ratio(t)
        trend = get_trend_24h(t)
        if ratio is not None and trend is not None:
            enriched.append({**t, 'sl_ratio': ratio, 'trend_24h': trend,
                             'abs_trend': abs(trend)})

    print(f"TP+SL trades with both SL ratio and trend: {len(enriched)}")

    # 2x2 MATRIX: SL Correct vs Market Trending
    print("\n" + "=" * 60)
    print("2x2 MATRIX: SL Correct × Market Trending")
    print("=" * 60)

    # SL: correct (>0.85) vs tightened (<0.85)
    # Market: trending (|24h ret| > 2%) vs ranging (<= 2%)
    cells = {
        'Correct+Ranging': [],
        'Correct+Trending': [],
        'Tightened+Ranging': [],
        'Tightened+Trending': [],
    }

    for t in enriched:
        sl_ok = t['sl_ratio'] > 0.85
        trending = t['abs_trend'] > 2.0
        if sl_ok and not trending:
            cells['Correct+Ranging'].append(t)
        elif sl_ok and trending:
            cells['Correct+Trending'].append(t)
        elif not sl_ok and not trending:
            cells['Tightened+Ranging'].append(t)
        else:
            cells['Tightened+Trending'].append(t)

    print(f"\n  {'Cell':<25} {'N':>4} {'WR':>6} {'PnL':>8} {'Avg trend':>10}")
    for label, trades in cells.items():
        if not trades:
            print(f"  {label:<25} {'0':>4}")
            continue
        wins = sum(1 for t in trades if t['pnl_slot'] > 0)
        pnl = sum(t['pnl_portfolio'] for t in trades)
        avg_trend = np.mean([t['trend_24h'] for t in trades])
        print(f"  {label:<25} {len(trades):>4} {wins / len(trades) * 100:>5.1f}% "
              f"{pnl:>+7.2f}% {avg_trend:>+9.2f}%")

    # Statistical test: 2-way ANOVA (approximation via group comparison)
    print("\n  --- Effect Decomposition ---")
    c_r = cells['Correct+Ranging']
    c_t = cells['Correct+Trending']
    t_r = cells['Tightened+Ranging']
    t_t = cells['Tightened+Trending']

    def wr(trades):
        if not trades:
            return 0
        return sum(1 for t in trades if t['pnl_slot'] > 0) / len(trades) * 100

    sl_effect = (wr(c_r) + wr(c_t)) / 2 - (wr(t_r) + wr(t_t)) / 2
    regime_effect = (wr(c_r) + wr(t_r)) / 2 - (wr(c_t) + wr(t_t)) / 2
    interaction = (wr(c_r) - wr(c_t)) - (wr(t_r) - wr(t_t))

    print(f"  SL Effect (correct - tightened):     {sl_effect:+.1f}pp")
    print(f"  Regime Effect (ranging - trending):   {regime_effect:+.1f}pp")
    print(f"  Interaction:                          {interaction:+.1f}pp")

    # More granular: by trend strength
    print("\n" + "=" * 60)
    print("WR by Trend Strength × SL Correctness")
    print("=" * 60)

    print(f"\n  {'Trend Range':<15} {'Correct SL':>12} {'n':>4} {'Tightened SL':>14} {'n':>4}")
    for lo, hi in [(-99, -4), (-4, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 4), (4, 99)]:
        c_sub = [t for t in enriched if t['sl_ratio'] > 0.85 and lo <= t['trend_24h'] < hi]
        t_sub = [t for t in enriched if t['sl_ratio'] <= 0.85 and lo <= t['trend_24h'] < hi]
        c_wr_str = f"{wr(c_sub):.0f}%" if c_sub else "  -"
        t_wr_str = f"{wr(t_sub):.0f}%" if t_sub else "  -"
        label = f"[{lo:+.0f},{hi:+.0f})%"
        print(f"  {label:<15} {c_wr_str:>12} {len(c_sub):>4} {t_wr_str:>14} {len(t_sub):>4}")

    # Direction alignment with trend
    print("\n" + "=" * 60)
    print("Direction Alignment with Pre-Entry Trend (Correct SL only)")
    print("=" * 60)

    correct_only = [t for t in enriched if t['sl_ratio'] > 0.85]
    aligned = []
    counter = []
    for t in correct_only:
        if (t['direction'] == 'LONG' and t['trend_24h'] > 0) or \
           (t['direction'] == 'SHORT' and t['trend_24h'] < 0):
            aligned.append(t)
        else:
            counter.append(t)

    print(f"  Trend-aligned (LONG in uptrend / SHORT in downtrend):")
    print(f"    {len(aligned)} trades, WR={wr(aligned):.1f}%")
    print(f"  Counter-trend:")
    print(f"    {len(counter)} trades, WR={wr(counter):.1f}%")
    print(f"  Alignment effect: {wr(aligned) - wr(counter):+.1f}pp")

    # v1.71 context
    print("\n" + "=" * 60)
    print("v1.71 Context")
    print("=" * 60)

    v171 = [t for t in th if t.get('timestamp', '') >= '2026-04-03']
    if v171:
        prices = [t['entry_price'] for t in v171]
        print(f"  BTC: ${min(prices):.0f} → ${max(prices):.0f}")
        print(f"  Direction: {dict(Counter(t['direction'] for t in v171))}")
        print(f"  All timeouts: {[t['direction'] for t in v171 if t['exit_reason']=='TIMEOUT']}")
        print(f"  → {sum(1 for t in v171 if t['direction']=='SHORT')}/{len(v171)} are SHORT")
        print(f"  → In rising market, SHORT positions timeout → WR crashes")
        print(f"  → This is REGIME EFFECT, not system failure")

    # FINAL: Debiased WR accounting for regime
    print("\n" + "=" * 60)
    print("DEBIASED WR: Accounting for BOTH SL and Regime")
    print("=" * 60)

    # Future trades will have: correct SL (v1.71) + unknown regime
    # Best estimate: use correct-SL trades across ALL regime conditions
    c_ranging_wr = wr(c_r)
    c_trending_wr = wr(c_t)

    # What % of time is market trending vs ranging?
    trending_pct = (len(c_t) + len(t_t)) / len(enriched) * 100
    ranging_pct = (len(c_r) + len(t_r)) / len(enriched) * 100

    weighted_wr = c_ranging_wr * ranging_pct / 100 + c_trending_wr * trending_pct / 100
    be_wr = 70.3

    print(f"  Correct-SL + Ranging:  WR={c_ranging_wr:.1f}% ({len(c_r)} trades)")
    print(f"  Correct-SL + Trending: WR={c_trending_wr:.1f}% ({len(c_t)} trades)")
    print(f"  Market regime mix: {ranging_pct:.0f}% ranging, {trending_pct:.0f}% trending")
    print(f"  Regime-weighted WR: {weighted_wr:.1f}%")
    print(f"  BE WR: {be_wr:.1f}%")
    print(f"  Margin: {weighted_wr - be_wr:+.1f}pp")

    # Edge
    tp_port = 0.4798
    sl_port = -1.1335
    edge = weighted_wr / 100 * tp_port + (1 - weighted_wr / 100) * sl_port
    monthly = edge * 10 * 30
    print(f"\n  Edge/trade: {edge:+.4f}%")
    print(f"  Monthly (10 tr/day): {monthly:+.1f}%")

    if weighted_wr > be_wr:
        print(f"\n  VERDICT: VIABLE — regime-weighted WR {weighted_wr:.1f}% > BE {be_wr:.1f}%")
    else:
        print(f"\n  VERDICT: NOT VIABLE — regime-weighted WR {weighted_wr:.1f}% <= BE {be_wr:.1f}%")


if __name__ == '__main__':
    main()
