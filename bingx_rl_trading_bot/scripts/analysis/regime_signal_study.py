"""
Regime Signal Study — Market State as Signal
==============================================
3-candle patterns failed. TP/SL structure failed.
What if the signal is MARKET REGIME, not patterns?

Tests:
1. Trend-following: enter in 24h trend direction only
2. Volatility filter: enter only when vol > threshold
3. Combined: trend + vol
4. WF 5-fold for each
5. vs Random baseline
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.scanner.pattern_scanner import compute_atr_ratio, LEVERAGE, FEE_PCT

df = pd.read_csv('data/btc_5m_270days_reclassified.csv')
n = len(df)
o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
ar = compute_atr_ratio(h, l, c)


def bt(bars, d, tp, sl, o, h, l, nb, ar, mb=576):
    fee = FEE_PCT * LEVERAGE
    pnls = []
    for idx in bars:
        if idx + 1 >= nb:
            continue
        entry = o[idx + 1]
        if entry <= 0:
            continue
        a = ar[idx + 1] if not np.isnan(ar[idx + 1]) else 1.0
        a = max(1.0, min(1.5, a))
        ta, sa = tp * a, sl * a
        if d == 'LONG':
            tpp = entry * (1 + ta / 100)
            slp = entry * (1 - sa / 100)
        else:
            tpp = entry * (1 - ta / 100)
            slp = entry * (1 + sa / 100)
        hit = False
        for j in range(idx + 2, min(idx + 2 + mb, nb)):
            if d == 'LONG':
                ht_ = h[j] >= tpp
                hs_ = l[j] <= slp
            else:
                ht_ = l[j] <= tpp
                hs_ = h[j] >= slp
            if ht_ and hs_:
                bo = o[j]
                if abs(tpp - bo) <= abs(slp - bo):
                    pnls.append(ta * LEVERAGE - fee)
                else:
                    pnls.append(-sa * LEVERAGE - fee)
                hit = True
                break
            elif ht_:
                pnls.append(ta * LEVERAGE - fee)
                hit = True
                break
            elif hs_:
                pnls.append(-sa * LEVERAGE - fee)
                hit = True
                break
        if not hit:
            eb = min(idx + 2 + mb, nb - 1)
            ep = o[eb]
            if d == 'LONG':
                ret = (ep - entry) / entry * 100
            else:
                ret = (entry - ep) / entry * 100
            pnls.append(ret * LEVERAGE - fee)
    return pnls


def compute_signals(o, h, l, c, n):
    """Compute regime signals for each bar."""
    trend_24h = np.full(n, np.nan)  # 24h return
    vol_24h = np.full(n, np.nan)    # 24h range %
    vol_48h = np.full(n, np.nan)    # 48h range %

    for i in range(576, n):
        trend_24h[i] = (c[i] - c[i - 288]) / c[i - 288] * 100
        vol_24h[i] = (max(h[i-288:i+1]) - min(l[i-288:i+1])) / c[i] * 100
        vol_48h[i] = (max(h[i-576:i+1]) - min(l[i-576:i+1])) / c[i] * 100

    return trend_24h, vol_24h, vol_48h


def wf_test(strategy_fn, label, tp, sl, n_folds=5):
    """Walk-forward test for a strategy function."""
    folds = []
    for fi in range(n_folds):
        oos_start = int(n * (fi + 1) / (n_folds + 1))
        oos_end = int(n * (fi + 2) / (n_folds + 1))
        oos_o = o[oos_start:oos_end]
        oos_h = h[oos_start:oos_end]
        oos_l = l[oos_start:oos_end]
        oos_c = c[oos_start:oos_end]
        oos_ar = ar[oos_start:oos_end]
        oos_n = oos_end - oos_start

        # Get signals for OOS period
        trend, vol24, vol48 = compute_signals(o, h, l, c, n)
        oos_trend = trend[oos_start:oos_end]
        oos_vol24 = vol24[oos_start:oos_end]
        oos_vol48 = vol48[oos_start:oos_end]

        # Generate entry bars and directions
        entries = strategy_fn(oos_trend, oos_vol24, oos_vol48, oos_n)

        total_pnl = 0
        total_tr = 0
        for bar, direction in entries:
            pnls = bt([bar], direction, tp, sl, oos_o, oos_h, oos_l, oos_n, oos_ar)
            total_pnl += sum(pnls)
            total_tr += len(pnls)

        ppt = total_pnl / total_tr if total_tr > 0 else 0
        folds.append({
            'fold': fi + 1, 'pnl': round(total_pnl, 1),
            'trades': total_tr, 'ppt': round(ppt, 4),
            'pass': total_pnl > 0,
        })

    passes = sum(1 for f in folds if f['pass'])
    total = sum(f['pnl'] for f in folds)
    return folds, passes, total


def main():
    trend_24h, vol_24h, vol_48h = compute_signals(o, h, l, c, n)

    # ================================================================
    # Strategy 1: Trend-Following (enter in 24h trend direction)
    # ================================================================
    print("=" * 60)
    print("Strategy 1: Trend-Following (24h direction)")
    print("=" * 60)

    def trend_strategy(trend, vol24, vol48, nb, interval=12, min_trend=0.5):
        entries = []
        for i in range(0, nb - 2, interval):
            if np.isnan(trend[i]):
                continue
            if abs(trend[i]) < min_trend:
                continue  # Skip low-trend periods
            d = 'LONG' if trend[i] > 0 else 'SHORT'
            entries.append((i, d))
        return entries

    for tp, sl in [(5.0, 2.0), (6.0, 1.0), (4.0, 1.5)]:
        for min_trend in [0.0, 0.5, 1.0, 2.0]:
            def strat(t, v24, v48, nb, mt=min_trend):
                return trend_strategy(t, v24, v48, nb, 12, mt)

            folds, passes, total = wf_test(strat, f"Trend mt={min_trend}", tp, sl)
            total_tr = sum(f['trades'] for f in folds)
            avg_ppt = total / total_tr if total_tr > 0 else 0
            print(f"  TP={tp}/SL={sl} trend>{min_trend}%: "
                  f"WF={passes}/5 Total={total:+.0f}% "
                  f"({total_tr} tr, {avg_ppt:+.4f}$/tr)")

    # ================================================================
    # Strategy 2: Volatility Filter (enter only when vol > threshold)
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Strategy 2: High-Vol Only (both directions)")
    print("=" * 60)

    def vol_strategy(trend, vol24, vol48, nb, interval=12, min_vol=4.0):
        entries = []
        for i in range(0, nb - 2, interval):
            if np.isnan(vol24[i]):
                continue
            if vol24[i] < min_vol:
                continue
            entries.append((i, 'LONG'))
            entries.append((i, 'SHORT'))
        return entries

    for tp, sl in [(6.0, 1.0), (5.0, 2.0)]:
        for min_vol in [0.0, 3.0, 4.0, 5.0, 6.0]:
            def strat(t, v24, v48, nb, mv=min_vol):
                return vol_strategy(t, v24, v48, nb, 12, mv)

            folds, passes, total = wf_test(strat, f"Vol>{min_vol}", tp, sl)
            total_tr = sum(f['trades'] for f in folds)
            avg_ppt = total / total_tr if total_tr > 0 else 0
            print(f"  TP={tp}/SL={sl} vol>{min_vol}%: "
                  f"WF={passes}/5 Total={total:+.0f}% "
                  f"({total_tr} tr, {avg_ppt:+.4f}$/tr)")

    # ================================================================
    # Strategy 3: Trend + Vol (most selective)
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Strategy 3: Trend + Vol Combined")
    print("=" * 60)

    def combined_strategy(trend, vol24, vol48, nb, interval=12,
                          min_trend=1.0, min_vol=4.0):
        entries = []
        for i in range(0, nb - 2, interval):
            if np.isnan(trend[i]) or np.isnan(vol24[i]):
                continue
            if abs(trend[i]) < min_trend or vol24[i] < min_vol:
                continue
            d = 'LONG' if trend[i] > 0 else 'SHORT'
            entries.append((i, d))
        return entries

    for tp, sl in [(6.0, 1.0), (5.0, 2.0), (4.0, 1.5)]:
        for min_trend, min_vol in [(0.5, 3.0), (1.0, 4.0), (1.5, 4.0),
                                    (2.0, 5.0), (1.0, 3.0)]:
            def strat(t, v24, v48, nb, mt=min_trend, mv=min_vol):
                return combined_strategy(t, v24, v48, nb, 12, mt, mv)

            folds, passes, total = wf_test(strat, f"T>{min_trend}+V>{min_vol}", tp, sl)
            total_tr = sum(f['trades'] for f in folds)
            avg_ppt = total / total_tr if total_tr > 0 else 0
            if total_tr > 50:
                print(f"  TP={tp}/SL={sl} T>{min_trend}+V>{min_vol}: "
                      f"WF={passes}/5 Total={total:+.0f}% "
                      f"({total_tr} tr, {avg_ppt:+.4f}$/tr)")

    # ================================================================
    # Strategy 4: Counter-Trend (mean reversion at high vol)
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Strategy 4: Counter-Trend (reverse 24h direction)")
    print("=" * 60)

    def counter_strategy(trend, vol24, vol48, nb, interval=12,
                         min_trend=1.0, min_vol=4.0):
        entries = []
        for i in range(0, nb - 2, interval):
            if np.isnan(trend[i]) or np.isnan(vol24[i]):
                continue
            if abs(trend[i]) < min_trend or vol24[i] < min_vol:
                continue
            d = 'SHORT' if trend[i] > 0 else 'LONG'  # OPPOSITE of trend
            entries.append((i, d))
        return entries

    for tp, sl in [(6.0, 1.0), (5.0, 2.0), (2.0, 5.0), (1.5, 4.0)]:
        for min_trend, min_vol in [(1.0, 4.0), (2.0, 4.0), (1.5, 3.0)]:
            def strat(t, v24, v48, nb, mt=min_trend, mv=min_vol):
                return counter_strategy(t, v24, v48, nb, 12, mt, mv)

            folds, passes, total = wf_test(strat, "Counter", tp, sl)
            total_tr = sum(f['trades'] for f in folds)
            avg_ppt = total / total_tr if total_tr > 0 else 0
            rr_label = f"R:R={'%.1f' % (tp/sl)}" if tp > sl else f"R:R={'%.2f' % (tp/sl)}"
            if total_tr > 30:
                print(f"  TP={tp}/SL={sl} ({rr_label}) T>{min_trend}+V>{min_vol}: "
                      f"WF={passes}/5 Total={total:+.0f}% "
                      f"({total_tr} tr, {avg_ppt:+.4f}$/tr)")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 60}")
    print("SUMMARY: Any WF >= 4/5?")
    print("=" * 60)
    print("  (Review above results for any strategy achieving 4/5 or 5/5 WF PASS)")


if __name__ == '__main__':
    main()
