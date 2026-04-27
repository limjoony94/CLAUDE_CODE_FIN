"""
Phase 2 Diagnostic: Entry Signal Isolation Test
================================================
M1-A entry trigger 자체에 edge가 있는지 확인.

방법:
  - 동일 entry: trend (1h+4h) + RSI cross + body + EMA9 + 15m D3 buffered
  - 동일 N=1, min_bars_between=2, friction 0.20%/trade
  - Exit replaced: 고정 N-bar timeout만. SL/trail/emergency 전부 OFF.
  - Test 3개 N: 12 bars (1h), 24 bars (2h), 48 bars (4h)

해석:
  - 어느 N이든 gross PnL > 0 → entry edge 있음, exit mechanics 조정 정당화
  - 모든 N gross PnL < 0 → entry edge 없음, M1-A 폐기 검토 (사용자 보고)

advisor 권고 (2026-04-27): 6 variants 아니고 3개 N values만. 결과 본 후 결정.
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from m1_bt_framework import M1ABot, prepare_data


def run_bt_fixed_exit(df_5m, h1_long, h4_long, d3_long, d3_short, valid_mask,
                      bot, exit_n_bars, friction=0.20):
    """Same entries as M1-A, but exit = fixed N-bar timeout only.
    Returns trades list."""
    n = len(df_5m)
    opens = df_5m['open'].values
    closes = df_5m['close'].values
    timestamps = df_5m['timestamp'].values

    in_pos = False
    pdir = None; pentry = None; pstart_idx = None
    cooldown_until = 0
    trades = []

    i = 0
    while i < n:
        if in_pos:
            held = i - pstart_idx
            if held >= exit_n_bars:
                exit_price = closes[i]
                gross_pct = ((exit_price / pentry - 1) * 100) if pdir == 'LONG' else ((1 - exit_price / pentry) * 100)
                net_pct = gross_pct - friction
                trades.append({
                    'entry_ts': str(timestamps[pstart_idx]),
                    'exit_ts': str(timestamps[i]),
                    'direction': pdir,
                    'entry': float(pentry),
                    'exit': float(exit_price),
                    'gross_pct': round(gross_pct, 4),
                    'net_pct': round(net_pct, 4),
                    'bars_held': held,
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
    return trades


def summarize(trades, label):
    if not trades:
        return {'label': label, 'n': 0}
    nets = [t['net_pct'] for t in trades]
    grosses = [t['gross_pct'] for t in trades]
    wins_gross = sum(1 for x in grosses if x > 0)
    wins_net = sum(1 for x in nets if x > 0)
    days = (pd.to_datetime(trades[-1]['exit_ts']) - pd.to_datetime(trades[0]['entry_ts'])).days
    return {
        'label': label,
        'n': len(trades),
        'days': days,
        'trades_per_day': round(len(trades)/days, 3),
        'gross_sum': round(sum(grosses), 2),
        'gross_avg': round(sum(grosses)/len(grosses), 4),
        'gross_wr_pct': round(100 * wins_gross / len(trades), 2),
        'net_sum': round(sum(nets), 2),
        'net_avg': round(sum(nets)/len(nets), 4),
        'net_wr_pct': round(100 * wins_net / len(trades), 2),
        'daily_gross_pct': round(sum(grosses)/days, 4) if days else 0,
        'daily_net_pct': round(sum(nets)/days, 4) if days else 0,
    }


def main():
    print("Loading + indicators...")
    df_5m, h1_long, h4_long, d3_long, d3_short, valid_mask = prepare_data(
        ROOT / 'data' / 'btc_5m_720days_binance.csv',
        ROOT / 'data' / 'btc_15m_720days.csv',
        ROOT / 'data' / 'btc_1h_720days.csv',
    )
    print(f"  5m: {len(df_5m):,}, valid: {int(valid_mask.sum()):,}\n")

    bot = M1ABot()

    results = []
    for n_bars in (12, 24, 48):
        label = f'fixed_exit_{n_bars}bars_{n_bars*5}min'
        print(f"BT: {label}")
        trades = run_bt_fixed_exit(df_5m, h1_long, h4_long, d3_long, d3_short, valid_mask,
                                    bot, exit_n_bars=n_bars, friction=0.20)
        s = summarize(trades, label)
        results.append(s)
        print(f"  n={s['n']} days={s['days']} per_day={s['trades_per_day']}")
        print(f"  GROSS: sum={s['gross_sum']:+.2f}% avg/trade={s['gross_avg']:+.4f}% WR={s['gross_wr_pct']}%")
        print(f"  NET  : sum={s['net_sum']:+.2f}% avg/trade={s['net_avg']:+.4f}% WR={s['net_wr_pct']}%")
        print(f"  daily: gross={s['daily_gross_pct']:+.4f}% net={s['daily_net_pct']:+.4f}%\n")

    print("=" * 70)
    print("VERDICT — Entry signal edge check (gross sum, friction-free)")
    print("=" * 70)
    any_positive = any(r['gross_sum'] > 0 for r in results)
    if any_positive:
        print("✓ EDGE EXISTS at some horizon — entry signal worth pursuing.")
        print("  Exit mechanics tuning is justified (single hypothesis, not sweep).")
        for r in results:
            mark = 'YES' if r['gross_sum'] > 0 else 'no '
            print(f"    {r['label']:<35} gross_sum={r['gross_sum']:+8.2f}% [{mark}]")
    else:
        print("✗ NO EDGE — entry signal alone has negative gross PnL at all horizons.")
        print("  Same failure mode as C1. Spec-tuning will produce overfit BT pass.")
        print("  Recommended action: report to user, consider M1 paradigm shift or shelve.")
        for r in results:
            print(f"    {r['label']:<35} gross_sum={r['gross_sum']:+8.2f}%")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'spec': 'M1-A entry, fixed-N-bar exit (no SL/trail/emergency)',
        'friction_per_trade_pct': 0.20,
        'horizons_bars': [12, 24, 48],
        'results': results,
        'verdict': 'EDGE_EXISTS' if any_positive else 'NO_EDGE',
    }
    p = ROOT / 'results' / f'm1_entry_isolation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
