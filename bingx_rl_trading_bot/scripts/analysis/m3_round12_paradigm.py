"""M3-R12 — Paradigm shift: π* pair trade + ω* funding harvest.

Pre-reg: claudedocs/m3_round12_paradigm_shift.md
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_critique_pipeline import prepare_all_data


# ==================== π* PAIR TRADE ====================

def backtest_pair_trade(df, valid_mask, z_entry=2.5, z_exit=0.5, timeout_bars=96, friction_rt=0.16):
    """Market-neutral BTC-ETH pair trade.

    Entry: |z| ≥ z_entry → take fading position (SHORT BTC + LONG ETH if z>0, etc.)
    Exit: |z| ≤ z_exit OR timeout
    Friction: friction_rt is full RT cost (0.04 maker × 2 legs × 2 sides = 0.16%).
    PnL: BTC return + opposite ETH return (1:1 dollar-neutral). Net market-neutral.
    """
    n = len(df)
    btc_close = df['close'].values
    eth_close = df['eth_close'].values
    z = df['ratio_z'].values
    timestamps = df['timestamp'].values

    in_pos = False
    direction = None  # 'BTC_SHORT_ETH_LONG' (z>0) or 'BTC_LONG_ETH_SHORT' (z<0)
    btc_entry = None; eth_entry = None; pstart = None

    trades = []
    for i in range(2, n):
        if not valid_mask[i]: continue
        if pd.isna(z[i]) or pd.isna(btc_close[i]) or pd.isna(eth_close[i]):
            continue

        if in_pos:
            held = i - pstart
            should_exit = abs(z[i]) <= z_exit or held >= timeout_bars
            if should_exit:
                # PnL: BTC delta + ETH delta (opposite sign per direction)
                btc_ret = (btc_close[i] / btc_entry - 1) * 100
                eth_ret = (eth_close[i] / eth_entry - 1) * 100
                if direction == 'BTC_SHORT_ETH_LONG':
                    gross = -btc_ret + eth_ret  # short BTC + long ETH
                else:  # BTC_LONG_ETH_SHORT
                    gross = btc_ret - eth_ret
                net = gross - friction_rt
                trades.append({
                    'entry_ts': str(timestamps[pstart]), 'exit_ts': str(timestamps[i]),
                    'direction': direction, 'gross_pct': round(gross, 4),
                    'net_pct': round(net, 4), 'bars_held': held, 'z_entry': z[pstart], 'z_exit': z[i],
                })
                in_pos = False

        if not in_pos:
            if abs(z[i]) >= z_entry:
                in_pos = True
                if z[i] > 0:
                    direction = 'BTC_SHORT_ETH_LONG'
                else:
                    direction = 'BTC_LONG_ETH_SHORT'
                btc_entry = btc_close[i]
                eth_entry = eth_close[i]
                pstart = i
    return trades


def trade_summary_pair(trades):
    if not trades:
        return None
    nets = [t['net_pct'] for t in trades]
    grosses = [t['gross_pct'] for t in trades]
    days = (pd.to_datetime(trades[-1]['exit_ts']) - pd.to_datetime(trades[0]['entry_ts'])).days
    if days == 0: days = 1
    wins = sum(1 for x in nets if x > 0)
    n = len(nets)
    win_pnls = [x for x in nets if x > 0]
    loss_pnls = [x for x in nets if x <= 0]
    rr = abs((sum(win_pnls)/max(1, len(win_pnls))) / (sum(loss_pnls)/max(1, len(loss_pnls)))) if loss_pnls else float('inf')
    return {
        'n': n, 'days': days, 'per_day': round(n/days, 3),
        'sum_net': round(sum(nets), 2), 'sum_gross': round(sum(grosses), 2),
        'avg_net': round(sum(nets)/n, 4), 'avg_gross': round(sum(grosses)/n, 4),
        'wr_pct': round(100 * wins / n, 2), 'rr': round(rr, 3),
        'daily_net': round(sum(nets)/days, 4),
    }


def test_pair_trade(df, valid_mask):
    print("\n" + "=" * 80); print("π* PAIR TRADE (true market-neutral)"); print("=" * 80)

    # Sweep entry/exit thresholds (small grid to avoid multiple-comparison)
    print(f"\n{'z_entry':>8} {'z_exit':>8} {'n':>5} {'days':>5} {'per_day':>8} {'daily_net':>12} {'WR':>6} {'RR':>6}")
    results = []
    for z_entry in (2.0, 2.5, 3.0):
        for z_exit in (0.0, 0.5, 1.0):
            trades = backtest_pair_trade(df, valid_mask, z_entry=z_entry, z_exit=z_exit, friction_rt=0.16)
            s = trade_summary_pair(trades)
            if s:
                results.append({'z_entry': z_entry, 'z_exit': z_exit, **s})
                print(f"{z_entry:>8.1f} {z_exit:>8.1f} {s['n']:>5} {s['days']:>5} {s['per_day']:>8.3f} "
                      f"{s['daily_net']:>+11.4f}% {s['wr_pct']:>5.1f}% {s['rr']:>5.2f}")
            else:
                print(f"{z_entry:>8.1f} {z_exit:>8.1f}  no trades")

    # Pre-reg condition checks on best config
    best = max(results, key=lambda r: r['daily_net']) if results else None
    if best:
        print(f"\nBest config: z_entry={best['z_entry']}, z_exit={best['z_exit']}")
        print(f"  daily={best['daily_net']:+.4f}%, n={best['n']}, per_day={best['per_day']}, WR={best['wr_pct']}, RR={best['rr']}")
        cond = {
            'c1_daily_positive': best['daily_net'] > 0,
            'c2_per_day_ge_0.5': best['per_day'] >= 0.5,
            'c3_wr_ge_50': best['wr_pct'] >= 50,
        }
        print(f"  pre-reg: c1={cond['c1_daily_positive']}, c2={cond['c2_per_day_ge_0.5']}, c3={cond['c3_wr_ge_50']}")
        return {'best': best, 'all_results': results, 'cond': cond}
    return {'best': None, 'all_results': [], 'cond': None}


# ==================== ω* FUNDING HARVEST ====================

def test_funding_harvest(df, valid_mask):
    print("\n" + "=" * 80); print("ω* FUNDING YIELD HARVEST"); print("=" * 80)

    # Funding rate is 8h. 8h = 32 fifteen-min bars.
    # Strategy: when funding > +threshold (longs pay), SHORT perp 1 cycle (8h).
    # PnL = funding received - price drift - friction
    # Approach: examine each 8h block, decide trade based on funding at start

    # Identify 8h boundaries (00:00, 08:00, 16:00 UTC = funding settlement)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour.values
    df['minute'] = pd.to_datetime(df['timestamp']).dt.minute.values
    settlement_mask = ((df['hour'].isin([0, 8, 16])) & (df['minute'] == 0)).values
    settlement_idx = np.where(settlement_mask & valid_mask)[0]

    print(f"  Settlement events found: {len(settlement_idx)}")

    cl = df['close'].values
    funding = df['funding_pct'].values  # The funding rate for the upcoming/just-settled period

    results = {}
    for thresh in (0.0, 0.005, 0.01, 0.015, 0.02):  # threshold in % (0.01 = 0.01%)
        cycles = []
        for k in range(len(settlement_idx) - 1):
            entry_idx = settlement_idx[k]
            exit_idx = settlement_idx[k + 1]
            if exit_idx - entry_idx != 32:  # not a full 8h cycle (data gap)
                continue
            f = funding[entry_idx]
            if pd.isna(f) or pd.isna(cl[entry_idx]) or pd.isna(cl[exit_idx]):
                continue
            if abs(f) < thresh:
                continue
            entry_price = cl[entry_idx]
            exit_price = cl[exit_idx]
            price_ret = (exit_price / entry_price - 1) * 100
            # If funding > 0, longs pay shorts → SHORT to receive
            # If funding < 0, shorts pay longs → LONG to receive
            if f > 0:
                # SHORT: PnL = -price_ret + funding_received
                pnl_gross = -price_ret + f  # f is in % (already)
            else:
                # LONG: PnL = +price_ret + |f|
                pnl_gross = price_ret + abs(f)
            friction_per_cycle = 0.08  # 0.04 × 2 (open + close)
            pnl_net = pnl_gross - friction_per_cycle
            cycles.append({'entry_idx': int(entry_idx), 'funding': f,
                            'price_ret': price_ret, 'gross': pnl_gross, 'net': pnl_net})
        if not cycles:
            results[thresh] = None; continue
        nets = [c['net'] for c in cycles]
        grosses = [c['gross'] for c in cycles]
        days = (pd.to_datetime(df['timestamp'].iloc[settlement_idx[len(settlement_idx)-1]])
                  - pd.to_datetime(df['timestamp'].iloc[settlement_idx[0]])).days
        if days == 0: days = 1
        wins = sum(1 for x in nets if x > 0)
        results[thresh] = {
            'n_cycles': len(cycles),
            'days': days,
            'per_day': round(len(cycles) / days, 3),
            'sum_net': round(sum(nets), 2),
            'sum_gross': round(sum(grosses), 2),
            'avg_net': round(sum(nets) / len(nets), 4),
            'avg_gross': round(sum(grosses) / len(grosses), 4),
            'wr_pct': round(100 * wins / len(nets), 2),
            'daily_net': round(sum(nets) / days, 4),
        }

    print(f"\n{'thresh':>8} {'n_cycles':>10} {'per_day':>8} {'daily_net':>12} {'avg_gross':>12} {'WR':>6}")
    for thresh, r in results.items():
        if r:
            print(f"{thresh:>8.3f} {r['n_cycles']:>10} {r['per_day']:>8.3f} {r['daily_net']:>+11.4f}% {r['avg_gross']:>+11.4f}% {r['wr_pct']:>5.1f}%")
        else:
            print(f"{thresh:>8.3f} no cycles")

    # Best config
    valid_results = {t: r for t, r in results.items() if r is not None}
    if valid_results:
        best = max(valid_results.items(), key=lambda kv: kv[1]['daily_net'])
        print(f"\nBest: threshold={best[0]}, daily={best[1]['daily_net']:+.4f}%, cycles/day={best[1]['per_day']}")
        cond = {
            'c1_daily_positive': best[1]['daily_net'] > 0,
            'c2_per_day_ge_1': best[1]['per_day'] >= 1,
            'c3_wr_ge_40': best[1]['wr_pct'] >= 40,
        }
        return {'best_threshold': best[0], 'best_metrics': best[1], 'all_results': results, 'cond': cond}
    return {'best_threshold': None, 'all_results': results, 'cond': None}


def main():
    print("Loading data...")
    df, h1, h4, base_valid, eth_valid, funding_valid = prepare_all_data()
    print(f"  bars: {len(df):,} | eth_valid: {int(eth_valid.sum()):,} | funding_valid: {int(funding_valid.sum()):,}\n")

    # π* pair trade — needs eth + ratio_z
    pair_results = test_pair_trade(df, eth_valid)
    # ω* funding harvest — needs funding rates
    funding_results = test_funding_harvest(df, funding_valid)

    # Summary
    print("\n" + "=" * 100)
    print("M3-R12 — PARADIGM SHIFT VERDICT")
    print("=" * 100)
    pair_pass = pair_results['cond'] and all(pair_results['cond'].values()) if pair_results['cond'] else False
    funding_pass = funding_results['cond'] and all(funding_results['cond'].values()) if funding_results['cond'] else False
    print(f"  π* pair trade   : {'PASS surface' if pair_pass else 'FAIL'} | best daily: {pair_results['best']['daily_net'] if pair_results['best'] else 'N/A'}")
    print(f"  ω* funding harvest: {'PASS surface' if funding_pass else 'FAIL'} | best daily: {funding_results['best_metrics']['daily_net'] if funding_results.get('best_metrics') else 'N/A'}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'pre_reg': 'claudedocs/m3_round12_paradigm_shift.md',
           'pair_trade': pair_results,
           'funding_harvest': funding_results,
           'pair_surface_pass': pair_pass,
           'funding_surface_pass': funding_pass}
    p = ROOT / 'results' / f'm3_r12_paradigm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
