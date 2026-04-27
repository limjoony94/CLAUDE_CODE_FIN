"""
Diagnostic 2: LIVE vs BT best_pnl 분포 비교
============================================
질문: LIVE TRAIL_TP가 break-even cap에 더 자주 걸리는가?
방법:
  - BT 943 trades의 best_pnl 분포 (이미 측정)
  - LIVE 11 TRAIL_TP는 best_pnl 직접 기록 안 됨 → log에서 추정
  - 11 TRAIL_TP의 trigger price와 entry로부터 best_pnl 추정 가능
    (trail trigger 시점에서 best_pnl - trail_dist = realized_pnl이고
     break-even cap 시 realized=0 → best_pnl = trail_dist)
"""
import sys, json, math, yaml, re
from datetime import datetime, timedelta
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import compute_atr, compute_channel, compute_fractal_swings


def load_csv_15m(path):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp').set_index('timestamp')
    df15 = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open']).reset_index()
    return [[int(r['timestamp'].timestamp()*1000), r['open'], r['high'], r['low'], r['close']] for _, r in df15.iterrows()]


def run_bt(candles, sig, s_idx, e_idx):
    opens = [c[1] for c in candles]; highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]; closes = [c[4] for c in candles]
    atr = compute_atr(highs, lows, closes, sig.atr_period)
    chh, chl = compute_channel(highs, lows, sig.channel_period)
    swl, swh = compute_fractal_swings(highs, lows, 10)
    in_pos = False; cd = 0
    pdir = pprice = psl = pbest = None; pheld = 0
    trades = []
    for i in range(s_idx, e_idx + 1):
        if in_pos:
            pheld += 1
            pbest = max(pbest, highs[i]) if pdir == 'LONG' else min(pbest, lows[i])
            er = sig.check_exit(direction=pdir, entry_price=pprice, best_price=pbest,
                current_high=highs[i], current_low=lows[i], current_close=closes[i],
                sl_price=psl, atr_val=atr[i] if not math.isnan(atr[i]) else atr[i-1],
                bars_held=pheld)
            if er:
                xp = er['exit_price']
                pnl = (xp/pprice - 1)*100 if pdir == 'LONG' else (1 - xp/pprice)*100
                pnl -= 0.10
                # Best_pnl absolute
                best_pnl = (pbest/pprice - 1)*100 if pdir == 'LONG' else (1 - pbest/pprice)*100
                trades.append({'reason': er['reason'], 'pnl1x': round(pnl, 4), 'bars': pheld,
                               'best_pnl': round(best_pnl, 4)})
                in_pos, cd, pdir = False, i + sig.min_bars_between_trades, None
        if not in_pos and i >= cd and i < e_idx:
            if math.isnan(atr[i]) or math.isnan(chh[i]): continue
            es = sig.check_entry(bar_open=opens[i], bar_high=highs[i], bar_low=lows[i],
                bar_close=closes[i], channel_high=chh[i], channel_low=chl[i],
                atr_val=atr[i], last_swing_low=swl[i], last_swing_high=swh[i])
            if es:
                ni = i + 1
                if ni > e_idx: continue
                pdir = es['direction']; pprice = opens[ni]; psl = es['sl_price']
                pheld = 0; in_pos = True
                pbest = highs[ni] if pdir == 'LONG' else lows[ni]
    return trades


def parse_live_trail_tp():
    """봇 로그에서 TRAIL_TP exits + entry 시점 ATR 매칭."""
    rows = []
    cur_entry = None
    log_files = sorted((ROOT / 'logs').glob('c1_breakout.log*'))
    for lf in log_files:
        try:
            text = lf.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        for line in text.splitlines():
            m_entry = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*ENTRY (LONG|SHORT) @ \$([\d.]+) \| SL=\$([\d.]+) \(([+\-\d.]+)%\) \| ATR=\$([\d.]+)', line)
            if m_entry:
                cur_entry = {'time': m_entry.group(1), 'direction': m_entry.group(2),
                             'entry_price': float(m_entry.group(3)),
                             'sl_pct': float(m_entry.group(5)),
                             'atr': float(m_entry.group(6))}
                continue
            m_slip = re.search(r'Exit slippage: ([+\-\d.]+)% \(trigger=\$([\d.]+) fill=\$([\d.]+)\)', line)
            if m_slip and cur_entry:
                trigger = float(m_slip.group(2))
                # Trail trigger formula: trigger = entry × (1 ± realized_pnl/100)
                # break-even cap: realized = max(0, best_pnl - trail_dist)
                # If trigger ≈ entry, realized=0 → best_pnl = trail_dist (≈ 2.5×ATR×100/entry)
                e = cur_entry['entry_price']
                if cur_entry['direction'] == 'LONG':
                    realized_pnl = (trigger / e - 1) * 100
                else:
                    realized_pnl = (1 - trigger / e) * 100
                # If realized > 0: best_pnl = realized + trail_dist (BT progressive)
                # If realized = 0 (break-even cap): best_pnl < trail_dist
                # We can compute best_pnl from realized + trail_dist using K=2.5
                # but progressive K=0.5 if best_pnl > 0.9%
                # Simpler: just record realized_pnl = inferred 'effective realized'
                trail_dist_estimate = 2.5 * cur_entry['atr'] / e * 100  # K=2.5 approx
                est_best_pnl = realized_pnl + trail_dist_estimate
                rows.append({
                    'entry_time': cur_entry['time'], 'direction': cur_entry['direction'],
                    'entry_price': e, 'atr_pct': cur_entry['atr']/e*100,
                    'trigger': trigger, 'realized_pnl': round(realized_pnl, 4),
                    'est_best_pnl': round(est_best_pnl, 4),
                    'is_break_even_cap': realized_pnl <= 0.05,  # 거의 0%면 cap
                })
                cur_entry = None
    return rows


def main():
    cfg = yaml.safe_load(open(ROOT / 'config' / 'c1_breakout_config.yaml'))
    sig = C1BreakoutSignal(cfg['strategy'])

    # BT
    print("=== BT 272d TRAIL_TP best_pnl 분포 ===")
    candles = load_csv_15m(str(ROOT / 'data' / 'btc_5m_270days_reclassified.csv'))
    t0 = datetime.fromtimestamp(candles[0][0]/1000)
    s_idx = next(i for i, c in enumerate(candles) if c[0] >= int((t0 + timedelta(days=60)).timestamp()*1000))
    e_idx = len(candles) - 1
    bt_trades = run_bt(candles, sig, s_idx, e_idx)
    bt_trail = [t for t in bt_trades if t['reason'] == 'TRAIL_TP']
    print(f"BT TRAIL_TP n={len(bt_trail)}")
    if bt_trail:
        bps = [t['best_pnl'] for t in bt_trail]
        print(f"  Mean : {statistics.mean(bps):.3f}%")
        print(f"  Med  : {statistics.median(bps):.3f}%")
        sorted_bps = sorted(bps)
        print(f"  P5   : {sorted_bps[int(0.05*len(sorted_bps))]:.3f}%")
        print(f"  P25  : {sorted_bps[int(0.25*len(sorted_bps))]:.3f}%")
        print(f"  P75  : {sorted_bps[int(0.75*len(sorted_bps))]:.3f}%")
        print(f"  P95  : {sorted_bps[int(0.95*len(sorted_bps))]:.3f}%")
        # Break-even cap rate (realized=0 → best_pnl < trail_dist)
        # avg trail_dist estimate
        # Better metric: realized = max(0, best - trail_dist) ≤ 0.05% 비율
        # Compute realized for BT
        realized_zero = sum(1 for t in bt_trail if abs((t['pnl1x'] + 0.10)) <= 0.05)  # remove fee
        print(f"  Break-even cap rate (realized ≤ 0.05%): {realized_zero}/{len(bt_trail)} ({100*realized_zero/len(bt_trail):.1f}%)")

    # LIVE
    print("\n=== LIVE TRAIL_TP 분포 (parsed from logs) ===")
    live_rows = parse_live_trail_tp()
    print(f"LIVE TRAIL_TP n={len(live_rows)}")
    if live_rows:
        rps = [r['realized_pnl'] for r in live_rows]
        print(f"  Realized PnL Mean: {statistics.mean(rps):.3f}%")
        print(f"  Realized PnL Med : {statistics.median(rps):.3f}%")
        sorted_rps = sorted(rps)
        for p in (5, 25, 50, 75, 95):
            print(f"  P{p} : {sorted_rps[int(p/100*len(sorted_rps))]:.3f}%")
        cap_rate = sum(1 for r in live_rows if r['is_break_even_cap'])
        print(f"  Break-even cap rate (realized ≤ 0.05%): {cap_rate}/{len(live_rows)} ({100*cap_rate/len(live_rows):.1f}%)")

        print("\n  LIVE TRAIL_TP samples:")
        for r in live_rows:
            print(f"    {r['entry_time']} {r['direction']:<5} ATR%={r['atr_pct']:.3f}% realized={r['realized_pnl']:+.3f}% cap={r['is_break_even_cap']}")

    # Verdict
    print("\n=== Verdict ===")
    if bt_trail and live_rows:
        bt_cap_rate = sum(1 for t in bt_trail if abs(t['pnl1x'] + 0.10) <= 0.05) / len(bt_trail)
        live_cap_rate = sum(1 for r in live_rows if r['is_break_even_cap']) / len(live_rows)
        bt_mean = statistics.mean([t['pnl1x'] for t in bt_trail])
        live_mean = statistics.mean([r['realized_pnl'] - 0.10 for r in live_rows])
        print(f"BT TRAIL_TP avg PnL 1x   : {bt_mean:+.4f}%")
        print(f"LIVE TRAIL_TP avg PnL 1x : {live_mean:+.4f}%")
        print(f"Gap                       : {live_mean-bt_mean:+.4f}pp")
        print(f"BT cap rate              : {100*bt_cap_rate:.1f}%")
        print(f"LIVE cap rate            : {100*live_cap_rate:.1f}%")
        if live_cap_rate > bt_cap_rate * 1.5:
            print("⚠️  LIVE break-even cap rate 50%+ 이상 → trail이 자주 entry로 회귀")
        else:
            print("✅ Cap rate은 BT 비슷 — trail 설계 자체는 OK, slippage가 dominant cause")

    out = {
        'date': datetime.now().isoformat(),
        'bt_trail_n': len(bt_trail) if bt_trail else 0,
        'bt_trail_mean': statistics.mean([t['best_pnl'] for t in bt_trail]) if bt_trail else 0,
        'live_trail_n': len(live_rows),
        'live_realized_mean': statistics.mean([r['realized_pnl'] for r in live_rows]) if live_rows else 0,
        'live_samples': live_rows,
    }
    p = ROOT / 'results' / f'diag2_best_pnl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
