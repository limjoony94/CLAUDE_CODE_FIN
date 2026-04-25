"""
F v4 Boundary Test — Emergency hit 빈도 추정 (2026-04-25)
=============================================================
F v4 (SL을 cycle check_exit로 이동) 적용 시 catastrophic loss risk 측정.

가설:
- 봇 cycle 15m 동안 가격이 fractal SL 지나 emergency 3%까지 진행하면 → catastrophic
- 이 빈도가 1% 이내면 F v4 안전, 5%+면 risk 큼

방법:
- 333d BT (max_sl 4.5)
- 각 SL-reason trade에서 SL hit bar 다음 2 bars (30분) 동안 가격 진행 측정
- Emergency 3% 도달 빈도 측정
"""
import sys, os, json, math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)
from scripts.production.c1_breakout.signals import C1BreakoutSignal

CONFIG = {
    'channel_period': 15, 'body_min_ratio': 0.4, 'atr_period': 14,
    'trail_K': 2.5, 'max_sl_atr': 4.5,  # production 적용 값
    'emergency_sl_pct': 3.0, 'max_hold_bars': 192,
    'sl_min_pct': 0.15, 'sl_max_pct': 3.0,
    'min_bars_between': 2, 'trail_activation_pct': 0.05,
    'fractal_lookback': 10,
    'progressive_trail': {'enabled': True, 'threshold_pct': 0.9, 'trail_K_post': 0.5},
}
LEVERAGE = 3
EMERGENCY_PCT = 3.0


def load_csv_15m(path):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp').set_index('timestamp')
    df15 = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open']).reset_index()
    return [[int(r['timestamp'].timestamp()*1000), r['open'], r['high'], r['low'], r['close']] for _, r in df15.iterrows()]


def run_bt_with_boundary(candles):
    """BT runs and for each SL-reason trade, captures next 2 bars range."""
    signal = C1BreakoutSignal(CONFIG)
    ts = [datetime.fromtimestamp(c[0]/1000, tz=timezone.utc).replace(tzinfo=None) for c in candles]
    opens, highs = [c[1] for c in candles], [c[2] for c in candles]
    lows, closes = [c[3] for c in candles], [c[4] for c in candles]
    n = len(closes)
    atr = compute_atr(highs, lows, closes, CONFIG['atr_period'])
    ch_h, ch_l = compute_channel(highs, lows, CONFIG['channel_period'])
    sw_l, sw_h = compute_fractal_swings(highs, lows, CONFIG['fractal_lookback'])

    trades = []; in_pos, cd = False, 0
    pdir = pprice = psl = pbest = None; pheld = 0; pentry_bar = -1

    for i in range(n):
        if in_pos:
            pheld += 1
            pbest = max(pbest, highs[i]) if pdir == 'LONG' else min(pbest, lows[i])
            er = signal.check_exit(direction=pdir, entry_price=pprice, best_price=pbest,
                current_high=highs[i], current_low=lows[i], current_close=closes[i],
                sl_price=psl, atr_val=atr[i] if not math.isnan(atr[i]) else atr[i-1],
                bars_held=pheld)
            if er:
                xp, rs = er['exit_price'], er['reason']
                # Capture next 2 bars (15m + 15m = 30min) for boundary analysis
                next_bars = []
                for j in range(i+1, min(i+3, n)):
                    next_bars.append({'high': highs[j], 'low': lows[j], 'close': closes[j]})
                trades.append({
                    'i_exit': i, 'dir': pdir, 'entry': pprice, 'sl_price': psl,
                    'exit': xp, 'reason': rs, 'bars_held': pheld,
                    'pnl_pct_1x': (xp/pprice - 1)*100 if pdir=='LONG' else (1-xp/pprice)*100,
                    'next_bars': next_bars,
                })
                in_pos, cd, pdir = False, i + CONFIG['min_bars_between'], None
        if not in_pos and i >= cd:
            if math.isnan(atr[i]) or math.isnan(ch_h[i]) or i+1 >= n: continue
            es = signal.check_entry(bar_open=opens[i], bar_high=highs[i], bar_low=lows[i],
                bar_close=closes[i], channel_high=ch_h[i], channel_low=ch_l[i],
                atr_val=atr[i], last_swing_low=sw_l[i], last_swing_high=sw_h[i])
            if es:
                pdir = es['direction']; pprice = opens[i+1]; psl = es['sl_price']
                pheld = 0; in_pos = True; pentry_bar = i+1
                pbest = highs[i+1] if pdir == 'LONG' else lows[i+1]
    return trades


def analyze_boundary(trades):
    """For SL-reason trades, check if next 2 bars hit emergency 3%."""
    sl_trades = [t for t in trades if t['reason'] == 'SL']
    print(f"BT total trades: {len(trades)}")
    print(f"SL-reason trades: {len(sl_trades)}\n")

    emg_hits = []
    for t in sl_trades:
        d = t['dir']; entry = t['entry']
        emg_price = entry * (1 - EMERGENCY_PCT/100) if d == 'LONG' else entry * (1 + EMERGENCY_PCT/100)
        worst_extension = 0  # how far past sl_price next 2 bars went
        emg_hit = False
        if not t['next_bars']:
            continue
        for b in t['next_bars']:
            if d == 'LONG':
                if b['low'] <= emg_price:
                    emg_hit = True
                # extension below sl_price
                ext = (t['sl_price'] - b['low']) / t['sl_price'] * 100
                worst_extension = max(worst_extension, ext)
            else:
                if b['high'] >= emg_price:
                    emg_hit = True
                ext = (b['high'] - t['sl_price']) / t['sl_price'] * 100
                worst_extension = max(worst_extension, ext)
        # Loss if F v4 + bot down: assume close at worst extension price
        # Cap at emergency_pct
        cap_loss_pct = min(EMERGENCY_PCT, max(abs(t['pnl_pct_1x']), worst_extension))
        emg_hits.append({
            'entry': entry, 'sl_price': t['sl_price'], 'sl_pnl': t['pnl_pct_1x'],
            'worst_extension_pct': round(worst_extension, 3),
            'emg_hit': emg_hit, 'cap_loss_1x': round(cap_loss_pct, 3),
        })

    n_emg = sum(1 for e in emg_hits if e['emg_hit'])
    n_total = len(emg_hits)
    avg_ext = sum(e['worst_extension_pct'] for e in emg_hits) / max(1, n_total)
    avg_cap = sum(e['cap_loss_1x'] for e in emg_hits) / max(1, n_total)
    severe_ext = sum(1 for e in emg_hits if e['worst_extension_pct'] > 1.5)

    print("=" * 80)
    print("F v4 Boundary Test — SL-reason trade 다음 2 bars 분석")
    print("=" * 80)
    print(f"SL-reason trades analyzed: {n_total}")
    print(f"  Emergency 3% hit (worst case F v4 catastrophic): {n_emg} ({100*n_emg/max(1,n_total):.1f}%)")
    print(f"  Severe extension > 1.5% past SL: {severe_ext} ({100*severe_ext/max(1,n_total):.1f}%)")
    print(f"  Avg extension past SL (next 2 bars): {avg_ext:+.3f}%")
    print(f"  Avg cap loss (1x): {avg_cap:.3f}%")

    # Distribution
    ext_buckets = {
        '0~0.5%': 0, '0.5~1%': 0, '1~1.5%': 0, '1.5~2%': 0, '2~3%': 0, '>3%': 0,
    }
    for e in emg_hits:
        ext = e['worst_extension_pct']
        if ext < 0.5: ext_buckets['0~0.5%'] += 1
        elif ext < 1: ext_buckets['0.5~1%'] += 1
        elif ext < 1.5: ext_buckets['1~1.5%'] += 1
        elif ext < 2: ext_buckets['1.5~2%'] += 1
        elif ext < 3: ext_buckets['2~3%'] += 1
        else: ext_buckets['>3%'] += 1
    print("\nExtension 분포 (SL 지난 후 다음 2 bars 최대 진행):")
    for k, v in ext_buckets.items():
        pct = 100*v/max(1,n_total)
        bar = '█' * int(pct/2)
        print(f"  {k:>10}: {v:>4} ({pct:>5.1f}%) {bar}")

    print(f"\nVerdict for F v4:")
    if 100*n_emg/max(1,n_total) <= 1:
        print(f"  ✅ Emergency 3% hit 빈도 매우 낮음 ({100*n_emg/max(1,n_total):.1f}%) — F v4 catastrophic risk 작음")
    elif 100*n_emg/max(1,n_total) <= 5:
        print(f"  ⚠️  Emergency 3% hit moderate ({100*n_emg/max(1,n_total):.1f}%) — F v4 주의 필요")
    else:
        print(f"  ❌ Emergency 3% hit 잦음 ({100*n_emg/max(1,n_total):.1f}%) — F v4 catastrophic risk 큼")

    return {'total_sl': n_total, 'emg_hit': n_emg, 'emg_hit_pct': 100*n_emg/max(1,n_total),
            'avg_extension_pct': avg_ext, 'avg_cap_loss': avg_cap,
            'severe_extension_pct': 100*severe_ext/max(1,n_total),
            'distribution': ext_buckets, 'samples': emg_hits[:50]}


def main():
    csv_path = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    if not csv_path.exists():
        print(f"ERR: {csv_path} not found")
        return
    print(f"Loading {csv_path.name}...")
    candles = load_csv_15m(str(csv_path))
    print(f"Got {len(candles)} 15m candles\n")

    print("Running BT (max_sl 4.5)...")
    trades = run_bt_with_boundary(candles)

    result = analyze_boundary(trades)

    out = {'date': datetime.now().isoformat(), 'config': CONFIG, 'result': result}
    path = ROOT / 'results' / f'f_v4_boundary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
