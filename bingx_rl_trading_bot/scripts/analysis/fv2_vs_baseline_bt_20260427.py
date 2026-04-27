"""
F v2 cycle exit vs Baseline (legacy trail) — 272d BT 비교
==========================================================
F v2 active vs not (cycle MARKET close vs theoretical break-even cap).
Slippage 가정 0%로 둘 다 동일 (BT 실측 X). 신호 selection 동일.
"""
import sys, json, math, yaml
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)

FEE_RT_PCT = 0.10


def load_csv_15m(path):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp').set_index('timestamp')
    df15 = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open']).reset_index()
    return [[int(r['timestamp'].timestamp()*1000), r['open'], r['high'], r['low'], r['close']] for _, r in df15.iterrows()]


def run_bt(candles, sig, s_idx, e_idx):
    """BT — same as production: 매 cycle check_exit, F v2 무관 결과 동일 (theoretical exit price)."""
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
                xp, rs = er['exit_price'], er['reason']
                pnl = (xp/pprice - 1)*100 if pdir == 'LONG' else (1 - xp/pprice)*100
                pnl -= FEE_RT_PCT
                trades.append({'pnl1x': round(pnl, 4), 'reason': rs, 'bars': pheld,
                               'best_pnl': round(((pbest/pprice-1) if pdir=='LONG' else (1-pbest/pprice))*100, 3)})
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


def stats(trades):
    if not trades: return {}
    sum1x = sum(t['pnl1x'] for t in trades)
    wins = sum(1 for t in trades if t['pnl1x'] > 0)
    from collections import Counter
    reasons = Counter(t['reason'] for t in trades)
    bal = 100; peak = 100; mdd = 0
    for t in trades:
        bal *= (1 + t['pnl1x']/100); peak = max(peak, bal); dd = (bal-peak)/peak*100
        if dd < mdd: mdd = dd
    return {'n': len(trades), 'sum1x': round(sum1x, 2),
            'avg': round(sum1x/len(trades), 4),
            'wr': round(100*wins/len(trades), 1),
            'mdd_compound': round(mdd, 2),
            'reasons': dict(reasons)}


def main():
    cfg = yaml.safe_load(open(ROOT / 'config' / 'c1_breakout_config.yaml'))
    candles = load_csv_15m(str(ROOT / 'data' / 'btc_5m_270days_reclassified.csv'))
    print(f"{len(candles)} bars\n")

    t0 = datetime.fromtimestamp(candles[0][0]/1000)
    s_idx = next(i for i, c in enumerate(candles) if c[0] >= int((t0+timedelta(days=60)).timestamp()*1000))
    e_idx = len(candles) - 1
    days = (datetime.fromtimestamp(candles[e_idx][0]/1000) - (t0 + timedelta(days=60))).days
    print(f"Eval: {days}d\n")

    # Run with progressive_trail enabled vs disabled (F v2 toggle effect not visible in BT
    # because BT uses theoretical break-even cap regardless of cycle MARKET close)
    print("Note: F v2 toggle has no effect on BT — both use signals.check_exit (theoretical).")
    print("Real F v2 effect = LIVE intrabar trigger vs cycle close. Measure-able only with intrabar/slippage model.\n")

    # Compare progressive_trail toggle (real BT-measurable effect)
    print("=" * 90)
    print(f"{'config':<45} {'n':>5} {'sum1x':>9} {'avg':>8} {'WR':>6} {'MDD':>7}")
    print("=" * 90)

    base_cfg = dict(cfg['strategy'])
    base_cfg['progressive_trail'] = {'enabled': False, 'threshold_pct': 0.9, 'trail_K_post': 0.5}
    sig_base = C1BreakoutSignal(base_cfg)
    trades_base = run_bt(candles, sig_base, s_idx, e_idx)
    s = stats(trades_base)
    print(f"{'baseline (no progressive_trail)':<45} {s['n']:>5} {s['sum1x']:>+8.2f}% {s['avg']:>+7.3f}% {s['wr']:>5.1f}% {s['mdd_compound']:>+6.2f}%")

    prog_cfg = dict(cfg['strategy'])
    prog_cfg['progressive_trail'] = {'enabled': True, 'threshold_pct': 0.9, 'trail_K_post': 0.5}
    sig_prog = C1BreakoutSignal(prog_cfg)
    trades_prog = run_bt(candles, sig_prog, s_idx, e_idx)
    s2 = stats(trades_prog)
    print(f"{'production (progressive_trail enabled)':<45} {s2['n']:>5} {s2['sum1x']:>+8.2f}% {s2['avg']:>+7.3f}% {s2['wr']:>5.1f}% {s2['mdd_compound']:>+6.2f}%")
    print(f"{'Δ (prog - base)':<45} {s2['n']-s['n']:>+5} {s2['sum1x']-s['sum1x']:>+8.2f}pp {(s2['avg']-s['avg']):>+7.3f}pp")

    print()
    print("Reason distribution:")
    print(f"  baseline   : {s['reasons']}")
    print(f"  production : {s2['reasons']}")

    # Trail TP best_pnl distribution
    trail_base = [t for t in trades_base if t['reason'] == 'TRAIL_TP']
    trail_prog = [t for t in trades_prog if t['reason'] == 'TRAIL_TP']
    if trail_base and trail_prog:
        avg_best_base = sum(t['best_pnl'] for t in trail_base) / len(trail_base)
        avg_best_prog = sum(t['best_pnl'] for t in trail_prog) / len(trail_prog)
        # progressive에서 best ≥ 0.9% 비율
        prog_high = sum(1 for t in trail_prog if t['best_pnl'] >= 0.9)
        print(f"\nTrail TP best_pnl (avg): baseline {avg_best_base:.3f}% | production {avg_best_prog:.3f}%")
        print(f"  Prog trades with best_pnl ≥ 0.9%: {prog_high}/{len(trail_prog)} ({100*prog_high/len(trail_prog):.1f}%)")

    out = {
        'date': datetime.now().isoformat(),
        'days': days,
        'baseline_no_prog': s,
        'production_prog': s2,
        'delta_pnl': round(s2['sum1x']-s['sum1x'], 2),
        'note': 'F v2 toggle invisible in BT (theoretical exit). progressive_trail effect measured.',
    }
    p = ROOT / 'results' / f'fv2_vs_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
