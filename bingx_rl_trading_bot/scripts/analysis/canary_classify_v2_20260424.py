"""
Canary Trades Class A/B/C Classification v2 — Time-Based Matching
===================================================================
v1 실패 원인: state.json entry는 fill price, log ENTRY는 signal price.
v2 수정: exit_time UTC 기반으로 역산 매칭.

Advisor 권고: F v2 전제 검증.
- Class A: Activation 미교차 → F v2 도움 없음
- Class B: Activation 교차 + baton looser than SL → F v2 도움 없음
- Class C: Baton tighter than SL but baton fired earlier than BT trail
  (drawdown < trail_dist_pct at baton trigger) → F v2 (cycle check_exit) 도움 가능
"""
import re
import json
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT / 'logs'

# From state.json, exit_time is UTC
CANARY_TRADES = [
    {'i': 34, 'dir': 'LONG',  'entry_fill': 78285.9, 'exit': 78905.0, 'pnl_pct': 2.07,  'reason': 'EXCHANGE_SL', 'exit_utc': '2026-04-22 14:16:56', 'bars': 16},
    {'i': 35, 'dir': 'LONG',  'entry_fill': 79250.0, 'exit': 78727.4, 'pnl_pct': -2.28, 'reason': 'EXCHANGE_SL', 'exit_utc': '2026-04-22 16:43:44', 'bars': 16},
    {'i': 36, 'dir': 'SHORT', 'entry_fill': 78535.7, 'exit': 78443.8, 'pnl_pct': 0.05,  'reason': 'EXCHANGE_SL', 'exit_utc': '2026-04-23 00:29:39', 'bars': 6},
    {'i': 37, 'dir': 'LONG',  'entry_fill': 78260.3, 'exit': 77805.8, 'pnl_pct': -2.04, 'reason': 'EXCHANGE_SL', 'exit_utc': '2026-04-23 08:33:34', 'bars': 12},
    {'i': 38, 'dir': 'SHORT', 'entry_fill': 77540.0, 'exit': 77188.2, 'pnl_pct': 1.06,  'reason': 'TRAIL_TP',    'exit_utc': '2026-04-23 10:15:06', 'bars': 4},
    {'i': 39, 'dir': 'LONG',  'entry_fill': 78140.0, 'exit': 77940.4, 'pnl_pct': -1.07, 'reason': 'EXCHANGE_SL', 'exit_utc': '2026-04-23 17:01:47', 'bars': 8},
    {'i': 40, 'dir': 'SHORT', 'entry_fill': 77059.6, 'exit': 77789.9, 'pnl_pct': -3.14, 'reason': 'EXCHANGE_SL', 'exit_utc': '2026-04-23 18:00:47', 'bars': 1},
    {'i': 41, 'dir': 'LONG',  'entry_fill': 78285.0, 'exit': 77942.0, 'pnl_pct': -1.61, 'reason': 'EXCHANGE_SL', 'exit_utc': '2026-04-24 01:32:08', 'bars': 12},
]

RE_TS = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+')
RE_ENTRY = re.compile(r'ENTRY (LONG|SHORT) @ \$(\d+\.?\d*) \| SL=\$(\d+\.?\d*) \((\d+\.?\d*)%\) \| ATR=\$(\d+\.?\d*)')
RE_BATON = re.compile(r'Trail BATON-TOUCH: STOP_MARKET @\$(\d+\.?\d*) \(best=\$(\d+\.?\d*)')


def log_ts_to_utc(ts_str):
    """Log is KST (UTC+9). Convert to UTC."""
    dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
    return dt - timedelta(hours=9)


def read_all_logs():
    files = sorted(LOG_DIR.glob('c1_breakout.log.2026-04-*'))
    files.append(LOG_DIR / 'c1_breakout.log')
    events = []  # (utc_dt, line)
    for f in files:
        if not f.exists():
            continue
        with open(f, encoding='utf-8', errors='ignore') as fp:
            for line in fp:
                m = RE_TS.match(line)
                if m:
                    events.append((log_ts_to_utc(m.group(1)), line.rstrip('\n')))
    events.sort(key=lambda x: x[0])
    return events


def analyze_trade(trade, events):
    """Find ENTRY log (direction+SL), BATON-TOUCH logs within hold window."""
    exit_utc = datetime.strptime(trade['exit_utc'], '%Y-%m-%d %H:%M:%S')
    # bars_held may be unreliable — use generous window (up to 48h before exit)
    start_utc = exit_utc - timedelta(hours=48)

    # Find ENTRY: most recent entry of same direction before exit_utc
    entry_log = None
    for utc, line in events:
        if utc > exit_utc:
            break
        if utc < start_utc:
            continue
        m = RE_ENTRY.search(line)
        if m and m.group(1) == trade['dir']:
            # Keep the latest match before exit
            entry_log = {
                'utc': utc, 'dir': m.group(1),
                'signal': float(m.group(2)), 'sl': float(m.group(3)),
                'sl_pct': float(m.group(4)), 'atr': float(m.group(5)),
            }

    # Find BATON-TOUCH between entry.utc and exit.utc
    batons = []
    if entry_log:
        for utc, line in events:
            if utc <= entry_log['utc']:
                continue
            if utc > exit_utc:
                break
            m = RE_BATON.search(line)
            if m:
                batons.append({
                    'utc': utc, 'trigger': float(m.group(1)), 'best': float(m.group(2)),
                })

    # Compute best_pnl max
    entry = trade['entry_fill']
    best_pnl_max = None
    last_baton = batons[-1] if batons else None
    if batons:
        if trade['dir'] == 'LONG':
            best_max = max(b['best'] for b in batons)
            best_pnl_max = (best_max / entry - 1) * 100
        else:
            best_min = min(b['best'] for b in batons)
            best_pnl_max = (1 - best_min / entry) * 100

    # Classify
    activation_crossed = bool(batons)
    baton_vs_sl = None
    if last_baton and entry_log:
        sl = entry_log['sl']
        bt = last_baton['trigger']
        if trade['dir'] == 'LONG':
            baton_vs_sl = 'tighter' if bt > sl else 'looser'
        else:
            baton_vs_sl = 'tighter' if bt < sl else 'looser'

    # Class
    if not activation_crossed:
        cls = 'A'
        desc = 'Activation 미교차 → F v2 도움 없음'
    elif baton_vs_sl == 'looser':
        cls = 'B'
        desc = 'Baton looser than SL (SL 먼저 hit 정상) → F v2 도움 없음'
    elif baton_vs_sl == 'tighter':
        # Class C: baton fired but was it at optimal BT point?
        # Simplified check: did baton fire earlier than BT trail_dist would allow?
        # BT trail_dist = K*ATR / exit_price * 100 ≈ 2.5 * ATR / exit * 100
        if entry_log:
            atr = entry_log['atr']
            trail_dist = 2.5 * atr / trade['exit'] * 100
            # Compute drawdown at baton fire
            if best_pnl_max is not None:
                if trade['dir'] == 'LONG':
                    cur_pnl_at_exit = (trade['exit'] / entry - 1) * 100
                else:
                    cur_pnl_at_exit = (1 - trade['exit'] / entry) * 100
                drawdown = best_pnl_max - cur_pnl_at_exit
                if drawdown < trail_dist * 0.95:  # baton fired earlier (BT would hold)
                    cls = 'C'
                    desc = f'Baton tighter, fired at drawdown {drawdown:.2f}% < BT trail {trail_dist:.2f}% → F v2 likely helps'
                else:
                    cls = 'B*'
                    desc = f'Baton tighter, fired at drawdown {drawdown:.2f}% >= BT trail {trail_dist:.2f}% → BT would also exit'
            else:
                cls = '?'
                desc = 'Baton tighter but best_pnl 데이터 없음'
        else:
            cls = '?'
            desc = 'Entry log 없음'
    else:
        cls = '?'
        desc = 'Ambiguous'

    return {
        'trade': trade, 'entry_log': entry_log, 'batons_count': len(batons),
        'last_baton': last_baton, 'best_pnl_max': best_pnl_max,
        'activation_crossed': activation_crossed, 'baton_vs_sl': baton_vs_sl,
        'class': cls, 'desc': desc,
    }


def main():
    events = read_all_logs()
    print(f"Loaded {len(events)} log events\n")

    results = [analyze_trade(t, events) for t in CANARY_TRADES]

    print("=" * 120)
    print("Canary Trades Class A/B/C v2 (Time-matched)")
    print("=" * 120)
    hdr = f"{'#':>3} {'dir':<5} {'entry':>9} {'SL':>9} {'exit':>9} {'pnl':>7} {'best_pnl':>10} {'batons':>7} {'class':>6}  desc"
    print(hdr); print("-" * 120)
    for r in results:
        t = r['trade']
        sl = r['entry_log']['sl'] if r['entry_log'] else '?'
        bp = f"{r['best_pnl_max']:.3f}%" if r['best_pnl_max'] is not None else 'n/a'
        print(f"{t['i']:>3} {t['dir']:<5} {t['entry_fill']:>9.1f} {sl:>9} {t['exit']:>9.1f} "
              f"{t['pnl_pct']:>+6.2f}% {bp:>10} {r['batons_count']:>7} {r['class']:>6}  {r['desc']}")

    # Distribution
    cls_count = {}
    for r in results:
        cls_count[r['class']] = cls_count.get(r['class'], 0) + 1
    print()
    print("Class distribution:")
    for c in sorted(cls_count):
        print(f"  {c}: {cls_count[c]} trades")

    # Losers only
    losers = [r for r in results if r['trade']['pnl_pct'] < 0]
    losers_cls = {}
    for r in losers:
        losers_cls[r['class']] = losers_cls.get(r['class'], 0) + 1
    print(f"\nLosers ({len(losers)}) class distribution:")
    for c in sorted(losers_cls):
        print(f"  {c}: {losers_cls[c]} trades")

    print("\n=== Verdict ===")
    c_count = cls_count.get('C', 0)
    a_count = cls_count.get('A', 0)
    b_count = cls_count.get('B', 0) + cls_count.get('B*', 0)
    total = len(results)
    if c_count >= total * 0.4:
        print(f"✅ Class C dominates ({c_count}/{total}) — F v2 is the right fix")
    elif a_count >= total * 0.5:
        print(f"⚠️ Class A dominates ({a_count}/{total}) — F v2 won't help; need different approach")
    elif b_count >= total * 0.5:
        print(f"⚠️ Class B dominates ({b_count}/{total}) — SL correctly fired; accept or adjust SL")
    else:
        print(f"Mixed: A={a_count} B={b_count} C={c_count} — situational")

    out = {'date': datetime.now().isoformat(), 'results': results, 'class_count': cls_count, 'losers_class': losers_cls}
    path = ROOT / 'results' / f'canary_classify_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
