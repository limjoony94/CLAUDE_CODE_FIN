"""
Canary Trades Class A/B/C 분류 — Advisor 권고 (2026-04-24)
============================================================
F v2 구현 전 전제 검증.

각 Canary trade (#34~#41)에 대해 로그 추출:
1. Entry time/price/SL
2. HOURLY best= snapshots during hold
3. BATON-TOUCH 발생 여부 (activation 교차 증거)
4. Exit detail

Classification:
- Class A: Activation 미교차 (no BATON log) → F v2 영향 없음
- Class B: Activation 교차 + baton_trigger > SL (LONG) or < SL (SHORT) → baton이 looser
           → SL 먼저 hit 정상 → F v2 영향 없음
- Class C: Activation 교차 + baton_trigger 위치가 tighter이지만 SL 먼저 hit
           → F v2의 check_exit + MARKET close만 해결 가능
"""
import re
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT / 'logs'

CANARY_TRADES = [
    # entry/exit times are from state.json (UTC)
    {'i': 34, 'dir': 'LONG',  'entry': 78285.9, 'exit': 78905.0, 'pnl_pct': 2.07,  'reason': 'EXCHANGE_SL', 'exit_time': '2026-04-22T14:16:56'},
    {'i': 35, 'dir': 'LONG',  'entry': 79250.0, 'exit': 78727.4, 'pnl_pct': -2.28, 'reason': 'EXCHANGE_SL', 'exit_time': '2026-04-22T16:43:44'},
    {'i': 36, 'dir': 'SHORT', 'entry': 78535.7, 'exit': 78443.8, 'pnl_pct': 0.05,  'reason': 'EXCHANGE_SL', 'exit_time': '2026-04-23T00:29:39'},
    {'i': 37, 'dir': 'LONG',  'entry': 78260.3, 'exit': 77805.8, 'pnl_pct': -2.04, 'reason': 'EXCHANGE_SL', 'exit_time': '2026-04-23T08:33:34'},
    {'i': 38, 'dir': 'SHORT', 'entry': 77540.0, 'exit': 77188.2, 'pnl_pct': 1.06,  'reason': 'TRAIL_TP',    'exit_time': '2026-04-23T10:15:06'},
    {'i': 39, 'dir': 'LONG',  'entry': 78140.0, 'exit': 77940.4, 'pnl_pct': -1.07, 'reason': 'EXCHANGE_SL', 'exit_time': '2026-04-23T17:01:47'},
    {'i': 40, 'dir': 'SHORT', 'entry': 77059.6, 'exit': 77789.9, 'pnl_pct': -3.14, 'reason': 'EXCHANGE_SL', 'exit_time': '2026-04-23T18:00:47'},
    {'i': 41, 'dir': 'LONG',  'entry': 78285.0, 'exit': 77942.0, 'pnl_pct': -1.61, 'reason': 'EXCHANGE_SL', 'exit_time': '2026-04-24T01:32:08'},
]


def read_all_logs():
    files = sorted(LOG_DIR.glob('c1_breakout.log.2026-04-*'))
    files.append(LOG_DIR / 'c1_breakout.log')
    lines = []
    for f in files:
        if not f.exists():
            continue
        with open(f, encoding='utf-8', errors='ignore') as fp:
            for line in fp:
                lines.append(line.rstrip('\n'))
    return lines


RE_TS = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+')
RE_ENTRY = re.compile(r'ENTRY (LONG|SHORT) @ \$(\d+\.?\d*) \| SL=\$(\d+\.?\d*)')
RE_HOURLY = re.compile(r'HOURLY.*?pos=\[(LONG|SHORT) @(\d+) best=(\d+) trail\??(\d+)? \((\d+)b\)\]')
RE_BATON = re.compile(r'Trail BATON-TOUCH: STOP_MARKET @\$(\d+\.?\d*) \(best=\$(\d+\.?\d*)')
RE_EXIT = re.compile(r'EXIT (LONG|SHORT) (\w+) \| PnL=([-+]?\d+\.?\d*)%')
RE_SKIPPED = re.compile(r'Trail TP: skipped at entry')
RE_GHOST = re.compile(r'GHOST: (LONG|SHORT) @ \$(\d+\.?\d*).*?exit=\$(\d+\.?\d*) PnL=([-+]?\d+\.?\d*)')


def parse_log_ts(line):
    m = RE_TS.match(line)
    if not m:
        return None
    # log is in KST (UTC+9). Convert to UTC.
    dt_kst = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
    # subtract 9h for UTC
    from datetime import timedelta
    return dt_kst - timedelta(hours=9)


def classify_trade(trade, lines):
    """For each trade, find ENTRY, HOURLY logs, BATON-TOUCH, EXIT."""
    exit_dt = datetime.strptime(trade['exit_time'][:19], '%Y-%m-%dT%H:%M:%S')
    # search window: look back up to 48h before exit
    from datetime import timedelta
    start_dt = exit_dt - timedelta(hours=48)
    end_dt = exit_dt + timedelta(minutes=5)

    trade_events = {
        'trade': trade,
        'entry_logs': [],
        'hourly_logs': [],
        'baton_logs': [],
        'exit_logs': [],
        'skipped_logs': [],
        'ghost_logs': [],
    }
    last_entry_idx = -1

    for i, line in enumerate(lines):
        ts = parse_log_ts(line)
        if ts is None:
            continue
        if ts < start_dt or ts > end_dt:
            continue

        m = RE_ENTRY.search(line)
        if m and float(m.group(2)) == trade['entry']:
            trade_events['entry_logs'].append((ts.isoformat(), line.strip()))
            last_entry_idx = i
            continue

        # After entry, track HOURLY and BATON for this trade
        if last_entry_idx >= 0 and i > last_entry_idx:
            m = RE_HOURLY.search(line)
            if m and m.group(1) == trade['dir']:
                dir_match = m.group(1)
                pos_entry = int(m.group(2))
                pos_best = int(m.group(3))
                if abs(pos_entry - int(trade['entry'])) < 10:
                    trade_events['hourly_logs'].append({
                        'ts': ts.isoformat(),
                        'best': pos_best,
                        'trail': m.group(4),
                        'bars': int(m.group(5)),
                    })
                    continue

            m = RE_BATON.search(line)
            if m:
                trade_events['baton_logs'].append({
                    'ts': ts.isoformat(),
                    'trigger': float(m.group(1)),
                    'best': float(m.group(2)),
                })
                continue

            m = RE_SKIPPED.search(line)
            if m:
                trade_events['skipped_logs'].append(ts.isoformat())
                continue

            m = RE_EXIT.search(line)
            if m:
                trade_events['exit_logs'].append({
                    'ts': ts.isoformat(),
                    'dir': m.group(1),
                    'reason': m.group(2),
                    'pnl': float(m.group(3)),
                })
                continue

            m = RE_GHOST.search(line)
            if m:
                trade_events['ghost_logs'].append({
                    'ts': ts.isoformat(),
                    'dir': m.group(1),
                    'entry': float(m.group(2)),
                    'exit': float(m.group(3)),
                    'pnl': float(m.group(4)),
                })

        # stop searching after we cross exit time by >5min
        if ts > exit_dt + timedelta(minutes=1):
            continue

    # Classify
    dir_ = trade['dir']; entry = trade['entry']
    max_best = None
    if trade_events['hourly_logs']:
        if dir_ == 'LONG':
            max_best = max(h['best'] for h in trade_events['hourly_logs'])
            best_pnl_pct = (max_best / entry - 1) * 100
        else:
            max_best = min(h['best'] for h in trade_events['hourly_logs'])
            best_pnl_pct = (1 - max_best / entry) * 100
    else:
        best_pnl_pct = None

    # Activation crossed?
    activation_crossed = bool(trade_events['baton_logs'])
    if not activation_crossed and best_pnl_pct is not None and best_pnl_pct > 0.05:
        # HOURLY best crossed but no baton log recorded — possibly between cycles
        activation_crossed = 'likely'

    # Baton trigger vs SL comparison
    baton_vs_sl = None
    if trade_events['baton_logs']:
        latest_baton = trade_events['baton_logs'][-1]
        baton_trigger = latest_baton['trigger']
        # SL not directly in event; derive from ENTRY log
        sl_price = None
        for ts, l in trade_events['entry_logs']:
            m = RE_ENTRY.search(l)
            if m:
                sl_price = float(m.group(3))
        if sl_price is not None:
            if dir_ == 'LONG':
                # baton > sl means baton is tighter (higher, closer to entry)
                baton_vs_sl = 'baton_tighter' if baton_trigger > sl_price else 'baton_looser'
            else:
                baton_vs_sl = 'baton_tighter' if baton_trigger < sl_price else 'baton_looser'

    # Classify
    if not activation_crossed or activation_crossed == 'likely' and not trade_events['baton_logs']:
        trade_class = 'A'
        desc = 'Activation never crossed → F v2 no help'
    elif baton_vs_sl == 'baton_looser':
        trade_class = 'B'
        desc = 'Baton looser than SL → SL first hit (correct) → F v2 no help'
    elif baton_vs_sl == 'baton_tighter':
        trade_class = 'C'
        desc = 'Baton tighter than SL but SL fired → F v2 might help'
    else:
        trade_class = '?'
        desc = 'Ambiguous (missing data)'

    trade_events['class'] = trade_class
    trade_events['desc'] = desc
    trade_events['best_pnl_pct'] = best_pnl_pct
    trade_events['activation_crossed'] = activation_crossed
    trade_events['baton_vs_sl'] = baton_vs_sl

    return trade_events


def main():
    lines = read_all_logs()
    print(f"Loaded {len(lines)} log lines\n")

    results = []
    for t in CANARY_TRADES:
        r = classify_trade(t, lines)
        results.append(r)

    # Summary
    print("=" * 95)
    print("Canary Trades Class A/B/C Classification (Advisor-driven)")
    print("=" * 95)
    print(f"{'#':>3} {'dir':<5} {'entry':>9} {'SL':>9} {'best':>9} {'best_pnl':>10} "
          f"{'baton?':>7} {'class':>6} {'desc'}")
    print("-" * 95)
    for r in results:
        t = r['trade']
        baton_count = len(r['baton_logs'])
        bp = f"{r['best_pnl_pct']:+.3f}%" if r['best_pnl_pct'] is not None else 'n/a'
        best = r['hourly_logs'][-1]['best'] if r['hourly_logs'] else '?'
        if r['hourly_logs']:
            if t['dir'] == 'LONG':
                best_max = max(h['best'] for h in r['hourly_logs'])
            else:
                best_max = min(h['best'] for h in r['hourly_logs'])
        else:
            best_max = '?'
        sl = '?'
        for ts, l in r['entry_logs']:
            m = RE_ENTRY.search(l)
            if m:
                sl = m.group(3)
                break
        print(f"#{t['i']:>2} {t['dir']:<5} {t['entry']:>9.1f} {sl:>9} {best_max:>9} "
              f"{bp:>10} {baton_count:>7} {r['class']:>6} {r['desc']}")

    classes = {'A': 0, 'B': 0, 'C': 0, '?': 0}
    for r in results:
        classes[r['class']] = classes.get(r['class'], 0) + 1
    print()
    print("Class distribution:")
    for c in ('A', 'B', 'C', '?'):
        print(f"  Class {c}: {classes[c]} trades")
    print()
    print("=== Verdict ===")
    if classes['A'] > classes['C']:
        print(f"⚠️  Class A dominates ({classes['A']}/{sum(classes.values())}) — F v2 will NOT move the needle for these trades.")
        print("   Alternative needed: tighter activation, LIMIT entry, or different mechanism.")
    elif classes['C'] >= classes['A']:
        print(f"✅ Class C significant ({classes['C']}/{sum(classes.values())}) — F v2 is the right fix.")
    else:
        print("Ambiguous — need more trades or deeper analysis.")

    out = {
        'generated_at': datetime.now().isoformat(),
        'results': results,
        'class_distribution': classes,
    }
    path = ROOT / 'results' / f'canary_trades_classify_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
