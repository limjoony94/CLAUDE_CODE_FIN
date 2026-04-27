"""
Slippage by ATR — F v2 cycle MARKET close에서 vol scaling 측정
================================================================
Advisor 관찰: #46 -1.28% vs #43/#45 -0.6% (#46는 ATR 2배). vol scaling 가능성?

방법:
1. 봇 로그에서 모든 "Exit slippage" 라인 추출 (F v2 시기 = 04-24~현재)
2. 각 trade의 entry 시점 ATR (log "ENTRY ... ATR=$X" 라인) 매칭
3. (slippage%, ATR%) 회귀 fit
4. F v3 calibration: timeout/spread_buffer를 ATR-conditional 정해야 하는지

목적: F v3 enable 시 vol-aware calibration 가능 여부 판단.
"""
import re, json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT / 'logs'


def parse_logs():
    """모든 log file에서 ENTRY + Exit slippage + EXIT 매칭."""
    rows = []  # list of dicts (entry, atr, slip, direction, reason, hold)
    log_files = sorted(LOG_DIR.glob('c1_breakout.log*'))

    for lf in log_files:
        try:
            text = lf.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        # iterate by line, build trade contexts
        cur_entry = None  # last ENTRY context
        for line in text.splitlines():
            m_entry = re.search(
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*ENTRY (LONG|SHORT) @ \$([\d.]+) \| SL=\$([\d.]+) \(([+\-\d.]+)%\) \| ATR=\$([\d.]+)',
                line)
            if m_entry:
                cur_entry = {
                    'time': m_entry.group(1),
                    'direction': m_entry.group(2),
                    'entry_price': float(m_entry.group(3)),
                    'sl_price': float(m_entry.group(4)),
                    'sl_pct': float(m_entry.group(5)),
                    'atr': float(m_entry.group(6)),
                }
                continue
            m_slip = re.search(
                r'Exit slippage: ([+\-\d.]+)% \(trigger=\$([\d.]+) fill=\$([\d.]+)\)',
                line)
            if m_slip:
                slip_pct = float(m_slip.group(1))
                trigger = float(m_slip.group(2))
                fill = float(m_slip.group(3))
                if cur_entry:
                    atr_pct = cur_entry['atr'] / cur_entry['entry_price'] * 100
                    rows.append({
                        'entry_time': cur_entry['time'],
                        'direction': cur_entry['direction'],
                        'entry_price': cur_entry['entry_price'],
                        'atr': cur_entry['atr'],
                        'atr_pct': round(atr_pct, 4),
                        'sl_pct': cur_entry['sl_pct'],
                        'trigger': trigger,
                        'fill': fill,
                        'slip_pct': slip_pct,
                    })
                continue
            m_exit = re.search(
                r'EXIT (LONG|SHORT) (TRAIL_TP|SL|EMERGENCY|TIMEOUT) \| PnL=([+\-\d.]+)% \| Hold=(\d+)b',
                line)
            if m_exit and rows and 'reason' not in rows[-1]:
                # attach to latest slip row (assumes EXIT follows Exit slippage)
                rows[-1]['reason'] = m_exit.group(2)
                rows[-1]['pnl_3x'] = float(m_exit.group(3))
                rows[-1]['hold_bars'] = int(m_exit.group(4))
    return rows


def main():
    rows = parse_logs()
    print(f"Found {len(rows)} TRAIL_TP exits with slippage record\n")

    if not rows:
        print("No data — exit"); return

    # Sort by time
    rows = sorted(rows, key=lambda r: r['entry_time'])

    print("=" * 120)
    print(f"{'#':>3} {'entry_time':<19} {'dir':<5} {'entry':>9} {'ATR$':>7} {'ATR%':>7} {'sl%':>6} {'trigger':>9} {'fill':>9} {'slip%':>7} {'hold':>5}")
    print("=" * 120)
    for i, r in enumerate(rows, 1):
        print(f"{i:>3} {r['entry_time']:<19} {r['direction']:<5} {r['entry_price']:>9.1f} "
              f"{r['atr']:>7.2f} {r['atr_pct']:>6.4f}% {r['sl_pct']:>5.2f}% "
              f"{r['trigger']:>9.1f} {r['fill']:>9.1f} {r['slip_pct']:>+6.4f}% {r.get('hold_bars',0):>5}")

    # Statistics + correlation
    print()
    n = len(rows)
    avg_slip = sum(r['slip_pct'] for r in rows) / n
    avg_atr_pct = sum(r['atr_pct'] for r in rows) / n
    print(f"Avg slip%: {avg_slip:+.4f}%   Avg ATR%: {avg_atr_pct:.4f}%")

    # Pearson correlation between ATR% and abs(slip%)
    if n >= 3:
        xs = [r['atr_pct'] for r in rows]
        ys = [abs(r['slip_pct']) for r in rows]
        mean_x = sum(xs)/n; mean_y = sum(ys)/n
        cov = sum((xs[i]-mean_x)*(ys[i]-mean_y) for i in range(n))
        var_x = sum((xs[i]-mean_x)**2 for i in range(n))
        var_y = sum((ys[i]-mean_y)**2 for i in range(n))
        if var_x > 0 and var_y > 0:
            r_corr = cov / ((var_x*var_y)**0.5)
            slope = cov / var_x  # |slip|% per ATR%
            print(f"Pearson corr(ATR%, |slip|%): r={r_corr:.4f}")
            print(f"Linear: |slip|% ≈ {slope:.4f} × ATR% + {mean_y - slope*mean_x:+.4f}")
        else:
            r_corr = 0; slope = 0
            print("Insufficient variance for correlation")

    # ATR% bucket
    print()
    print("--- ATR% buckets (slippage 분포) ---")
    sorted_atr = sorted(r['atr_pct'] for r in rows)
    if n >= 4:
        median = sorted_atr[n//2]
        low = [r for r in rows if r['atr_pct'] <= median]
        high = [r for r in rows if r['atr_pct'] > median]
        if low: print(f"  Low ATR  (≤{median:.4f}%): n={len(low)}, avg slip = {sum(r['slip_pct'] for r in low)/len(low):+.4f}%")
        if high: print(f"  High ATR (>{median:.4f}%): n={len(high)}, avg slip = {sum(r['slip_pct'] for r in high)/len(high):+.4f}%")

    print("\nVerdict for F v3 calibration:")
    if n >= 3 and var_x > 0 and var_y > 0 and abs(r_corr) > 0.5:
        print(f"  ⚠️  Slippage scales with ATR (|r|={abs(r_corr):.2f}). F v3 timeout/spread should be ATR-conditional.")
    elif n >= 3 and var_x > 0 and var_y > 0:
        print(f"  ➡️  Mild correlation (|r|={abs(r_corr):.2f}). F v3 fixed timeout 60s 일단 OK, 데이터 누적 후 재평가.")
    else:
        print(f"  Sample n={n} insufficient for confident calibration. F v3 fixed 60s OK.")

    out = {
        'date': datetime.now().isoformat(),
        'n_samples': n,
        'avg_slip_pct': round(avg_slip, 4),
        'avg_atr_pct': round(avg_atr_pct, 4),
        'pearson_r': round(r_corr, 4) if n >= 3 and var_x > 0 and var_y > 0 else None,
        'linear_slope_per_atr_pct': round(slope, 4) if n >= 3 and var_x > 0 and var_y > 0 else None,
        'samples': rows,
    }
    p = ROOT / 'results' / f'slippage_by_atr_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
