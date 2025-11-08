"""
V2 Bot Monitoring Script

실시간으로 V2 봇의 성능을 모니터링하고 V1과 비교합니다.

Usage:
    python scripts/production/monitor_v2_bot.py
"""

import os
from pathlib import Path
from datetime import datetime
import re

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


def parse_log_file(log_file):
    """로그 파일에서 거래 정보 추출"""
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 시작 시간
    start_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*STARTED', content)
    start_time = start_match.group(1) if start_match else "Unknown"

    # 포지션 진입
    entries = re.findall(r'POSITION ENTERED.*?Entry Price: \$([0-9,]+\.\d{2})', content, re.DOTALL)

    # 포지션 종료
    exits = []
    exit_pattern = r'POSITION EXITED - (.*?)\n.*?Exit Price: \$([0-9,]+\.\d{2}).*?P&L: ([+-]\d+\.\d{2})%'
    for match in re.finditer(exit_pattern, content, re.DOTALL):
        exits.append({
            'reason': match.group(1),
            'price': match.group(2),
            'pnl': float(match.group(3))
        })

    # 현재 포트폴리오
    portfolio_matches = re.findall(r'Portfolio: \$([0-9,]+\.\d{2}) \(([+-]\d+\.\d{2})%\)', content)
    if portfolio_matches:
        latest_portfolio = portfolio_matches[-1]
        portfolio = {
            'value': latest_portfolio[0],
            'return': float(latest_portfolio[1])
        }
    else:
        portfolio = {'value': '$10,000.00', 'return': 0.0}

    # 현재 포지션
    current_position = None
    holding_matches = re.findall(r'Holding (LONG|SHORT): P&L ([+-]\d+\.\d{2})% \| (\d+\.\d+)h', content)
    if holding_matches:
        last_holding = holding_matches[-1]
        current_position = {
            'side': last_holding[0],
            'pnl': float(last_holding[1]),
            'duration': float(last_holding[2])
        }

    return {
        'start_time': start_time,
        'entries': len(entries),
        'exits': exits,
        'portfolio': portfolio,
        'current_position': current_position
    }


def analyze_v2_performance(v2_log):
    """V2 성능 분석"""
    data = parse_log_file(v2_log)
    if not data:
        return None

    exits = data['exits']

    # 종료 사유별 카운트
    tp_exits = len([e for e in exits if e['reason'] == 'Take Profit'])
    sl_exits = len([e for e in exits if e['reason'] == 'Stop Loss'])
    max_hold_exits = len([e for e in exits if e['reason'] == 'Max Holding'])

    # 승률
    wins = len([e for e in exits if e['pnl'] > 0])
    total_trades = len(exits)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    # TP 도달률
    tp_rate = (tp_exits / total_trades * 100) if total_trades > 0 else 0

    return {
        'trades': total_trades,
        'tp_exits': tp_exits,
        'sl_exits': sl_exits,
        'max_hold_exits': max_hold_exits,
        'wins': wins,
        'losses': total_trades - wins,
        'win_rate': win_rate,
        'tp_rate': tp_rate,
        'portfolio_return': data['portfolio']['return'],
        'current_position': data['current_position']
    }


def main():
    print("="*80)
    print("V2 BOT MONITORING - Real-time Performance Tracker")
    print("="*80)
    print()

    # V1 로그 찾기
    v1_logs = sorted(LOGS_DIR.glob("combined_long_short_*.log"), reverse=True)
    v1_log = v1_logs[0] if v1_logs else None

    # V2 로그 찾기
    v2_logs = sorted(LOGS_DIR.glob("combined_v2_realistic_*.log"), reverse=True)
    v2_log = v2_logs[0] if v2_logs else None

    if v1_log:
        print("📊 V1 Bot (Original TP)")
        print("-" * 80)
        v1_data = parse_log_file(v1_log)
        if v1_data:
            print(f"  Log: {v1_log.name}")
            print(f"  Started: {v1_data['start_time']}")
            print(f"  Entries: {v1_data['entries']}")
            print(f"  Completed Trades: {len(v1_data['exits'])}")

            if v1_data['exits']:
                tp_exits = len([e for e in v1_data['exits'] if e['reason'] == 'Take Profit'])
                sl_exits = len([e for e in v1_data['exits'] if e['reason'] == 'Stop Loss'])
                max_hold = len([e for e in v1_data['exits'] if e['reason'] == 'Max Holding'])
                wins = len([e for e in v1_data['exits'] if e['pnl'] > 0])

                print(f"  Exit Reasons:")
                print(f"    - Take Profit: {tp_exits} ({tp_exits/len(v1_data['exits'])*100:.1f}%)")
                print(f"    - Stop Loss: {sl_exits}")
                print(f"    - Max Holding: {max_hold} ({max_hold/len(v1_data['exits'])*100:.1f}%)")
                print(f"  Win Rate: {wins}/{len(v1_data['exits'])} ({wins/len(v1_data['exits'])*100:.1f}%)")

            print(f"  Portfolio: {v1_data['portfolio']['value']} ({v1_data['portfolio']['return']:+.2f}%)")
        print()

    if v2_log:
        print("🚀 V2 Bot (Realistic TP)")
        print("-" * 80)
        v2_stats = analyze_v2_performance(v2_log)

        if v2_stats:
            print(f"  Log: {v2_log.name}")
            print(f"  Completed Trades: {v2_stats['trades']}")

            if v2_stats['trades'] > 0:
                print(f"  Exit Reasons:")
                print(f"    - Take Profit: {v2_stats['tp_exits']} ({v2_stats['tp_rate']:.1f}%) ✅")
                print(f"    - Stop Loss: {v2_stats['sl_exits']}")
                print(f"    - Max Holding: {v2_stats['max_hold_exits']} ({v2_stats['max_hold_exits']/v2_stats['trades']*100:.1f}%)")
                print(f"  Win Rate: {v2_stats['wins']}/{v2_stats['trades']} ({v2_stats['win_rate']:.1f}%)")
            else:
                print(f"  Status: No completed trades yet")

            if v2_stats['current_position']:
                pos = v2_stats['current_position']
                print(f"  Current Position: {pos['side']} P&L {pos['pnl']:+.2f}% ({pos['duration']:.1f}h)")

            print(f"  Portfolio Return: {v2_stats['portfolio_return']:+.2f}%")
        print()

    if v1_log and v2_log:
        print("📈 V1 vs V2 Comparison")
        print("-" * 80)

        v1_data = parse_log_file(v1_log)
        v2_stats = analyze_v2_performance(v2_log)

        if v1_data and v1_data['exits'] and v2_stats and v2_stats['trades'] > 0:
            v1_tp_rate = len([e for e in v1_data['exits'] if e['reason'] == 'Take Profit']) / len(v1_data['exits']) * 100
            v1_win_rate = len([e for e in v1_data['exits'] if e['pnl'] > 0]) / len(v1_data['exits']) * 100

            print(f"  TP Hit Rate:")
            print(f"    V1: {v1_tp_rate:.1f}% → V2: {v2_stats['tp_rate']:.1f}% (Δ {v2_stats['tp_rate']-v1_tp_rate:+.1f}%)")

            print(f"  Win Rate:")
            print(f"    V1: {v1_win_rate:.1f}% → V2: {v2_stats['win_rate']:.1f}% (Δ {v2_stats['win_rate']-v1_win_rate:+.1f}%)")

            print(f"  Portfolio Return:")
            print(f"    V1: {v1_data['portfolio']['return']:+.2f}% → V2: {v2_stats['portfolio_return']:+.2f}% (Δ {v2_stats['portfolio_return']-v1_data['portfolio']['return']:+.2f}%)")

            if v2_stats['tp_rate'] > v1_tp_rate:
                print("\n  ✅ IMPROVEMENT: V2 TP hit rate higher!")
            if v2_stats['win_rate'] > v1_win_rate:
                print("  ✅ IMPROVEMENT: V2 win rate higher!")
            if v2_stats['portfolio_return'] > v1_data['portfolio']['return']:
                print("  ✅ IMPROVEMENT: V2 returns better!")
        else:
            print("  ⏳ Waiting for V2 to complete more trades for comparison...")

    print()
    print("="*80)
    print("Run this script periodically to track V2 performance")
    print("="*80)


if __name__ == "__main__":
    main()
