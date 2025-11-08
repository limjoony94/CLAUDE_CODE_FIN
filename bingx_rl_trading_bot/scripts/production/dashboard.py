"""
Comprehensive Dashboard - 종합 대시보드

V2 봇의 모든 정보를 한눈에 표시합니다.

Usage:
    python scripts/production/dashboard.py
"""

import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


def get_bot_status():
    """봇 실행 상태 확인 (log timestamp 기반)"""
    # Check V2 bot status from log file
    v2_logs = sorted(LOGS_DIR.glob("combined_v2_realistic_*.log"), reverse=True)
    v2_running = False
    if v2_logs:
        try:
            with open(v2_logs[0], 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    time_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', last_line)
                    if time_match:
                        last_time = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S")
                        time_diff = (datetime.now() - last_time).total_seconds()
                        v2_running = time_diff < 360  # 6 minutes = 5 min check + 1 min buffer
        except:
            pass

    # Check V1 bot status from log file
    v1_logs = sorted(LOGS_DIR.glob("combined_long_short_2*.log"), reverse=True)
    v1_running = False
    if v1_logs:
        try:
            with open(v1_logs[0], 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    time_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', last_line)
                    if time_match:
                        last_time = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S")
                        time_diff = (datetime.now() - last_time).total_seconds()
                        v1_running = time_diff < 360
        except:
            pass

    return {
        'v2_running': v2_running,
        'v1_running': v1_running
    }


def parse_log(log_file):
    """로그 파일 파싱"""
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 시작 시간
    start_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*STARTED', content)
    start_time = start_match.group(1) if start_match else "Unknown"

    # 마지막 업데이트
    if lines:
        last_line = lines[-1]
        time_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', last_line)
        last_update = time_match.group(1) if time_match else "Unknown"
    else:
        last_update = "Unknown"

    # 거래 정보
    entries = len(re.findall(r'POSITION ENTERED', content))
    exits = re.findall(r'POSITION EXITED - (.*?)\n.*?P&L: ([+-]\d+\.\d{2})%', content, re.DOTALL)

    tp_exits = len([e for e in exits if e[0] == 'Take Profit'])
    sl_exits = len([e for e in exits if e[0] == 'Stop Loss'])
    max_hold_exits = len([e for e in exits if e[0] == 'Max Holding'])

    # 승률
    pnls = [float(e[1]) for e in exits]
    wins = len([p for p in pnls if p > 0])
    total_trades = len(pnls)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    # TP 도달률
    tp_rate = (tp_exits / total_trades * 100) if total_trades > 0 else 0

    # 포트폴리오
    portfolio_matches = re.findall(r'Portfolio: \$([0-9,]+\.\d{2}) \(([+-]\d+\.\d{2})%\)', content)
    if portfolio_matches:
        latest_portfolio = portfolio_matches[-1]
        portfolio_value = latest_portfolio[0]
        portfolio_return = float(latest_portfolio[1])
    else:
        portfolio_value = "10,000.00"
        portfolio_return = 0.0

    # 현재 포지션
    holding_matches = re.findall(r'Holding (LONG|SHORT): P&L ([+-]\d+\.\d{2})% \| (\d+\.\d+)h', content)
    if holding_matches:
        last_holding = holding_matches[-1]
        current_position = {
            'side': last_holding[0],
            'pnl': float(last_holding[1]),
            'duration': float(last_holding[2])
        }
    else:
        current_position = None

    # 최근 가격
    price_matches = re.findall(r'Current Price: \$([0-9,]+\.\d{2})', content)
    current_price = price_matches[-1] if price_matches else "Unknown"

    return {
        'start_time': start_time,
        'last_update': last_update,
        'entries': entries,
        'total_trades': total_trades,
        'tp_exits': tp_exits,
        'sl_exits': sl_exits,
        'max_hold_exits': max_hold_exits,
        'wins': wins,
        'losses': total_trades - wins,
        'win_rate': win_rate,
        'tp_rate': tp_rate,
        'portfolio_value': portfolio_value,
        'portfolio_return': portfolio_return,
        'current_position': current_position,
        'current_price': current_price
    }


def print_dashboard():
    """대시보드 출력"""
    print("\n" + "="*80)
    print("🎯 V2 BOT COMPREHENSIVE DASHBOARD")
    print("="*80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 봇 상태
    status = get_bot_status()
    print("🤖 BOT STATUS")
    print("-" * 80)
    v2_status = "🟢 RUNNING" if status['v2_running'] else "🔴 STOPPED"
    v1_status = "🟢 RUNNING" if status['v1_running'] else "🔴 STOPPED"
    print(f"  V2 Bot (Realistic TP): {v2_status}")
    print(f"  V1 Bot (Original TP):  {v1_status}")
    if status['v1_running']:
        print("  ⚠️  WARNING: V1 bot still running (should be stopped)")
    print()

    # V2 성능
    v2_logs = sorted(LOGS_DIR.glob("combined_v2_realistic_*.log"), reverse=True)
    if v2_logs:
        v2_data = parse_log(v2_logs[0])
        if v2_data:
            print("📊 V2 PERFORMANCE")
            print("-" * 80)
            print(f"  Log File: {v2_logs[0].name}")
            print(f"  Started: {v2_data['start_time']}")
            print(f"  Last Update: {v2_data['last_update']}")
            print()

            print(f"  💰 Portfolio: ${v2_data['portfolio_value']} ({v2_data['portfolio_return']:+.2f}%)")
            print(f"  💵 Current BTC Price: ${v2_data['current_price']}")
            print()

            if v2_data['current_position']:
                pos = v2_data['current_position']
                pos_emoji = "🟢" if pos['side'] == 'LONG' else "🔴"
                pnl_emoji = "📈" if pos['pnl'] > 0 else "📉"
                print(f"  {pos_emoji} Current Position: {pos['side']}")
                print(f"  {pnl_emoji} P&L: {pos['pnl']:+.2f}% | Duration: {pos['duration']:.1f}h")
            else:
                print(f"  ⏸️  No Active Position")
            print()

            print(f"  📈 Trading Statistics:")
            print(f"     Completed Trades: {v2_data['total_trades']}")
            print(f"     Win Rate: {v2_data['win_rate']:.1f}% ({v2_data['wins']}W / {v2_data['losses']}L)")
            print()

            if v2_data['total_trades'] > 0:
                print(f"  🎯 Exit Reasons:")
                print(f"     Take Profit:  {v2_data['tp_exits']} ({v2_data['tp_rate']:.1f}%)")
                print(f"     Stop Loss:    {v2_data['sl_exits']}")
                print(f"     Max Holding:  {v2_data['max_hold_exits']}")
                print()

                # 평가
                if v2_data['tp_rate'] >= 40:
                    tp_status = "✅ EXCELLENT"
                elif v2_data['tp_rate'] >= 20:
                    tp_status = "✅ GOOD"
                elif v2_data['tp_rate'] >= 10:
                    tp_status = "⚠️ ACCEPTABLE"
                else:
                    tp_status = "❌ LOW"

                if v2_data['win_rate'] >= 60:
                    wr_status = "✅ EXCELLENT"
                elif v2_data['win_rate'] >= 50:
                    wr_status = "✅ GOOD"
                else:
                    wr_status = "⚠️ NEEDS IMPROVEMENT"

                print(f"  📊 Performance Assessment:")
                print(f"     TP Hit Rate: {tp_status}")
                print(f"     Win Rate: {wr_status}")

    # V1 비교
    v1_logs = sorted(LOGS_DIR.glob("combined_long_short_2*.log"), reverse=True)
    if v1_logs:
        v1_data = parse_log(v1_logs[0])
        if v1_data and v1_data['total_trades'] > 0:
            print()
            print("📉 V1 COMPARISON (Original TP)")
            print("-" * 80)
            print(f"  Completed Trades: {v1_data['total_trades']}")
            print(f"  TP Hit Rate: {v1_data['tp_rate']:.1f}% (vs V2: {v2_data['tp_rate']:.1f}%)")
            print(f"  Win Rate: {v1_data['win_rate']:.1f}% (vs V2: {v2_data['win_rate']:.1f}%)")
            print(f"  Return: {v1_data['portfolio_return']:+.2f}% (vs V2: {v2_data['portfolio_return']:+.2f}%)")

            # 개선 지표
            tp_improvement = v2_data['tp_rate'] - v1_data['tp_rate']
            wr_improvement = v2_data['win_rate'] - v1_data['win_rate']
            return_improvement = v2_data['portfolio_return'] - v1_data['portfolio_return']

            print()
            print(f"  ✨ Improvements:")
            print(f"     TP Hit Rate: {tp_improvement:+.1f}%")
            print(f"     Win Rate: {wr_improvement:+.1f}%")
            print(f"     Return: {return_improvement:+.2f}%")

    print()
    print("="*80)
    print("💡 Quick Commands:")
    print("-" * 80)
    print("  Monitor V1 vs V2:  python scripts/production/monitor_v2_bot.py")
    print("  Check Alerts:      python scripts/production/auto_alert_system.py")
    print("  View V2 Log:       tail -f logs/combined_v2_realistic_*.log")
    print("="*80)
    print()


if __name__ == "__main__":
    print_dashboard()
