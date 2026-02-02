"""
Auto Alert System - 자동 경고 시스템

V2 봇의 비정상 상태를 자동으로 감지하고 경고합니다.

Features:
1. 봇 다운 감지 (5분 이상 업데이트 없음)
2. 큰 손실 감지 (일일 -3% 이상)
3. 연속 손실 감지 (3연속 손실)
4. TP 도달률 이상 감지 (10개 거래에 0%)

Usage:
    python scripts/production/auto_alert_system.py
"""

import os
import re
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


class AlertSystem:
    def __init__(self, log_file):
        self.log_file = log_file
        self.alerts = []

    def check_bot_down(self):
        """봇이 5분 이상 업데이트 없으면 경고"""
        if not os.path.exists(self.log_file):
            self.alerts.append("🚨 CRITICAL: Log file not found")
            return

        with open(self.log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            self.alerts.append("🚨 CRITICAL: Log file is empty")
            return

        # 마지막 로그 시간
        last_line = lines[-1]
        time_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', last_line)

        if time_match:
            last_time_str = time_match.group(1)
            last_time = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
            current_time = datetime.now()
            time_diff = (current_time - last_time).total_seconds()

            if time_diff > 360:  # 6 minutes (5 min check + 1 min buffer)
                self.alerts.append(f"🚨 CRITICAL: Bot down for {time_diff/60:.1f} minutes")
                self.alerts.append(f"   Last update: {last_time_str}")

    def check_large_loss(self):
        """일일 큰 손실 감지 (-3% 이상)"""
        if not os.path.exists(self.log_file):
            return

        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 최신 포트폴리오 상태
        portfolio_matches = re.findall(r'Portfolio: \$([0-9,]+\.\d{2}) \(([+-]\d+\.\d{2})%\)', content)

        if portfolio_matches:
            latest = portfolio_matches[-1]
            return_pct = float(latest[1])

            if return_pct <= -3.0:
                self.alerts.append(f"⚠️ WARNING: Large loss detected: {return_pct:.2f}%")
                self.alerts.append(f"   Portfolio: ${latest[0]}")

    def check_consecutive_losses(self):
        """연속 손실 감지 (3연속)"""
        if not os.path.exists(self.log_file):
            return

        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 종료된 거래의 P&L
        exit_pattern = r'POSITION EXITED.*?P&L: ([+-]\d+\.\d{2})%'
        pnls = [float(match) for match in re.findall(exit_pattern, content, re.DOTALL)]

        if len(pnls) >= 3:
            last_three = pnls[-3:]
            if all(pnl < 0 for pnl in last_three):
                self.alerts.append(f"⚠️ WARNING: 3 consecutive losses")
                self.alerts.append(f"   Last 3: {last_three[0]:.2f}%, {last_three[1]:.2f}%, {last_three[2]:.2f}%")

    def check_tp_hit_rate(self):
        """TP 도달률 이상 감지 (10개 거래에 0개)"""
        if not os.path.exists(self.log_file):
            return

        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 종료 사유
        exit_reasons = re.findall(r'POSITION EXITED - (.*?)\n', content)

        if len(exit_reasons) >= 10:
            tp_exits = len([r for r in exit_reasons if r == 'Take Profit'])
            tp_rate = tp_exits / len(exit_reasons) * 100

            if tp_rate == 0:
                self.alerts.append(f"⚠️ WARNING: 0% TP hit rate after {len(exit_reasons)} trades")
                self.alerts.append(f"   This suggests TPs may still be too high")

    def check_low_win_rate(self):
        """낮은 승률 감지 (20개 거래에 <45%)"""
        if not os.path.exists(self.log_file):
            return

        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # P&L
        exit_pattern = r'POSITION EXITED.*?P&L: ([+-]\d+\.\d{2})%'
        pnls = [float(match) for match in re.findall(exit_pattern, content, re.DOTALL)]

        if len(pnls) >= 20:
            wins = len([p for p in pnls if p > 0])
            win_rate = wins / len(pnls) * 100

            if win_rate < 45:
                self.alerts.append(f"⚠️ WARNING: Low win rate: {win_rate:.1f}% ({wins}/{len(pnls)})")
                self.alerts.append(f"   Expected: ≥50%")

    def check_high_probability_failures(self):
        """높은 확률 신호 실패 감지"""
        if not os.path.exists(self.log_file):
            return

        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 진입 확률과 P&L 매칭
        entries = []
        for match in re.finditer(r'POSITION ENTERED.*?Probability: ([\d.]+)', content, re.DOTALL):
            entries.append(float(match.group(1)))

        exits = []
        for match in re.finditer(r'POSITION EXITED.*?P&L: ([+-][\d.]+)%', content, re.DOTALL):
            exits.append(float(match.group(1)))

        if len(entries) >= 10 and len(entries) == len(exits):
            # prob > 0.7인 거래의 승률
            high_prob_trades = [(prob, pnl) for prob, pnl in zip(entries, exits) if prob >= 0.7]

            if len(high_prob_trades) >= 5:
                wins = len([pnl for prob, pnl in high_prob_trades if pnl > 0])
                wr = wins / len(high_prob_trades) * 100

                if wr < 60:
                    self.alerts.append(f"⚠️ WARNING: High-prob trades (≥0.7) failing")
                    self.alerts.append(f"   Win rate: {wr:.1f}% ({wins}/{len(high_prob_trades)})")
                    self.alerts.append(f"   Expected: ≥65%")

    def run_all_checks(self):
        """모든 체크 실행"""
        self.check_bot_down()
        self.check_large_loss()
        self.check_consecutive_losses()
        self.check_tp_hit_rate()
        self.check_low_win_rate()
        self.check_high_probability_failures()

        return self.alerts

    def print_status(self):
        """상태 출력"""
        print("="*80)
        print("AUTO ALERT SYSTEM - V2 Bot Health Check")
        print("="*80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log: {self.log_file.name if self.log_file else 'Not found'}")
        print()

        if self.alerts:
            print(f"🚨 {len(self.alerts)} ALERTS DETECTED:")
            print()
            for alert in self.alerts:
                print(alert)
        else:
            print("✅ All checks passed - Bot is healthy")

        print()
        print("="*80)


def main():
    # V2 로그 찾기
    v2_logs = sorted(LOGS_DIR.glob("combined_v2_realistic_*.log"), reverse=True)

    if not v2_logs:
        print("❌ ERROR: V2 log file not found")
        return

    v2_log = v2_logs[0]

    # Alert system 실행
    alert_system = AlertSystem(v2_log)
    alert_system.run_all_checks()
    alert_system.print_status()


if __name__ == "__main__":
    main()
