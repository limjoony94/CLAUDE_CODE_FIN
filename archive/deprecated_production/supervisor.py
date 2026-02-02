"""
24/7 Supervisor - 완전 자동화 관리 시스템

V2 봇을 24/7 모니터링하고 자동으로 관리합니다.

Features:
1. 봇 자동 재시작 (크래시 시)
2. 매일 자동 리포트 (아침 9시)
3. 실시간 알림 (경고 발생 시)
4. 성능 추적 및 로그

Usage:
    python scripts/production/supervisor.py

    # 백그라운드 실행 (권장)
    nohup python scripts/production/supervisor.py > logs/supervisor.log 2>&1 &
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import signal

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOGS_DIR = PROJECT_ROOT / "logs"
BOT_SCRIPT = PROJECT_ROOT / "scripts" / "production" / "combined_long_short_v2_realistic_tp.py"
DASHBOARD_SCRIPT = PROJECT_ROOT / "scripts" / "production" / "dashboard.py"
ALERT_SCRIPT = PROJECT_ROOT / "scripts" / "production" / "auto_alert_system.py"
ANALYST_SCRIPT = PROJECT_ROOT / "scripts" / "production" / "autonomous_analyst.py"

# Configuration
CHECK_INTERVAL = 60  # Check every 1 minute
DAILY_REPORT_HOUR = 9  # Send daily report at 9 AM
AUTONOMOUS_ANALYSIS_INTERVAL = 3600  # Run autonomous analysis every 1 hour
RESTART_DELAY = 10  # Wait 10 seconds before restart
MAX_RESTART_ATTEMPTS = 3  # Max restart attempts per hour


class Supervisor:
    def __init__(self):
        self.bot_process = None
        self.last_daily_report = None
        self.last_autonomous_analysis = None
        self.restart_count = 0
        self.restart_reset_time = datetime.now()
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.log("🛑 Shutdown signal received, cleaning up...")
        self.running = False
        if self.bot_process:
            self.bot_process.terminate()
            self.bot_process.wait()
        sys.exit(0)

    def log(self, message):
        """Supervisor log with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[SUPERVISOR {timestamp}] {message}")

    def is_bot_running(self):
        """Check if V2 bot is running (log-based check)"""
        v2_logs = sorted(LOGS_DIR.glob("combined_v2_realistic_*.log"), reverse=True)
        if not v2_logs:
            return False

        try:
            with open(v2_logs[0], 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines:
                    return False

                import re
                last_line = lines[-1]
                time_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', last_line)
                if time_match:
                    from datetime import datetime
                    last_time = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S")
                    time_diff = (datetime.now() - last_time).total_seconds()
                    return time_diff < 360  # 6 minutes
        except:
            pass

        return False

    def start_bot(self):
        """Start V2 bot"""
        try:
            self.log("🚀 Starting V2 bot...")
            self.bot_process = subprocess.Popen(
                [sys.executable, str(BOT_SCRIPT)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(5)  # Wait for startup

            if self.is_bot_running():
                self.log("✅ V2 bot started successfully")
                return True
            else:
                self.log("❌ V2 bot failed to start")
                return False
        except Exception as e:
            self.log(f"❌ Error starting bot: {e}")
            return False

    def restart_bot(self):
        """Restart V2 bot with safety checks"""
        # Reset restart counter if 1 hour has passed
        if (datetime.now() - self.restart_reset_time).total_seconds() > 3600:
            self.restart_count = 0
            self.restart_reset_time = datetime.now()

        # Check restart limit
        if self.restart_count >= MAX_RESTART_ATTEMPTS:
            self.log(f"🚨 CRITICAL: Reached max restart attempts ({MAX_RESTART_ATTEMPTS}/hour)")
            self.log("⏸️  Pausing auto-restart for 1 hour...")
            return False

        self.log(f"🔄 Restarting bot (attempt {self.restart_count + 1}/{MAX_RESTART_ATTEMPTS})...")

        # Stop existing process
        if self.bot_process:
            try:
                self.bot_process.terminate()
                self.bot_process.wait(timeout=10)
            except:
                pass

        time.sleep(RESTART_DELAY)

        # Start new process
        success = self.start_bot()
        if success:
            self.restart_count += 1

        return success

    def check_alerts(self):
        """Run alert system and return number of alerts"""
        try:
            result = subprocess.run(
                [sys.executable, str(ALERT_SCRIPT)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=15
            )

            output = result.stdout

            # Count alerts
            if "All checks passed" in output:
                return 0
            else:
                # Count lines starting with warning/critical symbols
                alerts = output.count("⚠️") + output.count("🚨")
                if alerts > 0:
                    self.log(f"⚠️  {alerts} alerts detected:")
                    for line in output.split('\n'):
                        if '⚠️' in line or '🚨' in line:
                            self.log(f"  {line}")
                return alerts
        except Exception as e:
            self.log(f"❌ Error checking alerts: {e}")
            return 0

    def send_daily_report(self):
        """Generate and log daily report"""
        try:
            self.log("=" * 80)
            self.log("📊 DAILY REPORT")
            self.log("=" * 80)

            # Run dashboard
            result = subprocess.run(
                [sys.executable, str(DASHBOARD_SCRIPT)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=30
            )

            # Log dashboard output
            for line in result.stdout.split('\n'):
                if line.strip():
                    self.log(f"  {line}")

            self.log("=" * 80)
            self.log("✅ Daily report completed")
            self.log("=" * 80)

            self.last_daily_report = datetime.now()
        except Exception as e:
            self.log(f"❌ Error generating daily report: {e}")

    def should_send_daily_report(self):
        """Check if it's time for daily report"""
        now = datetime.now()

        # If never sent or sent yesterday
        if self.last_daily_report is None or self.last_daily_report.date() < now.date():
            # Check if it's after DAILY_REPORT_HOUR
            if now.hour >= DAILY_REPORT_HOUR:
                return True

        return False

    def should_run_autonomous_analysis(self):
        """Check if it's time for autonomous analysis"""
        if self.last_autonomous_analysis is None:
            return True

        time_since_last = (datetime.now() - self.last_autonomous_analysis).total_seconds()
        return time_since_last >= AUTONOMOUS_ANALYSIS_INTERVAL

    def run_autonomous_analysis(self):
        """Run Claude's autonomous analysis"""
        try:
            self.log("=" * 80)
            self.log("🤖 AUTONOMOUS ANALYSIS - Claude's Critical Thinking")
            self.log("=" * 80)

            # Run analyst
            result = subprocess.run(
                [sys.executable, str(ANALYST_SCRIPT)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=60
            )

            # Log output
            for line in result.stdout.split('\n'):
                if line.strip():
                    self.log(f"  {line}")

            self.log("=" * 80)
            self.log("✅ Autonomous analysis completed")
            self.log("=" * 80)

            self.last_autonomous_analysis = datetime.now()
        except Exception as e:
            self.log(f"❌ Error in autonomous analysis: {e}")

    def run(self):
        """Main supervisor loop"""
        self.log("=" * 80)
        self.log("🎯 V2 BOT SUPERVISOR - STARTED")
        self.log("=" * 80)
        self.log(f"Check interval: {CHECK_INTERVAL} seconds")
        self.log(f"Daily report time: {DAILY_REPORT_HOUR}:00")
        self.log(f"Autonomous analysis: Every {AUTONOMOUS_ANALYSIS_INTERVAL//3600} hour(s)")
        self.log(f"Max restarts: {MAX_RESTART_ATTEMPTS}/hour")
        self.log("=" * 80)

        # Initial bot check
        if not self.is_bot_running():
            self.log("⚠️  V2 bot not running, starting...")
            self.start_bot()
        else:
            self.log("✅ V2 bot already running")

        # Main loop
        while self.running:
            try:
                # Check bot status
                if not self.is_bot_running():
                    self.log("🚨 ALERT: V2 bot stopped!")
                    self.restart_bot()

                # Check for alerts
                alert_count = self.check_alerts()

                # Send daily report if needed
                if self.should_send_daily_report():
                    self.send_daily_report()

                # Run autonomous analysis if needed
                if self.should_run_autonomous_analysis():
                    self.run_autonomous_analysis()

                # Wait for next check
                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                self.log("⏹️  Keyboard interrupt received")
                break
            except Exception as e:
                self.log(f"❌ Error in supervisor loop: {e}")
                time.sleep(CHECK_INTERVAL)

        self.log("👋 Supervisor shutdown complete")


if __name__ == "__main__":
    supervisor = Supervisor()
    supervisor.run()
