"""
Periodic Model Retraining Scheduler

목적: 주기적으로 모델을 재훈련하여 최신 시장 데이터 반영
- 설정된 간격(예: 매주)마다 자동 재훈련
- 재훈련 전 데이터 다운로드
- 재훈련 중 봇 모니터링 (포지션 청산 대기)
- 재훈련 완료 후 성능 검증

사용법:
    # 설정 확인
    python scripts/production/periodic_retraining_scheduler.py --check

    # 스케줄러 실행 (백그라운드)
    python scripts/production/periodic_retraining_scheduler.py --start

    # 즉시 재훈련 (테스트용)
    python scripts/production/periodic_retraining_scheduler.py --now

    # Windows Task Scheduler 설정 스크립트 생성
    python scripts/production/periodic_retraining_scheduler.py --create-task
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import json
from loguru import logger

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "production"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Scripts
TRAIN_ALL_SCRIPT = SCRIPTS_DIR / "train_all_models.py"
BOT_SCRIPT = SCRIPTS_DIR / "phase4_dynamic_testnet_trading.py"

# Configuration
CONFIG_FILE = PROJECT_ROOT / "config" / "retraining_config.json"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "enabled": True,
    "interval_days": 7,  # Retrain every 7 days
    "retraining_time": "02:00",  # 2 AM (low activity)
    "wait_for_position_close": True,
    "max_wait_hours": 4,  # Max wait for position to close
    "download_data_before_training": True,
    "validate_after_training": True,
    "last_training": None,
    "next_training": None
}


def load_config() -> dict:
    """Load retraining configuration"""
    if not CONFIG_FILE.exists():
        logger.info("No config file found, creating default...")
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    # Merge with defaults (in case new fields were added)
    merged = DEFAULT_CONFIG.copy()
    merged.update(config)

    return merged


def save_config(config: dict):
    """Save retraining configuration"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    logger.success(f"✅ Config saved: {CONFIG_FILE}")


def check_bot_running() -> bool:
    """Check if trading bot is currently running"""
    lock_file = RESULTS_DIR / "bot_instance.lock"
    return lock_file.exists()


def check_open_position() -> bool:
    """Check if bot has open positions"""
    state_file = RESULTS_DIR / "phase4_testnet_trading_state.json"

    if not state_file.exists():
        return False

    try:
        with open(state_file, 'r') as f:
            state = json.load(f)

        trades = state.get('trades', [])
        open_trades = [t for t in trades if t.get('status') == 'OPEN']

        return len(open_trades) > 0

    except Exception as e:
        logger.error(f"Error checking positions: {e}")
        return False


def wait_for_position_close(max_wait_hours: int = 4) -> bool:
    """
    Wait for open positions to close

    Returns:
        True if all positions closed, False if timeout
    """
    logger.info(f"Waiting for positions to close (max: {max_wait_hours}h)...")

    start_time = datetime.now()
    check_interval = 60  # Check every 1 minute

    while (datetime.now() - start_time).total_seconds() < max_wait_hours * 3600:
        if not check_open_position():
            logger.success("✅ All positions closed")
            return True

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"   Still waiting... ({elapsed/60:.0f} min elapsed)")
        time.sleep(check_interval)

    logger.warning(f"⚠️ Timeout: Positions still open after {max_wait_hours}h")
    return False


def run_retraining() -> bool:
    """
    Execute model retraining

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("STARTING MODEL RETRAINING")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        # Run training script
        result = subprocess.run(
            [sys.executable, str(TRAIN_ALL_SCRIPT), "--download-data"],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(PROJECT_ROOT)
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            logger.success(f"✅ Retraining completed successfully ({elapsed:.1f}s)")
            return True
        else:
            logger.error(f"❌ Retraining failed (return code: {result.returncode})")
            logger.error(f"Error output:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("❌ Retraining timed out after 1 hour")
        return False

    except Exception as e:
        logger.error(f"❌ Retraining failed with exception: {e}")
        return False


def perform_retraining_cycle(config: dict) -> bool:
    """
    Perform complete retraining cycle

    1. Check if bot is running
    2. Wait for positions to close (if configured)
    3. Run retraining
    4. Update config with last training time

    Returns:
        True if successful, False otherwise
    """
    logger.info("\n" + "=" * 80)
    logger.info("RETRAINING CYCLE START")
    logger.info("=" * 80)
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check bot status
    bot_running = check_bot_running()
    open_position = check_open_position()

    logger.info(f"Bot Running: {bot_running}")
    logger.info(f"Open Position: {open_position}")

    # Wait for position to close (if configured)
    if config.get('wait_for_position_close') and open_position:
        max_wait = config.get('max_wait_hours', 4)
        position_closed = wait_for_position_close(max_wait)

        if not position_closed:
            logger.warning("⚠️ Retraining aborted: Positions still open")
            return False

    # Run retraining
    success = run_retraining()

    if success:
        # Update config
        config['last_training'] = datetime.now().isoformat()

        # Calculate next training
        interval_days = config.get('interval_days', 7)
        config['next_training'] = (datetime.now() + timedelta(days=interval_days)).isoformat()

        save_config(config)

        logger.success("✅ Retraining cycle completed successfully")
    else:
        logger.error("❌ Retraining cycle failed")

    logger.info("=" * 80)

    return success


def calculate_next_training_time(config: dict) -> datetime:
    """Calculate when next training should occur"""
    interval_days = config.get('interval_days', 7)
    target_time_str = config.get('retraining_time', '02:00')  # "HH:MM"

    # Parse target time
    target_hour, target_minute = map(int, target_time_str.split(':'))

    # Last training time
    last_training_str = config.get('last_training')

    if last_training_str:
        last_training = datetime.fromisoformat(last_training_str)
    else:
        # First time: schedule for next occurrence of target time
        last_training = datetime.now() - timedelta(days=interval_days)

    # Next training = last + interval
    next_training = last_training + timedelta(days=interval_days)

    # Adjust to target time
    next_training = next_training.replace(hour=target_hour, minute=target_minute, second=0)

    # If calculated time is in the past, add interval days
    while next_training < datetime.now():
        next_training += timedelta(days=interval_days)

    return next_training


def scheduler_loop(config: dict):
    """Main scheduler loop"""
    logger.info("=" * 80)
    logger.info("RETRAINING SCHEDULER STARTED")
    logger.info("=" * 80)
    logger.info(f"Interval: Every {config.get('interval_days')} days")
    logger.info(f"Target Time: {config.get('retraining_time')}")
    logger.info("=" * 80)

    while True:
        # Calculate next training time
        next_training = calculate_next_training_time(config)
        config['next_training'] = next_training.isoformat()
        save_config(config)

        logger.info(f"\nNext training scheduled: {next_training.strftime('%Y-%m-%d %H:%M:%S')}")

        # Wait until next training time
        while datetime.now() < next_training:
            time_remaining = (next_training - datetime.now()).total_seconds()
            hours_remaining = time_remaining / 3600

            if hours_remaining > 1:
                logger.info(f"   {hours_remaining:.1f} hours until next training...")
                time.sleep(3600)  # Check every hour
            else:
                logger.info(f"   {time_remaining/60:.0f} minutes until next training...")
                time.sleep(60)  # Check every minute

        # Time to retrain!
        if config.get('enabled'):
            perform_retraining_cycle(config)
        else:
            logger.warning("⚠️ Retraining disabled in config, skipping...")

        # Reload config (in case it was modified)
        config = load_config()


def create_windows_task():
    """Create Windows Task Scheduler XML configuration"""
    task_xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Periodic model retraining for BingX Trading Bot</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2025-01-01T02:00:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByWeek>
        <DaysOfWeek>
          <Monday />
        </DaysOfWeek>
        <WeeksInterval>1</WeeksInterval>
      </ScheduleByWeek>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT2H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions>
    <Exec>
      <Command>{sys.executable}</Command>
      <Arguments>{TRAIN_ALL_SCRIPT} --download-data</Arguments>
      <WorkingDirectory>{PROJECT_ROOT}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>"""

    task_file = PROJECT_ROOT / "retraining_task.xml"
    with open(task_file, 'w', encoding='utf-16') as f:
        f.write(task_xml)

    logger.success(f"✅ Task XML created: {task_file}")
    logger.info("\nTo import into Windows Task Scheduler:")
    logger.info("1. Open Task Scheduler")
    logger.info("2. Click 'Import Task...'")
    logger.info(f"3. Select: {task_file}")
    logger.info("4. Adjust settings as needed")
    logger.info("5. Save and enable the task")


def main():
    parser = argparse.ArgumentParser(description="Periodic model retraining scheduler")
    parser.add_argument('--check', action='store_true',
                       help='Check configuration and next training time')
    parser.add_argument('--start', action='store_true',
                       help='Start scheduler loop (runs indefinitely)')
    parser.add_argument('--now', action='store_true',
                       help='Run retraining immediately (for testing)')
    parser.add_argument('--create-task', action='store_true',
                       help='Create Windows Task Scheduler XML')

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    if args.check:
        logger.info("=" * 80)
        logger.info("RETRAINING CONFIGURATION")
        logger.info("=" * 80)
        for key, value in config.items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 80)

        next_training = calculate_next_training_time(config)
        logger.info(f"\nNext training: {next_training.strftime('%Y-%m-%d %H:%M:%S')}")

    elif args.now:
        success = perform_retraining_cycle(config)
        sys.exit(0 if success else 1)

    elif args.start:
        scheduler_loop(config)

    elif args.create_task:
        create_windows_task()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
