"""
실시간 로그 모니터링 스크립트
Live Log Monitor for BingX Trading Bot
"""
import time
import sys
from pathlib import Path

def tail_log(log_file: Path, initial_lines: int = 50):
    """
    실시간으로 로그 파일을 모니터링

    Args:
        log_file: 로그 파일 경로
        initial_lines: 시작 시 표시할 이전 라인 수
    """
    print("=" * 80)
    print("🔴 LIVE LOG MONITOR - BingX Trading Bot")
    print("=" * 80)
    print(f"📂 Monitoring: {log_file}")
    print(f"⏱️  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 80)
    print()

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            # 파일 끝으로 이동
            f.seek(0, 2)
            file_size = f.tell()

            # 마지막 N줄 읽기
            f.seek(0)
            lines = f.readlines()
            for line in lines[-initial_lines:]:
                print(line, end='')

            print()
            print("=" * 80)
            print("📡 LIVE UPDATES (새 로그만 표시됩니다)")
            print("=" * 80)
            print()

            # 실시간 모니터링
            while True:
                line = f.readline()
                if line:
                    # 중요한 정보는 강조 표시
                    if 'ERROR' in line or '❌' in line:
                        print(f"\033[91m{line}\033[0m", end='')  # 빨간색
                    elif 'WARNING' in line or '⚠️' in line:
                        print(f"\033[93m{line}\033[0m", end='')  # 노란색
                    elif 'Position:' in line or 'P&L:' in line or '✅' in line:
                        print(f"\033[92m{line}\033[0m", end='')  # 초록색
                    elif 'Total Net P&L:' in line or 'Strategy Return:' in line:
                        print(f"\033[96m{line}\033[0m", end='')  # 청록색
                    else:
                        print(line, end='')
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)

    except KeyboardInterrupt:
        print()
        print()
        print("=" * 80)
        print("🛑 Log monitoring stopped by user")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_file = log_dir / "phase4_dynamic_testnet_trading_20251014.log"

    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        sys.exit(1)

    tail_log(log_file, initial_lines=50)
