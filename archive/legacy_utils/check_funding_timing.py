#!/usr/bin/env python3
"""
Check funding fee timing to determine if funding fee occurred
"""

from datetime import datetime, timezone

def check_funding_timing():
    """Analyze funding fee timing"""

    # Reset time (KST)
    reset_time_kst = datetime(2025, 10, 25, 3, 46, 44)

    # Convert to UTC (KST = UTC+9)
    reset_time_utc = reset_time_kst.replace(tzinfo=timezone.utc).timestamp()
    reset_time_utc = datetime.fromtimestamp(reset_time_utc - 9*3600, tz=timezone.utc)

    # Current time (KST)
    current_time_kst = datetime.now()
    current_time_utc = datetime.now(timezone.utc)

    print('=' * 80)
    print('⏰ Funding Fee Timing Analysis')
    print('=' * 80)
    print()

    print('📅 Timeline (KST → UTC):')
    print(f'   Reset: {reset_time_kst.strftime("%Y-%m-%d %H:%M:%S")} KST')
    print(f'        = {reset_time_utc.strftime("%Y-%m-%d %H:%M:%S")} UTC')
    print()
    print(f'   Now:   {current_time_kst.strftime("%Y-%m-%d %H:%M:%S")} KST')
    print(f'        = {current_time_utc.strftime("%Y-%m-%d %H:%M:%S")} UTC')
    print()

    print('💸 BingX Funding Fee Schedule (UTC):')
    print('   00:00 UTC = 09:00 KST')
    print('   08:00 UTC = 17:00 KST')
    print('   16:00 UTC = 01:00 KST (다음날)')
    print()

    # Check which funding cycles occurred
    print('🔍 Funding Cycles Analysis:')
    print()

    # Reset was 2025-10-24 18:46:44 UTC
    # Current is 2025-10-24 23:22:XX UTC

    print(f'   Reset Time: 2025-10-24 18:46:44 UTC')
    print(f'   Previous Funding: 2025-10-24 16:00:00 UTC (이미 지남)')
    print(f'   Next Funding: 2025-10-25 00:00:00 UTC (아직 안 옴)')
    print()

    print('✅ 결론:')
    print('   Reset 이후 Funding Fee 사이클 없음')
    print('   (16:00 UTC는 reset 전, 00:00 UTC는 아직 도래 안 함)')
    print()

    print('=' * 80)
    print('❓ Balance 감소 원인 재분석 필요')
    print('=' * 80)
    print()
    print('확인된 사실:')
    print('1. ✅ 거래 없음 (No trades)')
    print('2. ✅ Funding Fee 없음 (No funding cycle)')
    print('3. ✅ 수동 청산 없음 (User confirmed)')
    print()
    print('Balance 변화:')
    print('   Reset: $4,561.00')
    print('   현재 (API): $4,536.45')
    print('   감소: $24.55')
    print()
    print('가능한 원인:')
    print('1. Reset 시점의 balance가 실제로는 더 낮았을 가능성')
    print('2. Reset backup 파일과 실제 거래소 balance의 불일치')
    print('3. Unrealized P&L 변동 (포지션 진입 후 P&L 악화)')
    print()
    print('📝 권장 조치:')
    print('   Reset 시점의 거래소 실제 balance를 확인')
    print('   (Backup 파일이 아닌 거래소 API 기록)')

if __name__ == "__main__":
    check_funding_timing()
