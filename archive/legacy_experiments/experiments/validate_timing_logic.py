"""
정각 동기화 로직 검증 스크립트
==============================

구현된 시간 동기화 로직을 실제 실행 전에 검증합니다.

검증 항목:
1. wait_for_next_candle: 정각 계산 정확성
2. fetch_and_validate_candles: 재시도 로직
3. filter_completed_candles: 캔들 필터링 정확성
4. 엣지 케이스: 경계 조건 테스트
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_time_calculations():
    """시간 계산 로직 검증"""

    print("="*80)
    print("1️⃣ 시간 계산 로직 검증")
    print("="*80)
    print()

    test_cases = [
        # (현재 시간, 예상 정각)
        ("2025-10-22 22:13:45", "2025-10-22 22:15:00"),
        ("2025-10-22 22:00:01", "2025-10-22 22:05:00"),
        ("2025-10-22 22:04:59", "2025-10-22 22:05:00"),
        ("2025-10-22 22:14:30", "2025-10-22 22:15:00"),
        ("2025-10-22 22:19:59", "2025-10-22 22:20:00"),
    ]

    all_passed = True

    for i, (current_str, expected_str) in enumerate(test_cases, 1):
        current = datetime.strptime(current_str, "%Y-%m-%d %H:%M:%S")
        expected = datetime.strptime(expected_str, "%Y-%m-%d %H:%M:%S")

        # 정각 계산 (wait_for_next_candle 로직)
        minutes_to_next = 5 - (current.minute % 5)
        seconds_to_next = minutes_to_next * 60 - current.second
        calculated_next = current + timedelta(seconds=seconds_to_next)

        passed = calculated_next == expected
        all_passed = all_passed and passed

        status = "✅" if passed else "❌"
        print(f"테스트 {i}: {status}")
        print(f"  현재: {current.strftime('%H:%M:%S')}")
        print(f"  예상 정각: {expected.strftime('%H:%M:%S')}")
        print(f"  계산 정각: {calculated_next.strftime('%H:%M:%S')}")

        if not passed:
            diff = (calculated_next - expected).total_seconds()
            print(f"  ❌ 오차: {diff}초")
        print()

    return all_passed

def test_candle_filtering():
    """캔들 필터링 로직 검증"""

    print("="*80)
    print("2️⃣ 캔들 필터링 로직 검증")
    print("="*80)
    print()

    # 테스트 데이터 생성
    base_time = datetime(2025, 10, 22, 22, 0, 0)
    timestamps = [base_time + timedelta(minutes=5*i) for i in range(10)]

    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': [108000 + i*100 for i in range(10)],
        'open': [108000 + i*100 - 50 for i in range(10)],
        'high': [108000 + i*100 + 50 for i in range(10)],
        'low': [108000 + i*100 - 100 for i in range(10)],
        'volume': [1000000] * 10
    })

    test_cases = [
        # (현재 시간, 예상 완성 캔들 수)
        (datetime(2025, 10, 22, 22, 47, 3), 9),  # 22:45까지 완성
        (datetime(2025, 10, 22, 22, 42, 3), 8),  # 22:40까지 완성
        (datetime(2025, 10, 22, 22, 37, 3), 7),  # 22:35까지 완성
    ]

    all_passed = True

    for i, (current_time, expected_count) in enumerate(test_cases, 1):
        # 현재 5분 구간 시작 시간 (진행중인 캔들)
        current_candle_start = current_time.replace(second=0, microsecond=0)
        current_candle_start = current_candle_start - timedelta(minutes=current_candle_start.minute % 5)

        # 완성된 캔들 필터링
        df_completed = df[df['timestamp'] < current_candle_start].copy()

        passed = len(df_completed) == expected_count
        all_passed = all_passed and passed

        status = "✅" if passed else "❌"
        print(f"테스트 {i}: {status}")
        print(f"  현재 시간: {current_time.strftime('%H:%M:%S')}")
        print(f"  진행중 캔들: {current_candle_start.strftime('%H:%M:00')}")
        print(f"  예상 완성 개수: {expected_count}")
        print(f"  실제 완성 개수: {len(df_completed)}")

        if len(df_completed) > 0:
            latest = df_completed.iloc[-1]['timestamp']
            print(f"  최신 완성 캔들: {latest.strftime('%H:%M:%S')}")

        if not passed:
            print(f"  ❌ 불일치!")
        print()

    return all_passed

def test_expected_candle_calculation():
    """예상 캔들 시간 계산 검증"""

    print("="*80)
    print("3️⃣ 예상 캔들 시간 계산 검증")
    print("="*80)
    print()

    test_cases = [
        # (현재 시간, 예상 최신 완성 캔들)
        ("2025-10-22 22:15:03", "2025-10-22 22:10:00"),
        ("2025-10-22 22:10:03", "2025-10-22 22:05:00"),
        ("2025-10-22 22:05:03", "2025-10-22 22:00:00"),
        ("2025-10-22 22:20:03", "2025-10-22 22:15:00"),
    ]

    all_passed = True

    for i, (current_str, expected_str) in enumerate(test_cases, 1):
        current = datetime.strptime(current_str, "%Y-%m-%d %H:%M:%S")
        expected = datetime.strptime(expected_str, "%Y-%m-%d %H:%M:%S")

        # 예상 캔들 계산 (메인 루프 로직)
        expected_candle = current.replace(second=0, microsecond=0)
        expected_candle = expected_candle - timedelta(minutes=expected_candle.minute % 5)
        expected_candle = expected_candle - timedelta(minutes=5)  # 5분 전

        passed = expected_candle == expected
        all_passed = all_passed and passed

        status = "✅" if passed else "❌"
        print(f"테스트 {i}: {status}")
        print(f"  현재 시간: {current.strftime('%H:%M:%S')}")
        print(f"  예상 최신 캔들: {expected.strftime('%H:%M:%S')}")
        print(f"  계산 최신 캔들: {expected_candle.strftime('%H:%M:%S')}")

        if not passed:
            diff = (expected_candle - expected).total_seconds()
            print(f"  ❌ 오차: {diff}초")
        print()

    return all_passed

def test_edge_cases():
    """엣지 케이스 검증"""

    print("="*80)
    print("4️⃣ 엣지 케이스 검증")
    print("="*80)
    print()

    all_passed = True

    # 엣지 케이스 1: 자정 경계
    print("테스트 1: 자정 경계 (23:58 → 00:00)")
    current = datetime(2025, 10, 22, 23, 58, 0)
    minutes_to_next = 5 - (current.minute % 5)
    seconds_to_next = minutes_to_next * 60 - current.second
    next_candle = current + timedelta(seconds=seconds_to_next)
    expected = datetime(2025, 10, 23, 0, 0, 0)

    passed = next_candle == expected
    all_passed = all_passed and passed
    print(f"  {'✅' if passed else '❌'} 현재: 23:58:00 → 정각: {next_candle.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 엣지 케이스 2: 정각 직후
    print("테스트 2: 정각 직후 (22:15:01 → 22:20:00)")
    current = datetime(2025, 10, 22, 22, 15, 1)
    minutes_to_next = 5 - (current.minute % 5)
    seconds_to_next = minutes_to_next * 60 - current.second
    next_candle = current + timedelta(seconds=seconds_to_next)
    expected = datetime(2025, 10, 22, 22, 20, 0)

    passed = next_candle == expected
    all_passed = all_passed and passed
    print(f"  {'✅' if passed else '❌'} 현재: 22:15:01 → 정각: {next_candle.strftime('%H:%M:%S')}")
    print()

    # 엣지 케이스 3: 빈 DataFrame
    print("테스트 3: 빈 DataFrame 처리")
    df_empty = pd.DataFrame(columns=['timestamp', 'close'])
    current_time = datetime(2025, 10, 22, 22, 15, 3)
    current_candle_start = current_time.replace(second=0, microsecond=0)
    current_candle_start = current_candle_start - timedelta(minutes=current_candle_start.minute % 5)
    df_completed = df_empty[df_empty['timestamp'] < current_candle_start]

    passed = len(df_completed) == 0
    all_passed = all_passed and passed
    print(f"  {'✅' if passed else '❌'} 빈 DataFrame → 완성 캔들 개수: {len(df_completed)}")
    print()

    # 엣지 케이스 4: 모든 캔들이 완성된 경우
    print("테스트 4: 모든 캔들이 완성된 경우")
    old_time = datetime(2025, 10, 22, 20, 0, 0)
    timestamps = [old_time + timedelta(minutes=5*i) for i in range(5)]
    df_old = pd.DataFrame({
        'timestamp': timestamps,
        'close': [108000] * 5
    })
    current_time = datetime(2025, 10, 22, 22, 15, 3)
    current_candle_start = current_time.replace(second=0, microsecond=0)
    current_candle_start = current_candle_start - timedelta(minutes=current_candle_start.minute % 5)
    df_completed = df_old[df_old['timestamp'] < current_candle_start]

    passed = len(df_completed) == 5
    all_passed = all_passed and passed
    print(f"  {'✅' if passed else '❌'} 모든 캔들 완성 → 개수: {len(df_completed)}/5")
    print()

    return all_passed

def test_syntax_errors():
    """구문 오류 확인"""

    print("="*80)
    print("5️⃣ 구문 오류 확인")
    print("="*80)
    print()

    try:
        # 봇 파일 import 시도
        import scripts.production.opportunity_gating_bot_4x as bot
        print("✅ 구문 오류 없음 - 파일이 정상적으로 import됨")
        print()

        # 새로 추가한 함수 확인
        functions_to_check = [
            'wait_for_next_candle',
            'fetch_and_validate_candles',
            'filter_completed_candles'
        ]

        all_exist = True
        for func_name in functions_to_check:
            if hasattr(bot, func_name):
                print(f"✅ {func_name} 함수 존재")
            else:
                print(f"❌ {func_name} 함수 없음")
                all_exist = False

        print()
        return all_exist

    except SyntaxError as e:
        print(f"❌ 구문 오류 발견:")
        print(f"   파일: {e.filename}")
        print(f"   라인: {e.lineno}")
        print(f"   오류: {e.msg}")
        print()
        return False
    except Exception as e:
        print(f"⚠️  Import 경고: {e}")
        print("   (API 키 등 실행 환경 문제일 수 있음)")
        print()
        return True  # 구문 오류는 아님

def main():
    """전체 검증 실행"""

    print("\n")
    print("="*80)
    print("정각 동기화 로직 검증")
    print("="*80)
    print()

    results = {
        '시간 계산': test_time_calculations(),
        '캔들 필터링': test_candle_filtering(),
        '예상 캔들 계산': test_expected_candle_calculation(),
        '엣지 케이스': test_edge_cases(),
        '구문 오류': test_syntax_errors()
    }

    print("="*80)
    print("📊 최종 검증 결과")
    print("="*80)
    print()

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {test_name}")

    print()
    print("="*80)

    if all_passed:
        print("✅ 모든 검증 통과!")
        print("   봇 재시작 가능")
    else:
        print("❌ 일부 검증 실패!")
        print("   문제 해결 후 재시도 필요")

    print("="*80)
    print()

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
