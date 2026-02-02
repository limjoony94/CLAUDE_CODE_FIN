"""
서버 시간 딜레이 + 시간 동기화 측정 스크립트
==========================================

BingX API의 5분봉 데이터 제공 딜레이 및 시간 동기화 상태를 측정합니다.

측정 항목:
1. 시간 동기화: 로컬 시스템 시간 vs BingX 서버 시간
2. 서버 딜레이: 정각 직후(+1초, +2초, +3초, +5초, +10초) 데이터 요청
3. 캔들 제공 시점: 완성된 캔들이 언제 제공되는지 파악

결과:
- 시간 오차(time drift) 측정 및 경고
- 최소 대기 시간 결정 (서버 딜레이 + 안전 여유)
- 완성된 캔들 필터링 기준 확립
"""

import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient
from scripts.production.opportunity_gating_bot_4x import load_api_keys

def check_time_synchronization(client):
    """로컬 시스템 시간과 BingX 서버 시간 동기화 확인"""

    print("="*80)
    print("시간 동기화 체크")
    print("="*80)
    print()

    time_drifts = []

    # 5회 측정하여 평균 계산
    for i in range(5):
        # 로컬 시간 (요청 직전)
        local_time_before = datetime.now()

        # 서버로부터 최신 캔들 데이터 요청
        try:
            klines = client.get_klines("BTC-USDT", "5m", limit=1)

            # 로컬 시간 (응답 직후)
            local_time_after = datetime.now()

            if klines and len(klines) > 0:
                # 서버 캔들 시간
                df = pd.DataFrame(klines)
                df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
                server_candle_time = df.iloc[-1]['timestamp']

                # 네트워크 지연 시간 보정 (왕복 시간의 절반)
                network_delay = (local_time_after - local_time_before).total_seconds() / 2
                local_time_adjusted = local_time_before + timedelta(seconds=network_delay)

                # 시간 차이 계산 (서버 시간은 완성된 캔들 시작 시간이므로 현재 시간과 비교하기 위해 조정)
                # 현재 5분 구간의 시작 시간 계산
                local_candle_start = local_time_adjusted.replace(second=0, microsecond=0)
                local_candle_start = local_candle_start - timedelta(minutes=local_candle_start.minute % 5)

                # 시간 오차
                time_drift = (local_candle_start - server_candle_time).total_seconds()
                time_drifts.append(time_drift)

                print(f"측정 {i+1}:")
                print(f"  로컬 시간: {local_time_adjusted.strftime('%H:%M:%S.%f')[:-3]}")
                print(f"  서버 캔들: {server_candle_time.strftime('%H:%M:%S')}")
                print(f"  시간 오차: {time_drift:+.3f}초")

        except Exception as e:
            print(f"  ❌ 오류: {e}")

        if i < 4:
            time.sleep(1)

    print()

    if time_drifts:
        avg_drift = sum(time_drifts) / len(time_drifts)
        abs_drift = abs(avg_drift)

        print(f"📊 평균 시간 오차: {avg_drift:+.3f}초")
        print()

        # 시간 오차 평가
        if abs_drift < 0.5:
            print("✅ 시간 동기화 상태: 양호 (< 0.5초)")
            print("   정각 동기화 사용 가능")
        elif abs_drift < 1.0:
            print("⚠️  시간 동기화 상태: 주의 (0.5~1.0초)")
            print("   정각 동기화 사용 가능하지만 여유 시간 추가 권장")
        elif abs_drift < 2.0:
            print("⚠️  시간 동기화 상태: 불량 (1.0~2.0초)")
            print("   시스템 시간 동기화 권장 (NTP 설정)")
        else:
            print("❌ 시간 동기화 상태: 심각 (> 2.0초)")
            print("   ⚠️  정각 동기화 사용 불가")
            print("   ⚠️  반드시 시스템 시간을 NTP 서버와 동기화하세요")
            print()
            print("Windows 시간 동기화 방법:")
            print("  1. 설정 → 시간 및 언어 → 날짜 및 시간")
            print("  2. '지금 동기화' 클릭")
            print("  3. 또는 명령 프롬프트(관리자):")
            print("     w32tm /resync")

        print()
        return avg_drift

    else:
        print("❌ 시간 동기화 체크 실패\n")
        return None

def measure_candle_delay():
    """5분봉 완성 캔들 제공 시간 측정"""

    print("\n" + "="*80)
    print("BingX 서버 딜레이 + 시간 동기화 측정")
    print("="*80)
    print()

    # API 초기화
    api_keys = load_api_keys()
    client = BingXClient(
        api_key=api_keys['api_key'],
        secret_key=api_keys['secret_key'],
        testnet=api_keys['testnet']
    )

    print("✅ BingX API 연결 완료\n")

    # STEP 1: 시간 동기화 체크
    time_drift = check_time_synchronization(client)

    if time_drift is not None and abs(time_drift) > 2.0:
        print("="*80)
        print("⚠️  경고: 시간 오차가 너무 커서 서버 딜레이 측정 중단")
        print("   먼저 시스템 시간을 동기화하세요.")
        print("="*80)
        return

    # STEP 2: 서버 딜레이 측정
    print("="*80)
    print("서버 캔들 제공 딜레이 측정")
    print("="*80)
    print()

    # 다음 5분 정각까지 대기
    now = datetime.now()
    seconds_to_next = 300 - (now.minute % 5) * 60 - now.second
    next_candle_time = now + timedelta(seconds=seconds_to_next)

    print(f"현재 시간: {now.strftime('%H:%M:%S')}")
    print(f"다음 5분 정각: {next_candle_time.strftime('%H:%M:%S')}")
    print(f"대기 시간: {seconds_to_next}초\n")

    if seconds_to_next < 30:
        print("⚠️  정각이 너무 가까움 (30초 미만)")
        print("   다음 정각까지 추가 대기...\n")
        time.sleep(seconds_to_next + 5)
        seconds_to_next = 295
        next_candle_time = datetime.now() + timedelta(seconds=seconds_to_next)

    print(f"⏳ {seconds_to_next}초 대기 중...")
    time.sleep(seconds_to_next)

    print("\n" + "="*80)
    print("측정 시작 - 정각 직후 여러 시점에서 데이터 요청")
    print("="*80 + "\n")

    test_delays = [1, 2, 3, 5, 10]  # 정각 후 N초에 테스트
    results = []

    for delay_sec in test_delays:
        print(f"📊 테스트 {delay_sec}: 정각 +{delay_sec}초 후 요청")

        # 정각 + delay_sec 까지 대기
        time.sleep(delay_sec)

        # 요청 시작 시간
        request_time = datetime.now()

        # 데이터 요청
        try:
            klines = client.get_klines("BTC-USDT", "5m", limit=3)
            response_time = datetime.now()

            if klines and len(klines) > 0:
                # DataFrame 변환
                df = pd.DataFrame(klines)
                df['timestamp'] = pd.to_datetime(df['time'], unit='ms')

                # 최신 캔들 정보
                latest_candle = df.iloc[-1]
                latest_timestamp = latest_candle['timestamp']

                # 이전 캔들 정보
                prev_candle = df.iloc[-2] if len(df) > 1 else None
                prev_timestamp = prev_candle['timestamp'] if prev_candle is not None else None

                # 딜레이 계산
                network_delay = (response_time - request_time).total_seconds()

                # 완성된 정각 계산 (최신 캔들 시간의 5분 전)
                expected_completed_candle = latest_timestamp - timedelta(minutes=5)

                print(f"   요청 시간: {request_time.strftime('%H:%M:%S.%f')[:-3]}")
                print(f"   응답 시간: {response_time.strftime('%H:%M:%S.%f')[:-3]}")
                print(f"   네트워크 딜레이: {network_delay:.3f}초")
                print(f"   최신 캔들: {latest_timestamp.strftime('%H:%M:%S')}")
                if prev_timestamp:
                    print(f"   이전 캔들: {prev_timestamp.strftime('%H:%M:%S')}")

                # 정각 기준 계산
                candle_expected = next_candle_time.replace(microsecond=0)

                # 최신 캔들이 정각 캔들인지 확인
                is_new_candle = latest_timestamp >= candle_expected

                print(f"   정각 캔들: {candle_expected.strftime('%H:%M:%S')}")
                print(f"   정각 캔들 제공: {'✅ YES' if is_new_candle else '❌ NOT YET'}")

                results.append({
                    'delay_sec': delay_sec,
                    'request_time': request_time,
                    'response_time': response_time,
                    'network_delay': network_delay,
                    'latest_candle': latest_timestamp,
                    'is_new_candle': is_new_candle
                })

                print()

            else:
                print("   ❌ 데이터 요청 실패\n")

        except Exception as e:
            print(f"   ❌ 오류: {e}\n")

        # 다음 테스트까지 대기 (총 5분 대기 방지)
        if delay_sec < test_delays[-1]:
            wait_time = test_delays[test_delays.index(delay_sec) + 1] - delay_sec
            time.sleep(wait_time - 1)  # 1초는 이미 사용

    # 결과 분석
    print("\n" + "="*80)
    print("📊 종합 측정 결과")
    print("="*80 + "\n")

    # 시간 동기화 결과
    print("1️⃣ 시간 동기화 상태:")
    if time_drift is not None:
        print(f"   평균 시간 오차: {time_drift:+.3f}초")
        if abs(time_drift) < 0.5:
            print("   상태: ✅ 양호")
        elif abs(time_drift) < 1.0:
            print("   상태: ⚠️  주의 (여유 시간 추가 권장)")
        else:
            print("   상태: ⚠️  불량 (NTP 동기화 권장)")
    else:
        print("   상태: ❌ 측정 실패")
    print()

    # 서버 딜레이 결과
    print("2️⃣ 서버 캔들 제공 딜레이:")

    if results:
        # 정각 캔들이 처음 제공된 시점 찾기
        first_new_candle = None
        for r in results:
            if r['is_new_candle']:
                first_new_candle = r
                break

        if first_new_candle:
            min_delay = first_new_candle['delay_sec']
            avg_network_delay = sum(r['network_delay'] for r in results)/len(results)

            print(f"   정각 캔들 제공 시작: 정각 +{min_delay}초")
            print(f"   평균 네트워크 딜레이: {avg_network_delay:.3f}초")
            print()

            # 시간 오차를 반영한 권장 설정
            base_delay = min_delay
            safety_margin = 1  # 기본 안전 여유

            if time_drift is not None and abs(time_drift) >= 0.5:
                safety_margin = 2  # 시간 오차가 크면 여유 증가

            recommended_delay = base_delay + safety_margin

            print(f"3️⃣ 최종 권장 설정:")
            print(f"   서버 처리 시간: {min_delay}초")
            print(f"   시간 오차 반영: {'+1초 여유' if time_drift is not None and abs(time_drift) >= 0.5 else '미반영'}")
            print(f"   안전 여유: +{safety_margin}초")
            print(f"   ")
            print(f"   ✅ 권장 대기 시간: 정각 +{recommended_delay}초")
            print(f"   ")
            print(f"   예시:")
            print(f"     22:15:00 정각")
            print(f"     → 22:15:{recommended_delay:02d}에 데이터 요청")
            print(f"     → 완성된 캔들만 계산에 사용 (22:10:00 캔들까지)")
            print()

            # 추가 안전 장치 안내
            print(f"   📋 2중 안전장치:")
            print(f"     1. 정각 +{recommended_delay}초 대기 (서버 딜레이 + 여유)")
            print(f"     2. 완성된 캔들 필터링 (진행중 캔들 제외)")
            print(f"     → 백테스트와 동일한 조건 보장")
        else:
            print("⚠️  테스트 범위 내에서 정각 캔들 미제공")
            print("   더 긴 대기 시간 필요\n")

        # 상세 결과 테이블
        print("\n상세 측정 결과:")
        print("-" * 80)
        print(f"{'정각+':<8} {'네트워크':>10} {'최신 캔들':>20} {'정각 캔들':>12}")
        print("-" * 80)
        for r in results:
            status = "✅" if r['is_new_candle'] else "❌"
            print(f"{r['delay_sec']:>3}초     {r['network_delay']:>8.3f}초  {r['latest_candle'].strftime('%H:%M:%S'):>20}  {status:>12}")
        print("-" * 80)

    else:
        print("❌ 측정 실패 - 결과 없음")

    print("\n" + "="*80)
    print("측정 완료")
    print("="*80)

if __name__ == "__main__":
    try:
        measure_candle_delay()
    except KeyboardInterrupt:
        print("\n\n측정 중단됨")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
