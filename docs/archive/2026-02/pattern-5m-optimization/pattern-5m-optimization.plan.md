# Plan: pattern-5m-optimization

> Opus 4.6 전면 코드 리뷰 및 최적화

## 목표
- Opus 4.5로 작성된 14개 핵심 모듈을 Opus 4.6 관점에서 전면 재검토
- 코드 품질, 성능, 안정성, 구조적 개선점 도출 및 적용
- 봇 기능 변경 없이 내부 품질만 향상 (리팩토링)

## 범위

### 핵심 모듈 (14개)
| # | 모듈 | 책임 | 크기 |
|---|------|------|------|
| 1 | bot.py | 메인 루프 | 22KB |
| 2 | signals.py | 패턴 탐지 + Context Filter | 27KB |
| 3 | position_open.py | 진입 로직 | 23KB |
| 4 | constants.py | 패턴 정의 + TP/SL | 20KB |
| 5 | position_close.py | 청산 로직 | 21KB |
| 6 | orders.py | 주문 관리 | 18KB |
| 7 | exchange.py | BingX API | 15KB |
| 8 | models.py | 데이터클래스 | 11KB |
| 9 | position_monitor.py | 포지션 모니터링 | 12KB |
| 10 | state.py | 상태 저장 | 11KB |
| 11 | indicators.py | 기술 지표 | 8.4KB |
| 12 | config.py | 설정 로드 | 4.1KB |
| 13 | position.py | Facade | 1.2KB |
| 14 | utils/lock.py | 프로세스 잠금 | ~5KB |

### 보조 파일
- utils/logging_config.py
- pattern_5m_bot.py (엔트리포인트)

## 검토 기준

### 1. 코드 품질
- 중복 코드, dead code
- 불필요한 복잡성 (과도한 try/except, 중첩)
- 네이밍 일관성
- 매직 넘버

### 2. 성능
- 불필요한 API 호출
- 비효율적 연산 (반복 계산, 불필요 루프)
- 메모리 사용 패턴

### 3. 안정성
- 엣지 케이스 미처리
- Race condition 가능성
- 에러 복구 로직 검증
- 데이터 무결성

### 4. 구조
- 모듈 간 순환 의존성
- 단일 책임 원칙 위반
- 인터페이스 명확성

## 제약 조건
- 봇이 실시간 운영 중 — 기능 변경 금지
- TP/SL, 패턴, 전략 로직 변경 금지
- 외부 동작(API 호출 형식, 상태 파일 구조) 유지
- 테스트 통과 필수

## 성공 기준
- 모든 모듈 리뷰 완료
- 발견된 이슈에 대한 수정 적용
- 봇 재시작 후 정상 동작 확인
- Gap analysis match rate >= 90%
