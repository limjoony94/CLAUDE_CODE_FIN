# pattern_5m/ — 메인 봇 모듈 패키지

> **Version**: v1.56.2 | **Updated**: 2026-03-12

14개 모듈로 구성된 Pattern 5m 트레이딩 봇 패키지 (131패턴, 59L+72S).

## 모듈 구조

| 모듈 | 역할 |
|------|------|
| `bot.py` | 메인 루프, Early Exit, 헬스체크, Momentum Guard, MDD Sizing |
| `config.py` | YAML 설정 로드 + Dynamic Pattern 로딩 |
| `constants.py` | 패턴 목록, Per-pattern TP/SL, 리스크 파라미터, BOT_VERSION |
| `exchange.py` | BingX API 래퍼 (CCXT), Circuit Breaker, API Caching (TTL 5s) |
| `indicators.py` | 기술 지표 (RSI, ATR, 캔들 분류 12-type Ground Truth) |
| `models.py` | 데이터클래스 정의 |
| `orders.py` | 주문 생성/관리, TP/SL 배치/검증, Emergency SL, Cascade SL |
| `position.py` | 포지션 관리 (facade) |
| `position_open.py` | 진입 로직, 1/N sizing, Direction Cap, AggRisk check |
| `position_monitor.py` | 포지션 모니터링, Cascade SL tightening, Timeout |
| `position_close.py` | 청산, Crash Recovery, Exit classification |
| `signals.py` | 패턴 탐지 + Context Filter 인프라 (현재 비활성) |
| `state.py` | 상태 저장/복구 (atomic write, .bak) |
| `utils/` | lock, logging_config |

## 데이터 흐름

```
exchange.py → indicators.py → signals.py → position_open.py → orders.py
                                                ↓
                                  position_monitor.py → position_close.py
                                                ↓
                                  state.py → results/ (JSON)
```

## 주요 설정

- 전략 파라미터: `config/pattern_5m_config.yaml`
- 패턴/TP-SL: Dynamic (`results/dynamic_patterns.json`, 131패턴, ATR-scaled)
- 리스크: 일일손실 13%, Aggregate risk (counter 8%/with 15%), Direction Cap 7
- 포지션: Hedge mode, Max 9 slots, 1/N=11.1% sizing, Timeout 288bars (24h)
- 보호: Cascade SL (85% tightening), Emergency SL (closePosition=true), Momentum Guard
- API 키: `config/api_keys.yaml`
