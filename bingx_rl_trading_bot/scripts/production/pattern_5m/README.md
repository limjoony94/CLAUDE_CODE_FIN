# pattern_5m/ — 메인 봇 모듈 패키지

> **Version**: v1.27.0 | **Updated**: 2026-02-11

14개 모듈로 구성된 Pattern 5m 트레이딩 봇 패키지 (52패턴, 32L+20S).

## 모듈 구조

| 모듈 | 역할 |
|------|------|
| `bot.py` | 메인 루프, Early Exit, 헬스체크, 연속손실 pause |
| `config.py` | YAML 설정 로드 |
| `constants.py` | 52패턴 목록, Uniform TP 70% TP/SL, 리스크 파라미터 |
| `exchange.py` | BingX API 래퍼 (CCXT) |
| `indicators.py` | 기술 지표 (RSI, ATR, 캔들 분류) |
| `models.py` | 데이터클래스 정의 |
| `orders.py` | 주문 생성/관리, TP/SL 자동조정 |
| `position.py` | 포지션 관리 (facade) |
| `position_open.py` | 진입 로직, TP/SL 계산 |
| `position_monitor.py` | 포지션 모니터링 |
| `position_close.py` | 청산, 일일손실 7% 제한 |
| `signals.py` | 패턴 탐지 + Context Filter |
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
- 패턴/TP-SL: `constants.py` (52패턴 + Uniform TP 70%)
- 리스크: 일일손실 7%, 연속 3회 손실 pause (600s)
- API 키: `config/api_keys.yaml`
