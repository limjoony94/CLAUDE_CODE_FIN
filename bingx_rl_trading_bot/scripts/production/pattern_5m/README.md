# pattern_5m/ — 메인 봇 모듈 패키지

14개 모듈로 구성된 Pattern 5m 트레이딩 봇 패키지.

## 모듈 구조

| 모듈 | 역할 |
|------|------|
| `bot.py` | 메인 루프, Early Exit, 헬스체크 |
| `config.py` | YAML 설정 로드 |
| `constants.py` | 패턴 목록, Per-pattern TP/SL 맵 ★ |
| `exchange.py` | BingX API 래퍼 (CCXT) |
| `indicators.py` | 기술 지표 (RSI, ATR 등) |
| `models.py` | 데이터클래스 정의 |
| `orders.py` | 주문 생성/관리, TP/SL 자동조정 |
| `position.py` | 포지션 관리 (facade) |
| `position_open.py` | 진입 + leverage side fix ★ |
| `position_monitor.py` | 포지션 모니터링 |
| `position_close.py` | 청산 |
| `signals.py` | 패턴 탐지 + Context Filter + Regime |
| `state.py` | 상태 저장/복구 (JSON) |
| `utils/` | lock, logging_config |

★ = v1.22.0에서 수정

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
- 패턴/TP-SL: `constants.py` 하드코딩
- API 키: `config/api_keys.yaml`
