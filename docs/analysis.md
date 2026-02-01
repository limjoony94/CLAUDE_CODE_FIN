# CLAUDE_CODE_FIN 프로젝트 분석

**분석일**: 2026-02-01 | **분석 대상**: Pattern 5m v1.22.0

---

## 1. 프로젝트 개요

BingX 거래소에서 BTC-USDT 선물을 5분봉 캔들 패턴 기반으로 자동 매매하는 트레이딩 봇.
12개 캔들 분류 체계(DOJI, HAMMER, MARUBOZU 등)의 3-캔들 조합 패턴을 감지하여 롱/숏 진입.

**현재 운영 전략**: 12 패턴 (7 Long + 5 Short), Per-pattern TP/SL, WR 80.3%, PF 3.36

---

## 2. 디렉토리 구조

```
CLAUDE_CODE_FIN/                       # 루트
├── CLAUDE.md                          # 프로젝트 메타 문서 (핵심)
├── docs/                              # 루트 레벨 문서 (TECH_STACK, CODING_CONVENTIONS, GIT_WORKFLOW)
├── claudedocs/                        # 루트 레벨 연구 문서 (cleanup 분석)
└── bingx_rl_trading_bot/              # 메인 프로젝트 디렉토리
    ├── config/                        # 설정 파일 (API 키, 전략 config)
    ├── scripts/
    │   ├── production/                # ★ 운영 코드
    │   │   ├── pattern_5m_bot.py      #   엔트리포인트
    │   │   ├── pattern_5m/            #   14개 모듈 패키지
    │   │   └── engulf_5m/             #   (Archived)
    │   ├── analysis/                  # 연구/분석 스크립트
    │   ├── training/                  # 모델 학습
    │   ├── monitoring/                # 모니터링 스크립트
    │   ├── data/                      # 데이터 수집
    │   └── (debug, deprecated, etc.)
    ├── src/                           # 레거시 코드 (RL agent, XGBoost 등)
    ├── models/                        # 학습된 모델 파일 (~3500개 중 대부분)
    ├── data/                          # 시장 데이터 CSV
    ├── results/                       # 봇 상태/메트릭 JSON
    ├── logs/                          # 운영 로그
    ├── claudedocs/                    # 연구 리포트
    ├── archive/                       # 아카이브
    └── experimental/                  # 실험 (Random Masking 등)
```

**총 파일 수**: ~3,550개 (대부분 models/ 디렉토리의 pkl/txt 파일)

---

## 3. 핵심 모듈 (pattern_5m/)

| 모듈 | 역할 | 에이전트 관련도 |
|------|------|----------------|
| `bot.py` | 메인 루프, Early Exit, 헬스체크 | automation |
| `config.py` | YAML 설정 로드 | dev |
| `constants.py` | 패턴 목록, Per-pattern TP/SL 맵 | dev |
| `exchange.py` | BingX API 래퍼 (CCXT) | dev |
| `signals.py` | 패턴 탐지, Context Filter | dev |
| `orders.py` | 주문 생성/관리, TP/SL 자동조정 | dev |
| `position_open.py` | 진입 로직 + leverage fix | dev |
| `position_monitor.py` | 포지션 모니터링 | monitor |
| `position_close.py` | 청산 로직 | dev |
| `state.py` | 상태 저장/복구 (JSON) | monitor |
| `models.py` | 데이터클래스 정의 | dev |
| `indicators.py` | 기술 지표 (RSI, ATR 등) | dev |

---

## 4. 데이터 흐름

```
BingX API → exchange.py → indicators.py → signals.py → position_open.py → orders.py
                                                            ↓
                                              position_monitor.py → position_close.py
                                                            ↓
                                              state.py (JSON) → results/ (metrics)
```

---

## 5. 설정 파일

| 파일 | 용도 |
|------|------|
| `config/pattern_5m_config.yaml` | 전략 파라미터, 리스크, 레버리지 |
| `config/api_keys.yaml` | BingX API 키 (민감) |
| `pattern_5m/constants.py` | Per-pattern TP/SL 하드코딩 |

---

## 6. 의존성

- **Python 3.12** (WSL2 Ubuntu)
- **ccxt**: BingX 거래소 API
- **pandas, numpy**: 데이터 처리
- **xgboost**: 레거시 ML 모델 (현재 미사용)
- **pyyaml**: 설정 파일
- **ta / ta-lib**: 기술 지표 (추정)

---

## 7. 현재 상태 요약

- **v1.22.0** 운영 중: DN-D-BD 과적합 패턴 제거 완료
- 270일 백테스트 검증 완료 (WR 80.3%, PF 3.36, WF 5/5)
- Standard Research Protocol 확립 (MC test, WF validation)
- 모듈러 아키텍처 (14개 모듈)로 유지보수성 양호
- `src/` 하위는 레거시 (RL/XGBoost 기반) → 현재 pattern 전략과 무관

---

## 8. 개선 필요 사항

1. **실행 스크립트**: .bat 파일 → Linux shell 스크립트 필요 (WSL 환경)
2. **모니터링**: metrics.json 수동 확인 → 자동 리포팅 필요
3. **로그 관리**: 로그 로테이션/정리 자동화 부재
4. **테스트**: 단위 테스트 부재
5. **의존성 관리**: requirements.txt / pyproject.toml 미확인
6. **models/ 정리**: 3000+ 레거시 모델 파일 → archive 필요
