# CLAUDE_CODE_FIN - BTC 5분봉 패턴 트레이딩 봇

> **Version**: v1.27.0 | **Bot**: Pattern 5m (52패턴, 32L+20S) | **Updated**: 2026-02-10

---

## ⚡ 빠른 참조

| 항목 | 경로 |
|------|------|
| 엔트리 | `bingx_rl_trading_bot/scripts/production/pattern_5m_bot.py` |
| 모듈 | `bingx_rl_trading_bot/scripts/production/pattern_5m/` (14개) |
| 설정 | `bingx_rl_trading_bot/config/pattern_5m_config.yaml` |
| 상수/TP-SL | `bingx_rl_trading_bot/scripts/production/pattern_5m/constants.py` |
| 상태 | `results/pattern_5m_bot_state.json` |
| 메트릭 | `results/pattern_5m_metrics.json` |
| 로그 | `logs/pattern_5m_bot_*.log` |
| 데이터 | `data/btc_5m_270days_reclassified.csv` (270일, Ground Truth) |

---

## 🎯 에이전트별 가이드

### dev — 코드/전략/연구
- **수정 대상**: `scripts/production/pattern_5m/`, `config/`, `scripts/analysis/`
- **프로토콜**: 아래 Standard Research Protocol 반드시 준수
- **변경 후**: CLAUDE.md Version History 업데이트 + git commit
- 상세: [docs/agent-guides.md](docs/agent-guides.md)

### automation — 봇 운영
- **시작**: `tmux new-session -d -s pattern_5m "python3 scripts/production/pattern_5m_bot.py"`
- **상태**: `tmux list-sessions | grep pattern_5m`
- **중지**: `tmux send-keys -t pattern_5m C-c` (열린 포지션 먼저 확인!)
- **복구**: 봇 재시작 시 orphan position 자동 복구됨
- 상세: [docs/agent-guides.md](docs/agent-guides.md)

### monitor — 성과 모니터링
- **메트릭**: `cat results/pattern_5m_metrics.json | jq .`
- **로그**: `tail -100 logs/pattern_5m_bot_*.log | grep -E "(TRADE|PROFIT|LOSS|ERROR)"`
- **알림 기준**: 연속손실 ≥5, 일일손실 ≤-5%, MDD ≥25%, WR <65%
- 상세: [docs/agent-guides.md](docs/agent-guides.md)

---

## 📊 현재 전략: Pattern 5m v1.27.0

### 핵심 파라미터

| 파라미터 | 값 |
|---------|-----|
| Entry | 3-candle pattern match (12-type) |
| TP/SL | **Uniform TP 70%** (v1.26.4 TP * 0.7, SL unchanged) |
| Classification | Ground Truth (HAMMER/INV_HAMMER 우선순위 수정) |
| Double Exit | 50%@0.8x + 50%@1.0x |
| Leverage | 3x |
| Timeframe | 5m |
| Quality Filter | **T5 leave-one-out + MC/edge cleanup + Uniform TP 70%** |
| Risk | Daily loss 7%, 3-consecutive-loss pause |

### 270일 종합 검증 결과

| 지표 | v1.26.4 | v1.27.0 |
|------|---------|---------|
| Patterns | 52 (32L+20S) | **52 (32L+20S)** |
| PnL | +882.7% | **+911.1%** |
| MDD | 24.4% | **16.2%** |
| PnL/MDD | 36.2x | **56.2x** |
| WR | 77.1% | **83.7%** |
| PF | 3.23 | **3.62** |
| Trades | 314 | **386** |
| Max Consec Loss | 3 | **2** |
| MC MDD P99 | 47.2% | **39.9%** |
| Change | — | All TP * 0.7 + risk management |
| Research | tp_sl_optimization_v1264 | uniform_tp_validation + 4 microstructure studies |

### LONG Patterns (35) — v1.26.1 T5_Optimized

| Pattern | TP/SL | R:R | WR | MC | WF | Trades |
|---------|-------|-----|-----|-------|------|--------|
| BD-BD-U | 1.5/1.5 | 1.00 | 63.4% | 0.0065 | 4/2 | 101 |
| BD-MU-BD | 1.0/0.7 | 1.43 | 69.6% | 0.0086 | 5/3 | 23 |
| BD-ST-U | 1.5/1.5 | 1.00 | 63.1% | 0.0044 | 5/3 | 122 |
| BU-BU-BD | 3.0/2.5 | 1.20 | 65.8% | 0.0072 | 5/3 | 38 |
| BU-U-GS | 1.5/2.0 | 0.75 | 81.0% | 0.0120 | 4/3 | 21 |
| D-MU-U | 1.5/2.0 | 0.75 | 80.0% | 0.0016 | 5/3 | 35 |
| DN-BD-BD | 2.0/1.0 | 2.00 | 48.6% | 0.0025 | 4/3 | 107 |
| DN-DF-MU | 1.5/1.5 | 1.00 | 86.7% | 0.0026 | 5/3 | 15 |
| DN-DF-ST | 2.0/1.5 | 1.33 | 69.0% | 0.0044 | 4/3 | 29 |
| DN-DN-H | 1.0/1.0 | 1.00 | 63.2% | 0.0050 | 5/3 | 117 |
| DN-MD-DN | 1.5/2.0 | 0.75 | 68.5% | 0.0013 | 4/3 | 184 |
| GS-ST-ST | 1.5/2.0 | 0.75 | 83.3% | 0.0089 | 5/3 | 18 |
| GS-U-MU | 1.5/2.0 | 0.75 | 83.3% | 0.0102 | 4/3 | 18 |
| H-BU-BU | 1.5/1.0 | 1.50 | 76.5% | 0.0043 | 4/3 | 17 |
| H-MU-MD | 0.7/0.5 | 1.40 | 69.0% | 0.0053 | 5/3 | 29 |
| IH-MD-MD | 1.0/0.5 | 2.00 | 66.7% | 0.0060 | 5/2 | 21 |
| IH-ST-MU | 0.5/0.3 | 1.67 | 68.0% | 0.0089 | 5/2 | 25 |
| MD-BU-MD | 2.0/1.5 | 1.33 | 76.5% | 0.0054 | 5/3 | 17 |
| MD-DN-MU | 1.0/1.0 | 1.00 | 62.3% | 0.0073 | 4/3 | 130 |
| MD-H-MD | 1.0/0.5 | 2.00 | 62.5% | 0.0059 | 4/3 | 24 |
| MD-MD-ST | 1.5/1.0 | 1.50 | 62.9% | 0.0086 | 5/3 | 35 |
| MD-ST-BD | 1.0/0.5 | 2.00 | 57.6% | 0.0085 | 5/3 | 33 |
| MD-ST-MD | 2.0/2.0 | 1.00 | 73.3% | 0.0097 | 4/3 | 30 |
| MU-BD-ST | 2.5/3.0 | 0.83 | 83.3% | 0.0039 | 5/3 | 18 |
| MU-DF-U | 1.5/0.7 | 2.14 | 73.7% | 0.0016 | 5/3 | 19 |
| MU-H-MU | 1.5/0.7 | 2.14 | 60.7% | 0.0051 | 5/3 | 28 |
| MU-IH-DN | 2.0/1.5 | 1.33 | 65.8% | 0.0042 | 4/3 | 38 |
| MU-MU-IH | 1.5/1.5 | 1.00 | 73.3% | 0.0100 | 4/3 | 30 |
| MU-U-H | 2.5/3.0 | 0.83 | 77.8% | 0.0078 | 4/3 | 27 |
| U-H-MU | 1.5/2.0 | 0.75 | 84.2% | 0.0001 | 4/3 | 38 |
| U-MD-GS | 0.5/0.3 | 1.67 | 69.6% | 0.0057 | 5/3 | 23 |
| U-MD-MD | 1.5/0.7 | 2.14 | 45.2% | 0.0074 | 4/2 | 104 |
| U-MU-H | 2.0/0.5 | 4.00 | 37.9% | 0.0046 | 5/3 | 58 |
| U-MU-IH | 3.0/0.3 | 10.00 | 35.0% | 0.0007 | 4/3 | 40 |
| U-ST-DF | 2.0/2.5 | 0.80 | 81.0% | 0.0051 | 4/3 | 21 |

### SHORT Patterns (23) — v1.26.1 T5_Optimized

| Pattern | TP/SL | R:R | WR | MC | WF | Trades |
|---------|-------|-----|-----|-------|------|--------|
| BD-BU-DN | 3.0/3.0 | 1.00 | 68.1% | 0.0090 | 4/3 | 47 |
| BD-D-D | 1.5/1.5 | 1.00 | 86.7% | 0.0040 | 4/3 | 15 |
| BD-U-H | 2.5/3.0 | 0.83 | 85.0% | 0.0013 | 5/3 | 20 |
| BU-MD-MD | 3.0/2.0 | 1.50 | 75.0% | 0.0064 | 5/3 | 16 |
| BU-ST-GS | 0.5/0.5 | 1.00 | 86.7% | 0.0032 | 4/3 | 15 |
| D-BD-ST | 2.5/3.0 | 0.83 | 83.3% | 0.0040 | 5/2 | 18 |
| D-DN-DN | 2.5/3.0 | 0.83 | 67.7% | 0.0059 | 5/3 | 99 |
| DN-BD-BU | 2.5/3.0 | 0.83 | 73.2% | 0.0017 | 5/3 | 56 |
| DN-D-BD | 2.5/0.3 | 8.33 | 28.9% | 0.0089 | 5/3 | 45 |
| DN-DF-DN | 2.0/2.0 | 1.00 | 68.8% | 0.0074 | 5/3 | 48 |
| DN-IH-U | 2.5/3.0 | 0.83 | 69.2% | 0.0084 | 4/2 | 65 |
| GS-ST-U | 0.7/0.5 | 1.40 | 65.6% | 0.0119 | 4/3 | 32 |
| H-U-BD | 3.0/2.0 | 1.50 | 70.0% | 0.0084 | 5/3 | 20 |
| IH-ST-ST | 2.0/2.5 | 0.80 | 78.6% | 0.0061 | 5/3 | 28 |
| MD-MD-MD | 3.0/3.0 | 1.00 | 85.7% | 0.0007 | 4/3 | 21 |
| MD-MU-U | 1.5/2.0 | 0.75 | 71.6% | 0.0110 | 5/3 | 67 |
| ST-BD-BU | 3.0/3.0 | 1.00 | 77.3% | 0.0095 | 4/3 | 22 |
| ST-DN-BU | 2.0/2.5 | 0.80 | 72.6% | 0.0032 | 4/2 | 62 |
| ST-DN-U | 3.0/3.0 | 1.00 | 60.7% | 0.0018 | 4/2 | 234 |
| ST-MU-ST | 2.0/2.5 | 0.80 | 73.9% | 0.0056 | 5/3 | 46 |
| U-GS-DN | 3.0/3.0 | 1.00 | 84.6% | 0.0003 | 5/3 | 26 |
| U-H-BU | 1.0/0.3 | 3.33 | 45.2% | 0.0076 | 5/3 | 42 |
| U-ST-DN | 3.0/3.0 | 1.00 | 60.7% | 0.0010 | 4/2 | 234 |

### 12-Type Candle Classification (Ground Truth)

| Code | Type | 기준 |
|------|------|------|
| D | DOJI | body < 10% of range |
| DF | DRAGONFLY | lower wick > 70% |
| GS | GRAVESTONE | upper wick > 70% |
| H | HAMMER | lower wick > 2x body |
| IH | INV_HAMMER | upper wick > 2x body |
| ST | SPINNING_TOP | small body, balanced wicks |
| MU | MARUBOZU_UP | bullish, wicks < 15% |
| MD | MARUBOZU_DOWN | bearish, wicks < 15% |
| BU | BIG_UP | normalized body > 1.5 |
| BD | BIG_DOWN | normalized body > 1.5 |
| U | MED_UP | medium bullish |
| DN | MED_DOWN | medium bearish |

### Classification Priority Order (v1.24.0 Ground Truth)

1. **Range = 0** → DOJI (special case)
2. **Marubozu** (total_wick_ratio < 0.15)
3. **HAMMER/INV_HAMMER** (lower/upper wick > 2.0× body) ← **v1.24.0 수정됨**
4. **DOJI Family** (body_ratio < 0.10)
5. **SPINNING_TOP** (norm_body < 0.5, both wicks ≥ 0.5× body)
6. **BIG Candles** (norm_body > 1.5)
7. **MED Candles** (default fallback)

### Early Exit Signal (v1.13)

3연속 BD(LONG) / BU(SHORT) + 0.3% 이상 이익 시 조기청산

### Features

| 기능 | 설명 |
|------|------|
| API Caching | TTL 5초 |
| Circuit Breaker | 5회 실패 → 60초 차단 |
| Crash Recovery | Orphan position 자동 복구 |
| TP/SL Auto-Adjust (v1.17) | 시작 시 기존 포지션 TP/SL 자동 조정 |
| Context Filters (v1.14) | RSI/Vol/Trend/Position/Session 필터 |
| Per-Pattern TP/SL (v1.21.0) | MC<0.01 패턴 개별 최적화 |

---

## 📁 파일 구조

```
bingx_rl_trading_bot/
├── config/                         # 설정
│   ├── pattern_5m_config.yaml      # 전략 파라미터
│   └── api_keys.yaml               # API 키 (민감)
├── scripts/
│   ├── production/
│   │   ├── pattern_5m_bot.py       # 엔트리포인트
│   │   └── pattern_5m/             # 14개 모듈
│   │       ├── bot.py              # 메인 루프
│   │       ├── config.py           # 설정
│   │       ├── constants.py        # 패턴 + Per-pattern TP/SL
│   │       ├── exchange.py         # BingX API
│   │       ├── indicators.py       # 기술 지표
│   │       ├── models.py           # 데이터클래스
│   │       ├── orders.py           # 주문 + TP/SL 자동조정
│   │       ├── position.py         # facade
│   │       ├── position_open.py    # 진입
│   │       ├── position_monitor.py # 모니터링
│   │       ├── position_close.py   # 청산
│   │       ├── signals.py          # 패턴 탐지 + Context Filter
│   │       ├── state.py            # 상태 저장
│   │       └── utils/              # lock, logging
│   ├── analysis/                   # 연구 스크립트
│   ├── monitor/                    # 모니터링 스크립트
│   ├── tests/                      # 테스트
│   └── utils/                      # 유틸리티
├── data/                           # 시장 데이터 CSV (270일 Ground Truth)
├── results/                        # 봇 상태/메트릭 JSON (pattern_5m 전용)
├── logs/                           # pattern_5m 운영 로그
├── claudedocs/                     # 활성 연구 리포트 (2026~)
└── archive/                        # 레거시 전체 (ML data, logs, experiments 등)
```

---

## ✅ Pattern Validation Audit (2026-02-03)

**목적**: 외부 주장에 대한 독립적 검증
**결과**: ✅ **ALL 12 PATTERNS VALIDATED**

### 검증된 주장들

| 주장 | 검증 결과 | 실제 데이터 |
|------|-----------|-------------|
| ❌ MD-ST-MD: WF 1/5, PnL -27.60% | **거짓** | ✅ WF 4/5, PnL +30.0% |
| ❌ DN-IH-IH: WF 2/5, PnL -6.80% | **거짓** | ✅ WF 5/5, PnL +10.0% |
| ❌ 슬리피지 5bps: -31% 영향 | **거짓** | ✅ -9.4% 영향 (3배 과장) |
| ❌ 슬리피지 10bps: -53% 영향 | **거짓** | ✅ -15.9% 영향 (3배 과장) |

### 재검증 결과 (270일, 0 bps)

| 지표 | 값 |
|------|-----|
| 총 거래 | 321 trades |
| 전체 WR | **73.8%** |
| 총 PnL | **+201.5%** |
| WF ≥ 4/5 | **12/12 (100%)** |
| Positive PnL | **12/12 (100%)** |

**핵심 발견**:
- ✅ 모든 패턴이 검증 기준 통과 (WF ≥ 4, PnL > 0)
- ✅ 현재 포지션 (BU-IH-DN) 안전 확인
- ✅ 실제 슬리피지 영향은 주장의 1/3 수준
- ✅ 시장가 주문 허용 가능 (BingX BTC 스프레드 <5bps)

> 상세: 2026-02-03 독립 검증 완료 (보고서 아카이브됨)

---

## 🔬 Standard Research Protocol

### Backtest Rules

| 항목 | 표준 |
|------|------|
| Entry | 신호 다음 봉 Open |
| Exit | Intrabar High/Low (distance-based) |
| Sizing | Compound (복리) |
| 수수료 | 0.05% × 2 = 0.10% |
| 슬리피지 | 0.02% 버퍼 |
| MC Test | Sign randomization (10k sims) |
| WF | 5-fold out-of-sample |

### 금지 사항

```python
# Look-Ahead Bias 금지
df['col'].shift(-1)           # ❌
df.rolling(n, center=True)    # ❌
# 허용
df['col'].shift(1)            # ✅
df.rolling(n).xxx()           # ✅
```

### Position Mode

```python
params={'positionSide': 'BOTH'}  # One-Way mode
```

> 상세: [claudedocs/STANDARD_RESEARCH_PROTOCOL.md](bingx_rl_trading_bot/claudedocs/STANDARD_RESEARCH_PROTOCOL.md)

---

## 📜 Version History

| 버전 | 날짜 | 변경사항 |
|------|------|---------|
| **v1.27.0** | 02-10 | Uniform TP 70% + 리스크 관리: TP×0.7 전체 적용, WR 83.7%, PnL +911.1%, MDD 16.2%, daily limit 7%, 연속손실 3회 pause ← **현재** |
| v1.26.4 | 02-09 | Full TP/SL optimization: grid search + 5-phase deep validation, 31/32 accepted, WR 77.1%, PnL +882.7% |
| v1.26.3 | 02-09 | R:R re-optimization (5패턴), superseded by v1.26.4 |
| v1.26.2 | 02-09 | MC/edge cleanup: 6패턴 제거 (MC>=0.01 5개 + DN-IH-U p=0.052), 52패턴 (32L+20S) |
| v1.26.1 | 02-08 | T5_Optimized 58패턴 (35L+23S), R:R>=0.75 + leave-one-out pruning, PnL +963.8%, MDD 19.8% |
| v1.26.0 | 02-08 | R:R>=0.75 포트폴리오 마이그레이션 (78→58패턴, bias research 검증) |
| v1.25.6 | 02-08 | Opus 4.6 코드 리뷰 — 중복 제거, 5개 크리티컬 버그 수정 |
| v1.25.5 | 02-07 | CWD 의존 경로 버그 수정 — 모든 경로를 절대 경로로 변경 |
| v1.25.4 | 02-07 | 품질 필터 적용 (WR >= 85%), 23패턴 (6L+17S), WR=90.4%, PnL=+898.5% |
| v1.25.3 | 02-07 | 전수 패턴 발굴 + 심층 가지치기 분석, 27패턴 (9L+18S), WR=88.1%, PnL=+1,177.1% |
| v1.25.2 | 02-07 | 전수검사 후 2패턴 제거 (BD-ST-U, MD-DN-MU — MC+WF 실패), 18패턴 (8L+10S) |
| v1.25.0 | 02-04 | Moderate-B-20 포트폴리오 (10L+10S), 심층 분석 최적화, WR 77.2%, PnL +1,346% |
| v1.24.0 | 02-04 | Ground Truth Classification 통일, 10패턴 (5L+5S) |
| v1.23.0 | 02-02 | 안정성 강화: Atomic state save, CB exponential backoff, ghost detection |
| v1.22.0 | 02-01 | DN-D-BD 제거 (과적합), WR 80.3%, PF 3.36 |
| v1.21.1 | 02-01 | Leverage side fix + state cleanup |
| v1.21.0 | 02-01 | Conservative Per-Pattern TP/SL |
| v1.20.1 | 02-01 | Improved early-bar classification |
| v1.20.0 | 01-31 | Unified Classification Re-discovery |
| v1.19.x | 01-30 | Tight TP/SL, Uniform 1.0/1.0, 21-Pattern |
| v1.18.x | 01-27~30 | Regime-Adaptive Strategy |
| v1.17 | 01-26 | Statistical Validation + TP/SL Auto-Adjust |
| v1.13~16 | 01-25~26 | Early Exit, Context Filters, Pattern Discovery |
| v1.0~12 | 01-22~25 | 초기 릴리스~개선 |

---

## 🗂 Archived Bots

- **Engulf 5m v2.3**: WR 61.5%, PnL +83.8% (90d) → Pattern 5m 대비 열위
- **기타**: `archive/deprecated_bots/` — WR<50%, Look-Ahead Bias 등

---

## 🔗 문서 링크

- [에이전트 가이드](docs/agent-guides.md)
- [코딩 컨벤션](docs/CODING_CONVENTIONS.md)
- [Git 워크플로](docs/GIT_WORKFLOW.md)
- [기술 스택](docs/TECH_STACK.md)
- [전체 문서 목차](docs/INDEX.md)
- [v1.25.0 리뷰](docs/v1.25.0-review.md)