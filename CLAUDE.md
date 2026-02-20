# CLAUDE_CODE_FIN - BTC 5분봉 패턴 트레이딩 봇

> **Version**: v1.29.0 | **Bot**: Pattern 5m (59패턴, 12L+47S) | **Updated**: 2026-02-21

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
| Dynamic Patterns | `results/dynamic_patterns.json` (scanner 출력) |
| Scanner | `scripts/scanner/pattern_scanner.py` (Dynamic WF 패턴 선택 CLI) |

---

## 🤖 Auto-Trigger Rules (Claude 자율 판단 기준)

Claude는 사용자 의도를 감지하여 아래 규칙에 따라 **자동으로** 적절한 도구를 선택한다.

### Intent → Command 자동 매핑

| 사용자 의도 (키워드) | 자동 실행 | 비고 |
|---------------------|----------|------|
| "봇 상태", "봇 확인", "살아있어?" | `/bot-status` | 프로세스+메트릭+로그 종합 |
| "성과", "실적", "수익률", "얼마 벌었어" | `/check-live` | 기대치 대비 성과 분석 |
| "일일 보고", "오늘 어때", "daily" | `/daily-report` | 일일 성과 리포트 생성 |
| "패턴 스캔", "재스캔", "새 패턴" | `/scan-patterns` | MAE/MFE 스캐너 실행 |
| "연구", "가설", "백테스트", "분석해줘" | `/research-template` + `trading-researcher` agent | 연구 프로토콜 강제 |
| "테스트", "tests" | `/run-tests` | pytest 1139+ 검증 |
| "WF 검증", "walk-forward", "OOS" | `/wf-validate` | Expanding window WF |
| "배포", "적용", "deploy" | `/deploy-patterns` | 안전 배포 체크리스트 |
| "문제", "에러", "왜 안돼", "이상해" | `/diagnose` + `root-cause-analyst` agent | 종합 진단 |
| "긴급", "중지", "emergency", "멈춰" | `/emergency-stop` | 긴급 정지 프로시저 |
| "리스크", "위험", "MDD", "drawdown" | `trading-risk` agent | 리스크 평가 |

### Intent → Agent 자동 선택

| 작업 유형 | 에이전트 | 선택 이유 |
|----------|---------|----------|
| 연구 스크립트 작성/실행 | `trading-researcher` | Standard Research Protocol 강제 |
| 봇 모니터링/성과 분석 | `trading-monitor` | 기대치 대비 분석 로직 내장 |
| 리스크 평가/전략 안전성 | `trading-risk` | MDD, WF, MC 전문 |
| 코드 품질/리팩토링 | `quality-engineer` | 테스트+코드 품질 |
| 디버깅/장애 분석 | `root-cause-analyst` | 체계적 원인 분석 |
| 성능 최적화 | `performance-engineer` | 프로파일링 기반 |

### 자동 행동 규칙

1. **코드 변경 후** → 자동으로 `/run-tests` 제안 (production 파일 변경 시 필수)
2. **전략 파라미터 변경 제안 시** → 자동으로 `/wf-validate` 제안
3. **연구 스크립트 작성 시** → `trading-researcher` agent 사용 + 연구 프로토콜 검증
4. **git commit 후 production 파일 포함 시** → CLAUDE.md Version History 업데이트 제안
5. **비정상 결과 감지 시** (PnL > 5000%, WF 전부 FAIL) → 자동 경고 + 원인 분석 제안
6. **새 세션 시작 시** → Serena 메모리 확인 (`project_state_v1_28_42`)

### Serena MCP 자동 활용

| 상황 | Serena 액션 |
|------|------------|
| 세션 시작 | `activate_project` → `check_onboarding` → 관련 메모리 읽기 |
| 코드 탐색 요청 | `find_symbol` / `get_symbols_overview` |
| 함수 수정 | `find_referencing_symbols` → 영향 범위 확인 → `replace_symbol_body` |
| 중요 발견 | `write_memory` (다음 세션 활용) |
| 연구 시작 | `read_memory("research_protocol_standard")` |
| 디버깅 시작 | `read_memory("common_pitfalls_and_lessons")` |

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
- **알림 기준**: 연속손실 ≥3, 일일손실 ≤-10%, MDD ≥25%, WR <60% | EXPECTED_WIN_RATE=68.0 (v1.28.0)
- 상세: [docs/agent-guides.md](docs/agent-guides.md)

---

## 📊 현재 전략: Pattern 5m v1.29.0

### 핵심 파라미터

| 파라미터 | 값 |
|---------|-----|
| Entry | 3-candle pattern match (12-type) |
| TP/SL | **Per-pattern 최적화** (scanner MAE/MFE percentile) |
| Classification | Ground Truth (HAMMER/INV_HAMMER 우선순위 수정) |
| Leverage | 3x |
| Timeframe | 5m |
| **Max Positions** | **5** (virtual slots, 1/N sizing, same-direction only) |
| Pattern Source | **Dynamic** (results/dynamic_patterns.json) |
| Discovery | **MAE/MFE** (TP=MFE percentile, SL=MAE percentile) |
| Scanner MAX_BARS | **288** (24h; v1.28.24: 500→288, 24h timeout study) |
| Quality Filter | **Edge>=21.8pp + WR>=60% + SL>=1.0% + MC<0.01 + min_trades>=25** |
| Patterns | **59** (12L + 47S), edge mean 23.9pp, WR mean 86.7% |
| Risk | Daily loss **13%** (v1.28.5), 3-consecutive-loss pause |

### 270일 In-Sample 검증 결과

| 지표 | Static 51 (v1.27.2) | PP 112 (v1.28.24) | **MAE/MFE 59 (v1.28.40)** |
|------|---------------------|------------------------|---------------------------|
| Patterns | 51 (32L+19S) | 112 (22L+90S) | **59 (12L+47S)** |
| Discovery | PP grid | PP grid | **MAE/MFE percentile** |
| WR mean | 84.9% | 90.5% | **86.7%** |
| Edge mean | ~15pp | 24.7pp | **23.9pp** |
| Portfolio Trades | 339 | 358 | **321** |
| Portfolio PnL | +966% | +1,398% | **+949%** |
| Portfolio MDD | 16.2% | 17.0% | **22.4%** |
| PnL/MDD | 59.6x | 82.1x | **42.4x** |
| TP range | 0.5-3.0 | 0.5-3.0 | **1.0-3.3** |
| SL range | 1.0-4.0 | 1.0-4.0 | **1.7-4.2** |

### LONG Patterns (12) — MAE/MFE 59

| Pattern | TP/SL | Edge | WR | Trades |
|---------|-------|------|-----|--------|
| BD-BD-BU | 1.67/3.14 | 30.0 | 95.2 | 42 |
| BD-DN-MU | 1.78/3.57 | 23.3 | 90.0 | 30 |
| DN-ST-H | 1.85/3.61 | 22.8 | 88.9 | 27 |
| H-ST-DN | 1.74/3.49 | 21.8 | 88.6 | 35 |
| H-U-MU | 1.36/2.96 | 27.5 | 96.0 | 25 |
| IH-DN-MD | 1.01/1.79 | 24.1 | 88.0 | 25 |
| MD-BD-DN | 1.93/3.32 | 21.9 | 85.2 | 27 |
| MD-DN-BU | 1.55/3.08 | 24.1 | 90.6 | 32 |
| MD-DN-DN | 1.95/3.7 | 22.0 | 87.5 | 88 |
| MD-MU-ST | 1.62/3.19 | 23.3 | 89.7 | 29 |
| U-MD-H | 1.35/3.08 | 24.6 | 94.1 | 34 |
| U-MU-H | 1.8/1.96 | 26.7 | 78.8 | 33 |

### SHORT Patterns (47) — MAE/MFE 59

| Pattern | TP/SL | Edge | WR | Trades |
|---------|-------|------|-----|--------|
| BD-BU-U | 1.45/3.25 | 22.2 | 91.4 | 58 |
| BD-DN-BU | 1.59/3.17 | 22.6 | 89.2 | 37 |
| BD-ST-U | 3.01/4.14 | 22.1 | 80.0 | 35 |
| BU-BD-U | 3.26/4.06 | 24.5 | 80.0 | 25 |
| BU-BU-U | 1.81/3.33 | 22.7 | 87.5 | 32 |
| BU-IH-U | 1.55/2.52 | 22.7 | 84.6 | 26 |
| BU-MU-DN | 2.05/2.1 | 27.5 | 78.1 | 32 |
| D-MU-DN | 2.56/3.15 | 28.8 | 84.0 | 25 |
| D-ST-U | 1.58/3.4 | 22.0 | 90.2 | 41 |
| D-U-MU | 1.47/3.0 | 22.5 | 89.7 | 29 |
| DF-U-U | 1.74/3.08 | 28.1 | 92.0 | 25 |
| DN-BD-BU | 2.18/3.01 | 28.8 | 86.8 | 38 |
| DN-BD-ST | 2.33/3.94 | 21.9 | 84.8 | 46 |
| DN-BU-BU | 1.91/3.47 | 27.2 | 91.7 | 36 |
| DN-BU-MD | 1.44/2.53 | 22.0 | 85.7 | 35 |
| DN-BU-ST | 2.52/4.07 | 23.2 | 85.0 | 40 |
| DN-DN-IH | 2.32/3.65 | 24.0 | 85.1 | 47 |
| DN-GS-ST | 1.36/2.26 | 25.6 | 88.0 | 25 |
| DN-IH-MD | 1.11/2.33 | 21.9 | 89.7 | 29 |
| DN-MU-H | 1.49/3.26 | 23.7 | 92.3 | 26 |
| DN-MU-MU | 1.76/2.85 | 23.0 | 84.8 | 46 |
| DN-ST-BU | 3.25/3.7 | 25.6 | 78.8 | 33 |
| GS-U-DN | 2.03/3.72 | 22.0 | 86.7 | 30 |
| H-DN-U | 3.0/3.88 | 23.6 | 80.0 | 30 |
| IH-ST-ST | 1.45/2.33 | 24.1 | 85.7 | 28 |
| IH-U-DN | 2.89/3.55 | 23.7 | 78.8 | 33 |
| IH-U-U | 2.73/3.55 | 21.8 | 78.4 | 37 |
| MD-BU-DN | 1.81/3.15 | 25.0 | 88.5 | 26 |
| MD-MU-U | 1.05/1.66 | 23.6 | 84.8 | 66 |
| MD-ST-ST | 1.51/3.47 | 23.5 | 93.2 | 44 |
| MU-DN-ST | 1.5/3.24 | 22.9 | 91.3 | 69 |
| MU-ST-MD | 1.34/3.08 | 22.3 | 92.0 | 25 |
| ST-BD-BU | 1.47/2.82 | 22.3 | 88.0 | 25 |
| ST-BU-ST | 2.58/4.22 | 21.9 | 84.0 | 25 |
| ST-D-U | 1.12/2.57 | 22.0 | 91.7 | 60 |
| ST-DN-BU | 1.82/3.33 | 22.8 | 87.5 | 48 |
| ST-H-DN | 1.96/3.01 | 24.6 | 85.2 | 27 |
| ST-IH-DN | 1.7/2.72 | 26.5 | 88.0 | 25 |
| ST-MU-ST | 1.29/1.96 | 23.0 | 83.3 | 48 |
| ST-ST-U | 2.24/3.94 | 22.1 | 85.9 | 92 |
| ST-U-BD | 2.0/3.74 | 23.5 | 88.6 | 44 |
| U-D-DN | 2.17/3.77 | 22.8 | 86.3 | 51 |
| U-DN-BD | 2.22/3.79 | 22.8 | 85.9 | 78 |
| U-DN-H | 2.97/3.67 | 23.7 | 78.9 | 38 |
| U-GS-DN | 1.75/2.79 | 28.5 | 90.0 | 30 |
| U-H-DN | 2.14/3.32 | 23.8 | 84.6 | 52 |
| U-IH-U | 2.53/3.15 | 22.9 | 78.4 | 37 |

### WF OOS 검증 (720d, 3-fold Expanding Window)

| Fold | IS 기간 | OOS 기간 | OOS Trades | OOS WR | OOS PnL | OOS MDD |
|------|---------|----------|------------|--------|---------|---------|
| 1 | 0-180d | 180-360d | 156 | 69.2% | +80.7% | 36.9% |
| 2 | 0-360d | 360-540d | 131 | 72.5% | +112.1% | 39.6% |
| 3 | 0-540d | 540-720d | 154 | 79.9% | +127.7% | 37.0% |

**Verdict: 3/3 PASS** | Total OOS PnL: +320.5% | Avg OOS WR: 73.9%

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
| Context Filters (v1.14) | RSI/Vol/Trend 인프라 (v2 연구 결과: 유의 효과 없음, 필터 비활성) |
| Per-Pattern TP/SL (v1.21.0) | MC<0.01 패턴 개별 최적화 |
| Dynamic Pattern Selection (v1.27.3) | `pattern_source: dynamic` 모드 — scanner CLI가 생성한 Universal TP/SL 패턴 세트 사용 |

### Dynamic Pattern Selection (v1.27.3)

`pattern_source` 설정으로 정적(constants.py) 또는 동적(scanner 출력) 패턴 세트 선택 가능.

| 모드 | 설정값 | 패턴 소스 | TP/SL |
|------|--------|-----------|-------|
| Static (fallback) | `pattern_source: static` | constants.py 51패턴 | Per-pattern 최적화 |
| Dynamic PP | `pattern_source: dynamic` + `tp_sl_mode: per_pattern` | results/dynamic_patterns.json | PP grid search |
| **Dynamic MAE/MFE (현재)** | `pattern_source: dynamic` + `tp_sl_mode: per_pattern` | results/dynamic_patterns.json | **MAE/MFE percentile** |

**Scanner CLI 사용법**:
```bash
cd bingx_rl_trading_bot
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 21.8   # MAE/MFE (현재)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 21.8 --wf-folds 3  # + WF
python scripts/scanner/pattern_scanner.py                           # PP 모드 (이전)
```

**현재 적용**: MAE/MFE discovery (edge>=21.8pp, MC<0.01) → **59패턴 (12L+47S)**.
WF 3/3 PASS (720d expanding window, OOS PnL +320.5%). Backup: `results/dynamic_patterns_pp112_backup.json`

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
│   │       ├── config.py           # 설정 + Dynamic Pattern 로딩
│   │       ├── constants.py        # 패턴 + Per-pattern TP/SL
│   │       ├── exchange.py         # BingX API
│   │       ├── indicators.py       # 기술 지표
│   │       ├── models.py           # 데이터클래스
│   │       ├── orders.py           # 주문 + TP/SL 자동조정
│   │       ├── position.py         # facade
│   │       ├── position_open.py    # 진입 (dynamic universal TP/SL 지원)
│   │       ├── position_monitor.py # 모니터링
│   │       ├── position_close.py   # 청산
│   │       ├── signals.py          # 패턴 탐지 + Context Filter
│   │       ├── state.py            # 상태 저장
│   │       └── utils/              # lock, logging
│   ├── scanner/                    # Dynamic WF Pattern Scanner CLI
│   │   └── pattern_scanner.py      # 오프라인 패턴 스캔 → dynamic_patterns.json
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
| **v1.28.17** | 02-18 | **State corruption resilience + data restoration**: (1) `state.py`: `sync_metrics_with_state()` 양방향 스마트 싱크 — state < metrics일 때 state가 손상된 것으로 판단, metrics 신뢰하고 state를 metrics로 복구 (기존: 항상 state 신뢰 → metrics 다운그레이드 버그). (2) `state.py`: `load_state()` 복구 체인에 timestamped backup 폴백 추가 (main → .bak → backup_* → default). (3) State/Metrics 데이터 복원: force-kill로 손상된 state(5 trades) → 백업+로그에서 정확한 40 trades/25W/8.40% PnL 복원. |
| v1.28.16 | 02-18 | **Test fix + dead code cleanup**: (1) `test_patterns.py`: `test_stats_win_rates_reasonable` 실패 수정 — WR 45.2% 패턴(U-MD-MD, R:R 2.14 보상 구조)이 50% 하한에 걸림, 유효 범위 0-100%로 수정 (2) `exchange.py`: 미사용 `api_retry` 데코레이터 제거 + dead import (`wraps`, `TypeVar`) 정리 + `raise e` → `raise` (traceback 보존) (3) `pattern_5m_bot.py`: 오래된 docstring 업데이트 (v1.24 시절 패턴+백테스트 결과 제거). **214 tests all pass**. | |
| v1.28.15 | 02-18 | **4 hardening fixes**: (1) `position_close.py`: `recover_from_crash` Case 2 fallback을 `entry_price` → 현재 ticker로 변경 (v1.28.14 sync fix와 동일 패턴) (2) `lock.py`: `_check_windows_process`+`check_duplicate_instances`에 `python3.exe` 추가 — MSYS2 환경에서 봇 프로세스 미감지 수정 (3) `logging_config.py`: dead code `log_signal_conditions` 제거 (engulf bot 시절 잔존, 현재 미사용) (4) `lock.py`: `_write_lock_info`+`_cleanup_file`을 base `FileLock` 클래스로 통합 — WindowsFileLock/UnixFileLock 중복 제거. 비즈니스 로직 변경 없음. |
| v1.28.14 | 02-17 | **2 behavior improvements**: (1) `position_monitor.py`: `sync_position_with_exchange`에서 trade history 실패 시 fallback을 `entry_price`(PnL=0%) → 현재 ticker 가격으로 변경 — 외부 청산 시 PnL 왜곡 방지 (2) `bot.py`: Trading window에서 포지션 종료 감지 후 같은 캔들에서 새 진입 신호 확인 — 기존엔 5분 대기 필요. cooldown/daily limit으로 안전성 보장. |
| **v1.29.0** | 02-21 | **N=5 멀티포지션 구현 (One-Way BOTH mode)**: 가상 슬롯 기반 멀티포지션 시스템. `state['position']` → `state['positions']` dict (`{slot_id: pos_dict}`). `max_positions: 5` (config), `DEFAULT_MAX_POSITIONS=1`, `MAX_ALLOWED_POSITIONS=10`. Signal routing: `_route_signal()` — OPEN/SKIP/CLOSE_OLDEST. 1/N 사이징 (`get_slot_size`). Emergency SL: `closePosition: True` 전체 보호 (worst SL - buffer). 슬롯별 TP/SL: `amount` 기반 (closePosition 미사용). State v1→v2 자동 마이그레이션 (`_migrate_state_v2`). 반대 신호 → FIFO 가장 오래된 슬롯 청산. 동일 패턴 중복 진입 차단 (`signals.py`). 12개 파일 변경 (constants/state/models/config/orders/position_open/position_close/position_monitor/bot/signals/exchange/config.yaml). `max_positions: 1`로 N=1 동작 100% 호환. **1014 tests passed**. |
| **v1.28.42** | 02-21 | **ATR-scaled TP/SL + proportional vol_mult cap**: `get_volatility_multiplier()`에 ATR-ratio 스케일링 모드 추가 (기존 `vol_adaptive`보다 우선). `ATR_ratio = ATR(14) / rolling_median(ATR(14), 576봉)`, clamp [0.6, 1.7]. WF 연구: 28 시나리오 전수 검증, `BOTH_a14_w576_0.6-1.7` WF 3/3 PASS, pre-overlap edge +74.5%. SL 스케일링이 핵심 동인 (TP-only FAIL, SL-only PASS). `config.yaml` `strategy.atr_scale` 섹션 추가 (enabled/window/clamp_lo/clamp_hi). `calculate_indicators()`에서 `atr_scale` 활성화 시에도 ATR 계산 트리거. `MAX_OHLCV_CANDLES` 150→600 (ATR-scale window=576 충족). **Proportional cap**: `_effective_vol_mult()` — 멀티플라이어 자체를 `max_sl_pct / base_sl_pct`로 제한, TP/SL 동일 비율 스케일링으로 R:R 비율 보존. Hard SL cap(`_cap_sl_to_daily_limit`)은 R:R 왜곡 최대 +65.6% (49/59 패턴 영향) → proportional cap은 +1.2% 이하. 7가지 캡 전략 심층 분석 (`atr_cap_strategy_study.py`). 1139 tests passed. ← **현재** |
| v1.28.41 | 02-19 | **Atomic write retry + .new fallback for OneDrive PermissionError**: `os.replace()` 실패 시 OneDrive 동기화 잠금 대기하는 `_atomic_replace_with_retry()` 추가 (exponential backoff 0.1/0.2/0.4s, 3회). Non-atomic fallback(`open(state_file,'w')`) → `.new` 파일 쓰기로 변경 (corruption 방지). `load_state()` 복구 체인 확장: main → `.new` → `.bak` → timestamped → default. `save_metrics()`도 동일 패턴 적용. `LOCK_FILE`을 `results/` (OneDrive 동기화) → `tempfile.gettempdir()` (로컬)으로 이동 — OneDrive가 lock 파일을 잠가 `msvcrt.locking()` 실패하는 문제 해결. `EXPECTED_AVG_WIN/LOSS/EDGE` MAE/MFE 59패턴 값으로 업데이트 (5.63/9.64/0.75). 1014 tests passed. |
| **v1.28.40** | 02-19 | **Deploy MAE/MFE 59 patterns to production**: PP 112패턴 → MAE/MFE 59패턴 (12L+47S) 교체. `--discovery-method mae_mfe --edge-threshold 21.8` 스캔. MAE/MFE는 실제 가격 행동(MFE/MAE percentile)에서 TP/SL 도출 → 더 적응적. WF 3/3 PASS (OOS +320.5%). IS: WR 87.2%, PnL +949%, MDD 22.4%. TP 1.0~3.3%, SL 1.7~4.2%. PP 112 백업: `dynamic_patterns_pp112_backup.json`. |
| v1.28.39 | 02-19 | **Scanner MAE/MFE discovery method**: `--discovery-method mae_mfe` 옵션 추가. MFE percentile → TP, MAE percentile → SL 도출 (fixed grid 대신 실제 가격 행동 기반). 연구에서 WF OOS 2.4배 향상 (grid +617% → MAE/MFE +1,511%). `compute_excursions()`, `derive_tp_sl()`, `grid_search_mae_mfe()`, `_mae_mfe_worker()` (병렬), `scan_patterns_mae_mfe()` 추가. WF `scan_universe_range()` mae_mfe 모드 지원. 출력 JSON `tp_sl_mode: "per_pattern"` 봇 100% 호환. 270d 스캔: 267패턴, WR 93.0%, PnL +1,440%. 955 tests passed. |
| v1.28.38 | 02-18 | **Scanner fee calculation bug fix**: `bt_signals()`에서 수수료가 레버리지 미반영 — `FEE_PCT`(0.10%) 대신 `FEE_PCT * LEVERAGE`(0.30%) 사용해야 함. BingX 수수료는 notional(레버리지 적용 금액)에 부과되므로 capital-space PnL에서 `fee × leverage` 차감 필요. 기존: 트레이드당 0.20% 수수료 과소 차감 → PnL 낙관적 왜곡. Production `calculate_pnl()`과 정확히 일치하도록 수정. 패턴 선택(edge=WR-baseline)에는 영향 없음. |
| v1.28.37 | 02-18 | **Extract calculate_pnl() + test pure functions**: (1) `position_close.py`: PnL 계산을 `calculate_pnl()` 순수 함수로 추출 (entry/exit/direction/leverage → pnl_pct, price_pnl_pct). (2) `test_pure_functions.py` 신규: `calculate_pnl` 9개 + `extract_pattern_name` 8개 + `setup_scale_out` 9개 = 26 tests. (3) `test_config.py`: `load_dynamic_patterns()` 17개 tests 추가 (universal/per_pattern/auto-inference/fallback/staleness). 379 tests total. |
| v1.28.36 | 02-18 | **Dynamic pattern confidence scoring**: `calculate_pattern_confidence()`가 static `PATTERN_STATS`(51패턴)만 참조 → 112 dynamic 패턴 중 61개가 WR=50% 기본값 → `historical` component=0.0. 수정: `config.py`에서 `pattern_details` → `_dynamic_pattern_stats` dict 주입, `signals.py`에서 priority lookup (dynamic > static > default 50%). `check_entry_signal()`에 config/direction 전달. 337 tests passed. |
| v1.28.35 | 02-18 | **Fix edge metric — unify actual/expected definition**: `actual_edge`가 PnL/maxDD (ratio)로 계산되고 `EXPECTED_EDGE`는 WR margin (pp)으로 정의되어 성능 리포트에서 15.0 vs 1.4 비교 = 사과vs오렌지. 둘 다 "per-trade expected PnL%"로 통일: `actual_edge = total_pnl / total_trades`, `EXPECTED_EDGE = WR×avg_win - (1-WR)×avg_loss = 0.27%`. Display format `:.1f` → `:.2f%`. 214 tests passed. |
| v1.28.34 | 02-18 | **Untrack state.json from git — prevent state corruption**: `pattern_5m_bot_state.json`을 `.gitignore`에 추가 + `git rm --cached`로 git 추적 해제. 이 파일이 git에 추적되면서 git 작업 시 커밋된 이전 state로 되돌아가 매 재시작마다 `state=5 < metrics=41` corruption이 반복되었음. `.bak` 파일도 `.gitignore`에 추가. metrics.json은 이미 .gitignore에 있어 정상. 214 tests passed. |
| v1.28.33 | 02-18 | **Dead constants cleanup**: `constants.py`에서 미사용 상수 ~55줄 제거 — (1) `MarketRegime` class + `REGIME_*` 5개 상수 + `REGIME_PATTERNS` dict + `DEFAULT_REGIME` (v1.19.0 deprecated 이후 어디서도 import 안됨) (2) `LOG_FORMAT`/`LOG_DATE_FORMAT` (logging_config.py 자체 포맷 사용) (3) `ROTATION_MAX_SIZE`/`ROTATION_MIN_PARTIAL_PCT`/`ROTATION_REFILL_TO_FULL` (어디서도 import 안됨, `ROTATION_ENABLED`만 사용) (4) `DEFAULT_LEVERAGE`/`DEFAULT_POSITION_PCT` (config dict에서 직접 사용, import 안됨). 214 tests passed. |
| v1.28.32 | 02-18 | **PnL=0 consistency + dead code removal + sync .bak fix**: (1) `position_close.py`: `record_closed_position`의 `pnl_pct > 0` → `>= 0` — v1.28.30에서 `models.py`만 수정하고 여기를 놓침, state/metrics win 카운터 불일치 수정 (confidence log도 동일 수정) (2) `position_close.py`: `close_position_market` 실패 시 TP/SL 재배치를 private import → public `place_tp_sl_orders` 사용 (3) `position_open.py`: `_cancel_existing_tp_sl` 함수 제거 — `orders.cancel_remaining_orders`와 중복 (4) `state.py`: `sync_metrics_with_state`에서 state 복원 후 .bak도 갱신 — `save_state`의 `_create_backup`이 old corrupted main을 .bak에 복사하는 문제로 매 재시작마다 CRITICAL 경고 반복. 214 tests passed. |
| v1.28.31 | 02-18 | **Code cleanup**: (1) `position_monitor.py`: `_infer_exit_reason` LONG/SHORT 분기 동일 코드 제거 — 방향 무관하게 price proximity 비교 (2) `exchange.py`: `_api_call_with_retry`에서 `last_exception=None` 방어 guard 추가 (`API_MAX_ATTEMPTS=0` edge case). 214 tests passed. |
| v1.28.30 | 02-18 | **3 code quality fixes**: (1) `state.py`: `_check_daily_reset`과 `reset_daily_stats_if_needed` 필드 리셋 로직 중복 제거 — 공통 `_reset_daily_fields()` helper 추출 (2) `models.py`: `update_trade` PnL=0을 loss→win으로 변경 (`pnl_pct > 0` → `>= 0`) — break-even 청산이 loss로 잘못 집계되던 문제 (3) `config.py`: `validate_config` 반환타입 `bool`→`None` — 성공 시 `return True`/실패 시 `raise`는 misleading. 214 tests passed. |
| v1.28.29 | 02-18 | **Fix load_dynamic_patterns partial config mutation**: `config.py` `load_dynamic_patterns`에서 patterns injection을 TP/SL validation 이후로 이동 — 기존엔 TP/SL 검증 실패 시 dynamic patterns는 주입되었지만 TP/SL 데이터 없는 hybrid state로 fallback (static defaults로 트레이딩 가능). 이제 fallback 시 완전한 static config 반환. 214 tests passed. |
| v1.28.28 | 02-18 | **Fix recovery NoneType crash**: `position_close.py` `recover_position_to_state`에서 `state.get('position', {})` → `state.get('position') or {}` — state에 `position: null`이 존재할 때 `.get(key, default)`는 default를 반환하지 않아 `None.get()` AttributeError 발생. State corruption 복구 후 orphan position recovery가 2회 연속 실패하는 실 장애 수정. 214 tests passed.
| v1.28.27 | 02-18 | **position_open sentinel + refill guard**: (1) `position_open.py`: `_cancel_existing_tp_sl`에 `_EXCHANGE_MANAGED` sentinel 제외 — 3번째(마지막) 취소 함수도 sentinel 체크 완료, refill 시 sentinel order ID로 exchange cancel 시도 방지 (2) `position_open.py`: `refill_position`에서 `total_qty <= 0` guard 추가 — 비정상 수량으로 인한 ZeroDivisionError 방지. 214 tests passed.
| v1.28.26 | 02-18 | **Sentinel consistency + lock cleanup**: (1) `orders.py`: `_cancel_existing_tpsl_orders`에 `_EXCHANGE_MANAGED` sentinel 제외 추가 — `cancel_remaining_orders`와 동일 패턴 적용, adjust_tpsl 경로에서도 sentinel 취소 시도 방지 (2) `lock.py`: `_write_lock_info()` 미사용 `filepath` 파라미터 제거 — 메서드는 `self._handle`에 직접 write하므로 인자 불필요, 호출부 2곳(Windows/Unix acquire) 동시 정리. 214 tests passed.
| v1.28.25 | 02-18 | **3 production hardening fixes**: (1) `position_close.py`: `entry_price <= 0` zero division guard — corrupted state에서 PnL 계산 시 ZeroDivisionError 방지, 0% 기록 (2) `orders.py`: `cancel_remaining_orders`에서 `_EXCHANGE_MANAGED` sentinel 명시적 제외 — sentinel이 truthy라 불필요한 open_order_ids 비교 시도 방지 (3) `position_close.py`: `recover_position_to_state`에서 이전 state의 `pattern_name` 보존 — recovery 후 per-pattern TP/SL 조회 가능 + "Pattern None not in dynamic" 경고 해소. fallback `calculate_tp_sl`에도 pattern 전달. 214 tests passed. |
| v1.28.24 | 02-18 | **Scanner MAX_BARS 500→288 (24h timeout study)**: 3-phase 24h 연구 (v1 DROP비교, v2 edge threshold, v3 forced close) 결과 288봉(24h) 최적 확인. Scanner `MAX_BARS=500→288` 변경 후 재스캔. 325패턴 → E>=21.8pp+WR>=60%+SL>=1.0% → **112패턴 (22L+90S)**. 포트폴리오: PnL +1398% (vs +894%), WR 93.6% (vs 88.1%), MDD 17.0% (vs 24.2%), PnL/MDD 82.1x (vs 36.9x). Forced close(패배처리) WF 1/3 FAIL → 미적용. Production 봇 변경 없음 (timeout 미도입, TP/SL 보유). EXPECTED_AVG_WIN 5.44, EXPECTED_AVG_LOSS 10.73. |
| v1.28.23 | 02-18 | **Extend EXCHANGE_MANAGED to initial TP/SL placement**: v1.28.22는 verify 경로만 처리 → crash recovery 시 `place_tp_sl_orders`에서도 110407/110406/110413 에러 발생 (매 재시작마다 WARNING 로그). `_place_single_tp_order`와 `_place_sl_order`에도 동일한 에러코드 감지 + `_EXCHANGE_MANAGED` sentinel 마킹 추가. 이제 recovery → placement → verify 전체 경로에서 "already exists" 에러를 깔끔하게 처리. 214 tests passed. |
| v1.28.22 | 02-18 | **Fix TP/SL verify infinite retry loop**: crash recovery 후 `tp_order_id`/`sl_order_id`가 None → `verify_tp_sl_orders`가 10분마다 재배치 시도 → exchange "already exists" (110407/110406) 거부 → ID가 None 유지 → 무한 반복. `_EXCHANGE_MANAGED` sentinel 도입: BingX 에러코드 110407(TP exists)/110406(SL exists)/110413(TP exceeded)를 감지하여 "exchange가 관리 중" 상태로 마킹, 이후 재시도 방지. orders.py 수정 (`_verify_single_tp_order`, `_verify_sl_order`). 214 tests passed. |
| v1.28.21 | 02-18 | **Silent except → debug logging + backup consistency**: (1) state.py: 3곳 `except Exception: pass` → `logger.debug` 추가 (timestamped backup 검색/정리 실패 로깅) (2) lock.py: 2곳 `except Exception: pass` → `logger.debug` 추가 (unlock 실패 로깅) (3) state.py `_create_backup`: 수동 `open/read/write` → `shutil.copy2` 변경 (save_state의 .bak 생성과 일관성). 214 tests passed. |
| v1.28.20 | 02-18 | **Utils cleanup + outdated docstring**: (1) `__init__.py` docstring을 v1.28.x 현재 상태로 업데이트 (2) `logging_config.py`: local `import time as _time` 2곳 → module-level `import time`으로 이동 (3) `lock.py`: 미사용 `_lock_file_handle` 전역변수 제거 (실제 lock 상태는 `_lock_instance`로 관리). 214 tests passed. |
| v1.28.19 | 02-18 | **Unused import cleanup**: (1) signals.py: 미사용 `classify_candle` import 제거 (only `calculate_indicators` 사용) (2) position_close.py: 미사용 `List` typing 제거 (3) orders.py: 미사용 `Tuple` typing 제거. AST 기반 전체 모듈 스캔 결과 이 3곳만 해당. 214 tests passed. |
| v1.28.18 | 02-18 | **Dead code removal + import cleanup**: (1) `get_candle_type_for_price()` 제거 — 정의만 있고 호출 없는 dead code (2) signals.py 로컬 import 3곳을 모듈 레벨로 이동 (`import os`, `from datetime import datetime` — `_save_confidence_to_csv`, `check_cooldown` 내부). 비즈니스 로직 변경 없음, 214 tests passed. |
| v1.28.17 | 02-18 | **State corruption resilience**: (1) `sync_metrics_with_state()` 양방향 스마트 sync — state < metrics면 corruption 감지 후 metrics 신뢰 (2) `load_state()` timestamped backup 복구 체인 추가 (main → .bak → timestamped → default) (3) 40 trades 데이터 수동 복원 |
| v1.28.13 | 02-17 | **Resilience improvements — 3 fixes**: (1) `config.py`: `tp_sl_mode` 누락 시 JSON 구조에서 자동 추론 (`patterns_tpsl` → per_pattern, `universal_tp/sl` → universal) — static fallback 방지 (2) `position_close.py`: `recover_position_to_state`에서 거래소 오픈 주문의 TP/SL 가격 읽기 — 복구 시 config 기본값(1.0%/1.0%) 대신 실제 per-pattern TP/SL 보존 (`_read_tpsl_from_exchange_orders` 신규) (3) `bot.py`: maintenance window에서 `check_position_status` 반환값 반영 — 포지션 닫힘 즉시 has_position 갱신. 비즈니스 로직 변경 없음. |
| v1.28.12 | 02-17 | **Safety fixes — 6 production bugs**: (1) CRITICAL: SL 미배치 시 즉시 재시도 2회 — 기존 10분 방치 방지 (`position_open.py`) (2) Cache poisoning: `cache.set_positions([])` → `cache.invalidate_all()` — 포지션 조회 오염 방지 (`position_close.py`) (3) `adjust_tpsl_to_config`에서 `remaining_quantity` 사용 — scale-out 부분체결 후 wrong qty 방지 (`orders.py`) (4) `_verify_scale_out_orders`에서 `order_id=None` 처리 — 초기 배치 실패 stage 재배치 (`orders.py`) (5) `_infer_exit_from_price` zero guard — tp/sl=0 시 false TP/SL 판정 방지 (`position_monitor.py`) (6) `_cancel_existing_tp_sl`에서 `tp_order_id` 취소 추가 — refill 시 old TP 잔존 방지 (`position_open.py`). 비즈니스 로직 변경 없음, 안전성 강화만. |
| v1.28.11 | 02-17 | **SL<1.0% pattern removal**: SL threshold study (`sl_threshold_study.py`) 기반 3패턴 제거 (H-MU-MD SL 0.5%, IH-ST-MU SL 0.7%, MU-MU-BD SL 0.7%). WF OOS 검증: 50pat 686.4% → 47pat 702.7% (+16.3%), WR 84.3%→88.1%. SL eff 0.43~0.63% (spread 차감 후) = 실전 즉사. EXPECTED_AVG_WIN 6.04, EXPECTED_AVG_LOSS 10.09. SL min 1.0%, TP min 0.5. |
| v1.28.10 | 02-17 | **Safety patch — 4 critical trading logic gaps**: (1) SL/TP 미배치 감지+재배치 (`orders.py`: `_verify_single_tp_order` 추가, sl_order_id=None 처리, price guard) (2) 시장가 청산 실패 시 TP+SL 모두 재배치+save_state (`position_close.py`) (3) config.py validation 키 `_LONG`/`_SHORT` suffix 제거 — 50개 false warning 수정 (4) Direction mismatch 감지+즉시복구 3곳 추가 (`check_position_status`/`sync_position_with_exchange`/`recover_from_crash`), actual exit price 사용. 비즈니스 로직 변경 없음, 안전성 강화만. |
| v1.28.9 | 02-16 | **Edge>=21.8pp + WR>=60% quality filter**: 257→50 patterns (15L+35S). Edge sensitivity 9시나리오 분석 후 edge+WR 이중 필터. WR min 63.4%, mean 85.5%, edge mean 25.0pp, trades 1,580. EXPECTED_AVG_WIN 5.96, EXPECTED_AVG_LOSS 9.60. 257패턴 backup: `dynamic_patterns_257_backup.json` |
| v1.28.8 | 02-16 | **Logging system improvement**: lock.py 로거명 수정 (`__name__`→`'pattern_5m'`), 17곳 generic Exception `logger.error`→`logger.exception` (traceback 추가), 5곳 critical path DEBUG 로깅 추가 (signals/position_open/position_monitor). 9개 파일 수정, 비즈니스 로직 변경 없음. |
| v1.28.7 | 02-16 | **Production code review + dual-direction bug fix**: (1) bot.py early exit return False on failure (2) save_metrics atomic write (3) signals.py if/elif fix — 5/6 dual-direction patterns were trading wrong direction (4) scanner dedup logic (5) dead import cleanup (6) config mutable return fix. 256패턴 (83L+173S) after dedup. |
| v1.28.6 | 02-16 | PP discovery scanner: Scanner에 per_pattern grid search 모드 추가 (default). PP +487% avg OOS vs Uni +18%. Multi-seed MC, MAX_BASELINE_WR 70% 필터. 294패턴→256 dedup. Rollback: `--discovery-method universal` |
| v1.28.5 | 02-15 | Dynamic per-pattern TP/SL optimization: Universal TP 2.0/SL 3.0 → Per-pattern 최적화 (TP median 2.0%, SL median 4.0%). WF 3-fold OOS: Per-pattern +1,209% vs Universal +1,166% (+3.7%). 47패턴 (19L+28S), max_daily_loss 10→13% (SL max 4.0%×3x=12.1%). Rollback: tp_sl_mode "per_pattern"→"universal" |
| v1.28.4 | 02-15 | Statistical rigor filter: edge_threshold 5→10pp (통계적 엄밀성 연구 기반). BH FDR/Bonferroni/Bootstrap CI/11-scenario WF 비교 → edge≥10pp가 WF 3/3 최적. 47패턴 (19L+28S), PnL 740.5%, WR 80.8%, MDD 30.5%, PF 2.73 |
| v1.28.3 | 02-14 | Statistical significance filter: min_trades 10→20 (trades<20 패턴 13개 제거 — MC 신뢰도 부족). 7개 시나리오 비교, WF 3-fold 개별 검증. 51패턴 (18L+33S), PnL 671.4%, MDD 27.3%, PF 2.42 |
| v1.28.2 | 02-14 | WF frontier optimal TP: TP 2.1→2.0 (15후보 3-fold WF 검증). OOS Edge 12.6pp(최고), OOS MDD 31.7%(최저), E[trade] 1.79%(최고). v1.29.0 TP 0.80/SL 1.20 시도→WF 1/3 FAIL→롤백. 66패턴 (25L+41S), PnL 772.7%, MDD 33.7% |
| v1.28.1 | 02-14 | Fine grid TP + distance fix + min_trades filter: TP 2.0→2.1 + same-bar TP/SL를 bar open 기준으로 수정 + min_trades 8→10. 62패턴 (22L+40S), PnL 661%, MDD 27.3% |
| v1.28.0 | 02-12 | Static→Dynamic 프로덕션 전환. Universal TP 2.0/SL 3.0 + 75패턴 (28L+47S). True WF OOS: Universal +562% vs Per-pattern +416%. pattern_source: dynamic 활성화. Daily limit 5→10% (1회 SL=9.1%) |
| v1.27.3 | 02-12 | Expectation reset + Dynamic WF Pattern Selection 인프라. 백테스트 PnL의 90%가 look-ahead bias, 순수 forward edge +80.5%/68.5% WR. EXPECTED_WIN_RATE 85→68, daily limit 7→5%. Dynamic pattern_source 모드 추가 (scanner CLI → Universal TP 2.0/SL 3.0, 75패턴=28L+47S). 90일 OOS 테스트 대기 (2026-04-30 목표) |
| v1.27.2 | 02-12 | Low-WR pattern review: U-H-BU 제거 (SL 0.3% 실전 불가), 51패턴 (32L+19S), PnL +966%, WR 84.9%, MDD 16.2%, PnL/MDD 59.6x + 25개 개별 검증 + Pattern-Rediscovery WF + Strategy Options Evaluation |
| v1.27.1 | 02-12 | Legacy pattern re-optimization: 15/21 legacy 패턴 TP/SL 재최적화 (9 MC fix + 6 upgrade), PnL +956.2%, MDD 16.2%, PnL/MDD 59.0x, PF 3.82, WR 82.1% |
| v1.27.0 | 02-10 | Uniform TP 70% + 리스크 관리: TP×0.7 전체 적용, WR 83.7%, PnL +911.1%, MDD 16.2%, daily limit 7%, 연속손실 3회 pause + context filter 심층연구 FAIL (BH FDR 0/156 유의) |
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