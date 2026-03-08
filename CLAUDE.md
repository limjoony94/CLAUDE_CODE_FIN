# CLAUDE_CODE_FIN - BTC 5분봉 패턴 트레이딩 봇

> **Version**: v1.55.0 | **Bot**: Pattern 5m (131패턴, 59L+72S, Edge18pp+NeutralWindow+ATR Scanner+Holdout+MDD+Cap7+MomGuard1.5%15m1h+NposScanner+CascadeSL85+AggRisk8_15+ATRClamp05_15+TO288+ScannerCascade+MassCloseGuard+ExitClassify+PatternRecovery) | **Updated**: 2026-03-08

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

> **15m 봇** (비활성): `pattern_15m_bot.py` + `config/pattern_15m_config.yaml` + `results/dynamic_patterns_15m.json` — 거래 빈도 부족(0.21/day) + Multi-TF 필터 연구 7/7 STOP으로 비활성화

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
- **알림 기준**: 연속손실 ≥3, 일일손실 ≤-13%, MDD ≥25%, WR <50% | EXPECTED_WIN_RATE=61.6 (v1.54.0, N-pos+Cascade OOS, 131pat)
- 상세: [docs/agent-guides.md](docs/agent-guides.md)

---

## 📊 현재 전략: Pattern 5m v1.53.0

### 핵심 파라미터

| 파라미터 | 값 |
|---------|-----|
| Entry | 3-candle pattern match (12-type) |
| TP/SL | **Per-pattern ATR-scaled** (TP 0.85-2.80%, SL 1.44-5.95%, MAE/MFE + ATR scanner v2.4, v1.53.0) |
| Classification | Ground Truth (HAMMER/INV_HAMMER 우선순위 수정) |
| Leverage | **Fixed 3x** (v1.42.0: Adaptive 비활성화 — M4+M2 redundancy -46.94, P3 CascadeSL이 MDD 방어 대체) |
| Timeframe | 5m |
| **Max Positions** | **9** (virtual slots, 1/N=11.1% sizing, **mixed-direction** in Hedge) |
| **Position Mode** | **Hedge** (LONG/SHORT 독립 포지션, v1.30.0) |
| Pattern Source | **Dynamic** (results/dynamic_patterns.json, Neutral window ±1%) |
| Discovery | **MAE/MFE + ATR-scaled** (TP=MFE percentile, SL=MAE percentile, ATR scanner v2.4) |
| Scanner MAX_BARS | **288** (24h; v1.28.24: 500→288, 24h timeout study) |
| Quality Filter | **Edge>=18pp + WR>=60% + SL>=1.0% + MC<0.01 + min_trades>=25 + Holdout 7d** |
| Patterns | **131** (59L + 72S), ATR scanner v2.4 + Neutral window + WF 3/3 PASS + Holdout validation (v1.53.0 rescan, 303d, ATR clamp [0.5,1.5]) |
| **Direction Cap** | **7** (max same-direction positions, 7/9 = 78%, v1.36.1 — portfolio study: PnL/MDD 14.43x, corr loss -11%) |
| **Position Timeout** | **288 bars (24h)** — v1.48.0: 864→288 (timeout_sweep_study: OOS min +17.5%, scanner MAX_BARS 일치) |
| Risk | Daily loss **13%** (v1.28.5), **aggregate risk cap** (counter 8%/with 15%, v1.49.0: 5→8% counter) |

### v1.53.0 검증 요약

- **IS (1-pos)**: WR 95.4%, PnL +1,420%, MDD 27.0% | **N-pos IS (Cascade ON)**: 1162 trades, WR 71.3%, PnL +236.4%, MDD 1.37%, PnL/MDD 172.4x
- **131패턴 (59L+72S)**: TP 0.85-2.80%, SL 1.44-5.95%, Edge 18.0-31.8pp, Trades/pat 25-266
- 개별 패턴 상세: `results/dynamic_patterns.json` 참조

### WF OOS 검증 (v1.54.0, Cascade SL ON, 3-fold Expanding Window, N-pos)

| Fold | IS Bars | OOS Bars | IS Patterns | OOS Trades | OOS WR | OOS PnL |
|------|---------|----------|-------------|------------|--------|---------|
| 1 | 18,156 | 18,156 | 40 (35L+5S) | 310 | 61.0% | +32.7% |
| 2 | 36,312 | 18,156 | 94 (72L+22S) | 381 | 57.7% | +40.4% |
| 3 | 54,468 | 18,159 | 109 (62L+47S) | 426 | 66.0% | +55.8% |

**Verdict: 3/3 PASS** | Total OOS PnL: +128.9% (N-pos+Cascade) | Total OOS Trades: 1117 | Avg OOS WR: ~61.6%

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
| **Hedge Mode (v1.30.0)** | **LONG/SHORT 독립 포지션** — FIFO 대비 3배 PnL/MDD, 강제청산 제거 |
| **Position Timeout (v1.31.0→v1.48.0)** | **24h(288bars) 초과 포지션 시장가 청산** — v1.48.0: 864→288 (timeout_sweep_study: OOS min +17.5%, scanner MAX_BARS=288 일치, 24h+ trades net negative) |
| **WR Excess Filter (v1.31.0)** | **Random Walk WR 대비 진짜 엣지 > 5pp만 선별** — 레짐 편향 패턴 제거 |
| **Compact TP/SL (v1.33.0)** | **TP max 2.0%, SL max 2.5%** — 빠른 체결 (median 192b=16h vs Wide 318b=26.5h), 거래 빈도 +47% |
| **Direction Cap (v1.36.1)** | **Max 7 same-direction positions** — portfolio corr-loss study: PnL/MDD 14.43x (cap8 13.54x), corr loss 3.1% vs 3.5% |
| **Holdout Validation (v1.34.0)** | **Scanner --holdout-days 7** — 마지막 7일 OOS 검증, WR Excess<=0 패턴 제거 |
| **Scan Staleness (v1.34.0)** | **dynamic_patterns.json 90일 초과 시 WARNING** — 봇 시작 시 자동 체크 |
| **MDD Sizing (v1.34.0)** | **DD 5%→full, 20%→25% 선형 축소** — peak equity HWM 기반 동적 포지션 사이징 |
| **Trade History (v1.34.0)** | **거래 상세 영속화** — metrics.json에 전체 거래 이력 저장 (로그 로테이션 생존) |
| **ATR Scanner Integration (v1.35.0)** | **Scanner v2.2에 ATR-scaled TP/SL 기본 통합** — Scanner-Production 정합성 확보, `--no-atr`로 Fixed 모드 가능 |
| **Aggregate Risk Cap (v1.35.5→v1.49.0)** | **방향별 SL 노출 합산 제한** — counter **8%**, with **15%** cap |
| **Neutral Window Discovery (v1.36.3)** | Scanner가 start≈end price (±1%) 최장 구간 자동 발견. `--no-neutral`로 비활성화 |
| **Momentum Guard (v1.46.0)** | BTC >1.5%/15min 변동 시 역방향 진입 1h 차단. config `momentum_guard` |
| **Emergency SL Overhaul (v1.36.6)** | `closePosition:true` + 매 루프 `_ensure_emergency_sl_exists()` 선제 검증 |
| **N-pos Scanner (v1.38.1 default)** | Scanner에 N=9/compound/dir_cap/agg_risk/momentum 통합. `--no-npos`로 legacy |
| **Scanner Cascade SL (v1.54.0)** | Scanner N-pos에 Cascade SL 구현. `--no-cascade`로 비활성화. IS WR -15pp, MDD -32%, PnL/MDD +57% |
| **Cascade SL Tightening (v1.45.0)** | SL 피격 시 동일 방향 SL 거리 ×0.15 (85% 축소). 연쇄 적용 (0.15²=2.25%). config `cascade_sl_tightening` |
| **DISABLED (5개)** | Regime Sizing, Adaptive Leverage, Equity Curve, Correlation-Aware, Loss Burst Brake — 각 `enabled: true`로 재활성화 가능. Entry Optimization은 ROLLED BACK (코드 제거) |

### Dynamic Pattern Selection (v1.27.3)

`pattern_source` 설정으로 정적(constants.py) 또는 동적(scanner 출력) 패턴 세트 선택 가능.

| 모드 | 설정값 | 패턴 소스 | TP/SL |
|------|--------|-----------|-------|
| Static (fallback) | `pattern_source: static` | constants.py 51패턴 | Per-pattern 최적화 |
| Dynamic PP | `pattern_source: dynamic` + `tp_sl_mode: per_pattern` | results/dynamic_patterns.json | PP grid search |
| **Dynamic ATR (현재)** | `pattern_source: dynamic` + `tp_sl_mode: per_pattern` | results/dynamic_patterns.json | **MAE/MFE + ATR-scaled** |

**Scanner CLI 사용법** (v2.4):
```bash
cd bingx_rl_trading_bot
# 기본 (neutral + ATR + N-pos, v1.38.1~ default)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --wf-folds 3 --holdout-days 7
# Legacy 1-pos 모드 (빠른 반복용)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --wf-folds 3 --holdout-days 7 --no-npos
# 주요 옵션: --no-neutral, --no-atr, --neutral-tol 2.0, --atr-clamp-lo 0.5 --atr-clamp-hi 1.5, --n-slots 5 --direction-cap 4
```

**현재 적용**: MAE/MFE + ATR scanner + Neutral window (edge>=18pp, MC<0.01, --wf-folds 3, --holdout-days 7) → **131패턴 (59L+72S)** (v1.53.0 rescan).
WF N-pos 3/3 PASS, OOS +110.6% (aligned rescan). Neutral window ±1% 자동 탐색 (259d). ATR config: a14/w576/clamp[0.5,1.5]. Data: 303d. Backup: `results/dynamic_patterns_131pat_v1530_backup.json`

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
│   │   └── pattern_5m/             # 14개 모듈 (multi-TF 지원)
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
├── results/                        # 봇 상태/메트릭 JSON
├── logs/                           # pattern_5m 운영 로그
├── claudedocs/                     # 활성 연구 리포트 (2026~)
└── archive/                        # 레거시 전체 (ML data, logs, experiments 등)
```

---

## ✅ Pattern Validation Audit (2026-02-03)

12/12 패턴 독립 검증 통과. 외부 주장(WF FAIL, 슬리피지 -53%) 모두 거짓 확인 (실제 슬리피지 영향 1/3 수준). 상세: 보고서 아카이브됨.

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
params={'positionSide': 'LONG'}   # Hedge mode (v1.30.0)
params={'positionSide': 'SHORT'}  # Hedge mode (v1.30.0)
# params={'positionSide': 'BOTH'}  # One-Way mode (v1.29.x 이전)
```

> 상세: [claudedocs/STANDARD_RESEARCH_PROTOCOL.md](bingx_rl_trading_bot/claudedocs/STANDARD_RESEARCH_PROTOCOL.md)

---

## 📜 Version History (Recent)

> 전체 히스토리: [docs/VERSION_HISTORY.md](docs/VERSION_HISTORY.md)

| 버전 | 날짜 | 변경사항 |
|------|------|---------|
| **v1.55.0** | 03-08 | **Live 안정성 3종 개선** ← 현재. (1) N/A 패턴 방지: crash recovery 시 trade_history에서 pattern 복원 (2) Exit 분류 강화: near-SL 40%/near-TP 30% proximity 분류 (3) Mass closure guard: 3+ 동시 청산 시 API 재확인 |
| v1.54.0 | 03-07 | Scanner Cascade SL 구현 + EXIT 분류 개선. N-pos IS: WR 71.3%, PnL +236.4%, MDD 1.37%, PnL/MDD 172.4x. WF 3/3 PASS (OOS +128.9%). CASCADE_SL exit reason 추가 |
| v1.53.0 | 03-05 | Data 303d + Rescan 131pat (59L+72S). WF 3/3 PASS (OOS +110.6%, aligned). N-pos IS: WR 86.6%, PnL +220.8%, MDD 2.01% |
| (연구) | 03-05 | 5개 파라미터 Sweep **ALL KEEP baseline** — 최적화 공간 소진 |
| (연구) | 03-04 | Entry Optimization **ROLLBACK** — WF 94% PASS rate (비판별), 95% Cascade 의존 |
| v1.52.0 | 03-05 | Scanner ATR clamp [0.5,1.5] 정합성. 125패턴. WF 3/3 PASS |
| v1.51.0 | 03-05 | Momentum Guard threshold 1.0→1.5%. IS +5.3% |
| v1.50.0 | 03-05 | ATR clamp_hi 1.7→1.5. IS PnL/MDD +52% |
| v1.49.0 | 03-05 | AggRisk counter 5→8%. IS +29%, OOS min +12% |
| v1.48.0 | 03-05 | Timeout 864→288 (24h). OOS min +17.5%. Scanner MAX_BARS 일치 |
| v1.47.0 | 03-05 | ATR clamp_lo 0.6→0.5. IS +44%, MDD -25% |
| v1.46.0 | 03-05 | Momentum Guard lb3/cd12 (15min감지/1h보호). IS +32% |
| v1.45.0 | 03-05 | Cascade SL tighten_pct 75→85%. IS +44%, OOS min +64.4% |
| v1.44.0 | 03-05 | AggRisk Relaxation counter 3→5%, with 7→15%. IS +108% |
| v1.42.0 | 03-03 | Mechanism Stack — M2 Regime + M4 AdaptiveLev OFF. IS +260% |
| v1.41.0 | 03-03 | Cascade SL Tightening 도입. PnL/MDD 2.9x 향상 |

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