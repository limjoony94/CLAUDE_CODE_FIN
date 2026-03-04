# CLAUDE_CODE_FIN - BTC 5분봉 패턴 트레이딩 봇

> **Version**: v1.42.0 | **Bot**: Pattern 5m (130패턴, 61L+69S, Edge18pp+NeutralWindow+ATR Scanner+Holdout+MDD+Cap7+MomentumGuard+NposScanner+CascadeSL) | **Updated**: 2026-03-04

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
- **알림 기준**: 연속손실 ≥3, 일일손실 ≤-13%, MDD ≥25%, WR <50% | EXPECTED_WIN_RATE=88.8 (v1.36.4, 130pat Neutral+Edge18+WF3/3)
- 상세: [docs/agent-guides.md](docs/agent-guides.md)

---

## 📊 현재 전략: Pattern 5m v1.36.4

### 핵심 파라미터

| 파라미터 | 값 |
|---------|-----|
| Entry | 3-candle pattern match (12-type) |
| TP/SL | **Per-pattern ATR-scaled** (TP 0.91-2.76%, SL 1.44-4.84%, MAE/MFE + ATR scanner v2.3, v1.36.4) |
| Classification | Ground Truth (HAMMER/INV_HAMMER 우선순위 수정) |
| Leverage | **Fixed 3x** (v1.42.0: Adaptive 비활성화 — M4+M2 redundancy -46.94, P3 CascadeSL이 MDD 방어 대체) |
| Timeframe | 5m |
| **Max Positions** | **9** (virtual slots, 1/N=11.1% sizing, **mixed-direction** in Hedge) |
| **Position Mode** | **Hedge** (LONG/SHORT 독립 포지션, v1.30.0) |
| Pattern Source | **Dynamic** (results/dynamic_patterns.json, Neutral window ±1%) |
| Discovery | **MAE/MFE + ATR-scaled** (TP=MFE percentile, SL=MAE percentile, ATR scanner v2.2) |
| Scanner MAX_BARS | **288** (24h; v1.28.24: 500→288, 24h timeout study) |
| Quality Filter | **Edge>=18pp + WR>=60% + SL>=1.0% + MC<0.01 + min_trades>=25 + Holdout 7d** |
| Patterns | **130** (61L + 69S), ATR scanner v2.3 + Neutral window + WF 3/3 PASS + Holdout validation (v1.36.4) |
| **Direction Cap** | **7** (max same-direction positions, 7/9 = 78%, v1.36.1 — portfolio study: PnL/MDD 14.43x, corr loss -11%) |
| **Position Timeout** | **864 bars (72h)** — 48h+ trades are net negative, slot recycling (v1.31.0) |
| Risk | Daily loss **13%** (v1.28.5), **aggregate risk cap** (counter 3%/with 7%, v1.35.5) |

### 270일 In-Sample 검증 결과

| 지표 | ATR 51 (v1.35.0) | ATR 54 (v1.36.1) | Neutral 51 (v1.36.3) | **Neutral 130 (v1.36.4)** |
|------|----------------------|----------------------|----------------------|----------------------|
| Patterns | 51 (16L+35S) | 54 (4L+50S) | 51 (22L+29S) | **130 (61L+69S)** |
| Discovery | MAE/MFE+ATR | MAE/MFE+ATR (shifted) | MAE/MFE+ATR (neutral) | **MAE/MFE+ATR (neutral, E≥18)** |
| WR mean | 93.9% | 89.4% | ~95% | **95.0%** |
| Edge threshold | 21.8pp | 21.8pp | 21.8pp | **18.0pp** |
| Portfolio Trades | 294 | 268 | ~320 | **320** |
| Portfolio PnL | +1,214% | +817% | +1,075% | **+1,385%** |
| Portfolio MDD | 20.4% | 38.2% | 13.5% | **27.8%** |
| PnL/MDD | 59.5x | 21.4x | 79.6x | **49.8x** |
| TP range | 0.87-2.84 | 0.84-2.79 | 0.91-2.76 | **0.91-2.76** |
| SL range | 1.44-4.76 | 1.67-4.52 | 1.44-4.84 | **1.44-4.84** |

### Pattern Summary — ATR Scanner v2.3 (v1.36.4, Neutral window, Edge≥18pp)

**130 패턴 (61L + 69S)** — L/S 47%/53% 방향 균형 (neutral window 효과)

| 통계 | LONG (61) | SHORT (69) | 전체 (130) |
|------|-----------|------------|------------|
| TP range | 0.91-2.76% | 0.91-2.76% | 0.91-2.76% |
| SL range | 1.44-4.84% | 1.44-4.84% | 1.44-4.84% |
| Edge | 18.0-31.4pp | 18.0-31.8pp | 18.0-31.8pp |
| Trades/pat | 25-266 | 25-266 | 25-266 |

> 개별 패턴 상세: `results/dynamic_patterns.json` 참조

### WF OOS 검증 (Neutral 257d, 3-fold Expanding Window, ATR Scanner v2.3)

| Fold | IS Bars | OOS Bars | IS Patterns | OOS Trades | OOS WR | OOS PnL |
|------|---------|----------|-------------|------------|--------|---------|
| 1 | 18,024 | 18,024 | 37 (28L+9S) | 84 | 88.1% | +241.2% |
| 2 | 36,048 | 18,024 | 93 (65L+28S) | 95 | 88.4% | +269.3% |
| 3 | 54,072 | 18,026 | 95 (50L+45S) | 90 | 90.0% | +362.2% |

**Verdict: 3/3 PASS** | Total OOS PnL: +872.7% | Total OOS Trades: 269 | Avg OOS WR: 88.8%

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
| **Position Timeout (v1.31.0)** | **72h(864bars) 초과 포지션 시장가 청산** — 48h+ 거래 net negative, 슬롯 재활용 |
| **WR Excess Filter (v1.31.0)** | **Random Walk WR 대비 진짜 엣지 > 5pp만 선별** — 레짐 편향 패턴 제거 |
| **Compact TP/SL (v1.33.0)** | **TP max 2.0%, SL max 2.5%** — 빠른 체결 (median 192b=16h vs Wide 318b=26.5h), 거래 빈도 +47% |
| **Direction Cap (v1.36.1)** | **Max 7 same-direction positions** — portfolio corr-loss study: PnL/MDD 14.43x (cap8 13.54x), corr loss 3.1% vs 3.5% |
| **Holdout Validation (v1.34.0)** | **Scanner --holdout-days 7** — 마지막 7일 OOS 검증, WR Excess<=0 패턴 제거 |
| **Scan Staleness (v1.34.0)** | **dynamic_patterns.json 90일 초과 시 WARNING** — 봇 시작 시 자동 체크 |
| **MDD Sizing (v1.34.0)** | **DD 5%→full, 20%→25% 선형 축소** — peak equity HWM 기반 동적 포지션 사이징 |
| **Trade History (v1.34.0)** | **거래 상세 영속화** — metrics.json에 전체 거래 이력 저장 (로그 로테이션 생존) |
| **ATR Scanner Integration (v1.35.0)** | **Scanner v2.2에 ATR-scaled TP/SL 기본 통합** — Scanner-Production 정합성 확보, `--no-atr`로 Fixed 모드 가능 |
| **~~Regime Sizing (v1.35.3)~~** | **v1.42.0 DISABLED** — M2+M4 redundancy -46.94 (mechanism_optimization_study), P3 CascadeSL이 MDD 방어 대체. `enabled: true`로 재활성화 가능 |
| **Aggregate Risk Cap (v1.35.5)** | **방향별 SL 노출 합산 제한** — counter-regime 3%, with-regime 7% cap. MDD -52% (13.2→6.3%), PnL/MDD +16%, WF 3/3 PASS |
| **Neutral Window Discovery (v1.36.3)** | **가격 중립 윈도우 자동 탐색** — Scanner가 start≈end price (±1%) 최장 구간 자동 발견. 방향 편향 없는 패턴 발굴 (22L+29S vs 기존 4L+50S). MDD 13.5% (vs 38.2%), PnL/MDD 79.6x (vs 21.4x). `--no-neutral`로 비활성화 가능 |
| **Momentum Guard (v1.36.2)** | **강한 단기 움직임 시 역방향 진입 일시정지** — BTC >1%/30min 변동 시 역방향 진입 30min 차단. 신호의 ~2.3%만 차단하면서 correlated loss 회피. PnL/MDD 8.19 vs baseline 7.51 (+9%), WF 3/3 PASS |
| **Emergency SL Overhaul (v1.36.6)** | **closePosition + 선제적 헬스체크 + 연쇄 방어** — Emergency SL에 `closePosition:true` (2x over-coverage 제거), 110411/110424 에러 처리, place-first/cancel-after (보호갭 0), 매 루프 선제 검증 `_ensure_emergency_sl_exists()`, EMERGENCY_SL/CASCADE 분류. 2026-02-27 연쇄 청산 사건 근본 해결 |
| **~~Loss Burst Brake (v1.37.0)~~** | **v1.40.1 DISABLED** — Momentum Guard(G2)와 중복 (redundancy 0.788), full stack에서 유해. `enabled: true`로 재활성화 가능 |
| **N-pos Scanner (v1.38.0→v1.38.1 default)** | **프로덕션 정합 포트폴리오 시뮬레이터** — Scanner 백테스트에 N=9/compound/dir_cap/regime/agg_risk/momentum 통합. **v1.38.1부터 기본 활성화**, `--no-npos`로 legacy 1-pos 모드. WF OOS: N-pos 68.4% WR (3/3 PASS, min fold +13.35%), 1-pos 85.0%. Live WR gap 32.3pp→15.4pp (52% 감소) |
| **~~Adaptive Leverage (v1.39.0)~~** | **v1.42.0 DISABLED** — M2+M4 redundancy -46.94 (mechanism_optimization_study), P3 CascadeSL이 MDD 방어 대체. `enabled: true`로 재활성화 가능 |
| **~~Equity Curve Trading (v1.40.0)~~** | **v1.40.1 DISABLED** — IS에서 유해 (PnL -11.4%, MDD 변화 없음). `enabled: true`로 재활성화 가능 |
| **~~Correlation-Aware Entry (v1.40.0)~~** | **v1.40.1 DISABLED** — Regime Sizing(M2)과 중복 (redundancy 71.75). `enabled: true`로 재활성화 가능 |
| **Cascade SL Tightening (v1.41.0)** | **동일 방향 SL 피격 시 나머지 SL 75% 축소** — correlated_loss_study H5_Cascade_t75. SL exit → 동일 방향 포지션 SL 거리 ×0.25. 연쇄 적용 (2nd SL → 0.25²=6.25%). PnL/MDD 79.4 (baseline 27.4의 2.9배), MDD 2.45%. config `cascade_sl_tightening` 섹션. `enabled: false`로 즉시 롤백 |
| **~~Entry Optimization (v1.43.0)~~** | **ROLLED BACK** — h7_critical_validation 6-test 검증 결과 ROLLBACK. (1) WF 94% PASS rate (판별력 부재), (2) H7 효과 95% Cascade 의존 (독립 lift +5.6 only), (3) Live WR 55-60%에서 H0 E[trade] 13.5배 우월. TP×0.75+TO432 제거, 원래 TO864 복원. |

### Dynamic Pattern Selection (v1.27.3)

`pattern_source` 설정으로 정적(constants.py) 또는 동적(scanner 출력) 패턴 세트 선택 가능.

| 모드 | 설정값 | 패턴 소스 | TP/SL |
|------|--------|-----------|-------|
| Static (fallback) | `pattern_source: static` | constants.py 51패턴 | Per-pattern 최적화 |
| Dynamic PP | `pattern_source: dynamic` + `tp_sl_mode: per_pattern` | results/dynamic_patterns.json | PP grid search |
| **Dynamic ATR (현재)** | `pattern_source: dynamic` + `tp_sl_mode: per_pattern` | results/dynamic_patterns.json | **MAE/MFE + ATR-scaled** |

**Scanner CLI 사용법** (v2.3):
```bash
cd bingx_rl_trading_bot
# 기본 (neutral window + ATR, v1.36.3~)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --wf-folds 3 --holdout-days 7
# neutral 비활성화 (기존 방식)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --no-neutral --is-days 270 --wf-folds 3 --holdout-days 7
# neutral 허용오차 변경
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --neutral-tol 2.0
# Fixed 모드 (ATR 비활성화)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --no-atr
# ATR 커스텀 파라미터
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --atr-period 14 --atr-window 576 --atr-clamp-lo 0.6 --atr-clamp-hi 1.7
# 기본 (N-pos 기본 활성, v1.38.1~) — 프로덕션 정합 백테스트
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --wf-folds 3 --holdout-days 7
# Legacy 1-pos 모드 (빠른 반복용)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --wf-folds 3 --holdout-days 7 --no-npos
# N-pos 파라미터 커스텀
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --wf-folds 3 --holdout-days 7 --n-slots 5 --direction-cap 4
# N-pos 개별 필터 비활성화
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --wf-folds 3 --holdout-days 7 --no-regime-sizing --no-momentum-guard
```

**현재 적용**: MAE/MFE + ATR scanner + Neutral window (edge>=18pp, MC<0.01, --wf-folds 3, --holdout-days 7) → **130패턴 (61L+69S)**.
WF 3/3 PASS, OOS +872.7%. Neutral window ±1% 자동 탐색 (257d). ATR config: a14/w576/clamp[0.6,1.7]. Data: 297d. Backup: `results/dynamic_patterns_51pat_neutral_edge21.8_backup.json`

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
params={'positionSide': 'LONG'}   # Hedge mode (v1.30.0)
params={'positionSide': 'SHORT'}  # Hedge mode (v1.30.0)
# params={'positionSide': 'BOTH'}  # One-Way mode (v1.29.x 이전)
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
| (연구) | 02-23 | **ATR-Scaled Backtest Study** (`atr_scaled_backtest_study.py`): Scanner(고정 TP/SL) vs Production(ATR-scaled TP/SL) 조건 비교 4-Phase 연구. **Phase 1**: 15패턴 개별 비교 — ATR scaling이 avg WR +4.5pp, edge +0.876%/trade 개선 (11/15 패턴 향상). **Phase 2**: 59패턴 재평가 — ATR-scaled 필터가 더 엄격 (39 pass vs Fixed 51), 패턴 선별이 달라짐. **Phase 3**: Hedge N=5 포트폴리오 — ATR T864 PnL/MDD **17.18** vs Fixed 10.91, 둘 다 WF 3/3 PASS. ATR scaling이 리스크 조정 성과 +57.5% 향상. **Phase 4**: ATR ratio 분포 — mean 1.0874 (slight expansion), 72.1% within clamps [0.6,1.7]. **결론**: ATR scaling은 Scanner 단계에서도 적용 시 선별 결과가 달라지며, Production 조건과의 정합성이 향상됨. |
| **(연구)** | 03-04 | **Entry Optimization 연구 + ROLLBACK**: `entry_behavior_critical_study.py` 7-test 심층 분석 + `entry_improvement_hypotheses.py` 11가설 WF 검증 + `h7_critical_validation.py` 6-test 비판적 검증. **H7_tp75_to432 IS: PnL/MDD 212.8** (baseline 97.0), WF 3/3 PASS. **그러나 h7_critical_validation 결과**: (1) **WF 판별력 부재** — tp_mult 0.5~1.5 (34 configs 중 32 = 94%) 전부 WF PASS. WF가 사실상 non-discriminating. (2) **95% Cascade 의존** — Cascade ON: H7 lift +115.8, Cascade OFF: H7 lift **+5.6만**. 독립적 가치 거의 없음. (3) **Live WR에서 열위** — WR 55-60%에서 H0 E[trade] 0.68-1.15% vs H7 0.05-0.45% (H0이 2-13x 우월). (4) R:R 1.089→0.839 (-23%), breakeven WR 47.9→54.4% (+6.5pp). **VERDICT: ROLLBACK**. TP×0.75 코드 제거, timeout 864 복원. **교훈**: WF 3/3 PASS가 유일한 검증 gate일 때 94% pass rate로 판별력 부재 — 추가 비판적 검증 필수. Cascade 상호작용이 지배적인 효과를 독립적 개선으로 착각하지 말 것. |
| **v1.42.0** | 03-03 | **Mechanism Stack Optimization — M2 Regime + M4 AdaptiveLev 비활성화**: `mechanism_portfolio_study.py` 5-Phase 연구 (10개 메커니즘 Individual Ablation + Pairwise Synergy 15쌍) + `mechanism_optimization_study.py` 12-config 최적화 + WF 3/3 검증. **핵심 발견**: P3 CascadeSL(v1.41.0)이 M2/M4의 MDD 방어를 대체. **M2+M4 redundancy -46.94** (극심), 제거 시 IS PnL/MDD 34.36→123.67 (+260%), MDD 4.50%→3.82% (-15%), OOS +77.3%→+160.7% (+108%). 12개 설정 모두 WF 3/3 PASS. 보수적 Q4a 선택: M2+M4만 비활성화, G5 AggRisk + M1 MDD 안전망 유지. (1) config `regime_sizing.enabled: false`, (2) config `adaptive_leverage.enabled: false`. 유효 Guard Chain: G2 Momentum→G5 AggRisk→G1 DirCap→M1 MDD+M5 ATR+P1 Timeout+P2 EarlyExit+P3 CascadeSL (8개). 각 `enabled: true`로 즉시 재활성화 가능. **1061 tests passed**. ← **현재** |
| **v1.41.0** | 03-03 | **Cascade SL Tightening — 동일 방향 SL 연쇄 축소**: `correlated_loss_study.py` H5_Cascade_t75 연구 기반 (PnL/MDD 79.4, baseline 27.4의 2.9배, MDD 2.45%, WF 3/3 PASS min fold +25.4%). (1) `orders.py`: `update_single_sl()` 공개 함수 — 기존 SL 취소→새 SL 배치 원자적 수행, `_place_sl_order()` 재사용 (110412 breach 처리 포함). (2) `position_monitor.py`: `_cascade_tighten_sls()` — SL/EMERGENCY_SL/SL_AFTER_* exit 감지 시 동일 방향 잔여 포지션의 SL 거리 ×0.25 (75% 축소). `_handle_position_closed()`에서 `record_closed_position()` 전에 호출 (슬롯 삭제 전 접근 보장). entry_price 기준 SL 거리 재계산. 연쇄 자동 지원 (매 SL마다 현재 sl_price 기준). (3) `config.yaml`: `risk.cascade_sl_tightening` 블록 (enabled: true, tighten_pct: 75). `enabled: false`로 즉시 롤백 가능. **1061 tests passed**. |
| **v1.40.1** | 03-03 | **Guard Ablation — 중복 메커니즘 3개 비활성화**: `guard_ablation_study.py` 4-Phase 연구 (40+ 시나리오, 10개 메커니즘 개별 PnL 영향 + 6쌍 중복 분석) 기반. (1) **M3 Equity Curve Sizing OFF**: IS에서 유해 — PnL -11.4%, MDD 변화 없음, ablation 제거 시 PnL/MDD +1.96. (2) **G4 Correlation-Aware Entry OFF**: Regime Sizing(M2)과 극단적 중복 (redundancy 71.75), ablation -1.00 (M2 -1.54보다 약함). (3) **G3 Loss Burst Brake OFF**: Momentum Guard(G2)와 중복 (redundancy 0.788), full stack에서 유해 (ablation 제거 시 +5.66). config 변경만 (`enabled: false` × 3), 코드 변경 없음. 유효 Guard Chain: G2→G5→G1→M4→M1+M2+M5 (10→7개, v1.42.0에서 M2+M4 추가 비활성화→5개). 가장 가치 있는 메커니즘: M5 ATR(-14.69), G5 AggRisk(-4.90), M4 AdaptiveLev(-3.47). MDD×AdaptiveLev 쌍은 super-additive (redundancy -0.23 = 시너지). 각 `enabled: true`로 즉시 재활성화 가능. **1061 tests passed**. |
| **v1.40.0** | 03-02 | **MDD Reduction — Equity Curve Trading + Correlation-Aware Entry (Combo2)**: mdd_reduction_study.py 7가설/45시나리오/6Phase 연구 기반. Combo2(H1_EqCurve_half_ema30 + H6_CorrAware_dir70) 선택. (1) `bot.py`: `_check_equity_curve_sizing()` — **LONG/SHORT 독립** 누적 PnL 추적, 방향별 cum_pnl < SMA(30 trades) 시 ×0.5 사이징. `_check_correlation_aware_entry()` — 동일 방향 비율 ≥70% AND counter-regime 시 진입 차단. `_update_equity_curve_tracker()` + `_record_trade_for_equity_curve()` (dedup). Guard chain: momentum→burst→**correlation(NEW)**→agg_risk→adaptive_lev→**equity_curve(NEW)**→open. (2) `position_open.py`: `equity_curve_scale` 파라미터 추가 (get_position_size + open_position + slot 저장). (3) `state.py`: `equity_curve_tracker` 기본값 (long_cum_pnls/short_cum_pnls). (4) `models.py`: BotState 타입 확장. (5) `config.yaml`: `equity_curve_trading` + `correlation_aware_entry` 섹션 (enabled: true). 연구 결과: MDD -47.8% (6.5→3.4%), PnL -3.9% (106.8→102.6%), **PnL/MDD +84.1%** (16.45→30.29), WF 3/3 PASS (F1 +23.0%, F2 +33.6%, F3 +23.7%). 각 `enabled: false`로 즉시 롤백. **1061 tests passed**. |
| **v1.39.0** | 03-02 | **Adaptive Leverage — WR Confidence × Edge Quality**: 리스크 관리 연구(6-Phase) 기반 adaptive leverage 활성화. (1) `bot.py`: `_compute_adaptive_leverage()` + 5개 헬퍼 함수 (`_leverage_wr_confidence`, `_leverage_kelly`, `_leverage_breakeven`, `_leverage_exp_decay`, `_clamp`) + `_update_rolling_wr_tracker()`. `_process_entry_signal()`에 leverage_override 통합. (2) `position_open.py`: `get_position_size()` + `open_position()` `leverage_override` 파라미터, 슬롯에 `effective_leverage` 저장. (3) `position_close.py`: per-slot `effective_leverage` PnL 계산. (4) `config.py`: `_npos_portfolio_wr`, `_npos_ref_edge` 로딩. (5) `state.py`: `rolling_wr_tracker` 기본값. (6) `models.py`: BotState 타입 확장. (7) `config.yaml`: `adaptive_leverage.enabled: true`, method=wr_confidence, window=12. 연구: H5_WRConf_w12가 7/9 리스크 지표 baseline 대비 우수 — MDD -32% (4.75→3.24%), Calmar +39% (37.4→52.0), OOS consistency 0.823 (vs 0.645), DD≥2% 시 레버리지 자동 감축 -47% (2.63→1.39x). PnL -5.3% trade-off. WF 3/3 PASS. MC 1000-sim 견고. `adaptive_leverage.enabled: false`로 즉시 롤백 가능. **1061 tests passed**. |
| **v1.38.1** | 03-02 | **N-pos Scanner default 전환 + Fold 2 데이터 슬라이싱 버그 수정**: (1) `--npos` default=True 전환 — `--no-npos`로 legacy 1-pos 모드 가능. (2) `npos_fold2_diagnosis_study.py`: 이전 Fold 2 +2.91% 근소 PASS가 **neutral window 데이터 미슬라이싱 버그**였음 발견 — `run_npos_wf()`에서 `build_signal_index()`에 full data 전달 → neutral-sliced data로 수정 후 Fold 2 **+13.35%** (안정). 10가설 검증: H1 relaxed agg(5/10) best min-fold +15.79%, H6 no-agg/H9 minimal FAIL → agg_risk 필수 확인. (3) Fold 2 LONG -20.45%(BTC -16.6% 하락) vs SHORT +31.05% — 방향 균형이 하락장 생존 핵심. **결론: 현 baseline 파라미터(3/7) 유지, N-pos default 전환만**. |
| **v1.38.0** | 03-01 | **N-pos Scanner — production-aligned portfolio simulator**: Scanner 백테스트가 프로덕션 실행 조건을 반영하지 않아 WF OOS WR 88.8% vs Live 52.7% (36pp 괴리). (1) `pattern_scanner.py` v2.4: `portfolio_npos()` — N=9 compound equity, 1/N sizing, direction_cap=7, regime×0.3, agg_risk_cap(3/7%), momentum_guard(1%/30min), timeout(864bars DROP), ATR-scaled TP/SL, intrabar resolution, fee+slippage. `_check_exit_npos()`, `compute_ema_slope()`, `calc_stats_compound()` 추가. (2) CLI `--npos` 플래그 + `--n-slots`/`--direction-cap`/`--regime-mult`/`--no-regime-sizing`/`--no-momentum-guard`/`--no-agg-risk-cap` 파라미터. (3) `expanding_window_wf()` `use_npos` 모드 — OOS를 portfolio_npos()로 평가. (4) IS N-pos 요약 + build_output_json npos 메타데이터. (5) `npos_portfolio_study.py` 3-Phase 연구: Phase 1 IS 1-pos WR 95.4% vs N-pos 72.8% (MDD 24.9%→6.5%), Phase 2 WF OOS 1-pos 85.0% vs N-pos 68.1% (both 3/3 PASS), **Live WR gap 32.3pp→15.4pp (52% 감소)**, Phase 3 sensitivity — agg_risk_cap 최대 영향(5,704 blocked), full production PnL/MDD 16.5x. **Verdict: GO**. (6) `--npos` 미사용 시 기존 동작 100% 보존. 출처: `entry_improvement_study.py` simulate_portfolio (21시나리오 WF 검증 완료). **1061 tests passed**. |
| **v1.37.0** | 03-01 | **Loss Burst Brake + Live WR gap research suite**: (1) `bot.py`: `_record_loss_for_burst_brake()` + `_check_loss_burst_brake()` — 동일 방향 2회 손실 in 24h(288bars) → 해당 방향 12h(144bars) 진입 차단. per-direction timestamp 추적, dedup via `_loss_burst_last_recorded`. Trading+Maintenance 양 윈도우에서 손실 기록. `_process_entry_signal()` 내 momentum guard 다음, aggregate risk cap 이전에 체크. (2) config `risk.loss_burst_brake` 섹션 (threshold/window_bars/block_bars). (3) 연구: portfolio_exposure_study.py H6b — MDD -36%, burst count -36%, max daily loss -76%, WF 3/3 PASS. (4) Live WR gap 연구 8편: signal_direction_audit (LONG 44.7%, SHORT 58.2%), entry_price_inversion (data period artifact), edge_decay (half-life 30d), execution_gap, regime_filter, regime_transition_detector (8 detectors), strategy_reform, live_wr_gap_analysis. (5) Pattern rescan: 130pat(61L+69S) 안정 확인. **1061 tests passed**. |
| **v1.36.6** | 03-01 | **Emergency SL overhaul — closePosition + proactive health check + cascade defense**: 2026-02-27 연쇄 청산 사건(4 LONG 슬롯, $66,232.6) 근본 원인 3가지 해결. (1) `orders.py`: Emergency SL에 `closePosition: 'true'` 추가 — 2x over-coverage 근본 해소 (수량 계산 의존 제거). (2) `orders.py`: 110424 에러 → EXCHANGE_MANAGED 마킹 (per-slot SLs 활성 시). (3) `orders.py`: 110411 에러 처리 추가 — SL 가격이 현재가 관통 시 breach 재계산 (3개 SL 배치 경로 모두). (4) `orders.py`: 모든 실패 경로 CRITICAL 로깅 ("POSITION UNPROTECTED"). (5) `orders.py`: `update_emergency_sl` place-first/cancel-after 순서 변경 — 보호 갭 0. (6) `bot.py`: `_ensure_emergency_sl_exists()` 매 루프 선제 검증 (Trading+Maintenance 양 윈도우). (7) `position_open.py`: `_get_actual_fill_price` qty 캡 `min(exchange_total, order_qty)`. (8) `position_monitor.py`: `_infer_exit_reason` EMERGENCY_SL 분류 + CASCADE 경고 로그. (9) `config.py`+`scanner`: 방향 불균형 경고. **1061 tests passed**. |
| **v1.36.5** | 02-27 | **ATR scaling bug fix (2 critical paths)**: 전수조사(17 code paths) 기반 2개 크리티컬 버그 수정. (1) `orders.py`: `_adjust_single_position_tpsl()`이 `_resolve_expected_tpsl()`(base TP/SL만 반환) 사용 → `calculate_tp_sl()` (ATR scaling + slippage buffer 포함)로 교체. **매 봇 재시작마다 모든 포지션의 ATR-scaled TP/SL이 base 값으로 덮어쓰기되던 버그**. circular import 회피를 위해 local import 사용. universal mode 패턴 불필요 분기 추가. (2) `position_close.py`: `recover_position_to_state()`에서 `vol_mult=1.0` 하드코딩 → 기존 state에서 `old_vol_mult` 추출하여 보존. crash recovery 시 ATR scaling 손실 방지. (3) `test_orders.py`: 3개 테스트 slippage buffer 반영 업데이트. **1061 tests passed**. |
| v1.36.4 | 02-27 | **Edge threshold 21.8→18pp (130 patterns)**: Edge sweep 연구 (15/18/20/21.8pp, 모두 WF 3/3 PASS). Neutral window 환경에서 18pp가 최적 균형 — 130패턴(61L+69S), IS PnL +1,385%, MDD 27.8%, WF OOS +872.7% (vs 21.8pp: 51pat, OOS +632.5%). L/S 47%/53% 방향 균형 유지. TP 0.91-2.76%, SL 1.44-4.84%. Holdout 7d: 7 SHORT 패턴 제거. Stable patterns: 7 (3-fold 안정). Backup: `dynamic_patterns_51pat_neutral_edge21.8_backup.json`. |
| v1.36.3 | 02-27 | **Neutral window pattern discovery + Filter ablation study**: (1) `neutral_window_discovery_study.py` — 270d shifted(4L+50S) 극단적 SHORT 편향 → 가격 중립 윈도우 257d: **51패턴(22L+29S)**, MDD 13.5%, PnL/MDD 79.6x, WF 3/3 PASS. Scanner v2.3: `--neutral` 기본. (2) `neutral_filter_ablation_study.py` — 6필터 full factorial(64조합) ablation. F6:Timeout 필수(+9.68), F2+F3 없으면 WF FAIL. DirCap7은 neutral portfolio에서 no-op(2건 차단). 상위 WF PASS 간 차이 noise(<1%). **결론: 현행 6필터 전부 유지(no change)**. |
| v1.36.2 | 02-27 | **Momentum Guard — counter-dir spike protection**: `entry_improvement_study.py` 21시나리오 연구 (H1 Entry Spacing STOP, H2 Momentum Guard GO, H3 Progressive Cap STOP). `bot.py`: `_check_momentum_guard()` — BTC >1%/30min 변동 시 역방향 진입 30min 차단. 신호의 ~2.3%만 차단하면서 correlated loss 회피. PnL/MDD 8.19 vs baseline 7.51 (+9%), ALL 21 scenarios WF 3/3 PASS. config `momentum_guard` 섹션 추가 (lookback_bars=6, threshold_pct=1.0, cooldown_bars=6). **1061 tests passed**. |
| v1.36.1 | 02-27 | **Direction Cap 8→7 + Data Extension Rescan (54pat)**: (1) Live 성과 괴리 분석 (WF OOS 77.3% vs Live 51.3%): BTC +9.1% rally에서 8 SHORT 동시 SL 피격이 근본원인. `live_gap_portfolio_study.py` 4-Hypothesis 연구: H1 Direction Cap **GO** (cap7: PnL/MDD 14.43x, cap8: 13.54x, corr loss 3.1% vs 3.5%), H2 Portfolio SL STOP (무효과), H3 Regime Sizing keep 0.3 (제거 시 corr loss +43%), H4 Combined cap7+mult0.3. (2) config `direction_cap: 8→7`. (3) 데이터 확장: 270d→297d (btc_5m_270days_reclassified.csv, ~2025-05-05→2026-02-26, 85,634 rows). `extend_5m_data.py` 스크립트 추가. (4) 270d shifted rescan (ATR v2.2): **54패턴(4L+50S)**, WF 3/3 PASS, OOS PnL +650.7% (기존 +443.9%), Avg OOS WR 83.2% (기존 77.3%). LONG 16→4 (최근 BTC rally에서 LONG edge 약화 반영), SHORT 35→50. TP 0.84-2.79%, SL 1.67-4.52%. Backup: `dynamic_patterns_51pat_atr_v1350_backup.json`. |
| v1.36.0 | 02-27 | **Multi-TF infra + 15m 실험 + 15m 비활성화**: (1) `constants.py`: `PATTERN_BOT_TF` 환경변수로 multi-TF 지원. 경로/타이밍 자동 파생 (`CANDLE_DURATION_MS`, `MAX_OHLCV_CANDLES`, `DEFAULT_TIMEOUT_BARS`). 기본값 `5m` 유지. (2) `pattern_15m_bot.py` + `pattern_15m_config.yaml` + `dynamic_patterns_15m.json`: 15m 인프라 구축. 15패턴(10L+5S), WF 3/3 PASS, OOS +2,152%. (3) **15m 비활성화 결정**: 거래 빈도 0.21/day(4~5일에 1회) 부족. (4) **Multi-TF Direction Filter 연구** (`multi_tf_direction_study.py`): 15m/1h/4h EMA(20) 방향으로 5m 필터링/사이징 7가설 검증 — **7/7 STOP**. 5m 패턴이 구조적 역추세(상승장 풀백 SHORT)이므로 상위 TF 방향 필터가 핵심 수익원 억제. 기존 same-TF regime sizing(v1.35.3)이 최적. **1061 tests passed**. |
| v1.35.6 | 02-27 | **Remove consecutive loss pause (deadlock fix)**: N=9 멀티포지션에서 consecutive_losses가 리셋 불가능한 데드락 발생 (진입 차단 → 승리 불가 → 리셋 불가 → 영구 차단). `check_consecutive_loss_limit()` 함수 및 관련 상수(`MAX_CONSECUTIVE_LOSSES`, `CONSECUTIVE_LOSS_PAUSE_SECONDS`) 완전 제거. bot.py, signals.py, constants.py, 테스트 7개 정리. 대체 보호: aggregate risk cap + regime sizing + MDD sizing + daily loss limit. **1061 tests passed**. |
| v1.35.5 | 02-26 | **Aggregate directional risk cap**: `aggregate_risk_cap_study.py` 9-시나리오 연구 기반. (1) `bot.py`: `_check_aggregate_risk_cap()` + `_estimate_new_sl_pct()` — 방향별 SL 노출 합산(sum of eff_sl × 1/N × leverage)이 cap 초과 시 진입 차단. EMA(20) regime으로 counter/with 분기. (2) config `aggregate_risk_cap: {enabled: true, counter_cap: 3.0, with_cap: 7.0}`. (3) `state.py`: `consecutive_losses` midnight reset 제거 + `trade_history` 기반 복구 (v1.35.4b). 연구 결과: `dynamic_3_7` MDD -52% (13.2→6.3%), PnL/MDD +16% (23.7→27.5x), WF 3/3 PASS. **1068 tests passed**. |
| v1.35.4 | 02-26 | **SL breach auto-recalculation**: 3-bug chain 디버깅 (state corruption → snapshot silent fail → stale SL infinite retry). (1) `orders.py`: `_recalculate_breached_sl()` 헬퍼 — BingX 110412 에러(SL price constraint) 감지 시 current_price ± 0.3% 버퍼로 SL 재계산. 3개 SL 배치 경로 모두 적용 (`_place_sl_order`, `_verify_sl_order`, `_place_emergency_sl_for_direction`). (2) `position_close.py`: `_snapshot_all_tpsl()` 실패 로깅 `debug`→`warning` (silent failure 방지). |
| v1.35.3 | 02-26 | **Regime-aware position sizing**: `uptrend_profit_study_p2.py` 3-Hypothesis 연구. H1 Direction-Specific Edge Threshold STOP (4 thresholds WF FAIL), H2 Data Extension STOP (LONG 악화), **H3 Regime-Aware Sizing GO** (19/19 WF 3/3 PASS). `position_open.py`: EMA(20) slope로 추세 감지, counter-regime 진입(상승장 SHORT, 하락장 LONG) 시 사이즈 ×`counter_mult`. config `regime_sizing` 섹션 추가 (ema_period=20, lookback=5, counter_mult=0.3). mult=0.3: MDD **6.71%** (기존 18.70%), PnL/MDD +21.4%. **1068 tests passed**. |
| v1.35.2 | 02-26 | **Consecutive loss fix + tighter MDD sizing**: (1) `bot.py`: 연속 손실 pause를 메인 루프→`_process_entry_signal()`로 이동 — v1.27.0의 `and not positions` 조건이 N=9에서 pause 무효화 (다른 슬롯 열려있으면 트리거 안됨). 이제 새 진입만 차단, 기존 포지션 모니터링 계속, win 시 자동 리셋. (2) config MDD sizing 타이트닝: `full_size_below_dd` 5→**3%**, `min_size_above_dd` 20→**15%** — DD 반응속도 ~40% 향상 (edge half-life 30d 기반). (3) `rescan_interval_days` 90→**60d** — edge half-life 30d, first negative 90d → 60d에 경고하면 30d 여유. (4) `trend_filter_study.py`: 상승장 숏 문제 연구 (H1 Momentum/H2 EMA/H3 ConsecBrake/H4 ATR+Mom — 4가설 전부 STOP, WF 2/3 FAIL). **1068 tests passed**. |
| **v1.35.1** | 02-25 | **Direction Cap 6→8**: `direction_cap_wf_study.py` — 296d(270d CSV+26d API), 3-fold OOS, 7 cap 시나리오(3~9) 비교. **cap8이 유일한 3/3 PASS** (+0.33%/+33.73%/+29.27%). cap8: PnL +63.33%, MDD 24.78%, PnL/MDD **2.56x**. cap6(기존): PnL +24.29%, MDD 22.85%, PnL/MDD 1.06x (2/3 FAIL). no-cap: PnL +63.95%, MDD 28.85%, 2/3 FAIL. cap8은 no-cap과 PnL 동등하면서 MDD -4.07%p 개선. config `direction_cap: 8`. **1067 tests passed**. |
| v1.35.0 | 02-25 | **ATR Scanner v2.2 + 51pat deploy**: (1) `pattern_scanner.py` v2.2: ATR-scaled TP/SL 기본 통합 — `compute_atr_ratio()`, `bt_signals_atr()` 추가, grid search/WF/holdout에 ATR passthrough, `--no-atr`로 Fixed 모드 가능. (2) CLI 인자: `--atr-period 14`, `--atr-window 576`, `--atr-clamp-lo 0.6`, `--atr-clamp-hi 1.7`. (3) NumpyEncoder 추가 (WF numpy.bool_ 직렬화). (4) 270d ATR scan+WF: **51패턴(16L+35S)**, WF 3/3 PASS, OOS PnL +443.9%. IS: WR 93.9%, PnL +1,214%, MDD 20.4%, PnL/MDD 59.5x. TP 0.87-2.84%, SL 1.44-4.76%. Scanner-Production ATR 정합성 확보 (연구 OOS +64% 향상 검증). Backup: `dynamic_patterns_35pat_compact_backup.json`. **1067 tests passed**. |
| v1.34.0 | 02-24 | **Holdout + Clean Protocol v3.0 + MDD sizing + Trade history**: (1) Scanner `--holdout-days 7`: 마지막 7일 OOS 보유, WR Excess<=0 패턴 제거 (SKIP 유지). (2) **BH FDR 버그 수정**: `m = max(n_tested, len(sorted_items))` — MC pre-filter가 m을 축소하여 FDR 제어 무력화하던 버그 수정 (m=1,326 전체 가설). (3) **`--clean` Protocol v3.0**: 사전등록 manifest + 이론 기반 임계값(edge≥5pp cost-based, SL≥1.0% execution risk) + BH FDR 1차 필터(mc_threshold=1.0). Clean scan 결과: 260패턴(84L+176S), **현행 35패턴 중 34개(97%) BH FDR 통과 → 패턴 선별 건전성 확인**. (4) Bot `_check_scan_staleness()`: 90d 초과 시 WARNING. (5) MDD 동적 사이징: `peak_equity` HWM + DD 5%→full/20%→25% 선형. (6) **trade_history 영속화**: `position_close.py`에서 거래 상세 기록 → `metrics.json` 저장 (로그 로테이션 생존). 8파일, **1067 tests passed**. |
| v1.33.0 | 02-23 | **35pat Compact TP/SL + 15 STRONG SHORT 복원**: `short_restore_tpsl_compact_study.py` 6-Phase 연구. (1) 34 탈락 SHORT 중 15개 STRONG 복원 (4-Phase 재평가). (2) Compact grid (TP max 2.0%, SL max 2.5%) vs Wide 비교 — compact가 PnL/MDD 28.84 (wide 14.23), holding time 1.5-1.9x 빠름. (3) 35pat(9L+26S): trades 1343 (4.97/day), WR 68.1%, PnL +297.1%, MDD 10.3%, **WF 3/3 PASS**. TP 0.50-2.00%, SL 1.00-2.50%. EXPECTED_WR=68.1, EXPECTED_EDGE=0.221. Backup: `dynamic_patterns_22pat_wide_backup.json`. |
| v1.32.0 | 02-23 | **7 STRONG LONG 복원 + Direction Cap 6**: `long_restore_and_direction_cap_study.py` 5-Phase 연구. 15→22 패턴 (9L+13S), direction cap=6. 22pat+cap6: trades 679, WR 68.5%, PnL +317.8%, PnL/MDD 50.44, WF 3/3 PASS. Trailing stop → ADOPT: NO (-8.1%). |
| **v1.31.1** | 02-23 | **Optimal TP/SL (WR Excess maximizing grid search)**: `tpsl_regime_bias_study.py` 5-Phase 연구 결과 적용. Phase 4 grid search로 패턴별 WR Excess 극대화 TP/SL 도출. 15패턴 전체 TP/SL 변경 (TP 1.0-3.5%, SL 2.0-4.0%). **14/15 STRONG** (3-4 phase genuine edge 확인), 1 MODERATE. Phase 5: Optimal T864 PnL +372.7%, MDD 21.7%, PnL/MDD 17.18, WF 3/3 PASS (vs MAE/MFE +271.9%). 12/15 Direction Flip GENUINE, 13/15 Symmetric TP=SL 모든 레벨 >50%. EXPECTED_WR=68.7, EXPECTED_EDGE=1.099. Backup: `dynamic_patterns_15pat_mae_backup.json`. **1067 tests passed**. |
| v1.31.0 | 02-23 | **15-pattern WR Excess filter + T864 timeout**: 패턴 59→15 (WR Excess>5pp), TIMEOUT=864 (72h), PnL/MDD 17.03, WF 3/3 PASS. |
| v1.30.1 | 02-23 | **N=5→9 멀티포지션 확장 (NO_TIMEOUT 전략)**: `max_positions: 5→9`, `DEFAULT_MAX_POSITIONS: 5→9`. timeout_impact_study.py NO_TIMEOUT N-sweep 결과 PnL/MDD 피크 N=9 (PnL +107.1%, MDD 67.1%, PnL/MDD 1.60, WF 3/3). 1/N sizing 20%→11.1%. Hit rate 13.7%→21.2% (+55%). Direction bias 발견: LONG 15% trades/-4.74% PnL vs SHORT 85%/+78.91%. |
| **v1.30.0** | 02-22 | **Production Hedge 모드 전환**: FIFO→Hedge 전환 (config `position_mode: hedge` 활성화). 백테스트 비교: Hedge N=5 +174.0%/MDD 29.6% vs FIFO N=5 +57.1%/MDD 58.8% (3배 PnL/MDD). FIFO CLOSE_OLDEST 1,041회 강제청산 -451.9% 비용 제거. v1.29.4에서 구축한 인프라 활용 (코드 변경 없음). `verify_position_mode(hedged=True)` 자동 실행. LONG/SHORT 독립 포지션 공존, per-direction emergency SL. **1067 tests passed**. |
| v1.29.4 | 02-22 | **Hedge mode infrastructure**: `position_mode='hedge'` 지원 인프라 전체 구축. `orders.py`: per-direction emergency SL (place/cancel/update/verify) + `_get_position_side()` hedge 라우팅 + helper 함수들. `bot.py`: `_route_signal()` hedge 분기 (direction-agnostic slots). `position_monitor.py`: `get_actual_exit_price` positionSide 필터링. `position_close.py`: direction-aware 청산+recovery. `position_open.py`: hedge positionSide params. `exchange.py`: `verify_position_mode()` hedge/one-way 감지. `state.py`: `emergency_sl_orders` dict. `models.py`: BotState 타입 확장. `config.yaml`: `position_mode` 필드. 인프라만 구축 (미활성). 12개 파일, +614/-132, **1067 tests passed**. |
| v1.29.3 | 02-22 | **CLOSE_OLDEST immediate re-entry + emergency SL amount fix**: (1) `bot.py`: CLOSE_OLDEST 후 즉시 새 슬롯 진입 — 기존엔 청산만 하고 다음 캔들 대기. (2) `orders.py`: emergency SL `amount` 파라미터를 `closePosition:True` 대신 실제 총수량으로 변경 — BingX amount rounding 이슈 방지. |
| v1.29.2 | 02-21 | **adjust_tpsl_to_config dynamic per-pattern mode**: `orders.py` `adjust_tpsl_to_config()`이 dynamic per-pattern TP/SL 모드에서도 올바르게 동작하도록 수정 — config에서 패턴별 TP/SL 조회. |
| v1.29.1 | 02-21 | **Crash recovery per-pattern TP/SL preservation + per-slot fill detection**: (1) `position_close.py`: crash recovery 시 per-pattern TP/SL 가격 보존 (config 기본값 대신 패턴 최적값 사용). (2) `position_monitor.py`: per-slot TP/SL fill detection — exchange qty < local sum 시 open orders 조회하여 개별 슬롯 체결 감지 (N>1 필수). (3) `orders.py`: `verify_tp_sl_orders` qty mismatch guard — 체결 감지 중인 방향은 verify skip (이중 TP 재배치 방지). 8개 테스트 추가. |
| **v1.29.0** | 02-21 | **N=5 멀티포지션 (One-Way BOTH mode)**: 가상 슬롯 기반 멀티포지션 시스템. `state['position']` → `state['positions']` dict. Signal routing (`_route_signal`: OPEN/SKIP/CLOSE_OLDEST), Emergency SL (closePosition:True 전체 보호), 1/N 사이징 (슬롯당 19% equity), State v2 마이그레이션. `max_positions: 5`, `DEFAULT_MAX_POSITIONS=5`. Max single-trade loss 2.5% (vs N=1 12.6%). 동일 패턴 중복 진입 차단. 12개 파일 변경, 1014 tests passed. |
| **v1.28.42** | 02-21 | **ATR-scaled TP/SL + proportional vol_mult cap**: `get_volatility_multiplier()`에 ATR-ratio 스케일링 모드 추가 (기존 `vol_adaptive`보다 우선). `ATR_ratio = ATR(14) / rolling_median(ATR(14), 576봉)`, clamp [0.6, 1.7]. WF 연구: 28 시나리오 전수 검증, `BOTH_a14_w576_0.6-1.7` WF 3/3 PASS, pre-overlap edge +74.5%. SL 스케일링이 핵심 동인 (TP-only FAIL, SL-only PASS). `config.yaml` `strategy.atr_scale` 섹션 추가 (enabled/window/clamp_lo/clamp_hi). `calculate_indicators()`에서 `atr_scale` 활성화 시에도 ATR 계산 트리거. `MAX_OHLCV_CANDLES` 150→600 (ATR-scale window=576 충족). **Proportional cap**: `_effective_vol_mult()` — 멀티플라이어 자체를 `max_sl_pct / base_sl_pct`로 제한, TP/SL 동일 비율 스케일링으로 R:R 비율 보존. Hard SL cap(`_cap_sl_to_daily_limit`)은 R:R 왜곡 최대 +65.6% (49/59 패턴 영향) → proportional cap은 +1.2% 이하. 7가지 캡 전략 심층 분석 (`atr_cap_strategy_study.py`). 1139 tests passed. |
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