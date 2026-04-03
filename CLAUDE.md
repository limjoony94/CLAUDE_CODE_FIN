# CLAUDE_CODE_FIN - BTC 5분봉 패턴 트레이딩 봇

> **Version**: v1.71.0 | **Bot**: Pattern 5m (111패턴, 51L+60S, Edge18pp+NeutralWindow+ATR Scanner+Holdout+MDD+N7DC7+MomGuard1.5%15m1h+NposScanner+CascadeOFF+PreCascadeOFF+AggRisk10_15+ATRClamp10_15+TO576+MassCloseGuard+ExitClassify+PatternRecovery+RegimeFix+DupGuard+CodeAuditFix+MFEMedianTP+OrphanPrevention+PosTrackFix+MedianFallback+EmgSlRace+NaPrevent+EmgSlUpdate+CcxtTypeAdopt+SoftDelete+ScannerVolCap+VolAdaptOFF+TrailOFF+TPDecayOFF+TPLimitMaker) | **Updated**: 2026-04-03

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
| 데이터 | `data/btc_5m_270days_reclassified.csv` (303일, Ground Truth, 파일명은 레거시) |
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
- **알림 기준**: 연속손실 ≥3, 일일손실 ≤-13%, MDD ≥25%, WR <50% | EXPECTED_WIN_RATE=67.4 (v1.56.1 clean TP+SL, 129t) / post-fix 83.0% (53t)
- **Clean baseline**: `results/pattern_5m_baseline_post0305.json` — pre-03-05 오염 데이터 제외한 정확한 기대치
- **주의**: LONG WR 0% (03-05~08, 15t) — BTC 하락 레짐 편향. 소표본이므로 지속 모니터 필요
- 상세: [docs/agent-guides.md](docs/agent-guides.md)

---

## 🧬 시스템 본질 (2026-03-15 심층 분석 결론)

> **"BTC 5분봉 변동성 수확 시스템 (Volatility Harvester)"**
>
> 패턴이 가격 방향을 예측하는 시스템이 **아님**. TP/SL 비대칭(TP 1.25% < SL 3.43%) + 메커니즘 스택(Cascade SL, Timeout, TP Decay)이 BTC 5분봉의 변동성 구조를 체계적으로 수확하는 시스템.

### 실증 근거 (mechanism_stress_test + regime_robustness_stress)

| 검증 | 결과 | 의미 |
|------|------|------|
| 랜덤 진입 vs 패턴 진입 | 랜덤이 패턴의 86% 성과 | 패턴의 genuine 기여 ~14% |
| 방향 반전 (LONG↔SHORT) | 반전해도 **100% 윈도우 수익** | 방향 예측은 수익 원천 아님 |
| 레짐별 성과 | BULL/BEAR/SIDEWAYS 전부 수익 | 레짐 무관 작동 |
| Hurst 지수 | 0.58 (약한 trending) | Mean-reversion 시장이 아님에도 작동 |
| AC vs PnL 상관 | r=-0.28 (p=0.08, ns) | 자기상관이 성과를 예측하지 못함 |
| AC 추세 | +0.0001/일 (p=0.028) | Mean-reversion 약화 추세이나 성과에 무관 |

### 수익의 실제 원천

1. **TP < SL 비대칭**: 작은 TP(1.25%)를 빈번히 체결, 큰 SL(3.43%)은 드물게 피격 → 높은 WR
2. **Cascade SL**: SL 피격 시 동일 방향 포지션 SL 85% 축소 → 연쇄 손실 차단
3. **Timeout + TP Decay**: 노출 시간 제한 + TP 조기 체결 유도 → 슬롯 회전 가속
4. **신호 밀도**: 131 패턴 = 다양한 진입점 → 9슬롯 포트폴리오 효율적 활용

### 라이브 모니터링 기준

| 상태 | 기준 | 행동 |
|------|------|------|
| GREEN | WR > 60%, 정상 거래 빈도 | 유지 |
| YELLOW | WR 55-60% 7일 지속 | 주의 관찰 |
| RED | WR < 55% 7일 지속 | 포지션 축소 검토 |
| HALT | WR < 50% 14일 지속 | 봇 중지, 미시구조 변화 분석 |

### 현실적 기대치

| 항목 | 비관적 | 중립 | 낙관적 |
|------|--------|------|--------|
| WR | 65% | 72% | 80% |
| 월간 PnL | +5% | +15% | +30% |
| MDD | 15% | 10% | 5% |

> 주의: IS 83% WR, OOS +339.5%는 **백테스트 상한**. 라이브 성과는 구조적으로 하회 (슬리피지, API 지연, 미시구조 변동). 랜덤 진입 OOS +333%가 **실질 baseline**.

---

## 📊 현재 전략: Pattern 5m v1.71.0

### 핵심 파라미터

| 파라미터 | 값 |
|---------|-----|
| Entry | 3-candle pattern match (12-type) |
| TP/SL | **Per-pattern MFE Median TP** (TP 0.84-2.48%, SL 1.44-4.56%, MFE median from exc_stats, v1.67.0) |
| Classification | Ground Truth (HAMMER/INV_HAMMER 우선순위 수정) |
| Leverage | **Fixed 3x** |
| Timeframe | 5m |
| **Max Positions** | **7** (virtual slots, 1/N=14.3% sizing, **mixed-direction** in Hedge) — v1.71.0: 5→7 (lockout 12%→6%, OOS 동등, +40% trades) |
| **Position Mode** | **Hedge** (LONG/SHORT 독립 포지션) |
| Pattern Source | **Dynamic** (results/dynamic_patterns.json, Neutral window ±1%) |
| Discovery | **MAE/MFE + ATR-scaled** (TP=MFE percentile, SL=MAE percentile, ATR scanner v2.4) |
| Scanner MAX_BARS | **288** (24h) |
| Quality Filter | **Edge>=18pp + WR>=60% + SL>=1.0% + MC<0.01 + min_trades>=25 + Holdout 7d** |
| Patterns | **111** (51L + 60S), ATR scanner v2.4 + Neutral window + WF 5/5 PASS (v1.68.0 rescan, 325d) |
| **Direction Cap** | **7** (= max_positions, DC blocking 제거) — v1.71.0: DC=N으로 blocking 0건 |
| **Position Timeout** | **576 bars (48h)** — v1.69.1: 288→576 (timeout_cascade_study: TP decay replaces timeout, OOS +42.7%) |
| **ATR Entry Clamp** | **[1.0, 1.5]** — v1.69.1: expand-only (no low-vol shrinkage, OOS +12.4%) |
| **Vol Adaptation** | **OFF** — v1.69.1: mid-trade SL adjustment 비활성 (SL clustering 방지) |
| **Trail Stop** | **OFF** — v1.69.0: N-pos에서 R:R 0.11로 역효과 |
| **TP Decay** | **0.9975^bars** (HL 23.1h, bar 0부터, update 6bars) |
| **Cascade SL** | **OFF** — v1.70.0: 98% tighten DISABLED (SL 동일가격 집중 → 클러스터. Live 3d -102.64%. Natural cluster 8.9% max -7.9%) |
| **Preemptive Cascade** | **OFF** — v1.70.0: preemptive DISABLED (ALL SLs를 current±0.4%에 집중 → 동시 체결. 7 clusters/3d 원인) |
| Risk | Daily loss **13%**, **aggregate risk cap** (counter **10%**/with **15%**) |

### v1.67.0 검증 요약

- **TP Mode: MFE Median** — 각 패턴의 MFE(최대유리편향) 중앙값을 TP로 직접 사용
- **자유 파라미터 0개** (데이터 직접 사용, 튜닝 없음) → 과적합 위험 최저 (score 30.6)
- **IS P/M 149.3x**, OOS-5 **+385%** (baseline ×0.72: 134.8x / +380%)
- **TP 범위**: 0.84-1.70% (MFE median), SL: 1.44-5.95% (변경 없음)
- **WR Margin +36.6pp** (baseline +36.7pp와 동등)
- **Effective median scale ≈ 0.71** 이나 패턴별 적응 범위 0.37~1.44
- tp_calibration_study + overfit_diag: 12개 전략 중 과적합 스코어 최저, OOS5 최강
- Rollback: `tp_mode: scale` + `tp_scale_factor: 0.72` (v1.61.0 legacy)

### WF OOS 검증 (v1.67.0, MFE Median TP, 5-fold Expanding Window, N-pos)

| Fold | OOS PnL |
|------|---------|
| F1-F5 | +63%, +58%, +72%, +123%, +69% |
| **Total** | **+385%** |

**Verdict: 5/5 PASS** | IS P/M: 149.3x | Overfit score: 30.6 (MODERATE, lowest)

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
| **Direction Cap (v1.36.1→v1.65.0)** | **Max 6 same-direction positions** — v1.65.0: 7→6 (preemptive_param_crossval COMBO_B: OOS +410.7%, cluster exposure -14%) |
| **Holdout Validation (v1.34.0)** | **Scanner --holdout-days 7** — 마지막 7일 OOS 검증, WR Excess<=0 패턴 제거 |
| **Scan Staleness (v1.34.0)** | **dynamic_patterns.json 90일 초과 시 WARNING** — 봇 시작 시 자동 체크 |
| **MDD Sizing (v1.34.0)** | **DD 5%→full, 20%→25% 선형 축소** — peak equity HWM 기반 동적 포지션 사이징 |
| **Trade History (v1.34.0)** | **거래 상세 영속화** — metrics.json에 전체 거래 이력 저장 (로그 로테이션 생존) |
| **ATR Scanner Integration (v1.35.0)** | **Scanner v2.2에 ATR-scaled TP/SL 기본 통합** — Scanner-Production 정합성 확보, `--no-atr`로 Fixed 모드 가능 |
| **Aggregate Risk Cap (v1.35.5→v1.65.0)** | **방향별 SL 노출 합산 제한** — counter **10%**, with **15%** cap (v1.65.0: 999→10/15 재활성화) |
| **Neutral Window Discovery (v1.36.3)** | Scanner가 start≈end price (±1%) 최장 구간 자동 발견. `--no-neutral`로 비활성화 |
| **Momentum Guard (v1.46.0)** | BTC >1.5%/15min 변동 시 역방향 진입 1h 차단. config `momentum_guard` |
| **Emergency SL Overhaul (v1.36.6)** | `closePosition:true` + 매 루프 `_ensure_emergency_sl_exists()` 선제 검증 |
| **N-pos Scanner (v1.38.1 default)** | Scanner에 N=9/compound/dir_cap/agg_risk/momentum 통합. `--no-npos`로 legacy |
| **Scanner Cascade SL (v1.54.0)** | Scanner N-pos에 Cascade SL 구현. `--no-cascade`로 비활성화. IS WR -15pp, MDD -32%, PnL/MDD +57% |
| **Cascade SL Tightening (v1.45.0→v1.68.0)** | SL 피격 시 동일 방향 SL 거리 ×0.02 (**98% 축소**, v1.68.0). `_sl_price_original`(진입 시 원본 SL)에서 거리 계산. config `cascade_sl_tightening` |
| **Pre-emptive Cascade (v1.65.0→v1.68.0)** | **방향별 미실현 손실 > 2% 시 SL 선제 축소(98%)** — v1.68.0: threshold 4→2%, tighten 95→98%, ALL 포지션 loss 계산 (cascaded 포함). `_sl_price_original`(진입 시)에서 거리 계산. config `preemptive_cascade` |
| **Scanner Regime Fix (v1.56.0)** | Scanner DEFAULT_REGIME_MULT 0.3→1.0 — production v1.42.0에서 비활성화한 Regime Sizing을 scanner에서도 정합. IS PnL/MDD +83%, OOS +60% |
| **Duplicate Trade Guard (v1.56.1)** | `record_closed_position`에 중복 기록 방지 가드 추가 + `pattern_name` 필드 우선 사용. N/A 오염(22건) 정화 완료. TP+SL WR 61.6→67.4%, gap 11.2→5.4pp |
| **Code Audit Fixes (v1.56.2)** | (1) Cascade SL Place-first/Cancel-after (보호 갭 제거) (2) SL 실패 시 Emergency SL 즉시 호출 (3) Momentum cooldown state 영속화 (4) datetime 파싱 방어 (5) Hardcoded 300s→CANDLE_DURATION_MS (6) Always-truthy 수정. 1061 tests ALL PASSED |
| **MFE Median TP (v1.67.0)** | **Per-pattern MFE median as TP** — 각 패턴의 MFE 중앙값을 TP로 직접 사용. 자유파라미터 0개, 과적합 위험 최저(score 30.6). IS P/M 149.3x, OOS5 +385%, WF 5/5 PASS. Effective scale 0.37~1.44 (median 0.71). config `tp_mode: mfe_median`. Rollback: `tp_mode: scale` + `tp_scale_factor: 0.72` |
| **Orphan Prevention (v1.59.0)** | **3-layer defense against transient API zero-contract** — (1) ALL closure detections trigger 1s delay + fresh re-verify (removed ≥3 threshold) (2) Inter-direction exchange_map rebuild after closures (3) Post-closure orphan detection + auto-recovery. Root cause: BingX API transient 0-contract during order processing. 1073 tests ALL PASSED |
| **N/A Pattern Prevention (v1.59.4)** | **4-layer N/A cascade 방지** — (1) `_restore_none_pattern_slots()`: crash recovery 후 None 슬롯 로그 기반 복원 (2) `record_closed_position` last-resort log recovery (3) `cancel_remaining_orders` 3회 retry (4) Recovery 전 stale order cleanup. Root cause: BingX averaged entry → price matching 실패 → None cascade |
| **Emergency SL Update Fix (v1.59.5)** | **Cancel-first/Place-after + EXCHANGE_MANAGED 해소** — (1) `update_emergency_sl` cancel-first 패턴 (closePosition=true 1-per-direction) (2) `_find_close_position_order` 헬퍼: open_orders에서 실제 closePosition 주문 검색 (3) `_verify_emergency_sl_for_direction` EXCHANGE_MANAGED 시 실제 주문 adopt 또는 cancel-replace (4) `_cancel_emergency_sl_for_direction` EXCHANGE_MANAGED 시 실제 주문 찾아 취소. 1105 tests ALL PASSED |
| **CCXT Type Adopt Fix (v1.59.6)** | **CCXT STOP_MARKET→'market' 정규화 대응 + stopPrice 정합** — (1) `_find_close_position_order` type 매칭: CCXT `o.type` + raw `info.type` 이중 체크 (CCXT가 STOP_MARKET→'market'로 정규화하여 매칭 실패 → EXCHANGE_MANAGED 무한 루프 원인) (2) `_verify_emergency_sl_for_direction` adopt 시 `info.stopPrice` 우선 사용 (CCXT `stopPrice=None` 반환 quirk 대응) (3) `_place_emergency_sl_for_direction` adopt 로그에서도 동일 수정. Root cause: CCXT→BingX 타입 정규화 불일치로 `_find_close_position_order` 항상 None 반환 → adopt 불가 → EXCHANGE_MANAGED 영구화. 1107 tests ALL PASSED |
| **Soft-Delete Mass Closure (v1.60.0)** | **Two-cycle confirmation으로 false closure 근본 방지** — (1) `_resolve_pending_close_slots()` 신규: 이전 사이클에서 pending 마킹된 슬롯을 exchange 재확인 (0→confirmed closure, position exists→false alarm 복원) (2) Mass closure 감지 시 즉시 삭제 대신 `_pending_close=True` 마킹 + `save_state()` (3) `position_open.py`: pending 슬롯 진입 카운트 제외 (4) `bot.py`: pending 슬롯 direction cap/aggregate risk/모니터링 제외. Root cause: BingX transient 0-contract → `del positions[slot_id]` → `.bak` 덮어쓰기 → 원본 데이터 영구 유실 → orphan recovery가 동일 패턴 할당 → 동일 TP/SL. 1111 tests ALL PASSED |
| **Exp-Decay TP (v1.63.0)** | **TP × 0.997^bars_held (half-life 19.25h, bar 0부터)** — 진입 즉시부터 TP가 entry 방향으로 지수 감소. 슬롯 회전 가속 (+40% trades), OOS +10.9% 개선. WF 3/3 PASS. config `tp_decay.decay_rate`. IS PnL +457.5%, MDD 5.15%, PnL/MDD 88.8x. update_interval 6bars(30min) |
| **DISABLED (5개)** | Regime Sizing, Adaptive Leverage, Equity Curve, Correlation-Aware, Loss Burst Brake — 각 `enabled: true`로 재활성화 가능. Entry Optimization은 ROLLED BACK (코드 제거) |

### Dynamic Pattern Selection (v1.27.3)

`pattern_source` 설정으로 정적(constants.py) 또는 동적(scanner 출력) 패턴 세트 선택 가능.

| 모드 | 설정값 | 패턴 소스 | TP/SL |
|------|--------|-----------|-------|
| Static (fallback) | `pattern_source: static` | constants.py 51패턴 | Per-pattern 최적화 |
| Dynamic PP | `pattern_source: dynamic` + `tp_sl_mode: per_pattern` | results/dynamic_patterns.json | PP grid search |
| **Dynamic ATR + MFE Median (현재)** | `pattern_source: dynamic` + `tp_sl_mode: per_pattern` + `tp_mode: mfe_median` | results/dynamic_patterns.json | **MAE/MFE + ATR-scaled, TP=MFE median** |

**Scanner CLI 사용법** (v2.4):
```bash
cd bingx_rl_trading_bot
# 기본 (neutral + ATR + N-pos, v1.38.1~ default)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --wf-folds 3 --holdout-days 7
# Legacy 1-pos 모드 (빠른 반복용)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --wf-folds 3 --holdout-days 7 --no-npos
# 주요 옵션: --no-neutral, --no-atr, --neutral-tol 2.0, --atr-clamp-lo 0.5 --atr-clamp-hi 1.5, --n-slots 5 --direction-cap 4
```

**현재 적용**: MAE/MFE + ATR scanner + Neutral window (edge>=18pp, MC<0.01, --wf-folds 3, --holdout-days 7) → **111패턴 (51L+60S)** (v1.68.0, 325d data, 2026-03-17 scan) + **tp_mode: mfe_median** (v1.67.0).
WF N-pos 5/5 PASS, OOS +660.5% (parity scanner: timeout PnL + TP decay). IS PnL/MDD 151.6x. TP 0.84-1.70% (MFE median), SL 1.44-5.95%. Cascade 98%, Pre-emptive 2%/98%. Neutral window ±1% (259d). ATR: a14/w576/clamp[0.5,1.5]. Data: 325d (extended 2026-03-27).

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
├── data/                           # 시장 데이터 CSV (303일 Ground Truth)
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
| **v1.71.0** | 04-03 | **N=7 DC=7 + TP LIMIT maker fee (Lockout Reduction + Fee Optimization)** ← 현재. 21-cycle critical evaluation continuation. (1) N=5→7: lockout 12.2%→6.0% (절반), trades +40% (3.8→5.2/day). Scanner OOS +39.7% (N5 +42.0%와 noise 범위 -2.3pp). WF 4/5 동일. MDD +2.8pp (26.7→28.3%). (2) DC=5→7=N: DC blocking 완전 제거. (3) TP LIMIT order: TAKE_PROFIT_MARKET→regular LIMIT for maker fee (0.02% vs taker 0.05%). 650 TP × 0.03% × 3x / N = +11.7% PnL (BT에 미반영 보너스). Fallback to TAKE_PROFIT on error. (4) Methodology critique: pattern direction 52% accuracy × TP/SL asymmetry = WR 74.8%. 5 alternative signals ALL inferior. System = pattern timing + direction + TP/SL asymmetry. (5) Edge health: 10/10 months > BE, decay -0.5pp/month, 17mo remaining. (6) Stress test: max single event -10.5% (7 SL), liquidation at 33% move. 1117 tests ALL PASSED |
| v1.70.0 | 04-02 | CASCADE OFF + N=5 DC=4 (Cluster Elimination Restructure). 7-cycle critical evaluation (20+ studies, 4 background agents, corrected discrimination test). (1) CRITICAL: Cascade 98% tighten concentrates ALL SLs at current±0.4% → 100% cluster rate. Live 3d: 7 clusters -131.5%, non-cluster +116.5%. Root cause: 98% keep ratio always triggers 0.4% fallback → identical SL prices. (2) CASCADE OFF: reactive + preemptive both disabled. Natural cluster rate 8.9%, worst -7.9% (vs cascade -29%). (3) N=9→5: slot contention 감소, WR 58.5→59.4%. N=7/9 breakeven/negative. (4) DC=6→4: DC≥4에서 수익 급등 (IS +43.5% vs DC=3 +17.8%). DISCRIMINATING p=0.00 (1/20 shuffled positive). (5) Scanner OOS +31.3%, WF 4/5 PASS. MDD 21.3%. (6) Slippage margin 13bp (breakeven at +15bp extra). Monthly Sharpe 0.23, 5/10 positive months. Honest expectation: +18-26%/year after slippage. (7) Previous conclusions corrected: N=1 +767% = timeout DROP artifact; Partial Close +636% = dropped position artifact. Config-only change, no production code modified. 1117 tests expected PASS |
| v1.69.1 | 03-31 | Cascade entry-based fix + 17-study validation. (1) CRITICAL BUG FIX: cascade SL이 config tighten_pct(98%)를 무시하고 하드코딩 10% keep 사용 → entry-based + _sl_price_original로 복원 (BT R:R 4.84 vs Live 0.66 근본 원인). (2) Price validity: entry-based SL이 current price 넘어가면 current±$30 fallback. (3) Trail OFF: N-pos R:R 0.11로 역효과. (4) vol_adapt OFF: SL clustering 방지 (03-29 6건 동시 SL -27.82%). (5) ATR clamp [0.5,1.5]→[1.0,1.5]: expand-only. (6) Preemptive 2.0→1.5%: OOS +30.4%. (7) Timeout 288→576: decay가 대체. 17개 연구 + adversarial 검증: holdout +56.9%, bootstrap 0/1000 음수, 파라미터 ±10% 최악 +451%. WF 5/5 PASS, OOS +716%. MTM MDD 1.95-2.28%. 1117 tests ALL PASSED |
| v1.69.0 | 03-31 | Trail OFF + Cascade/Preemptive re-enable + Vol_adapt OFF. 7-study validated. Cascade OFF→ON: OOS -91%→+440%. |
| v1.68.0 | 03-28~29 | Market-Close Cascade (MCC) + Harness + 5 code fixes. Cascade 3단계 시도 후 MCC로 전환: (1) Entry-based SL tightening: 100% SKIP. (2) Current-price-based: -27.81% 피해. (3) MCC: SL exit시 같은 방향 전체 시장가 청산 (SL 수정 대신). BT sim +3999% WF. Live 검증 대기 중. (1) Config: cascade_tighten_pct 95→98, preemptive_cascade_pct 3→2, preemptive_tighten_pct 95→98 (23-iter adversarial harness, STRONG GO 8.2/10, bootstrap CI [+83%,+121%]). (2) Fix A: `_sl_price_original` 진입 시 설정 — v1.67.1 fix 완성. (3) Fix B: Pre-emptive loss 계산에 ALL 포지션 포함 (0건→trigger 가능). (4) Fix C v2: OPP_SIGNAL 시간 윈도우 카운터 (window_bars=4, 20min) — no-reset 과도 발동 수정. (5) Fix F: TP decay 실패 시 old TP 유지. (6) Fix G: Recovery 포지션 _sl_price_original 설정. Scanner parity: timeout PnL + TP decay + vol_adapt. Strategy Harness 시스템 구축 (adversarial_evaluator_v2, loop_evaluator, STRATEGY_LOOP.md). 111pat (51L+60S). WF 5/5 PASS, OOS +660.5%. 1111 tests ALL PASSED |
| v1.67.1 | 03-25 | **Cascade SL original distance fix**. Reactive/pre-emptive cascade가 `original_sl_distance`를 vol_adapt 이후 SL에서 계산하여 cascade SL이 entry 근처에 배치 → 가격 유효성 검증 SKIP → cascade 무력화 버그 수정. `_sl_price_original` (진입 시 원래 SL) 우선 사용. 시뮬레이션 차이 +0.3% (무), 라이브에서 cascade SKIP 비율 감소 기대. sim_production_parity_study + realistic_sim_study + cascade_tighten_sweep + overfit_check: 과적합 아님(FP=0, IS-OOS gap 동일). 1111 tests ALL PASSED |
| v1.67.0 | 03-24 | **MFE Median TP + OPP_SIGNAL Exit**. (1) `tp_mode: mfe_median`: 각 패턴의 MFE 중앙값을 TP로 직접 사용 — 자유파라미터 0개, 과적합 위험 최저(score 30.6). IS P/M 149.3x, OOS5 +385%. 패턴별 적응적 scale 0.37~1.44 (uniform ×0.72 대체). tp_calibration_study 12전략 비교 + overfit_diag 정량 평가. (2) OPP_SIGNAL Exit (v1.66.0): 2회 연속 반대 신호 + 수익>0.1% → 시장가 청산. opp_partial_close_study: 100% 전량청산이 50%/75%/25% 부분청산 대비 OOS 우위 (+18~27%). 1111 tests ALL PASSED |
| v1.65.0 | 03-15 | **Pre-emptive Cascade SL + 파라미터 재최적화**. (1) `_apply_preemptive_cascade()`: 방향별 미실현 손실 > 4% 시 SL 선제 95% 축소 — SL 피격 전 클러스터 방어 (sl_cluster_defense: IS P/M 127.3x, R:R 1.29, WF OOS +410.7% vs baseline +308.4%). (2) DirCap 7→6 (동시 방향 노출 감소). (3) AggRisk counter 999→10%, with 999→15% 재활성화. (4) 13개 심층 연구 기반 시스템 본질 재정의: "BTC 5m 변동성 수확기" (랜덤 진입 86% 성과, 방향 반전 100% 수익). `claudedocs/system_deep_analysis_20260315.md` 참조. 1111 tests ALL PASSED |
| v1.63.1 | 03-14 | Decay Rate 0.997→0.9975 (HL 23.1h). 10-point sweep 최적화. OOS +339.5%. Option B (TP→SL) 열위 확인. Config `decay_rate: 0.9975` |
| v1.63.0 | 03-14 | Exp-Decay TP (0.997^bars). Bar 0부터 TP 지수 감소 — `TP_dist × 0.997^bars_held` (half-life 19.25h). IS PnL +457.5%, OOS +332.5% vs baseline +299.7%. WF 3/3 PASS |
| v1.62.0 | 03-14 | Time-Decay TP (linear_144). 12h(144bars)부터 TP를 선형 감소, 24h(288bars)에 원본의 50%까지. OOS +334.4% (+11.6%). → v1.63.0에서 exp_decay로 교체 |
| v1.61.0 | 03-13 | **TP Scale Factor 0.5→0.72 (R:R Compensation Fix)**. Live TP×0.5 R:R=0.66 → WR margin +2.1pp (위험, SL 1회 복구에 TP 1.5회 필요). tp_factor_deep_study: ×0.72 IS PnL/MDD 106.8x, OOS +276.8%, R:R 1.009 (1 SL≈1 TP), BE WR 49.8%, WR margin +27.0pp. Config `tp_scale_factor: 0.72`. TP range: 0.61-2.02% (원본×0.72). Production 코드 변경 없음 (config only) |
| v1.60.1 | 03-13 | Scanner _effective_vol_mult Cap (Production Parity). Scanner N-pos 백테스트에 production `_effective_vol_mult` 캡 추가 — `min(r, MAX_DAILY_LOSS_PCT / LEVERAGE / sl_pct)`. 6곳 적용: (1) `bt_signals_atr()` 1-pos (2) `_check_exit_npos()` N-pos exit (3) Cascade SL ATR ratio (4-5) Aggregate risk 기존/신규 포지션 (6) 상수 `MAX_DAILY_LOSS_PCT=13`. Production은 이미 이 캡 적용 중이나 scanner에는 없어서 6/131 패턴(base_sl>4.333%)에서 IS 불일치. 적용 후: IS WR 77.8→83.5%, PnL +458→+449%, MDD 4.05→4.93%, PnL/MDD 113.0→91.1x. WF 3/3 PASS (OOS +299.1% vs +306.0%). Production 코드 변경 없음 (scanner only). 1111 tests ALL PASSED |
| v1.60.0 | 03-13 | Soft-Delete Mass Closure (Two-Cycle Confirmation). BingX transient 0-contract 응답에 의한 false closure 근본 방지. (1) `_resolve_pending_close_slots()` 신규 — 이전 사이클 pending 슬롯을 exchange 재확인 (0→confirmed, exists→false alarm 복원) (2) Mass closure 감지 시 `_pending_close=True` 마킹 (즉시 삭제 대신) (3) `position_open.py` pending 슬롯 진입 카운트 제외 (4) `bot.py` pending 슬롯 direction cap/aggregate risk/모니터링 제외. Root cause: transient 0-contract → `del positions[slot_id]` → `.bak` 덮어쓰기 → 원본 데이터 영구 유실 → orphan recovery가 동일 패턴 할당 → 동일 TP/SL (3 LONG slots 70778.8/67081.7 현상). 1111 tests ALL PASSED |
| v1.59.6 | 03-13 | CCXT Type Adopt Fix. `_find_close_position_order()` CCXT type 정규화 대응: CCXT가 STOP_MARKET→`'market'`로 정규화하여 `type in ('STOP_MARKET','STOP')` 매칭 항상 실패 → `info.type` 이중 체크 추가. `_verify`/`_place` adopt 시 `info.stopPrice` 우선 사용 (CCXT `stopPrice=None` quirk). Root cause: EXCHANGE_MANAGED 무한 루프의 실제 원인 — 주문은 거래소에 존재하지만 CCXT 타입 불일치로 발견 불가 → adopt 불가 → 영구 EXCHANGE_MANAGED. Live 교차검증: 20 open orders 전부 state와 일치 확인. 1107 tests ALL PASSED |
| v1.59.5 | 03-12 | Emergency SL Update Fix (Cancel-first + EXCHANGE_MANAGED 해소). (1) `update_emergency_sl()` cancel-first/place-after 패턴 전환 — BingX closePosition=true 1-per-direction 제약으로 place-first 시 110406 에러 → 가격 업데이트 불가. (2) `_find_close_position_order()` 헬퍼 추가 — open_orders에서 실제 closePosition 주문 검색. (3) `_verify_emergency_sl_for_direction` EXCHANGE_MANAGED 무한 루프 해소 — 실제 주문 찾아 adopt(가격 일치) 또는 cancel-replace(가격 불일치). (4) `_cancel_emergency_sl_for_direction` EXCHANGE_MANAGED 시 실제 주문 찾아 취소. Live 검증: emergency SL 정상 업데이트 확인. 1105 tests ALL PASSED |
| v1.59.4 | 03-12 | N/A Pattern Prevention + Orphan Order Retry. 4-layer N/A cascade 방지: (1) `_restore_none_pattern_slots()` — crash recovery 후 None-pattern 슬롯을 로그에서 복원 (Phase 5) (2) `record_closed_position` last-resort log recovery — trade_history N/A 오염 방지 (3) `cancel_remaining_orders` 3회 retry with backoff — 네트워크 에러 시 orphan 주문 방지 (4) `recover_position_to_state` stale order cleanup — recovery 전 잔류 TP/SL 주문 정리. Root cause: BingX averaged entry price defeats price-matching → pattern=None → .bak 오염 → 연쇄 N/A. 재시작 시 4개 None 슬롯 즉시 복원 + per-pattern TP/SL 재조정 확인. 1101 tests ALL PASSED |
| v1.59.3 | 03-12 | Emergency SL Race Condition Fix. `update_emergency_sl()`에서 place-first/cancel-after 패턴의 경쟁 조건 수정. 새 SL 배치 시 110406(already exists) → EXCHANGE_MANAGED 설정 → old 주문 취소 → 보호 없음 버그. Fix: 110406 시 old order가 "already existing" 주문이므로 취소하지 않고 old_id 유지. Live에서 SHORT emergency SL 누락 확인 (개별 SL만으로 커버 중). 1076 tests ALL PASSED |
| v1.59.2 | 03-12 | Median TP/SL Fallback. Recovery 포지션(pattern=None)이 config defaults(tp=1%, sl=1%)를 사용하여 실제 SL 범위(1.44~5.95%) 대비 위험하게 타이트한 문제 수정. (1) `calculate_tp_sl()`에 `_get_median_tpsl_fallback()` 추가 — per-pattern dict의 median TP/SL 사용 (TP=0.855%, SL=3.48%) (2) `_adjust_single_position_tpsl()`에서 pattern=None 시 `return False` 제거 — 재시작 시 median fallback으로 조정 가능. 거래소 교차검증: 4슬롯 9주문 전부 정합 확인. 1073+ tests ALL PASSED |
| v1.59.1 | 03-12 | Position Tracking Fix + SL×1.1 Revert. (1) Duplicate trade guard에서 TP/SL 주문 미취소 → orphan order 잔류 → ghost closure 연쇄 버그 수정 (`cancel_remaining_orders` 추가) (2) Orphan detection에서 stale snapshot 사용 → 첫 루프 slot 제거 후 재감지 → 중복 recovery 버그 수정 (`state.get('positions')` 재조회) (3) Duplicate recovery guard 추가 (`recovered=True` 슬롯 존재 시 동일 방향 재복구 차단) (4) SL×1.1 revert (corrected WF NON-DISC: +429.7% vs SL×1.0 +430.0%). Root cause chain: Cascade SL mass closure → dup guard skipped order cancel → orphan TP filled → ghost slot removal → recovery created duplicate. 1073 tests ALL PASSED |
| v1.59.0 | 03-12 | Orphan Prevention 3-layer defense. Root cause: BingX API transient 0-contract response → 2-slot direction이 ≥3 mass closure guard 미발동 → false closure → TP/SL 취소 → orphan. Fix: (1) ALL closures 1s delay + fresh re-verify (≥3 조건 제거) (2) Inter-direction exchange_map rebuild (3) Post-closure orphan detection + auto-recovery. 1073 tests ALL PASSED |
| v1.57.0 | 03-12 | TP Scale Factor 0.5. N-pos 슬롯 회전 최적화: Scanner MAE/MFE TP를 ×0.5 축소. IS PnL/MDD 88.5→113.0x (+28%), OOS +206→+306% (+49%). 10/10 MC wins, DISCRIMINATING, Cascade-INDEPENDENT. `tp_scale_factor: 0.5` in config |
| v1.56.2 | 03-12 | Code Audit 전수점검 + 7건 수정. (1) `update_single_sl` Place-first/Cancel-after (Cascade SL 보호 갭 제거) (2) SL 실패 시 Emergency SL 즉시 호출 (3) Momentum cooldown `save_state` 영속화 (4) `datetime.fromisoformat('')` 방어 (5) `except Exception: pass`→로깅 (6) always-truthy `or {}` 수정 (7) Hardcoded 300s→`CANDLE_DURATION_MS//1000`. 교차검증으로 57건 중 4건 FALSE POSITIVE 확인 (C-10,H-9,H-11,L-1일부). 1061 tests ALL PASSED |
| v1.56.1 | 03-11 | N/A 오염 정화 + Duplicate Guard. (1) trade_history 22건 N/A+dup 제거 (2) `record_closed_position` duplicate guard 추가 (3) `pattern_name` 필드 우선 사용. TP+SL WR 61.6→67.4%, gap 11.2→5.4pp (p=0.10, NOT significant) |
| v1.56.0 | 03-11 | Scanner regime_mult 정합 + 메커니즘 교차검증. Scanner DEFAULT_REGIME_MULT 0.3→1.0. IS PnL/MDD 48.3→88.5x (+83%). WF OOS +128.9→+206.1% (+60%). 6-mechanism 교차검증: M6 Regime 유해(-121.7%), M5 Timeout 필수(+80.7%), TOP3(Timeout+Cascade+Momentum) 최적 |
| (연구) | 03-11 | Timeout 교차검증 6-phase: 독립 효과 확인, Cascade 무관 (+193.7%), slot liberation 2124건. Mechanism 교차검증: 15-seed 전부 NON-DISC, AggRisk/DirCap은 IS 감소시키나 live risk guard 역할 |
| v1.55.0 | 03-08 | Live 안정성 3종 개선. (1) N/A 패턴 방지: crash recovery 시 trade_history에서 pattern 복원 (2) Exit 분류 강화: near-SL 40%/near-TP 30% proximity 분류 (3) Mass closure guard: 3+ 동시 청산 시 API 재확인 |
| v1.54.0 | 03-07 | Scanner Cascade SL 구현 + EXIT 분류 개선. N-pos IS: WR 71.3%, PnL +236.4%, MDD 4.87%(MTM), PnL/MDD 48.3x. WF 3/3 PASS (OOS +128.9%). CASCADE_SL exit reason 추가 |
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