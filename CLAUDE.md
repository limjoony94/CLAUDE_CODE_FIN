# CLAUDE_CODE_FIN - BTC 5분봉 패턴 트레이딩 봇

> **Version**: v1.59.4 | **Bot**: Pattern 5m (131패턴, 59L+72S, Edge18pp+NeutralWindow+ATR Scanner+Holdout+MDD+Cap7+MomGuard1.5%15m1h+NposScanner+CascadeSL85+AggRisk8_15+ATRClamp05_15+TO288+ScannerCascade+MassCloseGuard+ExitClassify+PatternRecovery+RegimeFix+DupGuard+CodeAuditFix+TPScale05+OrphanPrevention+PosTrackFix+MedianFallback+EmgSlRace+NaPrevent) | **Updated**: 2026-03-12

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

## 📊 현재 전략: Pattern 5m v1.57.0

### 핵심 파라미터

| 파라미터 | 값 |
|---------|-----|
| Entry | 3-candle pattern match (12-type) |
| TP/SL | **Per-pattern ATR-scaled × TP 0.5** (TP 0.43-1.40%, SL 1.44-5.95%, MAE/MFE + ATR scanner v2.4 + tp_scale_factor 0.5, v1.57.0) |
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

### v1.57.0 검증 요약

- **N-pos IS (TP×0.5, Cascade ON, regime_mult=1.0)**: WR 77.8%, PnL +458%, MDD 4.05%(MTM), **PnL/MDD 113.0x**
- **131패턴 (59L+72S)**: TP 0.43-1.40% (×0.5), SL 1.44-5.95%, Edge 18.0-31.8pp
- **v1.56.0→v1.57.0 개선**: PnL/MDD 88.5x→113.0x (+28%), OOS +206→+306% (+49%)
- **TP Scale Factor**: `tp_scale_factor: 0.5` — N-pos 슬롯 회전 최적화 (1-pos 최적 ≠ N-pos 최적)
- **판별력 검증**: DISCRIMINATING (random gap +83-146%), Cascade-INDEPENDENT (OFF gap +56.5% > ON +33.8%)
- 개별 패턴 상세: `results/dynamic_patterns.json` 참조 (원본 TP, config에서 ×0.5 적용)

### WF OOS 검증 (v1.57.0, TP×0.5, Cascade SL ON, regime_mult=1.0, 3-fold Expanding Window, N-pos)

| Fold | OOS PnL |
|------|---------|
| 1 | +56.8% |
| 2 | +79.2% |
| 3 | +170.0% |

**Verdict: 3/3 PASS** | Total OOS PnL: +306.0% (TP×0.5+Cascade, N-pos) | IS PnL/MDD: 113.0x

> 이전 v1.56.0 WF: OOS +206.1% → v1.57.0: +306.0% (+49% 향상, TP scale factor 효과)

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
| **Scanner Regime Fix (v1.56.0)** | Scanner DEFAULT_REGIME_MULT 0.3→1.0 — production v1.42.0에서 비활성화한 Regime Sizing을 scanner에서도 정합. IS PnL/MDD +83%, OOS +60% |
| **Duplicate Trade Guard (v1.56.1)** | `record_closed_position`에 중복 기록 방지 가드 추가 + `pattern_name` 필드 우선 사용. N/A 오염(22건) 정화 완료. TP+SL WR 61.6→67.4%, gap 11.2→5.4pp |
| **Code Audit Fixes (v1.56.2)** | (1) Cascade SL Place-first/Cancel-after (보호 갭 제거) (2) SL 실패 시 Emergency SL 즉시 호출 (3) Momentum cooldown state 영속화 (4) datetime 파싱 방어 (5) Hardcoded 300s→CANDLE_DURATION_MS (6) Always-truthy 수정. 1061 tests ALL PASSED |
| **TP Scale Factor (v1.57.0)** | **Post-discovery TP×0.5 스케일링** — N-pos 슬롯 회전 최적화. Scanner 1-pos 최적 TP를 N-pos 포트폴리오에 맞게 축소. IS PnL/MDD +28%, OOS +49%, 10/10 MC wins, DISCRIMINATING, Cascade-independent. config `tp_scale_factor` |
| **Orphan Prevention (v1.59.0)** | **3-layer defense against transient API zero-contract** — (1) ALL closure detections trigger 1s delay + fresh re-verify (removed ≥3 threshold) (2) Inter-direction exchange_map rebuild after closures (3) Post-closure orphan detection + auto-recovery. Root cause: BingX API transient 0-contract during order processing. 1073 tests ALL PASSED |
| **N/A Pattern Prevention (v1.59.4)** | **4-layer N/A cascade 방지** — (1) `_restore_none_pattern_slots()`: crash recovery 후 None 슬롯 로그 기반 복원 (2) `record_closed_position` last-resort log recovery (3) `cancel_remaining_orders` 3회 retry (4) Recovery 전 stale order cleanup. Root cause: BingX averaged entry → price matching 실패 → None cascade |
| **DISABLED (5개)** | Regime Sizing, Adaptive Leverage, Equity Curve, Correlation-Aware, Loss Burst Brake — 각 `enabled: true`로 재활성화 가능. Entry Optimization은 ROLLED BACK (코드 제거) |

### Dynamic Pattern Selection (v1.27.3)

`pattern_source` 설정으로 정적(constants.py) 또는 동적(scanner 출력) 패턴 세트 선택 가능.

| 모드 | 설정값 | 패턴 소스 | TP/SL |
|------|--------|-----------|-------|
| Static (fallback) | `pattern_source: static` | constants.py 51패턴 | Per-pattern 최적화 |
| Dynamic PP | `pattern_source: dynamic` + `tp_sl_mode: per_pattern` | results/dynamic_patterns.json | PP grid search |
| **Dynamic ATR + TP Scale (현재)** | `pattern_source: dynamic` + `tp_sl_mode: per_pattern` + `tp_scale_factor: 0.5` | results/dynamic_patterns.json | **MAE/MFE + ATR-scaled, TP×0.5** |

**Scanner CLI 사용법** (v2.4):
```bash
cd bingx_rl_trading_bot
# 기본 (neutral + ATR + N-pos, v1.38.1~ default)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --wf-folds 3 --holdout-days 7
# Legacy 1-pos 모드 (빠른 반복용)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --wf-folds 3 --holdout-days 7 --no-npos
# 주요 옵션: --no-neutral, --no-atr, --neutral-tol 2.0, --atr-clamp-lo 0.5 --atr-clamp-hi 1.5, --n-slots 5 --direction-cap 4
```

**현재 적용**: MAE/MFE + ATR scanner + Neutral window (edge>=18pp, MC<0.01, --wf-folds 3, --holdout-days 7) → **131패턴 (59L+72S)** (v1.56.0 rescan, regime_mult=1.0) + **tp_scale_factor=0.5** (v1.57.0).
WF N-pos 3/3 PASS, OOS +306.0%. IS PnL/MDD 113.0x. TP 0.43-1.40% (×0.5), SL 1.44-5.95%. Neutral window ±1% 자동 탐색 (259d). ATR config: a14/w576/clamp[0.5,1.5]. Data: 303d. Backup: `results/dynamic_patterns_131pat_v1530_backup.json`

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
| **v1.59.4** | 03-12 | **N/A Pattern Prevention + Orphan Order Retry** ← 현재. 4-layer N/A cascade 방지: (1) `_restore_none_pattern_slots()` — crash recovery 후 None-pattern 슬롯을 로그에서 복원 (Phase 5) (2) `record_closed_position` last-resort log recovery — trade_history N/A 오염 방지 (3) `cancel_remaining_orders` 3회 retry with backoff — 네트워크 에러 시 orphan 주문 방지 (4) `recover_position_to_state` stale order cleanup — recovery 전 잔류 TP/SL 주문 정리. Root cause: BingX averaged entry price defeats price-matching → pattern=None → .bak 오염 → 연쇄 N/A. 재시작 시 4개 None 슬롯 즉시 복원 + per-pattern TP/SL 재조정 확인. 1101 tests ALL PASSED |
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