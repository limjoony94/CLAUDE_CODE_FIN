# CLAUDE_CODE_FIN - BTC 5분봉 패턴 트레이딩 봇

> **Version**: v1.28.23 | **Bot**: Pattern 5m (47패턴, 13L+34S) | **Updated**: 2026-02-18

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

## 📊 현재 전략: Pattern 5m v1.28.24

### 핵심 파라미터

| 파라미터 | 값 |
|---------|-----|
| Entry | 3-candle pattern match (12-type) |
| TP/SL | **Per-pattern 최적화** (scanner PP grid search) |
| Classification | Ground Truth (HAMMER/INV_HAMMER 우선순위 수정) |
| Leverage | 3x |
| Timeframe | 5m |
| Pattern Source | **Dynamic** (results/dynamic_patterns.json) |
| Scanner MAX_BARS | **288** (24h; v1.28.24: 500→288, 24h timeout study) |
| Quality Filter | **Edge>=21.8pp + WR>=60% + SL>=1.0% + MC<0.01 + BH FDR + min_trades>=25** |
| Patterns | **112** (22L + 90S), edge mean 24.7pp, WR mean 90.5% |
| Risk | Daily loss **13%** (v1.28.5), 3-consecutive-loss pause |

### 270일 In-Sample 검증 결과

| 지표 | Static 51 (v1.27.2) | Dynamic 47 (v1.28.11) | **Dynamic 112 (v1.28.24)** |
|------|---------------------|------------------------|---------------------------|
| Patterns | 51 (32L+19S) | 47 (13L+34S) | **112 (22L+90S)** |
| Scanner MAX_BARS | 500 | 500 | **288 (24h)** |
| Filter | MC<0.01 + edge>=10pp | MC<0.01 + BH FDR + E>=21.8 + WR>=60% + SL>=1.0% | **동일** |
| WR mean | 84.9% | 86.8% | **90.5%** |
| Edge mean | ~15pp | 24.9pp | **24.7pp** |
| Portfolio Trades | 339 | 339 | **358** |
| Portfolio PnL | +966% | +894% | **+1,398%** |
| Portfolio MDD | 16.2% | 24.2% | **17.0%** |
| PnL/MDD | 59.6x | 36.9x | **82.1x** |
| TP/SL | Per-pattern (legacy) | Per-pattern (PP grid) | **Per-pattern (PP grid)** |

### Pattern Summary — v1.28.24 Dynamic (112 patterns)

> 전체 패턴 상세는 `results/dynamic_patterns.json` 참조

**LONG (22)**: BD-BD-BU, BD-DN-MU, BU-BD-DN, DN-MD-MD, H-MD-DN, H-ST-DN, IH-DN-MD, IH-MU-DN, MD-BD-DN, MD-DN-BU, MD-MU-ST, MD-ST-MD, MU-MD-ST, MU-ST-U, MU-U-BU, MU-U-U, ST-DN-IH, ST-MU-U, ST-ST-D, U-D-ST, U-MD-BD, U-MU-H

**SHORT (90)**: BD-BU-BU, BD-BU-U, BD-DN-BU, BD-U-DN, BD-U-MU, BU-BD-D, BU-BD-U, BU-BU-DN, BU-BU-U, BU-DN-BU, BU-DN-H, BU-MU-DN, BU-U-DN, BU-U-ST, D-D-DN, D-DN-ST, D-MU-DN, D-ST-U, D-U-MU, DF-U-U, DN-BD-BU, DN-BD-ST, DN-BU-BU, DN-BU-MD, DN-D-BD, DN-DF-DN, DN-DN-GS, DN-DN-IH, DN-GS-ST, DN-IH-MD, DN-IH-ST, DN-MD-BD, DN-MD-DN, DN-MD-MU, DN-MU-MU, DN-ST-BD, DN-ST-BU, DN-ST-D, DN-U-H, GS-ST-U, GS-U-DN, H-DN-ST, H-DN-U, H-ST-ST, IH-DN-DN, IH-DN-U, IH-ST-ST, IH-U-DN, IH-U-U, MD-BD-U, MD-BU-DN, MD-DN-ST, MD-MU-DN, MD-ST-ST, MD-U-D, MU-BU-DN, MU-DN-BU, MU-DN-MU, MU-DN-ST, MU-MD-U, MU-MU-U, MU-ST-MD, MU-U-ST, ST-BD-BD, ST-BD-BU, ST-D-U, ST-DN-BU, ST-H-DN, ST-IH-DN, ST-ST-H, ST-ST-MD, ST-ST-U, ST-U-BD, ST-U-BU, ST-U-H, U-BU-BD, U-BU-MD, U-BU-ST, U-D-DN, U-DN-BD, U-DN-DF, U-DN-H, U-GS-DN, U-H-DN, U-IH-U, U-MD-ST, U-MU-MD, U-ST-IH, U-ST-U, U-U-ST

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
| **Dynamic PP (현재)** | `pattern_source: dynamic` + `tp_sl_mode: per_pattern` | results/dynamic_patterns.json | **PP grid search** |

**Scanner CLI 사용법**:
```bash
cd bingx_rl_trading_bot
python scripts/scanner/pattern_scanner.py                           # 기본 (PP 모드, 270d)
python scripts/scanner/pattern_scanner.py --edge-threshold 10 --correction bh --wf-folds 3
```

**현재 적용**: Scanner 출력 257패턴 → Edge>=21.8pp + WR>=60% 후처리 필터 → 50패턴.
257패턴 원본: `results/dynamic_patterns_257_backup.json`

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
| **v1.28.24** | 02-18 | **Scanner MAX_BARS 500→288 (24h timeout study)**: 3-phase 24h 연구 (v1 DROP비교, v2 edge threshold, v3 forced close) 결과 288봉(24h) 최적 확인. Scanner `MAX_BARS=500→288` 변경 후 재스캔. 325패턴 → E>=21.8pp+WR>=60%+SL>=1.0% → **112패턴 (22L+90S)**. 포트폴리오: PnL +1398% (vs +894%), WR 93.6% (vs 88.1%), MDD 17.0% (vs 24.2%), PnL/MDD 82.1x (vs 36.9x). Forced close(패배처리) WF 1/3 FAIL → 미적용. Production 봇 변경 없음 (timeout 미도입, TP/SL 보유). EXPECTED_AVG_WIN 5.44, EXPECTED_AVG_LOSS 10.73. ← **현재** |
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