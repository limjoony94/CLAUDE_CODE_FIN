# CLAUDE_CODE_FIN - BTC 5분봉 패턴 트레이딩 봇

> **Version**: v1.25.0 | **Bot**: Pattern 5m (20패턴, 10L+10S) | **Updated**: 2026-02-04

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

## 📊 현재 전략: Pattern 5m v1.25.0

### 핵심 파라미터

| 파라미터 | 값 |
|---------|-----|
| Entry | 3-candle pattern match (12-type) |
| TP/SL | Per-pattern (0.3-2.0%) |
| Classification | Ground Truth (HAMMER/INV_HAMMER 우선순위 수정) |
| Double Exit | 50%@0.8x + 50%@1.0x |
| Leverage | 3x |
| Timeframe | 5m |

### 270일 종합 검증 결과

| 지표 | v1.25.0 |
|------|---------|
| Patterns | **20 (10L+10S)** |
| Trades | 1,936 |
| WR | **77.2%** |
| PnL | **+1,346%** (레버리지 3x) |
| PF | **1.77** |
| WF | **5/5** |
| Bootstrap 95% CI | WR [75.3%, 79.0%] |
| OOS (60일) | WR 76.2%, PnL +327% |
| Stress (15bps slip) | WR 66.4%, PnL +98% ✅ |

### LONG Patterns (10)

| Pattern | TP/SL | WR | MC | WF | Trades |
|---------|-------|-----|--------|------|--------|
| MD-BU-U | 0.5/2.0 | 92.1% | 0.0005 | 5/5 | 76 |
| MU-MU-U | 0.7/1.5 | 81.0% | 0.0050 | 4/5 | 84 |
| MU-U-MU | 1.5/2.0 | 70.1% | 0.0140 | 4/5 | 77 |
| BU-BU-BD | 0.7/1.5 | 80.3% | 0.0123 | 5/5 | 76 |
| ST-H-DN | 0.3/0.7 | 82.9% | 0.0172 | 5/5 | 76 |
| ST-MU-U | 0.5/1.0 | 78.4% | 0.0026 | 4/5 | 148 |
| DN-IH-ST | 0.5/0.7 | 73.2% | 0.0118 | 5/5 | 71 |
| IH-DN-DN | 0.7/1.0 | 72.2% | 0.0045 | 4/5 | 108 |
| MD-DN-MU | 1.0/1.0 | 62.3% | 0.0069 | 4/5 | 130 |
| BD-ST-U | 1.5/1.5 | 63.1% | 0.0030 | 5/5 | 122 |

### SHORT Patterns (10)

| Pattern | TP/SL | WR | MC | WF | Trades |
|---------|-------|-----|--------|------|--------|
| MD-ST-ST | 0.5/2.0 | 90.5% | 0.0012 | 5/5 | 95 |
| U-MU-BU | 0.5/2.0 | 91.0% | 0.0020 | 5/5 | 78 |
| MU-BU-DN | 0.7/2.0 | 88.3% | 0.0002 | 5/5 | 77 |
| ST-H-U | 0.5/2.0 | 91.2% | 0.0048 | 5/5 | 57 |
| ST-DN-H | 0.7/2.0 | 85.3% | 0.0079 | 5/5 | 75 |
| MD-MU-U | 1.0/2.0 | 81.2% | 0.0020 | 5/5 | 85 |
| BU-U-ST | 0.5/1.5 | 87.2% | 0.0001 | 5/5 | 125 |
| H-DN-ST | 0.5/1.5 | 85.7% | 0.0176 | 4/5 | 70 |
| DN-BD-BU | 0.7/1.0 | 72.0% | 0.0068 | 4/5 | 107 |
| DN-BU-U | 1.0/1.0 | 59.3% | 0.0180 | 4/5 | 199 |

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
│   │   └── pattern_5m/             # 14개 모듈 ★
│   │       ├── bot.py              # 메인 루프
│   │       ├── config.py           # 설정
│   │       ├── constants.py        # 패턴 + Per-pattern TP/SL ★
│   │       ├── exchange.py         # BingX API
│   │       ├── indicators.py       # 기술 지표
│   │       ├── models.py           # 데이터클래스
│   │       ├── orders.py           # 주문 + TP/SL 자동조정
│   │       ├── position.py         # facade
│   │       ├── position_open.py    # 진입 ★
│   │       ├── position_monitor.py # 모니터링
│   │       ├── position_close.py   # 청산
│   │       ├── signals.py          # 패턴 탐지 + Context Filter
│   │       ├── state.py            # 상태 저장
│   │       └── utils/              # lock, logging
│   ├── analysis/                   # 연구 스크립트
│   └── monitoring/                 # 모니터링 스크립트
├── data/                           # 시장 데이터 CSV
├── results/                        # 봇 상태/메트릭 JSON
├── logs/                           # 운영 로그
├── claudedocs/                     # 연구 리포트 아카이브
└── archive/                        # 레거시 아카이브
```

★ = v1.25.0에서 수정된 파일

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

> 상세: [claudedocs/PATTERN_VALIDATION_AUDIT_20260203.md](bingx_rl_trading_bot/claudedocs/PATTERN_VALIDATION_AUDIT_20260203.md)

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
| **v1.25.0** | 02-04 | Moderate-B-20 포트폴리오 (10L+10S), 심층 분석 최적화, WR 77.2%, PnL +1,346% ← **현재** |
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

- [프로젝트 분석](docs/analysis.md)
- [에이전트 가이드](docs/agent-guides.md)
- [구조 개선안](docs/restructure-plan.md)
- [정리 목록](docs/cleanup-list.md)
- [코딩 컨벤션](docs/CODING_CONVENTIONS.md)
- [Git 워크플로](docs/GIT_WORKFLOW.md)
- [기술 스택](docs/TECH_STACK.md)
- [전체 문서 목차](docs/INDEX.md)