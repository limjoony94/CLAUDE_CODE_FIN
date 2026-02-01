# CLAUDE_CODE_FIN - BTC 5분봉 패턴 트레이딩 봇

> **Version**: v1.23.0 | **Bot**: Pattern 5m (12패턴, 7L+5S) | **Updated**: 2026-02-02

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
| 데이터 | `data/btc_5m_270days.csv` (270일, Binance) |

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

## 📊 현재 전략: Pattern 5m v1.22.0

### 핵심 파라미터

| 파라미터 | 값 |
|---------|-----|
| Entry | 3-candle pattern match (12-type) |
| TP/SL | Per-pattern (1.0-2.0%) — MC<0.01만 개별 최적화 |
| Regime | 비활성화 (tight TP/SL은 레짐 독립적) |
| Double Exit | 50%@0.8x + 50%@1.0x |
| Leverage | 3x |
| Timeframe | 5m |

### 270일 검증 결과

| 지표 | v1.22.0 |
|------|---------|
| Patterns | 12 (7L+5S) |
| WR | **80.3%** |
| PF | **3.36** |
| WF | **5/5** |
| MDD | ~20% |

### LONG Patterns (7)

| Pattern | TP/SL | WR | MC | Note |
|---------|-------|-----|--------|------|
| U-MU-H | 1.5/1.5 | 68.4% | 0.0000 | optimized |
| MD-ST-MD | 2.0/2.0 | 70.8% | 0.0078 | optimized |
| GS-U-BD | 1.0/1.0 | 76.0% | 0.0372 | uniform |
| MD-MD-ST | 1.5/2.0 | 71.1% | 0.0002 | optimized |
| BU-IH-DN | 1.5/2.0 | 76.0% | 0.0022 | optimized |
| MD-H-MD | 1.0/1.0 | 83.3% | 0.0014 | uniform best |
| IH-MD-MD | 1.5/2.0 | 86.7% | 0.0020 | optimized |

### SHORT Patterns (5) — v1.22.0: DN-D-BD 제거

| Pattern | TP/SL | WR | MC | Note |
|---------|-------|-----|--------|------|
| BD-U-GS | 1.5/2.0 | 76.5% | 0.0042 | optimized |
| DN-GS-H | 1.0/1.0 | 80.0% | 0.0176 | uniform |
| U-DF-BU | 1.0/1.5 | 76.5% | 0.0010 | optimized |
| BD-GS-BD | 1.0/1.0 | 76.5% | 0.0120 | uniform |
| DN-IH-IH | 1.0/1.5 | 80.0% | 0.0000 | optimized |

### 12-Type Candle Classification

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

★ = v1.22.0에서 수정된 파일

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
| **v1.23.0** | 02-02 | 안정성 강화: Atomic state save, CB exponential backoff, ghost detection ← **현재** |
| **v1.22.0** | 02-01 | DN-D-BD 제거 (과적합), WR 80.3%, PF 3.36 |
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
