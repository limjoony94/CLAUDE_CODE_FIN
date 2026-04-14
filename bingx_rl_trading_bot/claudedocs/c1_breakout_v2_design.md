# C1 Breakout v2.6 — Strategy Design Document

> **Version**: v2.6.0 | **Date**: 2026-04-14 | **Asset**: BTC/USDT | **TF**: 15m

## 1. 전략 개요

15분봉 채널 돌파 + 프랙탈 SL + ATR 트레일링 TP.
18라운드 연구, 30+ 전략 변형, 10개 웹 레퍼런스 전략 비교를 통해 선별된 최종 전략.

### 검증 결과 (v2.5, 333일 백테스트, additive 1x)

| 지표 | 값 |
|------|-----|
| PnL (additive 1x) | +169.5% |
| PnL (compound 1x) | +417.9% |
| MDD | 5.4% |
| WR | 36.6% |
| R:R | 3.36 |
| Daily | +0.509% |
| Trades/day | 3.1 |
| MC Direction | p=0.000 DISC |
| WF OOS | 5/5 PASS (+153.9%) |
| 3-Way | ALL PASS (Train +61%, Valid +54%, Test +55%) |
| Progressive | 10/10 PASS |
| Param grid | 60/60 양수 (파라미터 고원) |
| Bootstrap 95% CI | [+109%, +234%] |

## 2. 진입 로직

### 채널 돌파 + Body 확인

```
조건 (LONG):
  1. close > max(high[i-15 : i])     ← 15봉 최고가 돌파 (현재봉 제외)
  2. |body| > 0.4 × range            ← 돌파봉이 확실한 방향성 (도지 제거)
  3. body > 0 (close > open)         ← body 방향이 돌파 방향과 일치
  
조건 (SHORT):
  1. close < min(low[i-15 : i])      ← 15봉 최저가 이탈 (현재봉 제외)
  2. |body| > 0.4 × range
  3. body < 0 (close < open)

진입가: o[i+1] (다음 봉 시가)
```

### 파라미터

| 파라미터 | 값 | 민감도 |
|---------|-----|--------|
| channel | 15 | 12-18 전부 양수 (고원) |
| body_min_ratio | 0.4 | 0.4-0.6 안정 |

## 3. 청산 로직

### 우선순위: SL → Emergency → Timeout → Trail (SL-first)

### SL: 프랙탈 기반 (동적)

```
LONG SL = max(last_swing_low, entry - 3.3 × ATR(14))
SHORT SL = min(last_swing_high, entry + 3.3 × ATR(14))

swing_low/high: lookback=10, causal (과거 데이터만, 미래 없음)
  현재봉 포함 과거 11봉 중 최저/최고인 바를 최신 swing으로 갱신
  
SL 거리 제한: 0.15% ~ 3.0% (범위 밖이면 진입 취소)
Exchange: STOP_MARKET @ fractal SL price (crash protection)
```

### TP: ATR 트레일링 (동적)

```
trail_distance = 2.5 × ATR(14)[current_bar] / close × 100  (% 단위)

LONG: best_price = max(all highs since entry)
      exit when: (best_pnl - cur_pnl) >= trail_distance
      
SHORT: best_price = min(all lows since entry)
       exit when: (best_pnl - cur_pnl) >= trail_distance

트레일 활성화: best_pnl > 0.05%
Exchange: TRAILING_STOP_MARKET @ trailingPercent (15분마다 ATR 갱신)
```

### Emergency SL

```
|unrealized_pnl| >= 3.0% → 즉시 청산 (gap/extreme move protection)
실효: 9% at 3x leverage
```

### Timeout

```
bars_held >= 192 (48시간) → 시장가 청산
```

### 파라미터

| 파라미터 | 값 | 민감도 |
|---------|-----|--------|
| max_sl_atr | 3.3 | 2.7-3.6 전부 양수 |
| trail_K | 2.5 | 2.0-3.0 안정 |
| emergency_sl | 3.0% | 고정 |
| max_hold | 192 bars | 고정 |

## 4. 포지션 관리

| 항목 | 값 |
|------|-----|
| 최대 동시 포지션 | 1 (N=1) |
| Exchange 레버리지 | 10x (BingX 계정 설정) |
| Trading 레버리지 | 3x (실제 포지션 크기) |
| 포지션 크기 | balance × 0.98 × 3 / price |
| 재진입 대기 | 청산 후 최소 2봉 (min_bars_between) |

## 5. 리스크 관리

| 항목 | 값 |
|------|-----|
| Emergency SL | 3.0% (9% at 3x) |
| Halt 브레이크 | 없음 (v2.4에서 전부 제거) |
| 안전장치 | SL + Trail + Emergency만 적용 |

## 6. 데이터 요구사항

| 항목 | 값 |
|------|-----|
| OHLCV | 15분봉 (candle_bars_fetch=100) |
| ATR warmup | 최소 14봉 (3.5시간) |
| Channel warmup | 최소 15봉 (3.75시간) |
| Fractal warmup | 최소 10봉 (2.5시간) |
| 총 warmup | 25봉 (6.25시간) |

## 7. Exchange 연동

| 항목 | 설정 |
|------|------|
| Exchange | BingX Perpetual |
| Position Mode | One-Way (positionSide=BOTH) |
| SL Order | STOP_MARKET (진입 시 즉시 배치, reduceOnly) |
| TP Order | TRAILING_STOP_MARKET (trailingPercent, 15분 ATR 갱신) |
| Polling Interval | 15분 (캔들 완성 시) |
| Fee | 0.10% RT (taker 0.05% × 2) |

## 8. 구현 아키텍처

```
scripts/production/
├── c1_breakout_bot.py       # 엔트리포인트 (로깅, 경로 설정)
└── c1_breakout/
    ├── __init__.py
    ├── signals.py           # C1BreakoutSignal (채널 돌파 + body + SL/TP 로직)
    ├── bot.py               # C1BreakoutBot (메인 루프, exchange, state, trail)
    ├── indicators.py        # compute_atr, compute_channel, compute_fractal_swings
    └── config.py            # load_config (YAML + defaults)
config/
└── c1_breakout_config.yaml  # 전략+리스크 파라미터 (유일한 설정 소스)
```

## 9. 모니터링 기준

| 상태 | 기준 | 행동 |
|------|------|------|
| GREEN | R:R > 2.5, 정상 빈도 (~3/day) | 유지 |
| YELLOW | R:R 1.5-2.5 or MDD 8-15% | 주의 관찰 |
| RED | R:R < 1.5 or MDD > 15% | 수동 점검 |

## 10. 버전 히스토리

| 버전 | 날짜 | 변경 |
|------|------|------|
| v2.0.0 | 2026-04-12 | 초기 설계. 18라운드 연구 기반. |
| v2.3.0 | 2026-04-13 | 21 Cycle 비판. 16건 버그 수정 (BUG#16~30). |
| v2.4.0 | 2026-04-13 | 18 Cycle 비판. Halt 제거, leverage 검증. |
| v2.5.0 | 2026-04-13 | 30 Cycle 비판. SL-first, lookback=10, Exch10x/Trade3x. |
| v2.6.0 | 2026-04-14 | 20 Cycle 비판. BUG#35 trail 90%→0.9%, SL ID sync, margin retry. |
