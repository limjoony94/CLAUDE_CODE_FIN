# VWAP Band Bot 전략 요약

**작성일**: 2025-12-05
**봇 파일**: `scripts/production/vwap_band_bot.py`
**상태**: 실행 중

---

## 1. 전략 개요

**전략명**: VWAP Band Mean Reversion (평균 회귀)
- **LONG 진입**: 가격 < VWAP 하단밴드 (과매도 → 상승 기대)
- **SHORT 진입**: 가격 > VWAP 상단밴드 (과매수 → 하락 기대)

---

## 2. 봇 설정 (현재 사용 중)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| VWAP_PERIOD | 20 | 5시간 (20 x 15분) |
| BAND_MULTIPLIER | 2.5 | 표준편차 배수 |
| TAKE_PROFIT_PCT | 3.0% | 익절 |
| STOP_LOSS_PCT | 1.0% | 손절 |
| COOLDOWN_CANDLES | 4 | 1시간 쿨다운 |
| MAX_HOLD_CANDLES | 48 | 12시간 최대 보유 |
| LEVERAGE | 4x | 실효 레버리지 |
| TIMEFRAME | 15m | 캔들 주기 |

---

## 3. 백테스트 결과

### 3.1 사용된 백테스트 파일
- **파일**: `results/vwap_15m_finetune_20251203_190022.csv`
- **데이터**: `BTCUSDT_5m_raw_105days_20251122_231051.csv`
- **기간**: 2025-08-09 ~ 2025-11-22 (105일)

### 3.2 현재 설정의 백테스트 결과

```
설정: VWAP 5h, Band 2.5x, TP 3%, SL 1%, vol_ma 20, cooldown 4, max_hold 12h
```

| 지표 | 값 |
|------|-----|
| **Total Return** | **+211.3%** |
| Trades | 170회 |
| Win Rate | 49.4% |
| Profit Factor | 1.46 |
| Max Drawdown | 36.0% |
| Sharpe Ratio | 5.88 |
| TP Exits | 16 (9%) |
| SL Exits | 68 (40%) |
| Timeout Exits | 86 (51%) |

### 3.3 주의: 다른 백테스트 파일들

| 파일 | VWAP Period | 결과 | 비고 |
|------|-------------|------|------|
| `vwap_strategy_backtest` | 24h, 48h, 72h | 손실 (-10~-22%) | **다른 설정** |
| `vwap_15m_extended` | 다양 | 대부분 손실 | **다른 설정** |
| `vwap_15m_finetune` | **5h** | **+211%** | **현재 봇 설정** |

⚠️ **혼동 주의**: VWAP Period가 다르면 결과가 완전히 다름

---

## 4. 실제 거래 기록

### 4.1 거래 현황 (2025-12-03 ~ 12-05)

| # | 방향 | 진입가 | 종료 방식 | 결과 |
|---|------|--------|----------|------|
| 1 | LONG | $92,168 | MAX_HOLD timeout (12h) | +6.5% |
| 2 | LONG | $92,725 | EXTERNAL_CLOSE | SL 추정 |
| 3 | LONG | $91,899 | EXTERNAL_CLOSE | SL 추정 |
| 4 | LONG | $91,899 | EXTERNAL_CLOSE | SL 추정 |
| 5 | LONG | $91,032 | **보유 중** | +0.7% |

### 4.2 실제 vs 백테스트 비교

| 구분 | 백테스트 | 실제 |
|------|----------|------|
| 거래 수 | 170회 | 5회 |
| Win Rate | 49% | 20% (1/5) |
| Timeout 비율 | 51% | 20% (1/5) |
| 기간 | 105일 | 2일 |

**결론**: 샘플 크기가 너무 작아 아직 판단 불가

---

## 5. 파일 위치

| 용도 | 경로 |
|------|------|
| 봇 코드 | `scripts/production/vwap_band_bot.py` |
| 모니터 | `scripts/monitoring/vwap_band_monitor.py` |
| 상태 파일 | `results/vwap_band_bot_state.json` |
| Lock 파일 | `results/vwap_band_bot.lock` |
| 로그 | `logs/vwap_band_bot_YYYYMMDD.log` |
| 백테스트 결과 | `results/vwap_15m_finetune_20251203_190022.csv` |

---

## 6. 운영 명령어

```bash
# 봇 시작
cd bingx_rl_trading_bot
python scripts/production/vwap_band_bot.py

# 모니터링
python scripts/monitoring/vwap_band_monitor.py

# 상태 확인
cat results/vwap_band_bot_state.json

# 로그 확인
tail -50 logs/vwap_band_bot_$(date +%Y%m%d).log

# PID 확인
cat results/vwap_band_bot.lock
```

---

## 7. 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2025-12-03 | VWAP Band Bot v2.0 배포 |
| 2025-12-04 | Hedge Mode reduce_only 버그 수정 |
| 2025-12-04 | 누락된 TP/SL 주문 생성 로직 추가 |

---

**마지막 업데이트**: 2025-12-05 04:30 KST
