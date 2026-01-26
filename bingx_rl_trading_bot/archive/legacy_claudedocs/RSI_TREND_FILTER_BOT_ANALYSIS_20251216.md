# RSI Trend Filter Bot v1.1 - 종합 분석 문서

**작성일**: 2025-12-16
**상태**: ✅ 운영 중
**파일**: `scripts/production/rsi_trend_filter_bot.py`
**설정**: `config/rsi_trend_filter_config.yaml`

---

## 1. 전략 개요

### 1.1 핵심 로직

```
Entry Rules:
  LONG:  Close > EMA(100) AND RSI(14) crosses above 40
  SHORT: Close < EMA(100) AND RSI(14) crosses below 60

Exit Rules:
  TP: 3.0%
  SL: 2.0%
  Leverage: 4x
```

### 1.2 전략 철학

| 요소 | 설명 |
|------|------|
| **Trend Filter** | EMA(100)으로 상승/하락 추세 판단 |
| **Entry Trigger** | RSI 교차 신호 (oversold/overbought 복귀) |
| **Risk:Reward** | 1:1.5 (SL 2% : TP 3%) |
| **Asymmetric Thresholds** | LONG: RSI 40, SHORT: RSI 60 (비대칭) |

### 1.3 신호 상세

**LONG 진입 조건**:
1. 현재가 > EMA(100) → 상승 추세 확인
2. RSI(14)가 40을 상향 돌파 → oversold 영역 탈출
3. 추세 방향으로 진입하면서 모멘텀 확인

**SHORT 진입 조건**:
1. 현재가 < EMA(100) → 하락 추세 확인
2. RSI(14)가 60을 하향 돌파 → overbought 영역 탈출
3. 추세 방향으로 진입하면서 모멘텀 확인

---

## 2. 검증 결과

### 2.1 Walk-Forward 검증 (7개 윈도우)

| Window | Test Period | Test PnL | Test WR% | Long PnL | Short PnL |
|--------|-------------|----------|----------|----------|-----------|
| 1 | 05/07~06/06 | +19.2% | 50.0% | +21.2% | -2.0% |
| 2 | 06/06~07/06 | +16.0% | 50.0% | -2.0% | +18.0% |
| 3 | 07/06~08/05 | **+39.2%** | 58.3% | +29.6% | +9.6% |
| 4 | 08/05~09/04 | +19.2% | 50.0% | -2.0% | +21.2% |
| 5 | 09/04~10/04 | +21.2% | 57.1% | +26.4% | -5.2% |
| 6 | 10/04~11/03 | +10.8% | 46.2% | -2.0% | +12.8% |
| 7 | 11/03~12/03 | -4.8% | 40.9% | -27.2% | +22.4% |

**종합 성과**:
- Positive Windows: **6/7 (85.7%)**
- Total Test PnL: **+120.8%**
- Average Test PnL: +17.3%/window
- Sharpe Ratio: **1.42**
- p-value: **0.013** (통계적 유의)

### 2.2 다른 전략 대비 비교

| 전략 | Return | WR% | MDD | WF Positive | Status |
|------|--------|-----|-----|-------------|--------|
| **RSI Trend Filter** | **+120.8%** | 50.3% | ~35% | **6/7** | ✅ 검증됨 |
| Trend Following | +313.6% | 51.5% | 24.3% | 미검증 | ⚠️ |
| EMA Crossover | +227.7% | 55.7% | 19.4% | 미검증 | ⚠️ |
| FIXED 1.5% SL | +160.8% | 49.5% | 56.5% | 미검증 | ⚠️ |
| Supertrend Trail | ~~+1276%~~ -283% | 31.4% | 97.8% | ❌ 버그 | ⛔ |

---

## 3. 코드 구조 분석

### 3.1 파일 구조

```
scripts/production/rsi_trend_filter_bot.py (885 lines)
├── Configuration (line 35-72)
├── Logging Setup (line 74-93)
├── Configuration Loading (line 95-120)
├── Exchange Setup (line 122-143)
├── State Management (line 145-380)
│   ├── load_state() / save_state()
│   ├── sync_position_with_exchange()
│   ├── get_actual_exit_price()
│   ├── recover_position_to_state()
│   └── record_closed_position()
├── Indicator Calculations (line 382-406)
│   ├── calculate_rsi()
│   ├── calculate_ema()
│   └── calculate_indicators()
├── Signal Generation (line 408-450)
│   └── check_entry_signal()
├── Order Execution (line 452-755)
│   ├── get_position_size()
│   ├── open_position()
│   ├── place_tp_sl_orders()
│   ├── check_position_status()
│   └── close_position_market()
└── Main Loop (line 757-884)
    ├── check_cooldown()
    ├── check_daily_loss_limit()
    └── run_bot()
```

### 3.2 핵심 함수 분석

#### 3.2.1 Signal Generation (line 412-450)

```python
def check_entry_signal(df, state, config):
    """
    LONG:  Close > EMA(100) AND RSI crosses above 40
    SHORT: Close < EMA(100) AND RSI crosses below 60
    """
    current = df.iloc[-1]
    previous = df.iloc[-2]

    close = current['close']
    ema = current['ema']
    rsi = current['rsi']
    rsi_prev = previous['rsi']

    above_ema = close > ema

    # LONG: 상승추세 + RSI 40 상향돌파
    if above_ema:
        if rsi > 40 and rsi_prev <= 40:
            return 'LONG', reason

    # SHORT: 하락추세 + RSI 60 하향돌파
    else:
        if rsi < 60 and rsi_prev >= 60:
            return 'SHORT', reason
```

**분석**:
- RSI 교차 감지: 이전 캔들 RSI vs 현재 캔들 RSI 비교
- EMA 필터: 추세 방향과 일치하는 신호만 허용
- 비대칭 임계값: LONG 40, SHORT 60 (보수적)

#### 3.2.2 Position Sync (line 200-276)

```python
def sync_position_with_exchange(exchange, state, config):
    """
    봇 재시작 시 거래소 포지션과 로컬 상태 동기화
    """
    # Case 1: 상태에 포지션 있음, 거래소에 없음 → 외부 청산 처리
    # Case 2: 상태에 포지션 없음, 거래소에 있음 → 포지션 복구
    # Case 3: 둘 다 있음 → 일치 확인 및 수량 업데이트
```

**분석**:
- 봇 재시작 시 상태 복구 로직 완비
- 외부 청산 감지 및 PnL 계산
- 거래소 실제 포지션 기준으로 동기화

#### 3.2.3 Order Execution (line 484-576)

```python
def open_position(exchange, state, config, signal, reason):
    # 1. 레버리지 설정
    exchange.set_leverage(leverage, symbol)

    # 2. 포지션 사이즈 계산
    quantity, available = get_position_size(exchange, config)

    # 3. 시장가 주문 실행
    order = exchange.create_market_order(...)

    # 4. 실제 체결가 확인 (주문 응답 또는 포지션 조회)
    actual_entry_price = float(order.get('average', ...))

    # 5. TP/SL 계산 (슬리피지 버퍼 적용)
    tp_price = entry * (1 + direction * (tp_pct - 0.05%) / 100)
    sl_price = entry * (1 - direction * (sl_pct - 0.05%) / 100)

    # 6. TP/SL 주문 배치
    place_tp_sl_orders(exchange, state, config)
```

**분석**:
- 실제 체결가 기반 TP/SL 계산 (슬리피지 고려)
- Hedge Mode 지원 (positionSide 파라미터)
- TAKE_PROFIT_MARKET / STOP_MARKET 주문 사용

### 3.3 리스크 관리

| 기능 | 설정값 | 설명 |
|------|--------|------|
| **Daily Loss Limit** | 15% | 일일 손실 한도 도달 시 거래 중단 |
| **Max Position Size** | $10,000 | 포지션 최대 크기 제한 |
| **Cooldown** | 4 candles (1시간) | 연속 진입 방지 |
| **Slippage Buffer** | 0.05% | TP/SL 계산 시 슬리피지 고려 |

---

## 4. 설정 파라미터

### 4.1 기본 설정 (config/rsi_trend_filter_config.yaml)

```yaml
symbol: "BTC-USDT"
timeframe: "15m"
leverage: 4
position_size_pct: 95

strategy:
  rsi_period: 14
  rsi_long_threshold: 40
  rsi_short_threshold: 60
  ema_period: 100
  tp_pct: 3.0
  sl_pct: 2.0
  cooldown_candles: 4

risk:
  max_daily_loss_pct: 15
  max_position_size_usd: 10000
```

### 4.2 파라미터 튜닝 가이드

| 파라미터 | 현재값 | 조정 방향 | 영향 |
|---------|--------|----------|------|
| `rsi_long_threshold` | 40 | ↑ 상승: 더 보수적 | 진입 감소, 승률 상승 |
| `rsi_short_threshold` | 60 | ↓ 하락: 더 보수적 | 진입 감소, 승률 상승 |
| `ema_period` | 100 | ↑ 상승: 느린 추세 | 신호 감소, 노이즈 감소 |
| `tp_pct` | 3.0 | ↑ 상승: 더 큰 목표 | 승률 하락, 수익폭 증가 |
| `sl_pct` | 2.0 | ↑ 상승: 여유 손절 | 손절 감소, 손실폭 증가 |
| `cooldown_candles` | 4 | ↑ 상승: 거래 감소 | 오버트레이딩 방지 |

---

## 5. 운영 가이드

### 5.1 시작/모니터링 명령어

```bash
# 봇 시작
cd bingx_rl_trading_bot
python scripts/production/rsi_trend_filter_bot.py

# 상태 확인
cat results/rsi_trend_filter_bot_state.json

# 로그 확인
tail -f logs/rsi_trend_filter_bot_YYYYMMDD.log
```

### 5.2 상태 파일 구조

```json
{
  "position": {
    "direction": "SHORT",
    "entry_price": 86035.8,
    "quantity": 0.016,
    "tp_price": 83497.7,
    "sl_price": 87713.5,
    "entry_time": "2025-12-16T16:21:34",
    "reason": "RSI crossed below 60 (RSI=56.6), below EMA100",
    "order_id": "...",
    "tp_order_id": "...",
    "sl_order_id": "...",
    "unrealized_pnl": 0.9443,
    "mark_price": 85976.8
  },
  "total_trades": 0,
  "total_pnl": 0,
  "winning_trades": 0,
  "daily_pnl": 0,
  "daily_trades": 0,
  "last_rsi": 56.55,
  "last_trade_date": "2025-12-16"
}
```

### 5.3 장애 대응

| 상황 | 증상 | 대응 |
|------|------|------|
| **봇 재시작** | 포지션 상태 불일치 | 자동 동기화 (sync_position_with_exchange) |
| **API 오류** | 연결 실패 | 30초 대기 후 재시도 |
| **일일 손실 한도** | 15% 손실 | 자동 거래 중단, 다음 날 재개 |
| **외부 청산** | 수동 청산 감지 | 실제 청산가 조회 후 PnL 계산 |

---

## 6. 코드 품질 평가

### 6.1 장점

| 항목 | 평가 |
|------|------|
| **Position Sync** | ✅ 재시작 시 거래소 동기화 완비 |
| **Actual Fill Price** | ✅ 실제 체결가 기반 TP/SL 계산 |
| **Error Handling** | ✅ try-except로 API 오류 처리 |
| **State Backup** | ✅ 상태 변경 전 백업 생성 |
| **Logging** | ✅ 상세한 로깅 (파일 + 콘솔) |
| **Config Separation** | ✅ YAML 설정 파일 분리 |

### 6.2 개선 가능 사항

| 항목 | 현재 | 개선안 |
|------|------|--------|
| **모니터링** | 없음 | 별도 모니터링 스크립트 추가 |
| **알림** | 없음 | Telegram/Discord 알림 연동 |
| **백테스트 일치** | 미확인 | 백테스트와 프로덕션 로직 검증 |
| **재진입 로직** | 없음 | TP 후 재진입 옵션 추가 |

---

## 7. 백테스트 vs 프로덕션 로직 비교

### 7.1 Entry Signal

| 항목 | 백테스트 | 프로덕션 | 일치 |
|------|----------|----------|------|
| RSI Period | 14 | 14 | ✅ |
| RSI Long Threshold | 40 | 40 | ✅ |
| RSI Short Threshold | 60 | 60 | ✅ |
| EMA Period | 100 | 100 | ✅ |
| RSI Crossover Check | `rsi > threshold AND rsi_prev <= threshold` | 동일 | ✅ |
| EMA Filter | `close > ema` for LONG | 동일 | ✅ |

### 7.2 Exit Logic

| 항목 | 백테스트 | 프로덕션 | 일치 |
|------|----------|----------|------|
| TP % | 3.0% | 3.0% - 0.05% (슬리피지) | ⚠️ 약간 다름 |
| SL % | 2.0% | 2.0% - 0.05% (슬리피지) | ⚠️ 약간 다름 |
| Fee 계산 | 0.05% x 2 | 0.05% x 2 | ✅ |
| Cooldown | 4 candles | 4 candles (60분) | ✅ |

**참고**: 프로덕션에서 슬리피지 버퍼 0.05%를 적용하여 실제 TP는 2.95%, SL은 1.95%로 설정됨. 이는 보수적 접근으로 백테스트 대비 약간 불리함.

---

## 8. 결론

### 8.1 전략 평가

| 평가 항목 | 점수 | 코멘트 |
|----------|------|--------|
| **검증 수준** | ★★★★★ | Walk-Forward 86% 통과, p=0.013 |
| **코드 품질** | ★★★★☆ | 안정적, 동기화 로직 완비 |
| **리스크 관리** | ★★★★☆ | 일일 손실 한도, 쿨다운 적용 |
| **운영 편의성** | ★★★☆☆ | 모니터링/알림 추가 필요 |
| **수익성** | ★★★★☆ | +120.8% (7개월), 안정적 |

### 8.2 추천 사항

1. **현재 전략 유지**: Walk-Forward 검증 통과한 유일한 전략
2. **모니터링 강화**: 별도 모니터링 스크립트 추가 권장
3. **알림 시스템**: 진입/청산 시 알림 연동 권장
4. **정기 검증**: 월 1회 성과 검토 및 파라미터 재검증

---

## 관련 파일

| 파일 | 설명 |
|------|------|
| `scripts/production/rsi_trend_filter_bot.py` | 프로덕션 봇 코드 |
| `config/rsi_trend_filter_config.yaml` | 설정 파일 |
| `results/rsi_trend_filter_bot_state.json` | 상태 파일 |
| `results/rsi_trend_filter_walkforward_20251216_031126.csv` | Walk-Forward 결과 |
| `results/rsi_deep_research_20251216_032642.csv` | RSI 전략 연구 결과 |
| `results/best_strategy_validation_20251216_032813.csv` | 최적 전략 검증 결과 |
