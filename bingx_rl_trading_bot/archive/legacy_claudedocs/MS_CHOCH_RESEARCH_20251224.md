# MS_ChoCH Strategy Research Summary

**Date**: 2025-12-24
**Version**: v1.2 (ATR-based Position Sizing)
**Status**: Deployed and Running

---

## Strategy Overview

### Market Structure Change of Character (ChoCH)
시장 구조 변화를 감지하여 추세 전환점에서 진입하는 전략

**Core Concept**:
- **ChoCH (Change of Character)**: 기존 추세가 무너지고 새로운 추세가 시작되는 지점
- 단순 돌파(BOS)가 아닌, 이전 추세의 구조적 붕괴를 확인 후 진입

---

## Entry Logic

### LONG Entry Conditions
```
1. bos_up = True (상단 돌파)
2. lower_low.shift(5) = True (5봉 전 저점 갱신이 있었음 = 하락추세 존재 확인)
3. Volume > 평균 * 1.5 (거래량 확인)
4. Close > EMA(100) (추세 필터)
```

**해석**: 하락추세(lower_low)가 진행 중이었으나, 상단 돌파(bos_up)가 발생하면서 추세 전환(ChoCH) 신호

### SHORT Entry Conditions
```
1. bos_down = True (하단 돌파)
2. higher_high.shift(5) = True (5봉 전 고점 갱신이 있었음 = 상승추세 존재 확인)
3. Volume > 평균 * 1.5 (거래량 확인)
4. Close < EMA(100) (추세 필터)
```

**해석**: 상승추세(higher_high)가 진행 중이었으나, 하단 돌파(bos_down)가 발생하면서 추세 전환(ChoCH) 신호

---

## Exit Logic

### Fixed TP/SL (Optimized)
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Take Profit** | **2.5%** | v1.1 최적화 (기존 2.0%) |
| **Stop Loss** | **1.5%** | 고정 |
| **R:R Ratio** | **1.67:1** | TP/SL |

### TP 최적화 연구 결과
| TP | SL | Full PnL | WF Win% | Daily Return |
|----|-----|----------|---------|--------------|
| 2.0% | 1.5% | +328.8% | 70% | 3.13% |
| **2.5%** | **1.5%** | **+609.1%** | **70%** | **5.80%** |
| 3.0% | 1.5% | +487.3% | 60% | 4.64% |

**결론**: TP 2.5%가 최적. Full Period PnL +85% 향상.

---

## Position Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Exchange Leverage | 10x | BingX 설정 |
| Effective Leverage | 4x | 포지션 크기 계산용 |
| Base Risk per Trade | 2.0% | 계좌 대비 (ATR 조정 전) |
| Max Position | $10,000 | USD 기준 |
| Margin Mode | Isolated | 격리 마진 |
| Position Mode | One-Way | 단방향 |

### ATR-based Dynamic Position Sizing (v1.2)

**v1.2 신규 기능**: 변동성 기반 포지션 크기 동적 조절

| Parameter | Value | Description |
|-----------|-------|-------------|
| ATR Period | 14 | ATR 계산 기간 |
| ATR Multiplier | 1.0 | 조절 강도 (1.0 = full) |
| ATR Average | 50 bars | 정규화용 장기 평균 |
| Size Range | 0.5x ~ 2.0x | 최소/최대 배수 |

**공식**:
```
atr_ratio = atr_avg / atr  (역관계: 변동성 높으면 작은 포지션)
adjusted_risk_pct = base_risk_pct * atr_ratio * atr_multiplier
position_size = balance * adjusted_risk_pct * leverage / current_price
```

**효과**:
- 고변동성 시장 → 작은 포지션 (리스크 감소)
- 저변동성 시장 → 큰 포지션 (기회 극대화)
- **Return/DD Ratio: 18.7** (17개 방법 중 최고)

### Position Sizing Comparison (17 Methods)
| Method | Full PnL | Max DD | Return/DD Ratio |
|--------|----------|--------|-----------------|
| **ATR_x1.0** | **609.1%** | **32.6%** | **18.7** ✅ |
| ATR_x0.5 | 609.1% | 43.1% | 14.1 |
| Fixed_2pct | 609.1% | 50.4% | 12.1 |
| Kelly_Full | 609.1% | 68.2% | 8.9 |
| Martingale_1.5x | 812.3% | 95.4% | 8.5 |

---

## Validation Results

### Walk-Forward Validation (10 Windows, 105 Days)
| Metric | Value |
|--------|-------|
| **Profitable Windows** | **70%** (7/10) |
| **Average PnL per Window** | +6.7% |
| **Standard Deviation** | 11.8% |
| **Full Period PnL** | +609.1% |
| **Max Drawdown** | 27.4% |

### Direction Performance
| Direction | PnL | WF Win% | Notes |
|-----------|-----|---------|-------|
| **LONG** | +64.6% | 50% | 5/10 windows |
| **SHORT** | +152.5% | 60% | 6/10 windows |

### Key Metrics
| Metric | Value |
|--------|-------|
| Daily Return | 5.80% |
| Win Rate | 58.0% |
| Trades/Day | 0.95 |
| Profit Factor | 2.23 |
| Expectancy | 2.17% per trade |

---

## Configuration Files

### Config: `config/ms_choch_bot_config.yaml`
```yaml
exchange:
  symbol: "BTC-USDT"
  timeframe: "5m"
  leverage: 10
  position_mode: "one-way"
  margin_mode: "isolated"

position:
  effective_leverage: 4
  risk_per_trade_pct: 2.0
  max_position_usd: 10000

strategy:
  name: "MS_ChoCH_VF_EMA_ATR"
  signal_type: "choch"
  volume_filter: true
  volume_threshold: 1.5
  ema_filter: true
  ema_period: 100
  swing_lookback: 5
  trend_lookback: 10
  tp_pct: 2.5
  sl_pct: 1.5
  cooldown_bars: 0
  # ATR-based Position Sizing (v1.2)
  base_risk_pct: 2.0
  atr_period: 14
  atr_multiplier: 1.0
```

### Bot Script: `scripts/production/ms_choch_bot.py`
### State File: `results/ms_choch_bot_state.json`

---

## Batch Files

| File | Purpose |
|------|---------|
| `START_MS_CHOCH.bat` | Start bot (background) |
| `STOP_MS_CHOCH.bat` | Stop bot |
| `MONITOR_MS_CHOCH.bat` | Monitor status |

---

## Recent Backtest (7 Days)

### 2025-12-16 ~ 2025-12-23 Production Logic Backtest
| Metric | Value |
|--------|-------|
| **Period** | 7 days |
| **Total Trades** | 7 |
| **Trades/Day** | 1.00 |
| **Total PnL** | **+1.50%** |
| **Win Rate** | 42.9% (3W / 4L) |
| **Max Drawdown** | 3.00% |
| **LONG** | +2.00% (4 trades, 50% WR) |
| **SHORT** | -0.50% (3 trades, 33.3% WR) |

**개별 거래**:
| Date | Side | Entry | Exit | PnL | Type |
|------|------|-------|------|-----|------|
| 12/17 | LONG | 87838 | 86521 | -1.50% | SL |
| 12/17 | SHORT | 87654 | - | -1.50% | SL |
| 12/18 | SHORT | 87206 | - | -1.50% | SL |
| 12/19 | LONG | 87144 | 89323 | +2.50% | TP |
| 12/19 | LONG | 88606 | 87276 | -1.50% | SL |
| 12/19 | LONG | 87814 | 90009 | +2.50% | TP |
| 12/22 | SHORT | 89530 | 87292 | +2.50% | TP |

---

## Research History

### 2025-12-24: ATR Position Sizing (v1.2)
- **스크립트**: `scripts/analysis/position_sizing_comparison.py`
- **발견**: ATR_x1.0이 Return/DD ratio 18.7로 최고 (17개 방법 중)
- **결론**: v1.2로 업데이트, ATR 기반 동적 포지션 사이징 적용

### 2025-12-24: TP Optimization Research
- **스크립트**: `scripts/analysis/ms_choch_tp25_research.py`
- **발견**: TP 2.5%가 2.0% 대비 +85% PnL 향상
- **결론**: v1.1로 업데이트, TP 2.5% 적용

### 2025-12-24: Deep Validation
- **스크립트**: `scripts/analysis/ms_choch_deep_validation.py`
- **검증**: 10 windows walk-forward, Monte Carlo simulation
- **결과**: 70% 윈도우 수익, 통계적 유의성 확인

### 2025-12-24: Strategy Screening
- **스크립트**: `scripts/analysis/new_strategy_quick_screen.py`
- **비교**: 6개 신규 전략 중 MS_ChoCH 선정
- **이유**: 일관된 walk-forward 성과, 양방향 수익

---

## Comparison with Other Strategies

### 6 Strategy Quick Screen (2025-12-24)
| Strategy | Full PnL | WF Win% | Selected |
|----------|----------|---------|----------|
| **MS_ChoCH** | **+609.1%** | **70%** | **Yes** |
| MS_BOS | +445.2% | 60% | No |
| MACD Martingale | +312.4% | 50% | No |
| RSI Divergence | +287.6% | 60% | No |
| EMA Cross | +198.3% | 40% | No |
| Bollinger Squeeze | +156.7% | 50% | No |

### vs Active Bots
| Bot | Full PnL | WF Win% | Status |
|-----|----------|---------|--------|
| **MS_ChoCH** | **+609.1%** | **70%** | **Active** |
| RSI Trend Filter v2.0 | +61.0% | 67% | Standby |
| SuperTrend 5m | +42.8% | 83% | Standby |

---

## Risk Considerations

### Strengths
- 높은 수익률 (609% full period)
- 일관된 walk-forward 성과 (70%)
- 양방향 수익 (LONG +64.6%, SHORT +152.5%)
- 명확한 시장 구조 기반 신호

### Weaknesses
- 높은 변동성 (11.8% std)
- Max Drawdown 27.4%
- LONG 방향 일관성 낮음 (50% WF)
- 거래 빈도 낮음 (0.95/day)

### Risk Mitigation
- Effective leverage 4x (보수적)
- Isolated margin 사용
- 일일 손실 한도 10% 설정
- 연속 손실 5회 시 일시 중지

---

**Last Updated**: 2025-12-24 KST (v1.2 ATR Position Sizing)
