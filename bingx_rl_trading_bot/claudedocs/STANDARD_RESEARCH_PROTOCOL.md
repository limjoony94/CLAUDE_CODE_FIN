# Trading Strategy Research Standard Protocol

**Version**: 2.0
**Created**: 2026-01-19
**Updated**: 2026-02-11
**Status**: ACTIVE

---

## Overview

본 문서는 거래 전략 연구의 모든 단계에서 반드시 준수해야 하는 표준 프로토콜을 정의합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STRATEGY RESEARCH LIFECYCLE                       │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────────┤
│ Phase 1  │ Phase 2  │ Phase 3  │ Phase 4  │ Phase 5  │ Phase 6     │
│ 데이터   │ 전략     │ 백테스트 │ 검증     │ 배포     │ 모니터링   │
│ 준비     │ 개발     │ 실행     │ 프레임워크│ 준비    │ & 리뷰     │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────────┘
```

---

## Phase 1: Data Preparation (데이터 준비)

### 1.1 데이터 요구사항

| 항목 | 최소 기준 | 권장 |
|------|----------|------|
| **기간** | 90일 | 270일+ |
| **캔들 수** | 25,920 (5m×90일) | 77,760+ |
| **데이터 소스** | BingX API | 검증된 거래소 |
| **타임프레임** | 5m (프로덕션 기준) | 5m |
| **Ground Truth** | `data/btc_5m_270days_reclassified.csv` | 270일 GT 분류 |

### 1.2 데이터 품질 체크리스트

```python
def validate_data(df):
    """데이터 품질 검증"""
    checks = {
        'no_nulls': df.isnull().sum().sum() == 0,
        'no_duplicates': df.index.is_unique,
        'chronological': df.index.is_monotonic_increasing,
        'no_gaps': check_candle_gaps(df),  # 누락된 캔들 체크
        'valid_ohlc': (df['high'] >= df['low']).all(),
        'volume_positive': (df['volume'] >= 0).all(),
    }
    return all(checks.values()), checks
```

### 1.3 데이터 저장 규칙

```
data/
├── btc_5m_270days_reclassified.csv  # Ground Truth 270일 데이터 (프로덕션 + 연구 공용)
├── raw/                              # 원본 데이터 (수정 금지)
│   └── btc_5m_YYYYMMDD.csv
└── cache/                            # 임시 캐시 (삭제 가능)
```

---

## Phase 2: Strategy Development (전략 개발)

### 2.1 신호 정의 규칙

#### Entry Signal 정의
```python
def generate_signal(df, i):
    """
    신호 생성 함수 표준 형식

    Args:
        df: OHLCV DataFrame
        i: 현재 인덱스 (마지막 완성된 캔들)

    Returns:
        'LONG', 'SHORT', or None

    주의:
        - i 인덱스의 데이터만 사용 (shift(1) 허용, shift(-1) 금지)
        - 미래 데이터 참조 절대 금지
    """
    current = df.iloc[i]
    prev = df.iloc[i-1] if i > 0 else None

    # 신호 로직...
    return signal
```

### 2.2 Look-Ahead Bias 방지 규칙

| 패턴 | 상태 | 설명 |
|------|------|------|
| `df['col'].shift(-1)` | ❌ 금지 | 미래 데이터 참조 |
| `df['col'].shift(-N)` | ❌ 금지 | N봉 미래 데이터 |
| `df.rolling(n, center=True)` | ❌ 금지 | 양방향 rolling |
| `df['col'].shift(1)` | ✅ 허용 | 과거 데이터만 |
| `df.rolling(n).mean()` | ✅ 허용 | 과거 방향 rolling |

### 2.3 Entry/Exit 타이밍 표준

```
신호 발생 시점: 캔들 N 종료 시점 (N의 close 확정 후)
Entry 실행:    캔들 N+1 시작 시점 (N+1의 open 가격)
Exit 체크:     캔들 N+2부터 (high/low로 TP/SL 체크)
```

```python
# ✅ 올바른 Entry 타이밍
signal_candle = df.iloc[-2]  # 마지막 완성된 캔들에서 신호
entry_price = df.iloc[-1]['open']  # 다음 캔들 시가로 진입

# ❌ 잘못된 Entry 타이밍
entry_price = signal_candle['close']  # 신호 캔들 종가 = Look-Ahead!
```

### 2.4 TP/SL 설정 표준

```python
def calculate_tp_sl(entry_price, direction, tp_pct, sl_pct):
    """TP/SL 계산"""
    if direction == 'LONG':
        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 - sl_pct / 100)
    else:  # SHORT
        tp_price = entry_price * (1 - tp_pct / 100)
        sl_price = entry_price * (1 + sl_pct / 100)
    return tp_price, sl_price
```

---

## Phase 3: Backtest Execution (백테스트 실행)

### 3.1 Position Sizing 표준

**방식: Compound (복리)**

```python
def calculate_position_size(balance, position_pct, leverage, price, max_usd=None):
    """
    Position Sizing 표준 공식

    Args:
        balance: 현재 잔고
        position_pct: 포지션 비율 (0.0 ~ 1.0)
        leverage: 실효 레버리지
        price: 현재 가격
        max_usd: 최대 포지션 크기 (USD)

    Returns:
        quantity: 포지션 수량
    """
    position_value = balance * position_pct
    if max_usd:
        position_value = min(position_value, max_usd)

    quantity = (position_value * leverage) / price
    return quantity
```

### 3.2 수수료 계산 표준

| 구분 | 비율 | 적용 시점 |
|------|------|----------|
| Entry Fee | 0.05% (Taker) | 진입 시 |
| Exit Fee | 0.05% (Taker) | 청산 시 |
| **Total** | **0.10%** | PnL에서 차감 |

```python
def calculate_pnl(entry_price, exit_price, direction, leverage, fee_pct=0.05):
    """PnL 계산 (수수료 포함)"""
    if direction == 'LONG':
        raw_pnl_pct = (exit_price / entry_price - 1) * 100
    else:
        raw_pnl_pct = (1 - exit_price / entry_price) * 100

    # 레버리지 적용
    leveraged_pnl = raw_pnl_pct * leverage

    # 수수료 차감 (양방향)
    total_fee = fee_pct * 2 * leverage
    net_pnl = leveraged_pnl - total_fee

    return net_pnl
```

### 3.3 Exit 감지 표준

```python
def check_exit(bar, position):
    """
    Exit 체크 (Intrabar High/Low 사용)

    Args:
        bar: 현재 캔들 데이터
        position: 포지션 정보 (direction, tp_price, sl_price)

    Returns:
        (exit_price, exit_reason) or (None, None)
    """
    if position['direction'] == 'LONG':
        # LONG: High가 TP에 도달하면 TP, Low가 SL에 도달하면 SL
        if bar['high'] >= position['tp_price']:
            return position['tp_price'], 'TP'
        elif bar['low'] <= position['sl_price']:
            return position['sl_price'], 'SL'
    else:  # SHORT
        # SHORT: Low가 TP에 도달하면 TP, High가 SL에 도달하면 SL
        if bar['low'] <= position['tp_price']:
            return position['tp_price'], 'TP'
        elif bar['high'] >= position['sl_price']:
            return position['sl_price'], 'SL'

    return None, None
```

### 3.4 Slippage 모델링 (권장)

```python
SLIPPAGE_BUFFER_PCT = 0.02  # 0.02% 슬리피지 버퍼

def apply_slippage(price, direction, is_entry=True):
    """슬리피지 적용"""
    if is_entry:
        # 진입 시: 불리한 방향
        if direction == 'LONG':
            return price * (1 + SLIPPAGE_BUFFER_PCT / 100)
        else:
            return price * (1 - SLIPPAGE_BUFFER_PCT / 100)
    else:
        # 청산 시: 불리한 방향
        if direction == 'LONG':
            return price * (1 - SLIPPAGE_BUFFER_PCT / 100)
        else:
            return price * (1 + SLIPPAGE_BUFFER_PCT / 100)
```

---

## Phase 4: Validation Framework (검증 프레임워크)

### 4.1 Two-Tier Validation (필수)

#### Type 1: Signal Quality Verification

**목적**: 신호 자체의 예측력 검증 (포지션 상태 무시)

**max_bars 설정 가이드**:
- `max_bars`는 TP/SL에 도달하기까지 충분한 시간을 허용해야 함
- 일반적 권장: `max_bars = 500` (5m 기준 약 42시간)
- 계산 기준: TP% / 평균 변동성(bar당) × 2 (여유 배수)
- 예: TP 2.5%, 변동성 0.1%/bar → 25 × 2 = 50 bars (최소)

```python
def type1_validation(df, signal_func, tp_pct, sl_pct, max_bars=500):
    """
    Type 1: 모든 신호에 대해 독립 평가
    - Entry: 신호 다음 봉 Open
    - Exit: High/Low로 TP/SL 체크
    """
    results = []

    for i in range(1, len(df) - max_bars):
        signal = signal_func(df, i)
        if signal is None:
            continue

        entry_idx = i + 1
        entry_price = df.iloc[entry_idx]['open']
        tp_price, sl_price = calculate_tp_sl(entry_price, signal, tp_pct, sl_pct)

        # Exit 탐색 (entry_idx + 1부터!)
        for j in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
            bar = df.iloc[j]
            exit_price, reason = check_exit(bar, {
                'direction': signal,
                'tp_price': tp_price,
                'sl_price': sl_price
            })

            if exit_price:
                pnl = calculate_pnl(entry_price, exit_price, signal, 1.0)
                results.append({'signal': signal, 'pnl': pnl, 'reason': reason})
                break

    # 통계
    wins = sum(1 for r in results if r['pnl'] > 0)
    return {
        'total_signals': len(results),
        'win_rate': wins / len(results) * 100 if results else 0,
        'expected_value': sum(r['pnl'] for r in results) / len(results) if results else 0,
        'long_wr': calculate_wr(results, 'LONG'),
        'short_wr': calculate_wr(results, 'SHORT'),
    }
```

**통과 기준**:

| 메트릭 | 최소 | 권장 |
|--------|------|------|
| 총 신호 수 | ≥ 100 | ≥ 300 |
| 승률 | ≥ 50% | ≥ 55% |
| 기대값 | > 0% | > 0.5% |
| LONG 수익 | ≥ 0 | > 0 |
| SHORT 수익 | ≥ 0 | > 0 |

#### Type 2: Actual Trading Verification

**목적**: 실제 거래 조건에서 성과 검증 (포지션 있으면 신호 무시)

```python
def type2_validation(df, signal_func, tp_pct, sl_pct, leverage, position_pct, fee_pct):
    """
    Type 2: 실제 거래 시뮬레이션
    - 한 번에 하나의 포지션만
    - 복리 효과 반영
    - 수수료 포함
    """
    balance = 100
    position = None
    trades = []

    for i in range(1, len(df)):
        bar = df.iloc[i]

        # 1. 포지션 청산 체크 (먼저!)
        if position is not None:
            exit_price, reason = check_exit(bar, position)
            if exit_price:
                pnl_pct = calculate_pnl(
                    position['entry_price'], exit_price,
                    position['direction'], leverage, fee_pct
                )
                pnl_dollar = balance * position_pct * (pnl_pct / 100)
                balance += pnl_dollar
                trades.append({
                    'direction': position['direction'],
                    'pnl_pct': pnl_pct,
                    'reason': reason,
                    'balance': balance
                })
                position = None

        # 2. 신규 진입 체크 (포지션 없을 때만)
        if position is None:
            signal = signal_func(df, i - 1)  # 이전 봉에서 신호
            if signal:
                entry_price = bar['open']
                tp_price, sl_price = calculate_tp_sl(entry_price, signal, tp_pct, sl_pct)
                position = {
                    'direction': signal,
                    'entry_price': entry_price,
                    'tp_price': tp_price,
                    'sl_price': sl_price
                }

    return {
        'total_pnl_pct': (balance - 100),
        'num_trades': len(trades),
        'win_rate': sum(1 for t in trades if t['pnl_pct'] > 0) / len(trades) * 100 if trades else 0,
        'max_drawdown': calculate_max_drawdown(trades),
        'trades': trades
    }
```

**통과 기준**:

| 메트릭 | 최소 | 권장 |
|--------|------|------|
| Total PnL | > 0% | > 30% |
| Walk-Forward | ≥ 3/5 (60%) | ≥ 4/5 (80%) |
| Monte Carlo p-value | < 0.02 | < 0.01 |
| Max Drawdown | < 50% | < 30% |
| Edge (Excess WR) | > 5pp over random | > 10pp |

### 4.2 Walk-Forward Analysis

```python
def walk_forward_validation(df, signal_func, params, n_folds=5):
    """
    Walk-Forward 검증
    - 데이터를 n_folds로 분할 (기본 5-fold)
    - 각 fold에서 독립적으로 백테스트
    - 3/5 이상 수익이면 PASS
    """
    fold_size = len(df) // n_folds
    results = []

    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size
        fold_df = df.iloc[start_idx:end_idx].copy()

        result = type2_validation(fold_df, signal_func, **params)
        results.append({
            'fold': i + 1,
            'pnl': result['total_pnl_pct'],
            'trades': result['num_trades'],
            'pass': result['total_pnl_pct'] > 0
        })

    pass_count = sum(1 for r in results if r['pass'])
    return {
        'pass_rate': pass_count / n_folds,
        'passes': pass_count,
        'total_folds': n_folds,
        'is_valid': pass_count >= n_folds / 2,
        'folds': results
    }
```

### 4.3 Monte Carlo Simulation (필수)

**방법: Sign Randomization (부호 무작위화)**

```python
def monte_carlo_sign_randomization(trades, n_simulations=10000):
    """
    Monte Carlo Sign Randomization Test
    - 각 거래의 부호(+/-)를 무작위로 뒤집어 10,000회 시뮬레이션
    - 실제 PnL이 시뮬레이션 분포의 상위 2%에 있으면 PASS (p < 0.02)
    - 거래 순서 섞기(shuffle)가 아닌 부호 뒤집기(sign flip)가 표준
    """
    import numpy as np

    actual_pnl = sum(t['pnl_pct'] for t in trades)
    pnl_values = np.array([t['pnl_pct'] for t in trades])

    count_ge = 0
    for _ in range(n_simulations):
        signs = np.random.choice([-1, 1], size=len(pnl_values))
        sim_pnl = np.sum(pnl_values * signs)
        if sim_pnl >= actual_pnl:
            count_ge += 1

    p_value = count_ge / n_simulations
    return {
        'p_value': p_value,
        'is_valid': p_value < 0.02,  # 필수: p < 0.02
        'is_strong': p_value < 0.01  # 권장: p < 0.01
    }
```

**Edge Test (v1.26.2+ 추가 검증)**

```python
def edge_test(trades, tp_pct, sl_pct):
    """
    Random Walk 대비 초과 승률 검증
    - Random walk WR = SL / (TP + SL) (distance-based baseline)
    - 초과 WR (edge) > 5pp이면 genuine edge
    - ALL profit comes from edge, not distance
    """
    random_wr = sl_pct / (tp_pct + sl_pct)
    actual_wr = sum(1 for t in trades if t['pnl_pct'] > 0) / len(trades)
    edge_pp = (actual_wr - random_wr) * 100

    return {
        'random_wr': random_wr * 100,
        'actual_wr': actual_wr * 100,
        'edge_pp': edge_pp,
        'has_edge': edge_pp > 5.0  # 최소 5pp 초과 승률
    }
```

---

## Phase 5: Deployment Preparation (배포 준비)

### 5.1 Pre-Deployment Checklist

```markdown
## 배포 전 체크리스트

### 검증 완료
- [ ] Type 1 통과: 승률 ___%, 기대값 ___%
- [ ] Type 2 통과: PnL ___%, DD ___%
- [ ] Walk-Forward: ___/8 passes
- [ ] Monte Carlo: ___% 수익 확률

### 코드 품질
- [ ] Look-Ahead Bias 감사 완료
- [ ] 예외 처리 적용 (ccxt.NetworkError 등)
- [ ] 로깅 구현
- [ ] Crash Recovery 구현

### 운영 준비
- [ ] API 키 설정
- [ ] 레버리지 설정 (거래소: 10x, 실효: 3x)
- [ ] 포지션 모드: One-Way
- [ ] 마진 모드: Cross
- [ ] 배치 스크립트 테스트

### 문서화
- [ ] 전략 파라미터 기록
- [ ] 백테스트 결과 기록
- [ ] CLAUDE.md 업데이트
- [ ] 버전 번호 증가
```

### 5.2 Configuration Template

```yaml
# config/pattern_5m_config.yaml (v1.27.0 기준)
symbol: 'BTC-USDT'
timeframe: '5m'
leverage: 3                    # 실효 레버리지
exchange_leverage: 10          # 거래소 설정
position_mode: 'one-way'
margin_mode: 'crossed'
position_size_pct: 95

strategy:
  # TP/SL은 PATTERN_OPTIMAL_TPSL에서 패턴별 관리 (Uniform TP 70%)
  default_tp_pct: 1.0          # fallback only
  default_sl_pct: 1.0          # fallback only
  cooldown_candles: 0

  # Context Filters
  use_context_filter: true     # RSI/Vol/Trend

risk:
  max_daily_loss_pct: 7        # v1.27.0: 10% → 7%
  max_position_size_usd: 10000
  consecutive_loss_pause: 3    # v1.27.0: 3연속 손실 → 600초 일시정지
  consecutive_loss_cooldown: 600

# 메타데이터
version: "1.27.0"
validated_date: "2026-02-10"
backtest_results:
  patterns: 52                 # 32L + 20S
  total_pnl: 911.1
  win_rate: 83.7
  max_drawdown: 16.2
  profit_factor: 3.62
  trades: 386
  walk_forward: "5/5"
  mc_p_value: 0.0000
```

---

## Phase 6: Monitoring & Review (모니터링 & 리뷰)

### 6.1 Performance Tracking

```python
class PerformanceMetrics:
    """프로덕션 성과 추적"""

    def __init__(self):
        self.trades = []
        self.expected_win_rate = 84.0  # v1.27.0 백테스트 기대치
        self.expected_avg_win = 2.5
        self.expected_avg_loss = 2.0

    def update_trade(self, pnl_pct):
        self.trades.append(pnl_pct)
        self._check_drift()

    def _check_drift(self):
        """백테스트-프로덕션 괴리 감지"""
        if len(self.trades) < 10:
            return

        actual_wr = sum(1 for t in self.trades if t > 0) / len(self.trades) * 100
        drift = abs(actual_wr - self.expected_win_rate)

        if drift > 15:  # 15% 이상 괴리
            logger.warning(f"⚠️ Performance drift detected: {drift:.1f}%")
```

### 6.2 Backtest vs Production Comparison

| 메트릭 | 백테스트 (v1.27.0) | 프로덕션 | 허용 괴리 |
|--------|-------------------|----------|----------|
| 승률 | 83.7% | TBD | ±10% |
| PnL/MDD | 56.2x | TBD | ±30% |
| Profit Factor | 3.62 | TBD | ±20% |
| 거래 빈도 | ~43/월 (386/9mo) | TBD | ±30% |

### 6.3 Strategy Lifecycle

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  RESEARCH   │ →  │  VALIDATED  │ →  │ PRODUCTION  │
│  (개발중)    │    │  (검증완료)  │    │  (운영중)    │
└─────────────┘    └─────────────┘    └─────────────┘
                          │                   │
                          │                   ▼
                          │           ┌─────────────┐
                          │           │  DEGRADED   │
                          │           │ (성능 저하)  │
                          │           └─────────────┘
                          │                   │
                          ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │   PAUSED    │ ←  │  ARCHIVED   │
                   │  (일시정지)  │    │   (폐기)     │
                   └─────────────┘    └─────────────┘
```

---

## Quick Reference Card

### 핵심 규칙 요약

| 항목 | 표준 |
|------|------|
| **Entry 타이밍** | 신호 다음 봉 Open |
| **Exit 타이밍** | Intrabar High/Low |
| **Position Sizing** | Compound (복리) |
| **수수료** | 0.05% × 2 = 0.10% |
| **슬리피지** | 0.02% 버퍼 |
| **Look-Ahead 금지** | shift(-N), center=True |

### 검증 기준 요약

| 검증 | 필수 통과 조건 |
|------|---------------|
| **Type 1** | 신호 ≥100, 승률 ≥50%, 기대값 >0 |
| **Type 2** | PnL >0%, WF ≥3/5 (60%), DD <50% |
| **Monte Carlo** | Sign randomization p < 0.02 (10k sims) |
| **Edge Test** | Excess WR > 5pp over random walk baseline |

### 파일 위치

```
claudedocs/
├── STANDARD_RESEARCH_PROTOCOL.md                ← 본 문서
├── PRODUCTION_TRADING_LOGIC_ANALYSIS_20260204.md ← 트레이딩 로직 분석
└── [과거 연구 리포트]

scripts/analysis/                                ← 연구 스크립트 (45+)
├── uniform_tp_validation.py                     ← v1.27.0 핵심 검증
├── tp_sl_optimization_v1264.py                  ← v1.26.4 TP/SL 최적화
├── tp_sl_deep_validation.py                     ← 5-phase deep validation
└── ...

results/                                         ← 검증 결과 JSON
├── uniform_tp_validation.json
├── tp_sl_optimization_v1264.json
└── tp_sl_deep_validation.json
```

---

## Appendix: Validation Result Template

```markdown
# Strategy Validation Report: [전략명]

**Date**: YYYY-MM-DD
**Version**: X.X

## Type 1: Signal Quality
| Metric | Result | Status |
|--------|--------|--------|
| Total Signals | ___ | ≥100? |
| Win Rate | ___% | ≥50%? |
| Expected Value | ___% | >0%? |
| LONG WR | ___% | |
| SHORT WR | ___% | |

## Type 2: Actual Trading
| Metric | Result | Status |
|--------|--------|--------|
| Total PnL | ___% | >0%? |
| Trades | ___ | |
| Win Rate | ___% | |
| Max DD | ___% | <50%? |

## Walk-Forward (5 Folds)
| Fold | PnL | Status |
|------|-----|--------|
| 1 | ___% | PASS/FAIL |
| 2 | ___% | PASS/FAIL |
| 3 | ___% | PASS/FAIL |
| 4 | ___% | PASS/FAIL |
| 5 | ___% | PASS/FAIL |

**Pass Rate**: ___/5 (___%)

## Monte Carlo Sign Randomization (10,000 sims)
- p-value: ___ (<0.02?)

## Edge Test
- Random WR: ___% | Actual WR: ___% | Edge: ___pp (>5pp?)

## Final Decision
- [ ] APPROVED for Production
- [ ] NEEDS IMPROVEMENT
- [ ] REJECTED

**Reviewer**: ___
**Approved Date**: ___
```

---

**Document History**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-19 | Initial release |
| 2.0 | 2026-02-11 | v1.27.0 기준 업데이트: MC sign randomization 10k sims, WF 5-fold, Edge Test 추가, 데이터 270일, 검증 기준 강화 |
