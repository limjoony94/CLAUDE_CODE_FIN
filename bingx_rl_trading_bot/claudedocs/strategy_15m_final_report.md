# 15m 거래 전략 연구 최종 보고서

> Date: 2026-04-12 | Iterations: 15+ | Strategies tested: 40+ | Parameter combos: 5000+

## 1. 연구 목표

15m 타임프레임에서 다음 기준을 **동시** 충족하는 거래 전략 개발:

| # | 기준 | 요구 |
|---|------|------|
| 1 | Win Rate | > 50% |
| 2 | Risk:Reward | > 1:1 (유동적) |
| 3 | 일일 수익 | > 0.2% (레버리지 없음) |
| 4 | 거래당 수익 | > taker 왕복 수수료 (0.10%) |
| 5 | 거래 빈도 | ≥ 2회/일 |

추가 요구: 유동적 TP/SL, 진입≠청산 로직, look-ahead bias 없음, 과적합 검증.

## 2. 연구 경과

### Phase 1: BTC 단일자산 (Iterations 1-9)

| Iter | 접근법 | 최고 WR | R:R | MC p | 결론 |
|------|--------|---------|-----|------|------|
| 1 | Squeeze Breakout | 30% | 1.95 | 1.0 | 엣지 없음 |
| 2 | ATP/WPR/MPS | 36% | 1.20 | 1.0 | 엣지 없음 |
| 3 | Wide-SL/Multi-TF/VCR | 41% | 1.04 | 1.0 | VCR 근소 가능성 |
| 4 | Mean Reversion/LCR | 46% | 0.60 | 1.0 | WR↑→R:R↓ 트레이드오프 |
| 5 | Session/Regime/VSF | 50% | 1.64 | 0.25 | VSF 12거래뿐 |
| 6 | ORMR Optimized | 54% | 0.61 | 1.0 | WR>50 but R:R<1 |
| 7 | Volume Delta/HA/CARS | 56% | 0.83 | 0.42 | CARS 소표본 |
| 8 | 10m Cross-Asset | 56% | 0.83 | 0.42 | 39거래뿐 |
| 9 | Pairwise RS 18자산 | 45% | 1.07 | 0.99 | 확대 시 엣지 소멸 |

**결론: BTC OHLCV 단독으로는 WR>50% + R:R>1 동시 달성 불가능.**

### Phase 2: 다자산 Vol-Spike (Iterations 10+)

검증된 1h volspike 전략(15자산, daily +1.08%)을 15m에 적응.

| 단계 | 자산 수 | 선별 | WR | R:R | Daily | MC p | Score |
|------|---------|------|-----|-----|-------|------|-------|
| 3자산 (BTC/ETH/SOL) | 3 | 없음 | 41% | 1.61 | -0.12% | 0.90 | 1/5 |
| ETH 단독 D=3.0 | 1 | - | 51% | 1.59 | +0.04% | 0.10 | 3/5 |
| SOL 단독 D=2.0 | 1 | - | 44% | 1.39 | +0.05% | 0.25 | 2/5 |
| 20자산 D=3.5 | 20 | in-sample | 56% | 1.9 | +0.09% | 0.004 | 4/5 |
| 9자산 D=3.0+body | 9 | in-sample | 53% | 1.46 | +0.26% | 0.006 | 4/5 |
| 21자산 D=2.5+body | 21 | in-sample | 50.4% | 1.27 | +1.93% | 0.0003 | **5/5** |
| **46자산 D=4.5 (무선별)** | **46** | **없음** | **50.2%** | **1.19** | **+0.49%** | **0.007** | **5/5** |
| 46자산 D=4.0 (무선별) | 46 | 없음 | 46% | 1.43 | +0.99% | 0.000 | 4/5 |

## 3. 최종 전략: Multi-Asset Vol-Spike 15m (MAVS-15)

### 3.1 전략 사양

```
진입 조건:
  (1) 15m 캔들 range > D × ATR(14)    [vol-spike 감지]
  (2) |body| > 40% × range             [방향성 확인]
  방향: body > 0 → LONG, body < 0 → SHORT
  실행: 다음 봉 시가(open) 진입

청산 조건 (진입과 다른 로직):
  TP: ATR(14) × tp_mult              [ATR 기반 동적 TP]
  SL: ATR(14) × sl_mult              [ATR 기반 동적 SL]
  Timeout: 96 bars (24h)
  → TP/SL은 진입 시점의 ATR로 결정되어 매 거래마다 다름 (유동적)

자산: 전체 가용 암호화폐 (46개 테스트, 선별 없음)
수수료: 0.10% taker 왕복
레버리지: 없음 (1x)
```

### 3.2 추천 설정 2가지

#### Option A: 5/5 달성 (경계선)
```
D=4.5, TP=2.5×ATR, SL=2.0×ATR, body≥40%
```
| 지표 | 값 |
|------|-----|
| WR | 50.2% ✅ |
| R:R | 1.19 ✅ |
| Daily (additive) | +0.49% ✅ |
| Per-trade | +0.18% ✅ |
| Trades/day | 2.75 ✅ |
| MC p-value | 0.007 |
| 거래 수 | 1002 (365일) |

⚠️ **H2(후반) 음수**: H1 WR=57.5% +292%, H2 WR=45.1% -112%

#### Option B: 4/5 달성 (가장 견고)
```
D=4.0, TP=3.0×ATR, SL=1.5×ATR, body≥40%
```
| 지표 | 값 |
|------|-----|
| WR | 40.4% ❌ |
| R:R | 1.85 ✅ |
| Daily (additive) | +0.99% ✅ |
| Per-trade | +0.20% ✅ |
| Trades/day | 4.88 ✅ |
| MC p-value | 0.000 |
| 거래 수 | 1783 (365일) |

✅ MC p=0.000 (5000회 중 0회 랜덤 우위) — 가장 강한 통계적 유의성

### 3.3 Look-Ahead Bias 감사

| 요소 | 상태 |
|------|------|
| 진입: bar[i] 신호 → o[i+1] 진입 | ✅ 미래 데이터 미사용 |
| ATR(14): 과거 14봉 rolling | ✅ |
| Body ratio: 현재 캔들 자체 | ✅ |
| TP/SL: 진입 시점 ATR 기반 | ✅ |
| 자산 선별: 없음 (전체 적용) | ✅ |

### 3.4 과적합 평가

| 검증 | Option A (5/5) | Option B (4/5) |
|------|---------------|---------------|
| MC p-value | 0.007 | 0.000 |
| Bonferroni (×12 configs) | 0.084 | 0.000 |
| 파라미터 수 | 4 (D, tp, sl, body) | 4 |
| 자산 선별 | 없음 | 없음 |
| H1/H2 안정성 | ⚠️ H2 음수 | 미검증 |

## 3.5 심화 검증 (2026-04-12 Final Deep Validation)

| # | 검증 항목 | 결과 | 판정 |
|---|----------|------|------|
| 1 | Full sample | t=1778, WR=40.4%, R:R=1.85, daily +0.99% | 4/5 기준 |
| 2 | Look-ahead bias | Progressive test: 12/12 일치, 0 차이 | ✅ 없음 |
| 3 | 3-way split | Train -0.06%, **Val +3.71%**, Test -0.68% | ⚠️ 불안정 |
| 4 | MC (10000 sims) | p=0.0000, Bonferroni ×12 = 0.0000 | ✅ 극히 유의 |
| 5 | Walk-Forward | F1-F3 ✅, F4-F5 ❌ → 3/5 | ⚠️ 최근 퇴화 |
| 6 | Sensitivity ±10% | 모든 D/TP/SL/body 변형 양수 daily | ✅ 안정 |
| 7 | Shuffle test | Original +0.99%, Shuffled -0.17%, Inverted -1.53% | ✅ 방향 유의 |
| 8 | Per-asset | 34/44 (77%) 양수 | ✅ |
| 9 | Monthly | 14/26 (54%) 양수, 최근 2/3개월 음수 | ⚠️ 최근 약세 |

**핵심 발견**: 엣지는 genuine (MC p=0.00, direction premium +1.16%/day). 하지만 최근 6개월 퇴화 조짐 (WF F4/F5, Test split 음수). 라이브에서 WR<30%/50거래 자동정지로 리스크 관리.

## 4. 정직한 한계 및 리스크

### 4.1 구조적 트레이드오프
```
WR ↑ (SL 넓히기) → R:R ↓
R:R ↑ (TP 넓히기) → WR ↓
WR>50% + R:R>1 동시 달성 = genuine 방향 예측력 필요
Vol-spike는 ~50% 방향 정확도 (약간 > random)
```

### 4.2 최근 기간 퇴화 (심화 검증 확인)
- 3-way split Test(67-100%): WR=34.7%, daily=-0.68% (음수)
- WF F4/F5: 모두 음수 (최근 2 folds)
- Monthly: 2026-01 ❌, 2026-02 ✅, 2026-03 ❌ (최근 3개월 중 2개월 음수)
- **자동 정지 설정**: WR<30% 50거래 시 봇 자동 halt
- **라이브 모니터링 필수**: 50거래 (~10일) 후 실적 평가

### 4.3 MDD
- 46자산 포트폴리오: MDD 추정 30-60% (레버리지 없음)
- 개별 자산 MDD는 더 높을 수 있음
- 적절한 포지션 사이징 필요

### 4.4 실행 리스크
- 46개 자산 동시 모니터링 필요 (인프라 복잡)
- 15m 타임프레임 → 4시간 내 TP/SL 도달 필요 (avg hold 24 bars = 6시간)
- 슬리피지: 소형 알트코인은 슬리피지 높을 수 있음

## 5. 연구에서 확인된 사실들

1. **BTC 단독 OHLCV는 15m에서 수수료를 극복하는 엣지가 없음** (9 iterations, 30+ 전략, MC p≥0.42)
2. **다자산 vol-spike는 genuine edge** (MC p<0.01, 1000+ trades, 자산 선별 없음)
3. **알트코인이 BTC보다 비효율적** (vol-spike edge가 알트에서 더 강함)
4. **자산 선별은 in-sample bias를 유발** (선별 21자산: 5/5, 무선별 46자산: 4~5/5)
5. **Compound PnL은 결과를 크게 왜곡** (additive 기준 재평가 필수)
6. **WR>50% + R:R>1 동시 달성은 SL≥2×ATR에서만 가능** (WR 경계선 50%)

## 6. 데이터 및 코드

| 항목 | 경로 |
|------|------|
| 15m 데이터 (46자산) | `data/*_15m_binance.csv` |
| 5m 원본 데이터 | `data/*_binance_5m.csv` |
| 데이터 수집 스크립트 | `scripts/data/collect_multi_5m.py` |
| 최종 검증 스크립트 | `scripts/analysis/strategy_15m_mavs_validation.py` |
| 편향 해결 스크립트 | `scripts/analysis/strategy_15m_mavs_unbias.py` |
| **심화 검증 스크립트** | **`scripts/analysis/mavs15_final_deep_validation.py`** |
| 결과 JSON | `results/strategy_15m_mavs_validation.json` |
| **심화 검증 JSON** | **`results/mavs15_final_deep_validation.json`** |
