# Pattern 5m v1.27.0 종합 연구 리포트

> **작성일**: 2026-02-11
> **대상**: BTC 5m Pattern Trading Bot v1.27.0
> **범위**: v1.26.0 ~ v1.27.0 전체 연구 시리즈 (21개 연구, 16개 결과 JSON)
> **데이터**: 270일 (77,760 bars), Ground Truth 12-type classification

---

## 1. Executive Summary

v1.27.0 Pattern 5m 봇에 대해 21개 독립 연구를 수행한 결과, **현재 설정이 Walk-Forward OOS 기준으로 최적에 근접**해 있음을 확인했다. 테스트한 모든 개선 경로 — TP/SL 조정, 패턴 필터링, 신호 스케줄링, SL 축소, 패턴 추가/제거, 구조 변경 — 가 OOS에서 현재 대비 개선에 실패했다. 또한 과적합 위험에 대한 6-phase 심층 연구를 수행하여, 개별 패턴 수준의 불안정성에도 불구하고 포트폴리오 수준의 견고성을 데이터로 확인했다. 최종 연구로 라이브 31거래 데이터 기반 미래 리스크 정량화 및 전략 붕괴 탐지 시스템을 설계하여, 실전 운영에 필요한 모니터링 체계를 완성했다.

### 핵심 지표

| 지표 | v1.26.4 | v1.27.0 | 변화 |
|------|---------|---------|------|
| Patterns | 52 (32L+20S) | 52 (32L+20S) | - |
| PnL (270d) | +882.7% | **+911.1%** | +28.4pp |
| WR | 77.1% | **83.7%** | +6.6pp |
| MDD | 24.4% | **16.2%** | -8.2pp |
| PnL/MDD | 36.2x | **56.2x** | +55% |
| PF | 3.23 | **3.62** | +12% |
| Trades | 314 | **386** | +23% |
| Max Consec Loss | 3 | **2** | -1 |
| MC p-value | 0.0000 | **0.0000** | - |
| WF OOS | - | **5/5 pass** | - |

### 연구 결론 총괄

| # | 연구 | 결론 | WF OOS 영향 |
|---|------|------|-------------|
| 1 | Uniform TP 70% | **PASS** — 생산 적용 | PnL +3.2%, MDD -33% |
| 2 | Risk Management | **PASS** — 7% daily limit 적용 | MDD 보호 강화 |
| 3 | R:R >= 1.0 Optimization | **FAIL** — 22/52만 가능, MC 실패 | 적용 불가 |
| 4 | Dual-TP Stability | **INFO** — 70% 단일 TP가 최적 | Dual 대비 우위 |
| 5 | Trade Microstructure | **INFO** — SL 방향전환 80%, TP race 이해 | 현재 TP/SL 적정 확인 |
| 6 | Distance-Edge Decomposition | **INFO** — 100% 수익이 edge 기반 | Edge 실재 확인 |
| 7 | Context Filter v2 | **FAIL** — BH FDR 0/156 유의 | 필터 불필요 |
| 8 | MC Filter WF | **FAIL** — 제거 시 OOS 악화 | 52개 유지 |
| 9 | Signal Scheduling (F) | **FAIL** — FIFO 자연선택이 이미 최적 | 쿨다운 악화 |
| 10 | SL Reduction (B) | **FAIL** — WR↓ + MDD↑ 이중타격 | -22~30% |
| 11 | Pattern Correlation (D) | **FAIL** — Jaccard>0.2 = 0쌍 | 무의미 |
| 12 | Reverse Edge | **INFO** — 21/52 anti-pattern 확인 | 방향 편향 없음 |
| 13 | New Pattern WF | **FAIL** — 52개 통과하나 포트폴리오 악화 | PnL -37%, MDD +10.5pp |
| 14 | Pattern Combination | **FAIL** — In-sample 과적합 | OOS -13.1% |
| 15 | Multi-Position | **FAIL** — PnL/MDD 악화 | 12.8→12.2x |
| 16 | Time-Based Exit | **FAIL** — 조기 청산 시 WR 급락 | 500bar 최적 |
| 17 | ATR-Adaptive TP/SL | **FAIL** — 모든 배수에서 fixed 열등 | MC 실패까지 |
| 18 | Combined Best | **FAIL** — 단일 최선도 baseline 미달 | - |
| 19 | Overfitting & Forward Decay | **MODERATE RISK** — 개별 불안정, 포트폴리오 견고 | Bootstrap P(PnL>0)=100% |
| 20 | Future Risk Stress Test | **INFO** — 라이브 마진 정량화, 방향 edge 확인 | 방향 edge +751% |
| 21 | R:R & Detection Deep Study | **CRITICAL FIX** — break-even 오류 수정 (3pp→7pp) | 탐지 한계 규명 |

**생산 반영**: Uniform TP 70% + 7% daily limit + 3-loss pause만 적용. 나머지 20개 연구는 변경 불필요 확인. 모니터링: 3단계 티어드 알림 시스템 권고.

---

## 2. 연구 방법론

### 2.1 검증 프레임워크

모든 연구는 동일한 표준 프로토콜을 준수한다:

```
Entry:      신호 다음 봉 Open
Exit:       Intrabar High/Low (distance-based)
Position:   1-position-at-a-time (default), FIFO priority
Sizing:     Compound (복리), 3x leverage
Fee:        0.05% x 2 = 0.10%
Slippage:   0.02% buffer
MC Test:    Sign randomization (10,000 sims, p < 0.01)
WF:         Expanding-window, 5 periods, 4 OOS folds
```

### 2.2 Walk-Forward 구조

```
Period 1 (54d) | Period 2 (54d) | Period 3 (54d) | Period 4 (54d) | Period 5 (54d)
───────────────┼────────────────┼────────────────┼────────────────┼────────────────
[TRAIN]        → [OOS Fold 1]
[───── TRAIN ─────]             → [OOS Fold 2]
[──────────── TRAIN ────────────]               → [OOS Fold 3]
[─────────────────────── TRAIN ─────────────────]               → [OOS Fold 4]
```

### 2.3 판단 기준

| 기준 | 임계값 |
|------|--------|
| MC p-value | < 0.01 (개별), < 0.001 (포트폴리오) |
| WF fold pass | >= 4/5 (개별), 전 fold 양수 (포트폴리오) |
| Edge (excess WR) | >= 5pp over random walk baseline |
| OOS PnL | Baseline 대비 양수 |
| OOS MDD | Baseline 대비 악화 없음 |

---

## 3. Phase 1: 전략 최적화 연구 (v1.26.0 ~ v1.26.4)

### 3.1 R:R >= 1.0 Portfolio Migration (v1.26.0 ~ v1.26.1)

**목적**: WR 기반 포트폴리오(v1.25.4)에서 R:R 기반으로 전환

| 버전 | 연구 | 결과 |
|------|------|------|
| v1 | R:R >= 0.75 grid search | 78개 패턴, compound overflow (7.2e86) |
| v2 | Simple returns 수정 | MDD 99%, 여전히 과적합 |
| v3 | 1-pos + BE+15% safety | 58개 패턴, PnL +963.8%, MDD 19.8% |

- **핵심 교훈**: Compound return 시뮬레이션은 overflow 위험. Simple returns + 1-pos-at-a-time이 유일하게 안정적.
- **TP/SL Bias Research**: Random baseline binomial test → 77/78 패턴이 genuine edge 보유 확인.

### 3.2 MC/Edge Cleanup (v1.26.2)

58개 → 52개 패턴으로 정리:

| 제거 패턴 | 사유 | MC p-value |
|-----------|------|------------|
| BU-U-GS | MC >= 0.01 | 0.012 |
| GS-U-MU | MC >= 0.01 | 0.0102 |
| MU-MU-IH | MC >= 0.01 | 0.01 |
| GS-ST-U | MC >= 0.01 | 0.0119 |
| MD-MU-U | MC >= 0.01 | 0.011 |
| DN-IH-U | No edge | p=0.052 |

### 3.3 Full TP/SL Optimization (v1.26.4)

**방법**: 52개 패턴 전수 grid search → 32개 후보 → 5-phase deep validation

| Phase | 검증 | 결과 |
|-------|------|------|
| 1 | CV stability | 32개 CV 안정 |
| 2 | Plateau test | 그리드 이웃 안정성 확인 |
| 3 | Edge test | Excess WR >= 5pp |
| 4 | OOS validation | 31/32 통과 |
| 5 | Composite score | 31개 최종 승인 |

- **기각**: DN-MD-DN (excess WR 4.2pp < 5pp threshold)
- **결과**: PnL +882.7%, WR 77.1%, MDD 24.4%, PF 3.23

---

## 4. Phase 2: v1.27.0 핵심 연구 (2026-02-10)

### 4.1 Uniform TP 70% Validation

**가설**: 모든 TP를 70%로 축소하면 더 빠른 trade resolution → 더 많은 거래 기회

**Robustness Plateau 분석**:

| TP Scale | PnL | MDD | PnL/MDD | Trades |
|----------|-----|-----|---------|--------|
| 60% | +834.1% | 18.7% | 44.6x | 424 |
| 65% | +876.3% | 17.3% | 50.6x | 404 |
| **70%** | **+911.1%** | **16.2%** | **56.2x** | **386** |
| 75% | +895.8% | 17.8% | 50.3x | 368 |
| 80% | +882.7% | 18.9% | 46.7x | 348 |
| 85% | +866.4% | 20.1% | 43.1x | 332 |
| 100% (v1.26.4) | +882.7% | 24.4% | 36.2x | 314 |

- **67-85% 범위 전체**: PnL/MDD 44-56x (narrow optimum 아님, 넓은 plateau)
- **개별 패턴 MC**: 15/52 실패하나 포트폴리오 MC p=0.0000 → 분산효과
- **WF 8-phase validation**: 5/5 pass, edge 평균 +18.7pp 보존

**판정**: **PASS** — 생산 적용.

### 4.2 Risk Management Research

**Daily Loss Limit Sweep**:

| Limit | PnL | MDD | PnL/MDD | Trigger Days |
|-------|-----|-----|---------|-------------|
| 5% | +868.3% | 15.3% | 56.71x | 25 |
| **7%** | **+911.1%** | **16.2%** | **56.2x** | **15** |
| 10% (이전) | +882.7% | 24.4% | 36.2x | 2 |
| 15% | +882.7% | 24.4% | 36.2x | 0 |

**Kelly Criterion**: Full Kelly 53.2%, 현재 포지션 사이즈 5.1% (매우 보수적).
**Probability of Ruin**: 20% 손실 시 85.9% 파산 확률, 30% 시 23.6%.
**Consecutive Loss Pause**: 3회 연속 → 600초 대기가 최적.

**판정**: **PASS** — 7% daily limit + 3-loss pause 적용.

### 4.3 Trade Microstructure Analysis

**TP vs SL Race Dynamics** (3,941 trades):

| 지표 | 값 |
|------|-----|
| TP 평균 도달 시간 | 206 bars |
| SL 평균 도달 시간 | 192 bars |
| SL 근접 후 TP 도달 (near-miss) | 1.7% (13/755) |
| SL 손실 중 방향 전환 비율 | **79.9%** (603/755) |
| SL 평균 도달 거리 (TP 대비) | 28.2% |

**핵심 인사이트**: SL 손실의 80%가 방향 전환으로 발생. TP를 줄여도 이 손실은 회피 불가. SL은 이미 적정 거리.

### 4.4 Distance-Edge Decomposition

**이익의 원천 분석**:

```
Random Walk 기대 WR = SL / (TP + SL)
                    = 평균 53.2% (distance만으로)

실제 WR             = 83.7% (v1.27.0)
Edge (excess WR)    = +30.5pp

Distance-only PnL   = -262.5% (음수!)
Edge-only PnL       = +1,173.6% (전체 이익 초과)
```

**결론**: **수익의 100%가 genuine pattern edge에서 발생**. Distance bias(TP < SL)는 단독으로 손실을 유발. Edge가 전부.

### 4.5 Context Filter Deep Study (v2)

**범위**: 52 patterns x 3 dimensions (RSI zone, Volume, Trend) = 156 combinations

| Phase | 결과 |
|-------|------|
| Raw WR differences | 20개 패턴에서 >= 15pp 차이 관찰 |
| Fisher's exact test | 다수 nominal p < 0.05 |
| **Benjamini-Hochberg FDR** | **0/156 유의 (전멸)** |

- MU-H-MU volume filter: 37.8pp 차이 → BH 보정 후 비유의
- DN-BD-BD RSI filter: 28.3pp 차이 → BH 보정 후 비유의
- **모든 효과가 다중검정에서 소멸** → `PATTERN_CONTEXT_FILTERS = {}` 유지

### 4.6 R:R >= 1.0 Optimization

**Grid Search**: 52 patterns x TP/SL grid (R:R >= 1.0 constraint)

| 결과 | 수 |
|------|-----|
| 유효 조합 | 22/52 |
| MC 실패 | 29/52 |
| WF 실패 | 1/52 |

**결론**: R:R >= 1.0 강제 시 과반이 MC 실패. 현재 R:R < 1.0 구조가 genuine edge 유지에 필수적.

### 4.7 Dual-TP Stability Analysis

**50%@0.8x + 50%@1.0x** vs **100%@0.7x (Uniform)**:

- Dual-TP: 높은 partial WR이나 절대 PnL 열등
- Uniform 70%: PnL/MDD 56.2x로 단일 TP가 최적
- **판정**: Uniform 70% 단일 TP 유지.

---

## 5. Phase 3: 개선 시도 연구 (2026-02-11)

### 5.1 MC Filter Walk-Forward Validation

**가설**: 개별 MC 실패 패턴을 제거하면 OOS 개선될 것

| Strategy | OOS PnL | MDD | Fold 승률 |
|----------|---------|-----|----------|
| A: 52 전체 (baseline) | +640.7% | 18.0% | 4/4 |
| B: MC p>=0.01 제거 | +200.7% | 35.5% | 0/4 |
| C: MC p>=0.05 제거 | +351.5% | 23.5% | 0/4 |
| D: 3-패턴 제거 | +619.0% | 20.4% | 2/4 |
| E: Always-fail 17개 제거 | +647.2% | 23.0% | 2/4 |

**MC 테스트 불안정성**: 훈련 기간에 따라 실패 패턴이 완전히 달라짐.

| 훈련 기간 | MC 실패 수 (/52) |
|-----------|-----------------|
| 54일 (1기) | 46개 (88%) |
| 108일 (2기) | 36개 (69%) |
| 162일 (3기) | 28개 (54%) |
| 270일 (전체) | 23개 (44%) |
| 항상 통과 | **5개** (10%) |

**결론**: MC 테스트는 짧은 기간에서 불안정. 포트폴리오 분산효과가 개별 패턴 유의성보다 중요. **52개 전체 유지**.

### 5.2 F/B/D Portfolio Improvement Research

#### F — Signal Scheduling (쿨다운)

| 쿨다운 (bars) | PnL | vs Baseline |
|--------------|-----|-------------|
| 0 (현재) | +808.7% | baseline |
| 1 | +800.4% | -1.0% |
| 3 | +605.4% | -25.1% |
| 6 | +520.3% | -35.7% |

- 차단 신호의 84.9%가 평균 PnL이 실행된 것보다 낮음 → **FIFO 자연선택이 이미 효과적**
- Oracle DP (미래 정보) 상한: +2,034.4% (현재의 2.5배) — 도달 불가

#### B — SL Reduction

| 전략 | PnL | WR | MDD |
|------|-----|-----|-----|
| 현재 SL | +808.7% | 83.7% | 16.2% |
| SL x 0.9 | +643.4% | 80.1% | 22.8% |
| SL x 0.8 | +563.8% | 77.3% | 28.4% |
| SL cap 2.5% | +628.1% | 81.2% | 20.9% |

**결론**: SL 축소는 WR 하락 + MDD 증가 이중타격. Grid search 최적값 추가 축소 불필요.

#### D — Pattern Correlation

| 지표 | 값 |
|------|-----|
| Jaccard > 0.20 pairs | **0쌍** |
| 미러 패턴 | 1쌍 (BD-U-H / H-U-BD) |
| 미러 제거 영향 | PnL +1.1%, MDD 악화 |

**결론**: 52개 패턴 간 신호 중복 사실상 없음. 미러 패턴도 포트폴리오 기여 양수.

### 5.3 Reverse Edge & New Pattern Research

#### Reverse Edge (52 패턴 반대 방향)

| 결과 | 수 |
|------|-----|
| 반대 방향 양의 edge | 1/52 (무의미) |
| Anti-pattern (반대 < -10pp) | **21/52** |
| 양방향 음의 edge | 104개 |
| Polarized (한쪽만 edge) | 218개 |

- **Continuation vs Reversal**: +1.67pp vs +1.52pp → 통계적 차이 없음. **방향 편향 없음**.

#### Full Universe Scan (3,456 조합)

| Phase | 결과 |
|-------|------|
| 전체 스캔 | 63개 신규 MC < 0.01 발견 |
| TP/SL 최적화 | 54/63 MC 통과 |
| 개별 WF | **52/54 통과 (96%)** |
| 포트폴리오 통합 WF | **NOT BENEFICIAL** |

**포트폴리오 통합 결과**:

| 포트폴리오 | OOS PnL | MDD | PF |
|-----------|---------|-----|-----|
| 52 (현재) | +640.7% | 18.0% | 3.10 |
| 52 + 52 (확장) | +403.5% | 28.5% | 2.42 |
| **변화** | **-37.2%** | **+10.5pp** | **-22%** |

**원인**: 1-pos-at-a-time 제약에서 신규 패턴이 기존 우수 신호를 차단.

### 5.4 Pattern Combination Optimization

#### In-Sample 결과 (과적합 함정)

| 방법 | 최적 패턴 수 | In-Sample PnL | vs 52패턴 |
|------|-------------|---------------|----------|
| BWD Elimination | 40 | +977% | +21% |
| FWD Selection | 29 | +950% | +18% |
| Random Size Sweep | 52 | best PnL/MDD | 전체 최적 |

#### TRUE Walk-Forward 결과 (현실)

| 전략 | OOS PnL | MDD | PnL/MDD |
|------|---------|-----|---------|
| 52 전체 (baseline) | +640.7% | 18.0% | 35.6x |
| BWD 최적 부분집합 | +627.6% | 26.7% | 23.5x |
| **변화** | **-2.0%** | **+48%** | **-34%** |

**핵심 발견**: "나쁜 패턴"이 fold마다 다름 → 일관된 제거 대상 없음 → 부분집합 선택은 과적합.

### 5.5 Structural Improvement Research

#### Multi-Position (동시 포지션)

| 설정 | OOS PnL | WR | MDD | PnL/MDD |
|------|---------|-----|-----|---------|
| 1-pos (현재) | +305.2% | 74.5% | 23.8% | **12.8x** |
| 2-pos any-dir | +316.0% | 73.1% | 25.9% | 12.2x |
| 2-pos same-dir | +300.3% | 72.4% | 37.9% | 7.9x |
| 3-pos any-dir | +314.9% | 73.1% | 32.9% | 9.6x |

- 2-pos PnL 소폭 상승(+3.5%)이나 MDD 악화 → **PnL/MDD 하락**

#### Time-Based Exit (최대 보유 기간)

**보유 시간별 성과 분포**:

| 보유 시간 | 거래 수 | 평균 PnL | WR | 총 PnL |
|-----------|--------|---------|-----|--------|
| 0-1h | 38 | +2.04% | 92.1% | +77.3% |
| 1-3h | 73 | +2.19% | 90.4% | +160.2% |
| 3-6h | 54 | +1.13% | 81.5% | +60.9% |
| 6-12h | 75 | +1.60% | 82.7% | +119.7% |
| 12-24h | 66 | +1.33% | 81.8% | +88.0% |
| **24h+** | **97** | **-1.35%** | **42.3%** | **-131.3%** |

**WF OOS 최대 보유 기간 비교**:

| MaxHold | OOS PnL | MDD | PnL/MDD | MC |
|---------|---------|-----|---------|-----|
| 36b (3h) | +113.0% | 32.4% | 3.5x | 0.0429 |
| 144b (12h) | +242.4% | 28.2% | 8.6x | 0.0000 |
| 288b (24h) | +285.6% | 30.4% | 9.4x | 0.0000 |
| **500b (현재)** | **+305.2%** | **23.8%** | **12.8x** | **0.0000** |

- 24h+ 트레이드가 손실 주범(-131.3%)이나, 조기 청산 시 WR 급락
- **500bar 제한이 OOS 최적**

#### ATR-Adaptive TP/SL

| 설정 | OOS PnL | MDD | PnL/MDD | MC |
|------|---------|-----|---------|-----|
| **fixed (현재)** | **+305.2%** | **23.8%** | **12.8x** | **0.0000** |
| ATR x 0.5 | +203.4% | 30.8% | 6.6x | 0.0020 |
| ATR x 0.7 | +109.2% | 32.0% | 3.4x | **0.0661** |
| ATR x 1.0 | +220.2% | 45.3% | 4.9x | 0.0018 |

- ATR(14) P90/P10 = 4.48x (높은 변동폭)
- **모든 ATR 배수에서 fixed 대비 열등**. ATR 0.7은 MC까지 실패.
- 고정 TP/SL이 ATR-adaptive보다 안정적

---

## 6. Phase 4: 과적합 & 미래 적용성 연구 (2026-02-11)

### 6.1 Temporal Stability — 패턴 edge의 시간적 안정성

270일을 9개 30일 구간으로 분할하여 포트폴리오 성과 측정:

| 구간 | PnL | Trades | WR | MDD | PnL/MDD |
|------|-----|--------|-----|-----|---------|
| 0-30일 | +41.2% | 41 | 80.5% | 16.9% | 2.4x |
| 30-60일 | +20.9% | 44 | 70.5% | 19.6% | 1.1x |
| 60-90일 | +41.2% | 41 | 73.2% | 16.6% | 2.5x |
| 90-120일 | +44.4% | 38 | 78.9% | 12.2% | 3.6x |
| 120-150일 | +20.1% | 32 | 68.8% | 11.8% | 1.7x |
| 150-180일 | +62.0% | 55 | 76.4% | 23.8% | 2.6x |
| 180-210일 | +58.1% | 67 | 76.1% | 18.1% | 3.2x |
| 210-240일 | +55.1% | 53 | 73.6% | 14.8% | 3.7x |
| 240-270일 | +32.5% | 39 | 76.9% | 22.6% | 1.4x |

- **전 9개월 양수**: 한 달도 손실 없음 (최저 +20.1%)
- **전반 평균 +36.9% vs 후반 +45.5%**: 감쇠 없음, 오히려 후반 개선
- **Jaccard 0.120**: 30일 단위 "좋은 패턴" 집합 유사도 매우 낮음
- Mostly positive (>=70% windows): **4/52만** — 대부분의 패턴이 구간별로 불안정

**핵심 긴장점**: 포트폴리오 전체는 항상 양수이지만, 그 안에서 기여하는 패턴은 매달 달라진다.

### 6.2 True Out-of-Time Test — 순수 미래 예측력

가장 중요한 과적합 직접 측정. 전반 135일로만 패턴 선택 후 후반 135일 성과:

| 방법 | 후반 135일 PnL | 패턴 수 | WR | MDD | MC |
|------|---------------|---------|-----|-----|-----|
| A: 전반 135일로만 선택 | **+65.8%** | 15 | 74.5% | 22.1% | 0.054 |
| B: 전체 270일로 선택 (현재) | **+227.2%** | 52 | 75.6% | 23.8% | 0.0002 |
| **Look-ahead gap** | **+161.4%** | - | - | - | - |

- **Gap +161.4%**: 전체 성과의 **71%가 look-ahead에 의존**
- **52개 중 15개만** 전반부 데이터만으로도 선택됨 → 37개(71%)는 "뒤돌아본 결과" 추가된 것
- **그러나 Method A도 +65.8% 양수**: 순수 미래 예측으로도 수익성 유지
- MC p=0.054로 5% 경계선 — 15개만으로는 통계적 유의성이 약함

**해석**: 과적합 요소가 명확히 존재하지만, 과적합을 제거해도 수익성은 유지된다.

### 6.3 Regime Dependence — 시장 국면 의존성

**변동성별 성과**:

| 국면 | PnL | Trades | WR | 평균 PnL |
|------|-----|--------|-----|---------|
| 저변동 | +57.5% | 58 | 77.6% | +0.99% |
| 중변동 | +65.9% | 94 | 72.3% | +0.70% |
| **고변동** | **+249.8%** | **249** | 75.5% | +1.00% |

**추세별 성과**:

| 국면 | PnL | Trades | WR | 평균 PnL |
|------|-----|--------|-----|---------|
| **하락** | **+224.6%** | **191** | **77.5%** | **+1.18%** |
| 횡보 | +61.0% | 62 | 75.8% | +0.98% |
| 상승 | +87.6% | 148 | 71.6% | +0.59% |

- **수익 집중**: 고변동(66.9%) + 하락추세(60.1%)에 수익 집중
- **그러나 모든 국면 양수**: 10건 이상 거래 국면 중 손실 국면 = **0**
- **국면 집중 리스크**: LOW (특정 국면에서만 작동하는 전략이 아님)

**미래 리스크**: 장기간 저변동+상승 국면이 지속되면 거래 빈도와 수익성이 감소할 수 있음.

### 6.4 Edge Decay Curve — 전략 수명 추정

첫 N일로 패턴 선택 후 다음 30일 OOS 성과:

| 훈련 일수 | OOS 30일 PnL | Trades | WR | MC | 선택 패턴 수 |
|-----------|-------------|--------|-----|------|------------|
| 30일 | +11.9% | 42 | 69.0% | 0.285 | 22 |
| 60일 | +30.0% | 36 | 77.8% | 0.033 | 27 |
| 90일 | +10.4% | 44 | 68.2% | 0.312 | 35 |
| 120일 | +22.7% | 34 | 70.6% | 0.099 | 32 |
| 150일 | +36.7% | 61 | 77.0% | 0.092 | 33 |
| 180일 | +13.7% | 64 | 70.3% | 0.314 | 33 |
| 210일 | +4.8% | 45 | 64.4% | 0.430 | 36 |
| 240일 | +12.2% | 42 | 69.0% | 0.290 | 34 |

- **Slope -0.040**: 약한 음의 추세 (훈련 기간이 길수록 OOS 소폭 감소)
- **전 구간 양수**: 최저 +4.8%도 양수
- MC p-value 대부분 높음 → 30일 단위로는 통계적 유의성 확보 어려움 (표본 부족)

**해석**: 명확한 edge decay는 관찰되지 않으나, 훈련 데이터가 길다고 OOS가 좋아지지도 않음. 전략이 특정 시기에 최적화되기보다는, 일관된 약한 edge를 포트폴리오로 증폭하는 구조.

### 6.5 Pattern Survival Analysis — 패턴 생존율

270일을 3개 90일 구간으로 분할하여 패턴별 수익성 검증:

| 구간 | PnL | WR | Trades | MC |
|------|-----|-----|--------|-----|
| Period 1 (0-90일) | +103.2% | 74.6% | 126 | 0.0050 |
| Period 2 (90-180일) | +121.6% | 74.8% | 123 | 0.0025 |
| Period 3 (180-270일) | +142.8% | 75.5% | 155 | 0.0055 |

**개별 패턴 생존율**:

| 분류 | 수 | 비율 |
|------|-----|------|
| 3기간 전부 양수 (Robust) | **31** | **60%** |
| 2/3 이상 양수 | 47 | 90% |
| 전부 음수 | 0 | 0% |

**Robust-only (31개) vs Full (52개)**:

| 포트폴리오 | PnL | Trades | WR | MDD | PnL/MDD |
|-----------|-----|--------|-----|-----|---------|
| **Robust only (31)** | +355.7% | 375 | **77.9%** | **15.8%** | **22.6x** |
| Full 52 | +376.0% | 402 | 75.1% | 27.5% | 13.7x |

- Robust-only: PnL 약간 낮지만(-5%) **MDD 절반(27.5→15.8%)** → **PnL/MDD 1.6배 우수**
- 21개 "fragile" 패턴은 PnL +20.3% 기여하면서 MDD +11.7pp 추가

### 6.6 Bootstrap Confidence Intervals — 통계적 확신도

Block bootstrap (30일 블록, 5,000회 리샘플):

| 지표 | 95% CI | 관찰값 |
|------|--------|--------|
| PnL | **[+287.4%, +458.8%]** | +376.0% |
| WR | [72.9%, 77.1%] | 75.1% |
| MDD | [22.2%, 32.7%] | 27.5% |

| 확률 | 값 |
|------|-----|
| P(PnL > 0) | **100.0%** (5,000/5,000) |
| P(PnL > 100%) | **100.0%** |
| P(WR > 60%) | **100.0%** |
| P(MDD < 30%) | 90.5% |
| **Worst 5% PnL** | **+301.9%** |
| Worst 5% MDD | 32.7% |

- 5,000번 리샘플 중 **한 번도 손실 없음**
- 최악의 5% 시나리오에서도 **+302% 수익**

### 6.7 과적합 종합 판정

| 테스트 | 결과 | 판정 |
|--------|------|------|
| 시간적 안정성 (Jaccard) | 0.120 (낮음) | ⚠️ 과적합 신호 |
| Look-ahead gap | +161.4% (큰 gap) | ⚠️ 과적합 신호 |
| Edge decay | -0.040 (약한 감쇠) | ⚠️ 과적합 신호 |
| 국면 의존성 | 손실 국면 없음 | ✅ 견고 |
| 패턴 생존율 | 60% 전구간 양수 | ✅ 견고 |
| Bootstrap CI | P(PnL>0) = 100% | ✅ 견고 |

**과적합 신호 3/6, 견고 신호 3/6 → MODERATE RISK**

**솔직한 해석**:
- 52개 중 37개(71%)는 전체 270일을 봐야만 선택되는 패턴 — look-ahead 의존
- 그러나 look-ahead를 제거해도(15개만) 후반부 +65.8% 양수
- 개별 "좋은 패턴"은 매달 바뀌지만, 52개의 분산이 이를 흡수
- Bootstrap 5,000회 중 손실 0회 — 포트폴리오 수준 edge는 통계적으로 견고
- **백테스트 WR 84%는 과적합된 수치**, 실제 라이브에서는 **67-75% 수준**이 현실적 기대치

---

## 7. 핵심 발견 및 교훈

### 7.1 과적합의 이중 구조

이번 연구 시리즈의 가장 중요한 발견은 **"개별 수준 과적합 + 포트폴리오 수준 견고성"의 공존**이다:

```
개별 패턴 수준:
  - Jaccard 0.12 → "좋은 패턴" 집합이 매달 변함
  - 71%가 look-ahead에 의존하여 선택됨
  - 개별 MC 실패율 44% (270일), 88% (54일)
  → 과적합 존재

포트폴리오 수준:
  - 9개월 연속 양수 (최저 +20.1%)
  - Bootstrap 5,000회 손실 0회
  - 모든 시장 국면 양수
  - 15개 보수적 선택으로도 +65.8% 양수
  → 견고한 edge 존재
```

**이것이 의미하는 바**: 52개 패턴 각각의 edge는 노이즈에 취약하지만, 분산된 포트폴리오로 합치면 노이즈가 상쇄되고 genuine edge만 남는다. 이는 개별 종목 선택은 불확실하지만 인덱스 펀드는 안정적인 것과 같은 원리이다.

**현실적 기대치 조정**:
- 백테스트 WR 84% → 라이브 기대 WR **67-75%**
- 백테스트 PnL +911% → 라이브 기대 PnL **보수적으로 1/3~1/2 수준**
- Bootstrap worst 5% PnL +302% → 이것이 현실적 하한 기대치

### 7.2 Edge의 본질

```
수익 원천 = 100% genuine pattern edge
           (distance bias만으로는 -262.5% 손실)

Random Walk WR = SL/(TP+SL) = 53.2%
Actual WR     = 83.7%
Edge          = +30.5pp (순수 패턴 예측력)
```

52개 패턴의 3-candle sequence가 단기 가격 방향을 **random walk 대비 30.5pp** 초과 예측한다. 이 edge는:
- TP/SL distance의 산물이 아님 (distance-only는 손실)
- Context filter로 강화할 수 없음 (BH FDR 전멸)
- 반대 방향에서는 존재하지 않음 (1/52만 양수)
- 다중검정 보정 후에도 견고함 (포트폴리오 MC p=0.0000)

### 7.3 포트폴리오 분산의 위력

| 현상 | 설명 |
|------|------|
| 15/52 개별 MC 실패 | 포트폴리오 MC p=0.0000 |
| MC 실패 패턴 제거 시 | OOS -440% (massive failure) |
| "나쁜 패턴" fold별 변동 | 일관된 제거 대상 없음 |
| 52→104 패턴 확장 | Signal competition → PnL -37% |

**결론**: 분산효과가 개별 패턴 품질보다 중요. 패턴 추가도 제거도 포트폴리오를 악화시킨다.

### 7.4 1-Position-at-a-Time 제약의 역할

- 전체 신호의 **84.9%**가 FIFO에 의해 차단
- 차단된 신호의 평균 PnL이 실행된 것보다 **낮음** → 자연선택
- Oracle DP 상한: +2,034.4% (현재의 2.5배) — 미래 정보 없이 도달 불가
- Multi-position으로 풀면 WR↓ MDD↑ → 순 악화

### 7.5 최적화의 한계

| 시도 | In-Sample | OOS | Gap |
|------|-----------|-----|-----|
| BWD Elimination | +977% (+21%) | -13.1% | **-34%** |
| FWD Selection | +950% (+18%) | N/A | - |
| New Patterns | 개별 96% WF pass | -37.2% | **-133%** |
| SL Reduction | N/A | -22~30% | - |
| ATR Adaptive | 일부 양수 | 전부 음수 | - |

**메타 교훈**: In-sample 최적화 결과는 OOS로 전이되지 않는다. 270일 데이터에서도 과적합이 발생한다.

### 7.6 라이브 성과 검증 및 미래 리스크 (연구 #20, #21)

31거래 라이브 데이터를 기반으로 **미래 리스크 정량화**와 **전략 붕괴 탐지** 시스템을 설계했다.

#### 7.6.1 Break-Even 마진 수정 (CRITICAL)

초기 연구(#20)에서 `pattern_5m_metrics.json`의 `recent_wins/losses` 버퍼를 직접 사용하여 break-even WR 64.8%, 마진 3.0pp라는 위험한 결론을 도출했다. 후속 검증(#21)에서 **버퍼 데이터 왜곡**을 발견:

| 항목 | 버퍼(잘못된 값) | 실제(수정값) | 차이 |
|------|----------------|-------------|------|
| Wins/Losses | 20W/12L | **21W/10L** | 버퍼 크기 불일치 |
| avg_win | 3.094 | **3.994** | +29% |
| avg_loss | 5.690 | **6.186** | +9% |
| R:R | 0.544 | **0.646** | +19% |
| Break-even WR | 64.8% | **60.8%** | -4.0pp |
| **마진** | **3.0pp** | **7.0pp** | **+4.0pp** |

`recent_wins/losses`는 롤링 버퍼로, 실제 거래 수(21W/10L)와 버퍼 엔트리 수(20W/12L)가 불일치한다. **항상 `total_pnl`, `winning_trades`, `losing_trades`에서 역산해야 정확한 평균을 얻을 수 있다.**

#### 7.6.2 R:R 불리 원인 규명

| 분석 항목 | 결과 |
|-----------|------|
| 설계 R:R 평균 | 0.478 (52개 패턴 TP/SL 비율) |
| TP < SL 비율 | **92.3%** (48/52 패턴) |
| 백테스트 R:R | 0.523 |
| 라이브 R:R | **0.646** (백테스트보다 양호) |
| Exit 구성 | TP 71.4%, SL 12.4%, TIMEOUT 16.2% |

R:R 불리는 **설계 의도**(작고 빈번한 승리 + 드문 큰 손실, 높은 WR로 보상)이며, 라이브 anomaly가 아니다.

#### 7.6.3 전략 붕괴 탐지 한계

Working WR(68%)과 Broken WR(60%)의 **8pp 갭**이 동시에 낮은 FP + 높은 TP를 불가능하게 만든다:

| 탐지 방법 | FP(68%) | TP(60%) | TP(55%) | 탐지(60%) |
|-----------|---------|---------|---------|-----------|
| Rolling WR(30,<50%) | 27.0% | 83.6% | 97.9% | 85t |
| Rolling WR(40,<50%) | 11.8% | 66.5% | 93.0% | 97t |
| CUSUM(0.40, 6) | 11.1% | 68.1% | 95.1% | 98t |
| Bayesian(0.8, 0.7) | **8.5%** | 69.0% | 96.2% | 73t |
| PnL < -25% | 13.4% | 72.6% | 96.6% | 64t |
| WR(30,<45%) AND DD>20% | 12.7% | 62.1% | 91.1% | 94t |

FP<25% AND TP(60%)>70%를 동시에 만족하는 **Pareto-optimal 규칙은 존재하지 않는다**.

#### 7.6.4 실용적 권고: 3단계 알림 시스템

| 단계 | 조건 | 행동 |
|------|------|------|
| Green | PnL > -15%, WR > 50% | 정상 운영 |
| **Yellow** | PnL < -25% (FP 13%, TP60 73%) | 포지션 사이즈 절반, 주시 |
| **Red** | WR(30)<45% AND DD>20% (FP 13%, TP60 62%) | 봇 정지, 수동 검토 |

#### 7.6.5 Forward Monte Carlo (라이브 파라미터 기반)

| Horizon | Mean PnL | P(손실) | P(>50% 손실) | 95% MDD |
|---------|----------|---------|-------------|---------|
| 50거래 | +13.1% | 33.2% | 1.4% | 53.9% |
| 100거래 | +26.5% | 23.8% | 3.8% | 72.8% |
| 200거래 | +52.0% | 18.1% | 3.6% | 96.7% |
| 500거래 | +129.5% | 7.2% | 2.6% | 127.6% |

#### 7.6.6 방향 Edge 검증

| 테스트 | 결과 |
|--------|------|
| 방향 반전 | PnL -464% → edge +751% |
| 무작위 방향 (200회) | P(원본보다 우수) = **0.0%** |
| 40% 패턴 고장 | 평균 PnL +166%, P(손실) = 0.0% |

52개 패턴의 방향 예측력은 **실재하며, 부분적 고장에도 견고**하다.

---

## 7. Common Mistakes 정리

이번 연구 시리즈에서 확인된, 반복하지 말아야 할 실수들:

| # | 실수 | 올바른 접근 |
|---|------|------------|
| 1 | PnL vs TP/SL 단위 혼동 | 봇 PnL = 가격이동 x leverage, TP/SL = 가격 거리 |
| 2 | 개별 MC 실패 → 제거 | 포트폴리오 분산효과 > 개별 유의성 |
| 3 | 같은 데이터로 백테스트 | 반드시 expanding-window WF |
| 4 | SL 축소 = 리스크 감소 | WR↓ + MDD↑ 이중타격 |
| 5 | In-sample 좋음 = OOS 좋음 | WF OOS만이 유효한 판단 기준 |
| 6 | 개별 WF 통과 = 포트폴리오 개선 | 1-pos signal competition 고려 |
| 7 | 패턴 부분집합 최적화 | "나쁜 패턴"이 기간마다 변함 |
| 8 | Multi-position = 무조건 개선 | PnL/MDD 악화, 사이즈 분산 필요 |
| 9 | 백테스트 WR = 실전 WR | 270일 최적화 WR 84%는 과적합 포함. 실전 67-75% 기대 |
| 10 | 개별 패턴 안정성 = 포트폴리오 안정성 | 개별 Jaccard 0.12이나 포트폴리오 9개월 연속 양수 |
| 11 | recent_wins/losses 버퍼 직접 사용 | 롤링 버퍼 크기 ≠ 실제 거래 수. total_pnl + trade counts에서 역산 |
| 12 | Break-even margin 3pp 공포 | 버퍼 왜곡. 실제 마진 7pp (수정값 60.8% BE vs 라이브 67.7%) |

---

## 8. 연구 자산 목록

### 8.1 Research Scripts (17개)

| 스크립트 | Phase | 주요 내용 |
|---------|-------|----------|
| `tp_ge_sl_research.py` | v1.26.0 | R:R >= 0.75 초기 연구 |
| `tp_ge_sl_research_v2.py` | v1.26.0 | Compound return 수정 |
| `tp_ge_sl_research_v3.py` | v1.26.0 | Simple returns 최종 |
| `tp_sl_bias_research.py` | v1.26.1 | Random baseline binomial |
| `risk_management_research.py` | v1.27.0 | Daily limit, Kelly, ruin |
| `rr_optimization_research.py` | v1.27.0 | R:R >= 1.0 grid search |
| `dual_tp_stability_research.py` | v1.27.0 | Dual-TP FWR 분석 |
| `trade_microstructure_research.py` | v1.27.0 | TP/SL race, MFE/MAE |
| `context_filter_research_v2.py` | v1.27.0 | BH FDR context filter |
| `distance_edge_decomposition.py` | v1.27.0 | WR 분해: distance + edge |
| `portfolio_improvement_research.py` | v1.27.0+ | F/B/D 3경로 |
| `reverse_edge_research.py` | v1.27.0+ | Reverse edge + universe scan |
| `new_pattern_wf_validation.py` | v1.27.0+ | 63 new pattern WF |
| `pattern_combination_research.py` | v1.27.0+ | BWD/FWD/TRUE WF |
| `structural_improvement_research.py` | v1.27.0+ | Multi-pos, time, ATR |
| `overfitting_forward_decay_research.py` | v1.27.0+ | 6-phase 과적합/미래 적용성 |
| `future_risk_stress_test.py` | v1.27.0+ | 6-phase 미래 리스크 정량화 |
| `rr_detection_deep_study.py` | v1.27.0+ | R:R 원인 + 전략 붕괴 탐지 |

### 8.2 Validation Scripts (5개)

| 스크립트 | 내용 |
|---------|------|
| `tp_sl_deep_validation.py` | 5-phase deep validation |
| `uniform_tp_validation.py` | Uniform TP 70% 8-phase |
| `mc_filter_wf_validation.py` | MC filter WF A~E |
| `new_pattern_wf_validation.py` | New pattern portfolio WF |
| `wf_validation_v1263.py` | v1.26.3 WF (superseded) |

### 8.3 Result JSONs (16개)

모든 결과는 `bingx_rl_trading_bot/results/` 디렉토리에 저장:

```
tp_ge_sl_research.json          tp_sl_bias_research.json
tp_ge_sl_research_v2.json       risk_management_research.json
tp_ge_sl_research_v3.json       rr_optimization_research.json
dual_tp_stability_research.json trade_microstructure_research.json
context_filter_research_v2.json mc_filter_wf_validation.json
portfolio_improvement_research.json
reverse_edge_research.json      new_pattern_wf_validation.json
pattern_combination_research.json
structural_improvement_research.json
overfitting_forward_decay_research.json
future_risk_stress_test.json    rr_detection_deep_study.json
uniform_tp_validation.json      tp_sl_deep_validation.json
```

---

## 9. 최종 결론 및 권고

### 9.1 현재 상태

v1.27.0은 **270일 데이터에서 도달 가능한 최적점에 근접**해 있다:

- 52개 패턴: 추가도 제거도 OOS 악화
- TP/SL: Uniform 70% + grid search 최적값, ATR-adaptive 열등
- 구조: 1-pos FIFO가 multi-pos 대비 최적
- 보유 기간: 500bar 최대가 최적
- 필터: Context filter 무효, MC filter 무효
- 리스크: 7% daily limit + 3-loss pause 적용

### 9.2 과적합 위험 평가

6-phase 과적합 연구 결과, **MODERATE RISK** (3/6 과적합 신호, 3/6 robust):

| 구분 | 과적합 신호 | Robust 신호 |
|------|------------|-------------|
| 패턴 안정성 | Jaccard 0.12 (개별 불안정) | 9개월 모두 양수 PnL |
| Look-ahead 의존 | 71% gap (OOS +65.8% vs +227.2%) | 15개 first-half 패턴도 양수 |
| Edge decay | 약한 음의 기울기 (-0.040) | 모든 기간 양수 |
| Regime 의존 | — | 모든 9개 regime 수익 |
| 생존 분석 | 40% 패턴 1기간 이상 실패 | 60% 패턴 3기간 모두 생존 |
| Bootstrap CI | — | P(PnL>0) = 100%, CI [+287%, +459%] |

**핵심 통찰**: 개별 패턴은 불안정하지만 포트폴리오 수준에서 robust. 라이브 WR 기대치는 **67-75%** (백테스트 84%가 아님).

### 9.3 추가 최적화 비권고

테스트한 모든 차원에서 현재 설정이 WF OOS 최적이므로, **추가 파라미터 조정은 과적합 위험만 증가**시킨다.

### 9.4 향후 가능한 연구 방향

현재 설정 변경이 아닌, **근본적으로 다른 접근**이 필요:

| 방향 | 설명 | 난이도 |
|------|------|--------|
| 다른 타임프레임 | 15m, 1h 패턴 별도 봇 | 중 |
| 다른 자산 | ETH, SOL 등 동일 전략 적용 | 중 |
| 4-candle 패턴 | 3→4 sequence 확장 | 고 |
| Regime detection | 시장 국면별 패턴 활성화 | 고 (context filter v2 실패 참고) |
| 실시간 데이터 축적 | 270일→540일 데이터로 재검증 | 저 (시간 필요) |
| **Robust-only 포트폴리오** | 31개 생존 패턴만 사용 (PnL/MDD 22.6x) | **저** |

### 9.5 운영 권고

1. **현재 설정 유지**: v1.27.0 52패턴, Uniform TP 70%, 1-pos, FIFO
2. **3단계 모니터링**:
   - **Green**: 정상 운영 (PnL > -15%, WR > 50%)
   - **Yellow**: PnL < -25% → 포지션 사이즈 절반, 주시 (FP 13%, TP60 73%)
   - **Red**: WR(30)<45% AND DD>20% → 봇 정지, 수동 검토 (FP 13%, TP60 62%)
3. **재검증 시점**: 90일 후 (540일 데이터) 또는 Red alert 발동 시
4. **과적합 경계**: 향후 최적화 시도 시 반드시 expanding-window WF OOS 기준 적용
5. **라이브 기대치**: 백테스트 WR 84% → 라이브 기대 67-75%, Break-even 60.8% (마진 7pp)
6. **버퍼 데이터 경고**: `recent_wins/losses`는 롤링 버퍼 — 정확한 평균은 `total_pnl`에서 역산
7. **보수적 대안 준비**: WR 60% 미만 지속 시 31개 robust-only 포트폴리오로 전환 검토
8. **탐지 한계 인지**: WR 68%→60% 구간은 통계적으로 구별 곤란 (1539거래 = 34개월 필요)

---

*End of Report*
