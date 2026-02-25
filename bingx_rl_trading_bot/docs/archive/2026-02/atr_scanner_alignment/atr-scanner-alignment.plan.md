# ATR Scanner-Production Alignment — Research Plan

> **Feature**: atr-scanner-alignment
> **Type**: Research (production 변경은 결과에 따라 결정)
> **Bot**: BTC 5m Pattern Trading, BingX, 3x Leverage, Hedge Mode, N=9
> **Baseline**: v1.33.0 (35 patterns, 9L+26S, Compact TP/SL, WR 68.1%, edge 0.221%)
> **Created**: 2026-02-24

---

## 1. Problem Statement

**Scanner-Production 불일치**: Scanner는 고정 TP/SL로 패턴을 선별하지만, Production은 ATR-scaled TP/SL로 실제 거래한다.

| 단계 | TP/SL 방식 | 문제 |
|------|-----------|------|
| Scanner (pattern_scanner.py) | Fixed TP/SL | ATR 미적용 |
| Production (position_open.py) | ATR(14)/median(576) scaled, clamp [0.6,1.7] | ATR 적용 |

**이전 연구 결과** (`atr_scaled_backtest_study.py`, 2026-02-23):
- Fixed filter: 51 patterns pass → ATR filter: 39 patterns pass (선별 결과 다름)
- ATR T864: PnL/MDD **17.18** vs Fixed T864: **10.91** (+57.5%)
- 개별 패턴: ATR scaling이 avg WR +4.5pp, edge +0.876%/trade 개선 (11/15 향상)
- **결론**: ATR scaling이 리스크 조정 성과를 개선하지만, Scanner에 미반영

**핵심 질문**: Scanner에 ATR scaling을 통합하면 패턴 선별이 개선되어 WF OOS 성과가 향상되는가?

---

## 2. Hypotheses

| ID | 가설 | Go/No-Go 기준 |
|----|------|--------------|
| H1 | ATR-integrated scanner가 Fixed scanner 대비 더 적은 수의 더 강한 패턴을 선별한다 | ATR avg WR Excess > Fixed avg WR Excess |
| H2 | ATR-integrated scanner 패턴으로 구성한 포트폴리오가 Fixed 대비 WF OOS PnL/MDD 향상 | ATR WF PnL/MDD > Fixed WF PnL/MDD, 3/3 PASS |
| H3 | ATR 파라미터 (period, window, clamp) fine-tuning으로 추가 개선 가능 | Best params WF PnL/MDD > default(a14/w576/0.6-1.7) PnL/MDD |

---

## 3. Research Design — 4-Phase Go/No-Go

```
Phase 1 (ATR Scanner) → H1 GO? →
Phase 2 (Portfolio + WF) → H2 GO? →
Phase 3 (Parameter Sweep) → H3 GO? →
Phase 4 (Production Candidate) → Deploy decision
```

### Phase 1: ATR-Integrated Scanner Backtest

**목적**: Scanner의 `bt_signals()` + `scan_patterns_mae_mfe()`에 ATR scaling을 적용하여 패턴 선별 결과 비교

**방법**:
1. 270일 5m 데이터에서 ATR(14)/median(576) 비율 시계열 사전 계산
2. `bt_signals()` 변형: 각 신호 시점의 ATR ratio로 base TP/SL을 스케일링
   - `effective_tp = base_tp * clamp(atr_ratio, 0.6, 1.7)`
   - `effective_sl = base_sl * clamp(atr_ratio, 0.6, 1.7)`
3. MAE/MFE discovery도 ATR-scaled 조건에서 수행
4. 동일 quality filter 적용 (edge>=21.8pp, WR>=60%, SL>=1.0%, MC<0.01, min_trades>=25, WR Excess>5pp)
5. Fixed vs ATR 패턴 목록 비교

**Go/No-Go Gate**:
- GO: ATR 패턴의 avg WR Excess가 Fixed 대비 +2pp 이상 또는 패턴 수가 유의하게 다름
- STOP: 차이 없음 (이미 Phase 2 atr_scaled_backtest_study에서 확인한 것과 동일)

### Phase 2: Portfolio WF 검증

**목적**: ATR-selected 패턴으로 구성한 포트폴리오의 WF OOS 성과 비교

**방법**:
1. Fixed-selected 포트폴리오: Hedge N=9, Direction Cap 6, T864, Compact TP/SL
2. ATR-selected 포트폴리오: 동일 설정, ATR-scaled execution
3. 3-fold Expanding Window WF (IS: 0-240d, 0-480d, 0-540d / OOS: 각 240d)
4. Comparison metrics: OOS PnL, OOS MDD, OOS PnL/MDD, OOS WR

**Go/No-Go Gate**:
- GO: ATR portfolio WF 3/3 PASS + OOS PnL/MDD > Fixed OOS PnL/MDD
- STOP: WF < 3/3 또는 Fixed 대비 열위

### Phase 3: Parameter Sweep (Optional, H2 GO 시)

**목적**: ATR 파라미터 최적화

**파라미터 공간**:
| 파라미터 | 현재 | 후보 범위 | 설명 |
|---------|------|----------|------|
| ATR period | 14 | [7, 14, 21, 28] | ATR 계산 기간 |
| Median window | 576 | [288, 576, 864, 1152] | Rolling median 기간 (1-4일) |
| Clamp low | 0.6 | [0.5, 0.6, 0.7, 0.8] | 최소 스케일링 비율 |
| Clamp high | 1.7 | [1.3, 1.5, 1.7, 2.0] | 최대 스케일링 비율 |

**방법**:
1. Full grid: 4 × 4 × 4 × 4 = 256 조합 → 계산 비용 높음
2. **2단계 축소**: (a) period/window sweep (16조합), best에서 (b) clamp sweep (16조합) = 32조합
3. 각 조합에서 Phase 1 + Phase 2 (WF) 반복
4. Best params 선택 기준: WF 3/3 PASS + OOS PnL/MDD 최대

**Go/No-Go Gate**:
- GO: Best params WF PnL/MDD > default params PnL/MDD + 10% 이상
- STOP: Default params가 이미 최적 또는 차이 미미

### Phase 4: Production Candidate 평가

**목적**: 최종 결과를 production에 적용할지 결정

**평가 기준**:
1. WF 3/3 PASS (필수)
2. WF OOS PnL/MDD > 현재 v1.33.0 baseline
3. 패턴 수 >= 15 (거래 빈도 유지)
4. ATR-aware scanner의 복잡도 vs 성과 개선 trade-off

**결과 옵션**:
- **A. Scanner 교체**: ATR-integrated scanner로 `pattern_scanner.py` 업데이트
- **B. Scanner 병렬**: `--atr-aware` 플래그 추가 (기존 모드 보존)
- **C. STOP**: 유의미한 개선 없어 현재 유지

---

## 4. Implementation Plan

### 스크립트: `scripts/analysis/atr_scanner_alignment_study.py`

**구조**:
```python
# Phase 1: ATR-integrated backtest
def compute_atr_ratio_series(df, period=14, window=576, clamp_lo=0.6, clamp_hi=1.7):
    """전체 데이터에서 ATR ratio 시계열 사전 계산"""

def bt_signals_atr(df, pattern, direction, tp, sl, atr_ratios, ...):
    """ATR-scaled TP/SL로 백테스트 (bt_signals 변형)"""

def scan_patterns_atr(df, atr_ratios, quality_filters):
    """ATR 조건에서 MAE/MFE discovery + quality filter"""

# Phase 2: Portfolio WF
def portfolio_backtest_atr(df, patterns, atr_ratios, n_positions, direction_cap, timeout):
    """Hedge N=9 포트폴리오 백테스트 (ATR-scaled)"""

def wf_validate(df, scan_func, portfolio_func, n_folds=3):
    """Expanding Window WF (scan→portfolio→OOS 평가)"""

# Phase 3: Parameter Sweep
def param_sweep(df, param_grid, wf_func):
    """2단계 파라미터 탐색"""

# Phase 4: Summary
def generate_production_candidate(best_params, wf_results):
    """최종 결과 정리 + dynamic_patterns.json 생성"""
```

### 데이터

| 데이터 | 경로 | 용도 |
|--------|------|------|
| 5m 270일 | `data/btc_5m_270days_reclassified.csv` | IS/OOS |
| Dynamic patterns | `results/dynamic_patterns.json` | 현재 baseline |

### 의존성

- `scripts/production/pattern_5m/signals.py` → `classify_candle` (production 일관성)
- `scripts/scanner/pattern_scanner.py` → `compute_excursions`, `derive_tp_sl` (MAE/MFE)
- 기존 ATR study 결과: `results/atr_scaled_backtest_study.json` (Phase 1 참고)

---

## 5. Expected Outcomes

| 시나리오 | 확률 | 결과 |
|---------|------|------|
| H1 GO + H2 GO + H3 GO | 20% | Scanner 교체/업그레이드, 새 패턴 세트 배포 |
| H1 GO + H2 GO + H3 STOP | 40% | Default ATR params로 scanner 업그레이드 |
| H1 GO + H2 STOP | 25% | 개별 패턴은 개선되지만 포트폴리오 수준 효과 없음 → 유지 |
| H1 STOP | 15% | Scanner-Production 불일치가 성과에 영향 없음 → 유지 |

### 이전 연구와의 차이

| 항목 | atr_scaled_backtest_study (02-23) | 이번 연구 |
|------|----------------------------------|----------|
| Scanner | Fixed TP/SL로 선별 후 ATR 비교 | ATR 조건에서 직접 선별 |
| MAE/MFE | Fixed TP/SL 기준 | ATR-scaled TP/SL 기준 |
| WF | 15패턴 기존 세트 | ATR 선별 새 세트 |
| Parameter | a14/w576/0.6-1.7 고정 | Sweep 포함 |
| 포트폴리오 | N=5, 15패턴 | N=9, Direction Cap 6, Compact grid |

---

## 6. Risks & Mitigations

| 리스크 | 완화 |
|--------|------|
| ATR ratio 계산 → Look-ahead bias | ATR ratio는 t-1 시점 값 사용 (shift(1)) |
| 파라미터 sweep → overfitting | WF OOS 필수, MC test |
| 계산 비용 (256 조합 × WF) | 2단계 축소 (32조합), 병렬 처리 |
| Compact TP/SL 범위와 ATR 충돌 | Proportional cap 유지 (R:R 보존) |

---

## 7. Success Criteria

1. **최소**: H1 GO/STOP 판정 + Fixed vs ATR 선별 차이 정량화
2. **기대**: H2 GO + WF 3/3 PASS + PnL/MDD > baseline
3. **최상**: H3 GO + 최적 파라미터 발견 + Scanner 업그레이드 배포

---

*Plan created: 2026-02-24 | Feature: atr-scanner-alignment*
