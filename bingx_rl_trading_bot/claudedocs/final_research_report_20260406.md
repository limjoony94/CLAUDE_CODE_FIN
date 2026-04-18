# BTC 5m 패턴 트레이딩 시스템 — 최종 연구 보고서

> 2026-04-04 ~ 04-06 | 22개 스크립트 | 50+ 전략 변형 | 17회 비판 사이클

## 결론

**OHLC 데이터만으로는 BTC 5분봉에서 거래 비용(수수료 + 펀딩)을 초과하는 일관된 edge를 추출할 수 없다.**

## 확정된 사실

### 1. 시장에 edge는 존재한다 (Perfect Foresight +5.30%/trade)
방향을 100% 정확히 예측하면 TP=5%/SL=2%에서 +5.30%/trade. Edge가 시장에 풍부하게 존재하나, OHLC 기반 신호로는 포착 불가.

### 2. 테스트한 모든 OHLC 신호가 실패
| 신호 유형 | WF 최고 | DISC? | 결과 |
|----------|------:|-------|------|
| 3-candle 패턴 (현행) | 5/5* | NON-DISC | *timeout DROP 환상 |
| 대안 분류 (ATR/KMeans/Fuzzy) | 동등 | - | 분류와 WR 무상관 (r=0.0008) |
| R:R 역전 (TP>SL) | 2/5 | NON-DISC p=1.0 | 시기 의존 |
| 추세 추종 (24h) | 3/5 | - | 총 PnL 음수 |
| 변동성 필터 | 2/5 | - | 필터 강화 시 악화 |
| Mean Reversion lb=10 | 4/5 | NON-DISC p=0.35 | 허위양성 (Bonferroni 미통과, 11% FP rate) |
| 스트래들 (양방향) | 2/5 | - | 시기 의존 |
| Timeout 연장 (8일) | 5/10 | - | 동전 던지기 |

### 3. 라이브 손실의 근본 원인
| 원인 | PnL 기여 | 상태 |
|------|--------:|------|
| SL 축소 (cascade/vol_adapt) | -31.29% | v1.70에서 수정 |
| CASCADE_SL 직접 | -13.73% | v1.70에서 수정 |
| TIMEOUT | -14.01% | 구조적 |
| CRASH/MCC/MARKET | -8.44% | 운영 |
| 수수료 | -30.30% | 불변 |
| 정상 SL 거래 | +28.40% | 작동하나 수수료 미만 |
| **합계** | **-26.39%** | |

### 4. SL 정확도가 유일한 핵심 변수 (+34.3pp 효과)
정상 SL에서 WR 85.9% (t=4.34, p=0.0000). 하지만 이 WR에서도 TP+SL PnL은 +2.09% (707건) = 거래당 +0.005% (수수료 0.043%의 1/9).

### 5. Timeout이 두 번째 핵심 장벽
- BT timeout DROP이 WR을 15-20pp 과대평가 (모든 이전 BT의 공통 결함)
- 1-pos BT timeout rate 42% vs Live N-pos 8.5% (5배 차이)
- Timeout 연장(8일)하면 edge 양수이나 WF 5/10 (비일관적)

### 6. 펀딩 비용은 방향 균형 시 상쇄
50/50 LONG/SHORT 포트폴리오에서 순 펀딩 ≈ 0. DC=4 최악(4/3)에서도 +0.033%/trade로 미미.

## 유일하게 살아남은 양수 구조

TP=5%/SL=2%/Timeout=8일/Random 방향:
- Edge: +0.172%/trade (수수료+순펀딩 후)
- WF: 5/10 (비일관적, 동전 던지기 수준)
- Monthly: +1.6% ($43 at $2,668 자본)
- 50% 확률로 월간 손실
- 실용적 가치 없음

## 방법론 결함 발견

1. **bt_signals_atr() timeout DROP**: 모든 Scanner/WF 검증의 근본 결함. Timeout 거래를 제거하여 WR을 15-20pp 과대평가.
2. **1-pos BT ≠ N-pos 프로덕션**: Timeout rate 5배 차이. 1-pos 결론을 production에 적용 불가.
3. **다중 검정**: 50+ 전략을 같은 데이터에서 테스트. WF 4/5 = 11% 허위양성율 × 50 tests = ~5.5 기대 허위양성.

## 향후 방향

| 방향 | 가능성 | 필요 사항 |
|------|--------|----------|
| 외부 데이터 (orderbook, funding, OI) | 미지 | BingX API 연동, 새 연구 |
| 다른 시장 (알트코인, 낮은 수수료) | 미지 | 시장 탐색 |
| 더 긴 타임프레임 (4h+) | 약간 유망 | 적은 거래, 큰 보유기간 |
| Market making / Arbitrage | 근본적 전환 | 완전 새 시스템 |
| 현행 봇 중단 | 현실적 | 추가 손실 방지 |

## 연구 스크립트 목록

scripts/analysis/ 아래:
- candle_classification_rigor_study.py, classification_v3_study.py
- classification_deep_study.py, classification_corrected_study.py
- signal_system_verification.py, signal_system_redesign.py
- rr_rebalance_study.py, fee_and_edge_critique.py
- critical_verification.py, timeout_deep_critique.py
- single_vs_both_dir.py, fuzzy_boundary_study.py
- v171_interim_critique.py, wr858_critique.py
- regime_vs_sl_critique.py, forward_risk_simulation.py
- honest_profitability.py, stratified_ttest.py
- rr_inversion_study.py, straddle_test.py
- regime_signal_study.py, theoretical_upper_bound.py
- edge_extraction_study.py, meanrev_validation.py
- timing_not_direction.py, final_verification.py
