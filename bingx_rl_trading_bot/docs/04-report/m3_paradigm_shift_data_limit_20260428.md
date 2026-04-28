# M3-R12 Paradigm Shift — Data Limit 발견 (Final)

> **Date**: 2026-04-28
> **Authority**: 사용자 옵션 C 명시 + "수익성 있는 모델 찾을 때까지"
> **Result**: 2 paradigm 후보 검증 — 둘 다 진정한 paradigm shift 아니거나 fail. **True paradigm shift는 현 dataset에서 data-limited**.

---

## 1. R12 결과

### π* Pair Trade (true market-neutral)
9 configs (3 z_entry × 3 z_exit) sweep:
| z_entry | z_exit | n | per_day | daily_net | WR | RR |
|---------|--------|---|---------|-----------|-----|-----|
| 2.0 | 0.0 | 256 | 0.83 | **-0.199%** | 47.3% | 0.82 |
| 2.0 | 0.5 | 610 | 1.97 | -0.417% | 56.2% | 0.40 |
| ... | ... | ... | ... | (모두 음수) | ... | ... |

**전부 daily 음수**. RR < 1 (winning trades smaller than losing trades). 9/9 FAIL.

**해석**: BTC-ETH spread mean-reversion이 friction 0.16% RT 압도 못함. β (directional spread)가 fail한 동일 이유 — spread가 거의 noise.

### ω* Funding Yield Harvest
Threshold sweep 0.000 ~ 0.020:
| Threshold | n_cycles | per_day | daily_net | gross/cycle | WR |
|-----------|----------|---------|-----------|-------------|-----|
| 0.000 | 267 | 3.0 | +0.096% | +0.112 | 44.6% |
| 0.005 | 192 | 2.16 | +0.040% | +0.099 | 46.4% |
| **0.010** | **97** | **1.09** | **+0.108%** ⭐ | +0.179 | 51.5% |
| 0.015 | 0 (no cycles) | – | – | – | – |
| 0.020 | 0 | – | – | – | – |

**Surface PASS at threshold=0.01** (3/3 pre-reg conditions). 하지만:

### Decomposition (advisor 권고로 검증)

97 cycles at threshold=0.01:
- **Mean carry (funding contribution)**: **0.010%** per cycle
- **Mean directional (price drift) contribution**: **0.169%** per cycle
- **Mean gross**: 0.179% per cycle
- **Funding fraction**: **5.6%**
- **Directional fraction**: **94.4%**

**중대 finding**: ω*는 **paradigm shift 아님**. 94% directional + 6% carry = funding 신호 가지고 directional counter-trend bet 한 것. γ/ξ family가 다른 exit framework로 rediscover된 것.

**기존 γ와 차이**:
- γ: Counter-trend on funding extreme + trail/SL exit framework (15m bars) → C3 FAIL
- ω*: Counter-trend on funding extreme + **fixed 8h hold** + no SL → SURFACE PASS
- 같은 pattern as R9b: trail framework 제거 시 directional alpha가 surface positive
- R9c OOS test = FAIL (WF 2/5, bootstrap pos_rate 9%) — ω*도 동일 운명 high probability

### OOS test 진행 안 함 (advisor 정렬)

R9c가 동일 framework (R9b sweep → R9c OOS) FAIL 확인. ω* OOS도 확률 매우 높게 FAIL. 자원 절약하고 "γ family rediscovered, not paradigm" 결론으로 마무리.

---

## 2. 진짜 paradigm shift는 왜 안 됐나 — Data Limit

현재 dataset:
- BTC perp 15m OHLCV (BingX, 720 days)
- ETH 5m → 15m OHLCV (Binance, 365 days)
- BingX funding rates (89 valid days for the 8h cycles)

진짜 paradigm shift에 필요한 data:

| Paradigm | 필요 data | 현재 보유 |
|----------|-----------|-----------|
| **Cross-exchange basis arb** | Binance perp + BingX perp 동시 가격 | ❌ Single source 의심 |
| **Spot-perp basis trade** | BTC spot 가격 (separate from perp) | ❌ |
| **True funding harvest with delta hedge** | Spot + perp simultaneous | ❌ Spot 부재 |
| **Triangular arbitrage** | BTC/USDT, ETH/USDT, BTC/ETH 가격 | ❌ BTC/ETH 동시 |
| **Calendar / quarterly futures** | 분기 만료 contract 가격 | ❌ |
| **Volatility selling (covered call)** | Options chain | ❌ |
| **Market making / LP** | Order book history (book level) | ❌ |
| **Order flow / footprint** | Trade-level data | ❌ |

**가용 data로 가능한 paradigm**:
1. **Pair trade BTC-ETH** (R12 π*에서 fail 확인)
2. **Funding-based directional MR** (R12 ω* — γ family rediscovery)
3. **Calendar effects** (time-of-day, day-of-week) — 표면적이며 sample 작음

---

## 3. 사용자 결정 영역 (재제시)

12 rounds × 17 directional mechanisms × 2 paradigm 후보 누적:

### 옵션 D-1: 새 data 확보 + 진짜 paradigm shift
- **BTC spot data** (Binance/Coinbase) → spot-perp basis, true funding harvest
- **Multi-exchange perp data** → cross-exchange basis arb
- **Order book history** → liquidity provision, microstructure
- 추정 작업: 데이터 수집 1-2주 + 분석 2-3주

### 옵션 A: Stop & accept (이전 권고 유지)
12 rounds 누적 evidence + 2 paradigm fail = capital 다른 곳 활용. **가장 강한 evidence-based**.

### 옵션 E: BingX bot infra을 수익형 다른 service로 재활용
- Copy-trading service 운영 (bot → signal provider)
- 거래 자동화 SaaS (다른 사람들의 bot 운영)
- **현 코드베이스 가치 활용 + research 끝내기**

### 옵션 F: Different asset/exchange
- 같은 framework로 다른 asset 시도 (R10 evidence: 같은 noise floor 가능성 큼)
- 또는 traditional market (stocks/forex) 전환

---

## 4. Files

- `claudedocs/m3_round12_paradigm_shift.md` — pre-reg
- `scripts/analysis/m3_round12_paradigm.py` — π* + ω* runner
- `results/m3_r12_paradigm_*.json` — raw
- `docs/04-report/m3_paradigm_shift_data_limit_20260428.md` — this report
- `docs/04-report/m3_final_arc_20260428.md` — directional arc final (R1-R11)

## 5. Standing instruction

**가용 데이터로 paradigm shift 시도 → 모두 directional alpha 변형이거나 spread fail**. 사용자 explicit instruction "수익성 모델 찾을 때까지" 지속 진행 위해선 **새 data 확보 (옵션 D-1) 또는 paradigm 자체 재정의 (옵션 E)** 필요.

Assistant 자체 D-1 data acquisition 진행 안 함 (사용자 capital/시간 결정 영역). 사용자 명시 redirect 후 해당 옵션 PDCA Plan 시작.
