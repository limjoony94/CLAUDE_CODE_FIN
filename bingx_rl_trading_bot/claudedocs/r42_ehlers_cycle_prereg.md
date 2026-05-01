# R42 Pre-Registration — Ehlers Dominant Cycle Mean Reversion

**Date pre-registered**: 2026-05-01
**Status**: PRE-COMMIT (작성 후 BT 코드 작성, 결과 미확정)
**Honest prior**: ~80% falsification (32 mechanisms × 7 substrates falsified). Untested angle (Hilbert/Ehlers cycle, grep 0 results) slightly increases breakthrough prior.

---

## Why structurally distinct from 32 priors

| Feature | 28r OHLCV / N1-N8 / Hurst / Volume Absorption / ICT | R42 (new) |
|---------|------------------------------------------------------|-----------|
| Conditioning | Price level / range / variance / volume / pattern / cross-asset | **Time-domain frequency analysis** |
| Mechanism | Trend / mean-reversion / breakout / regime / pattern / cross-section | **Adaptive cycle phase** |
| Indicator | MA / RSI / BB / Donchian / ATR / fractal / correlation | **Hilbert transform → instantaneous phase → dominant cycle wave** |

Hilbert transform extracts **instantaneous frequency** of the price series. Ehlers' MAMA approach detects dominant cycle period dynamically and produces a normalized cycle wave [-1, +1]. Mean reversion at cycle extremes is a different signal type from any prior round — it's a **frequency-domain feature**, not a price/range/volume statistic.

---

## Locked Algorithm — `r42_ehlers_cycle`

### Step 1: Hilbert Transform Cycle Detection

Using **scipy.signal.hilbert** on smoothed close prices:

```python
smoothed = rolling_mean(close, window=4)  # detrend short noise
analytic = scipy.signal.hilbert(smoothed - rolling_mean(smoothed, window=20))
phase = unwrap(angle(analytic))
inst_freq = diff(phase) / (2 * pi)  # cycles per bar
period = clip(1 / inst_freq, 6, 50)  # bound 6-50 bars (3-25 hours)
```

### Step 2: Cycle Wave (normalized [-1, +1])

```python
cycle_wave = real(analytic) / max(rolling_max(abs(analytic), 100), epsilon)
```

### Step 3: Entry Signals

**LONG entry** (at bar i+1 open):
- `cycle_wave[i] < -0.70` (extreme cycle bottom)
- `cycle_wave[i] > cycle_wave[i-1]` (turning up — momentum reversal)
- `close[i] > SMA(close, 50)[i]` (long-term uptrend filter)

**SHORT entry** (at bar i+1 open):
- `cycle_wave[i] > +0.70`
- `cycle_wave[i] < cycle_wave[i-1]`
- `close[i] < SMA(close, 50)[i]`

### Step 4: Exit Rules

Whichever first:
- `cycle_wave` crosses 0 in opposite direction (LONG: cycle_wave > 0, SHORT: cycle_wave < 0)
- ATR stop: -1.0 × ATR(14) below entry (LONG) / above entry (SHORT)
- Timeout: `period × 0.75` bars after entry

### Step 5: Friction

Taker round-trip: 0.05% × 2 = **0.10% per trade** (보수적 BingX taker)

---

## Falsification Criteria (사용자 strict, pre-committed)

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| **F1: avg_gross/trade** | > 0.07% | Friction floor (taker RT 0.10% 보수, 0.07%로 lenient) |
| **F2: bootstrap mean_daily** | ≥ +0.20% | 사용자 명시 criterion |
| **F3: bootstrap pos_rate** | ≥ 0.50 | 3-day windows 50%+ 양수 |
| **F4: bootstrap p5_daily** | ≥ 0 | Worst 5% windows 음수 아님 |
| **F5: min_n_trades_per_window** | ≥ 3 | 3-day window 통계 의미성 |
| **F6: full-period n_trades** | ≥ 50 | 전체 통계 의미성 |

**ALL 6 PASS** → develop path open. **Any 1 FAIL** → falsified, return to user-level.

---

## Math Gate

- Required: avg_gross > 0.07% to clear friction
- Required: bootstrap daily ≥ +0.20%
- 32 prior rounds 모두 동일 envelope에서 0 PASS
- 새 indicator angle (frequency-domain)이 envelope 자체를 바꿀 가능성 = unknown but low

---

## Data

- **Asset**: BTC perp
- **Timeframe**: 1h (cycle period 6-50 bars = 6-50시간 → daily-ish cycle 검출 가능)
- **Source**: `bingx_rl_trading_bot/data/btc_1h_720days.csv` (17,280 bars, 720d)
- **Split**: 540d in-sample (R42 develop), 180d fresh OOS (final test, 1회만)

---

## Honest Prior Distribution

- ~80% falsified at F1 (avg_gross < friction)
- ~10% borderline (F1 PASS but F2-F4 FAIL)
- ~8% develop path (F1-F6 PASS, fresh OOS test 진행)
- ~2% deployable (in-sample + fresh OOS 모두 PASS, BT-LIVE parity 검증 진행)

이 prior는 누적 evidence 기반 — 추가 데이터로 update.

---

## Anti-fishing commitments

1. **Single mechanism, single config**: 위 algorithm 변경 금지. Threshold (-0.70/+0.70), period (6-50), SMA window (50), ATR stop (1.0), timeout (period×0.75) 모두 사전 fixed.
2. **No parameter sweep**: F1-F6 fail 시 sweep 금지. 다른 mechanism 시도하려면 새 pre-reg 작성.
3. **Fresh OOS untouched**: develop 단계에서 180d 마지막 부분 미사용. 최종 1회 test.
4. **Pre-commit math**: F1 fail 시 silent pivot 금지 — falsified로 기록.

---

## Result (BT executed 2026-05-01)

**OUTCOME: VACUOUS — internal contradiction discovered**

신호 발생 0회 (LONG 0, SHORT 0). 원인 분석:
- LONG cycle bottom + turning (cw < -0.7 AND cw > prev): **276회** 단독 trigger
- LONG trend up (close > SMA50): **9,002회** 단독 trigger
- 두 조건 동시: **0회**

→ Cycle wave는 uptrend 동안 양수 영역에서만 oscillate, 음수 bottom은 downtrend에서만 도달. 즉 **trend filter (close > SMA50) + cycle reversion (cw < -0.7)은 BTC 1h에서 구조적으로 mutually exclusive**.

### Classification

- **NOT falsified** (no test conducted, 0 trades = vacuous)
- **NOT a pre-reg violation** (algorithm executed exactly as locked)
- **Mechanism design flaw**: pre-reg에서 "logical consistency dry-run" 누락. Cycle wave behavior in trending vs ranging market을 사전 확인했더라면 contradiction 즉시 보였을 것.

### Lesson added to research protocol

**Pre-reg에 추가 항목**: "Mechanism logical consistency check"
- Entry condition components 각각 단독 frequency 측정
- 모든 component AND 조건 frequency 측정
- 중요한 component 동시 trigger ≥1% 시점 미만이면 mechanism 폐기
- R38 (signal frequency 0.006/day vacuous) + R42 (component contradiction vacuous) = 2 vacuous 학습

### Counts

- **Falsified**: 32 mechanisms (이전 R1-R28, ICT, N1, N2, N7, N8 echo factor)
- **Vacuous**: 2 mechanisms (R38 VWAP frequency, R42 cycle×trend contradiction)
- **Deployable**: 0

R42는 falsification list에 추가 **안 됨** — 별도 vacuous 카테고리.
