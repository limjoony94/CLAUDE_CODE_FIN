# P0 Pre-Commit Amendment #003 — Maker Scenario Flag for Friction Model

**Amendment date**: 2026-05-01
**Original**: `experiments/p0/precommit_amendment_001.md` (3-scenario taker friction)
**Trigger**: Advisor non-blocking flag #4 — stress 0.20% RT는 taker-only 가정. Maker (post-only limits) strategies는 부적절한 friction 적용 시 false-fail 위험.

---

## Disclosure

`precommit_amendment_001.md`의 3 scenario (optimistic/realistic/stress) 모두 **taker-only execution 가정**:
- `taker_fee_pct: 0.045` per side (BingX retail taker fee)
- `slippage_pct`: market order에 대한 slippage 추정

Maker (post-only limit) execution은 다른 친화 구조:
- Maker fee: -0.020% rebate (BingX) 또는 +0.020% (some exchanges) — exchange/tier별 차이
- Slippage: maker fill probability < 100% (entry/exit 모두) — fill 안 되면 trade 자체 안 발생
- Adverse selection: maker breakout entry는 50% 이하 fill rate (memory: round25_maker_adverse_selection)

---

## Rule

**Default**: 모든 P2-P6 hypothesis test는 taker scenario (realistic + stress) 적용
**Exception**: P3에서 maker-only (post-only) mechanism이 식별되면 그 mechanism은 별도 maker scenario 평가

### Maker Scenario Definition (when applicable)

```python
SCENARIOS["maker_optimistic"] = FrictionParams(
    taker_fee_pct=0.0,           # taker 미사용
    slippage_pct=0.0,             # ideal limit fill
    fill_probability=0.5,         # adverse selection 보수
    rebate_pct=-0.020,            # BingX maker rebate
)
SCENARIOS["maker_realistic"] = FrictionParams(
    taker_fee_pct=0.0,
    slippage_pct=0.0,
    fill_probability=0.40,        # 40% maker fill (advisor history)
    rebate_pct=-0.010,            # 보수
)
```

Maker test는 `friction_model.maker_simulate(...)` 별도 entry point. taker model과 혼합 금지.

---

## Anti-Fishing Locks

1. ❌ Strategy를 taker-friction에서 fail 후 maker-friction으로 retry 금지 (mechanism 사전 분류 필수)
2. ❌ Maker scenario를 default로 사용 금지 (always taker first, maker only when explicitly post-only mechanism)
3. ❌ Mixed taker+maker 시나리오 임의 정의 금지 (unjustified parameter)
4. ✅ Maker scenario 활성화 시 fill_probability를 P5 force-flow detection에서 측정 후 calibration

---

## Implementation Note

Phase A (P2-P5) 1차 통과 시점에서는 taker only로 집중. Maker scenario 코드는 stub만 유지, 실제 활성화는 P3에서 maker mechanism 발견 시.

`friction_model.py` API:
```python
from validators.friction_model import get_friction, FrictionParams

# Default usage (taker)
fr = get_friction("realistic")  # taker realistic
cost = round_trip_cost_pct(fr)  # 0.16%

# Maker (when explicitly applicable)
fr_m = get_friction("maker_realistic")  # raises NotImplementedError until P3 activation
```

NotImplementedError는 P3에서 maker mechanism 식별 시 amendment 004로 활성화.

---

**Pre-commit signed**: Claude Code agent, 2026-05-01.
