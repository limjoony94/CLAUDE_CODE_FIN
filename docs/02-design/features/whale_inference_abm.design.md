---
template: design
version: 1.2 (ABM-customized, G0-only scope)
description: ABM core (Phase 0) design — implementation-ready spec for the G0 gate
variables:
  - feature: whale_inference_abm
  - phase: G0 (ABM build)
  - date: 2026-05-01
  - author: 임준영 + advisor + Claude Opus 4.7
  - architecture_ref: bingx_rl_trading_bot/claudedocs/whale_inference_abm_architecture_v1.1.md
  - plan_ref: docs/01-plan/features/whale_inference_abm.plan.md
---

# whale_inference_abm Design Document — G0 (ABM Build) v0.2

> **Summary**: Implementation-ready spec for Phase 0 (G0) ABM core: continuous double auction orderbook, 5 canonical agents, friction model, wealth-weighted sizing, open-system + frozen-admission window, deterministic event loop, logging schema. Includes ABIDES-vs-custom 5-day spike framework + G1-G4 preconditions.
>
> **Project**: CLAUDE_CODE_FIN
> **Phase scope**: G0 ONLY (advisor binding decision #1). G1-G4 designs deferred until G0 passes.
> **v1 scope (advisor B3 decision)**: ABM v1 = **cash-margin spot-like dynamics**. NO funding rate, NO liquidations, NO leverage. Substrate findings will NOT cover leverage-driven / funding-driven / liquidation-cascade mechanisms (those are known classes from 28-round R5/R8/R26). Acceptance: G3 substrate hypothesis space is narrowed by construction; this is intentional to keep claim sharp.
> **Author**: 임준영 + advisor + Claude Opus 4.7
> **Date**: 2026-05-01
> **Status**: Design v0.2 (advisor-reviewed, 3 BLOCKING + 3 FLAG + 3 NOTE patches applied)
> **Plan reference**: [`whale_inference_abm.plan.md`](../../01-plan/features/whale_inference_abm.plan.md)
> **Architecture reference**: [`whale_inference_abm_architecture_v1.1.md`](../../../bingx_rl_trading_bot/claudedocs/whale_inference_abm_architecture_v1.1.md)

---

## 1. Overview

### 1.1 Design Goals

1. **Phase 0 implementer can start coding from this doc** without further architecture or plan re-reads
2. **Determinism** — same seed → byte-identical trade tape (G1 null-baseline + G3 substrate audit hard requirement)
3. **Implementation flexibility within ABIDES-vs-custom spike** — design specifies behavior + interfaces, NOT internal implementation
4. **Friction model from Day 1** — no "we'll add friction later" trap (architecture v1.1 Section 6.4)
5. **Open-system honored at ABM level + tractability honored at extraction level** — frozen-admission window for G3
6. **Scope limit explicit (advisor B3 patch)**: v1 ABM = cash-margin spot-like dynamics. NO funding rate, NO liquidations, NO leverage in v1. Acknowledged consequence: G3 cannot discover substrates that depend on those mechanisms (already-known classes from R5/R8/R26). v1 narrows the discovery space to interaction-pattern substrate (multi-agent coordination, orderbook signature artifacts, regime-feedback) — sharper test, smaller surface.

### 1.2 Design Principles

- **Single-process determinism**: priority queue, no async, no parallel agents within a run
- **Behavior over implementation**: agent decision functions specified by formula + I/O contract, not by class hierarchy
- **G0 acceptance = empirical not aesthetic**: smoke test asserts trajectories, not code style
- **Documentation parity**: every G0 unit test has 1-line comment explaining which architecture/design line it enforces

---

## 2. Architecture

### 2.1 Component Diagram (G0 modules)

```
                 ┌──────────────────────────────────┐
                 │      Simulation Driver           │
                 │    (event loop + scheduler)      │
                 └──────────────┬───────────────────┘
                                │ pops events
                ┌───────────────┴───────────────┐
                │                                │
        ┌───────▼────────┐               ┌──────▼──────────┐
        │   Orderbook    │◄──────────────┤  Agent Registry │
        │  (CDA, L2)     │   submit_order│  (5 canonicals  │
        └───────┬────────┘               │   + admissions) │
                │ trade events           └──────┬──────────┘
        ┌───────▼────────┐                      │ wealth/PnL
        │  Trade Tape    │                      │
        │  (event log)   │                      │
        └───────┬────────┘                      │
                │                               │
        ┌───────▼─────────────────────────────▼─┐
        │       Wealth Tracker / Friction       │
        │   (capital ledger, fee deduction)     │
        └───────────────────────────────────────┘
                                │
                ┌───────────────▼───────────────┐
                │    Logger (structlog JSON)    │
                │  per-trade + per-bar snapshot │
                └───────────────────────────────┘
```

### 2.2 Data Flow (one event loop iteration)

```
1. Driver pops next event from priority queue (key: timestamp, sequence_no, agent_id)
2. Event type dispatched:
   - AGENT_DECISION: invoke agent.decide(orderbook_view) → Order or Hold
   - ORDER_MATCH: orderbook.match() → Trade event(s)
   - ADMISSION: registry adds new agent (open phase only)
   - BAR_TICK: emit per-bar snapshot, schedule next bar
3. Resulting events pushed back to queue with deterministic ordering
4. Wealth tracker updated; friction deducted on every trade
5. Logger emits structured event record
6. Loop continues until terminal_time reached or all-agents-bankrupt
```

### 2.3 Module Dependencies

| Module | Depends On | Imported By |
|--------|------------|-------------|
| `orderbook` | (nothing) | simulation, agents (read-only views) |
| `agents.base` | orderbook (types only) | agents.* |
| `agents.{family}` | agents.base, orderbook (read-only) | registry |
| `registry` | agents.* | simulation |
| `friction` | trade events (types) | wealth |
| `wealth` | friction, trade events | simulation, agents (read-only) |
| `simulation` | orderbook, registry, wealth, logger | (top-level) |
| `logger` | (nothing — pure structlog wrapper) | all |

Rule: agents NEVER write to orderbook directly. Agents return `Order` objects from `decide()`; simulation owns submission. This is the determinism boundary.

---

## 3. ABM Core Components

### 3.1 Orderbook (Continuous Double Auction)

**Behavior**:
- Two sorted books: bids (descending price), asks (ascending price)
- Each level: list of `(agent_id, size, sequence_no)` FIFO within level
- Order types: `LIMIT(side, price, size)`, `MARKET(side, size)`, `CANCEL(order_id)`
- Match: market orders walk opposite book until filled or empty; limit orders cross if marketable, otherwise rest
- No iceberg, no IOC variations, no stop in v1

**Interface**:
```python
class Orderbook:
    def submit(order: Order, agent_id: str, sequence_no: int) -> List[Trade]: ...
    def cancel(order_id: OrderID) -> bool: ...
    def best_bid() -> Optional[Tuple[price, size]]: ...
    def best_ask() -> Optional[Tuple[price, size]]: ...
    def snapshot(depth: int = 10) -> OrderbookSnapshot: ...  # read-only view
    def state_hash() -> str: ...  # SHA256 for determinism test
```

**State invariants** (asserted in `__post_match__`):
- bids[0].price < asks[0].price (no cross after match)
- All sizes > 0 (zero-size levels removed)
- FIFO order preserved within each level

### 3.2 Event Loop / Scheduler

**Behavior**:
- Priority queue (heapq) keyed on `(timestamp_ns, sequence_no, agent_id)`
- `sequence_no` is global monotonic counter assigned at event creation
- Tie-break: earlier timestamp wins; same timestamp → lower sequence_no wins; same both → lexicographic agent_id wins (deterministic)
- `time` is sim-internal int64 nanoseconds, NOT wall-clock
- Agents schedule their NEXT decision via `schedule_decision(delay_ns)` returning from `decide()`

**Interface**:
```python
class Scheduler:
    def __init__(seed: int, terminal_time_ns: int): ...
    def push(event: Event) -> None: ...
    def pop_next() -> Event: ...
    def now() -> int: ...  # current sim timestamp
    def is_done() -> bool: ...
```

**Determinism rule**: Scheduler is the ONLY consumer of the seed. Agents receive a derived sub-seed from registry (seeded from main seed + agent_id hash). No agent calls `random.*` directly; all RNG via `agent.rng` (a `numpy.random.Generator` instance).

### 3.3 Time Model

- Bar duration: 60 seconds sim-time = 60 × 10⁹ ns
- BAR_TICK events scheduled at `t = bar_index × 60e9`, never depending on agent activity
- Per-bar snapshot includes: orderbook L10 depth, last trade price, total volume in bar, agent wealth distribution

---

## 4. Five Canonical Agents (G1 baseline set)

Each agent is a `Decision Function` + parametric form. NOT a class with state beyond what's documented. State must be reconstructible from (seed, parameters, observed events) for determinism.

### 4.1 Momentum Agent

**Decision**:
```
look_back_window = N bars
mid_price[t] = (best_bid + best_ask) / 2
signal = sign(mid_price[t] - mid_price[t - N])
if signal != 0 AND |signal| > confirmation_threshold:
    emit MARKET order on signal direction, size = wealth_fraction × current_wealth / mid_price
```
**Parameters**: `N ∈ {3, 5, 10}` bars, `confirmation_threshold = 0` (any signal triggers in baseline), `wealth_fraction = 0.05`
**Decision frequency**: every bar (schedule_delay = 60e9 ns)

### 4.2 Mean-Reversion Agent

**Decision**:
```
ma_window = N bars, MA[t] = mean(mid_price[t-N], ..., mid_price[t-1])  # N PRIOR prices, EXCLUDING current
deviation = (mid_price[t] - MA[t]) / MA[t]
if |deviation| > threshold:
    emit MARKET order opposite to deviation direction, size = wealth_fraction × current_wealth / mid_price
```
**MA semantics (clarified per advisor Day 1-7 checkpoint v0.4 patch)**: MA at time t is the mean of the N prior prices, NOT including the current mid. Including current would make the price contribute to its own deviation calculation, attenuating the signal. Implementation: append current to history AFTER computing deviation.

**Parameters (v0.6 calibration)**: `N ∈ {10, 20, 30}` bars, `threshold = 0.001` (0.1%) — calibrated post-G0 diagnostic 2026-05-01, was 0.005 in v0.4 but produced 0 trades in 1k-bar smoke. See `results/g0_smoke/meanrev_diagnostic.md`. `wealth_fraction = 0.05`
**Decision frequency**: every bar

### 4.3 Market-Maker Agent

**Decision**:
```
target_spread = base_spread + inventory_skew × current_inventory
mid = (best_bid + best_ask) / 2
emit LIMIT BID at mid - target_spread/2, size = base_size
emit LIMIT ASK at mid + target_spread/2, size = base_size
cancel previous unfilled quotes before re-quoting
```
**Parameters**: `base_spread = 0.001` (10 bps), `inventory_skew = 0.0001 per unit`, `base_size = 0.01 × current_wealth`
**Decision frequency**: every 10 seconds (schedule_delay = 10e9 ns) — more frequent than directional agents
**State**: `current_inventory` (long_qty - short_qty) — reconstructible from trade history

### 4.4 Random Agent

**Decision**:
```
side = rng.choice(['buy', 'sell'])
order_type = rng.choice(['MARKET', 'LIMIT'])
if LIMIT:
    price_offset = rng.uniform(-0.01, 0.01) × mid_price
size = wealth_fraction × current_wealth / mid_price
```
**Parameters**: `wealth_fraction = 0.02`
**Decision frequency**: Poisson arrivals with rate λ = 1/120s (avg 1 trade per 2 minutes)

### 4.5 Piggyback Agent

**Decision**:
```
top_performer_id = argmax over agents of (wealth[t] / wealth[t - lookback])
last_action = trade_tape.last_trade_by(top_performer_id)
if last_action.timestamp > t - delay:
    emit MARKET order copying last_action.side, size = wealth_fraction × current_wealth / mid_price
```
**Parameters**: `lookback = 1000 bars`, `delay = 60s` (1-bar lag), `wealth_fraction = 0.03`
**Decision frequency**: every bar
**Anti-self-reference**: Piggyback agent excluded from "top_performer_id" candidates (would be circular). ALL piggyback-family agents excluded (not just self) to prevent piggyback chain artifacts.
**Cold-start (advisor B2 patch)**: For `t < lookback × BAR_DURATION_NS` (i.e., first 1000 bars), piggyback agent emits NO trades and NO quotes (simply returns Hold from `decide()`). After cold-start window, normal piggyback logic activates. Test: assert piggyback agents have 0 trades in `bar_index < 1000` in smoke test.

**Wealth-growth metric (advisor v0.4 checkpoint patch)**:
- **Window**: rolling `lookback = 1000 bars` (matches `lookback` parameter above; sliding window of bars[t-1000:t])
- **Metric**: ratio `wealth[t] / wealth[t-lookback]`. Higher = more growth. Avoids signed-return ambiguity for negative-equity edge cases.
- **Bankrupt agents**: EXCLUDED from leaderboard. They cannot be "followed" — bankrupt agents have no future actions to copy.
- **Cold-start interaction**: leaderboard only computable when ABM has run ≥ `lookback` bars. Piggyback agent's own cold-start guarantees no premature leaderboard reads.
- **Computed by**: simulation/wealth_tracker maintains rolling wealth snapshots per agent; provides leaderboard on demand to context dict.

### 4.6 Agent Population Composition (G0 smoke test)

| Agent | Count | Initial Wealth |
|-------|-------|----------------|
| Momentum | 3 (N=3, 5, 10) | 1000 each |
| Mean-Reversion | 3 (N=10, 20, 30) | 1000 each |
| Market-Maker | 2 (different base_spread) | 1000 each |
| Random | 5 | 1000 each |
| Piggyback | 2 | 1000 each |
| **Total** | **15 initial agents** | **15000 total wealth** |

Open-system admissions add more agents (Section 6).

### 4.6.5 Simulation Driver Responsibilities (advisor v0.4 checkpoint patch)

The `Simulation` class (Day 8-10) owns dispatch + cross-cutting state that agents cannot manage themselves:

| Responsibility | Detail |
|----------------|--------|
| **Active orders tracking** | `dict[AgentID, set[OrderID]]` — for cancel-and-requote (e.g., MarketMaker every 10s) |
| **Cancel-and-requote (MM)** | Before submitting MM's new LIMIT intents, sim cancels all `active_orders_by_agent[mm_id]` via `orderbook.cancel()`. Agent emits intents only; sim handles cancellation. |
| **Order ID + sequence assignment** | Agent emits `OrderIntent`; sim wraps into `Order` with `order_id = agent.next_order_id()` and `sequence_no = scheduler.next_sequence_no()`. |
| **Wealth update on fills** | After `orderbook.submit()` returns trades, sim invokes `wealth_tracker.apply_trade(trade)` for both sides + `friction.fee()` deduction. |
| **Bankruptcy detection + removal** | After wealth update, if `agent.current_wealth ≤ BANKRUPTCY_THRESHOLD`, sim calls `registry.remove_agent()` and emits `AGENT_REMOVED` event in tape. |
| **Wealth snapshot for leaderboard** | Per-bar (or per-T-bars), sim calls `wealth_tracker.snapshot()` to maintain rolling wealth history needed by Piggyback context. |
| **Context construction for `agent.decide()`** | Sim builds the `context` dict: `wealth_growth_leaderboard`, `last_actions_by_agent`, `piggyback_excluded_ids`. Agent receives read-only view. |
| **Push next decision event** | After `agent.decide()`, sim pushes `AgentDecisionEvent` at `now() + agent.next_decision_delay_ns()` (or `next_bar_start + agent.decision_offset_ns` for bar-aligned agents — TBD in `simulation.py` design). |
| **MarketMaker inventory update** | Sim detects fills involving MM as participant; calls `mm_agent.update_inventory(signed_size)`. |

**Determinism rule**: simulation MUST push events in deterministic order. When the same scheduler tick triggers multiple downstream events (e.g., trade match → wealth update → next decision schedule), order = (1) wealth update, (2) bankruptcy check, (3) re-quote schedule, (4) tape emit. Document this ordering in `simulation.py` docstring.

### 4.7 Decision Jitter (advisor B1 patch — alphabetical-order artifact 제거)

**Problem identified**: All directional agents (momentum/mean-rev/piggyback) decide "every bar" → same timestamp. Tie-break = `agent_id` lexicographic. Result: agents whose ID starts with 'a' always trade first within each bar across the entire run. G2 wealth concentration may then reflect alphabetical-order first-mover advantage rather than strategy.

**Fix**: per-agent jitter offset drawn from agent's seeded sub-RNG at registry time:
```python
# in registry.add_agent(agent):
agent.decision_offset_ns = agent.rng.uniform(0, BAR_DURATION_NS)  # [0, 60e9]
```

When agent schedules its next decision:
```python
# instead of: schedule_decision(timestamp = bar_start)
schedule_decision(timestamp = bar_start + agent.decision_offset_ns)
```

**Properties**:
- Determinism preserved (offset is from seeded RNG; same seed → same offsets)
- Alphabetical-order artifact removed (offsets distributed within bar)
- Decision frequency unchanged (still 1 decision per bar for directional agents)
- Per-agent offset is fixed for the run (drawn once at registry time, NOT re-drawn each bar)

**Applies to**: momentum, mean-reversion, piggyback (per-bar deciders). Market-maker decides every 10s with its own jitter `rng.uniform(0, 10e9)`. Random agent already Poisson-arrival, no jitter needed.

**Test**: integration test asserts that across 1000-bar run, the FIRST trader within each bar varies (≥ 5 distinct first-traders observed across bars), not always the same agent.

---

## 5. Wealth-Weighted Sizing

**Mechanism**: Order size scales with current wealth. Bankrupt agents (wealth ≤ 0) are removed from registry but their identifier preserved in trade_tape for inverse machinery.

**Formula**:
```
size = wealth_fraction × current_wealth / mid_price
size = clip(size, MIN_ORDER_SIZE, MAX_ORDER_SIZE)
size = round to LOT_STEP
```

**Constants**:
- `MIN_ORDER_SIZE = 0.0001` (BTC equivalent)
- `MAX_ORDER_SIZE = 1.0` (BTC equivalent)
- `LOT_STEP = 0.0001`

**Wealth update on trade**:
```
realized_pnl = (exit_price - entry_price) × position_size × side_sign
fees = friction.fee(order_role, notional)  # see Section 6
new_wealth = old_wealth + realized_pnl - fees
```

**Bankruptcy threshold**: `wealth ≤ 1.0` (1 USDT equivalent) → remove from registry. Triggers AGENT_REMOVED event in tape.

---

## 6. Friction Model

**Critical**: in ABM from Day 1, NOT added later (architecture v1.1 Section 6.4 enforcement).

**Fee schedule** (BingX rate):
- Taker fee: 0.05% of notional
- Maker fee: 0.02% of notional

**Role assignment**:
- MARKET order → always taker (fee 0.05%)
- LIMIT order that crosses on submission → taker for crossed quantity
- LIMIT order that rests → maker when later matched (fee 0.02%)

**Spread**: emergent from orderbook (no synthetic spread injection). Market-maker agent quotes set tightest spread; market structure determines actual.

**Slippage**: emergent from book depth. MARKET orders walking the book naturally pay worse prices for larger size.

**Funding rate**: NOT modeled in v1. **Decision finalized (advisor B3 patch + user 2026-05-01)**: this is permanent v1 scope, not "deferred if needed." v2 may add funding if v1 yields substrate findings worth extending. v1 substrate hypotheses CANNOT depend on funding by construction.

**Liquidations**: NOT modeled in v1 (no leverage in ABM v1, all orders fully collateralized). Same decision logic as funding above.

**Consequence acknowledged**: any "29th attempt" mechanism class found by this ABM cannot be a leverage-cascade or funding-skim mechanism. Those are already-explored mechanism families (R5/R8/R13/R26). v1 ABM searches the space NOT explored by 28-round retail BTC perp work: emergent multi-agent interaction patterns within spot-like dynamics. This is the deliberate narrowing for sharp G3 test.

**Friction interface**:
```python
class Friction:
    def fee(role: Literal["maker", "taker"], notional: float) -> float: ...
    def slippage_observed(submitted_size: float, executed_avg_price: float, expected_price: float) -> float: ...  # diagnostic, not deducted
```

---

## 7. Open-System + Frozen-Admission Window

**Two-phase ABM run** (architecture v1.1 patch 3):

```
Phase 1: OPEN          (t = 0 to T_open)
  - New agents may join via ADMISSION events
  - Joining strategy distribution: uniform over 5 canonical families
  - Joining wealth: 100 (smaller than initial 1000 to ensure incumbents have first-mover wealth advantage)
  - Joining rate: Poisson(λ = 1/300s) (avg 1 new agent per 5 minutes)
  - Used for G2 wealth-concentration evaluation

Phase 2: FROZEN        (T_open to T_open + T_extract)
  - ADMISSION events DISABLED in scheduler
  - Existing agents continue trading
  - Inverse machinery applied to trajectories from this window only
  - Used for G3 substrate extraction
```

**Default values for G0 smoke test**:
- `T_open = 7000 sim-bars × 60e9 ns = 420e12 ns ≈ 7 sim-hours`
- `T_extract = 3000 sim-bars × 60e9 ns = 180e12 ns ≈ 3 sim-hours`
- Total: 10000 sim-bars

**Implementation**:
```python
class AdmissionScheduler:
    def __init__(open_until_ns: int, joining_rate_lambda: float): ...
    def next_admission_event(now_ns: int) -> Optional[Event]:
        if now_ns >= self.open_until_ns:
            return None  # frozen
        delay_ns = rng.exponential(1 / joining_rate_lambda)
        return AdmissionEvent(timestamp=now_ns + delay_ns, ...)
```

**Documentation requirement**: every G3 substrate result must include `extraction_window = (T_open, T_open + T_extract)` for reproducibility.

---

## 8. Logging Schema

**Format**: structlog JSON, 1 record per line (newline-delimited JSON, NDJSON), parquet rollup hourly.

**Per-trade event** (`trade_tape.ndjson`):
```json
{
  "event_type": "TRADE",
  "timestamp_ns": 1234567890000000000,
  "sequence_no": 42,
  "trade_id": "t_00000042",
  "buyer_agent_id": "momentum_n5_seed1234",
  "seller_agent_id": "mm_a_seed5678",
  "price": 50000.5,
  "size": 0.01,
  "buyer_role": "taker",
  "seller_role": "maker",
  "buyer_fee": 0.025,
  "seller_fee": 0.010
}
```

**Per-bar snapshot** (`bar_snapshots.ndjson`):
```json
{
  "event_type": "BAR_SNAPSHOT",
  "timestamp_ns": 600000000000,
  "bar_index": 10,
  "best_bid": 50000.0,
  "best_ask": 50001.0,
  "mid_price": 50000.5,
  "bid_depth_l10": [[50000.0, 0.5], [49999.5, 0.3], ...],
  "ask_depth_l10": [[50001.0, 0.4], [50001.5, 0.6], ...],
  "bar_volume": 0.85,
  "wealth_dist": {"momentum_n3_seed111": 1023.4, "mm_a_seed5678": 998.7, ...}
}
```

**Per-decision log** (`agent_decisions.ndjson`, optional, ON for G0 smoke + G1 IRL training):
```json
{
  "event_type": "DECISION",
  "timestamp_ns": 1234567890000000000,
  "agent_id": "momentum_n5_seed1234",
  "agent_family": "momentum",
  "observed_state": {"orderbook_imbalance": 0.12, "trend_regime": 1},
  "action": {"type": "MARKET", "side": "buy", "size": 0.01},
  "reason_code": "signal_positive_above_threshold"
}
```

**Storage**: `${ABM_DATA_DIR}/g0_smoke/{run_id}/{trade_tape,bar_snapshots,agent_decisions}.ndjson`

**ABM_DATA_DIR hard-fail enforcement (advisor v0.4 checkpoint patch)**:
Logger initialization MUST hard-fail (RuntimeError) if:
1. `ABM_DATA_DIR` env var is unset
2. `ABM_DATA_DIR` value contains the substring `"OneDrive"` (case-insensitive)

Rationale: BUG#58 in trading bot codebase = OneDrive sync lock corrupted state.json. Same risk applies to high-frequency NDJSON writes from per-decision logger (~9M records per smoke run). Hard fail at init prevents silent corruption discovered at G0 acceptance review.

```python
# In logger init
import os
data_dir = os.environ.get("ABM_DATA_DIR")
if not data_dir:
    raise RuntimeError("ABM_DATA_DIR not set. Per design v0.2 F2 / v0.4 patch, must point to NON-OneDrive path.")
if "onedrive" in data_dir.lower():
    raise RuntimeError(f"ABM_DATA_DIR={data_dir} contains 'OneDrive'. Use local-only path.")
```

---

## 9. Determinism Model + Reproducibility Tests

### 9.1 Determinism Architecture (advisor binding decision #3)

- **Single-process**, event-driven discrete simulation
- **Priority queue** keyed on `(timestamp_ns, sequence_no, agent_id)` — total order on events
- `sequence_no` = global monotonic int, assigned at `Event` instantiation
- `agent_id` = string, lexicographic compare for tie-break
- All RNG flows from main `seed`:
  - Scheduler RNG: `seed`
  - Per-agent RNG: `seed_for_agent = hash(seed, agent_id) mod 2^32`
- No `random.*`, no `np.random.*` outside controlled `Generator` instances
- No wall-clock time use
- No parallel execution

**What this rules out**:
- multiprocessing.Pool agents
- asyncio agent decisions
- threading.Thread anywhere in sim path
- `time.time()` calls

### 9.2 Reproducibility Tests

```python
# tests/test_determinism.py

def test_trade_tape_byte_identical_same_process():
    """Same seed → 3 reruns in same Python process → identical SHA256."""
    hashes = []
    for _ in range(3):
        sim = Simulation(seed=42, terminal_time_ns=600 * 60 * 10**9)  # 10 sim-hours
        tape = sim.run()
        hashes.append(hashlib.sha256(tape.to_bytes()).hexdigest())
    assert len(set(hashes)) == 1, f"Non-deterministic: {hashes}"

def test_trade_tape_byte_identical_cross_process():
    """Same seed → fresh Python interpreter → identical SHA256."""
    h1 = subprocess.check_output(["python", "-m", "abm.cli", "--seed=42", "--hash"]).strip()
    h2 = subprocess.check_output(["python", "-m", "abm.cli", "--seed=42", "--hash"]).strip()
    assert h1 == h2

def test_trade_tape_differs_different_seed():
    """Sanity: different seed → different hash."""
    sim1 = Simulation(seed=42, terminal_time_ns=600 * 60 * 10**9)
    sim2 = Simulation(seed=43, terminal_time_ns=600 * 60 * 10**9)
    assert sim1.run().to_bytes() != sim2.run().to_bytes()
```

**Determinism budget**: 3-second tolerance for 10-sim-hour run. If reruns differ, build is broken — DO NOT release.

---

## 10. ABIDES-vs-Custom Decision (SPIKE SKIPPED — 2026-05-01)

### 10.0 Decision: Custom Build (advisor reconcile call 2026-05-01)

**Status**: SPIKE SKIPPED. Custom build proceeds directly.

**Evidence triggering skip**:
- ABIDES JPMorgan repo `jpmorganchase/abides-jpmc-public` archived 2025-06-02 (read-only)
- Repo statement: "we do not do technical support, nor consulting"
- Codebase composition: 81% Jupyter Notebook, 19% Python (research-shape, not production-shape)
- No pip distribution; install via `git clone` + `install.sh` only
- Python version compatibility unverified (no maintenance = unbumped pins likely)
- Community fork (`abides-sim/abides`) exists but fragmented; no clear canonical path

**Decision logic**:
- Spike framework criterion 2 ("D5 estimate > 10 person-days → custom") triggered by inspection: archived + notebook-heavy + no-pip = D5 well above 10 days without running the spike
- Spike framework tie-breaker (custom default for borderline) reinforced by archive status
- Running D1-D5 on archived code adds zero information — outcome already determined

**Additional context (CLAUDE_CODE_FIN-specific)**:
- BUG#58 precedent (OneDrive sync lock on state.json) — ABIDES + notebook checkpoints + intermediate artifacts on OneDrive = friction risk asymmetric to custom code with controlled I/O
- 5-day spike budget rolls into custom build → total Phase 0 unchanged at 3 weeks (15 days)

**Reference orderbook implementations to READ (not depend on)**:
- `abides-markets/orderbook.py` in archived ABIDES (reference for CDA pattern)
- `mbtgateway` (smaller, readable matching engine)
- `lobster-tools` (academic, well-documented limit order book)
- **30-minute pre-build read budget**: spend ≤30 minutes reading one of the above BEFORE writing custom orderbook line one. Saves ~1 day of false starts.

### 10.1 Custom Build Implementation Order (replaces 10.1-10.6 spike framework)

| Days | Deliverable |
|------|-------------|
| 1-3 | Orderbook (CDA, state_hash, invariants) + scheduler + determinism scaffolding |
| 4-7 | 5 canonical agents per Section 4 + unit tests |
| 8-10 | Wealth tracker + friction + bankruptcy + admission scheduler + frozen-window |
| 11-13 | Logger NDJSON + per-decision log + integration tests |
| 14-15 | Smoke 1k/10k bars + reproducibility tests (same/cross-process) + schema diff vs BingX Phase 1 + G0 acceptance review |

Total: 15 days = 3 weeks. No schedule slip vs original timeline.

### 10.2 (DEPRECATED) Original Spike Framework

The original 5-day ABIDES-vs-custom spike framework (5 deliverables D1-D5, decision criteria, tie-breaker) is preserved in git history (commit `78f802d` design v0.1, commit `adfe215` design v0.2 Section 10) for reference but no longer executes. Decision recorded as if spike completed Day 5 with criterion 2 triggered.

### 10.2 Five Required Deliverables

1. **D1**: Run vanilla ABIDES with 2 simple agents + 1 market-maker on synthetic timeline. Produce trade tape NDJSON.
2. **D2**: Patch ABIDES to add wealth-weighted order sizing (Section 5). Verify simulation still runs to completion.
3. **D3**: Patch ABIDES to add a 4th agent type (mean-reversion per Section 4.2). Verify it integrates with registry.
4. **D4**: Document where crypto-perp specifics (funding, liquidations, leverage) WOULD live in ABIDES architecture. File paths + modification surface.
5. **D5**: Estimate person-days for D4 extensions. Range estimate (low/expected/high).

### 10.3 Decision Criteria (priority order)

1. **Per-patch effort gate**: if ANY of D2/D3 takes > 1 day each → ABIDES extension cost too high → **custom**
2. **Crypto-perp extension gate**: if D5 estimate > 10 person-days → **custom** (would rewrite half of ABIDES anyway). NOTE (advisor B3 patch): since v1 scope drops funding/liquidation/leverage, "crypto-perp extension" here means only what's needed for spot-like cash-margin parity with BingX trade tape format. D5 should re-estimate against this narrower scope.
3. **Greenlight**: D2/D3 each ≤ 1 day AND D5 ≤ 10 days → **ABIDES**
4. **Tie-breaker (advisor F3 disambiguation)**: **custom**, but fires ONLY when criteria 1-3 are inconclusive — specifically when D5 estimate range straddles the 10-day threshold (e.g., low=8, expected=10, high=14 days). The tie-breaker does NOT override a clean greenlight (criterion 3 met unambiguously); it only resolves borderline cases in favor of custom.

### 10.4 Spike Abandon Trigger

Any single deliverable fails 2 attempts → **switch to custom on day 4-5** with whatever budget remains.

### 10.5 Post-Spike Budget

- If ABIDES chosen: 1 week patch-and-extend → G0 smoke test in week 3
- If custom chosen: 2 weeks build → G0 smoke test in week 3

Both paths converge on Phase 0 = 3 weeks total.

### 10.6 Spike Output Document

`results/g0_smoke/abides_spike_summary.md` with:
- Per-deliverable: time spent, success/failure, code patches if applicable
- Decision: ABIDES or custom
- Rationale: which criteria triggered
- Next-step plan with day-level granularity

---

## 11. Test Plan (G0 Acceptance)

### 11.1 Unit Tests

| Module | Test cases |
|--------|------------|
| `orderbook` | (a) limit add/cancel/match, (b) market walk-the-book, (c) state invariants, (d) state_hash() consistency |
| `agents.momentum` | (a) signal computation for trending series, (b) flat series no-signal, (c) wealth-weighted size correct |
| `agents.mean_reversion` | (a) deviation threshold trigger, (b) size correct |
| `agents.market_maker` | (a) quote on both sides, (b) cancel-and-requote on inventory skew |
| `agents.random` | (a) Poisson arrival statistical test (chi-square), (b) action distribution uniform |
| `agents.piggyback` | (a) top performer detection, (b) self-reference exclusion, (c) lag respected |
| `friction` | (a) taker fee 0.05% on MARKET, (b) maker fee 0.02% on resting LIMIT, (c) crossed-LIMIT taker on cross qty |
| `wealth` | (a) PnL update on trade, (b) bankruptcy removal, (c) order_size respects MIN/MAX |
| `scheduler` | (a) priority order, (b) tie-break by sequence_no, (c) terminal_time stop |
| `admission` | (a) admissions occur during open, (b) NO admissions after T_open, (c) joining wealth correct |

### 11.2 Integration Tests

- `test_smoke_1k_bars`: 1000-bar simulation completes without crash, produces non-empty trade_tape, all 15 initial agents have ≥ 1 trade
- `test_smoke_10k_bars`: 10000-bar simulation completes; admission events present pre-T_open, absent post-T_open; final wealth distribution non-uniform
- `test_friction_present`: agents lose money even when their strategy is "correct" by some margin (sanity that friction deducts)
- `test_no_lookahead_in_decisions`: agent.decide() called with snapshot at time T cannot reference any event with timestamp > T (causality)

### 11.3 Determinism Tests

(Section 9.2 above — `test_trade_tape_byte_identical_*`)

### 11.4 G0 Pass Criterion (verbatim from plan + advisor N3 addition)

All of:
- ABM platform decision finalized (spike completed)
- Continuous double auction operational
- 5 canonical agents implemented + unit-tested
- Wealth-weighted sizing operational
- Friction model integrated
- Open-system admission events + frozen-admission window mechanism
- Logging schema producing valid NDJSON
- Smoke test: 1000-bar run produces non-trivial price evolution + all 5 agent families active + no crashes
- Reproducibility: SHA256 trade-tape identity verified (same-process AND cross-process)
- **Schema diff (advisor N3)**: ABM `bar_snapshots` schema vs BingX Phase 1 L2 collector output schema diff'd; field-by-field reconciliation plan documented in `results/g0_smoke/schema_parity.md`. Mismatched fields require either (a) ABM logging schema patch or (b) G4-stage adapter layer documented now. CANNOT pass G0 without this artifact.
- Per-agent decision jitter test (Section 4.7) passes: ≥ 5 distinct first-traders observed across 1000 bars
- Piggyback cold-start test (Section 4.5) passes: 0 piggyback trades in `bar_index < 1000`

---

## 12. Anti-Circularity Reducibility — Algorithm Detail

(Advisor binding decision #4. NOT implemented in G0, but interface owed by G0 logging schema. Full implementation is G3 work.)

### 12.0 Operational Targets vs Final Calibration (advisor N2 patch)

Algorithm details below are **operational targets** for G3 implementation. Final thresholds (R² boundary, MI nat threshold, AST allowed-symbols set) will be **calibrated at G3 entry (week 13)** based on actual candidate substrate characteristics — pre-specifying them in detail without seeing real candidates risks ossifying definitions in ways that don't match the substrate space we encounter.

Pre-registration discipline (architecture v1.1 patch 2) covers any threshold change: each calibration produces a git-committed prereg file BEFORE substrate evaluation runs. Threshold movement after results are visible is excluded.

### 12.1 Per-Substrate Audit (3 layers, cheap-first)

For each candidate substrate `f_candidate(trajectory) → scalar`:

**Layer A — Symbolic match**:
- Check if `f_candidate`'s expression uses ONLY parameters of an explicit_strategy `s_i` and standard arithmetic
- If yes → mark "explicit-derived"
- Implementation: AST inspection of `f_candidate` source; reject if AST contains only allowed-symbols set

**Layer B — Linear reducibility (OLS R²)**:
- For each `s_i`, compute `readout(s_i, trajectory)` (scalar feature derived from explicit strategy)
- Fit `f_candidate ≈ Σ α_i · readout_i + β` via OLS on training trajectories
- If R² > 0.9 → mark "explicit-derived"

**Layer C — Conditional mutual information**:
- Compute `I_total = I(f_candidate; future_action_T+1)` — predictive power of candidate alone
- Compute `I_cond = I(f_candidate; future_action_T+1 | explicit_readouts_joint)` — predictive power of candidate AFTER conditioning on explicit strategies
- If `I_cond ≤ THRESHOLD_NATS` → mark "explicit-derived"
- MI estimator: KSG (Kraskov-Stögbauer-Grassberger) for continuous, plug-in MLE for discrete

**Calibration plan for THRESHOLD_NATS (advisor N1 patch)**:
Initial value `0.05 nats` is a placeholder. At G3 entry (week 13):
1. Generate K=1000 independent random feature functions (e.g., random linear combinations of orderbook columns + small noise)
2. For each random feature, compute `I_cond` against actual ABM trajectories
3. Set `THRESHOLD_NATS` = 95th percentile of the random-feature `I_cond` distribution
4. Pre-register the calibrated threshold + the random-feature generator seed BEFORE any candidate substrate evaluation
5. Document calibration result in `g3_substrate/calibration_{date}.json`

Rationale: this defines "explicit-derived" as "no more predictive than what 95% of random features achieve." Defensible threshold instead of arbitrary number.

### 12.2 Pass Criteria

ALL of: A rejects "explicit-derived" AND B rejects "explicit-derived" AND C rejects "explicit-derived" AND predictive_lift ≥ 5%

### 12.3 Operational Artifacts

- `substrate/audit.py` — implements `audit_layer_a()`, `audit_layer_b()`, `audit_layer_c()` returning `(passed: bool, evidence: dict)`
- `g3_substrate/audits/{substrate_id}.json` — per-substrate audit result with versioned thresholds
- Threshold change → new pre-registration (architecture v1.1 patch 2 + this design)

### 12.4 What G0 Logging Owes Layer C

Per-decision log (Section 8) must include:
- `observed_state` rich enough to reconstruct `explicit_readouts_joint(t)` post-hoc
- `action` to compute `future_action_T+1`
- `agent_family` to filter by strategy class

This is why per-decision log is REQUIRED for G0 smoke (not "optional"). Section 8 updated accordingly.

---

## 13. G1-G4 Design Preconditions (Interface G0 Owes)

(Advisor binding decision #1 — 1-page section listing what G0 must produce so G1+ designs aren't blocked)

### 13.1 For G1 (3-anchor MVP)

- **Trade tape NDJSON** with all fields in Section 8 (per-trade event)
- **Per-decision log NDJSON** for IRL training (Section 8 + Section 12.4)
- **Agent_family field** in every record (for null-baseline calibration grouped by family)
- **Reproducibility**: G1 evaluation needs identical trajectories across reruns to compare anchors fairly
- **Bar snapshot NDJSON** with orderbook imbalance computable from L10 depth (for IRL state discretization)

### 13.2 For G2 (Wealth-concentration validity)

- **Wealth distribution per bar** (Section 8 bar snapshot `wealth_dist` field)
- **Agent admission events logged** (so post-hoc Gini calculation reproducible)
- **Agent removal events logged** (bankruptcies)
- **Initial wealth distribution recorded in run metadata**

### 13.3 For G3 (New-substrate discovery)

- **Frozen-admission window timestamps** in run metadata (so extraction window unambiguous)
- **explicit_strategies catalog** stored with each run as JSON: list of agent decision functions + parameter signatures (Section 4 specs serialized)
- **Per-decision log** (Section 12.4 requirement)
- **Substrate prereg directory** structure: `prereg/{substrate_id}_{date}_{git_hash}.md`

### 13.4 For G4 (Real-data forward-predictive validity)

- **Detector function interface** committed (architecture v1.1 Section 5):
  ```python
  class SubstrateDetector:
      def detect(l2_window: L2Snapshot) -> Signal: ...
  ```
- **L2 snapshot schema** defined in G0 (must match BingX Phase 1 collector output schema)
- This requires G0 to verify schema parity with `bingx_rl_trading_bot/scripts/data_pipeline/...` before declaring G0 complete

---

## 14. Implementation Order (Phase 0)

```
Week 1 (Days 1-5): ABIDES-vs-Custom Spike
  Day 1: ABIDES install, vanilla run (D1)
  Day 2: D2 wealth-weighted patch
  Day 3: D3 4th agent patch
  Day 4: D4 crypto-perp surface documentation
  Day 5: D5 person-days estimate + decision + summary doc

Week 2 (Days 6-10): Build (path determined by spike)
  Day 6-7: Orderbook + scheduler + determinism scaffolding
  Day 8-9: 5 canonical agents + unit tests
  Day 10: Wealth tracker + friction + bankruptcy logic

Week 3 (Days 11-15): Integration, logging, smoke
  Day 11-12: Logger (NDJSON) + admission scheduler + frozen-window
  Day 13: Smoke test 1k bars + 10k bars
  Day 14: Reproducibility tests (same-process + cross-process)
  Day 15: G0 acceptance review against Section 11.4 checklist
```

---

## 15. Coding Conventions (ABM project, Python research)

### 15.1 Naming

| Target | Rule | Example |
|--------|------|---------|
| Modules | snake_case | `orderbook.py`, `agents/momentum.py` |
| Classes | PascalCase | `Orderbook`, `MomentumAgent`, `Friction` |
| Functions | snake_case | `submit_order()`, `next_admission_event()` |
| Constants | UPPER_SNAKE_CASE | `MIN_ORDER_SIZE`, `LOT_STEP` |
| Test files | `test_{module}.py` | `test_orderbook.py` |
| Agent IDs | `{family}_{params}_seed{N}` | `momentum_n5_seed1234`, `mm_b_seed5678` |

### 15.2 Imports

```python
# 1. stdlib
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Tuple

# 2. third-party
import numpy as np
import structlog

# 3. local
from abm.orderbook import Orderbook, OrderbookSnapshot
from abm.agents.base import Agent
```

### 15.3 Type Hints

All public functions and class methods type-hinted. `mypy --strict` clean.

### 15.4 Docstrings

Module: 1-line purpose. Class: 1-paragraph behavior + invariants. Function: NumPy-style docstring for non-trivial logic, 1-line for trivial.

### 15.5 No Comments Beyond `# WHY`

Per CLAUDE.md global instructions: comments only for non-obvious WHY (constraint, invariant, workaround).

---

## 16. Risks and Open Items

| Item | Status | Resolution path |
|------|--------|-----------------|
| ABIDES API may have changed since last public release | Open | Day 1 of spike resolves |
| structlog NDJSON parquet rollup performance at scale | Open | Defer until volume hurts; pure NDJSON acceptable for G0 smoke |
| Determinism may break if numpy minor version differs | Open | Pin requirements.txt to exact versions; CI runs versioned environment |
| MI estimator (KSG) computational cost for Layer C | Deferred to G3 | Sample if needed; Layer C is rarest-applied of three |
| L2 snapshot schema parity with BingX Phase 1 collector | Open | Section 11.4 G0 acceptance criterion (per N3 patch) |
| **F1 — structlog wall-clock leak**: structlog default config emits `event_time` from wall-clock → cross-process determinism hash mismatch | Flag | During implementation: configure structlog to use sim-time OR strip wall-clock fields BEFORE SHA256 in `test_trade_tape_byte_identical_*`. Test catches this if missed. |
| **F2 — Per-decision log volume on OneDrive**: ~9M records per smoke run (60 events/bar × 15 agents × 10000 bars). BUG#58 (state.json OneDrive sync lock) precedent suggests OneDrive may sync-lock during high-frequency NDJSON writes | Flag | Write per-decision logs to `${ABM_DATA_DIR}` set to non-OneDrive path (e.g., `C:\abm_runtime\` outside OneDrive). On G0 completion, optional move to OneDrive for archival. Document path requirement in README. |
| Cash-margin scope vs perp-realistic gap | Accepted (B3 (a)) | v1 explicitly does not cover funding/liquidation. v2 may extend if v1 G3 yields. Documented in Section 1.1 + 6 + 1 header. |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-05-01 | Initial G0-only design from architecture v1.1 + advisor 4 binding decisions | 임준영 + advisor + Claude Opus 4.7 |
| 0.2 | 2026-05-01 | Advisor review patches: B1 decision jitter, B2 piggyback cold-start, B3 v1 scope = cash-margin spot-like (user (a)), F1 structlog risk, F2 OneDrive log volume risk, F3 ABIDES tie-breaker disambiguation, N1 MI threshold calibration, N2 Section 12.0 phrasing, N3 schema diff G0 acceptance | 임준영 + advisor + Claude Opus 4.7 |
| 0.3 | 2026-05-01 | ABIDES archived 2025-06-02 discovered → spike SKIPPED → custom build proceeds directly. Section 10 rewritten with skip rationale + custom build 15-day implementation order. Total Phase 0 unchanged at 3 weeks. | 임준영 + advisor + Claude Opus 4.7 |
| 0.4 | 2026-05-01 | Day 1-7 advisor checkpoint: MeanReversion MA semantics clarified (excludes current price); Piggyback wealth-growth metric specified (rolling lookback ratio, bankrupt excluded, all-piggyback excluded); Simulation driver responsibilities enumerated (Section 4.6.5 NEW: cancel-and-requote, order_id+seq assignment, wealth update, bankruptcy detection, leaderboard maintenance, context construction, deterministic event ordering); Logger ABM_DATA_DIR hard-fail enforcement (Section 8). | 임준영 + advisor + Claude Opus 4.7 |
| 0.6 | 2026-05-01 | G0 → G1 transition advisor caveat #2: MeanReversion threshold 0.005 → 0.001 (5× looser). 1k-bar diagnostic showed 0 MR trades at 0.5% threshold; 0.1% threshold produces 62 MR trade legs (G1 IRL signal sufficient). Strategy character preserved (still contrarian fade). See `results/g0_smoke/meanrev_diagnostic.md`. | 임준영 + advisor + Claude Opus 4.7 |
| 0.7 | 2026-05-01 | G1 → G2 transition advisor signoff caveats: (a) IRL anchor demoted from "primary inverse-recovery" to "policy validation tool" — the implemented MVP is behavioral cloning (P(a\|s) histogram with empirical-tercile state discretization), not MaxEnt IRL. Substrate hypothesis generation at G3 should lean on Signature + Parametric (interpretable cluster/family labels) rather than IRL P(a\|s) tables. (b) MeanReversion explicitly accepted as "narrow-trigger family" — G3 substrate hypotheses must demonstrate predictive lift on at least 3 of 5 families (excluding MR-only signals) since MR has insufficient sample size for statistical validation. (c) G2 prerequisite: leaderboard caching + orderbook strict-mode toggle landed (10k bar smoke runtime 6956s → ~38s, 185× speedup). | 임준영 + advisor + Claude Opus 4.7 |
