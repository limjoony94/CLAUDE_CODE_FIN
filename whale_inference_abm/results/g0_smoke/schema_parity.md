# BingX Phase 1 L2 Collector ↔ ABM Logger Schema Parity

**Date**: 2026-05-01
**Purpose**: G0 acceptance criterion (advisor N3 patch). Document schema diff between
ABM bar_snapshot/trade output and BingX Phase 1 L2 collector output for G4 substrate
detector deployment.
**Source**: `bingx_rl_trading_bot/scripts/data_pipeline/bingx_l2_collector.py`

---

## 1. Depth (Orderbook) Snapshot

### BingX Phase 1 collector schema (parquet rows)
| Field | Type | Notes |
|-------|------|-------|
| `event_ts_ms` | int | UTC milliseconds (`time.time() * 1000`) |
| `symbol` | str | "BTC-USDT" |
| `bid_px_0` ... `bid_px_19` | float | Top-20 bid prices (highest first) |
| `bid_qty_0` ... `bid_qty_19` | float | Quantities at each level |
| `ask_px_0` ... `ask_px_19` | float | Top-20 ask prices (lowest first) |
| `ask_qty_0` ... `ask_qty_19` | float | Quantities at each level |

Subscription: `BTC-USDT@depth20` (top-20 levels)

### ABM `bar_snapshots.ndjson` schema
| Field | Type | Notes |
|-------|------|-------|
| `event_type` | "BAR_SNAPSHOT" | constant |
| `timestamp_ns` | int | sim-internal nanoseconds |
| `best_bid` | float \| null | derived from depth |
| `best_ask` | float \| null | derived from depth |
| `mid_price` | float \| null | (best_bid + best_ask) / 2 |
| `spread` | float \| null | best_ask - best_bid |
| `bid_depth_l10` | list[[price, size]] | top-10 levels (descending price) |
| `ask_depth_l10` | list[[price, size]] | top-10 levels (ascending price) |
| `wealth_dist` | dict[agent_id, wealth] | per-agent MTM (ABM-only) |

### Schema diff

| Category | BingX | ABM | Mismatch | G4 adapter required |
|----------|-------|-----|----------|---------------------|
| Time unit | `event_ts_ms` (ms) | `timestamp_ns` (ns) | YES | `ms = ns / 1_000_000` |
| Depth levels | 20 | 10 | YES (BingX richer) | Truncate BingX[20]→[10] OR expand ABM→20 |
| Depth representation | Flat columns (`bid_px_i`, `bid_qty_i`) | Nested list[[price, size]] | YES | Adapter: row → list of (px, qty) tuples |
| Symbol field | Yes (`symbol="BTC-USDT"`) | No (single-asset ABM) | YES | Adapter adds constant "BTC-USDT" or strips |
| Best bid/ask explicit | No (derived) | Yes (precomputed) | NO (derivable) | None |
| Mid/spread precomputed | No | Yes | NO | Adapter computes |
| Wealth distribution | Absent | Per-agent wealth | YES (ABM-only) | G4 detector can't use wealth_dist on real L2 |

---

## 2. Trade Record

### BingX Phase 1 collector schema
| Field | Type | Notes |
|-------|------|-------|
| `event_ts_ms` | int | UTC milliseconds |
| `symbol` | str | "BTC-USDT" |
| `price` | float | Trade price |
| `qty` | float | Trade quantity |
| `is_buyer_maker` | bool | True = buyer was maker (passive), seller hit buyer's bid |
| `trade_id` | str | BingX exchange trade ID |

### ABM `trade_tape.ndjson` schema
| Field | Type | Notes |
|-------|------|-------|
| `event_type` | "TRADE" | constant |
| `timestamp_ns` | int | sim ns |
| `sequence_no` | int | per-event monotonic |
| `trade_id` | str | "t_{counter}" |
| `buyer_agent_id` | str | ABM agent ID |
| `seller_agent_id` | str | ABM agent ID |
| `buyer_order_id` | str | per-agent order counter |
| `seller_order_id` | str | per-agent order counter |
| `price` | float | Trade price |
| `size` | float | Trade quantity (= BingX `qty`) |
| `buyer_role` | "taker" \| "maker" | |
| `seller_role` | "taker" \| "maker" | |

### Schema diff

| Category | BingX | ABM | Mismatch | G4 adapter required |
|----------|-------|-----|----------|---------------------|
| Time unit | ms | ns | YES | as above |
| Quantity field name | `qty` | `size` | YES (rename) | Adapter |
| Maker/taker | `is_buyer_maker` (bool) | `buyer_role` + `seller_role` (enum) | YES | True → buyer=MAKER, seller=TAKER (hit aggressively); False → swap |
| Trade ID | exchange-assigned | sim counter | NO (cosmetic) | None |
| Agent identity | **ABSENT (anonymized)** | Present | YES | G4 substrate detector MUST NOT depend on agent identity |
| Sequence no | absent | present | NO (ABM-only) | Adapter strips |

---

## 3. Reconciliation Plan (G4 stage)

**Direction**: G4 detector developed against ABM data first, then adapter wraps BingX
data into ABM-compatible structure for evaluation.

**Adapter responsibilities** (`deployment/l2_loader.py`, future Day-G4 code):
1. Read BingX parquet → DataFrame
2. Convert `event_ts_ms` to `timestamp_ns` (× 1_000_000)
3. Reshape flat depth columns into list of (price, qty) tuples — truncate to L10
   for parity with ABM bar_snapshot, OR keep L20 with adapter that subsets per-detector
4. Convert `is_buyer_maker` → role enums:
   ```python
   if is_buyer_maker:
       buyer_role, seller_role = Role.MAKER, Role.TAKER
   else:
       buyer_role, seller_role = Role.TAKER, Role.MAKER
   ```
5. Synthesize `buyer_agent_id = "bingx_anon"`, `seller_agent_id = "bingx_anon"` —
   detector blocked from agent-specific signals
6. Compute `mid_price`, `spread`, `best_bid`, `best_ask` from depth columns

**Critical anti-leakage check** (G4 review):
- Substrate detector function `f(orderbook, trade_tape) → signal` MUST be agent-id-blind
- Detector code review at G4 entry: grep for `agent_id` references in feature code; reject if found
- This is automatic for substrates derived from G3 (only orderbook + trade tape features)
  but worth re-checking when migrating from synthetic to real data

---

## 4. G0 Acceptance Status

- [x] BingX schema documented
- [x] ABM schema documented
- [x] Field-by-field diff table produced
- [x] Reconciliation plan written
- [x] Anti-leakage check identified (G4 stage)
- [ ] G4 adapter code (deferred to G4 — out of G0 scope per design v0.3 Section 13.4)

**G0 acceptance criterion N3 (advisor v0.4 patch)**: SATISFIED. ABM schema is convertible
to BingX schema via documented adapter; G4 unblocked.

---

## 5. Outstanding Issues (deferred to G4)

1. **Aggregated trades vs per-trade in ABM**: BingX `@trade` channel emits aggregated
   trades; ABM emits per-match trades. Adapter must aggregate ABM trades within ms
   window for direct comparison. OR ABM detector accepts both granularities.
2. **L10 vs L20 depth**: ABM bar_snapshot truncates to L10. BingX has L20. G4 detector
   should specify which level depth it requires; if > L10, expand ABM bar_snapshot
   serialization (small change in `Logger.bar_snapshot()` to use `depth=20`).
3. **Wall-clock vs sim-time alignment**: BingX `event_ts_ms` is real wall-clock.
   ABM `timestamp_ns` is sim-internal. For G4 comparison runs, the sim-tagged ABM
   trajectory and the wall-clock-tagged BingX recording exist on different time axes.
   Adapter MUST not attempt to align timestamps directly — use elapsed-time-from-start
   or normalized-time relative ordering instead.
