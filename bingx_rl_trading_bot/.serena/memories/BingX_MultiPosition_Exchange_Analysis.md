# BingX BTC 5m Pattern Bot: Exchange API & Multi-Position Architecture Analysis

## Executive Summary

**Critical Finding**: BingX in **one-way mode** (`positionSide: 'BOTH'`) enforces **ONE active position per symbol**. The bot architecture is fundamentally designed around single-position trading with strict enforcement at the exchange layer.

### Key Constraints
- **Mode**: One-Way (`positionSide: 'BOTH'`) — NOT hedge mode
- **Multiple Positions**: NOT supported simultaneously in one-way mode
- **Direction**: Can hold LONG **XOR** SHORT, never both at same time
- **Enforcement**: Bot actively checks for existing positions before opening new ones

---

## 1. Signal Generation (signals.py)

### Entry Signal Detection
```python
def check_entry_signal(df, state, config) -> Tuple[Optional[str], Optional[str]]:
    # Returns (signal, reason) where signal ∈ {'LONG', 'SHORT', None}
```

**Key Properties**:
- Returns single direction per call
- Direction-agnostic — doesn't check current position state
- Pattern matching via if/elif (LONG takes priority, SHORT is fallback)
- 59 validated patterns: 12 LONG + 47 SHORT
- Confidence scoring independent of position state

**Important**: Signal generation is **multi-position capable**. It can generate LONG or SHORT without knowing if a position already exists.

### Pattern Validation
- 3-candle pattern matching (e.g., 'BD-BD-U')
- Configurable long_patterns and short_patterns lists
- Classification confidence scoring based on:
  - Clarity of candle formation (40%)
  - Historical win rate (30%)
  - Regime alignment (30%)

---

## 2. Exchange Layer (exchange.py)

### BingX Configuration
```python
exchange = ccxt.bingx({
    'apiKey': api_key,
    'secret': secret_key,
    'options': {
        'defaultType': 'swap',           # Futures trading
        'positionSide': 'BOTH',          # ONE-WAY MODE
        'recvWindow': 60000
    }
})
```

### Position Fetching
```python
def fetch_positions_cached(exchange, symbol, cache, force_refresh=False):
    # Returns: List[Dict] of positions
    # Each position has: side ('long'/'short'), contracts (qty), entryPrice, etc.
```

**API Call**:
```python
positions = exchange.fetch_positions([symbol])
```

**BingX Response Structure**:
```python
[
    {
        'side': 'long',          # Only if > 0 contracts
        'contracts': 123.45,     # Absolute quantity
        'entryPrice': 45123.50,
        'markPrice': 45125.00,
        'unrealizedPnl': 50.00,
        # ... other fields
    },
    # SHORT position would appear as separate entry if exists
]
```

### Position Mode Enforcement
```python
def verify_position_mode(exchange, config):
    positions = exchange.fetch_positions([symbol])
    
    # CRITICAL CHECK
    has_long = any(p.get('side') == 'long' and float(p.get('contracts', 0)) > 0 for p in positions)
    has_short = any(p.get('side') == 'short' and float(p.get('contracts', 0)) > 0 for p in positions)
    
    if has_long and has_short:
        logger.error("🔴 CRITICAL: Both LONG and SHORT positions exist!")
        return False  # ARCHITECTURE VIOLATION
```

**This check enforces**: BingX one-way mode can't have both LONG and SHORT simultaneously. If it does, it's a critical error.

### Order Placement
```python
order = exchange.create_market_order(
    symbol=symbol,
    side='buy',  # or 'sell'
    amount=quantity,
    params={'positionSide': 'BOTH'}  # ONE-WAY mode
)
```

**BOTH parameter meaning**:
- NOT "both directions supported"
- Rather: "I'm using one-way mode (implied: no hedge mode)"
- Tells BingX to use ONE position per symbol, not separate LONG/SHORT positions

### Leverage Setting
```python
exchange.set_leverage(leverage, symbol, params={'side': 'BOTH'})
```

**BOTH here**: Same meaning — one-way mode. Leverage applies to whichever direction is open.

---

## 3. Position Management (position_open.py)

### The One-Position Constraint
```python
def _verify_no_existing_position(exchange, state, config, ...):
    positions = fetch_positions_cached(exchange, config['symbol'], ...)
    for pos in positions:
        if abs(float(pos.get('contracts', 0))) > 0:
            logger.warning("Position already exists on exchange")
            sync_position_with_exchange(...)
            return False  # BLOCKS new position opening
    return True  # Safe to open new position
```

**This function is called at line 115 of `open_position()`** — EVERY single time the bot tries to open a position. If ANY position (LONG or SHORT) exists, it returns False and blocks the open.

### Position Extraction Logic
```python
# Called after receiving order, ensures correct entry price
positions = fetch_positions_cached(exchange, symbol, ...)
pos_side = 'long' if signal == 'LONG' else 'short'
for pos in positions:
    if pos.get('side') == pos_side and float(pos.get('contracts', 0)) > 0:
        actual_entry_price = float(pos.get('entryPrice', ...))
        actual_quantity = float(pos.get('contracts', ...))
        break
```

**Key insight**: Code assumes at most ONE long and ONE short position can exist. Would need restructuring for multiple LONG or multiple SHORT positions.

---

## 4. Position Monitoring (position_monitor.py)

### Synchronization Logic
```python
def sync_position_with_exchange(exchange, state, config, ...):
    positions = fetch_positions_cached(...)
    
    # Extract at most one long, one short
    exchange_long = None
    exchange_short = None
    for pos in positions:
        if pos.get('side') == 'long' and float(pos.get('contracts', 0)) > 0:
            exchange_long = pos
        elif pos.get('side') == 'short' and float(pos.get('contracts', 0)) > 0:
            exchange_short = pos
    
    state_position = state.get('position')
    
    # Case 1: State has position, exchange doesn't
    if state_position and not exchange_long and not exchange_short:
        # Position was closed externally
        record_closed_position(...)
        return True
    
    # Case 2: Exchange has position, state doesn't
    if not state_position:
        if exchange_long:
            recover_position_to_state(state, ..., exchange_long, 'LONG', ...)
        elif exchange_short:
            recover_position_to_state(state, ..., exchange_short, 'SHORT', ...)
    
    # Case 3: Direction mismatch
    if state_position:
        state_dir = state_position.get('direction')
        if state_dir == 'LONG' and not exchange_long and exchange_short:
            # Serious: state says LONG but exchange has SHORT
            record_closed_position(state, ...)  # Close state entry
            recover_position_to_state(state, ..., exchange_short, 'SHORT', ...)  # Recover SHORT
```

**State Structure**:
```python
state['position'] = {
    'direction': 'LONG' or 'SHORT',
    'entry_price': float,
    'quantity': float,
    'entry_time': datetime_iso,
    'reason': str,
    'tp_order_id': str or None,
    'sl_order_id': str or None,
    # ... other fields
}
```

**Critical property**: `state['position']` is either None or a single dict. Never a list of dicts. The entire bot architecture assumes ONE position at a time.

---

## 5. BingX API Capabilities

### One-Way Mode (Current: `positionSide: 'BOTH'`)
| Feature | Support |
|---------|---------|
| Single symbol, one direction at a time | ✅ YES |
| LONG and SHORT simultaneously | ❌ NO |
| Switch direction (close LONG, open SHORT) | ✅ YES |
| Shared margin pool | ✅ YES |
| Leverage per symbol | ✅ YES |
| Multiple orders per position (TP + SL) | ✅ YES |

### Hedge Mode (NOT used: would use `positionSide: 'LONG'/'SHORT'`)
| Feature | Support |
|---------|---------|
| Single symbol, one direction at a time | ✅ YES |
| LONG and SHORT simultaneously | ✅ YES |
| Independent TP/SL per direction | ✅ YES |
| Separate margin per position | ✅ YES |
| Leverage per position | ✅ YES |

**Current bot**: Explicitly configured for one-way mode. Switching to hedge would require:
1. Parameter change from `BOTH` to `LONG`/`SHORT`
2. State restructuring (single position → position list)
3. Sync logic rewrite (3 states → 7+ states)
4. Independent order management per position

---

## 6. Multi-Position Feasibility Assessment

### What Would Need to Change

#### FEASIBLE (Minimal changes)
1. **Signal generation** (signals.py)
   - ✓ Already direction-agnostic
   - ✓ No state dependency
   - ✓ Can return LONG or SHORT independently

2. **Exchange setup** (exchange.py)
   - ✓ CCXT supports hedge mode natively
   - ✓ Just change BOTH → LONG/SHORT in order params
   - ✓ Position fetching already handles multiple sides

#### COMPLEX (Major refactoring)
1. **Position opening** (position_open.py)
   - ❌ `_verify_no_existing_position()` blocks if any position exists
   - Needs: Filter check by direction only (not absolute)
   - Needs: Allow LONG+LONG, SHORT+SHORT, or LONG+SHORT if hedge mode

2. **Position state** (state.py)
   - ❌ `state['position']` is single dict
   - Needs: `state['positions']` = List[dict] with ID tracking
   - Needs: Lookup by direction or position ID
   - Ripple effect: All code accessing state['position'] breaks

3. **Position monitoring** (position_monitor.py)
   - ❌ Sync logic assumes ≤1 active per direction
   - Current: 3 branches (no pos, state pos, exchange pos)
   - Needed: 7+ branches (2³ combinations per direction, then combined)
   - Complex: Handle partial sync (1 position matches, 1 doesn't)

4. **Order management** (orders.py)
   - ❌ TP/SL assumes one pair per position
   - Needs: Independent TP/SL order lists per position
   - Needs: Order ID mapping to position ID
   - Complex: Scale-out with partial fills

5. **State persistence** (models.py)
   - ❌ Metrics tracking assumes single active position
   - Needs: Aggregate metrics across multiple positions
   - Needs: Individual PnL tracking per position

6. **Configuration** (pattern_5m_config.yaml)
   - ✓ Add hedge_mode: true flag
   - ✓ Add max_positions_per_direction: 1 or 2
   - Simple changes

### Refactoring Scope Summary
| Layer | Current | Required Change | Effort |
|-------|---------|-----------------|--------|
| Signal Gen | 1 function | None | 0% |
| Exchange | Mode setup | Update mode flag | 5% |
| Position Open | Single check | Direction-filtered check | 15% |
| Position State | Single dict | List of dicts | 20% |
| Position Monitor | 3-way sync | 7+ branches | 35% |
| Orders | TP/SL pair | Multiple pairs per position | 15% |
| Metrics | Single position | Aggregate + per-position | 10% |
| **Total** | - | - | **100% → ~3-4 weeks** |

---

## 7. Current Architecture Diagram

```
Daily Loop
  ├─ Fetch OHLCV → classify candles
  ├─ Check entry signal (signals.py)
  │   └─ Returns: 'LONG' or 'SHORT' or None
  │
  ├─ If signal:
  │   ├─ Verify no existing position (exchange.py)
  │   │   └─ fetch_positions_cached()
  │   │   └─ Check: ANY position with contracts > 0?
  │   │   └─ If yes: Block new position
  │   │   └─ If no: Proceed
  │   │
  │   ├─ Open position (position_open.py)
  │   │   ├─ Create market order (side = buy/sell)
  │   │   ├─ params: {'positionSide': 'BOTH'}  ← One-way mode
  │   │   └─ Update state['position'] = {...}
  │   │
  │   └─ Place TP/SL orders (orders.py)
  │       ├─ TP order (take profit)
  │       └─ SL order (stop loss)
  │
  └─ Monitor position
      ├─ Sync with exchange (position_monitor.py)
      │   ├─ Fetch exchange positions
      │   ├─ Compare with state['position']
      │   └─ Handle mismatches (3 cases)
      │
      └─ If position closed:
          ├─ Record closed position (position_close.py)
          └─ Reset state['position'] = None
```

---

## 8. Key Findings Summary

### Signal Generation
- ✓ **Already multi-capable**: Can return LONG or SHORT independently
- ✓ **No state coupling**: Doesn't check current position
- ✓ **No refactor needed**: Works as-is for multi-position

### Exchange Layer
- ✓ **Hedge mode available**: BingX CCXT supports it natively
- ✓ **API capability**: Can fetch multiple positions per symbol
- ⚠️ **Config change needed**: Switch from 'BOTH' to 'LONG'/'SHORT' params

### Position Management
- ❌ **Single position only**: Architecture enforced at every level
- ❌ **State is scalar**: `position` is dict, not list
- ❌ **Sync is limited**: 3-way logic, not 7+ branches
- ❌ **Orders not scalable**: One TP + one SL assumption

### Practical Implications
1. **Current design is stable**: Single-position constraint prevents bugs
2. **Multi-position is possible**: Requires systematic refactoring
3. **Minimum viable approach**: Same-direction scale-out (LONG+LONG, not LONG+SHORT)
4. **Not urgent**: Current bot performing well (73.9% OOS WR, +949% IS PnL)

---

## 9. Recommendations

### If exploring multi-position:
1. **Phase 0**: Backtest LONG+SHORT simultaneously. Is WR better or worse?
2. **Phase 1**: Implement hedge mode setup + position list state
3. **Phase 2**: Rewrite sync logic for 7+ branches
4. **Phase 3**: Extensive crash recovery testing

### Current recommendation:
- **Keep one-position-only design** — it's proven stable
- **Signal generation is already multi-capable** — reuse as-is if refactoring
- **Future evolution**: Hedge mode is 3-4 week project, not immediate need
