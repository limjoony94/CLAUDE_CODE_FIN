# Root Cause Analysis: Missing Closed Position in QUANT_MONITOR

**Date**: 2025-10-24 01:47 KST
**Issue**: Position closed at 00:35:10 not showing in monitor

---

## Problem Summary

**Missing Position:**
- Entry: 2025-10-23 18:15:08
- Exit: 2025-10-24 00:35:10
- P&L: +$51.60
- Status: **Not recorded in state file**

---

## Root Cause

### 1. Order ID vs Position ID Mismatch

**BingX API Behavior:**
- Order 생성 → `order_id` 반환
- Position 생성 → `position_id` 할당 (**별도 값**)

**Example:**
```
Order ID:    1981288264506753024  (봇이 저장)
Position ID: 1981288265204482048  (Exchange가 생성)
```

### 2. Current Bot Implementation

**Entry (`enter_position_with_protection`):**
```python
return {
    'entry_order': entry_order,  # Contains order_id
    # ❌ NO position_id!
}
```

**State File:**
```json
{
  "order_id": "1981288264506753024",  # Order ID only
  // ❌ Missing: "position_id_exchange"
}
```

### 3. Reconciliation Failure

**Bot Restart Process:**
1. Find stale OPEN trade in state
2. Call `get_position_close_details(order_id=...)`
3. Search Position History for matching `order_id`

**`get_position_close_details` (Line 366):**
```python
pos_id = str(pos.get('id', ''))  # Position ID from API
if pos_id == str(order_id):      # ❌ Never matches!
```

**Position History API Response:**
```json
{
  "positionId": "1981288265204482048",  // ✅ Has Position ID
  "avgPrice": 109206.8,
  // ❌ NO Order ID field!
}
```

**Result:** No match found → "Position not found" → Removed as stale

---

## Data Flow Diagram

```
Entry:
  create_order() → order_id: 1981288264506753024
                → (BingX creates position internally)
                → position_id: 1981288265204482048 ❌ NOT CAPTURED

State:
  trades: [{
    "order_id": "1981288264506753024"  ✅ Saved
    // "position_id_exchange": ???     ❌ Missing
  }]

Bot Restart:
  1. Find OPEN trade with order_id: 1981288264506753024
  2. get_position_close_details(order_id="1981288264506753024")
  3. Position History returns:
     [{
       "positionId": "1981288265204482048",  ← Different ID!
       // No "orderId" field
     }]
  4. Match fails (1981288264506753024 ≠ 1981288265204482048)
  5. Return None → "Position not found"
  6. Remove as stale ❌
```

---

## API Analysis

### Position History API (fetchPositionHistory)
**Available Fields:**
```json
{
  "positionId": "...",       // ✅ Position ID
  "avgPrice": 109206.8,
  "avgClosePrice": 109953.4,
  "realisedProfit": 60.5,
  "netProfit": 51.6,
  // ❌ NO Order ID fields:
  // ❌ NO "orderId"
  // ❌ NO "openOrderId"
  // ❌ NO "closeOrderId"
}
```

**Conclusion:** Cannot match by Order ID!

### Open Positions API (fetch_positions)
**Available Fields:**
```json
{
  "id": "1981391450257178624",
  "positionId": "1981391450257178624",  // ✅ Has Position ID!
  "symbol": "BTC-USDT",
  "positionAmt": 0.0846,
  "avgPrice": 109942.8
}
```

**Conclusion:** Position ID is available after entry!

---

## Solution

### 1. Capture Position ID on Entry

**Modify `enter_position_with_protection`:**
```python
def enter_position_with_protection(self, ...):
    # ... create entry order ...

    # ✅ NEW: Fetch position to get position_id
    import time
    time.sleep(1)  # Wait for position to be created

    positions = self.exchange.fetch_positions([ccxt_symbol])
    position_id = None

    for pos in positions:
        if abs(float(pos.get('contracts', 0))) == quantity:
            position_id = pos.get('id')
            break

    return {
        'entry_order': entry_order,
        'stop_loss_order': stop_loss_order,
        'stop_loss_price': stop_loss_price,
        'price_sl_pct': price_sl_pct,
        'position_id': position_id  # ✅ NEW
    }
```

### 2. Save Position ID in State

**Production Bot:**
```python
# After entry
result = client.enter_position_with_protection(...)

new_trade = {
    "order_id": result['entry_order']['id'],
    "position_id_exchange": result.get('position_id'),  # ✅ NEW
    ...
}
```

### 3. Fix Reconciliation Logic

**Option A: Use Position ID**
```python
def get_position_close_details(self, position_id: str, symbol: str):
    # ... fetch history ...

    for pos in history:
        pos_id = str(pos.get('id', ''))
        if pos_id == str(position_id):  # ✅ Correct match
            return {...}
```

**Option B: Match by Entry Price + Quantity + Time**
```python
def get_position_close_details(self, entry_price: float, quantity: float,
                               entry_time: datetime, symbol: str):
    # ... fetch history ...

    for pos in history:
        # Match by characteristics
        if (abs(pos['avgPrice'] - entry_price) < 1.0 and
            abs(pos['quantity'] - quantity) < 0.0001 and
            abs(pos['timestamp'] - entry_time_ms) < 60000):  # 1min window
            return {...}
```

---

## Implementation Plan

### Phase 1: Quick Fix (Immediate)
1. ✅ Manual recovery: Add missing trade to state file
2. ⏳ Monitor for next occurrence

### Phase 2: Permanent Fix (This Session)
1. Modify `enter_position_with_protection` to capture position_id
2. Update production bot to save position_id_exchange
3. Update `get_position_close_details` to use position_id
4. Update reconciliation logic

### Phase 3: Testing
1. Test entry → verify position_id saved
2. Test bot restart → verify reconciliation works
3. Test closed position → verify monitoring update

---

## Impact Assessment

**Current Impact:**
- **Frequency**: Happens on every bot restart with open positions
- **Severity**: High (missing trade records)
- **Data Loss**: Closed positions not recorded

**After Fix:**
- ✅ All positions tracked correctly
- ✅ Reconciliation works reliably
- ✅ QUANT_MONITOR shows accurate data

---

## Related Files

**Need Modification:**
1. `src/api/bingx_client.py` (enter_position_with_protection)
2. `src/api/bingx_client.py` (get_position_close_details)
3. `scripts/production/opportunity_gating_bot_4x.py` (entry logic)

**Testing:**
1. Manual test: Entry → check position_id
2. Integration test: Bot restart → reconciliation
3. End-to-end test: Full cycle with monitoring

---

## Lessons Learned

1. **Always capture all IDs** from exchange APIs
2. **Verify data availability** before implementing matching logic
3. **Test reconciliation logic** with real exchange data
4. **Document ID relationships** (Order ID ≠ Position ID)
5. **Manual trades** already have position_id_exchange (from reconciliation)

---

**Status**: Root cause identified, solution designed, ready for implementation
