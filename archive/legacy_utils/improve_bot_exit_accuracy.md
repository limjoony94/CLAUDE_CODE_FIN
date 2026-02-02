# Bot Exit Accuracy Improvement Plan

## 현재 문제

봇이 포지션 청산 시 예상 체결가로 P&L을 계산하지만, **실제 거래소 체결 결과는 슬리피지로 인해 다를 수 있음**.

### 예시 (실제 발견된 케이스)
```yaml
Bot 계산 (부정확):
  Entry Price: $110,357.70
  Exit Price: $110,168.20 (예상)
  Net P&L: $0.87

거래소 실제 (ground truth):
  Entry Price: $110,307.69 (슬리피지)
  Exit Price: $110,153.85 (슬리피지)
  Entry Fee: $0.36 (봇이 0으로 기록)
  Net P&L: $0.14

차이: 6배 과대평가
```

## 해결 방안

### Step 1: 청산 후 거래소 실제 체결 결과 조회

봇 코드의 청산 로직 개선:
```python
# 현재 (opportunity_gating_bot_4x.py Line 2517-2609)
close_result = client.close_position(...)
# 0.5초 대기 후 fetch_my_trades로 조회 (불안정)
# → 체결 정보를 찾지 못하면 fallback

# 개선안
close_result = client.close_position(...)
close_order_id = close_result.get('id')

# 2초 대기 (거래소가 Position History에 기록할 시간)
time.sleep(2.0)

# Position History API로 실제 체결 결과 조회 ✅
close_details = client.get_position_close_details(
    position_id=position.get('position_id_exchange'),
    symbol=SYMBOL
)

if close_details:
    # 거래소 ground truth 사용
    actual_exit_price = close_details['exit_price']
    actual_pnl_usd = close_details['realized_pnl']
    actual_net_pnl = close_details['net_profit']
    actual_exit_time = close_details['close_time']
else:
    # Fallback: 기존 로직
    # (하지만 이 경우는 거의 없어야 함)
```

### Step 2: Position ID 추적 개선

현재 봇이 `position_id_exchange`를 저장하는지 확인 필요:
- 진입 시: `enter_position_with_protection`이 position_id 반환
- State에 저장: `position['position_id_exchange']`

확인 사항:
- [ ] 진입 시 position_id가 state에 저장되는가?
- [ ] 청산 시 position_id를 사용해 조회하는가?

### Step 3: 수수료 정확성 개선

거래소 API 응답:
- `realized_pnl`: Gross P&L (before fees)
- `net_profit`: Net P&L (after fees)
- `commission`: 실제 수수료

State file 업데이트:
```python
position.update({
    'exit_price': close_details['exit_price'],
    'pnl_usd': close_details['realized_pnl'],  # Gross
    'pnl_usd_net': close_details['net_profit'],  # Net
    'exit_fee': close_details['realized_pnl'] - close_details['net_profit'],  # 역산
    'total_fee': entry_fee + exit_fee
})
```

## 구현 계획

### Phase 1: 현재 봇 코드 개선 (Lines 2517-2665)
1. `fetch_my_trades` 대신 `get_position_close_details` 사용
2. 대기 시간 0.5초 → 2.0초 증가
3. Position ID 기반 조회로 변경
4. Fallback 로직 유지 (안전성)

### Phase 2: 테스트
1. 다음 청산 시 실제 체결 결과 조회 확인
2. State file의 P&L이 거래소 실제 기록과 일치하는지 검증
3. 로그에 비교 정보 출력

### Phase 3: 자동 Reconciliation
1. 청산 10초 후 자동으로 `reconcile_from_exchange.py` 실행
2. 이중 검증: 봇 계산 vs 거래소 ground truth
3. 차이가 있으면 경고 로그

## 기대 효과

1. **정확한 P&L 추적**: 슬리피지 완전 반영
2. **정확한 수수료**: Entry + Exit 수수료 모두 거래소 실제 값
3. **신뢰할 수 있는 성과 측정**: Backtest vs Production 비교 가능
4. **자동 검증**: Reconciliation으로 이중 확인

## 구현 우선순위

🔴 **High Priority**: Position ID 기반 조회 로직 추가
🟡 **Medium Priority**: 대기 시간 증가 및 fallback 개선
🟢 **Low Priority**: 자동 reconciliation (이미 수동 스크립트 있음)
