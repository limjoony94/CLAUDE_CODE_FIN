# RSI Zone Bot v1.3.1 개선사항

**날짜**: 2025-12-11
**버전**: v1.3.1 (Fine-Tuned + Auto-Backup)

---

## 1. 주요 개선사항 요약

### v1.3 → v1.3.1 변경사항

| 기능 | 설명 |
|------|------|
| **YAML 설정 분리** | 모든 파라미터를 `config/rsi_zone_bot_config.yaml`로 분리 |
| **State 자동 백업** | 거래 발생 시 + 주기적 백업 (30분마다) |
| **거래소 동기화 강화** | Deep Sync (5분마다) + Quick Sync (매 루프) |
| **백업 복구 기능** | State 손상 시 자동 백업에서 복구 |

---

## 2. YAML 설정 파일 구조

**파일 위치**: `config/rsi_zone_bot_config.yaml`

### 설정 섹션

```yaml
# 전략 파라미터
strategy:
  rsi_period: 14
  rsi_oversold_zone: 35
  rsi_recovery_threshold: 40
  rsi_overbought_zone: 65
  rsi_decline_threshold: 60
  ema_trend: 200

# 청산 파라미터
exit:
  take_profit_pct: 2.4
  stop_loss_pct: 1.4
  cooldown_candles: 4
  max_hold_candles: 9999

# 본절손절 설정
breakeven:
  enabled: true
  trigger_pct: 1.2
  buffer_pct: 0.15

# 레버리지 설정
leverage:
  exchange_leverage: 10
  effective_leverage: 4
  position_size_pct: 0.95

# 백업 설정
backup:
  enabled: true
  max_backups: 10
  backup_on_trade: true
  backup_interval_minutes: 30

# 동기화 설정
sync:
  enabled: true
  deep_sync_interval_minutes: 5
  quick_sync_on_loop: true
  verify_orders_on_sync: true
  auto_recreate_orders: true
```

---

## 3. Claude Code 활용 가이드

### 파라미터 변경 요청 예시

```
# TP 변경
"TP를 2.5%로 변경해줘"
→ config/rsi_zone_bot_config.yaml에서 exit.take_profit_pct 수정

# SL 변경
"손절을 1.5%로 변경해줘"
→ config/rsi_zone_bot_config.yaml에서 exit.stop_loss_pct 수정

# 레버리지 변경
"레버리지를 5배로 변경해줘"
→ config/rsi_zone_bot_config.yaml에서 leverage.effective_leverage 수정

# 쿨다운 변경
"쿨다운을 2시간으로 변경해줘"
→ config/rsi_zone_bot_config.yaml에서 exit.cooldown_candles: 8 (15m x 8 = 2h)

# 본절손절 비활성화
"본절손절 기능을 끄고 싶어"
→ config/rsi_zone_bot_config.yaml에서 breakeven.enabled: false
```

### 상태 확인 요청 예시

```
# 현재 상태 확인
"봇 상태 확인해줘"
→ results/rsi_zone_bot_state.json 읽기

# 거래 내역 확인
"최근 거래 내역 보여줘"
→ state.json의 trading_history 확인

# 백업 목록 확인
"백업 파일 목록 보여줘"
→ results/backups/ 디렉토리 확인
```

### 백테스트/분석 요청 예시

```
# 파라미터 최적화
"TP 2.0~3.0% 범위에서 최적값 찾아줘"
→ scripts/analysis/rsi_zone_fine_tuning.py 참고하여 그리드 서치

# Walk-Forward 검증
"이 파라미터로 Walk-Forward 검증해줘"
→ scripts/analysis/rsi_zone_walkforward_validation.py 실행
```

---

## 4. State 자동 백업 시스템

### 백업 트리거

| 트리거 | 파일명 패턴 | 설명 |
|--------|------------|------|
| startup | `rsi_zone_bot_state_YYYYMMDD_HHMMSS_startup.json` | 봇 시작 시 |
| trade | `rsi_zone_bot_state_YYYYMMDD_HHMMSS_trade.json` | 포지션 진입/청산 시 |
| periodic | `rsi_zone_bot_state_YYYYMMDD_HHMMSS_periodic.json` | 30분마다 |

### 백업 위치

```
results/backups/
├── rsi_zone_bot_state_20251211_123456_startup.json
├── rsi_zone_bot_state_20251211_134500_trade.json
├── rsi_zone_bot_state_20251211_140000_periodic.json
└── ...
```

### 백업 복구

State 파일이 손상되면 자동으로 가장 최근 백업에서 복구를 시도합니다.

```python
# 수동 복구 (필요 시)
from scripts.production.rsi_zone_bot import recover_from_backup, list_backups

# 백업 목록 확인
backups = list_backups()
for b in backups:
    print(f"{b['filename']} - {b['reason']} - PnL: {b['total_pnl']:.2f}%")

# 복구
state = recover_from_backup()
```

---

## 5. 거래소 동기화 로직

### Quick Sync (매 루프)

- 포지션 존재 여부 확인
- State와 거래소 포지션 불일치 감지
- TP/SL 주문 존재 확인

### Deep Sync (5분마다)

- 전체 포지션 검증
- 수량/진입가 불일치 수정
- TP/SL 주문 자동 재생성
- 고아 주문(orphan orders) 정리
- 잔고 업데이트

### 동기화 로그 예시

```
2025-12-11 14:30:00 🔄 Running Deep Sync with Exchange...
2025-12-11 14:30:00    Balance: Equity=$400.63, Available=$400.63
2025-12-11 14:30:00    Open Orders: 2
2025-12-11 14:30:00 🔄 Deep Sync completed
```

---

## 6. 파일 구조

```
bingx_rl_trading_bot/
├── config/
│   └── rsi_zone_bot_config.yaml     ← 설정 파일 (NEW)
│
├── scripts/
│   └── production/
│       └── rsi_zone_bot.py          ← v1.3.1 (Updated)
│
├── results/
│   ├── rsi_zone_bot_state.json      ← 현재 상태
│   └── backups/                     ← 백업 디렉토리 (NEW)
│       └── rsi_zone_bot_state_*.json
│
└── claudedocs/
    └── RSI_ZONE_BOT_V131_IMPROVEMENTS.md  ← 이 문서
```

---

## 7. 주의사항

### 설정 변경 후

1. YAML 파일 수정 후 봇을 **재시작**해야 적용됩니다.
2. 봇 실행 중 YAML 파일을 수정해도 즉시 반영되지 않습니다.

### 백업 관련

1. 최대 10개의 백업만 유지됩니다 (설정 가능).
2. 오래된 백업은 자동으로 삭제됩니다.
3. 중요한 백업은 별도로 보관하세요.

### 동기화 관련

1. Deep Sync는 API 호출이 많으므로 간격을 너무 짧게 설정하지 마세요.
2. `sync.enabled: false`로 설정하면 동기화가 비활성화됩니다.

---

## 8. 변경 이력

| 버전 | 날짜 | 변경사항 |
|------|------|---------|
| v1.3.1 | 2025-12-11 | YAML 설정, 자동 백업, Deep Sync 추가 |
| v1.3 | 2025-12-11 | TP 2.4%, SL 1.4%, BE_SL 1.2% (891조합 최적화) |
| v1.2 | 2025-12-10 | TP 2.5%, SL 1.5%, BE_SL 1.5% |
| v1.1 | 2025-12-10 | TP 2.0%, SL 1.5%, BE_SL 1.0% |
