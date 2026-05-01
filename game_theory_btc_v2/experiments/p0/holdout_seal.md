# Sealed OOS Holdout Commit — v2 Mandate § 0.2

**Seal date**: 2026-05-01
**Authority**: Pre-committed before any feature exploration. **CANNOT be modified post-execution.**

---

## Seal Boundary

### Primary boundary
- **Sealed window**: last 25% of fetched data (per mandate § 0.2)
- **For 720d backtest** → last 180d sealed
- **Anchor**: P0.2 fetch 완료 시점의 `max(timestamp)`을 anchor T로 정의
- **Sealed range**: `[T - 180d, T]` UTC (inclusive of T, exclusive of T - 180d for boundary clarity)

### Concrete computation (Day-of-fetch에 산정)
P0.2 완료 후 `experiments/p0/fetch_summary.md`에 다음 명시:
- `T_anchor` (UTC, ms): exact anchor timestamp from fetch
- `T_seal_start = T_anchor - 180 * 86400 * 1000` (UTC, ms)
- `seal_window`: human-readable (예: `2025-11-03T13:15:30Z ~ 2026-05-01T13:15:30Z`)

기록 후 본 문서에 amendment 추가 (boundary 자체는 변경 불가, 단지 concrete value 기록).

### Secondary boundary (P3 MAP-Elites용)
- 본 mandate § 0.2는 last 25%만 명시. P3에서 추가 walk-forward 5-fold + holdout 10%이 별도로 sealed.
- **Sealed window for P3 final eval**: 추가 last 10% of 720d = last 72d.
- 즉, 540d (74%) 자유 fitting + 108d (P3 WF/regime split) + 72d (sealed final eval).

---

## Sealed Operations Rule

P0.5 hypothesis 정량 정의 → P5 force-flow 정밀화까지 **다음 활동 전면 금지**:

1. ❌ Sealed window 데이터로 plot 그리기
2. ❌ Sealed window 통계 (mean, std, distribution) 계산
3. ❌ Sealed window 위에 backtest run
4. ❌ Sealed window 위에 hyperparameter search
5. ❌ Sealed window event ("Trump tariff cascade", "FTX collapse" 등) 시각 확인 후 strategy adjust
6. ❌ Sealed window를 통과한 데이터로 visualization 작성 (예: full-period chart)

**허용 활동 (메타 정보만)**:
- ✅ Sealed window의 row count, NaN count, gap rate 확인 (data quality audit)
- ✅ Sealed window 첫/마지막 timestamp 표시
- ✅ Sealed window file size

---

## Final Evaluation Window (P6 only)

P6 (LIVE-readiness) 단계에서 단 1회 sealed window 위에서 portfolio 평가:
- 6-criteria gate
- WF + regime stability check
- Strategy 재조정 절대 금지 — 평가 결과만 보고

평가 후 부정 결과 → strategy revision 없이 honest closure. P0-P5 단계로 돌아가는 것은 허용되나, 그 경우 **새 sealed window** 정의 필요 (overlapping reuse 금지).

---

## Anti-Fishing Locks

1. **No "just glance"**: P0.5에서 H1-H9 feature distribution 시각화 시 sealed window는 explicitly excluded.
2. **No reverse-engineering**: P5 force-flow 정밀화에서 known cascade event (예: 2025-10-10, 2025-11-19) 시점이 sealed window에 포함되었다면, 그 event는 검증 불가 (그 event 모르는 척하는 게 아니라 정량 검증을 그 event 기반으로 안 하는 것).
3. **No silent extension**: sealed window 부족하다 판단 시 extend 금지. P0-P5 결과만으로 P6 진입 또는 closure.

---

## Pre-commit Signatures

- **Pre-committed by**: Claude Code agent
- **On behalf of**: 임준영 (project owner)
- **Date**: 2026-05-01
- **Mandate version**: v2
- **Subject to**: Anti-fishing charter § 0.2 (mandate)

본 seal은 mandate § 0.7 honest closure의 일부. 위반 시 자동 P0-P6 closure + new seal로 재진입.

---

## Amendment Section — P0.2 Closure Concrete Values (2026-05-01)

**Locked at**: 2026-05-01T14:00:00+00:00 UTC (P0.2 fetch closure)
**Mutability**: IMMUTABLE post-this-section. 변경 시 새 seal 무효, 모든 P0-P5 작업 reset 필요.

```
T_anchor (UTC ms):           1777680000000
T_anchor_iso:                2026-05-01T14:00:00+00:00
T_seal_start (UTC ms):       1762128000000
T_seal_start_iso:            2025-11-02T14:00:00+00:00
T_p3_holdout_start (UTC ms): 1771336800000
T_p3_holdout_start_iso:      2026-02-18T14:00:00+00:00
sealed_window_days:          180  (last 25% of 720d primary)
p3_extra_holdout_days:       72   (additional 10% for P3 final eval)
```

### Sealed File Slices (rows where t_close_ms >= T_seal_start_ms)

| File | Total rows | Sealed rows | Sealed % | Free for P0.5-P5 |
|------|-----------|-------------|----------|------------------|
| `btc_perp_1d_1500d.parquet` | 1,500 | 181 | 12.07% | 1,319 |
| `btc_perp_1h_720d.parquet` | 17,280 | 4,321 | 25.01% | 12,959 |
| `btc_perp_5m_720d.parquet` | 207,360 | 51,840 | 25.00% | 155,520 |
| `btc_perp_1m_365d.parquet` | 525,600 | 259,199 | 49.31% | 266,401 |
| `btc_spot_1h_720d.parquet` | 17,280 | 4,321 | 25.01% | 12,959 |

### Forward Collector Files (NOT sealed — accumulating)

| File | Status |
|------|--------|
| `oi_forward.parquet` | 28d snapshot, accumulating forward |
| `ls_account_forward.parquet` | 20.8d, accumulating |
| `ls_position_forward.parquet` | 20.8d, accumulating |
| `ls_global_forward.parquet` | 20.8d, accumulating |
| `taker_volume_forward.parquet` | 20.8d, accumulating |

Phase B 활성화는 forward collector 데이터가 **60d 누적된 후**. 이 시점에 별도 sealed boundary 정의 (Phase B holdout amendment 003).

### Funding Cross-Check Files

| File | Note |
|------|------|
| `btc_funding_binance_720d.parquet` | Primary, sealed last 180d 적용 (8h grid → ~540 records sealed) |
| `btc_funding_bybit_620d.parquet` | Cross-check 354d only (Bybit pagination cap), sealed last 180d 적용 (~540 records sealed) |

### Multi-Asset File

`multi_asset_1d_800d.parquet` — 5 coins × 800 rows = 4,000 rows total. Per-symbol sealed last 180d:
- 각 symbol 별 ~180/800 = 22.5% sealed
- ETH/SOL/BNB/XRP/DOGE 모두 동일 boundary

---

## Operational Rules from Now (P0.5 진입 시점부터)

1. **모든 코드/notebook**은 sealed boundary filter 명시 적용:
   ```python
   T_SEAL_START_MS = 1762128000000  # 2025-11-02T14:00:00 UTC
   df_free = df[df["t_close_ms"] < T_SEAL_START_MS]  # P0.5-P5 전용
   df_sealed = df[df["t_close_ms"] >= T_SEAL_START_MS]  # P6 단 1회만
   ```
2. **Validator unit test 추가** (P0.3 deliverable D4 의무):
   - sealed 데이터 위에서 fitting 실행 시 assertion fail
3. **Plot/visualization 정책**: full-period chart 작성 시 sealed window는 별도 색 + 명시적 "DO NOT INTERPRET" 라벨
4. **P3 진입 시**: P3 final eval window (`T_p3_holdout_start ~ T_seal_start`)는 P3 fitting 단계 이후 1회만 사용

---

**Pre-commit signed (P0.2 closure)**: Claude Code agent, 2026-05-01T14:00:00 UTC.
이 amendment 변경 시 P0-P5 모든 작업 reset + 새 seal 정의.

