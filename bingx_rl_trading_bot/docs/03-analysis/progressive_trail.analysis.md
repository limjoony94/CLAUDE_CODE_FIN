# Analysis: Progressive Trail — Design ↔ Implementation Gap

> **Feature**: progressive_trail
> **Date**: 2026-04-21
> **Phase**: Check (Gap Analysis)
> **Design**: `docs/02-design/features/progressive_trail.design.md`
> **Commit**: `3a669f1`

---

## Match Rate: **98%** — FULL GO

```
┌─────────────────────────────────────────────┐
│  Overall Match Rate: 98%                    │
├─────────────────────────────────────────────┤
│  signals.py check_exit branch:     100%     │
│  signals.py helper fn:             100%     │
│  config.yaml section:              100%     │
│  bot.py baton-touch dynamic K:     100%     │
│  Unit tests (6+ required):         100% (8) │
│  Doc/Memory deliverables:           90%     │
└─────────────────────────────────────────────┘
```

---

## 1. Required Production Changes — Full Match

### 1.1 signals.py — ✅ 100%
| Design Requirement | Implementation | Status |
|---|---|---|
| `get_effective_trail_k(best_pnl)` helper | `signals.py:105-109` | ✅ Exact |
| `prog_enabled AND best_pnl >= threshold` → `tk_post` | L107 | ✅ |
| Else → `trail_K` | L109 | ✅ |
| `check_exit` uses effective K | L161: `k_effective = self.get_effective_trail_k(best_pnl)` | ✅ |

**Improvement over design sketch**: Design §5.1 shows inline `prog_cfg = self.cfg.get(...)` inside `check_exit`. Impl captures `prog_trail_enabled/threshold/K_post` in `__init__` (L31-34) and routes via helper. **DRY + hot-path O(1) read**.

### 1.2 config.yaml — ✅ 100%
| Key | Design | Actual |
|---|---|---|
| enabled | false | false ✅ |
| threshold_pct | 0.9 | 0.9 ✅ |
| trail_K_post | 0.5 | 0.5 ✅ |
| Traceability comment | — | "9/9 CORE + 5/5 WARN FULL GO" @ L27-29 (extra) |

### 1.3 bot.py baton-touch dynamic K — ✅ 100%
| Design Requirement | Implementation |
|---|---|
| BUG#61b quadratic uses dynamic K | `bot.py:1092`: `trail_K = self.signal.get_effective_trail_k(best_pnl)` ✅ |
| Single source of truth (signals↔bot) | Shared helper — 수식 100% 일치 구조적 보장 ✅ |

### 1.4 Unit Tests — ✅ 100% (8 cases vs 6 required)
| Design Case | Test |
|---|---|
| enabled=false → baseline | `test_progressive_disabled_uses_base_trail_k` ✅ |
| best_pnl < thr → base K | `test_progressive_enabled_below_threshold_uses_base_k` ✅ |
| best_pnl >= thr → post K | `test_progressive_enabled_above_threshold_uses_post_k` (exact 0.9 boundary) ✅ |
| Boundary | Covered inside above (0.9 exact) ✅ |
| LONG/SHORT symmetry | `test_progressive_check_exit_long_applies_post_k_after_threshold` + `_short_symmetry` ✅ |
| NaN ATR safety | `test_progressive_nan_atr_safety` ✅ |

**Extra regression guards (Design X, Impl O)**:
- `test_progressive_check_exit_long_baseline_holds_at_same_point` — delta vs base K
- `test_progressive_config_defaults_when_missing` — defensive defaults

---

## 2. Gap List

### ✅ Critical: 0
### ⚠️ Minor: 1
1. **analysis.md 자체가 §8 checklist에 포함** — 본 문서로 해결됨

### Previously flagged, now resolved
- Memory note: ✅ `memory/progressive_trail_20260421.md` 작성 완료
- CLAUDE.md v4.8.0 entry: ✅ commit 3a669f1에 포함
- MEMORY.md Research History: ✅ reference 추가

---

## 3. Architecture / Parity Compliance

| Check | Result |
|---|---|
| signals.py ↔ bot.py formula parity | ✅ Single helper (`self.signal.get_effective_trail_k`) — BUG#61 재발 구조적 방지 |
| BUG#61b quadratic uses dynamic K | ✅ L1092 |
| Baton-touch STOP_MARKET trigger dynamic K | ✅ via `_calc_trail_trigger_price` |
| Pre-activation TRAILING_STOP_MARKET (structural limit) | ⚠️ Unchanged — design accepted (§3), best_pnl<0.05% 구간은 progressive 무관 |
| Backtest-live parity (22 checks) | Maintained — dynamic K preserves invariant |
| Regression (127 pytest) | ✅ 127/127 PASS |

---

## 4. Design 명세 vs 구현 refactor 차이 정리

| Design | Implementation | 평가 |
|--------|----------------|------|
| `check_exit` 내부에서 `prog_cfg = self.cfg.get(...)` inline lookup | `__init__`에서 캐시 + helper fn | **개선** (hot-path 최적화) |
| `k = k_post if best_pnl >= thr else trail_K` 직접 작성 | `get_effective_trail_k` 추상화 | **개선** (재사용성, 테스트 용이) |
| bot.py에서 helper fn 직접 call | 동일 helper 재사용 | ✅ 설계 의도와 완전 일치 |

---

## 5. Verdict

**Match Rate 98% ≥ 90% → advance to `/pdca report`**

Production code gap 사실상 0. Helper-based refactor는 design sketch를 개선하면서 의미는 유지.
단일 helper(`get_effective_trail_k`)가 signals.py와 bot.py 양쪽에서 호출되어 Design §7 Risk
"BUG#61 재발 가능성 — signals.py와 bot.py 수식 100% 동일성" **구조적으로 차단**.

### 다음 단계 권장
1. `/pdca report progressive_trail` — 완료 리포트 생성
2. 30일 LIVE 관찰 (config `enabled: false` 유지)
3. Live 관찰 후 `enabled: true` 활성화 조건 검토

---

## 6. Files Examined

- `docs/02-design/features/progressive_trail.design.md`
- `scripts/production/c1_breakout/signals.py` (L31-34, 100-109, 161-167)
- `scripts/production/c1_breakout/bot.py` (L1066-1116, 1190-1340)
- `config/c1_breakout_config.yaml` (L27-35)
- `scripts/tests/test_progressive_trail.py` (8 cases)
- `scripts/analysis/progressive_trail_full_validation.py`
- `results/progressive_trail_validation_20260421_003533.json`
