# Gap Analysis: pattern_5m v1.34.0

> Date: 2026-02-24 | Phase: Check | Analyzer: gap-detector (manual)

---

## Plan vs Implementation Comparison

### Plan: v1.34.0 — 주기적 재스캔 + 7일 Holdout + MDD 동적 사이징

3가지 방어 메커니즘 + BH FDR 버그 수정 + Clean Protocol 추가

---

## Change 1: Scanner `--holdout-days` (7일 Holdout 검증)

| Plan 항목 | 구현 상태 | 상세 |
|-----------|----------|------|
| `--holdout-days` CLI 인자 | **DONE** | L1481, default=7 |
| `holdout_validate()` 함수 | **DONE** | L1360-1456, WR Excess > 0 검증 |
| main() holdout split | **DONE** | L1605-1611, `df_holdout = df[-holdout_bars:]` |
| Holdout FAIL 제거 + SKIP 유지 | **DONE** | L1702-1738, FAIL → removed, SKIP → kept |
| 출력 JSON `holdout_validation` | **DONE** | L1346-1351, `build_output_json()` |
| **Match**: 5/5 | | |

### Change 1 추가 (Plan 외): `--clean` Protocol v3.0

| 추가 항목 | 구현 상태 | 상세 |
|-----------|----------|------|
| `--clean` CLI flag | **DONE** | L1484-1485 |
| BH FDR 버그 수정 (`m=n_tested`) | **DONE** | L235, `m = max(n_tested, len(sorted_items))` |
| Pre-registration manifest | **DONE** | L1522-1546, `clean_scan_manifest.json` |
| Post-BH practical filters | **DONE** | L1665-1690, edge >= 5pp + SL >= 1.0% |
| `mc_threshold=1.0` override | **DONE** | L1496, clean mode에서 MC pre-filter 비활성 |
| **Match**: 5/5 | | |

---

## Change 2: Bot 스캔 Staleness 체크

| Plan 항목 | 구현 상태 | 상세 |
|-----------|----------|------|
| `_check_scan_staleness()` 함수 | **DONE** | bot.py L225-241 |
| `rescan_interval_days` config | **DONE** | config.yaml L32, default=90 |
| 봇 시작 시 호출 | **DONE** | bot.py L280-281 (`run()` 초기화) |
| 경고만 출력 (봇 차단 없음) | **DONE** | `logger.warning()` only |
| **Match**: 4/4 | | |

---

## Change 3: MDD 기반 동적 사이징

| Plan 항목 | 구현 상태 | 상세 |
|-----------|----------|------|
| `peak_equity` default state | **DONE** | state.py L112 |
| `update_peak_equity()` | **DONE** | state.py L211-214 |
| `get_position_size(state=)` 파라미터 | **DONE** | position_open.py L39 |
| MDD scale 계산 | **DONE** | position_open.py L66-85 |
| `per_slot_equity * mdd_scale` | **DONE** | position_open.py L85 |
| Peak equity 갱신 (bot.py) | **DONE** | bot.py L393, `update_peak_equity(state, equity)` |
| `mdd_sizing` config section | **DONE** | config.yaml L17-21 |
| **Match**: 7/7 | | |

---

## 추가 구현 (Plan 외)

| 항목 | 구현 상태 | 상세 |
|------|----------|------|
| `trade_history` persistence | **DONE** | models.py L162, L185, L254, L275 |
| `trade_detail` in `record_closed_position` | **DONE** | position_close.py L194-207 |
| `BOT_VERSION = "1.34.0"` | **DONE** | constants.py L15 |
| **Match**: 3/3 | | |

---

## Version & Config 일관성

| 항목 | 상태 |
|------|------|
| `BOT_VERSION` = "1.34.0" | OK |
| config `mdd_sizing.enabled` = true | OK |
| config `rescan_interval_days` = 90 | OK |
| Tests: **1067 passed** | OK |

---

## Gap Summary

| 항목 | Plan | 구현 | Gap |
|------|------|------|-----|
| Change 1: Holdout | 5 items | 5/5 DONE | 0 |
| Change 1+: Clean Protocol | (추가) | 5/5 DONE | 0 |
| Change 2: Staleness | 4 items | 4/4 DONE | 0 |
| Change 3: MDD Sizing | 7 items | 7/7 DONE | 0 |
| 추가: trade_history | (추가) | 3/3 DONE | 0 |
| Tests | 전체 통과 | 1067 passed | 0 |
| **TOTAL** | **24 items** | **24/24 DONE** | **0 gaps** |

---

## Match Rate: **100%**

모든 계획 항목이 구현되었으며, 추가로 Clean Protocol v3.0과 trade_history persistence가 구현됨.

---

## Observations

1. **Clean Protocol은 Plan에 없던 추가 구현** — BH FDR 버그 발견 후 session 중 추가
2. **Clean scan 결과**: 260패턴 발견, 현재 production 35패턴 중 34개(97%)가 Clean 통과
3. **Holdout 검증 한계**: 7일은 대부분 패턴에 충분한 거래를 생성하지 못함 (209/271 SKIP)
4. **CLAUDE.md Version History 미업데이트** — commit 후 업데이트 필요
