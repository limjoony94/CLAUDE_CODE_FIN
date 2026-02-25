# Archive Index — 2026-02

## pattern_5m (v1.34.0)

| Document | Description | Match Rate |
|----------|-------------|------------|
| [pattern_5m.plan.md](pattern_5m/pattern_5m.plan.md) | Plan: 3 defensive mechanisms + Clean Protocol |
| [pattern_5m.analysis.md](pattern_5m/pattern_5m.analysis.md) | Gap Analysis: 24/24 items | **100%** |
| [pattern_5m.report.md](pattern_5m/pattern_5m.report.md) | PDCA Completion Report | **100%** |

**Summary**: Scanner holdout validation, BH FDR bug fix, Clean Protocol v3.0, MDD dynamic sizing, scan staleness check, trade_history persistence. 8 files modified, 1067 tests passed.

**Archived**: 2026-02-24

---

## mtf_direction_filter (Research — STOP)

| Document | Description | Outcome |
|----------|-------------|---------|
| [mtf_direction_filter.plan.md](mtf_direction_filter/mtf_direction_filter.plan.md) | Plan: 6-Phase MTF Direction Filter + Tight TP |
| [mtf_direction_filter.report.md](mtf_direction_filter/mtf_direction_filter.report.md) | PDCA Research Report | **STOP** |

**Summary**: 1h 3-candle 패턴 방향 예측력 검증 (15 strong, best 70.3%) → 5m 진입 필터로 적용 시 WR +2.5pp / trades -74% → 총 edge 감소 → 가설 기각. Phase 1-2 GO, Phase 3/3X STOP. 코드 변경 없음, v1.33.0 유지.

**Artifacts**: `scripts/analysis/mtf_direction_study.py`, `scripts/analysis/mtf_filter_backtest.py`, `results/mtf_direction_study.json`, `results/mtf_filter_backtest.json`, `data/btc_1h_720days.csv`

**Archived**: 2026-02-24

---

## atr_scanner_alignment (Research — GO → B_SCANNER_REPLACE_DEFAULT)

| Document | Description | Outcome |
|----------|-------------|---------|
| [atr-scanner-alignment.plan.md](atr_scanner_alignment/atr-scanner-alignment.plan.md) | Plan: 4-Phase ATR Scanner-Production Alignment |
| [atr-scanner-alignment.report.md](atr_scanner_alignment/atr-scanner-alignment.report.md) | PDCA Research Report | **GO** |

**Summary**: Scanner-Production TP/SL 불일치 해소 연구. H1 GO: ATR scanner 20pat vs Fixed 15pat (+5, +1.1pp WR Excess). H2 GO: ATR WF 3/3 PASS, OOS +108.1% vs Fixed +65.9% (+64%). H3 STOP: 32/32 param combos WF PASS, best +1.4% vs default (robust). Action: B_SCANNER_REPLACE_DEFAULT.

**Artifacts**: `scripts/analysis/atr_scanner_alignment_study.py`, `results/atr_scanner_alignment_study.json`

**Archived**: 2026-02-24
