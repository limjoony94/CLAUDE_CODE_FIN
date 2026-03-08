# Pattern Trading Bot - Changelog

> **Last Updated**: 2026-03-08
> **Current Version**: v1.55.0

---

## [2026-03-08] - v1.55.0: Critical Resilience Fixes & Baseline Verification

### Added
- Pattern recovery from trade history (`_recover_pattern_from_history()`) — eliminates N/A pattern cascades from API crashes
- 3-tier exit classification system (near-SL 40%, near-TP 30%, cascade SL) for accurate performance attribution
- Mass closure prevention via re-fetch sanity check (3+ simultaneous closures trigger verification)
- Cascade SL integration into Scanner N-pos evaluation for metric alignment
- 4 critical research studies confirming mechanism dominance (86% of PnL) and WF non-discrimination
- Safety validation checklist at bot startup (12-item gate: API key, exchange conn, position state, emergency SL, pattern discovery, scan staleness)
- Enhanced logging for N/A recovery events, exit classification, mass closure alerts

### Changed
- EXPECTED_WIN_RATE: 71.0% → 61.6% (conservative OOS+live slippage alignment)
- `position_close.py`: Added pattern history recovery on N/A state
- `position_monitor.py`: Enhanced exit inference with price proximity classification
- `pattern_scanner.py`: Integrated cascade SL into portfolio_npos() evaluation

### Fixed
- API 109500 network error cascade → false mass closure → -67.68% loss (CRITICAL)
- Exit reason attribution ambiguity (MARKET/UNKNOWN → 95% accuracy classification)
- Simultaneous position closure glitch (re-fetch verification prevents false cascades)
- Scanner-Production WF metric divergence (cascade SL now aligned)
- Expected WR estimate drift (conservative 15pp live slippage buffer established)
- Pre-03-05 data contamination (state truncated, clean 44-trade baseline re-established)
- 53 unauthorized trades from old pattern sets (22 N/A cascades, 31 legacy patterns)

### Documented
- **Mechanism Stack Dominance**: 86% of PnL from guards (cascade SL 34%, AggRisk 28%, DirCap 15%, etc.), only 14% from pattern discovery
- **WF Non-Discrimination**: 30/30 random signals pass WF validation → WF is mechanism validator only, not edge validator
- **SHORT Structural Weakness**: -20pp vs LONG in uptrends (mean reversion + vol clustering); architectural, unfixable at entry level
- **Holdtime Insignificance**: 25/25 position duration tests non-discriminating → outcome ≠ duration driver
- **Live Gap Analysis**: Expected 61.6% vs Actual 65.9% (within 15pp buffer); no systematic underperformance
- **Research Protocol**: All studies follow Standard Research Protocol with WF validation + MC testing

### Known Limitations
- SHORT direction structurally weak in bull regimes (directional regime detection required; not yet implemented)
- WF validation cannot discriminate genuine patterns from random signals (mechanism stack is primary edge, not patterns)
- 15pp expected-to-actual variance buffer remains conservative until >200 live trades accumulated

---

## [2026-03-05] - v1.54.0: Post-Crisis Recovery & Pattern Set Refresh

### Added
- API error handling improvements (retry logic, timeout handling)
- Emergency SL preservation across connection losses
- Position state reconstruction from API history

### Changed
- Pattern set refreshed with MAE/MFE scanner v2.4
- ATR configuration aligned: clamp [0.5, 1.5]

### Fixed
- Position data inconsistencies from API 109500 errors
- False mass closure cascades

---

## [2026-03-03] - v1.53.0: Mechanism Stack Optimization (Baseline)

### Added
- Cascade SL Tightening (v1.45.0) — SL exit reduces same-direction SL by 85%
- Aggregate Risk Cap (v1.49.0) — counter 8%, with 15% directional exposure limits
- Momentum Guard (v1.46.0) — detects 1.5% 15-min moves, blocks reverse entry for 1h
- Position Timeout (v1.48.0) — 288-bar (24h) forced exit

### Changed
- N-pos Scanner default (v1.38.1) — compound equity evaluation with direction cap + agg risk
- Direction Cap: 7 (max same-direction positions in 9-slot portfolio)
- EXPECTED_WIN_RATE: 76.6% (OOS N-pos aligned rescan, 131 patterns)

### Metrics
- **IS (N-pos)**: WR 86.6%, PnL +220.8%, MDD 2.01%, PnL/MDD 109.7x
- **WF OOS**: 3/3 PASS, F1 +25.3%, F2 +30.5%, F3 +54.9%, Total +110.6%, Avg WR 76.6%
- **Pattern Set**: 131 patterns (59L + 72S), Edge 18-31.8pp, TP 0.85-2.80%, SL 1.44-5.95%

---

## [2026-02-27] - v1.51.0-v1.52.0: ATR & Risk Tuning

### Changed
- v1.52.0: ATR clamp alignment [0.5, 1.5]
- v1.51.0: Momentum Guard threshold 1.0% → 1.5%
- v1.50.0: ATR clamp_hi 1.7 → 1.5

---

## [2026-02-20] - v1.48.0: Position Timeout Optimization

### Changed
- Position Timeout: 864 bars (72h) → 288 bars (24h)
- Rationale: timeout_sweep_study showed OOS min +17.5% improvement; aligns with scanner MAX_BARS=288

---

## [2026-02-15] - v1.45.0: Cascade SL Tightening

### Added
- **Cascade SL Mechanism**: SL exit → same-direction SL reduced by 85% (tighten_pct)
- Effect: IS PnL +44%, OOS min fold +64.4%
- Purpose: Stop drawdown amplification when consecutive losses cluster

---

## [2026-02-10] - v1.42.0: Mechanism Stack Finalization

### Changed
- **Disabled**: Regime Sizing (M2), Adaptive Leverage (M4) — redundancy -46.94
- **Retained**: Cascade SL (P3), AggRisk (G5), Direction Cap (G1), Momentum Guard (G2), ATR Scaling (M5)
- Rationale: guard_ablation_study.py + mechanism_portfolio_study.py confirmed Cascade SL + AggRisk essential

### Metrics
- IS: PnL/MDD 34.36 → 123.67 (+260%), MDD 4.50% → 3.82%
- OOS: +77.3% → +160.7%, min fold +6.1% → +44.0%

---

## [2026-01-30] - v1.38.1: N-pos Scanner Default

### Changed
- Scanner `--npos` flag: default=True, `--no-npos` for legacy 1-pos mode
- Rationale: 3/3 WF PASS, 15pp live gap reduction (32.3pp → 15.4pp)

### Metrics
- **N-pos IS**: WR 72.8%, MDD 6.5%, PnL/MDD 22.3x
- **WF OOS**: WR 68.4%, PnL +37.8% (vs 1-pos +872.7% — realistic N-pos variance)

---

## [2026-01-25] - v1.36.3: Neutral Window Discovery

### Added
- Automatic neutral window detection (±1% start/end price tolerance)
- Scanner identifies longest 'neutral' candle cluster in lookback
- Pattern source expands from static 51 to dynamic discovery

---

## [2026-01-20] - v1.30.0: Hedge Mode Implementation

### Changed
- Position Mode: One-Way BOTH → Hedge (LONG/BOTH + SHORT/BOTH independent)
- Effect: PnL/MDD 3x better (5.88 vs 0.97), removes forced closure on direction reversal

---

## [2026-01-15] - v1.27.3: Dynamic Pattern Selection

### Added
- `pattern_source: dynamic` mode — bot loads patterns from `results/dynamic_patterns.json`
- Scanner generates universal TP/SL per pattern (MAE/MFE + ATR scaling)

---

## [2026-01-10] - v1.21.0: Per-Pattern TP/SL

### Added
- Individual TP/SL optimization for each pattern (vs global TP/SL)
- Grid search over TP percentile (30-85), SL percentile (15-70)
- Pattern quality filter: MC < 0.01, min_trades >= 25

---

## [2025-12-20] - v1.17.0: Early Exit Signal

### Added
- 3-candle sequence detection: 3×BD (LONG) or 3×BU (SHORT) + 0.3% profit → early exit
- Purpose: Capture quick reversions, avoid reversal traps

---

## [2025-12-15] - v1.14.0: Context Filters

### Added
- RSI, Volume, Trend filter infrastructure
- v2 research later found non-discriminating; filters inactive (config `enabled: false`)

---

## [2025-12-01] - v1.13.0: Foundation Baseline

### Added
- 3-candle pattern matching (12-type classification)
- Fixed 3x leverage, N=9 Hedge mode positions
- TP/SL per-pattern ATR-scaled discovery (MAE/MFE scanner)

---

## Archive

Earlier versions (v0.x - v1.12.x) archived in `docs/VERSION_HISTORY.md`
Deprecated bots (Engulf 5m, others) archived in `archive/deprecated_bots/`

---

**Maintained by**: Claude Code
**Last Verification**: 2026-03-08
**Status**: Active Development
