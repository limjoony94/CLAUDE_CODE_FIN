# P0 Pre-Commit — Zero-Base Inventory + Theory Grounding

**Pre-commit date**: 2026-05-01
**Priority**: P0 (foundational)
**Duration**: 7 days hard limit

---

## Hypothesis

7일 이내에 다음 7개 deliverable 모두 working code + documented + reproducible 상태로 구축할 수 있다.

## 7 Deliverables (PASS criteria)

| # | Deliverable | PASS criterion | 측정 방법 |
|---|-------------|---------------|----------|
| D1 | BTC 1h+5m+1d 720d data | 모든 timeframe 720d, gap rate <5%, NaN policy 명시, lookahead-free (timestamp = candle close 시점) | `pandas.read_parquet` + describe + audit script |
| D2 | BingX/Binance API access (CCXT) | `fetch_ohlcv` × 3 timeframe 작동 + funding/OI history fetch | unit test in `tests/test_data_fetch.py` |
| D3 | Realistic friction model | `friction_model.py`: taker 0.045%/side + slippage 0.02-0.05%/side parameterized + funding cost 8h 별도 | unit test: known input → known fee output |
| D4 | 6-criteria validator | `bootstrap_six_criteria.py`: 3-day random window, B=10000, dict output. Constant +0.5%/day series PASS, random walk FAIL | unit test in `tests/test_validator.py` |
| D5 | Baseline buy-and-hold benchmark | BTC 720d buy-and-hold daily P&L (friction 적용) + 1× constant long perp + funding cost + random entry baseline. 6-criteria 결과 표 | `experiments/p0_baselines/result.md` |
| D6 | H1-H9 정량 정의 | 각 가설마다 feature definition (lookahead-aware) + entry/exit rules + parameter space + expected sample size | `memory/p0_hypothesis_registry.md` |
| D7 | Reference candidate strategies (~30 mechanism minimal def) | 6 family (momentum/breakout/reversion/pattern/regime/cross-section) × ~5 each, minimal definition (entry/exit signature). full sweep은 P3에서 | `scripts/analysis/mechanism_catalog.py` 또는 `memory/p0_mechanism_catalog.md` |

## FAIL criteria

- 7일 후 1개 이상 deliverable 미완성 → P0 FAIL
- 데이터 quality audit fail (예: lookahead leak 발견, gap >5%) → 해당 deliverable FAIL
- Validator unit test 통과 못 함 → D3/D4 FAIL
- Honesty 위반 (e.g. fake metrics, silent re-scope) → 즉시 P0 closure + mandate 재검토

## PARTIAL acceptance

- 7개 중 5-6개 통과 + 누락 1-2개에 대한 명확한 blocker 식별 → PARTIAL.
- PARTIAL 시 GO/NOGO for P1은 사용자 결정.

## Stopping Rule

- **Hard limit**: 7 days from session start (2026-05-01 → 2026-05-08).
- D2 (BingX API access)에서 rate-limit / IP block 등 blocker 발생 시 24h 내 보고.
- 데이터 수집 중 timestamp 정책 (UTC vs exchange-native) 모호 시 보고 후 결정.

## Anti-Fishing Locks

1. **No silent pivot**: deliverable 정의는 본 문서가 ground truth. 변경 시 사용자 승인 필요.
2. **No fake validation**: validator unit test는 reproducible cell-level seed 필수.
3. **No data backfill**: P0.2 fetch 후 데이터 quality fail 시 재fetch. "good enough"로 진행 금지.
4. **No closure cherry-pick**: Day 7에 7/7 모두 보고. 5/7 + cherry-pick 4 highlights 금지.

## Closure Output (Day 7)

`experiments/p0/result.md`에 다음 포함:
- 7-deliverable status table (PASS/FAIL/PARTIAL)
- 각 deliverable 별 evidence (file path, unit test result, screenshot 등)
- 발생한 blocker + 해결 또는 escalate
- GO/NOGO recommendation for P1
- P0에서 학습한 anti-pattern (memory에 archive)

---

**Pre-commit signed**: Claude Code agent, 2026-05-01.
**Cannot be modified post-execution.** 변경 사항은 별도 amendment 문서로.
