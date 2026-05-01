# whale_inference_abm

Open-system Agent-Based Market simulation with wealth concentration and 3-anchor inverse strategy recovery.

## Documents

- **Architecture v1.1**: `../bingx_rl_trading_bot/claudedocs/whale_inference_abm_architecture_v1.1.md`
- **Plan v0.1**: `../docs/01-plan/features/whale_inference_abm.plan.md`
- **Design v0.3**: `../docs/02-design/features/whale_inference_abm.design.md`

## Status

**Phase 0 (G0): ABM build** — IN PROGRESS (Days 1-15).

Spike skipped — ABIDES JPMorgan archived 2025-06-02. Custom build directly.

## Setup

```bash
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
pytest
```

## Critical environment requirement (advisor F2 patch)

Set `ABM_DATA_DIR` to a NON-OneDrive path before running simulations:

```bash
set ABM_DATA_DIR=C:\abm_runtime
```

OneDrive sync lock (BUG#58 precedent in trading bot) caused state.json corruption — same risk applies to high-frequency NDJSON writes from per-decision logger. Use local-only path.

## Phase 0 implementation order

| Days | Deliverable |
|------|-------------|
| 1-3 | Orderbook (CDA) + scheduler + determinism |
| 4-7 | 5 canonical agents + unit tests |
| 8-10 | Wealth + friction + admission scheduler + frozen-window |
| 11-13 | Logger NDJSON + per-decision log + integration tests |
| 14-15 | Smoke tests + reproducibility + schema diff vs BingX Phase 1 + G0 acceptance |
