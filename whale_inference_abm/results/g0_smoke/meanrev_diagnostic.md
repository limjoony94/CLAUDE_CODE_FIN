# MeanReversion Diagnostic at 1k Bars (advisor caveat #2)

**Date**: 2026-05-01
**Trigger**: G0 → G1 transition advisor caveat #2 — MeanReversion produced 0 trades in 100-bar smoke; verify whether 1k-bar smoke also shows under-trading before committing G1 to a 5-family canonical set.

## Pre-calibration finding (threshold = 0.5%, design v0.4 default)

```
1000-bar smoke (seed=42, post-v0.5 MM/Random calibration):
  Total trades: 17087
  Bar snapshots WITH mid: 1000/1000 (100%)
  Family trade-leg counts:
    market_maker: 16909
    momentum:     15208
    random:        2048
    piggyback:        9
    mean_reversion:   0    ← 12,295 decisions made, ALL Hold
  Mid range: 49775.24 → 50101.21 (0.655%)
```

Per advisor decision tree: 0 legs = **"Real problem"** — IRL training cannot recover what isn't observed.

## Threshold sweep (1k bars, seed=42)

| threshold | Total trades | MR legs | Verdict |
|-----------|--------------|---------|---------|
| 0.5% (design v0.4 default) | 17,087 | 0 | Fail — narrow trigger never fires |
| 0.3% | 17,087 | 0 | Fail |
| 0.2% | 17,087 | 0 | Fail |
| **0.1%** | **19,699** | **62** | **Pass** — IRL signal sufficient |

Root cause: Mid range in 1k-bar smoke is 0.655%, but MA tracks mid closely so MA-mid deviation rarely exceeds even 0.2%. Threshold ≥ 0.2% never triggers under realistic ABM dynamics with the v0.5-calibrated agent population.

## v0.6 calibration

- `MeanReversionAgent.threshold` default: 0.005 → **0.001** (5× looser)
- Strategy character preserved (still contrarian fade against MA)
- Trigger now realistic given ABM mid-range characteristics
- Test factory `_meanrev()` retains explicit `threshold=0.005` so unit tests are unaffected
- Design v0.6 Section 4.2 patched to document calibrated value

## Post-calibration verification (1k bars, seed=42)

```
Total trades: 21,873
Family trade-leg counts:
  market_maker:  21,708 (49.6%)
  momentum:      19,914 (45.5%)
  random:         2,034
  mean_reversion:    62  ← was 0
  piggyback:         28
```

MeanRev: 0 → 62 legs. Per advisor decision tree: **≥ 50 = G1 unblocked**.

## Decision

- v0.6 patch applied: threshold 0.005 → 0.001
- All 5 canonical families now produce observable behavior at 1k bars
- G1 IRL training has signal for all 5 families to recover
- T-G1 unblocked

## Caveats remaining (do NOT block G1)

1. **MeanRev still rare** (62 / 21,873 = 0.28%). IRL training will see less MR data than other families. Expect higher variance in MR policy recovery. Mitigation: G1 IRL evaluation should report per-family recovery accuracy, not aggregate.
2. **Piggyback also rare** (28 trades) but expected — cold-start lookback=10 means active for last 990 bars only, and piggyback strategy is conditional on having a "top performer" worth following.
3. **Calibration is empirical**, not theoretically motivated. Threshold 0.001 was chosen because it produces ≥ 50 trades at 1k bars under v0.5-calibrated ABM dynamics. Different ABM populations or longer runs may suggest different optima.
