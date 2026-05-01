# Game-Theoretic Force-Flow BTC v2

**Status**: P0 (zero-base inventory) — started 2026-05-01
**Mandate**: `memory/000_session_start.md`
**Capital scope**: S0 ($1.5K BingX retail) → S1 ($10K-$100K)
**Parent project**: separate from `bingx_rl_trading_bot/` (R26/C1 lineage shelved)

---

## Theoretical Foundation

게임이론 + 시장 미시구조 기반 BTC perp 거래 framework. 핵심 통찰:

1. **가격 = 2D long-short 동적 균형** (Brock-Hommes HAM, Kyle 1985)
2. **OI rotation mechanics** — `long_open=long_close` 매칭 시 OI 불변하나 cost basis 회전. 저가에서 long-side가 적은 자본으로 더 큰 force 투사 가능 (Wyckoff Accumulation, Brunnermeier-Pedersen 2005)
3. **SL/TP forced flow** — 패자/승자가 강제·자발적으로 반대방향 force 투사 → 다음 trader가 absorber 위치 가능
4. **Multi-niche optimization** (NK-model PLS-complete, MAP-Elites): 단일 global optimum 추구는 mathematical 잘못된 framing

---

## Project Structure

```
game_theory_btc_v2/
├── data/                    # parquet/csv (gitignored — 큰 파일)
├── scripts/
│   ├── data/               # fetch_*.py, audit_*.py
│   ├── analysis/           # mechanism backtests
│   └── validators/         # friction_model.py, bootstrap_six_criteria.py
├── experiments/
│   ├── p0/                 # P0 zero-base
│   ├── p1/                 # BingX API inventory
│   └── ...                 # P2-P6 후속
├── memory/                  # session memory (markdown)
├── notebooks/               # exploratory (gitignored)
├── tests/                   # pytest
├── logs/                    # runtime logs (gitignored)
├── config/                  # local config (api keys path 등)
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

---

## Anti-Fishing Charter (요약)

전체 framework의 모든 priority에 강제:

1. **Single-attempt pre-commit** — hypothesis + PASS/FAIL + stopping rule 사전 등록. silent pivot 금지.
2. **Fresh OOS holdout** — 마지막 90d (또는 25%)는 fitting 단계에서 sealed.
3. **Lookahead audit** — feature lag-shift sensitivity. lag 0 vs lag 1 dramatic 차이 → leak 의심.
4. **Realistic friction** — BingX taker 0.045%/side + slippage 0.02-0.05%/side = RT 0.13-0.20%. Funding cost 8h 별도.
5. **6-Criteria gate** — 3-day random window bootstrap (B=10000):
   - `mean ≥ target_daily` / `p5 ≥ 0` / `pos_rate ≥ 0.5`
   - `p_beats_baseline ≥ 0.55` / `MaxDD ≥ -X%` / `Sharpe ≥ 1.5`
6. **Honesty terminal value** — PARTIAL은 PARTIAL로 closure.

---

## Capital-Stage Strategy (현재 scope: S0-S1)

| Stage | Capital | Optimal Strategy | Binding Friction |
|-------|---------|------------------|------------------|
| S0 | $1.5K | Friction-aware low-frequency high-edge (Funding Arb baseline + Force-flow reversal) | Taker fee + slippage |
| S1 | $10K-$100K | MAP-Elites multi-mechanism + Kelly fractional | 동일 |

S2 ($100K+) 이상은 별도 mandate (Almgren-Chriss execution, mean-field self-impact).

---

## Priority Schedule

```
P0  Zero-base inventory + theory grounding + env setup     (7d)  ← CURRENT
P1  BingX API + 공개 데이터 인벤토리                        (1d)
P2  Force-Flow Reversal Hypothesis (H3-H4)                  (3-5d)
P3  MAP-Elites on Mechanism × Regime Grid (H8)              (5-7d)
P4  Risk-Aware Thompson Sampling Bandit (H9)                (3-5d)
P5  Force-Flow Detection 정밀화 (P2 PASS/PARTIAL 시만)      (5-10d)
P6  통합 Portfolio + LIVE-readiness                          (3-5d)
```

각 priority entry 시 `experiments/p{N}/precommit.md` 의무 작성.

---

## Honesty Disclosure

§ 10.1 복리 수학표는 **목표가 아니라 envelope 인식용**:

| Daily | $1.5K → $1M | Probability |
|-------|------------|-------------|
| Funding Arb only (~0.019%/day) | ~100년 | 99% |
| v2 mandate full success (0.10-0.20%/day) | 6.5-13년 | 30-40% |
| Aggressive snowball (0.50%/day) | 3.6년 | 5-10% |
| Whale anecdotes (1%/day) | 2.5년 | <2% |

Survival bias 반대편 항상 인지. PASS 못 한 priority → honest closure.

---

## References

학문 references는 mandate § 11 참조 (memory/000_session_start.md).
