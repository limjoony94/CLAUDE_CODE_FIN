# Hedge vs One-Way Re-Verification Study (2026-03-08)

Script: `scripts/analysis/hedge_vs_oneway_reverification.py`
Results: `results/hedge_vs_oneway_reverification.json`
Config: 303d neutral(259d), N=9, DirCap=7, Cascade 85%, AggRisk 8/15, Timeout 288, ATR [0.5,1.5]

## Experiment 1: IS Comparison (5 strategies)

| Strategy | Trades | WR | PnL | MDD | PnL/MDD |
|----------|--------|-----|------|-----|---------|
| **Hedge (현행)** | 4,070 | 55.3% | +502.5% | 7.5% | 67.35 |
| FIFO | 7,220 | 46.6% | -13.9% | 26.8% | 0 |
| Close-ALL | 12,390 | 36.1% | -93.5% | 93.5% | 0 |
| Smart-OneWay | 2,967 | 56.7% | +452.5% | 14.6% | 31.04 |
| Hedge-NoCap | 4,078 | 55.3% | +519.4% | 7.5% | 69.6 |

FIFO forced closures: 5,517건 → WR -8.7pp, PnL 파괴

## Experiment 2: WF 3-Fold

| Strategy | Pass | OOS Avg PnL | OOS Min |
|----------|------|-------------|---------|
| Hedge | 3/3 PASS | +58.4% | +43.5% |
| FIFO | 1/3 FAIL | -5.2% | -17.5% |
| Close-ALL | 0/3 FAIL | -47.5% | -55.4% |
| Smart-OneWay | 3/3 PASS | +58.4% | +24.5% |
| Hedge-NoCap | 3/3 PASS | +59.5% | +39.6% |

## Experiment 3: Random Discrimination (20 seeds)

| Strategy | Random Pass | Verdict |
|----------|------------|---------|
| Hedge | 20/20 (100%) | NON-DISCRIMINATING |
| FIFO | 0/20 (0%) | DISCRIMINATING (= always loses) |
| Close-ALL | 0/20 (0%) | DISCRIMINATING (= always loses) |
| Smart-OneWay | 19/20 (95%) | NON-DISCRIMINATING |
| Hedge-NoCap | 20/20 (100%) | NON-DISCRIMINATING |

FIFO 0% = 강제 청산이 어떤 신호로도 수익 불가능하게 만듦

## Experiment 4: Mechanism Ablation (Hedge)

| Config | PnL | MDD | PnL/MDD |
|--------|-----|-----|---------|
| Full | +502.5% | 7.5% | 67.35 |
| No Cascade | -50.3% | 54.5% | 0 |
| No AggRisk | +685.9% | 10.9% | 62.91 |
| No Momentum | +474.1% | 7.5% | 63.54 |
| Bare | -45.3% | 53.9% | 0 |

## Conclusions
- **Hedge 우위 재확인**: v1.30.0 원본(PnL/MDD 5.88 vs 0.97)보다 더 극적 차이
- **FIFO는 현행 메커니즘 스택에서 수익 불가능** (forced close 5,517건이 전략 파괴)
- **Smart-OneWay는 차선**: PnL/MDD 31.04 (Hedge의 46%), 거래수 -27%
- **Cascade SL이 핵심 수익원**: 제거 시 PnL +502% → -50%
- **현행 Hedge 모드 유지가 최적** — 변경 근거 없음
- v1.30.0 결과와 일관됨 (데이터 303d, 메커니즘 full stack으로도 동일 결론)
