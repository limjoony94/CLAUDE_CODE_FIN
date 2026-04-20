# Design: Progressive Trail Full Validation

> **Feature**: progressive_trail
> **Date**: 2026-04-21
> **Phase**: Design
> **Plan**: `docs/01-plan/features/progressive_trail.plan.md`

---

## 1. Architecture

```
scripts/analysis/progressive_trail_full_validation.py  (NEW, ~600 lines)
 ├─ 재사용:
 │   ├─ intrabar_trail_impact (ibt)           — BT 엔진, atr14/c15/h15/l15/o15/ch_h/ch_l/sw_l/sw_h/sig
 │   ├─ c1_intrabar_parity.apply_slippage, SLIPPAGE (MED base)
 │   ├─ regime_filter_lowvol_study.run_bt_with_regime
 │   ├─ regime_filter_trend_study.precompute_trend_pass
 │   └─ progressive_trail_extended.make_check_exit_profit (monkey-patch factory)
 │
 ├─ 신규 함수:
 │   ├─ run_variant(label, tk_base, tk_post, threshold, slip, regime) → trades
 │   ├─ strict_expanding_wf(trades, n=5) → (fold_pnls, pos_count)
 │   ├─ oos_split_boundaries(tk_base, tk_post, thr) → 5 boundaries dict
 │   ├─ three_way_split(trades, split=(0.6,0.2,0.2)) → dict
 │   ├─ bootstrap_3day(trades, n=1000) → metrics dict
 │   ├─ bootstrap_relative(cand_trades, base_trades, n=1000) → P(cand>base)
 │   ├─ neighborhood_eval(thr_grid, tk_grid, regime) → 25-combo results
 │   ├─ parameter_consistency_halves(tk_base, tk_post, thr) → half1/half2 delta
 │   ├─ mc_direction_test(trades, n=999) → p-value
 │   ├─ fold2_regime_recheck(trades) → fold 2 breakdown
 │   ├─ slippage_sensitivity_3scenario(tk_base, tk_post, thr, regime) → 3 results
 │   └─ evaluate_go_9flags(all_results) → flag dict + pass count
 │
 └─ Output: results/progressive_trail_validation_{stamp}.json + stdout report
```

**핵심 설계**: Baseline(tk=2.5 고정) vs Candidate(tk_base=2.5, tk_post=0.5, thr=0.9) 동일 파이프라인 평가 — **단일 변수(profit-conditional K) 효과 순수 측정**.

---

## 2. Configurations

### 2.1 Combos (최종 target 중심)
```python
BASE_STRAT = {
    'max_sl_atr': 4.0, 'trail_K': 2.5, 'max_hold_bars': 192,
    'body_min_ratio': 0.60,
}
TREND_FILTER = {'lb': 192, 'thr_pct': 1.0}

VARIANTS = {
    'baseline':  dict(tk_base=2.5, tk_post=2.5, threshold=99.0),  # no switch
    'prog_0.9_0.5': dict(tk_base=2.5, tk_post=0.5, threshold=0.9),  # TARGET
    'prog_0.8_0.5': dict(tk_base=2.5, tk_post=0.5, threshold=0.8),  # alt
    'prog_1.0_0.5': dict(tk_base=2.5, tk_post=0.5, threshold=1.0),  # alt
}
```

### 2.2 Slippage 3 Scenarios (재사용 c1_intrabar_parity.SLIPPAGE)
```python
SLIP_LOW  = {k: v * 0.5 for k, v in SLIPPAGE.items()}
SLIP_MED  = SLIPPAGE                     # entry 0.05 / sl 0.15 / trail 0.05 / emerg 0.30
SLIP_HIGH = {k: v * 2.0 for k, v in SLIPPAGE.items()}
```

### 2.3 Neighborhood Grid (Sharp Peak 검사)
```python
NEIGHBOR_THR = [0.7, 0.8, 0.9, 1.0, 1.1]
NEIGHBOR_TKT = [0.3, 0.4, 0.5, 0.6, 0.7]
# 5×5 = 25 combos, 각각 bar_close SLIP_MED PnL
```

### 2.4 WF 분할
```python
STRICT_WF_FOLDS = 5
# Train: [0, train_end), OOS: [train_end, fold_end)
# fold i: train_end=i*OOS_W+WARMUP, fold_end=train_end+OOS_W
# WARMUP=26, OOS_W=(n15 - WARMUP) // 5

OOS_SPLIT_BOUNDARIES = [
    (0.60, 0.20, 0.20),
    (0.50, 0.25, 0.25),
    (0.70, 0.15, 0.15),
    (0.55, 0.15, 0.30),
    (0.60, 0.15, 0.25),
]
```

### 2.5 Bootstrap
```python
BOOT_N_SAMPLES = 1000
BOOT_WINDOW_BARS = 288  # 3 days × 96
BOOT_START_MIN = 220    # post-warmup
BOOT_SEED = 42
```

---

## 3. Test Battery (9 Core + 3 Warnings)

### 3.1 Test 1: Full Period (Clean + Slip3)
**Input**: variants × 3 slippage
**Output**: PnL, MDD, WR, trades_per_day, daily, ex_top5 per cell
**Pass**: `prog` ≥ `baseline + 10pp` at SLIP_LOW and SLIP_MED

```python
def test_full_period():
    for variant, cfg in VARIANTS.items():
        for slip_name, slip in [('LOW',SLIP_LOW),('MED',SLIP_MED),('HIGH',SLIP_HIGH)]:
            trades = run_variant(variant, slip=slip, regime=trend_passes, **cfg)
            metrics[variant][slip_name] = compute_metrics(trades)
```

### 3.2 Test 2: Strict Expanding WF 5-fold
**Implementation**:
```python
def strict_expanding_wf(tk_base, tk_post, thr, n=5):
    WARMUP = 26; OOS_W = (ibt.n15 - WARMUP) // n
    fold_pnls = []
    for i in range(n):
        train_end = WARMUP + i*OOS_W
        fold_end = train_end + OOS_W
        # OOS range만 trades 필터 (clean BT 후 filter)
        all_trades = run_variant('eval', tk_base, tk_post, thr, SLIP_MED, passes)
        oos_trades = [t for t in all_trades if train_end <= t['entry_bar'] < fold_end]
        fold_pnls.append(sum(t['net'] for t in oos_trades))
    return fold_pnls, sum(1 for p in fold_pnls if p>0)
```
**Pass**: 5/5 positive

### 3.3 Test 3: OOS Split 5 Boundaries
**Input**: 5 split ratios (60/20/20, 50/25/25, 70/15/15, 55/15/30, 60/15/25)
**Output**: train/val/test PnL per split
**Pass**: 5/5 splits have `val > 0 AND test > 0`

### 3.4 Test 4: 3-Way Split (60/20/20 default)
```python
def three_way_split(trades, ratios=(0.6,0.2,0.2)):
    total = ibt.n15
    tr_end = int(total*ratios[0])
    vl_end = tr_end + int(total*ratios[1])
    train = [t for t in trades if t['entry_bar'] < tr_end]
    val = [t for t in trades if tr_end <= t['entry_bar'] < vl_end]
    test = [t for t in trades if t['entry_bar'] >= vl_end]
    return {'train': sum(t['net'] for t in train),
            'val': sum(t['net'] for t in val),
            'test': sum(t['net'] for t in test)}
```
**Pass**: All 3 positive

### 3.5 Test 5: Bootstrap 3-Day
```python
def bootstrap_3day(trades):
    rng = random.Random(BOOT_SEED)
    pnls = []
    for _ in range(BOOT_N_SAMPLES):
        s = rng.randint(BOOT_START_MIN, ibt.n15 - BOOT_WINDOW_BARS - 1)
        e = s + BOOT_WINDOW_BARS
        pnls.append(sum(t['net'] for t in trades if s <= t['entry_bar'] < e))
    return {
        'mean': mean(pnls), 'median': median(pnls), 'std': stdev(pnls),
        'pos_pct': sum(1 for p in pnls if p>0)/BOOT_N_SAMPLES*100,
        'sharpe': mean(pnls)/stdev(pnls) if stdev(pnls)>0 else 0,
        'p5': sorted(pnls)[50],
        'p_loss_2pp': sum(1 for p in pnls if p<-2)/BOOT_N_SAMPLES*100,
    }
```
**Pass** (core): `mean>0 AND pos_pct>=55 AND p5>=-3.5`

### 3.6 Test 5b: Bootstrap Relative (vs Baseline Per-Window)
```python
def bootstrap_relative(cand_trades, base_trades):
    rng = random.Random(BOOT_SEED)
    c_wins = 0
    for _ in range(BOOT_N_SAMPLES):
        s = rng.randint(BOOT_START_MIN, ibt.n15 - BOOT_WINDOW_BARS - 1)
        e = s + BOOT_WINDOW_BARS
        cp = sum(t['net'] for t in cand_trades if s <= t['entry_bar'] < e)
        bp = sum(t['net'] for t in base_trades if s <= t['entry_bar'] < e)
        if cp > bp: c_wins += 1
    return c_wins / BOOT_N_SAMPLES * 100
```
**Pass**: `P(cand>base per-window) >= 55%` (not coinflip)

### 3.7 Test 6: Neighborhood 25 Combos
**Input**: 5×5 grid (thr × tkT)
**Output**: 25 PnL values, count of GO candidates
**GO criterion**: 각 combo가 baseline + 10pp 이상
**Pass**: GO count >= 8 (32%) — non-sharp-peak

### 3.8 Test 7: Parameter Consistency 2-Half
**Implementation**:
```python
def parameter_consistency_halves(tk_base, tk_post, thr):
    mid = ibt.n15 // 2
    all_trades = run_variant('eval', tk_base, tk_post, thr, SLIP_MED, passes)
    baseline_trades = run_variant('baseline', 2.5, 2.5, 99, SLIP_MED, passes)

    h1_cand = sum(t['net'] for t in all_trades if t['entry_bar'] < mid)
    h2_cand = sum(t['net'] for t in all_trades if t['entry_bar'] >= mid)
    h1_base = sum(t['net'] for t in baseline_trades if t['entry_bar'] < mid)
    h2_base = sum(t['net'] for t in baseline_trades if t['entry_bar'] >= mid)
    return {
        'h1_delta': h1_cand - h1_base, 'h2_delta': h2_cand - h2_base,
        'h1_cand': h1_cand, 'h2_cand': h2_cand,
        'h1_base': h1_base, 'h2_base': h2_base,
    }
```
**Pass**: `h1_delta >= +5pp AND h2_delta >= +5pp` — structural Δ 양 half 유지

### 3.9 Test 8: MC Direction Test (999 sims)
```python
def mc_direction_test(trades, n_sims=999):
    rng = random.Random(BOOT_SEED)
    real_pnl = sum(t['net'] for t in trades)
    beat = 0
    for _ in range(n_sims):
        shuffled = [t['net'] * rng.choice([-1, 1]) for t in trades]
        if real_pnl > sum(shuffled): beat += 1
    return 1 - beat / n_sims  # p-value (lower = better)
```
**Pass** (warning): `p < 0.01`

### 3.10 Test 9: Fold 2 Regime Re-check
**Purpose**: Fold 2(2025-07-11~09-15, 저변동성)에서 progressive가 baseline 대비 여전히 양수 우위인지.
**Implementation**:
```python
def fold2_regime_recheck(cand_trades, base_trades):
    WARMUP = 26; OOS_W = (ibt.n15 - WARMUP) // 5
    fold2_start = WARMUP + 1*OOS_W
    fold2_end = fold2_start + OOS_W
    cand_f2 = sum(t['net'] for t in cand_trades if fold2_start <= t['entry_bar'] < fold2_end)
    base_f2 = sum(t['net'] for t in base_trades if fold2_start <= t['entry_bar'] < fold2_end)
    return {'cand_fold2': cand_f2, 'base_fold2': base_f2, 'delta': cand_f2 - base_f2}
```
**Pass** (warning): `delta >= +3pp AND cand_fold2 > 0`

---

## 4. GO Protocol — 9-Flag Core + 4 Warning

```python
def evaluate_go_9flags(results):
    flags = {
        # Core (9 required)
        'f1_full_clean_pnl_gain':     results['full']['clean']['prog_0.9_0.5']['pnl']
                                       - results['full']['clean']['baseline']['pnl'] >= 10,
        'f2_full_slip_med_pnl_gain':  results['full']['slip_med']['prog_0.9_0.5']['pnl']
                                       - results['full']['slip_med']['baseline']['pnl'] >= 10,
        'f3_strict_wf_5of5':          results['strict_wf']['prog_0.9_0.5']['pos'] == 5,
        'f4_oos_split_5of5':          sum(1 for sp in results['oos_split']
                                          if sp['val']>0 and sp['test']>0) == 5,
        'f5_3way_all_positive':       all(v>0 for v in results['3way']['prog_0.9_0.5'].values()),
        'f6_bootstrap_core_3of3':     (results['bootstrap']['prog_0.9_0.5']['mean']>0
                                       and results['bootstrap']['prog_0.9_0.5']['pos_pct']>=55
                                       and results['bootstrap']['prog_0.9_0.5']['p5']>=-3.5),
        'f7_bootstrap_relative_55':   results['bootstrap_relative'] >= 55,
        'f8_neighborhood_8plus':      results['neighborhood']['go_count'] >= 8,
        'f9_structural_both_halves':  (results['consistency']['h1_delta']>=5
                                       and results['consistency']['h2_delta']>=5),
    }
    warnings = {
        'w1_mc_p_under_0.01':  results['mc']['p_value'] < 0.01,
        'w2_f6_ex_top5_pass':  results['full']['slip_med']['prog_0.9_0.5']['ex_top5'] > 0,
        'w3_mdd_improved':     (results['full']['slip_med']['baseline']['mdd']
                                - results['full']['slip_med']['prog_0.9_0.5']['mdd']) >= 1.0,
        'w4_sharpe_improved':  results['bootstrap']['prog_0.9_0.5']['sharpe'] >= 0.45,
    }
    return flags, warnings
```

**Deploy 조건**:
- 9/9 Core PASS → production config deploy (enabled=false)
- 3/4 Warning 이상 → 30일 LIVE 관찰 후 enabled=true 검토
- Core fail 1+ → STOP 재설계

---

## 5. Production Code Changes (After GO)

### 5.1 signals.py — `check_exit` modification
```python
# scripts/production/c1_breakout/signals.py (line ~100)
def check_exit(self, direction, entry_price, best_price, current_close,
               atr, bh, sl, trail_act_pct=None):
    # ... existing SL/emergency/timeout checks ...

    # Trail TP with progressive K
    if direction == 'LONG':
        best_pnl = (best_price / entry_price - 1) * 100
        cur_pnl = (current_close / entry_price - 1) * 100
    else:
        best_pnl = (1 - best_price / entry_price) * 100
        cur_pnl = (1 - current_close / entry_price) * 100

    trail_act = trail_act_pct or self.cfg.get('trail_activation_pct', 0.05)
    if best_pnl > trail_act and not math.isnan(atr) and atr > 0:
        # NEW: progressive K
        prog_cfg = self.cfg.get('progressive_trail', {}) or {}
        if prog_cfg.get('enabled', False):
            thr = prog_cfg.get('threshold_pct', 0.9)
            k_post = prog_cfg.get('trail_K_post', 0.5)
            k = k_post if best_pnl >= thr else self.cfg['trail_K']
        else:
            k = self.cfg['trail_K']

        trail_dist_pct = k * atr / current_close * 100
        drawdown = best_pnl - cur_pnl
        if drawdown >= trail_dist_pct:
            # ... realized calc, return TRAIL_TP ...
```

### 5.2 config/c1_breakout_config.yaml — new section
```yaml
progressive_trail:
  enabled: false              # validation 후 true 전환
  threshold_pct: 0.9          # best_pnl 기준
  trail_K_post: 0.5           # threshold 초과 시 적용
```

### 5.3 Bot.py — baton-touch 호환성
BUG#61b baton-touch trail 수식이 `trail_K` 변수를 사용하므로, progressive 활성화 시에도 `_calc_trail_trigger_price(best_price, entry, k_effective)` 호출 시 k 값을 동적으로 전달해야 함.

```python
# bot.py _update_exchange_trail (line ~500-ish)
k_cur = self._get_effective_trail_k(best_pnl)  # NEW helper
trigger = self._calc_trail_trigger_price(best, entry, k_cur, atr)
```

---

## 6. Unit Tests (≥6 cases)

```python
# scripts/tests/test_progressive_trail.py
def test_progressive_disabled_equals_baseline():
    # enabled=false 이면 k=trail_K fixed 동작

def test_progressive_early_phase_uses_base_k():
    # best_pnl < threshold → k=2.5

def test_progressive_late_phase_uses_post_k():
    # best_pnl >= threshold → k=0.5

def test_progressive_threshold_boundary():
    # best_pnl == threshold 경계

def test_progressive_long_vs_short_symmetry():
    # LONG과 SHORT에서 동일 로직 대칭 적용

def test_progressive_atr_zero_safety():
    # atr=0 or NaN 방어
```

---

## 7. Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| tkT=0.5 tight trail이 1:1 실거래에서 whipsaw 증가 | Bootstrap 3-day pos% + p5 + 30일 LIVE 수집 |
| Baton-touch trail 수식이 K 변경에 비대칭 | bot.py에서 dynamic k 적용 + 단위 테스트 검증 |
| Neighborhood 8+ 미달 — sharp peak | STOP, 추가 exit 재설계 불가로 판정 |
| 2-half delta consistency 실패 | 실거래 live 활성 연기, 추가 데이터 수집 |
| BUG#61 재발 가능성 (progressive 활성 시 새 edge case) | signals.py 수식과 bot.py 수식 100% 동일성 pytest fix |

---

## 8. Deliverables Checklist

- [ ] `scripts/analysis/progressive_trail_full_validation.py` (~600 lines)
- [ ] `results/progressive_trail_validation_{stamp}.json`
- [ ] `scripts/production/c1_breakout/signals.py` — check_exit progressive branch
- [ ] `config/c1_breakout_config.yaml` — progressive_trail section
- [ ] `scripts/production/c1_breakout/bot.py` — dynamic k in trail update
- [ ] `scripts/tests/test_progressive_trail.py` (6+ cases)
- [ ] `docs/03-analysis/progressive_trail.analysis.md`
- [ ] `docs/04-report/features/progressive_trail.report.md`
- [ ] `memory/progressive_trail_20260421.md`
- [ ] CLAUDE.md version history entry (v4.8.0 예정)

---

## 9. Execution Order

1. **Day 1 AM**: Script scaffolding + Tests 1~4 (full, strict WF, OOS split, 3-way)
2. **Day 1 PM**: Tests 5~6 (bootstrap, bootstrap_relative)
3. **Day 2 AM**: Tests 7~9 (neighborhood, consistency, MC, fold2)
4. **Day 2 PM**: GO 판정 → Design 갱신
5. **Day 3 AM**: Production code (signals/bot/config) + unit tests
6. **Day 3 PM**: Final gap-analyze + Report + Memory

**각 Day 종료 시 commit + push**.
