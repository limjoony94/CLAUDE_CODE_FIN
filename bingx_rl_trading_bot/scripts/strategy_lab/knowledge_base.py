"""Strategy Lab Knowledge Base — 28+ rounds × 4 alpha families.

Encoded from memory/* learnings (M3 rounds R20-R41, Path-B R1-R23, DeFi-R1, R5, R8, R26, C1).
Each round is a (mechanism, substrate, execution) point with in_sample / OOS metrics + falsification reason.

Source memory files:
  - m3_28round_comprehensive_final_20260428.md
  - final_envelope_ceiling_20260429.md
  - path_b_*.md, l2_microstructure_falsified.md
  - r26_postmortem_20260501.md, c1_breakout_postmortem_20260427.md
  - Various round-specific memos

This is the empirical prior. ANY new strategy proposal is cross-checked against these.
"""
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Taxonomy — mechanism / substrate / execution dimensions
# ============================================================================

MECHANISM_CATEGORIES = [
    'trend_following',         # C1, R20, breakout, MA cross
    'mean_reversion',          # R23 VWAP, R8 Donchian fade
    'pattern_based',           # R21 reversal, R24 ICT, candle patterns
    'grid_oscillation',        # R26
    'funding_carry',           # R5, R13
    'microstructure',          # L2 OBI/OFI/Kyle-λ
    'cross_sectional',         # Path-B XS momentum
    'defi_yield_rotation',     # DeFi-R1
    'volatility_compression',  # R37 NR7+BB
    'session_time_filter',     # R28
    'arbitrage',               # basis, funding spread
]

SUBSTRATE_CATEGORIES = [
    'BTC_perp_OHLCV_5m',
    'BTC_perp_OHLCV_15m',
    'BTC_perp_OHLCV_1h',
    'BTC_perp_OHLCV_daily',
    'BTC_perp_L2_orderbook',
    'BTC_perp_trade_tape',
    'BTC_funding_only',
    'BTC_spot_perp_pair',
    'multi_coin_crypto_8',     # cross-sectional ~8 coins
    'multi_coin_crypto_30',    # broader
    'defi_L2_pools',
    'L2_microstructure_tick',
]

EXECUTION_CATEGORIES = [
    'taker_only',                # market in/out
    'maker_entry_taker_exit',    # LIMIT in, MARKET out
    'maker_full_cycle',          # LIMIT in/out (R8 maker, R26 TP)
    'maker_taker_mixed',         # R26: maker entry/TP, taker SL
    'spot_perp_hedge',           # cash-and-carry
    'cross_exchange',            # geographic arbitrage
    'pool_rotation',             # DeFi monthly rebal
]

FRICTION_PROFILES = {
    'BingX_retail_taker':    {'rt_pct': 0.10, 'maker_pct': 0.04, 'funding_8h': 0.01, 'slippage_typ': 0.02},
    'BingX_retail_maker':    {'rt_pct': 0.04, 'maker_pct': 0.04, 'funding_8h': 0.01, 'slippage_typ': 0.00},
    'L2_arbitrum_swap':      {'rt_pct': 0.40, 'maker_pct': None, 'funding_8h': 0.00, 'slippage_typ': 0.05},  # gas+spread
    'institutional_taker':   {'rt_pct': 0.001, 'maker_pct': 0.0005, 'funding_8h': 0.01, 'slippage_typ': 0.005},  # for reference only
}

VERDICT_TYPES = ['DEPLOYABLE', 'L2_PASS_NOT_L3', 'L1_ONLY', 'FALSIFIED', 'VACUOUS']


@dataclass
class RoundEvidence:
    """One round of empirical evidence — used as prior for new proposals."""
    round_id: str
    date: str                      # YYYY-MM-DD
    mechanism: str                 # one of MECHANISM_CATEGORIES
    substrate: str                 # one of SUBSTRATE_CATEGORIES
    execution: str                 # one of EXECUTION_CATEGORIES
    friction_profile: str          # one of FRICTION_PROFILES keys
    in_sample_metric: dict         # {avg_gross_pct, daily_pct, wr, n_trades, ...}
    oos_metric: Optional[dict]     # WF / multi-window / 3-way metric
    verdict: str                   # one of VERDICT_TYPES
    falsification_reason: Optional[str]
    capital_assumption_usd: float  # what capital scale tested
    memo_ref: str                  # memory/* path
    notes: str = ''


# ============================================================================
# 28+ rounds encoded — empirical prior
# ============================================================================

KNOWLEDGE_BASE = [
    # ====== R5 — DEPLOYABLE (only one) ======
    RoundEvidence(
        round_id='R5_single_carry',
        date='2026-04-29',
        mechanism='funding_carry',
        substrate='BTC_spot_perp_pair',
        execution='spot_perp_hedge',
        friction_profile='BingX_retail_maker',
        in_sample_metric={'apy_pct': 3.28, 'usd_per_yr_on_500': 16.4},
        oos_metric={'apy_pct': 3.28, 'verdict': 'PASS', 'sharpe': None},
        verdict='DEPLOYABLE',
        falsification_reason=None,
        capital_assumption_usd=500,
        memo_ref='final_envelope_ceiling_20260429.md',
        notes='Only deployable result across 28 rounds. ~$49/yr at $1500 capital.',
    ),

    # ====== R8 — Donchian baseline (taker positive) ======
    RoundEvidence(
        round_id='R8_donchian_1h',
        date='2026-04-30',
        mechanism='trend_following',
        substrate='BTC_perp_OHLCV_1h',
        execution='taker_only',
        friction_profile='BingX_retail_taker',
        in_sample_metric={'avg_gross_pct': 0.04, 'wr': None},
        oos_metric={'verdict': 'NEAR_BREAK_EVEN'},
        verdict='L1_ONLY',
        falsification_reason='avg_gross +0.04% < friction 0.07%',
        capital_assumption_usd=500,
        memo_ref='round25_maker_adverse_selection.md',
        notes='Original taker baseline before R25 maker variant',
    ),

    # ====== R13 — multi-coin carry diluted ======
    RoundEvidence(
        round_id='R13_multicoin_carry',
        date='2026-04-30',
        mechanism='funding_carry',
        substrate='multi_coin_crypto_8',
        execution='spot_perp_hedge',
        friction_profile='BingX_retail_maker',
        in_sample_metric={'apy_pct': 2.85},
        oos_metric={'apy_pct': 2.85, 'verdict': 'L2_FAIL_T4_MAGNITUDE'},
        verdict='L2_PASS_NOT_L3',
        falsification_reason='Equal-weight averages below R5 single-coin (3.28% > 2.85%). Friction-as-fraction-of-position-size dominates.',
        capital_assumption_usd=1500,
        memo_ref='path_b_r13_multicoin_dilutes.md',
        notes='Equal-weight dilutes. Dispersion-weighted untested.',
    ),

    # ====== R21 — best individual M3 mechanism ======
    RoundEvidence(
        round_id='M3_R21_pattern_reversal',
        date='2026-04-28',
        mechanism='pattern_based',
        substrate='BTC_perp_OHLCV_5m',
        execution='taker_only',
        friction_profile='BingX_retail_taker',
        in_sample_metric={'avg_gross_pct': 0.010, 'n_trades': None},
        oos_metric={'verdict': 'L2_FAIL'},
        verdict='L1_ONLY',
        falsification_reason='avg_gross +0.010% × 7 = +0.07% × hr-1, < taker friction 0.07% RT. 20× gap to strict criterion +0.20%.',
        capital_assumption_usd=500,
        memo_ref='m3_28round_comprehensive_final_20260428.md',
        notes='Best single mechanism in 28-round comprehensive. Pattern reversal at extreme + structural exit.',
    ),

    # ====== R24 — ICT anti-edge ======
    RoundEvidence(
        round_id='R24_ict_liquidity_sweep',
        date='2026-04-30',
        mechanism='pattern_based',
        substrate='BTC_perp_OHLCV_1h',
        execution='taker_only',
        friction_profile='BingX_retail_taker',
        in_sample_metric={'avg_gross_pct': -0.05, 'n_trades': 541, 'wr': 0.357, 'r_to_r': 1.43},
        oos_metric={'verdict': 'WORSE_THAN_RANDOM', 'cum_pct': -50, 'random_p95': -42},
        verdict='FALSIFIED',
        falsification_reason='avg_gross negative pre-friction. Pattern-based strategy systematically anti-selected (publication bias). 0/3 HARD, 4/11 PASS.',
        capital_assumption_usd=500,
        memo_ref='round24_ict_anti_edge.md',
        notes='TradingView SMC publication-bias evidence',
    ),

    # ====== R25 — maker-entry adverse selection ======
    RoundEvidence(
        round_id='R25_maker_entry',
        date='2026-04-30',
        mechanism='trend_following',
        substrate='BTC_perp_OHLCV_1h',
        execution='maker_entry_taker_exit',
        friction_profile='BingX_retail_maker',
        in_sample_metric={'avg_gross_pct': -0.236},
        oos_metric={'verdict': 'FALSIFIED'},
        verdict='FALSIFIED',
        falsification_reason='Maker entry on breakout = adverse selection (Glosten-Milgrom). Limit-buy fills only on failed breakouts. Universal corollary: ALL momentum/breakout 28 prior rounds degrade with maker-entry.',
        capital_assumption_usd=500,
        memo_ref='round25_maker_adverse_selection.md',
        notes='Mean-reversion / grid trading would be different class.',
    ),

    # ====== R26 — Grid SHELVED (this session) ======
    RoundEvidence(
        round_id='R26_grid_ranging',
        date='2026-05-01',
        mechanism='grid_oscillation',
        substrate='BTC_perp_OHLCV_1h',
        execution='maker_taker_mixed',
        friction_profile='BingX_retail_maker',
        in_sample_metric={'apy_pct_BT': 169.5, 'daily_BT': 0.51},
        oos_metric={'verdict': 'L3_FAIL_BT_LIVE_PARITY', 'multi_window_n20_pos': '3/20', 'sign_test_p': 0.0013},
        verdict='FALSIFIED',
        falsification_reason='BT-LIVE parity bug (M1-M6: re-arm 누락, marketable LIMIT, halt missing, intra-1h fill, funding 무시, balance compounding 무시). LIVE 14d -12.86% vs BT +0.51%/day. Postmortem with 5-gate prevention.',
        capital_assumption_usd=500,
        memo_ref='r26_postmortem_20260501.md',
        notes='SHELVED 2026-05-01. Same pattern as C1 (BT positive → LIVE negative).',
    ),

    # ====== C1 — Breakout SHELVED ======
    RoundEvidence(
        round_id='C1_breakout_v26',
        date='2026-04-27',
        mechanism='trend_following',
        substrate='BTC_perp_OHLCV_15m',
        execution='maker_taker_mixed',
        friction_profile='BingX_retail_maker',
        in_sample_metric={'apy_pct_BT': 169.5, 'daily_BT': 0.51, 'wf_passes': '5/5'},
        oos_metric={'verdict': 'L3_FAIL_BT_LIVE_PARITY', 'live_14d_pct': -12.86},
        verdict='FALSIFIED',
        falsification_reason='Channel breakout. LIVE -12.86%/14d vs BT P5. D3 distribution check found foundation problem (BT model not representing LIVE market). Postmortem identified BT-LIVE parity gap.',
        capital_assumption_usd=500,
        memo_ref='c1_breakout_postmortem_20260427.md',
        notes='SHELVED 2026-04-27. First instance of BT-positive → LIVE-negative pattern.',
    ),

    # ====== L2 microstructure 4 features ======
    RoundEvidence(
        round_id='L2_microstructure_OBI_OFI_KyleLambda_QueueDepletion',
        date='2026-04-30',
        mechanism='microstructure',
        substrate='BTC_perp_L2_orderbook',
        execution='taker_only',
        friction_profile='BingX_retail_taker',
        in_sample_metric={'avg_gross_range': '[+0.0008%, +0.0024%]', 'hit_rate_F3_F4': '[0.529, 0.549]'},
        oos_metric={'verdict': 'FALSIFIED', 'pass_rate': '0/4'},
        verdict='FALSIFIED',
        falsification_reason='avg_gross 30-90× below 0.07% taker friction. Predictive signal exists but magnitude insufficient. Substrate change does NOT lift edge above friction.',
        capital_assumption_usd=500,
        memo_ref='l2_microstructure_falsified.md',
        notes='18h L2 sample. 4 distinct features. 27 mechanisms × 5 substrates evidence convergence.',
    ),

    # ====== Path-B R1 — XS momentum ======
    RoundEvidence(
        round_id='PathB_R1_xs_momentum',
        date='2026-04-29',
        mechanism='cross_sectional',
        substrate='multi_coin_crypto_8',
        execution='maker_full_cycle',
        friction_profile='BingX_retail_maker',
        in_sample_metric={'weekly_pct': 0.13, 'gross_post_friction': 0.025},
        oos_metric={'wf_passes': '3/5', 'bootstrap_pos_rate': 0.484, 'verdict': 'L2_BORDERLINE'},
        verdict='L2_PASS_NOT_L3',
        falsification_reason='WF 3/5 PASS, bootstrap 48.4% borderline FAIL, train/test sign FAIL. ~$45/yr if deployed.',
        capital_assumption_usd=1500,
        memo_ref='path_b_synthesis_borderline.md',
        notes='First round with edge > friction in 28 attempts. Borderline.',
    ),

    # ====== DeFi-R1 ======
    RoundEvidence(
        round_id='DeFi_R1_yield_rotation',
        date='2026-04-29',
        mechanism='defi_yield_rotation',
        substrate='defi_L2_pools',
        execution='pool_rotation',
        friction_profile='L2_arbitrum_swap',
        in_sample_metric={'apy_gross_pct': 4.92, 'apy_net_pct': 1.77, 'sharpe': 1.52, 'max_dd_pct': 0.95},
        oos_metric={'wf_passes': '3/5', 'bootstrap': 0.64, 't4_magnitude': 'FAIL'},
        verdict='L2_PASS_NOT_L3',
        falsification_reason='4/5 PASS, T4 magnitude FAIL (1.77%/yr = $26/yr on $1500). Friction 64% of gross.',
        capital_assumption_usd=1500,
        memo_ref='final_envelope_ceiling_20260429.md',
        notes='Top-3 trailing 30d APY median, monthly rebalance. Sharpe excellent.',
    ),

    # ====== M3 R26 BB-squeeze (NOT same as R26 grid) ======
    RoundEvidence(
        round_id='M3_R26_bb_squeeze',
        date='2026-04-28',
        mechanism='volatility_compression',
        substrate='BTC_perp_OHLCV_5m',
        execution='taker_only',
        friction_profile='BingX_retail_taker',
        in_sample_metric={'avg_gross_pct': 0.005, 'wr': None},
        oos_metric={'verdict': 'NEGATIVE', 't_stat': -4.18},
        verdict='FALSIFIED',
        falsification_reason='t-test confirms statistically significantly negative. Volatility compression breakout fails on retail OHLCV.',
        capital_assumption_usd=500,
        memo_ref='m3_28round_comprehensive_final_20260428.md',
        notes='M3 round naming clash with grid R26 (different study).',
    ),

    # ====== R37 — Volatility compression NR7+BB squeeze ======
    RoundEvidence(
        round_id='M3_R37_compression_breakout',
        date='2026-04-29',
        mechanism='volatility_compression',
        substrate='BTC_perp_OHLCV_5m',
        execution='taker_only',
        friction_profile='BingX_retail_taker',
        in_sample_metric={'avg_gross_pct': -0.014},
        oos_metric={'wf_passes': '1/5', 'bootstrap_pos_rate': 0.413, 'train_test_split': 'FAIL', 'verdict': 'L2_FAIL'},
        verdict='FALSIFIED',
        falsification_reason='6 structurally distinct mechanism classes all fail same envelope. Variance-conditional entry insufficient.',
        capital_assumption_usd=500,
        memo_ref='m3_r37_compression_6th_negative.md',
        notes='6th OOS negative. Decisive evidence envelope limit.',
    ),

    # ====== R38 — VWAP vacuous ======
    RoundEvidence(
        round_id='M3_R38_vwap_reversion',
        date='2026-04-29',
        mechanism='mean_reversion',
        substrate='BTC_perp_OHLCV_5m',
        execution='taker_only',
        friction_profile='BingX_retail_taker',
        in_sample_metric={'n_trades': 4, 'trades_per_day': 0.006},
        oos_metric={'verdict': 'VACUOUS'},
        verdict='VACUOUS',
        falsification_reason='Only 4 signals/720d (0.006/day). Vacuous test. NOT a fail/pass — frequency gate not met.',
        capital_assumption_usd=500,
        memo_ref='m3_r38_inconclusive_vacuous.md',
        notes='Process lesson: pre-reg must include min signal frequency ≥0.5/day.',
    ),

    # ====== R41 — MACD minimal arithmetic falsification ======
    RoundEvidence(
        round_id='M3_R41_macd_minimal',
        date='2026-04-29',
        mechanism='trend_following',
        substrate='BTC_perp_OHLCV_5m',
        execution='taker_only',
        friction_profile='BingX_retail_taker',
        in_sample_metric={'avg_gross_pct': 0.034, 'n_trades': 2760},
        oos_metric={'verdict': 'ARITHMETIC_FAIL', 'inequality': '+0.034% < +0.07% friction'},
        verdict='FALSIFIED',
        falsification_reason='avg_gross +0.034% × n=2760 < friction +0.07%. Arithmetic falsification (not statistical).',
        capital_assumption_usd=500,
        memo_ref='m3_28round_comprehensive_final_20260428.md',
        notes='New evidence type: arithmetic inequality. Closes M3 OHLCV envelope.',
    ),

    # ====== Trade-tape envelope (3 rounds combined) ======
    RoundEvidence(
        round_id='TradeTape_R1_R2_continuation_fade',
        date='2026-04-29',
        mechanism='microstructure',
        substrate='BTC_perp_trade_tape',
        execution='taker_only',
        friction_profile='BingX_retail_taker',
        in_sample_metric={'avg_gross_range': '[+0.010%, +0.050%]'},
        oos_metric={'verdict': 'FALSIFIED', 'env': '3 ROUNDS'},
        verdict='FALSIFIED',
        falsification_reason='3 independent rounds (continuation, fade, R41 OHLCV) all avg_gross < friction 0.07%. Bar-level retail BTC perp directional mechanisms = arithmetically falsified envelope.',
        capital_assumption_usd=500,
        memo_ref='trade_tape_envelope_closed_friction_floor.md',
        notes='Closes trade-tape envelope.',
    ),
]


# ============================================================================
# Aggregations + helper functions
# ============================================================================

def get_tested_combinations():
    """Return set of (mechanism, substrate, execution) tuples already tested."""
    return {(r.mechanism, r.substrate, r.execution) for r in KNOWLEDGE_BASE}


def get_falsified_combinations():
    """Return tested combinations with FALSIFIED verdict."""
    return {(r.mechanism, r.substrate, r.execution)
            for r in KNOWLEDGE_BASE if r.verdict == 'FALSIFIED'}


def get_deployable_rounds():
    """Currently deployable strategies."""
    return [r for r in KNOWLEDGE_BASE if r.verdict == 'DEPLOYABLE']


def get_l2_borderline():
    """L2-pass strategies that didn't reach L3 (paper trade candidates)."""
    return [r for r in KNOWLEDGE_BASE if r.verdict == 'L2_PASS_NOT_L3']


def empirical_friction_floor_pct():
    """The binding constraint per memory: 0.07% taker RT for BingX retail."""
    return FRICTION_PROFILES['BingX_retail_taker']['rt_pct']


def envelope_summary():
    """Summary of evidence accumulated."""
    n = len(KNOWLEDGE_BASE)
    n_falsified = sum(1 for r in KNOWLEDGE_BASE if r.verdict == 'FALSIFIED')
    n_deployable = sum(1 for r in KNOWLEDGE_BASE if r.verdict == 'DEPLOYABLE')
    n_l2 = sum(1 for r in KNOWLEDGE_BASE if r.verdict == 'L2_PASS_NOT_L3')
    n_l1 = sum(1 for r in KNOWLEDGE_BASE if r.verdict == 'L1_ONLY')
    n_vacuous = sum(1 for r in KNOWLEDGE_BASE if r.verdict == 'VACUOUS')
    return {
        'n_rounds_encoded': n,
        'falsified': n_falsified,
        'deployable': n_deployable,
        'l2_pass_not_l3': n_l2,
        'l1_only': n_l1,
        'vacuous': n_vacuous,
        'mechanisms_tested': len({r.mechanism for r in KNOWLEDGE_BASE}),
        'substrates_tested': len({r.substrate for r in KNOWLEDGE_BASE}),
        'friction_floor_pct': empirical_friction_floor_pct(),
        'capital_binding_evidence': 'retail $500-1500 envelope, friction-as-fraction dominates',
    }


if __name__ == '__main__':
    import json
    print('=' * 80)
    print('Strategy Lab Knowledge Base — Empirical Prior')
    print('=' * 80)
    summ = envelope_summary()
    for k, v in summ.items():
        print(f'  {k}: {v}')
    print()
    print('Tested (mechanism, substrate, execution) combinations:')
    for combo in sorted(get_tested_combinations()):
        print(f'  {combo}')
