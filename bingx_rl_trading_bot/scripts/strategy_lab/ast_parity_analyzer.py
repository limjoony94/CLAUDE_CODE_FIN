"""AST Parity Analyzer — Gate 1 readiness via static code analysis.

Compare LIVE bot code vs BT script:
  1. Enumerate LIVE methods touching position state (_on_fill, _replace_*, force_close_*, check_halt*, balance update)
  2. For each LIVE method, search BT for explicit analog (function call, comparable logic block)
  3. Missing analog → Gate 1 FAIL signal

This is the self-validation core: feeding the system R26 BT + LIVE code (without memory hints)
should detect re-arm/marketable LIMIT/halt/funding/balance compounding gaps automatically.
"""
import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ============================================================================
# Live behavior signatures — what to look for in LIVE bot code
# ============================================================================

LIVE_BEHAVIORS = {
    'level_rearm_after_fill': {
        'live_signatures': [
            r'_replace_grid_level',
            r'_replace_level',
            r'replace.*after.*fill',
            r'level\.filled\s*=\s*False',
        ],
        'bt_signatures': [  # what BT code should show analog
            r'(buy|sell)_filled\[\w+\]\s*=\s*False',
            r'level\.filled\s*=\s*False',
            r're_arm|rearm|replace_level',
        ],
        'description': 'TP/SL fill 후 같은 grid level 재배치 (re-arm)',
    },
    'marketable_limit_taker_fill': {
        'live_signatures': [
            r'create_limit_(buy|sell)_order.*positionSide',
        ],
        'bt_signatures': [
            r'open\[\w+\]\s*[<>]\s*lvl|level',  # marketable check on bar open
            r'taker_fric.*open',
            r'marketable',
        ],
        'description': 'LIMIT @ P 가격이 market price 반대편이면 immediate taker fill',
    },
    'halt_session_anchor_equity': {
        'live_signatures': [
            r'check_halts',
            r'halt_daily_loss_pct',
            r'start_capital',
            r'equity.*halt|halt.*equity',
        ],
        'bt_signatures': [
            r'session_anchor|halt_anchor|halt_triggered',
            r'cum_loss.*halt|halt.*cum_loss',
            r'equity_pct|equity_usd.*halt',
        ],
        'description': 'Session-cum 손실 ≥ halt threshold 시 force close + sys.exit',
    },
    'funding_fee_drag': {
        'live_signatures': [
            r'funding_rate|funding_fee',
            r'funding.*8h|8h.*funding',
        ],
        'bt_signatures': [
            r'cum_funding|funding_drag|funding_pct',
            r'funding_per_8h|funding_8h',
        ],
        'description': '8h마다 open notional × funding rate drag',
    },
    'balance_compounding_per_level': {
        'live_signatures': [
            r'auto_size_from_balance',
            r'notional_callback',
            r'_compute_per_level_notional',
            r'balance.*per_level|per_level.*balance',
        ],
        'bt_signatures': [
            r'apply_pnl_to_balance|update.*balance',
            r'per_level.*recompute|per_level_now',
            r'balance\s*[+\-]=',
        ],
        'description': 'balance 변동 시 per_level_notional 자동 재계산',
    },
    'position_sl_exchange_stop_market': {
        'live_signatures': [
            r'STOP_MARKET',
            r'per_position_stop_loss',
            r'sl_order_id',
        ],
        'bt_signatures': [
            r'sl_price|sl_pct',
            r'per_pos_sl|PER_POS_SL',
            r'stop.*market.*sl',
        ],
        'description': 'Exchange-side per-position STOP_MARKET (intra-bar SL)',
    },
}


# ============================================================================
# Code analysis
# ============================================================================

def collect_python_source(path: Path) -> str:
    """Concatenate all .py files in a directory (or read single file)."""
    if path.is_file():
        return path.read_text(encoding='utf-8')
    out = []
    for p in sorted(path.rglob('*.py')):
        out.append(f'\n# ===== {p.name} =====\n')
        try:
            out.append(p.read_text(encoding='utf-8'))
        except Exception as e:
            out.append(f'# ERROR reading: {e}')
    return '\n'.join(out)


def detect_signatures(code: str, signatures: list[str]) -> list[str]:
    """Return list of matched signatures."""
    found = []
    for sig in signatures:
        if re.search(sig, code, re.IGNORECASE):
            found.append(sig)
    return found


def parity_audit(live_path: Path, bt_path: Path) -> dict:
    """Compare LIVE code vs BT code for the 6 LIVE_BEHAVIORS."""
    live_code = collect_python_source(live_path)
    bt_code = collect_python_source(bt_path)

    findings = []
    n_present_in_live = 0
    n_missing_in_bt = 0

    for behavior_name, spec in LIVE_BEHAVIORS.items():
        live_matches = detect_signatures(live_code, spec['live_signatures'])
        bt_matches = detect_signatures(bt_code, spec['bt_signatures'])

        present_in_live = bool(live_matches)
        present_in_bt = bool(bt_matches)

        # Discrepancy: LIVE has it but BT doesn't
        gap = present_in_live and not present_in_bt

        finding = {
            'behavior': behavior_name,
            'description': spec['description'],
            'present_in_live': present_in_live,
            'live_matches': live_matches,
            'present_in_bt': present_in_bt,
            'bt_matches': bt_matches,
            'gap_detected': gap,
        }
        findings.append(finding)

        if present_in_live:
            n_present_in_live += 1
        if gap:
            n_missing_in_bt += 1

    # Score: fraction of LIVE behaviors that have BT analog
    if n_present_in_live > 0:
        coverage = (n_present_in_live - n_missing_in_bt) / n_present_in_live
    else:
        coverage = 1.0  # nothing to cover

    if n_missing_in_bt >= 3:
        verdict = 'GATE1_FAIL — 3+ LIVE behaviors not modeled in BT'
        gate1_score = 0.0
    elif n_missing_in_bt >= 1:
        verdict = f'GATE1_PARTIAL — {n_missing_in_bt} LIVE behavior(s) missing in BT'
        gate1_score = 0.5
    else:
        verdict = 'GATE1_OK — all LIVE behaviors have BT analog'
        gate1_score = 1.0

    return {
        'live_path': str(live_path),
        'bt_path': str(bt_path),
        'n_behaviors_checked': len(LIVE_BEHAVIORS),
        'n_present_in_live': n_present_in_live,
        'n_missing_in_bt': n_missing_in_bt,
        'coverage_pct': coverage * 100,
        'gate1_score': gate1_score,
        'verdict': verdict,
        'findings': findings,
    }


def report(audit: dict, verbose: bool = True):
    print('=' * 100)
    print(f'AST Parity Audit')
    print('=' * 100)
    print(f'LIVE: {audit["live_path"]}')
    print(f'BT:   {audit["bt_path"]}')
    print()
    print(f'Behaviors checked:        {audit["n_behaviors_checked"]}')
    print(f'Present in LIVE:          {audit["n_present_in_live"]}')
    print(f'Missing analog in BT:     {audit["n_missing_in_bt"]}')
    print(f'Coverage:                 {audit["coverage_pct"]:.1f}%')
    print(f'Gate 1 score:             {audit["gate1_score"]:.2f}')
    print(f'Verdict:                  {audit["verdict"]}')
    print()
    if verbose:
        print('Findings:')
        for f in audit['findings']:
            mark = '🔴 GAP ' if f['gap_detected'] else ('✅' if f['present_in_live'] and f['present_in_bt'] else '— ')
            print(f'  {mark} {f["behavior"]}')
            print(f'      {f["description"]}')
            print(f'      LIVE matches: {f["live_matches"] or "(none)"}')
            print(f'      BT matches:   {f["bt_matches"] or "(none)"}')
            print()


if __name__ == '__main__':
    # Self-validation: R26 LIVE vs ORIGINAL R26 BT (round26_grid_ranging.py)
    project_root = Path(__file__).resolve().parent.parent.parent
    live_dir = project_root / 'scripts' / 'production' / 'r26_grid'
    bt_path = project_root / 'scripts' / 'analysis' / 'round26_grid_ranging.py'

    if not live_dir.exists():
        print(f'LIVE path not found: {live_dir}')
    elif not bt_path.exists():
        print(f'BT path not found: {bt_path}')
    else:
        audit = parity_audit(live_dir, bt_path)
        report(audit)
