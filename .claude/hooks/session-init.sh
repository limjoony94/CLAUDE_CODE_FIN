#!/bin/bash
# Claude Code Session Init Hook — C1 Breakout v2.6
# Outputs project context as additionalContext for Claude

cat <<'ENDJSON'
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "C1 Breakout v2.6 (15m BTC) LIVE. Entry: scripts/production/c1_breakout_bot.py. Modules: scripts/production/c1_breakout/. Config: config/c1_breakout_config.yaml. State: results/c1_breakout_state.json. Log: logs/c1_breakout.log. Tests: scripts/tests/ (113 cases, coverage 71%). Docs: claudedocs/c1_breakout_v2_design.md, BUG_HISTORY.md, BACKTEST_LIVE_PARITY.md. Commands: /bot-status, /check-live, /daily-report, /diagnose, /emergency-stop, /run-tests, /research-template, /wf-validate, /deploy-patterns (note: /scan-patterns is DEPRECATED for C1). Bot is LIVE — do not modify production files without approval. Legacy (Pattern 5m/MAVS/CP/BTV/Volspike): archive/cleanup_20260418/."
  }
}
ENDJSON

exit 0
