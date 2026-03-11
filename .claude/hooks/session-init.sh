#!/bin/bash
# Claude Code Session Init Hook — Pattern 5m Trading Bot
# Outputs project context as additionalContext for Claude

cat <<'ENDJSON'
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "Pattern 5m Bot v1.55.0 project. Serena project: bingx_rl_trading_bot. Key memories: project_state_v1_28_42, research_protocol_standard, common_pitfalls_and_lessons. Custom commands: /bot-status, /check-live, /scan-patterns, /research-template, /run-tests, /wf-validate, /daily-report, /diagnose, /emergency-stop, /deploy-patterns. Bot is LIVE — do not modify production files without approval."
  }
}
ENDJSON

exit 0
