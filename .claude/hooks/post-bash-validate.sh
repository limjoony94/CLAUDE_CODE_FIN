#!/bin/bash
# PostToolUse hook — Detect test failures and research anomalies after Bash execution

INPUT=$(cat)

# Extract stdout from the tool result (single-line python for MSYS2 compat)
STDOUT=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('toolResult',{}).get('stdout','') or d.get('stdout',''))" 2>/dev/null)

CONTEXT=""

# Detect pytest failures (avoid false positive on "0 failed")
if echo "$STDOUT" | grep -qE "FAILED|ERRORS|[1-9][0-9]* failed"; then
    CONTEXT="${CONTEXT} TEST FAILURES DETECTED: Run /run-tests for details. Do NOT proceed with deployment or commits until tests pass."
fi

# Detect WF validation failures
if echo "$STDOUT" | grep -qi "WF.*FAIL\|fold.*FAIL\|OOS.*negative"; then
    CONTEXT="${CONTEXT} WF VALIDATION FAILURE: Strategy change does not pass walk-forward. Do NOT deploy. Review with /wf-validate."
fi

# Detect research anomalies (unrealistic returns — 5+ digit PnL)
if echo "$STDOUT" | grep -qE "PnL.*\+[0-9]{5,}%|return.*[0-9]{5,}%"; then
    CONTEXT="${CONTEXT} RESEARCH ANOMALY: Unrealistically high returns detected. Check for look-ahead bias, missing leverage, or classification bug."
fi

if [ -n "$CONTEXT" ]; then
    echo "{\"hookSpecificOutput\":{\"hookEventName\":\"PostToolUse\",\"additionalContext\":\"${CONTEXT}\"}}"
fi

exit 0
