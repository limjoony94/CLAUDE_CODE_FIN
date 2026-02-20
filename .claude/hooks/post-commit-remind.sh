#!/bin/bash
# PostToolUse hook — After git commit, remind about CLAUDE.md and version updates

INPUT=$(cat)

# Extract command (single-line python for MSYS2 compat)
COMMAND=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('toolInput',{}).get('command',''))" 2>/dev/null)

# Only trigger on git commit commands
if echo "$COMMAND" | grep -q "git commit"; then
    DIFF=$(cd "C:/Users/J/OneDrive/CLAUDE_CODE_FIN" && git diff HEAD~1 --name-only 2>/dev/null)

    if echo "$DIFF" | grep -q "scripts/production/pattern_5m/\|config/pattern_5m_config"; then
        echo '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"Production files were committed. Consider: (1) Update CLAUDE.md Version History if version changed. (2) Run /run-tests to verify. (3) Check if bot needs restart for changes to take effect."}}'
    fi
fi

exit 0
