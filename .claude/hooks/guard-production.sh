#!/bin/bash
# Claude Code PreToolUse Hook — Guard production files
# Warns (but doesn't block) when editing production bot files

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('toolInput',{}).get('file_path',''))" 2>/dev/null)

if echo "$FILE_PATH" | grep -qE "scripts/production/c1_breakout/|scripts/production/c1_breakout_bot\.py|config/c1_breakout_config\.yaml"; then
  echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","message":"WARNING: Editing live C1 Breakout production file. Bot is actively trading. Ensure changes are tested and approved.","decision":"allow"}}' >&2
fi

exit 0
