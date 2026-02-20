#!/bin/bash
# Stop hook — Remind Claude to save context when session ends
# Fires when Claude finishes responding

cat <<'ENDJSON'
{
  "hookSpecificOutput": {
    "hookEventName": "Stop",
    "additionalContext": "Session ending. If significant work was done: (1) Update Serena memory if new lessons learned (write_memory). (2) Update MEMORY.md if new common mistakes discovered. (3) Update CLAUDE.md Version History if production code changed."
  }
}
ENDJSON

exit 0
