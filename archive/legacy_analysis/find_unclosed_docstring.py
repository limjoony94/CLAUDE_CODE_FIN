#!/usr/bin/env python3
"""
Find unclosed docstrings
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
bot_file = PROJECT_ROOT / 'scripts' / 'production' / 'opportunity_gating_bot_4x.py'

with open(bot_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

triple_quote_count = 0
in_docstring = False
docstring_start = None

print("📊 Tracking docstring opens/closes:\n")

for i, line in enumerate(lines, 1):
    if '"""' in line:
        count = line.count('"""')
        triple_quote_count += count

        # Determine state
        if not in_docstring:
            in_docstring = True
            docstring_start = i
            state = "OPEN"
        else:
            in_docstring = False
            state = "CLOSE"

        print(f"Line {i:4d}: {'>>>  ' if in_docstring else '     '}{line.strip()[:60]}")
        print(f"           Count: {count}, State: {state}, Running total: {triple_quote_count}")

if triple_quote_count % 2 != 0:
    print(f"\n❌ UNMATCHED: Total triple-quote count is ODD ({triple_quote_count})")
    print(f"   Last unclosed docstring started at line: {docstring_start}")
else:
    print(f"\n✅ All triple quotes matched (total: {triple_quote_count})")
