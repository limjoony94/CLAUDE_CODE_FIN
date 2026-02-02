#!/usr/bin/env python3
"""
Binary search to find syntax error line
"""
import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
bot_file = PROJECT_ROOT / 'scripts' / 'production' / 'opportunity_gating_bot_4x.py'

with open(bot_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Binary search for the problematic line
left, right = 1, len(lines)
last_good = 0

while left <= right:
    mid = (left + right) // 2
    code = ''.join(lines[:mid])

    try:
        ast.parse(code)
        # This prefix parses OK
        last_good = mid
        left = mid + 1
    except SyntaxError:
        # This prefix has error
        right = mid - 1

print(f"✅ File parses correctly up to line: {last_good}")
print(f"❌ Syntax error introduced at or after line: {last_good + 1}")
print(f"\nLine {last_good + 1}:")
if last_good < len(lines):
    print(f"   {lines[last_good]}")
