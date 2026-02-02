#!/usr/bin/env python3
"""
Find syntax error in production bot
"""
import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
bot_file = PROJECT_ROOT / 'scripts' / 'production' / 'opportunity_gating_bot_4x.py'

try:
    with open(bot_file, 'r', encoding='utf-8') as f:
        code = f.read()

    ast.parse(code)
    print("✅ No syntax errors found!")

except SyntaxError as e:
    print(f"❌ SyntaxError found:")
    print(f"   File: {e.filename}")
    print(f"   Line: {e.lineno}")
    print(f"   Offset: {e.offset}")
    print(f"   Text: {e.text}")
    print(f"   Message: {e.msg}")

    # Show context
    print(f"\n📄 Context (lines {max(1, e.lineno-5)}-{e.lineno+5}):")
    with open(bot_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(max(0, e.lineno-6), min(len(lines), e.lineno+5)):
            marker = ">>>" if i+1 == e.lineno else "   "
            print(f"{marker} {i+1:4d}: {lines[i].rstrip()}")
