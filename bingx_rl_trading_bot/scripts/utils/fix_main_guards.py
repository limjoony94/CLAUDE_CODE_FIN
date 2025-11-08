"""
Fix __main__ guards in production scripts that were missing them
"""
import re
from pathlib import Path

def fix_main_guard(file_path):
    """Add proper indentation after 'if __name__ == "__main__":' guard"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the line with 'if __name__ == "__main__":'
    guard_line_idx = None
    for i, line in enumerate(lines):
        if 'if __name__ == "__main__":' in line:
            guard_line_idx = i
            break

    if guard_line_idx is None:
        print(f"No __main__ guard found in {file_path.name}")
        return False

    # Check if the next non-empty line is already indented
    next_code_line = guard_line_idx + 1
    while next_code_line < len(lines) and not lines[next_code_line].strip():
        next_code_line += 1

    if next_code_line < len(lines):
        # Check if already properly indented
        if lines[next_code_line].startswith('    '):
            print(f"✅ {file_path.name} already properly indented")
            return False

        # Need to indent all lines after the guard
        print(f"🔧 Fixing indentation in {file_path.name}...")
        fixed_lines = lines[:guard_line_idx + 1]  # Keep everything up to and including guard

        for line in lines[guard_line_idx + 1:]:
            if line.strip():  # If line is not empty
                fixed_lines.append('    ' + line)
            else:
                fixed_lines.append(line)

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)

        print(f"✅ Fixed {file_path.name}")
        return True

    return False

# Fix backtest_hybrid_v4.py
scripts_dir = Path(__file__).parent.parent / "production"
files_to_fix = [
    "backtest_hybrid_v4.py",
]

print("=" * 80)
print("Fixing __main__ guard indentation")
print("=" * 80)

for filename in files_to_fix:
    file_path = scripts_dir / filename
    if file_path.exists():
        fix_main_guard(file_path)
    else:
        print(f"❌ File not found: {filename}")

print("\n✅ Done!")
