"""
Add __main__ guards to scripts that don't have them
"""
from pathlib import Path

def needs_guard(file_path):
    """Check if file needs __main__ guard"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Already has guard
    if 'if __name__ == "__main__":' in content or "if __name__ == '__main__':" in content:
        return False, None

    # Find where executable code starts (after imports and class/function definitions)
    lines = content.split('\n')

    # Find first line that looks like executable code (not comment, import, class, def)
    first_exec_line = None
    in_docstring = False
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track docstrings
        if '"""' in stripped or "'''" in stripped:
            in_docstring = not in_docstring
            continue

        if in_docstring:
            continue

        # Skip empty lines, comments, imports, class/function definitions
        if not stripped:
            continue
        if stripped.startswith('#'):
            continue
        if stripped.startswith('import ') or stripped.startswith('from '):
            continue
        if stripped.startswith('class ') or stripped.startswith('def '):
            # Skip until we find the end of the class/function
            continue

        # This looks like executable code
        if not stripped.startswith('@'):  # Not a decorator
            first_exec_line = i
            break

    if first_exec_line is None:
        return False, None  # No executable code found

    return True, first_exec_line

def add_guard(file_path, first_exec_line):
    """Add __main__ guard to file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Split into before and after
    before = lines[:first_exec_line]
    after = lines[first_exec_line:]

    # Create new content
    new_lines = before
    new_lines.append('\n')
    new_lines.append('if __name__ == "__main__":\n')

    # Indent all executable lines
    for line in after:
        if line.strip():  # Non-empty line
            new_lines.append('    ' + line)
        else:  # Empty line
            new_lines.append(line)

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"✅ Added guard to {file_path.name}")

# Process scripts
scripts_dir = Path(__file__).parent.parent / "production"
scripts_to_check = [
    "backtest_regime_specific_v5.py",
    "optimize_hybrid_thresholds.py",
    "test_ultraconservative.py",
]

print("=" * 80)
print("Adding __main__ guards to production scripts")
print("=" * 80)

for script_name in scripts_to_check:
    script_path = scripts_dir / script_name
    if not script_path.exists():
        print(f"❌ Not found: {script_name}")
        continue

    needs, first_line = needs_guard(script_path)
    if needs:
        print(f"\n🔧 Processing: {script_name}")
        print(f"   First executable line: {first_line}")
        add_guard(script_path, first_line)
    else:
        print(f"✅ Already has guard: {script_name}")

print("\n" + "=" * 80)
print("✅ Done!")
print("=" * 80)
