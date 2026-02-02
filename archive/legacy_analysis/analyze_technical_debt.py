"""
Technical Debt Analysis Tool

비판적 분석을 통한 기술 부채 식별:
1. 설계 부채: 아키텍처, 모듈 의존성
2. 코드 부채: 중복 코드, 복잡도, 일관성
3. 테스트 부채: 테스트 커버리지
4. 인프라 부채: 설정, 로깅, 에러 핸들링
5. 혼잡도 부채: 불필요한 파일, 조직화
6. 중복 코드 부채: 같은 로직 반복
"""

import os
import ast
import re
from pathlib import Path
from collections import defaultdict, Counter
import hashlib

PROJECT_ROOT = Path(__file__).parent.parent


class TechnicalDebtAnalyzer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.python_files = []
        self.analysis_results = {
            'design_debt': {},
            'code_debt': {},
            'test_debt': {},
            'infrastructure_debt': {},
            'clutter_debt': {},
            'duplication_debt': {}
        }

    def scan_project(self):
        """Scan all Python files in project"""
        for py_file in self.project_root.rglob('*.py'):
            # Skip __pycache__ and .venv
            if '__pycache__' in str(py_file) or '.venv' in str(py_file):
                continue
            self.python_files.append(py_file)

        print(f"Found {len(self.python_files)} Python files")

    def analyze_design_debt(self):
        """Analyze architecture and design issues"""
        print("\n" + "=" * 80)
        print("1. DESIGN DEBT ANALYSIS")
        print("=" * 80)

        # Count files by directory
        dir_counts = defaultdict(int)
        for f in self.python_files:
            rel_path = f.relative_to(self.project_root)
            if len(rel_path.parts) > 1:
                top_dir = rel_path.parts[0]
                dir_counts[top_dir] += 1

        print("\nFile Distribution:")
        for dir_name, count in sorted(dir_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {dir_name}/: {count} files")

        # Identify versioned files (indicates design debt)
        versioned_files = defaultdict(list)
        for f in self.python_files:
            match = re.search(r'_v(\d+)', f.stem)
            if match:
                base_name = re.sub(r'_v\d+', '', f.stem)
                versioned_files[base_name].append((f, int(match.group(1))))

        if versioned_files:
            print("\n⚠️ VERSIONED FILES (Design Debt):")
            for base_name, versions in versioned_files.items():
                versions_sorted = sorted(versions, key=lambda x: x[1])
                print(f"\n  {base_name}:")
                for file_path, version in versions_sorted:
                    print(f"    v{version}: {file_path.name}")

            self.analysis_results['design_debt']['versioned_files'] = versioned_files

        # Check for multiple similar modules
        similar_modules = defaultdict(list)
        for f in self.python_files:
            if 'trading_env' in f.name:
                similar_modules['trading_env'].append(f)
            elif 'xgboost_trader' in f.name:
                similar_modules['xgboost_trader'].append(f)
            elif 'train_xgboost' in f.name:
                similar_modules['train_xgboost'].append(f)

        if similar_modules:
            print("\n⚠️ SIMILAR MODULES (Potential Duplication):")
            for module_type, files in similar_modules.items():
                if len(files) > 1:
                    print(f"\n  {module_type}: {len(files)} variations")
                    for f in files:
                        print(f"    - {f.name}")

    def analyze_code_debt(self):
        """Analyze code quality issues"""
        print("\n" + "=" * 80)
        print("2. CODE DEBT ANALYSIS")
        print("=" * 80)

        large_files = []
        complex_files = []
        long_functions = []

        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                # Check file size
                if len(lines) > 500:
                    large_files.append((py_file, len(lines)))

                # Parse AST for complexity
                try:
                    tree = ast.parse(content)

                    # Count functions and classes
                    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

                    # Check for long functions
                    for func in functions:
                        func_lines = func.end_lineno - func.lineno
                        if func_lines > 100:
                            long_functions.append((py_file, func.name, func_lines))

                    # High complexity indicator: many functions in one file
                    if len(functions) > 20:
                        complex_files.append((py_file, len(functions), len(classes)))

                except SyntaxError:
                    pass

            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")

        if large_files:
            print("\n⚠️ LARGE FILES (>500 lines):")
            for file_path, line_count in sorted(large_files, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {file_path.name}: {line_count} lines")

        if long_functions:
            print("\n⚠️ LONG FUNCTIONS (>100 lines):")
            for file_path, func_name, line_count in sorted(long_functions, key=lambda x: x[2], reverse=True)[:10]:
                print(f"  {file_path.name}::{func_name}(): {line_count} lines")

        if complex_files:
            print("\n⚠️ COMPLEX FILES (>20 functions):")
            for file_path, func_count, class_count in sorted(complex_files, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {file_path.name}: {func_count} functions, {class_count} classes")

    def analyze_test_debt(self):
        """Analyze test coverage"""
        print("\n" + "=" * 80)
        print("3. TEST DEBT ANALYSIS")
        print("=" * 80)

        # Find test files
        test_files = [f for f in self.python_files if 'test' in f.name.lower()]
        src_files = [f for f in self.python_files if 'src' in str(f)]
        script_files = [f for f in self.python_files if 'scripts' in str(f)]

        print(f"\nTest Files: {len(test_files)}")
        print(f"Source Files (src/): {len(src_files)}")
        print(f"Script Files (scripts/): {len(script_files)}")

        if len(src_files) > 0:
            coverage = (len(test_files) / len(src_files)) * 100
            print(f"\nTest Coverage Estimate: {coverage:.1f}%")

            if coverage < 20:
                print("  🚨 CRITICAL: Very low test coverage!")
            elif coverage < 50:
                print("  ⚠️ WARNING: Low test coverage")
            else:
                print("  ✅ Good test coverage")
        else:
            print("\n  ⚠️ No source files found in src/")

        print(f"\nTest files found:")
        for test_file in test_files:
            print(f"  - {test_file.relative_to(self.project_root)}")

    def analyze_infrastructure_debt(self):
        """Analyze logging, error handling, configuration"""
        print("\n" + "=" * 80)
        print("4. INFRASTRUCTURE DEBT ANALYSIS")
        print("=" * 80)

        has_logger = False
        has_config = False
        has_error_handling = False

        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                    if 'import logging' in content or 'from logger import' in content:
                        has_logger = True

                    if 'config' in content.lower():
                        has_config = True

                    if 'try:' in content and 'except' in content:
                        has_error_handling = True

            except Exception:
                pass

        print("\nInfrastructure Components:")
        print(f"  Logging: {'✅ Present' if has_logger else '❌ Missing'}")
        print(f"  Configuration Management: {'✅ Present' if has_config else '❌ Missing'}")
        print(f"  Error Handling: {'✅ Present' if has_error_handling else '❌ Missing'}")

        # Check for hardcoded values
        hardcoded_issues = []
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # Look for common hardcoded patterns
                    if 'API_KEY' in content and ('=' in content or 'API_KEY"' in content):
                        hardcoded_issues.append((py_file, "Potential hardcoded API key"))

                    # Look for direct file paths
                    if re.search(r'["\'][A-Z]:[\\\/]', content):
                        hardcoded_issues.append((py_file, "Absolute file paths"))

            except Exception:
                pass

        if hardcoded_issues:
            print("\n⚠️ HARDCODED VALUES:")
            for file_path, issue in hardcoded_issues[:10]:
                print(f"  {file_path.name}: {issue}")

    def analyze_clutter_debt(self):
        """Analyze project organization and clutter"""
        print("\n" + "=" * 80)
        print("5. CLUTTER DEBT ANALYSIS")
        print("=" * 80)

        # Find potentially obsolete files
        obsolete_patterns = [
            'debug_', 'test_', 'temp_', 'old_', 'backup_',
            '_old', '_temp', '_backup', '_copy'
        ]

        potential_obsolete = []
        for f in self.python_files:
            for pattern in obsolete_patterns:
                if pattern in f.name.lower():
                    potential_obsolete.append(f)
                    break

        if potential_obsolete:
            print(f"\n⚠️ POTENTIALLY OBSOLETE FILES ({len(potential_obsolete)}):")
            for f in sorted(potential_obsolete)[:20]:
                print(f"  - {f.relative_to(self.project_root)}")

        # Check scripts directory bloat
        scripts_dir = self.project_root / 'scripts'
        if scripts_dir.exists():
            script_files = list(scripts_dir.glob('*.py'))
            print(f"\n⚠️ SCRIPTS DIRECTORY BLOAT:")
            print(f"  Total scripts: {len(script_files)}")

            if len(script_files) > 30:
                print(f"  🚨 CRITICAL: Too many scripts ({len(script_files)})!")
                print(f"  Recommendation: Organize into subdirectories or archive old experiments")

    def analyze_duplication_debt(self):
        """Analyze code duplication"""
        print("\n" + "=" * 80)
        print("6. DUPLICATION DEBT ANALYSIS")
        print("=" * 80)

        # Simple duplication detection: same filename patterns
        filename_groups = defaultdict(list)

        for f in self.python_files:
            # Extract base name without version/suffix
            base = re.sub(r'_v\d+|_improved|_fixed|_final|_phase\d+', '', f.stem)
            filename_groups[base].append(f)

        duplicate_groups = {k: v for k, v in filename_groups.items() if len(v) > 1}

        if duplicate_groups:
            print(f"\n⚠️ DUPLICATE FILENAME PATTERNS ({len(duplicate_groups)} groups):")
            for base_name, files in sorted(duplicate_groups.items(), key=lambda x: len(x[1]), reverse=True)[:15]:
                if len(files) > 1:
                    print(f"\n  {base_name} ({len(files)} variations):")
                    for f in files:
                        print(f"    - {f.name}")

    def generate_recommendations(self):
        """Generate actionable recommendations"""
        print("\n" + "=" * 80)
        print("TECHNICAL DEBT REMEDIATION PLAN")
        print("=" * 80)

        recommendations = []

        # Design Debt
        if 'versioned_files' in self.analysis_results['design_debt']:
            versioned_count = len(self.analysis_results['design_debt']['versioned_files'])
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Design Debt',
                'issue': f'{versioned_count} sets of versioned files',
                'action': 'Consolidate to single canonical version, archive old versions',
                'effort': 'Medium'
            })

        # Scripts bloat
        scripts_dir = self.project_root / 'scripts'
        if scripts_dir.exists():
            script_count = len(list(scripts_dir.glob('*.py')))
            if script_count > 30:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Clutter Debt',
                    'issue': f'{script_count} scripts in flat structure',
                    'action': 'Organize into: experiments/, production/, utils/, archive/',
                    'effort': 'Low'
                })

        # Test debt
        test_files = [f for f in self.python_files if 'test' in f.name.lower()]
        src_files = [f for f in self.python_files if 'src' in str(f)]
        if len(src_files) > 0:
            coverage = (len(test_files) / len(src_files)) * 100
            if coverage < 50:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Test Debt',
                    'issue': f'Low test coverage ({coverage:.1f}%)',
                    'action': 'Add unit tests for core modules (src/models/, src/api/)',
                    'effort': 'High'
                })

        # Print recommendations
        print("\nPrioritized Actions:\n")

        for i, rec in enumerate(sorted(recommendations, key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x['priority']]), 1):
            print(f"{i}. [{rec['priority']}] {rec['category']}")
            print(f"   Issue: {rec['issue']}")
            print(f"   Action: {rec['action']}")
            print(f"   Effort: {rec['effort']}")
            print()

    def run_analysis(self):
        """Run complete technical debt analysis"""
        print("=" * 80)
        print("TECHNICAL DEBT ANALYSIS")
        print("=" * 80)
        print(f"Project: {self.project_root}")
        print()

        self.scan_project()
        self.analyze_design_debt()
        self.analyze_code_debt()
        self.analyze_test_debt()
        self.analyze_infrastructure_debt()
        self.analyze_clutter_debt()
        self.analyze_duplication_debt()
        self.generate_recommendations()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)


if __name__ == '__main__':
    analyzer = TechnicalDebtAnalyzer(PROJECT_ROOT)
    analyzer.run_analysis()
