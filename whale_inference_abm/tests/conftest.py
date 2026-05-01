"""Pytest config — adds repo root to path so `from abm.* import ...` works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
