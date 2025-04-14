# unit_tests/conftest.py
import sys
from pathlib import Path

# Insert the project root (the parent directory of unit_tests) into sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest_plugins = [
    "unit_tests.fixtures.vcr",
    "unit_tests.fixtures.snapshot",
]