import sys
from pathlib import Path

# plf (parser library) lives in the sibling repo. Prefer an installed
# package; fall back to the repo path for local development.
try:
    import plf  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "plf-python"))
