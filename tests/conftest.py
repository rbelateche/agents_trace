import sys
from pathlib import Path

# Make `app` importable from tests/ (mirrors how uvicorn runs it: PYTHONPATH=backend)
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
