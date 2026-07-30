import os
import sys
from pathlib import Path

os.environ.setdefault("CLUBELO_ENABLED", "false")

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))
