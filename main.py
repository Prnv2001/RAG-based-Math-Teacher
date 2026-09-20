import sys
from pathlib import Path

# Ensure 'src/' is included in Python path for cloud deployments (Render, Railway, etc.)
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from math_teacher.main import app

__all__ = ["app"]
