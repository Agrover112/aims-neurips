import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # modal_*.py modules live at the repo root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "submission"))
