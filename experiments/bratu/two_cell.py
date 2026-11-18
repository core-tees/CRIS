"""Run the paper's two-cell Bratu comparison using PETSc."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from run_bratu import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--case", "two-cell"]
    main()
