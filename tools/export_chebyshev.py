"""Package a fitted Chebyshev tree for the native transport solver."""

from pathlib import Path
import argparse
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--tree", type=Path, required=True)
p.add_argument("--exponent", type=float, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
tree = np.fromfile(a.tree, dtype="<f8")
if not len(tree) or len(tree) % 74 or not np.isfinite(tree).all():
    p.error("Invalid tree")
if not 0.2 <= a.exponent <= 2:
    p.error("Exponent must lie in [0.2,2]")
a.out.parent.mkdir(parents=True, exist_ok=True)
with a.out.open("wb") as f:
    np.asarray([-1024, len(tree) // 74], dtype="<i4").tofile(f)
    np.asarray([a.exponent], dtype="<f8").tofile(f)
    tree.tofile(f)
print(a.out)
