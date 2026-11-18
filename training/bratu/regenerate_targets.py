"""Regenerate lower-branch targets at the supplied Bratu training inputs."""

from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np
from scipy.special import lambertw

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "data/bratu/regularized_training")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    checks = {}
    for name in ("interior", "validation"):
        rows = np.fromfile(a.source / f"{name}.bin", "<f8").reshape(-1, 4)
        z = -rows[:, 1] * np.exp(rows[:, 0])
        if not np.all(np.isfinite(z) & (z >= -1 / np.e) & (z <= 0)):
            raise ValueError(f"{name} contains an input outside the lower-root domain")
        values = rows[:, 0] - lambertw(z, 0).real
        checks[name] = {"count": len(rows), "max_change": float(np.max(np.abs(values - rows[:, 2])))}
        rows[:, 2] = values
        rows.astype("<f8").tofile(a.out / f"{name}.bin")
    for name in ("regularity.bin", "validation_reg.bin", "initial.bin"):
        shutil.copy2(a.source / name, a.out / name)
    manifest = {"source": str(a.source), "method": "principal Lambert W branch",
                "labels": checks, "initial_sha256": hashlib.sha256((a.out / "initial.bin").read_bytes()).hexdigest()}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
