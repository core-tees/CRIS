"""Regenerate Bratu targets at the saved paper inputs with a native PETSc solver."""

from pathlib import Path
import argparse, subprocess, struct, json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in ["train", "validation", "test", "test_near_fold"]:
        src = ROOT / "data/bratu" / (name + ".bin")
        old = np.fromfile(src, "<f8", offset=16).reshape(-1, 4)
        raw = np.zeros((len(old), 5))
        raw[:, :2] = old[:, :2]
        pref = (a.out / name).resolve()
        raw.tofile(str(pref) + ".raw")
        subprocess.run(
            [str(a.solver.resolve()), "1", str(pref) + ".raw", str(pref), "1"],
            check=True,
        )
        z = np.fromfile(str(pref) + ".repeat1.labels", "<f8").reshape(-1, 4)
        root = z[:, 0]
        labels = np.column_stack((raw[:, :2], root, 1 - raw[:, 1] * np.exp(root)))
        with pref.with_suffix(".bin").open("wb") as f:
            f.write(b"BRATUD01" + struct.pack("<Q", len(raw)))
            labels.astype("<f8").tofile(f)
        rows.append(
            dict(
                split=name,
                count=len(raw),
                max_change=float(abs(root - old[:, 2]).max()),
            )
        )
    (a.out / "RESULTS.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
