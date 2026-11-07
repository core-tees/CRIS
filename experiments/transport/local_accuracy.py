"""Evaluate saved transport inverses on the fixed accuracy sets; no fitting."""

from pathlib import Path
import argparse
import json
import sys
import numpy as np
from numpy.polynomial.chebyshev import chebvander

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/transport"))


def network(checkpoint, raw):
    z = np.load(checkpoint)
    inputs = raw[:, [0, 1, 3]].copy()
    inputs[:, 1] = np.log1p(inputs[:, 1])
    x = 2 * (inputs - z["low"]) / (z["high"] - z["low"]) - 1
    widths, v, offset = z["widths"].astype(int), z["v"], 0
    for layer, (ni, no) in enumerate(zip(widths[:-1], widths[1:])):
        w = v[offset : offset + ni * no].reshape(no, ni)
        offset += ni * no
        b = v[offset : offset + no]
        offset += no
        x = x @ w.T + b
        if layer < len(widths) - 2:
            x = np.tanh(x)
    return (x[:, 0] + 1) / 2


def chebyshev(path, x):
    tree = np.fromfile(path, dtype="<f8").reshape(-1, 74)
    out = np.empty(len(x))

    def visit(node, ids):
        if not len(ids):
            return
        row = tree[node]
        if row[0] >= 0:
            mask = x[ids, int(row[0])] <= row[3]
            visit(int(row[1]), ids[mask])
            visit(int(row[2]), ids[~mask])
        else:
            z = (x[ids] - row[4:7]) * row[7:10] - 1
            basis = chebvander(z[:, 0], 3)
            for j in (1, 2):
                basis = (
                    basis[:, :, None] * chebvander(z[:, j], 3)[:, None, :]
                ).reshape(len(ids), -1)
            out[ids] = basis @ row[10:]

    visit(0, np.arange(len(x)))
    return out


def metrics(pred, target):
    error = pred - target
    assert pred.shape == target.shape and np.isfinite(pred).all()
    return dict(
        count=len(error),
        rmse=float(np.sqrt(np.mean(error**2))),
        maximum=float(np.max(abs(error))),
        p99=float(np.quantile(abs(error), 0.99)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/transport_local")
    parser.add_argument(
        "--network", type=Path, default=ROOT / "models/transport/network.npz"
    )
    parser.add_argument(
        "--tree", type=Path, default=ROOT / "models/transport/cheb_1024.tree"
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/fitting")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in ("validation", "endpoint", "quadratic", "singular"):
        z = np.load(args.data / f"{name}.npz")
        raw, x, y = z["raw"], z["x"], z["y"]
        predictions = {
            "neural": network(args.network, raw),
            "chebyshev": chebyshev(args.tree, x),
        }
        for model, pred in predictions.items():
            rows.append(dict(dataset=name, model=model, **metrics(pred, y)))
    (args.out / "RESULTS.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf8"
    )
    for row in rows:
        if row["dataset"] == "validation":
            print(f"{row['model']}: validation RMSE {row['rmse']:.9g}")


if __name__ == "__main__":
    main()
