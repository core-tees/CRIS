"""Check the deployed Bratu map and derivative against an independent evaluation."""

from pathlib import Path
import argparse
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def network(weights, alpha, beta):
    alpha = np.asarray(alpha, dtype=float)
    beta = np.broadcast_to(beta, alpha.shape)
    a = alpha.copy()
    above, below = a > 8, a < -90
    a[above] = 8 + 5 * np.tanh((a[above] - 8) / 5)
    a[below] = -90 + 5 * np.tanh((a[below] + 90) / 5)
    x = np.column_stack((np.arcsinh(a / 5), (beta - .02) / .005))
    offset = 0
    for i in range(7):
        ni, no = (2 if i == 0 else 16), (1 if i == 6 else 16)
        matrix = weights[offset:offset + ni * no].reshape(no, ni)
        offset += ni * no
        bias = weights[offset:offset + no]
        offset += no
        x = x @ matrix.T + bias
        if i < 6:
            x = np.tanh(x)
    if offset != len(weights):
        raise ValueError("Incorrect model size")
    return a + x[:, 0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--executable", type=Path, default=ROOT / "build-linux/petsc_bratu")
    p.add_argument("--model", type=Path, default=ROOT / "models/bratu/regularized.bin")
    args = p.parse_args()
    rng = np.random.default_rng(20261117)
    alpha = np.concatenate((rng.uniform(-100, 20, 1000), [-120, -95, -90, 8, 12, 25]))
    beta = rng.uniform(.015, .025, len(alpha))
    h = 1e-4
    weights = np.fromfile(args.model, "<f8")
    value = network(weights, alpha, beta)
    slope = (network(weights, alpha + h, beta) - network(weights, alpha - h, beta)) / (2 * h)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)
        np.savetxt(path / "queries.txt", np.column_stack((alpha, beta)), fmt="%.17g")
        command = [str(args.executable.resolve()), str(ROOT / "data/bratu/current"),
                   "spatial", "learned", str(path / "unused.txt"), str(path),
                   "-bratu_model", str(args.model.resolve()), "-fit_queries", str(path / "queries.txt")]
        subprocess.run(command, check=True)
        actual = np.loadtxt(path / "fit_values.csv", delimiter=",")
    value_error = np.max(np.abs(actual[:, 0] - value))
    slope_error = np.max(np.abs(actual[:, 1] - slope))
    print(f"Map maximum difference: {value_error:.3g}; derivative difference: {slope_error:.3g}")
    if value_error > 1e-10 or slope_error > 1e-6:
        raise SystemExit("Bratu model verification failed")


if __name__ == "__main__":
    main()
