"""Fit adaptive cubic Chebyshev leaves to elemental input/root pairs."""

from pathlib import Path
import argparse, json, sys, time
import numpy as np
from numpy.polynomial.chebyshev import chebvander

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/transport"))
from local_accuracy import chebyshev, metrics


def basis(x, lo, hi):
    z = 2 * (x - lo) / (hi - lo) - 1
    a = chebvander(z[:, 0], 3)
    for j in (1, 2):
        a = (a[:, :, None] * chebvander(z[:, j], 3)[:, None, :]).reshape(len(x), -1)
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/fitting"))
    p.add_argument("--out", type=Path, default=Path("runs/chebyshev_fit"))
    p.add_argument("--leaves", type=int, default=1024)
    p.add_argument(
        "--samples",
        type=int,
        help="Use the first N training samples for a smaller example",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    if a.leaves < 1:
        p.error("--leaves must be positive")
    source = a.data / "train_cheb.npz"
    if not source.exists():
        source = a.data / "train.npz"
    z = np.load(source)
    x = z["x"][: a.samples]
    y = z["y"][: a.samples]
    if len(y) < 128:
        p.error("At least 128 training samples are needed")
    a.out.mkdir(parents=True, exist_ok=False)
    if a.device == "cpu":
        from scipy.linalg import lstsq
    if a.device == "cuda":
        import torch

        tx = torch.as_tensor(x, device="cuda")
        ty = torch.as_tensor(y, device="cuda")
    start = time.perf_counter()

    def fit(ids, lo, hi):
        if a.device == "cpu":
            mat = basis(x[ids], lo, hi)
            coef = lstsq(mat, y[ids], cond=1e-11, lapack_driver="gelsy")[0]
            error = mat @ coef - y[ids]
            sse = float(error @ error)
        else:
            t = (
                2
                * (tx[ids] - torch.as_tensor(lo, device="cuda"))
                / torch.as_tensor(hi - lo, device="cuda")
                - 1
            )

            def v(j):
                u = t[:, j]
                return torch.stack(
                    (torch.ones_like(u), u, 2 * u * u - 1, 4 * u * u * u - 3), 1
                )

            mat = v(0)
            for j in (1, 2):
                mat = (mat[:, :, None] * v(j)[:, None, :]).reshape(len(ids), -1)
            c = torch.linalg.lstsq(mat, ty[ids], driver="gels").solution
            error = mat @ c - ty[ids]
            sse = float(error @ error)
            coef = c.cpu().numpy()
        if not np.isfinite(coef).all():
            raise RuntimeError("Nonfinite coefficients")
        return dict(ids=ids, lo=lo, hi=hi, c=coef, sse=sse)

    root = fit(np.arange(len(y)), np.full(3, -1.0), np.ones(3))
    leaves = [root]
    while len(leaves) < a.leaves:
        eligible = [n for n in leaves if len(n["ids"]) >= 512 and n["sse"] >= 0]
        if not eligible:
            raise RuntimeError(
                "Not enough samples for requested leaf count; use fewer leaves or more samples"
            )
        node = max(eligible, key=lambda n: n["sse"])
        best = None
        for axis in range(3):
            cut = float(np.median(np.unique(x[node["ids"], axis])))
            mask = x[node["ids"], axis] <= cut
            if (
                cut <= node["lo"][axis] + 1e-12
                or cut >= node["hi"][axis] - 1e-12
                or min(mask.sum(), (~mask).sum()) < 128
            ):
                continue
            hi = node["hi"].copy()
            hi[axis] = cut
            lo = node["lo"].copy()
            lo[axis] = cut
            kids = [
                fit(node["ids"][mask], node["lo"], hi),
                fit(node["ids"][~mask], lo, node["hi"]),
            ]
            score = sum(k["sse"] for k in kids)
            if best is None or score < best[0]:
                best = (score, axis, cut, kids)
        if best is None:
            node["sse"] = -1
            continue
        leaves = [n for n in leaves if n is not node]
        node.update(axis=best[1], cut=best[2], children=best[3])
        leaves.extend(best[3])
    rows = []

    def encode(node):
        index = len(rows)
        r = np.zeros(74)
        rows.append(r)
        r[0] = -1
        r[4:7] = node["lo"]
        r[7:10] = 2 / (node["hi"] - node["lo"])
        r[10:] = node["c"]
        if "children" in node:
            r[0] = node["axis"]
            r[3] = node["cut"]
            r[1] = encode(node["children"][0])
            r[2] = encode(node["children"][1])
        return index

    encode(root)
    path = a.out / "inverse.tree"
    np.asarray(rows, dtype="<f8").tofile(path)
    seconds = time.perf_counter() - start
    validation = np.load(a.data / "validation.npz")
    report = dict(
        samples=len(y),
        leaves=len(leaves),
        fit_seconds=seconds,
        device=a.device,
        validation=metrics(chebyshev(path, validation["x"]), validation["y"]),
    )
    (a.out / "fit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
