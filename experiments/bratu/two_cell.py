"""Reproduce the paper two-cell basin using ordinary Newton."""

from pathlib import Path
import argparse, json
import numpy as np
from scipy.optimize import brentq
import inverse as base

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/bratu"


def two_cell(method, beta, coupling):
    x, y = np.meshgrid(np.linspace(0, 3, 91), np.linspace(0, 3, 91))
    starts = np.column_stack([x.ravel(), y.ravel()])
    u = starts.copy()
    fold = np.log((1 - coupling) / beta)
    fun = lambda s: (1 - coupling) * s - 0.65 - beta * np.exp(s)
    if fun(fold) <= 0:
        return dict(status="no_verified_symmetric_lower_reference")
    lower = brentq(fun, -20, fold, xtol=1e-14)
    active = np.ones(len(u), bool)
    status = np.full(len(u), "iteration_limit", object)
    counts = np.zeros(len(u), int)
    outside = np.zeros(len(u), int)
    unsupported = np.zeros(len(u), int)
    first_exit = np.full(len(u), -1, int)
    for iteration in range(81):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        state = u[ids]
        a = 0.65 + coupling * state[:, ::-1]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            rho = np.e * beta * np.exp(a)
            valid = np.all(rho < 1, axis=1)
            support = np.all((a >= -2) & (a <= 6) & (rho <= 1 - 1e-8), axis=1)
            outside[ids] += ~valid
            unsupported[ids] += ~support
            new = (first_exit[ids] < 0) & ~valid
            first_exit[ids[new]] = iteration
            if method == "srdm":
                reaction = beta * np.exp(state)
                g = state - a - reaction
                d = 1 - reaction
                off = np.full_like(d, -coupling)
            else:
                phi, slope = base.inverse(method, a, beta, True)
                g = state - phi
                d = np.ones_like(state)
                off = -coupling * slope
            bounded = np.all(np.abs(state) < 100, axis=1)
            finite = np.all(np.isfinite(g), axis=1)
            bad = ~bounded | ~finite
            status[ids[~bounded]] = "magnitude_guard"
            status[ids[bounded & ~finite]] = "nonfinite_or_undefined_inverse"
            active[ids[bad]] = False
            done = ~bad & (np.max(np.abs(g), axis=1) <= 1e-10)
            status[ids[done]] = "map_converged"
            active[ids[done]] = False
            move = ~bad & ~done
            if iteration == 80:
                break
            det = d[:, 0] * d[:, 1] - off[:, 0] * off[:, 1]
            delta = np.column_stack(
                [
                    (-d[:, 1] * g[:, 0] + off[:, 0] * g[:, 1]) / det,
                    (off[:, 1] * g[:, 0] - d[:, 0] * g[:, 1]) / det,
                ]
            )
            u[ids[move]] += delta[move]
            counts[ids[move]] += 1
    with np.errstate(over="ignore", invalid="ignore"):
        rn = np.max(np.abs(u - 0.65 - coupling * u[:, ::-1] - beta * np.exp(u)), axis=1)
        err = np.max(np.abs(u - lower), axis=1)
    captured = (status == "map_converged") & (rn <= 1e-4) & (err <= 1e-3)
    initial = np.all(
        np.e * beta * np.exp(0.65 + coupling * starts[:, ::-1]) < 1, axis=1
    )
    label = f"two_b{beta:g}_c{coupling:g}_{method}"
    np.savez_compressed(
        OUT / "two_cell" / f"{label}.npz",
        starts=starts,
        final=u,
        status=status.astype(str),
        updates=counts,
        captured=captured,
        physical_residual=rn,
        reference_error=err,
        initially_admissible=initial,
        first_domain_exit=first_exit,
        unsupported_evaluations=unsupported,
        outside_evaluations=outside,
    )
    if beta == 0.1 and coupling == 0.25 and method == "exact":
        assert int(np.sum(captured & initial)) == 6028
    return dict(
        status="complete",
        method=method,
        beta=beta,
        coupling=coupling,
        reference=lower,
        captured=int(captured.sum()),
        captured_initially_admissible=int(np.sum(captured & initial)),
        starts=len(u),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    OUT = a.out.resolve()
    (OUT / "two_cell").mkdir(parents=True, exist_ok=True)
    rows = [two_cell(m, 0.1, 0.25) for m in ("srdm", "exact", "mse")]
    (OUT / "two_cell_summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), layout="constrained")
    for ax, method, title in zip(
        axes, ["srdm", "mse"], ["Original formulation", "Learned CRIS"]
    ):
        z = np.load(OUT / "two_cell" / f"two_b0.1_c0.25_{method}.npz")
        ax.scatter(
            z["starts"][:, 0],
            z["starts"][:, 1],
            c=z["captured"],
            cmap="Blues",
            s=4,
            vmin=0,
            vmax=1,
        )
        ax.set(title=title, xlabel="Initial U1", ylabel="Initial U2", aspect="equal")
    fig.suptitle("Dark blue: convergence to the selected lower solution", fontsize=10)
    fig.savefig(OUT / "convergence.png", dpi=160)
    plt.close(fig)
