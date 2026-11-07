"""Compare six supervised Bratu checkpoints on one fixed spatial problem.

Run with NumPy, SciPy and Matplotlib. Bundled inputs reproduce the calculations
without training. Use --plot-only to rebuild the figure from saved measurements.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import lambertw

DATA = ROOT / "runs/bratu_field"
DATA.mkdir(parents=True, exist_ok=True)
INPUT = ROOT / "data/bratu/field_study.npz"
RESULT = DATA / "bratu_checkpoint_transfer.json"
FIELDS = DATA / "bratu_checkpoint_transfer_fields.npz"
NAMES = [f"milestone_{i}.bin" for i in range(5, 10)] + ["final_checkpoint.bin"]
TOL = 1e-10


def prepare():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)


from inverse import network


def setup():
    nx, ny, nz = 19, 19, 9

    def adjacency(n):
        return sparse.diags(
            [np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n), format="csr"
        )

    A = (
        400 * sparse.kron(sparse.eye(nz), sparse.kron(sparse.eye(ny), adjacency(nx)))
        + 400 * sparse.kron(sparse.eye(nz), sparse.kron(adjacency(ny), sparse.eye(nx)))
        + 100 * sparse.kron(adjacency(nz), sparse.eye(nx * ny))
    ).tocsr() / 1800
    b = np.zeros((nz, ny, nx))
    b[0, :, :] = 100 * np.sin(np.pi * np.linspace(0, 1, 21)[1:-1])[None, :] / 1800
    return A, b.ravel(), 2 / 1800


def exact(a, beta):
    rho = np.e * beta * np.exp(a)
    if np.any(rho >= 1):
        raise ValueError(
            "Local fold reached during the fixed-start accuracy comparison"
        )
    w = lambertw(-beta * np.exp(a), 0).real
    return a - w, 1 / (1 + w)


def metrics(values):
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mean_absolute": float(np.mean(abs(values))),
        "maximum": float(np.max(abs(values))),
        "signed_mean": float(np.mean(values)),
        "positive_fraction": float(np.mean(values > 0)),
    }


def solve(A, b, beta, weights=None):
    u = np.zeros(len(b))
    I = sparse.eye(len(b), format="csr")
    states = []
    for step in range(81):
        states.append(u.copy())
        a = A @ u + b
        if weights is None:
            g = u - a - beta * np.exp(u)
            J = I - A - sparse.diags(beta * np.exp(u))
        else:
            prediction, da = network(weights, a, beta, True)
            g = u - prediction
            J = I - sparse.diags(da) @ A
        if np.max(abs(g)) <= TOL:
            return u, states, step, float(np.max(abs(g)))
        if step == 80:
            raise RuntimeError("Fixed-start comparison exceeded 80 updates")
        u += spsolve(J, -g)
        if not np.isfinite(u).all() or np.max(abs(u)) >= 100:
            raise RuntimeError("Fixed-start comparison left the numerical state range")


def run():
    prepare()
    data = np.load(INPUT)
    A, b, beta = setup()
    reference, _, _, ref_residual = solve(A, b, beta)
    agreement = float(np.max(abs(reference - data["archived_reference"])))
    assert agreement < 1e-11
    ar = A @ reference + b
    true_reference, _ = exact(ar, beta)
    assert np.max(abs(reference - true_reference)) < 1e-10
    test = data["test"]
    independent_test_roots, _ = exact(test[:, 0], test[:, 1])
    assert np.max(abs(independent_test_roots - test[:, 2])) < 3e-11
    rows, fields = [], {"reference": reference}
    for index, name in enumerate(NAMES):
        weights = np.fromfile(ROOT / "models/bratu" / name, dtype="<f8")
        u, states, steps, defect = solve(A, b, beta, weights)
        a = A @ u + b
        pred, _ = network(weights, a, beta)
        root, _ = exact(a, beta)
        local_final = pred - root
        delta_reference = network(weights, ar, beta)[0] - true_reference
        test_error = (
            network(weights, test[:, 0], test[:, 1])[0] - independent_test_roots
        )
        trajectory = []
        max_rho, amin, amax = 0.0, float("inf"), -float("inf")
        for state in states:
            ai = A @ state + b
            yi, _ = exact(ai, beta)
            trajectory.append(network(weights, ai, beta)[0] - yi)
            max_rho = max(max_rho, float(np.max(np.e * beta * np.exp(ai))))
            amin, amax = min(amin, float(ai.min())), max(amax, float(ai.max()))
        assert amin >= -2 and amax <= 6 and max_rho < 1 - 1e-8
        # Numerically evaluate the averaged exact-CRIS Jacobian in the
        # single-step error identity, with an independent quadrature check.
        gamma = []
        identity_error = []
        for order in [16, 32]:
            nodes, quadrature_weights = np.polynomial.legendre.leggauss(order)
            davg = np.zeros(len(b))
            for t, qw in zip((nodes + 1) / 2, quadrature_weights / 2):
                _, da = exact(A @ (reference + t * (u - reference)) + b, beta)
                davg += qw * da
            Jbar = sparse.eye(len(b), format="csr") - sparse.diags(davg) @ A
            z = spsolve(Jbar, np.ones(len(b)))
            assert (
                z.min() > 0
            )  # A Z-matrix with a positive inverse-image of 1 is an M-matrix.
            gamma.append(float(z.max()))
            identity_error.append(
                float(
                    np.max(
                        abs(
                            Jbar @ (u - reference)
                            - (u - root - (reference - true_reference))
                        )
                    )
                )
            )
        assert np.isclose(gamma[0], gamma[1], rtol=1e-10)
        assert max(identity_error) < 1e-11
        row = {
            "checkpoint": str(data["checkpoint_names"][index]),
            "sha256": str(data["checkpoint_sha256"][index]),
            "accepted_updates": int(data["updates"][index]),
            "validation_rmse": float(data["validation_rmse"][index]),
            "test_error": metrics(test_error),
            "reference_inputs_error": metrics(delta_reference),
            "final_inputs_error": metrics(local_final),
            "trajectory_inputs_error": metrics(np.concatenate(trajectory)),
            "trajectory_evaluations": len(states) * len(b),
            "field_error": metrics(u - reference),
            "newton_updates": steps,
            "learned_equation_defect_inf": defect,
            "physical_residual_inf": float(np.max(abs(u - a - beta * np.exp(u)))),
            "trajectory_max_fold_coordinate": max_rho,
            "trajectory_a_range": [amin, amax],
            "averaged_inverse_inf_norm": gamma[-1],
            "averaged_jacobian_identity_error": max(identity_error),
            "numerical_max_error_bound": gamma[-1]
            * (
                metrics(local_final)["maximum"]
                + defect
                + np.max(abs(reference - true_reference))
            ),
        }
        assert row["field_error"]["maximum"] <= row["numerical_max_error_bound"]
        rows.append(row)
        fields[f"checkpoint_{index}"] = u
        print(json.dumps(row), flush=True)
    assert np.isclose(rows[-1]["field_error"]["rmse"], 2.8433349140725957e-5, rtol=1e-7)
    result = {
        "protocol": {
            "nodes": [21, 21, 11],
            "unknowns": len(b),
            "beta": beta,
            "initial_state": "Zero interior; prescribed fixed boundary values",
            "solver": "Full Newton with sparse direct factorization",
            "residual_tolerance": TOL,
            "checkpoint_selection": "All retained supervised milestones 5 through 9 and final; selected before field evaluation",
            "test_count": len(test),
            "test_distribution": "Independent matched Bratu preparation distribution",
            "reference_inputs": "Same embedded exact-discrete-reference inputs for every checkpoint",
            "trajectory_inputs": "Each checkpoint own iterates, including initial and final; descriptive, not IID",
            "averaged_jacobian": "16- and 32-point Gauss-Legendre agreement; floating-point diagnostic, not interval certification",
        },
        "reference_residual_inf": ref_residual,
        "reference_agreement_inf": agreement,
        "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "results": rows,
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(FIELDS, **fields)


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = json.loads(RESULT.read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.25), layout="constrained", sharex=True
    )
    checkpoints = np.arange(1, len(result["results"]) + 1)
    for ax, metric, letter, title in zip(
        axes, ["rmse", "maximum"], "ab", ["Average error", "Maximum error"]
    ):
        for key, label, color, marker in [
            ("test_error", "Test inputs", "#28618C", "o"),
            ("reference_inputs_error", "Cell inputs", "#008B8B", "s"),
            ("field_error", "Field solution", "#343434", "^"),
        ]:
            y = [r[key][metric] for r in result["results"]]
            ax.semilogy(
                checkpoints,
                y,
                "-",
                marker=marker,
                color=color,
                ms=4,
                lw=1.2,
                label=label,
            )
        if metric == "maximum":
            bounds = [r["numerical_max_error_bound"] for r in result["results"]]
            assert all(
                r["field_error"]["maximum"] <= bound
                for r, bound in zip(result["results"], bounds)
            )
            ax.semilogy(
                checkpoints,
                bounds,
                "--",
                color="#B65A19",
                lw=1.5,
                label="Field-error bound",
            )
        ax.set_xlabel("Training stage")
        ax.set_xticks(checkpoints)
        ax.set_ylabel("RMSE" if metric == "rmse" else "Maximum error")
        ax.set_title(r"$\mathbf{" + letter + r"}$  " + title, loc="left")
        ax.grid(alpha=0.15)
    fig.legend(
        *axes[1].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    out = DATA / "bratu_checkpoint_transfer"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        run()
    elif not RESULT.exists():
        parser.error("Run the calculation before using --plot-only")
    plot()
