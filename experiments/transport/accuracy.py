"""CPU accuracy studies from the paper. No GPU timing measurements."""

from pathlib import Path
import argparse
import json
import numpy as np
from accuracy_solver import column, norne

ROOT = Path(__file__).resolve().parents[2]


def error(a, b):
    e = a - b
    return dict(
        rmse=float(np.sqrt(np.mean(e * e))),
        mae=float(np.mean(abs(e))),
        maximum=float(abs(e).max()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study", choices=["checkpoints", "refinement", "norne-reference"])
    ap.add_argument("--build", type=Path, default=ROOT / "build-linux")
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--limit", type=int, help="Only first N configurations, for a quick check"
    )
    a = ap.parse_args()
    out = a.out or ROOT / "runs" / a.study
    out.mkdir(parents=True, exist_ok=False)

    def exe(name):
        for p in (
            a.build / name,
            a.build / (name + ".exe"),
            a.build / "Release" / (name + ".exe"),
        ):
            if p.is_file():
                return str(p.resolve())
        raise FileNotFoundError(f"Build {name} with CMake first")

    rows = []
    if a.study == "checkpoints":
        stage = json.loads((ROOT / "models/transport/STAGES.json").read_text())
        reference = out / "original"
        column(exe("petsc_transport"), "original", 256, 150, reference, front=True)
        ref = np.fromfile(str(reference) + ".t50.field", dtype="<f8")
        configs = [(s["leaves"], s) for s in stage]
        for n, s in configs[: a.limit]:
            prefix = out / f"leaves{n}"
            summary = column(exe("petsc_transport"),
                             ROOT / f"models/transport/cheb_{n}.tree",
                             256, 150, prefix, front=True)
            summary.update(
                leaves=n,
                local=s["scores"]["quadratic"],
                field=error(np.fromfile(str(prefix) + ".t50.field", "<f8"), ref),
            )
            rows.append(summary)
            print(f"Completed {prefix.name}", flush=True)
    elif a.study == "refinement":
        configs = json.loads(
            (ROOT / "experiments/transport/refinement.json").read_text()
        )
        for c in configs[: a.limit]:
            prefix = out / f"beta{c['beta']:g}_N{c['N']}"
            summary = column(exe("petsc_transport"),
                             ROOT / "models/transport/shared976_quadratic.model",
                             c["N"], c["beta"], prefix)
            reference = out / f"original_beta{c['beta']:g}_N{c['N']}"
            column(exe("petsc_transport"), "original", c["N"], c["beta"], reference)
            discrete = np.fromfile(str(reference) + ".t200.field", "<f8")
            u = np.fromfile(str(prefix) + ".t200.field", "<f8")
            flux = lambda v: v * v / (v * v + 0.4 * (1 - v) ** 2)
            shock = 250 + 200 * (flux(0.25) - flux(0.05)) / 0.2
            h = 1000 / c["N"]
            fraction = np.clip((shock - np.arange(c["N"]) * h) / h, 0, 1)
            analytic = 0.05 + 0.2 * fraction
            summary.update(
                N=c["N"],
                beta=c["beta"],
                inverse=error(u, discrete),
                total=error(u, analytic),
                discretization=error(discrete, analytic),
            )
            rows.append(summary)
            print(f"Completed {prefix.name}", flush=True)
    else:
        for cap in [1000, 500][: a.limit]:
            prefix = out / f"dt{cap}"
            rows.append(norne(exe("petsc_transport"), cap, prefix))
    (out / "RESULTS.json").write_text(json.dumps(rows, indent=2))
    if a.study != "norne-reference":
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if a.study == "checkpoints":
            fig, ax = plt.subplots(layout="constrained")
            for key, label in [("local", "Local inverse"), ("field", "Coupled field")]:
                ax.loglog(
                    [r["leaves"] for r in rows],
                    [r[key]["rmse"] for r in rows],
                    "o-",
                    label=label,
                )
            ax.set_xlabel("Chebyshev leaves")
        else:
            betas = sorted({r["beta"] for r in rows})
            fig, axes = plt.subplots(
                1,
                len(betas),
                figsize=(4 * len(betas), 3.6),
                squeeze=False,
                layout="constrained",
            )
            for ax, beta in zip(axes.flat, betas):
                cohort = [r for r in rows if r["beta"] == beta]
                for key, style in [
                    ("inverse", "o-"),
                    ("discretization", "s--"),
                    ("total", "^:"),
                ]:
                    ax.loglog(
                        [1000 / r["N"] for r in cohort],
                        [r[key]["rmse"] for r in cohort],
                        style,
                        label=key.capitalize(),
                    )
                ax.set(xlabel="Cell width", title=rf"$\beta={beta:g}$")
                widths = [1000 / r["N"] for r in cohort]
                ticks = np.geomspace(min(widths), max(widths), 3)
                ax.set_xticks(ticks, [f"{v:.2g}" for v in ticks])
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for ax in fig.axes:
            ax.set_ylabel("RMSE")
            ax.spines[["top", "right"]].set_visible(False)
        fig.axes[0].legend()
        fig.savefig(out / "accuracy.png", dpi=160)
        plt.close(fig)
    print(out / "RESULTS.json")


if __name__ == "__main__":
    main()
