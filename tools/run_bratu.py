"""Run the paper's Bratu comparisons with PETSc and one shared inverse."""

from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import shlex
import subprocess

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/bratu/current"
METHODS = {"original": "classical", "aspin": "aspin", "neural": "learned"}


def cases_for(case, limit):
    if case == "two-cell":
        grid = np.linspace(-6, 6, 91)
        rows = [(i, 36, 150, x, y) for i, (y, x) in
                enumerate((y, x) for y in grid for x in grid)]
    elif case == "spatial":
        rows = [(0, 36, 150, 0., 0.)]
    else:
        # Case zero is the main example, followed by the remaining combinations.
        rows = [(0, 36, 150, 0., 0.)]
        rows += [(i, lam, q, 0., 0.) for i, (lam, q) in enumerate(
            ((lam, q) for lam in (30, 32, 34, 36) for q in (100, 125, 150, 175)
             if (lam, q) != (36, 150)), 1)]
    return rows if limit is None else rows[:limit]


def root_pair():
    # The common symmetric solutions of the two-cell system.
    coupling = 0.9836181378951395
    beta, source = 36 / 1800, -150 / 1800
    f = lambda u: (1 - coupling) * u - source - beta * np.exp(u)
    fold = np.log((1 - coupling) / beta)
    return (brentq(f, -100, fold), brentq(f, fold, 10))


def summarize(case, output):
    rows = []
    for f in sorted(output.glob("*/cases.csv")):
        with f.open(newline="") as stream:
            for row in csv.DictReader(stream):
                row["method"] = f.parent.name
                rows.append(row)
    for f in sorted(output.glob("*/*/cases.csv")):
        with f.open(newline="") as stream:
            for row in csv.DictReader(stream):
                row["method"] = f.parent.parent.name
                rows.append(row)
    lower, upper = root_pair()
    for row in rows:
        row["outcome"] = "not_converged"
        if int(row["accepted"]) != 1:
            continue
        if case == "two-cell":
            values = np.array([float(row["final1"]), float(row["final2"])])
            row["outcome"] = "other_root"
            for name, root in (("lower", lower), ("upper", upper)):
                if np.max(np.abs(values - root)) < 1e-3:
                    row["outcome"] = name
                    break
        else:
            method = row["method"]
            subdir = output / method
            if case == "sweep":
                subdir /= f"case_{int(row['case']):02d}"
            state = np.fromfile(subdir / f"state_{row['case']}.bin", "<f8", offset=8)
            reference = np.fromfile(DATA / "reference" / f"state_{row['case']}.bin", "<f8", offset=8)
            if state.shape != reference.shape or not state.size:
                raise ValueError("State and reference dimensions differ")
            row["reference_max_difference"] = float(np.max(np.abs(state - reference)))
            row["outcome"] = "desired" if row["reference_max_difference"] < 1e-3 else "other_root"
    (output / "summary.json").write_text(json.dumps(rows, indent=2))
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        if selected:
            counts = {name: sum(r["outcome"] == name for r in selected)
                      for name in ("lower", "upper", "desired", "other_root", "not_converged")}
            print(f"{method}: {counts}")
    if case == "two-cell" and len(rows) >= 3:
        plot_two_cell(rows, output, lower, upper)


def plot_two_cell(rows, output, lower, upper):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    methods = [m for m in METHODS if any(r["method"] == m for r in rows)]
    fig, axes = plt.subplots(1, len(methods), figsize=(3.1 * len(methods), 3.3),
                             squeeze=False, layout="constrained")
    colors = {"not_converged": "#d2a9bb", "other_root": "#999999",
              "upper": "#bd7727", "lower": "#28618c"}
    for ax, method in zip(axes[0], methods):
        selected = [r for r in rows if r["method"] == method]
        ax.scatter([float(r["initial1"]) for r in selected],
                   [float(r["initial2"]) for r in selected],
                   c=[colors[r["outcome"]] for r in selected], s=3, marker="s")
        ax.scatter([lower, upper], [lower, upper], marker="*", s=65,
                   c=[colors["lower"], colors["upper"]], edgecolors="white")
        ax.set(title={"original": "Classical", "neural": "CRIS", "aspin": "ASPIN"}[method],
               xlim=(-6, 6), ylim=(-6, 6), xlabel="Initial $U_1$", aspect="equal")
    axes[0, 0].set_ylabel("Initial $U_2$")
    labels = [("lower", "Lower solution"), ("upper", "Upper solution"),
              ("not_converged", "No convergence")]
    if any(r["outcome"] == "other_root" for r in rows):
        labels.append(("other_root", "Other solution"))
    fig.legend(handles=[Patch(facecolor=colors[key], label=label) for key, label in labels],
               loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(output / "convergence.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["two-cell", "spatial", "sweep"], default="two-cell")
    parser.add_argument("--method", choices=["all", *METHODS], default="all")
    parser.add_argument("--newton", choices=["ordinary", "backtracking"], default="ordinary")
    parser.add_argument("--executable", type=Path, default=ROOT / "build-linux/petsc_bratu")
    parser.add_argument("--model", type=Path, default=ROOT / "models/bratu/regularized.bin")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/bratu")
    parser.add_argument("--limit", type=int, help="Run only the first N starting states or cases")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel processes for the 3D sweep")
    parser.add_argument("--dry-run", action="store_true")
    a = parser.parse_args()
    if a.limit is not None and a.limit < 1:
        parser.error("--limit must be positive")
    if a.jobs < 1 or (a.jobs != 1 and a.case != "sweep"):
        parser.error("--jobs must be positive and applies only to the 3D sweep")
    methods = METHODS if a.method == "all" else [a.method]
    starts = cases_for(a.case, a.limit)
    output = a.out.resolve()
    graph = "two_cell" if a.case == "two-cell" else "spatial"
    jobs = []
    for method in methods:
        groups = [[row] for row in starts] if a.case == "sweep" else [starts]
        for group in groups:
            folder = output / method
            if a.case == "sweep":
                folder /= f"case_{group[0][0]:02d}"
            command = [str(a.executable.resolve()), str(DATA), graph, METHODS[method],
                       str(folder / "inputs.txt"), str(folder),
                       "-bratu_model", str(a.model.resolve()),
                       "-snes_linesearch_type", "basic" if a.newton == "ordinary" else "bt",
                       "-snes_divergence_tolerance", "1e300", "-ksp_max_it", "10000"]
            jobs.append((method, folder, group, command))
    if a.dry_run:
        for _, _, _, command in jobs:
            print(shlex.join(command))
        return
    if not a.executable.is_file():
        parser.error("Build petsc_bratu first; see docs/BUILD.md")
    if not a.model.is_file():
        parser.error("Bratu model not found")
    output.mkdir(parents=True, exist_ok=False)
    (output / "settings.json").write_text(json.dumps({"case": a.case, "newton": a.newton,
        "commands": [command for _, _, _, command in jobs]}, indent=2))
    def run(job):
        method, folder, group, command = job
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "inputs.txt").write_text("".join(
            f"{i} {lam} {q} {x:.17g} {y:.17g} -\n" for i, lam, q, x, y in group))
        print(f"Running {method}: {folder.relative_to(output)}", flush=True)
        with (folder / "run.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(run, jobs))
    summarize(a.case, output)


if __name__ == "__main__":
    main()
