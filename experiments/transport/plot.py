"""Plot work and progress from new PETSc trajectories."""

from pathlib import Path
import argparse, csv, json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    rows = []
    for path in sorted(a.directory.glob("*.summary.json")):
        prefix = path.name.removesuffix(".summary.json")
        s = json.loads(path.read_text())
        events = [
            json.loads(v)
            for v in (a.directory / (prefix + ".jsonl")).read_text().splitlines()
        ]
        accepted = [v for v in events if v["accepted"]]
        label = prefix.replace("_r1", "")
        axes[0].plot(
            [0] + [v["time_before"] + v["dt"] for v in accepted],
            range(len(accepted) + 1),
            label=label,
        )
        axes[1].bar(label, s["newton"])
        axes[2].bar(label, s["krylov"])
        rows.append(
            dict(
                method=label,
                complete=s["complete_requested_steps"],
                time=s["time"],
                newton=s["newton"],
                krylov=s["krylov"],
                cuts=s["cuts"],
            )
        )
    if not rows:
        p.error("No trajectory summaries found")
    axes[0].set(xlabel="Simulation time", ylabel="Accepted steps")
    axes[0].legend()
    axes[1].set_ylabel("Newton updates")
    axes[2].set_ylabel("Krylov iterations")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(a.directory / "comparison.png", dpi=160)
    plt.close(fig)
    with (a.directory / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    original = next((r for r in rows if r["method"] == "original"), None)
    errors = []
    if original and original["complete"]:

        def field(method):
            path = a.directory / (method + ".field")
            if not path.exists():
                path = a.directory / (method + "_r1.field")
            return np.fromfile(path, dtype="<f8")

        reference = field("original")
        for row in rows:
            if (
                row["method"] != "original"
                and row["complete"]
                and np.isclose(row["time"], original["time"])
            ):
                delta = field(row["method"]) - reference
                errors.append(
                    dict(
                        method=row["method"],
                        rmse=float(np.sqrt(np.mean(delta**2))),
                        maximum=float(abs(delta).max()),
                    )
                )
    (a.directory / "field_errors.json").write_text(json.dumps(errors, indent=2))
    print(json.dumps({"work": rows, "field_errors": errors}, indent=2))


if __name__ == "__main__":
    main()
