"""Compare original transport with neural and Chebyshev CRIS."""

from pathlib import Path
import argparse, json, shlex, struct, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"original": "srdm", "neural": "nn", "chebyshev": "cheb"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        choices=["column", "edfm", "norne", "spe10", "layer50", "layer75", "layer84"],
        default="column",
    )
    p.add_argument("--exponent", type=float, choices=[2.0, 0.8, 0.2], default=2.0)
    p.add_argument("--method", choices=["all", *METHODS], default="all")
    p.add_argument(
        "--newton", choices=["ordinary", "backtracking"], default="backtracking"
    )
    p.add_argument(
        "--executable", type=Path, default=ROOT / "build-linux/petsc_transport"
    )
    p.add_argument(
        "--model", type=Path, help="Deployment model for one selected CRIS method"
    )
    p.add_argument("--steps", type=int, help="Limit scheduled intervals")
    p.add_argument("--dt", type=float, help="One independent step, without subdivision")
    p.add_argument("--out", type=Path, default=ROOT / "runs/transport")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.model and a.method not in ("neural", "chebyshev"):
        p.error("--model requires one CRIS method")
    if a.dt is not None and a.dt <= 0:
        p.error("--dt must be positive")
    config = json.loads((ROOT / "experiments/transport/cases.json").read_text())
    law = {2.0: "quadratic", 0.8: "p08", 0.2: "singular"}[a.exponent]
    policy = {"ordinary": "basic", "backtracking": "bt"}[a.newton]
    jobs = []
    for method in METHODS if a.method == "all" else [a.method]:
        tag = METHODS[method]
        candidates = [f"{a.case}_{law}_{tag}_{policy}"]
        if law == "p08":
            candidates.append(f"{a.case}_singular_{tag}_{policy}_p08")
        candidates.append(f"{a.case}_{law}_{tag}")
        name = next((n for n in candidates if n in config["presets"]), None)
        if name is None:
            p.error(
                f"No released configuration for {a.case}, p={a.exponent}, {method}, {a.newton}"
            )
        opts = config["defaults"] | config["presets"][name]
        if opts["-snes_linesearch_type"] != policy:
            p.error(f"No released {a.newton} configuration for {method} in this case")
        for flag in ("-graph", "-initial_file", "-model"):
            opts[flag] = str(ROOT / opts[flag])
        if a.model:
            opts["-model"] = str(a.model.resolve())
        opts["-trajectory_repeats"] = "1"
        if a.steps is not None:
            opts["-steps"] = str(a.steps)
        if a.dt is not None:
            with open(opts["-graph"], "rb") as f:
                _, n, edges, steps, _ = struct.unpack("<5q", f.read(40))
                f.seek(40 + 24 * n + 8 * (n + 1) + 12 * edges)
                first = struct.unpack("<d", f.read(8))[0]
            opts.update(
                {
                    "-dt_scale": str(a.dt / first),
                    "-steps": "1",
                    "-max_cuts": "0",
                    "-snes_max_it": "2000",
                }
            )
        prefix = a.out.resolve() / method
        opts["-output"] = str(prefix)
        if "-log_view" in opts:
            opts["-log_view"] = ":" + str(prefix) + ".petsc.log"
        cmd = [str(a.executable.resolve())]
        for key, value in opts.items():
            cmd.append(key)
            if value is not None:
                cmd.append(str(value))
        jobs.append((method, cmd))
    if a.dry_run:
        for _, cmd in jobs:
            print(shlex.join(cmd))
        return
    if not a.executable.is_file():
        p.error("Build PETSc executable first; see docs/BUILD.md")
    a.out.mkdir(parents=True, exist_ok=False)
    for method, cmd in jobs:
        print(f"Running {method}", flush=True)
        with (a.out / f"{method}.log").open("w") as log:
            subprocess.run(
                cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    (a.out / "settings.json").write_text(
        json.dumps(
            {"case": a.case, "exponent": a.exponent, "commands": dict(jobs)}, indent=2
        )
    )
    subprocess.run(
        [sys.executable, str(ROOT / "experiments/transport/plot.py"), str(a.out)],
        check=True,
    )


if __name__ == "__main__":
    main()
