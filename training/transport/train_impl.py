"""Three-input FCN: full-data FP64 scaled Cholesky LM."""

import argparse, json, math, os, signal, time, traceback
from pathlib import Path
import numpy as np
import torch
import lm_core as core
import data
from evidence import evidence

STOP = False
WIDTHS = [3] + [13] * 6 + [1]


def configure():
    core.WIDTHS = WIDTHS
    core.LAYERS = []
    core.SIZE = 0
    for ni, no in zip(WIDTHS[:-1], WIDTHS[1:]):
        core.LAYERS.append((core.SIZE, ni, no))
        core.SIZE += ni * no + no
    assert core.SIZE == sum((a + 1) * b for a, b in zip(WIDTHS[:-1], WIDTHS[1:]))


def sync(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def assemble(v, x, y, chunk=8192, gram_rows=65536):
    G = torch.zeros((len(v), len(v)), device=v.device, dtype=v.dtype)
    g = torch.zeros_like(v)
    total = v.new_zeros(())
    jt = gt = 0.0
    for begin in range(0, len(x), gram_rows):
        end = min(begin + gram_rows, len(x))
        sync(v.device)
        tick = time.perf_counter()
        J = torch.empty((end - begin, len(v)), device=v.device, dtype=v.dtype)
        r = torch.empty(end - begin, device=v.device, dtype=v.dtype)
        for a in range(begin, end, chunk):
            b = min(a + chunk, end)
            p, j = core.forward(v, x[a:b], True)
            J[a - begin : b - begin].copy_(j)
            r[a - begin : b - begin].copy_(p - y[a:b])
        sync(v.device)
        jt += time.perf_counter() - tick
        tick = time.perf_counter()
        G.addmm_(J.T, J, beta=1.0, alpha=1.0 / len(x))
        g.add_(J.T @ r, alpha=1.0 / len(x))
        total += r.square().sum()
        sync(v.device)
        gt += time.perf_counter() - tick
        del J, r, j, p
    return G, g, 0.5 * float(total) / len(x), dict(jacobian_seconds=jt, gram_seconds=gt)


def preflight(device):
    configure()
    v = core.initial(20261016, device)
    x = torch.randn(11, 3, device=device)
    y = torch.randn(11, device=device)
    p, J = core.forward(v, x, True)
    ref = torch.autograd.functional.jacobian(lambda vv: core.forward(vv, x[:3]), v)
    je = float((J[:3] - ref).abs().max())
    assert je < 2e-12, je
    G, g, old, timing = assemble(v, x, y, 3, 7)
    ge = float((G - J.T @ J / len(x)).abs().max())
    gg = float((g - J.T @ (p - y) / len(x)).abs().max())
    assert max(ge, gg) < 2e-12, (ge, gg)
    step, lr = core.direction(G, g, 0.1)
    assert step is not None and lr < 1e-8
    ds = step * 0.001
    new = core.objective(v + ds, x, y)
    assert new < old
    reg = 0.03
    small = G[:7, :7]
    vv = v[:7]
    nr, ev = evidence(small, vv, 0.2, reg, len(x))
    eig = torch.linalg.eigvalsh(small)
    gamma = float((eig / (eig + reg)).sum())
    assert abs(ev["gamma"] - gamma) < 1e-8
    return dict(
        evidence_gamma_error=abs(ev["gamma"] - gamma),
        parameters=core.SIZE,
        architecture=WIDTHS,
        jacobian_max_error=je,
        gram_max_error=ge,
        gradient_max_error=gg,
        linear_residual=lr,
        descent_checked=True,
        labels=data.audit(),
    )


@torch.no_grad()
def metrics(v, x, y, endpoint=None):
    errors = []
    pred = []
    for i in range(0, len(x), 65536):
        p = (core.forward(v, x[i : i + 65536]) + 1) / 2
        errors.append(p - y[i : i + 65536])
        pred.append(p)
    e = torch.cat(errors)
    p = torch.cat(pred)
    ae = e.abs()
    result = dict(
        mse=float(e.square().mean()),
        rmse=float(e.square().mean().sqrt()),
        max_error=float(ae.max()),
        median_absolute_error=float(torch.quantile(ae, 0.5)),
        p99_error=float(torch.quantile(ae, 0.99)),
        below_zero=int((p < 0).sum()),
        above_one=int((p > 1).sum()),
        minimum=float(p.min()),
        maximum=float(p.max()),
    )
    if endpoint is not None:
        for key, mask in [("broad", ~endpoint), ("endpoint", endpoint)]:
            result[key + "_rmse"] = (
                float(e[mask].square().mean().sqrt()) if bool(mask.any()) else None
            )
    return result


def save(out, name, v, iteration, accepted, active, mu):
    temp = out / (name + ".tmp.npz")
    np.savez(
        temp,
        v=v.detach().cpu().numpy(),
        widths=WIDTHS,
        low=data.LOW,
        high=data.HIGH,
        output_low=0.0,
        output_high=1.0,
        iteration=iteration,
        accepted=accepted,
        active=active,
        mu=mu,
    )
    temp.replace(out / (name + ".npz"))


def run(a):
    device = torch.device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    core.atomic(out / "PREFLIGHT_PASSED.json", preflight(device))
    sets = {}
    for name in ("train", "validation", "endpoint", "quadratic", "singular"):
        z = np.load(Path(a.data) / (name + ".npz"))
        sets[name] = (
            torch.as_tensor(z["x"], device=device),
            torch.as_tensor(z["y"], device=device),
            torch.as_tensor(z["endpoint"], device=device),
        )
    x, physical_y, train_mask = sets["train"]
    y = 2 * physical_y - 1
    v = core.initial(a.seed, device)
    n = min(a.active, len(x))
    mu = 0.01
    iteration = accepted = 0
    reg = 0.001 / n if a.method == "br" else 0.0
    last_evidence = 0
    ev = None
    best = math.inf
    last_best = last_growth = gap_checks = 0
    milestones = {}
    best_metrics = None
    identity = dict(
        architecture=WIDTHS,
        parameters=core.SIZE,
        seed=a.seed,
        method=a.method,
        evidence_every=25,
        data_growth=a.growth,
        initial_active=n,
        pool=len(x),
        validation=len(sets["validation"][0]),
        full_active_gradient=True,
        full_active_curvature=True,
        precision="float64",
        optimizer="scaled Cholesky LM",
        backtracking_halvings=6,
        sampling=json.loads((Path(a.data) / "READY.json").read_text()),
        seconds=a.seconds,
        job=os.getenv("SLURM_JOB_ID"),
        growth=(
            "50 accepted without best improvement AND gap >1.5 for 3 checks AND 50 accepted since growth; double up to pool"
            if a.growth
            else "fixed active set"
        ),
        stopping="budget, explicit STOP, numerical failure or damping ceiling 1e16; no RMSE or slow-progress stop",
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    )
    core.atomic(out / "IDENTITY.json", identity)
    save(out, "initial", v, iteration, accepted, n, mu)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()

    def record(extra):
        nonlocal best, last_best, best_metrics, gap_checks
        tr = metrics(v, x[:n], physical_y[:n], train_mask[:n])
        va = metrics(v, *sets["validation"])
        ratio = va["mse"] / max(tr["mse"], 1e-300)
        gap_checks = gap_checks + 1 if ratio > 1.5 else 0
        if va["mse"] < best:
            best = va["mse"]
            last_best = accepted
            best_metrics = va
            save(out, "best", v, iteration, accepted, n, mu)
        diagnostics = {
            name: metrics(v, *sets[name])
            for name in ("endpoint", "quadratic", "singular")
        }
        elapsed = time.perf_counter() - start
        for target in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            if va["rmse"] <= target and str(target) not in milestones:
                milestones[str(target)] = dict(
                    seconds=elapsed,
                    iteration=iteration,
                    accepted=accepted,
                    active=n,
                    train_rmse=tr["rmse"],
                    validation_rmse=va["rmse"],
                )
        row = dict(
            iteration=iteration,
            accepted=accepted,
            active=n,
            curvature=n,
            mu=mu,
            seconds=elapsed,
            train=tr,
            validation=va,
            diagnostics=diagnostics,
            regularization=reg,
            evidence=ev,
            best_validation_mse=best,
            validation_train_mse_ratio=ratio,
            accepted_since_best=accepted - last_best,
            gap_checks=gap_checks,
            extra=extra,
        )
        core.atomic(out / "progress.json", row)
        core.atomic(out / "MILESTONES.json", milestones)
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        save(out, "latest", v, iteration, accepted, n, mu)
        print(
            f"updates={accepted} validation_rmse={row['validation']['rmse']:.6g}",
            flush=True,
        )
        return row

    row = record(dict(event="initial"))
    reason = "training_budget"
    while time.perf_counter() - start < a.seconds:
        if STOP or (out / "STOP").exists():
            reason = "author_or_scheduler_stop"
            break
        iteration += 1
        tick = time.perf_counter()
        G, g, old, timings = assemble(v, x[:n], y[:n])
        attempts = []
        took = False
        if a.method == "br" and accepted - last_evidence >= 25:
            sync(device)
            te = time.perf_counter()
            reg, ev = evidence(G, v, 2 * old, reg, n)
            sync(device)
            timings["evidence_seconds"] = time.perf_counter() - te
            last_evidence = accepted
        if reg:
            G.diagonal().add_(reg)
            g.add_(v, alpha=reg)
            old += 0.5 * reg * float(v @ v)
        for trial in range(15):
            if STOP or time.perf_counter() - start >= a.seconds:
                break
            used_mu = mu
            ts = time.perf_counter()
            step, lr = core.direction(G, g, mu)
            sync(device)
            solve_seconds = time.perf_counter() - ts
            if step is None or not bool(torch.isfinite(step).all()):
                attempts.append(dict(mu=mu, failure="factorization_or_nonfinite"))
                mu = min(mu * 10, 1e16)
                continue
            alpha = 1.0
            rho = -1.0
            te = time.perf_counter()
            for back in range(7):
                ds = step * alpha
                predicted = float(-g @ ds - 0.5 * ds @ (G @ ds))
                candidate = core.objective(v + ds, x[:n], y[:n], 65536)
                candidate += 0.5 * reg * float((v + ds) @ (v + ds))
                rho = (
                    (old - candidate) / predicted
                    if predicted > 0 and math.isfinite(candidate)
                    else -1.0
                )
                if rho > 1e-4:
                    v += ds
                    took = True
                    accepted += 1
                    mu = max(1e-14, mu * max(1 / 3, 1 - (2 * rho - 1) ** 3))
                    break
                alpha *= 0.5
            attempts.append(
                dict(
                    mu=used_mu,
                    rho=rho,
                    alpha=alpha,
                    backtracks=back,
                    linear_residual=lr,
                    solve_seconds=solve_seconds,
                    candidate_seconds=time.perf_counter() - te,
                )
            )
            if took:
                break
            mu = min(mu * 10, 1e16)
            if mu >= 1e16:
                break
        del G, g
        detail = dict(
            iteration=iteration,
            accepted=accepted,
            took_step=took,
            active=n,
            curvature=n,
            mu=mu,
            seconds=time.perf_counter() - start,
            iteration_seconds=time.perf_counter() - tick,
            timings=timings,
            attempts=attempts,
        )
        with (out / "iterations.jsonl").open("a") as f:
            f.write(json.dumps(detail) + "\n")
        if iteration == 1 or iteration % 5 == 0 or not took:
            row = record(detail)
            if (
                a.growth
                and accepted - last_best >= 50
                and gap_checks >= 3
                and accepted - last_growth >= 50
            ):
                if n < len(x):
                    before = n
                    n = min(2 * n, len(x))
                    last_growth = accepted
                    gap_checks = 0
                    event = dict(
                        before=before,
                        after=n,
                        iteration=iteration,
                        accepted=accepted,
                        trigger=row,
                    )
                    core.atomic(out / "last_adaptation.json", event)
                    with (out / "adaptations.jsonl").open("a") as f:
                        f.write(json.dumps(event) + "\n")
                    record(dict(event="data_growth", before=before, after=n))
                elif not (out / "DATA_LIMIT.json").exists():
                    core.atomic(out / "DATA_LIMIT.json", row)
        if not took and mu >= 1e16:
            reason = "damping_limit"
            break
        if a.smoke and iteration >= 2:
            reason = "smoke_complete"
            break
    row = record(dict(event="end", reason=reason))
    core.atomic(
        out / "COMPLETE.json",
        dict(
            reason=reason,
            iteration=iteration,
            accepted=accepted,
            seconds=time.perf_counter() - start,
            best_validation=best_metrics,
            final_validation=row["validation"],
            active=n,
            peak_gpu_memory_bytes=(
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        ),
    )


def stop(*_):
    global STOP
    STOP = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/transport_nn")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=4500)
    ap.add_argument("--active", type=int, default=65536)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--width", type=int, default=13)
    ap.add_argument("--method", choices=["lm", "br"], default="lm")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--growth", action="store_true")
    a = ap.parse_args()
    WIDTHS = [3] + [a.width] * 6 + [1]
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, stop)
    try:
        if a.preflight:
            print(json.dumps(preflight(a.device), indent=2))
        else:
            run(a)
    except Exception as e:
        if not a.preflight and Path(a.out).exists():
            core.atomic(
                Path(a.out) / "FAILURE.json",
                dict(error=str(e), traceback=traceback.format_exc()),
            )
        raise
