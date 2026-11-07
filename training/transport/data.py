"""Three-input sampling design; native PETSc solves supply root targets."""

import numpy as np

LOW = np.array([0.0, 0.0, 0.2])
HIGH = np.array([1.0, np.log1p(2.5e6), 2.0])


def transform(raw):
    x = raw[:, [0, 1, 3]].copy()
    x[:, 1] = np.log1p(x[:, 1])
    return 2 * (x - LOW) / (HIGH - LOW) - 1


def sample(power, seed, kind="mixed"):
    import torch

    u = (
        torch.quasirandom.SobolEngine(9, scramble=True, seed=seed)
        .draw_base2(power, dtype=torch.float64)
        .numpy()
    )
    raw = np.column_stack(
        (
            u[:, 0],
            np.expm1(u[:, 1] * HIGH[1]),
            np.full(len(u), 2.0),
            0.2 + 1.8 * u[:, 3],
            np.full(len(u), 0.4),
        )
    )
    endpoint = u[:, 5] < 0.2
    if kind == "broad":
        endpoint[:] = False
    if kind == "endpoint":
        endpoint[:] = True
    distance = 10 ** (-8 + u[:, 0] * (np.log10(0.05) + 8))
    raw[endpoint, 0] = np.where(
        u[endpoint, 8] < 0.5, distance[endpoint], 1 - distance[endpoint]
    )
    mix = u[:, 6]
    raw[mix < 0.15, 3] = 0.2
    raw[(mix >= 0.15) & (mix < 0.30), 3] = 2.0
    return raw, endpoint


def audit():
    raw, _ = sample(8, 181001, "mixed")
    x = transform(raw)
    assert raw.shape == (256, 5) and np.isfinite(x).all() and abs(x).max() <= 1
    return dict(sampler="fixed Sobol design", label_solver="PETSc local solves")
