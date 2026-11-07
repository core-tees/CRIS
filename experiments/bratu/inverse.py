from pathlib import Path
import numpy as np
from scipy.special import lambertw

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"mse": np.fromfile(ROOT / "models/bratu/final_checkpoint.bin", dtype="<f8")}


def network(weights, a, beta, derivative=False):
    shape = np.shape(a)
    a = np.asarray(a).ravel()
    beta = np.broadcast_to(beta, shape).ravel()
    x = np.column_stack(
        [(a + 2) / 4 - 1, 2 * np.log1p(beta / 1e-6) / np.log1p(1e6) - 1]
    )
    dx = (
        np.column_stack([np.full(a.size, 0.25), np.zeros(a.size)])
        if derivative
        else None
    )
    w = weights
    offset = 0
    width = 2
    for _ in range(4):
        matrix = w[offset : offset + 20 * width].reshape(20, width)
        offset += 20 * width
        bias = w[offset : offset + 20]
        offset += 20
        x = np.tanh(x @ matrix.T + bias)
        if derivative:
            dx = (dx @ matrix.T) * (1 - x * x)
        width = 20
    u = 4.5 * (x @ w[offset : offset + 20] + w[offset + 20] + 1) - 2
    du = 4.5 * (dx @ w[offset : offset + 20]) if derivative else None
    return u.reshape(shape), None if du is None else du.reshape(shape)


def inverse(name, a, beta, derivative=False):
    if name != "exact":
        return network(MODELS[name], a, beta, derivative)
    q = np.asarray(beta) * np.exp(a)
    valid = q <= 1 / np.e
    z = np.where(valid, -q, 0)
    w = lambertw(z, 0).real
    u = np.where(valid, a - w, np.nan)
    return u, np.where(valid, 1 / (1 + w), np.nan) if derivative else None
