"""Scaled Levenberg–Marquardt kernels for the transport inverse."""

import json, math
from pathlib import Path
import torch

WIDTHS = [3, 20, 20, 20, 1]
LAYERS = []
SIZE = 0
for ni, no in zip(WIDTHS[:-1], WIDTHS[1:]):
    LAYERS.append((SIZE, ni, no))
    SIZE += ni * no + no


def atomic(path, value):
    p = Path(path)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(value, indent=2))
    q.replace(p)


def forward(v, x, jacobian=False):
    acts = [x]
    ws = []
    for k, (offset, ni, no) in enumerate(LAYERS):
        w = v[offset : offset + ni * no].view(no, ni)
        b = v[offset + ni * no : offset + ni * no + no]
        z = acts[-1] @ w.T + b
        acts.append(z.tanh() if k < len(LAYERS) - 1 else z)
        ws.append(w)
    y = acts[-1].flatten()
    if not jacobian:
        return y
    parts = [None] * len(LAYERS)
    delta = torch.ones_like(acts[-1])
    for k in range(len(LAYERS) - 1, -1, -1):
        parts[k] = torch.cat(
            ((delta[:, :, None] * acts[k][:, None, :]).flatten(1), delta), 1
        )
        if k:
            delta = (delta @ ws[k]) * (1 - acts[k].square())
    return y, torch.cat(parts, 1)


def initial(seed, device):
    gen = torch.Generator().manual_seed(seed)
    p = []
    for _, ni, no in LAYERS:
        p.extend(
            [
                torch.randn(no * ni, generator=gen) * math.sqrt(2 / (ni + no)),
                torch.zeros(no),
            ]
        )
    return torch.cat(p).to(device)


@torch.no_grad()
def objective(v, x, y, block=65536):
    loss = v.new_zeros(())
    for i in range(0, len(x), block):
        r = forward(v, x[i : i + block]) - y[i : i + block]
        part = 0.5 * r.square().sum() / len(x)
        loss.add_(part)
    return float(loss)


def direction(G, g, mu):
    d = G.diagonal().sqrt().clamp_min(1e-8)
    A = G / d[:, None] / d[None, :]
    A.diagonal().add_(mu)
    L, info = torch.linalg.cholesky_ex(A)
    if int(info):
        return None, None
    s = torch.cholesky_solve((-g / d)[:, None], L).flatten() / d
    residual = float(
        (G @ s + mu * d.square() * s + g).norm() / g.norm().clamp_min(1e-30)
    )
    return s, residual
