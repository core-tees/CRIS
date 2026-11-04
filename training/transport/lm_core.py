"""Scaled Levenberg–Marquardt kernels for the transport inverse."""
import argparse, hashlib, json, math, os, platform, struct, time
from pathlib import Path
import torch

WIDTHS = [3, 20, 20, 20, 1]
LAYERS = []
SIZE = 0
for ni, no in zip(WIDTHS[:-1], WIDTHS[1:]):
    LAYERS.append((SIZE, ni, no))
    SIZE += ni*no + no


def atomic(path, value):
    p = Path(path); q = p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(value, indent=2)); q.replace(p)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forward(v, x, jacobian=False):
    acts = [x]; ws = []
    for k, (offset, ni, no) in enumerate(LAYERS):
        w = v[offset:offset+ni*no].view(no, ni)
        b = v[offset+ni*no:offset+ni*no+no]
        z = acts[-1] @ w.T + b
        acts.append(z.tanh() if k < len(LAYERS)-1 else z)
        ws.append(w)
    y = acts[-1].flatten()
    if not jacobian: return y
    parts = [None]*len(LAYERS)
    delta = torch.ones_like(acts[-1])
    for k in range(len(LAYERS)-1, -1, -1):
        parts[k] = torch.cat(((delta[:,:,None]*acts[k][:,None,:]).flatten(1), delta), 1)
        if k: delta = (delta @ ws[k])*(1-acts[k].square())
    return y, torch.cat(parts, 1)


def initial(seed, device):
    gen = torch.Generator().manual_seed(seed)
    p = []
    for _, ni, no in LAYERS:
        p.extend([torch.randn(no*ni, generator=gen)*math.sqrt(2/(ni+no)), torch.zeros(no)])
    return torch.cat(p).to(device)


def objective(v, x, y, gradient=False, block=65536):
    loss = v.new_zeros(()); g = torch.zeros_like(v) if gradient else None
    for i in range(0,len(x),block):
        with torch.set_grad_enabled(gradient):
            vv = v.detach().requires_grad_(gradient)
            r = forward(vv,x[i:i+block])-y[i:i+block]
            part = .5*r.square().sum()/len(x)
            if gradient: g.add_(torch.autograd.grad(part,vv)[0])
            loss.add_(part.detach())
    return float(loss), g


@torch.no_grad()
def normal(v, x, block):
    G = v.new_zeros((SIZE,SIZE))
    for i in range(0,len(x),block):
        _, J = forward(v,x[i:i+block],True)
        G.addmm_(J.T,J,beta=1,alpha=1/len(x))
    return G


def direction(G,g,mu):
    d = G.diagonal().sqrt().clamp_min(1e-8)
    A = G/d[:,None]/d[None,:]; A.diagonal().add_(mu)
    L, info = torch.linalg.cholesky_ex(A)
    if int(info): return None, None
    s = torch.cholesky_solve((-g/d)[:,None],L).flatten()/d
    residual = float((G@s+mu*d.square()*s+g).norm()/g.norm().clamp_min(1e-30))
    return s,residual


