from pathlib import Path
import json, hashlib
import numpy as np
import argparse

a = argparse.ArgumentParser()
a.add_argument("--checkpoint", type=Path, required=True)
a.add_argument("--out", type=Path, default=Path("runs/exported_models"))
a.add_argument("--verify-paper", action="store_true")
args = a.parse_args()
P = Path(__file__).resolve().parents[1]
O = args.out
O.mkdir(parents=True, exist_ok=True)
source = args.checkpoint
digest = hashlib.sha256(source.read_bytes()).hexdigest()
if args.verify_paper:
    assert digest == "9ddfc726a219fadd131300b8d42848e851bb082d1c0c1b38acfeabd7da5c8ebb"
z = np.load(source)
widths = z["widths"].astype(int)
v = z["v"]
layers = []
k = 0
for ni, no in zip(widths[:-1], widths[1:]):
    w = v[k : k + ni * no].reshape(no, ni)
    k += ni * no
    b = v[k : k + no]
    k += no
    layers.append((w, b))
assert k == len(v) == 976 and widths.tolist() == [
    3,
    13,
    13,
    13,
    13,
    13,
    13,
    1,
], "This exporter uses the paper architecture: 3 inputs, six width-13 hidden layers, one output."


def forward(ls, x, dx):
    for j, (w, b) in enumerate(ls):
        x = x @ w.T + b
        dx = dx @ w.T
        if j < len(ls) - 1:
            x = np.tanh(x)
            dx *= 1 - x * x
    return (x[:, 0] + 1) / 2, dx[:, 0] / 2


rng = np.random.default_rng(24680)
a = rng.random(2000)
old = rng.random(2000)
beta = np.expm1(rng.uniform(0, np.log1p(2.5e6), 2000))
report = {}
for name, p in [("quadratic", 2.0), ("intermediate", 0.8), ("singular", 0.2)]:
    const = 2 * (np.array([p]) - z["low"][2:]) / (z["high"][2:] - z["low"][2:]) - 1
    w, b = layers[0]
    wf = np.column_stack((w[:, :2], np.zeros(len(w))))
    bf = b + w[:, 2:] @ const
    fold = [(wf, bf)] + layers[1:]
    path = O / ("shared976_" + name + ".model")
    with path.open("wb") as f:
        for t in ([len(widths) - 1], widths, [2] * (len(widths) - 2) + [3], [2, 1, 0]):
            np.asarray(t, "<i4").tofile(f)
        for t in ([0, 0, 0], [1, 2.5e6, 1], [0], [1]):
            np.asarray(t, "<f8").tofile(f)
        for w, b in fold:
            w.astype("<f8").tofile(f)
            b.astype("<f8").tofile(f)
    raw = np.column_stack(
        (old / (1 + beta) + beta / (1 + beta) * a, np.log1p(beta), np.full(len(a), p))
    )
    x = 2 * (raw - z["low"]) / (z["high"] - z["low"]) - 1
    dx = np.zeros_like(x)
    dx[:, 0] = 2 * beta / (1 + beta)
    y, d = forward(layers, x, dx)
    xf = np.column_stack((x[:, :2], 2 * old - 1))
    dxf = np.column_stack((dx[:, :2], np.zeros(len(a))))
    yf, df = forward(fold, xf, dxf)
    assert max(abs(y - yf)) < 1e-12 and max(abs(d - df)) < 1e-10
    np.column_stack([a, beta, old, y, d]).astype("<f8").tofile(
        O / ("shared976_" + name + ".queries")
    )
    report[name] = dict(
        checkpoint_sha256=digest,
        parameters=int(k),
        exponent=p,
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        value_parity=float(max(abs(y - yf))),
        derivative_parity=float(max(abs(d - df))),
    )
if args.verify_paper:
    assert (
        report["intermediate"]["model_sha256"]
        == "a9f9011efbdcf842b5f013be36da62abc475e9f30633521a545bcb04132b8cb1"
    )
(O / "EXPORT_IDENTITY.json").write_text(json.dumps(report, indent=2))
