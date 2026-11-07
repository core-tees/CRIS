import torch


def evidence(G, v, mean_square, reg, n):
    A = (G + G.T) * 0.5
    A.diagonal().add_(reg)
    scale = A.diagonal().sqrt()
    C = A / scale[:, None] / scale[None, :]
    L, info = torch.linalg.cholesky_ex(C)
    method = "scaled_cholesky"
    min_eigen = None
    if int(info) == 0:
        trace = float((torch.cholesky_inverse(L).diagonal() / scale.square()).sum())
        gamma = max(1e-9, min(len(v), len(v) - reg * trace))
    else:
        # PSD Gram eigenvalues may be slightly negative through roundoff.
        eig = torch.linalg.eigvalsh(((G + G.T) * 0.5).cpu())
        min_eigen = float(eig.min())
        largest = float(eig.max())
        if min_eigen < -1e-10 * max(largest, 1.0):
            raise RuntimeError("Gram has material negative eigenvalues")
        eig = eig.clamp(min=0)
        gamma = float((eig / (eig + reg)).sum())
        method = "psd_eigenvalue_fallback"
    alpha = gamma / max(float(v @ v), 1e-30)
    beta = (n - gamma) / max(n * mean_square, 1e-30)
    return alpha / (beta * n), dict(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lambda_sum=alpha / beta,
        lambda_mean=alpha / (beta * n),
        trace_method=method,
        min_eigenvalue=min_eigen,
    )
