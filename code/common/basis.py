"""Chebyshev / Legendre polynomial frequency basis.

Extracted from code/0601code/chebyshev_basis.py -- keeps only the
build_freq_basis function needed by MIONet.
"""
import torch


def build_freq_basis(freqs, order=16, basis_type="chebyshev"):
    """Build polynomial frequency basis tensor.

    Args:
        freqs: (N,) or (B, N) frequency tensor in [-1, 1]
        order: polynomial order K
        basis_type: "chebyshev" or "legendre"

    Returns:
        (..., K+1, N) basis tensor
    """
    if basis_type == "chebyshev":
        return _chebyshev_basis(freqs, order)
    elif basis_type == "legendre":
        return _legendre_basis(freqs, order)
    else:
        raise ValueError(f"Unknown basis_type: {basis_type}")


def _chebyshev_basis(freqs, order):
    """T_n(x) recurrence: T_0=1, T_1=x, T_n=2*x*T_{n-1} - T_{n-2}."""
    x = torch.as_tensor(freqs, dtype=torch.float32)
    basis = [torch.ones_like(x), x]
    for n in range(2, order + 1):
        basis.append(2 * x * basis[-1] - basis[-2])
    return torch.stack(basis, dim=-2)  # (..., K+1, N)


def _legendre_basis(freqs, order):
    """P_n(x) recurrence."""
    x = torch.as_tensor(freqs, dtype=torch.float32)
    basis = [torch.ones_like(x), x]
    for n in range(2, order + 1):
        p = ((2 * n - 1) * x * basis[-1] - (n - 1) * basis[-2]) / n
        basis.append(p)
    return torch.stack(basis, dim=-2)
