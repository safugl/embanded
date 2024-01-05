"""Linear algebra utilities for torch implementation."""

import math
import torch


def matrix_block_indexing(A, rows):
    """Get blocks in a matrix, A[rows,:][:,rows]."""
    return A[rows[:, None], rows]


def matrix_add_to_diagonal(A, B):
    """Add vector B to diagonal elements in A, i.e., A + torch.diag(B)."""
    torch.ravel(A)[::A.shape[1]+1] += B
    return A


def matrix_get_diagonal_elements(A):
    """Get diagonal elements in A, i.e,. torch.diag(A)."""
    return torch.ravel(A)[::A.shape[1]+1]


def matrix_inv_cholesky(A):
    """Approximate matrix inverse via the Cholesky decomposition."""
    L = torch.linalg.cholesky(A)  # pylint: disable=E1102
    return torch.cholesky_inverse(L)


def matrix_centering(A):
    """Take a matrix A and return a centered version."""
    mu = torch.mean(A, axis=0, keepdims=True)
    return A - mu, mu


def matern_type_kernel(num_dim: int, h: float,
                       dtype: torch.dtype,
                       device: torch.device):
    """See main paper for definition."""
    if (isinstance(h, float) is not True) or (h <= 0):
        raise TypeError(f'h must be a positive float, not {h}')
    x = torch.arange(num_dim, dtype=dtype, device=device)[None, ...]
    Omega = (
        (1 + math.sqrt(3) * torch.abs(x.T - x) / h) *
        torch.exp(-math.sqrt(3) * torch.abs(x.T - x) / h)
    )
    return Omega


def one_hot_encoding(A, dtype=torch.float64, device='cpu'):
    """Define one-hot encoding based on A.

    Examples
    --------
    A = torch.tensor([1, 1, 1, 4, 5, 6, 1])
    one_hot_encoding(A)
        tensor([[1., 0., 0., 0.],
                [1., 0., 0., 0.],
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
                [1., 0., 0., 0.]], dtype=torch.float64)
    A=torch.arange(5)
    one_hot_encoding(A)
        tensor([[1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 0., 1., 0., 0.],
                [0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 1.]], dtype=torch.float64)
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError('{A} must be a tensor')
    if not A.ndim == 1:
        raise TypeError('{A} must be one-dimensional')

    categories = torch.unique(A)

    num_categories = len(categories)

    mat = torch.zeros((len(A), num_categories), dtype=dtype, device=device)
    for j, val in enumerate(categories):
        mat[A == val, j] = 1.
    return mat
