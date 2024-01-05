"""Linear algebra utilities for numpy implementation."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def matrix_block_indexing(A, rows):
    """Get blocks in a matrix, A[rows,:][:,rows]."""
    return A[rows[:, None], rows]


def matrix_add_to_diagonal(A, B):
    """Add vector B to diagonal elements in A, i.e., A + np.diag(B)."""
    A.ravel()[::A.shape[1]+1] += B
    return A


def matrix_get_diagonal_elements(A):
    """Get diagonal elements in A, i.e,. np.diag(A)."""
    return A.ravel()[::A.shape[1]+1]


def matrix_inv_cholesky(A):
    """Approximate matrix inverse via the Cholesky decomposition."""
    L = cho_factor(A, lower=True, overwrite_a=False, check_finite=False)
    return cho_solve(L, np.eye(A.shape[1],dtype=A.dtype),
                     overwrite_b=False,
                     check_finite=False)


def matrix_centering(A):
    """Take a matrix A and return a centered version."""
    mu = np.mean(A, axis=0, keepdims=True)
    return A - mu, mu


def matern_type_kernel(num_dim: int, h: float, dtype=np.float64):
    """See main paper for definition."""
    if (isinstance(h, float) is not True) or (h <= 0) or (h == np.inf):
        raise TypeError(f'h must be a positive float, not {h}')
    x = np.arange(num_dim, dtype=dtype)[None, ...]
    Omega = (
        (1 + np.sqrt(3) * np.abs(x.T - x) / h) *
        np.exp(-np.sqrt(3) * np.abs(x.T - x) / h)
    )
    return Omega


def one_hot_encoding(A, dtype=np.float64):
    """Define one-hot encoding based on A.

    Examples
    --------
    A = np.array([1, 1, 1, 4, 5, 6, 1])
    one_hot_encoding(A):
        array([[1., 0., 0., 0.],
               [1., 0., 0., 0.],
               [1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.],
               [1., 0., 0., 0.]])

    A=np.arange(5)
    one_hot_encoding(A)
        array([[1., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0.],
               [0., 0., 1., 0., 0.],
               [0., 0., 0., 1., 0.],
               [0., 0., 0., 0., 1.]])
    """
    if not isinstance(A, np.ndarray):
        raise TypeError('{A} must be an array')
    if not A.ndim == 1:
        raise TypeError('{A} must be one-dimensional')

    categories = np.unique(A)

    num_categories = len(categories)

    mat = np.zeros((len(A), num_categories), dtype=dtype)
    for j, val in enumerate(categories):
        mat[A == val, j] = 1.
    return mat
