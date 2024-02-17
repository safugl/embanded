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
    return np.einsum('ii->i', A)


def matrix_inv_cholesky(A,compute_logdet=False):
    """Approximate matrix inverse via the Cholesky decomposition.
    
    Examples
    --------
    Define an array, A as follows:
        array([[9, 3, 1, 5],
               [3, 7, 5, 1],
               [1, 5, 9, 2],
               [5, 1, 2, 6]])
     
    A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
    # Check matrix.
    assert np.isclose(A,A.T).all()
    
    O, d = matrix_inv_cholesky(A,True)
    B = np.linalg.inv(A)
    np.testing.assert_allclose(O,B)
    np.testing.assert_allclose(d,np.linalg.slogdet(B)[1])
    np.testing.assert_allclose(d,np.log(np.linalg.det(B)))
    """
    if not isinstance(A, np.ndarray):
        raise TypeError('{A} must have size (D x D)')
    if not A.ndim == 2:
        raise TypeError('{A} must have size (D x D)')
    if not A.shape[0] == A.shape[1]:
        raise TypeError('{A} must have size (D x D)')
    L = cho_factor(A, lower=True, overwrite_a=False, check_finite=False)
    O = cho_solve(L, np.eye(A.shape[1], dtype=A.dtype),
                     overwrite_b=False,
                     check_finite=False)

    if compute_logdet is True:
        logdet = - 2 * np.sum(np.log(np.diag(L[0])))
    else:
        logdet = None

    return O, logdet


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


def matrix_trace_of_product(A, B):
    """Return np.trace(A@B)."""
    if not A.shape[0] == B.shape[1]:
        raise TypeError('{A} and {B.T} must have size (M x N)')
    if not A.shape[1] == B.shape[0]:
        raise TypeError('{A} and {B.T} must have size (M x N)')
    # Notice that A and B have to be real matrices
    return np.einsum('ij,ji->', A, B)


def matrix_blockdiag_rotation(A, mat_indexer, B=None):
    """Take a vector A of size (D x 1) and a block diagonal matrix, B, of size
    (D x D) and estimate [A_1'@B_11@A_1, ..., A_f'@B_ff@A_f, ...], where f 
    indicates block index. If B is empty, then it is assumed to be the 
    identity matrix. This only works since blocks in B coincides with indexes
    in mat_indexer. The below examples illustrates this behavior.
    
    Examples
    --------
    The following examples illustrate the behavior when B is an array:
        array([[1.        , 2.        , 3.        , 0.        , 0.        ],
               [2.        , 4.        , 6.        , 0.        , 0.        ],
               [3.        , 6.        , 9.        , 0.        , 0.        ],
               [0.        , 0.        , 0.        , 0.70807342, 0.7651474 ],
               [0.        , 0.        , 0.        , 0.7651474 , 0.82682181]])
        
    and when mat_indexer is an array:
        array([[1., 0.],
               [1., 0.],
               [1., 0.],
               [0., 1.],
               [0., 1.]])
    
    Example 1:
        mat_indexer = one_hot_encoding(np.array([1, 1, 1, 2, 2]))
        A = np.random.randn(5,1)
        B11 = np.arange(1,4)[:,None]@np.arange(1,4)[None,:]
        B22 = np.sin(np.arange(1,3)[:,None])@np.sin(np.arange(1,3)[None,:])
        B = np.concatenate([np.c_[B11,np.zeros((3,2))], 
                            np.c_[np.zeros((2,3)),B22]],axis=0)
    
        estimate = matrix_blockdiag_rotation(A,mat_indexer,B);
        compare = np.c_[A[:3].T@B11@A[:3], A[3:].T@B22@A[3:]]
        np.testing.assert_allclose(compare,estimate)
        
    Example 2:
        compare = np.c_[A[:3].T@A[:3], A[3:].T@A[3:]]
        estimate = matrix_blockdiag_rotation(A,mat_indexer)
        np.testing.assert_allclose(compare,estimate)
        
    Notes
    --------
    An alternative approach could have been:
        O = (A.T@B)@np.multiply(A, mat_indexer)
        % When B = None
        O = np.power(A,2).T@ mat_indexer
    """
    if not isinstance(A, np.ndarray):
        raise TypeError('{A} must have size (D X 1)')
    if not A.shape[1] == 1:
        raise TypeError('{A} must have size (D X 1)')

    if B is not None:
        if not isinstance(B, np.ndarray):
            raise TypeError('{B} must have size (D X D)')
        if not (A.shape[0], A.shape[0]) == B.shape:
            raise TypeError('{B} must have size (D x D)')
        O = np.einsum('ij,jk->ik',
                      np.einsum('ji,jk->ik', A, B),
                      np.einsum('ji,jk->jk', A, mat_indexer))
    else:
        O = np.einsum('ji,jk->ik', np.power(A, 2), mat_indexer)
    return O


def matrix_block_trace(A, mat_indexer, B=None):
    """Take a matrix A and a block diagonal matrix, B, and estimate
    [trace(A_1@B_11), ..., trace(A_f@B_ff), ...], where f indicates block 
    index. If B is empty, then it is assumed to be the identity matrix. Both
    A and B will have size (D x D). This only works since blocks in B 
    coincides with indexes in mat_indexer. The below examples illustrates 
    this behavior.
    
    Examples
    --------
    The following examples illustrate the behavior when B is an array:
        array([[1.        , 2.        , 3.        , 0.        , 0.        ],
               [2.        , 4.        , 6.        , 0.        , 0.        ],
               [3.        , 6.        , 9.        , 0.        , 0.        ],
               [0.        , 0.        , 0.        , 0.70807342, 0.7651474 ],
               [0.        , 0.        , 0.        , 0.7651474 , 0.82682181]])
        
    and when mat_indexer is an array:
        array([[1., 0.],
               [1., 0.],
               [1., 0.],
               [0., 1.],
               [0., 1.]])
    
    Example 1:
        mat_indexer = one_hot_encoding(np.array([1, 1, 1, 2, 2]))
        A = np.random.randn(5,1)
        A = A@A.T
        B11 = np.arange(1,4)[:,None]@np.arange(1,4)[None,:]
        B22 = np.sin(np.arange(1,3)[:,None])@np.sin(np.arange(1,3)[None,:])
        B = np.concatenate([np.c_[B11,np.zeros((3,2))], 
                            np.c_[np.zeros((2,3)),B22]],axis=0)
    
        estimate = matrix_block_trace(A,mat_indexer,B)
        compare = np.c_[np.trace(A[:3,:3].T@B11), np.trace(A[3:,3:].T@B22)]
        np.testing.assert_allclose(compare.ravel(),estimate.ravel())
        
    Example 2:
        compare = np.c_[np.trace(A[:3,:3]), np.trace(A[3:,3:])]
        estimate = matrix_block_trace(A,mat_indexer)
        np.testing.assert_allclose(compare.ravel(),estimate.ravel())
        
    Notes
    --------
    An alternative approach could have been:
        O = matrix_get_diagonal_elements(B@A)@mat_indexer
        % When B = None
        O = matrix_get_diagonal_elements(A)@mat_indexer
        
    
    """
    if not isinstance(A, np.ndarray):
        raise TypeError('{A} must have size (D X D)')
    if not A.shape[1] == A.shape[0]:
        raise TypeError('{A} must have size (D X D)')

    if B is not None:
        if not isinstance(B, np.ndarray):
            raise TypeError('{B} must have size (D X D)')
        if not (A.shape[0], A.shape[0]) == B.shape:
            raise TypeError('{B} must have size (D x D)')
        O = np.einsum('ii,ij->j',
                      np.einsum('ik,kj->ij', A, B),
                      mat_indexer)
    else:
        O = np.einsum('ii,ij->j', A, mat_indexer)

    return O
