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
    return torch.einsum('ii->i', A)


def matrix_inv_cholesky(A, compute_logdet=False):
    """Approximate matrix inverse via the Cholesky decomposition.

    Examples
    --------
    Define a tensor, A as follows:
        tensor([[9., 3., 1., 5.],
                [3., 7., 5., 1.],
                [1., 5., 9., 2.],
                [5., 1., 2., 6.]], dtype=torch.float64)
     
    A = torch.tensor([[9, 3, 1, 5], [3, 7, 5, 1], 
                      [1, 5, 9, 2], [5, 1, 2, 6]],dtype=torch.float64)
    # Check matrix.
    assert np.isclose(A,A.T).all()

    O, d = matrix_inv_cholesky(A,True)
    B = np.linalg.inv(A)
    np.testing.assert_allclose(O,B)
    np.testing.assert_allclose(d,np.linalg.slogdet(B)[1])
    np.testing.assert_allclose(d,np.log(np.linalg.det(B)))
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError('{A} must have size (D x D)')
    if not A.ndim == 2:
        raise TypeError('{A} must have size (D x D)')
    if not A.shape[0] == A.shape[1]:
        raise TypeError('{A} must have size (D x D)')
    L = torch.linalg.cholesky(A)  # pylint: disable=E1102
    O = torch.cholesky_inverse(L)

    if compute_logdet is True:
        logdet = - 2 * torch.sum(torch.log(torch.diag(L)))
    else:
        logdet = None

    return O, logdet


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


def matrix_trace_of_product(A, B):
    """Return torch.trace(A@B)."""
    if not A.shape[0] == B.shape[1]:
        raise TypeError('{A} and {B.T} must have size (M x N)')
    if not A.shape[1] == B.shape[0]:
        raise TypeError('{A} and {B.T} must have size (M x N)')
    return torch.einsum('ij,ji->', A, B)


def matrix_blockdiag_rotation(A, mat_indexer, B=None):
    """Take a vector A of size (D x 1) and a block diagonal matrix, B, of size
    (D x D) and estimate [A_1'@B_11@A_1, ..., A_f'@B_ff@A_f, ...], where f 
    indicates block index. If B is empty, then it is assumed to be the 
    identity matrix. This only works since blocks in B coincides with indexes
    in mat_indexer. The below examples illustrates this behavior.
    
    Examples
    --------
    The following examples illustrate the behavior when B is a tensor:
        tensor([[1.0000, 2.0000, 3.0000, 0.0000, 0.0000],
                [2.0000, 4.0000, 6.0000, 0.0000, 0.0000],
                [3.0000, 6.0000, 9.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.7081, 0.7651],
                [0.0000, 0.0000, 0.0000, 0.7651, 0.8268]])
        
    and when mat_indexer is an array:
        tensor([[1., 0.],
                [1., 0.],
                [1., 0.],
                [0., 1.],
                [0., 1.]], dtype=torch.float64)
    
    Example 1:
        mat_indexer = one_hot_encoding(torch.tensor([1, 1, 1, 2, 2]))
        A = torch.randn(5,1,dtype=torch.float64)
        B11 = torch.arange(1,4)[:,None]@torch.arange(1,4)[None,:]
        B22 = torch.sin(torch.arange(1,3)[:,None])@torch.sin(torch.arange(1,3)[None,:])
        B = torch.cat([torch.cat([B11,torch.zeros((3,2))],dim=1), 
                            torch.cat([torch.zeros((2,3)),B22],dim=1)],dim=0)
        B = B.to(dtype=torch.float64)
        B22 = B22.to(dtype=torch.float64)
        B11 = B11.to(dtype=torch.float64)
        estimate = matrix_blockdiag_rotation(A,mat_indexer,B);
        compare = torch.cat([A[:3].T@B11@A[:3], A[3:].T@B22@A[3:]],dim=1)
        torch.testing.assert_close(compare,estimate)
        
    Example 2:
        compare = torch.cat([A[:3].T@A[:3], A[3:].T@A[3:]],dim=1)
        estimate = matrix_blockdiag_rotation(A,mat_indexer)
        torch.testing.assert_close(compare,estimate)
        
    Notes
    --------
    An alternative approach could have been:
        O = (A.T@B)@torch.multiply(A, mat_indexer)
        % When B = None
        O = torch.pow(A,2).T@ mat_indexer
        
    
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError('{A} must have size (D X 1)')
    if not A.shape[1] == 1:
        raise TypeError('{A} must have size (D X 1)')

    if B is not None:
        if not isinstance(B, torch.Tensor):
            raise TypeError('{B} must have size (D X D)')
        if not (A.shape[0], A.shape[0]) == B.shape:
            raise TypeError('{B} must have size (D x D)')
        O = torch.einsum('ij,jk->ik',
                         torch.einsum('ji,jk->ik', A, B),
                         torch.einsum('ji,jk->jk', A, mat_indexer))
    else:
        O = torch.einsum('ji,jk->ik', torch.pow(A, 2), mat_indexer)

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
    The following examples illustrate the behavior when B is a tensor:
        tensor([[1.0000, 2.0000, 3.0000, 0.0000, 0.0000],
                [2.0000, 4.0000, 6.0000, 0.0000, 0.0000],
                [3.0000, 6.0000, 9.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.7081, 0.7651],
                [0.0000, 0.0000, 0.0000, 0.7651, 0.8268]])
        
    and when mat_indexer is an array:
        tensor([[1., 0.],
                [1., 0.],
                [1., 0.],
                [0., 1.],
                [0., 1.]], dtype=torch.float64)
    
    
    Example 1:
        mat_indexer = one_hot_encoding(torch.tensor([1, 1, 1, 2, 2]))
        A = torch.randn(5,1,dtype=torch.float64)
        A = A@A.T
        B11 = torch.arange(1,4)[:,None]@torch.arange(1,4)[None,:]
        B22 = torch.sin(torch.arange(1,3)[:,None])@torch.sin(torch.arange(1,3)[None,:])
        B = torch.cat([torch.cat([B11,torch.zeros((3,2))],dim=1), 
                            torch.cat([torch.zeros((2,3)),B22],dim=1)],dim=0)
        B = B.to(dtype=torch.float64)
        B22 = B22.to(dtype=torch.float64)
        B11 = B11.to(dtype=torch.float64)
        
        estimate = matrix_block_trace(A,mat_indexer,B)
        compare = torch.cat([torch.trace(A[:3,:3].T@B11).ravel(), 
                             torch.trace(A[3:,3:].T@B22).ravel()])
        torch.testing.assert_close(estimate,compare)
        
    
    Example 2:
        compare = torch.cat([torch.trace(A[:3,:3]).ravel(), 
                             torch.trace(A[3:,3:]).ravel()])
        estimate = matrix_block_trace(A,mat_indexer)
        torch.testing.assert_close(estimate,compare)
        
    Notes
    --------
    An alternative approach could have been:
        O = matrix_get_diagonal_elements(B@A)@mat_indexer
        % When B = None
        O = matrix_get_diagonal_elements(A)@mat_indexer
         
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError('{A} must have size (D X D)')
    if not A.shape[1] == A.shape[0]:
        raise TypeError('{A} must have size (D X D)')

    if B is not None:
        if not isinstance(B, torch.Tensor):
            raise TypeError('{B} must have size (D X D)')
        if not (A.shape[0], A.shape[0]) == B.shape:
            raise TypeError('{B} must have size (D x D)')
        O = torch.einsum('ii,ij->j',
                         torch.einsum('ik,kj->ij', A, B),
                         mat_indexer)
    else:
        O = torch.einsum('ii,ij->j', A, mat_indexer)

    return O
