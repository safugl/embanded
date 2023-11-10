"""embanded model utils"""

import copy
import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve


def matrix_block_indexing(A, rows):
    """Get blocks in a matrix, A[rows,:][:,rows]"""
    return A[rows[:, None], rows]


def matrix_add_to_diagonal(A, B):
    """Add vector B to diagonal elements in A, i.e., A + np.diag(B)"""
    A.ravel()[::A.shape[1]+1] += B
    return A


def matrix_get_diagonal_elements(A):
    """Get diagonal elements in A, i.e,. np.diag(A)"""
    return A.ravel()[::A.shape[1]+1]


def matrix_inv_cholesky(A):
    """Approximate matrix inverse via the Cholesky decomposition. """
    L = cho_factor(A, lower=True, overwrite_a=False, check_finite=False)
    I = np.eye(A.shape[1])
    return cho_solve(L, I, overwrite_b=False, check_finite=False)


def matrix_centering(A):
    """Takes a matrix A, copies it and returns a centered version"""
    B = copy.deepcopy(A)
    B_offset = B.mean(axis=0, keepdims=True)
    B -= B_offset
    return B, B_offset


def matern_type_kernel(num_dim: int, h: float):
    """See main paper for definition"""
    if (isinstance(h, float) is not True) or (h <= 0) or (h == np.inf):
        raise TypeError(f'h must be a positive float, not {h}')

    x = np.arange(num_dim)[None, ...]
    Omega = (
        (1 + np.sqrt(3) * np.abs(x.T - x) / h) *
        np.exp(-np.sqrt(3) * np.abs(x.T - x) / h)
    )
    return Omega


def one_hot_encoding(input_vector):
    """Transform a vector of ints into a one hot encoding."""
    predictor_groups = np.unique(input_vector)
    assert (predictor_groups == np.arange(
        len(predictor_groups))).all(), "Wrong usage."
    num_groups = len(predictor_groups)

    mat = np.zeros((len(input_vector), num_groups), dtype=np.float64)
    for j in range(num_groups):
        mat[input_vector == j, j] = 1
    return mat


def prepare_smoothness_cov(F, smoothness_param):
    """Create Omega and Omega_inv from input list F and smoothness_param"""

    num_features = len(F)
    Omega = []
    Omega_inv = []
    for j in range(num_features):

        #  The number of columns in F{f}
        D_j = F[j].shape[1]

        # The smoothness parameter
        h_j = smoothness_param[j]

        if h_j is None:
            # If it is specified as NaN then assume that it
            # should be an identity matrix
            Omega_j = np.eye(D_j)
            Omega_j_inv = np.eye(D_j)
        else:

            # When h[j] is a scalar then define Omega as follows:
            Omega_j = matern_type_kernel(D_j, h_j)
            Omega_j_inv = matrix_inv_cholesky(Omega_j)

        Omega.append(Omega_j)
        Omega_inv.append(Omega_j_inv)

    # Prepare a matrix over entire Lambda
    Omega = scipy.linalg.block_diag(*Omega)

    # The inverse of this matrix
    Omega_inv = scipy.linalg.block_diag(*Omega_inv)

    return Omega, Omega_inv


def get_hyperparams_from_tuple(hyper_params):
    """Grab hyper parameters from tuple."""
    eta = hyper_params[0]
    tau = hyper_params[1]
    phi = hyper_params[2]
    kappa = hyper_params[3]
    return eta, tau, phi, kappa


def fit_model_without_smoothness(X, y,
                                 hyper_params,
                                 initialization_params,
                                 max_iterations,
                                 mat_indexer):
    """Default model fit without smoothness"""

    covXy = X.T @ y
    covX = X.T @ X

    num_feat = len(initialization_params[0])
    num_obs = X.shape[0]

    # Store lambdas and nu parameters
    summary = {"lambdas": np.zeros((max_iterations, num_feat), dtype=X.dtype),
               "nu": np.zeros(max_iterations, dtype=X.dtype)}

    eta, tau, phi, kappa = get_hyperparams_from_tuple(hyper_params)

    # Initialize the parameters
    lambdas = copy.deepcopy(initialization_params[0])
    nu = copy.deepcopy(initialization_params[1])

    for iteration in range(max_iterations):

        # Store in the summaries
        summary['lambdas'][iteration, :] = lambdas
        summary['nu'][iteration] = nu

        # Compute Sigma
        Sigma = compute_covariance(nu, covX, lambdas, mat_indexer)

        Sigma_diags = matrix_get_diagonal_elements(Sigma)

        W = 1.0 / nu * np.matmul(Sigma, covXy)

        if iteration < max_iterations-1:
            D_j = np.sum(mat_indexer, axis=0)
            E = np.power(W, 2).T@mat_indexer + Sigma_diags@mat_indexer
            lambdas = ((E + 2 * tau) / (D_j + 2 * eta + 2)).ravel()

            r = y - X @ W

            nu = ((r.T @ r).item() + np.trace(covX @ Sigma) +
                  2 * kappa) / (num_obs + 2 + 2 * phi)

    return W, summary, Sigma


def fit_model_with_smoothness(X, y,
                              hyper_params,
                              initialization_params,
                              max_iterations,
                              mat_indexer,
                              Omega_inv):
    """Default model fit with smoothness"""

    covXy = X.T @ y
    covX = X.T @ X

    num_feat = len(initialization_params[0])
    num_obs = X.shape[0]

    # Store lambdas and nu parameters
    summary = {"lambdas": np.zeros((max_iterations, num_feat), dtype=X.dtype),
               "nu": np.zeros(max_iterations, dtype=X.dtype)}

    eta, tau, phi, kappa = get_hyperparams_from_tuple(hyper_params)

    # Initialize the parameters
    lambdas = copy.deepcopy(initialization_params[0])
    nu = copy.deepcopy(initialization_params[1])

    for iteration in range(max_iterations):

        # Store in the summaries
        summary['lambdas'][iteration, :] = lambdas
        summary['nu'][iteration] = nu

        # Compute Sigma
        Sigma = compute_covariance(nu, covX, lambdas, mat_indexer, Omega_inv)

        W = 1.0 / nu * np.matmul(Sigma, covXy)

        if iteration < max_iterations-1:

            # In this case we also have to iterate over feature groups
            for f in range(num_feat):

                index = np.where(mat_indexer[:, f] == 1)[0]
                D_f = np.sum(mat_indexer[:, f] == 1)

                # Get rows in W
                mu_f = W[index]

                # Get blocks in Sigma and Omega_inv
                Sigma_ff = matrix_block_indexing(Sigma, index)
                Omega_inv_ff = matrix_block_indexing(Omega_inv, index)

                # Update lambdas
                lambdas[f] = ((mu_f.T @ Omega_inv_ff @ mu_f
                               + np.trace(Omega_inv_ff @ Sigma_ff)).item()
                              + 2 * tau) / (D_f + 2 * eta + 2)

            r = y - X @ W

            nu = ((r.T @ r).item() + np.trace(covX @ Sigma) +
                  2 * kappa) / (num_obs + 2 + 2 * phi)

    return W, summary, Sigma


def fit_model_multidimensional(X, y,
                               hyper_params,
                               initialization_params,
                               max_iterations,
                               mat_indexer,
                               Omega_inv):
    """Muldimensional model fit with- or withoout smoothness"""

    covXy = X.T @ y
    covX = X.T @ X

    num_feat = len(initialization_params[0])
    num_obs = X.shape[0]

    # Store lambdas and nu parameters
    summary = {"lambdas": np.zeros((max_iterations, num_feat), dtype=X.dtype),
               "nu": np.zeros(max_iterations, dtype=X.dtype)}

    eta, tau, phi, kappa = get_hyperparams_from_tuple(hyper_params)

    # Initialize the parameters
    lambdas = copy.deepcopy(initialization_params[0])
    nu = copy.deepcopy(initialization_params[1])

    for iteration in range(max_iterations):

        # Store in the summaries
        summary['lambdas'][iteration, :] = lambdas
        summary['nu'][iteration] = nu

        # Compute Sigma
        Sigma = compute_covariance(nu, covX, lambdas, mat_indexer, Omega_inv)

        # Compute W
        W = 1.0 / nu * np.matmul(Sigma, covXy)

        if iteration < max_iterations-1:

            # In this case we also have to iterate over feature groups
            for f in range(num_feat):

                index = np.where(mat_indexer[:, f] == 1)[0]
                D_f = np.sum(mat_indexer[:, f] == 1)

                # Get rows in W
                mu_f = W[index]

                # Get blocks in Sigma and Omega_inv
                Sigma_ff = matrix_block_indexing(Sigma, index)

                if Omega_inv is not None:
                    Omega_inv_ff = matrix_block_indexing(Omega_inv, index)

                    # Update with smoothness term
                    E = (np.trace(mu_f @ mu_f.T @ Omega_inv_ff) +
                         np.trace(Omega_inv_ff @ Sigma_ff))

                else:
                    # Update without smoothness
                    E = (np.trace(mu_f @ mu_f.T) + np.trace(Sigma_ff))

                # Update lambdas
                lambdas[f] = (E + 2 * tau) / (y.shape[1] * D_f + 2 * eta + 2)

            # Make point-predictions
            prediction = X @ W

            # Flatten prediction and target
            target = np.reshape(y, (y.shape[0]*y.shape[1], 1))
            prediction = np.reshape(prediction, (y.shape[0]*y.shape[1], 1))

            # Update nu
            nu = ((target - prediction).T @ (target - prediction)).item()
            nu += np.trace(covX @ Sigma) + 2 * kappa
            nu /= (num_obs + 2 + 2 * phi)

    return W, summary, Sigma


def compute_covariance(nu, covX, lambdas, mat_indexer, Omega_inv=None):
    """Compute inv( 1/nu  X.T X + L Omega_inv)"""

    # Create a vector that will form the diagonal elements in the L matrix
    lambdas_diag = mat_indexer@lambdas

    if Omega_inv is not None:
        L = 1.0 / nu * covX
        L += np.expand_dims(1.0/lambdas_diag, -1) * Omega_inv
    else:
        L = matrix_add_to_diagonal(1.0 / nu * covX, 1 / lambdas_diag)

    # Compute Sigma
    Sigma = matrix_inv_cholesky(L)

    return Sigma
