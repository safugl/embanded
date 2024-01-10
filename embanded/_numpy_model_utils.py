"""Model utilities for numpy implementation."""

import copy
import time
import numpy as np
import scipy

from ._numpy_linalg_utils import (
    one_hot_encoding, matern_type_kernel, matrix_inv_cholesky,
    matrix_add_to_diagonal, matrix_block_indexing, matrix_trace_of_product,
    matrix_blockdiag_rotation, matrix_block_trace,
)


def get_hyperparams_from_tuple(hyper_params):
    """Grab hyper parameters from tuple."""
    eta = hyper_params[0]
    tau = hyper_params[1]
    phi = hyper_params[2]
    kappa = hyper_params[3]

    return eta, tau, phi, kappa


def create_matrix_indexer(F, dtype=np.float64):
    """Create a matrix for indexing columns/rows in Sigma based on F.

    Example
    -------
    F = [np.random.randn(10,2), np.random.randn(10,3), np.random.randn(10,2)]
    create_matrix_indexer(F)
        array([[1., 0., 0.],
               [1., 0., 0.],
               [0., 1., 0.],
               [0., 1., 0.],
               [0., 1., 0.],
               [0., 0., 1.],
               [0., 0., 1.]])
    """
    # Create a column vector that contains indices
    columns_group = [np.ones(F[j].shape[1], dtype=dtype) * j
                     for j in range(len(F))]
    columns_group = np.concatenate(columns_group, axis=0).ravel()

    # Create a matrix that indexes predictor groups
    mat_indexer = one_hot_encoding(columns_group, dtype)
    return mat_indexer


def prepare_smoothness_cov(F, smoothness_param, dtype=np.float64):
    """Use F and smoothness_param to define Omega and Omega_inv.

    Example
    -------
    F = [np.random.randn(10,2), np.random.randn(10,5)]
    O, O_i = prepare_smoothness_cov(F,[None, 2.])
    print(np.round(O,4))
        [[1.     0.     0.     0.     0.     0.     0.    ]
         [0.     1.     0.     0.     0.     0.     0.    ]
         [0.     0.     1.     0.7849 0.4834 0.2678 0.1397]
         [0.     0.     0.7849 1.     0.7849 0.4834 0.2678]
         [0.     0.     0.4834 0.7849 1.     0.7849 0.4834]
         [0.     0.     0.2678 0.4834 0.7849 1.     0.7849]
         [0.     0.     0.1397 0.2678 0.4834 0.7849 1.    ]]
    """
    num_features = len(F)
    if len(smoothness_param) != num_features:
        raise TypeError('Unexpected length of {smoothness_param}')
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
            Omega_j = np.eye(D_j, dtype=dtype)
            Omega_j_inv = np.eye(D_j, dtype=dtype)
        else:

            # When h[j] is a scalar then define Omega as follows:
            Omega_j = matern_type_kernel(D_j, h_j, dtype=dtype)
            Omega_j_inv = matrix_inv_cholesky(Omega_j)

        Omega.append(Omega_j)
        Omega_inv.append(Omega_j_inv)

    # Prepare a matrix over entire Lambda
    Omega = scipy.linalg.block_diag(*Omega)

    # The inverse of this matrix
    Omega_inv = scipy.linalg.block_diag(*Omega_inv)

    return Omega, Omega_inv


def compute_covariance(nu, covX, lambdas, mat_indexer, Omega_inv=None):
    """Compute inv( 1/nu  X.T X + L Omega_inv)."""
    # Create a vector that will form the diagonal 1/lambda_j elements
    lambdas_diag_inv = mat_indexer @ (1./lambdas)

    if Omega_inv is not None:
        L = 1.0 / nu * covX + lambdas_diag_inv[:, None] * Omega_inv
    else:
        L = matrix_add_to_diagonal(1.0 / nu * covX, lambdas_diag_inv)

    # Compute Sigma
    Sigma = matrix_inv_cholesky(L)

    return Sigma


def fit_model_vectorized(X, y,
                         hyper_params,
                         initialization_params,
                         max_iterations,
                         mat_indexer,
                         Omega_inv,
                         verbose):
    """Fit a model and assume that y is 1D."""
    if verbose is True:
        if Omega_inv is None:
            print('Fitting a model without smoothness, assuming 1D y.')
        else:
            print('Fitting a model with smoothness, assuming 1D y.')

    covXy = X.T @ y
    covX = X.T @ X
    # covX_flat = covX.T.reshape(-1)
    num_feat = len(initialization_params[0])
    num_obs = X.shape[0]

    # Store lambdas and nu parameters
    summary = {"lambdas": np.zeros((max_iterations, num_feat), dtype=X.dtype),
               "nu": np.zeros(max_iterations, dtype=X.dtype)}

    eta, tau, phi, kappa = get_hyperparams_from_tuple(hyper_params)

    # Initialize the parameters
    lambdas = copy.deepcopy(initialization_params[0])
    nu = copy.deepcopy(initialization_params[1])

    if verbose is True:
        print(f'nu is initialized to {nu}.')
        print(f'lambdas are initialized to {lambdas}.')
        start_time = time.time()

    # Number of predictors in each group
    D = np.sum(mat_indexer, axis=0)

    for iteration in range(max_iterations):

        # Store in the summaries
        summary['lambdas'][iteration, :] = lambdas
        summary['nu'][iteration] = nu

        # Compute Sigma
        Sigma = compute_covariance(nu, covX, lambdas, mat_indexer, Omega_inv)

        # Estimate weigths
        W = 1.0 / nu * np.matmul(Sigma, covXy)

        if iteration < max_iterations-1:

            # Update the lambdas.
            lambdas = (
                (
                    matrix_blockdiag_rotation(W, mat_indexer, Omega_inv) +
                    matrix_block_trace(Sigma, mat_indexer, Omega_inv) +
                    2 * tau) / (D + 2 * eta + 2)
            ).ravel()

            nu = (
                (np.sum(np.power(y - X @ W, 2)) +
                 matrix_trace_of_product(covX, Sigma) +
                 2 * kappa) / (num_obs + 2 + 2 * phi)
            )

        if verbose is True:
            print(f'At iteration {iteration} of {max_iterations}')
            time_elapsed = time.time()-start_time
            print(f'Time elapsed: {time_elapsed}')

    return W, summary, Sigma


def fit_model_multidimensional(X, y,
                               hyper_params,
                               initialization_params,
                               max_iterations,
                               mat_indexer,
                               Omega_inv,
                               verbose):
    """Muldimensional model fit with- or without smoothness."""
    if verbose is True:
        if Omega_inv is None:
            print('Fitting a *multi_dimensional* model without smoothness.')
        else:
            print('Fitting a *multi_dimensional* model with smoothness.')

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

    if verbose is True:
        print(f'nu is initialized to {nu}.')
        print(f'lambdas are initialized to {lambdas}.')
        start_time = time.time()

    # The number of predictors in each predictor group
    D = np.sum(mat_indexer, axis=0)

    # Prepare indices for block indexing, and prepare Omega_inv terms for each
    # predictor group
    indices = []
    Omega_inv_groups = []
    for f in range(mat_indexer.shape[1]):
        index = np.where(mat_indexer[:, f] == 1)[0]
        indices.append(index)

        if Omega_inv is not None:
            Omega_inv_groups.append(matrix_block_indexing(Omega_inv, index))

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

                index = indices[f]
                D_f = D[f]

                # Get rows in W
                mu_f = W[index]

                # Get blocks in Sigma and Omega_inv
                Sigma_ff = matrix_block_indexing(Sigma, index)

                if Omega_inv is not None:
                    Omega_inv_ff = Omega_inv_groups[f]

                    # Update with smoothness term
                    if mu_f.shape[1] > mu_f.shape[0]:
                        E = (np.trace(mu_f @ mu_f.T @ Omega_inv_ff) +
                             matrix_trace_of_product(Omega_inv_ff, Sigma_ff))
                    else:
                        E = (np.trace(mu_f.T @ Omega_inv_ff @ mu_f) +
                             matrix_trace_of_product(Omega_inv_ff, Sigma_ff))

                else:
                    # Update without smoothness
                    E = (
                        (matrix_trace_of_product(mu_f, mu_f.T) +
                         np.trace(Sigma_ff))
                    )

                # Update lambdas
                lambdas[f] = (E + 2 * tau) / (y.shape[1] * D_f + 2 * eta + 2)

            # Make point-predictions
            prediction = X @ W

            # Estimate residual sum of squares
            rss = np.sum(np.power(y - prediction, 2))

            # Update nu
            nu = (
                (rss + matrix_trace_of_product(covX, Sigma) + 2 * kappa)
                / (num_obs + 2 + 2 * phi)
            )

        if verbose is True:
            print(f'At iteration {iteration} of {max_iterations}')
            time_elapsed = time.time()-start_time
            print(f'Time elapsed: {time_elapsed}')

    return W, summary, Sigma
