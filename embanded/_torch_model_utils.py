"""Model utilities for torch implementation."""

import copy
import time
import math
import torch

from ._torch_linalg_utils import (
    one_hot_encoding, matern_type_kernel, matrix_inv_cholesky,
    matrix_block_indexing, matrix_trace_of_product,
    matrix_blockdiag_rotation, matrix_block_trace,
)


def get_hyperparams_from_tuple(hyper_params):
    """Grab hyper parameters from tuple."""
    eta = hyper_params[0]
    tau = hyper_params[1]
    phi = hyper_params[2]
    kappa = hyper_params[3]
    return eta, tau, phi, kappa


def create_matrix_indexer(F,  dtype=torch.float64, device='cpu'):
    """Create a matrix for indexing columns/rows in Sigma based on F.

    Example
    -------
    F = [torch.randn(10,2), torch.randn(10,3), torch.randn(10,2)]
    create_matrix_indexer(F)
        tensor([[1., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [0., 0., 1.]], dtype=torch.float64)
    """
    # Create a column vector that contains indices
    columns_group = [torch.ones(F[j].shape[1], dtype=dtype,
                                device=device) * j for j in range(len(F))]
    columns_group = torch.concatenate(columns_group, axis=0).ravel()

    # Create a matrix that indexes predictor groups
    mat_indexer = one_hot_encoding(columns_group, dtype, device)
    return mat_indexer


def prepare_smoothness_cov(F, smoothness_param,
                           dtype=torch.float64,
                           device='cpu'):
    """Use F and smoothness_param to define Omega and Omega_inv.

    Example
    -------
    F = [torch.randn(10,2), torch.randn(10,5)]
    O, O_i = prepare_smoothness_cov(F,[None, 2.])
    print(O)
    tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.0000, 0.7849, 0.4834, 0.2678, 0.1397],
            [0.0000, 0.0000, 0.7849, 1.0000, 0.7849, 0.4834, 0.2678],
            [0.0000, 0.0000, 0.4834, 0.7849, 1.0000, 0.7849, 0.4834],
            [0.0000, 0.0000, 0.2678, 0.4834, 0.7849, 1.0000, 0.7849],
            [0.0000, 0.0000, 0.1397, 0.2678, 0.4834, 0.7849, 1.0000]],
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
            Omega_j = torch.eye(D_j, dtype=dtype, device=device)
            Omega_j_inv = torch.eye(D_j, dtype=dtype, device=device)
        else:

            # When h[j] is a scalar then define Omega as follows:
            Omega_j = matern_type_kernel(D_j, h_j, device=device, dtype=dtype)
            Omega_j_inv, __ = matrix_inv_cholesky(Omega_j)

        Omega.append(Omega_j)
        Omega_inv.append(Omega_j_inv)

    # Prepare a matrix over entire Lambda
    Omega = torch.block_diag(*Omega)

    # The inverse of this matrix
    Omega_inv = torch.block_diag(*Omega_inv)

    return Omega, Omega_inv


def compute_covariance(nu, covX, lambdas_diag,
                       Omega_inv=None, compute_score=False):
    """Compute inv( 1/nu  X.T X + L Omega_inv) as well as logdet.

    Example
    -------
    Below, we provide covX on the following form
        tensor([[16.4331,  1.1891,  3.6651],
                [ 1.1891,  7.9407,  2.0888],
                [ 3.6651,  2.0888, 13.1385]], dtype=torch.float64)
    
        
    torch.manual_seed(0)
    X = torch.randn(10,3,dtype=torch.float64)
    covX = X.T@X
    lambdas_diag = torch.rand(3,dtype=torch.float64)
    nu = 1.
    Sigma, logdet = compute_covariance(nu, covX, lambdas_diag, None, True)
    Sigma_ref = torch.linalg.inv(covX + torch.diag(1./lambdas_diag))
    torch.testing.assert_close(Sigma, Sigma_ref)
    torch.testing.assert_close(logdet,torch.log(torch.linalg.det(Sigma_ref)))
    """

    if Omega_inv is not None:
        # Case with smoothness prior
        Lambda_inv = (1./lambdas_diag)[:, None] * Omega_inv
    else:
        # Case without smoothness prior
        Lambda_inv = torch.diag(1./lambdas_diag)

    # Estimate Sigma.
    Sigma, logdet = (
        matrix_inv_cholesky(1.0 / nu * covX + Lambda_inv, compute_score)
    )

    return Sigma, logdet


def fit_model_vectorized(X, y,
                         hyper_params,
                         initialization_params,
                         max_iterations,
                         mat_indexer,
                         Omega_inv,
                         compute_score,
                         early_stopping_tol,
                         verbose):
    """Fit a model and assume that y is 1D."""
    if verbose is True:
        if Omega_inv is None:
            print('Fitting a model without smoothness, assuming 1D y.')
        else:
            print('Fitting a model with smoothness, assuming 1D y.')
    if compute_score is True:
        logdetOmega = 0
        if Omega_inv is not None:
            logdetOmega = -torch.linalg.slogdet(Omega_inv)[1]  # pylint: disable=E1102

    covXy = X.T @ y
    covX = X.T @ X

    num_feat = len(initialization_params[0])
    num_obs = X.shape[0]
    P = y.shape[1]

    # Store lambdas and nu parameters
    summary = {"lambdas": torch.zeros((max_iterations, num_feat),
                                      dtype=X.dtype,
                                      device=X.device),
               "nu": torch.zeros(max_iterations,
                                 dtype=X.dtype,
                                 device=X.device),
               "score": torch.zeros(max_iterations,
                                    dtype=X.dtype,
                                    device=X.device)*torch.nan}

    eta, tau, phi, kappa = get_hyperparams_from_tuple(hyper_params)

    # Initialize the parameters
    lambdas = copy.deepcopy(initialization_params[0])
    nu = copy.deepcopy(initialization_params[1])

    if verbose is True:
        print(f'nu is initialized to {nu}.')
        print(f'lambdas are initialized to {lambdas}.')
        start_time = time.time()

    # Number of predictors in each group
    D = torch.sum(mat_indexer, axis=0)

    for iteration in range(max_iterations):

        # Store in the summaries
        summary['lambdas'][iteration, :] = lambdas
        summary['nu'][iteration] = nu

        # Create a vector that will form the diagonal lambda_j elements
        lambdas_diag = mat_indexer @ lambdas

        # Compute Sigma
        Sigma, logdet = compute_covariance(nu, covX, lambdas_diag,
                                           Omega_inv, compute_score)

        # Estimate weigths
        W = 1.0 / nu * torch.matmul(Sigma, covXy)

        # Estimate residuals
        residuals = y - X @ W

        if compute_score is True:
            summary["score"][iteration] = (
                - 1 / (2 * nu) * torch.sum(y * residuals)
                - P / 2 * (
                    + (-1) * logdet
                    + logdetOmega
                    + torch.sum(torch.log(lambdas_diag))
                    + num_obs * math.log(nu)
                )
                - torch.sum(torch.log(torch.pow(lambdas, (1 + eta))))
                - torch.sum(tau / lambdas)
                - math.log(math.pow(nu, (1 + phi)))
                - kappa / nu
            )

            if early_stopping_tol:
                if iteration > 1:
                    score_diff = (
                        summary["score"][iteration] -
                        summary["score"][iteration-1])
                    if score_diff < early_stopping_tol:
                        break

        if iteration < max_iterations-1:

            # Update the lambdas.
            lambdas = (
                (
                    matrix_blockdiag_rotation(W, mat_indexer, Omega_inv) +
                    matrix_block_trace(Sigma, mat_indexer, Omega_inv) +
                    2 * tau) / (D + 2 * eta + 2)
            ).ravel()

            nu = (
                (torch.sum(torch.pow(y - X @ W, 2)) +
                 matrix_trace_of_product(covX, Sigma) +
                 2 * kappa) / (num_obs + 2 + 2 * phi)
            )

        if verbose is True:
            print(f'At iteration {iteration} of {max_iterations}')
            time_elapsed = time.time()-start_time
            print(f'Time elapsed: {time_elapsed}')

    if early_stopping_tol:
        summary['lambdas'] = summary['lambdas'][:iteration+1, :]
        summary['nu'] = summary['nu'][:iteration+1]
        summary['score'] = summary['score'][:iteration+1]

    return W, summary, Sigma


def fit_model_multidimensional(X, y,
                               hyper_params,
                               initialization_params,
                               max_iterations,
                               mat_indexer,
                               Omega_inv,
                               compute_score,
                               early_stopping_tol,
                               verbose):
    """Muldimensional model fit with- or without smoothness."""
    if verbose is True:
        if Omega_inv is None:
            print('Fitting a *multi_dimensional* model without smoothness.')
        else:
            print('Fitting a *multi_dimensional* model with smoothness.')
    if compute_score is True:
        logdetOmega = 0
        if Omega_inv is not None:
            logdetOmega = -torch.linalg.slogdet(Omega_inv)[1]  # pylint: disable=E1102

    covXy = X.T @ y
    covX = X.T @ X

    num_feat = len(initialization_params[0])
    num_obs = X.shape[0]
    P = y.shape[1]

    # Store lambdas and nu parameters
    summary = {"lambdas": torch.zeros((max_iterations, num_feat),
                                      dtype=X.dtype,
                                      device=X.device),
               "nu": torch.zeros(max_iterations,
                                 dtype=X.dtype,
                                 device=X.device),
               "score": torch.zeros(max_iterations,
                                    dtype=X.dtype,
                                    device=X.device)*torch.nan}

    eta, tau, phi, kappa = get_hyperparams_from_tuple(hyper_params)

    # Initialize the parameters
    lambdas = copy.deepcopy(initialization_params[0])
    nu = copy.deepcopy(initialization_params[1])

    if verbose is True:
        print(f'nu is initialized to {nu}.')
        print(f'lambdas are initialized to {lambdas}.')
        start_time = time.time()

    # The number of predictors in each predictor group
    D = torch.sum(mat_indexer, axis=0)

    # Prepare indices for block indexing, and prepare Omega_inv terms for each
    # predictor group
    indices = []
    Omega_inv_groups = []
    for f in range(mat_indexer.shape[1]):
        index = torch.where(mat_indexer[:, f] == 1)[0]
        indices.append(index)

        if Omega_inv is not None:
            Omega_inv_groups.append(matrix_block_indexing(Omega_inv, index))

    for iteration in range(max_iterations):

        # Store in the summaries
        summary['lambdas'][iteration, :] = lambdas
        summary['nu'][iteration] = nu

        # Create a vector that will form the diagonal lambda_j elements
        lambdas_diag = mat_indexer @ lambdas

        # Compute Sigma
        Sigma, logdet = compute_covariance(nu, covX, lambdas_diag,
                                           Omega_inv, compute_score)

        # Estimate weigths
        W = 1.0 / nu * torch.matmul(Sigma, covXy)

        # Estimate residuals
        residuals = y - X @ W

        if compute_score is True:
            summary["score"][iteration] = (
                - 1 / (2 * nu) * torch.sum(y * residuals)
                - P / 2 * (
                    + (-1) * logdet
                    + logdetOmega
                    + torch.sum(torch.log(lambdas_diag))
                    + num_obs * math.log(nu)
                )
                - torch.sum(torch.log(torch.pow(lambdas, (1 + eta))))
                - torch.sum(tau / lambdas)
                - math.log(math.pow(nu, (1 + phi)))
                - kappa / nu
            )

            if early_stopping_tol:
                if iteration > 1:
                    score_diff = (
                        summary["score"][iteration] -
                        summary["score"][iteration-1])
                    if score_diff < early_stopping_tol:
                        break

        if iteration < max_iterations-1:

            # In this case we also have to iterate over feature groups
            for f in range(num_feat):

                index = indices[f]
                D_f = D[f]

                # Get rows in W
                mu_f = W[index, :]

                # Get blocks in Sigma and Omega_inv
                Sigma_ff = matrix_block_indexing(Sigma, index)

                # Update lambda_f. In this case, W is a real matrix of
                # size [D X P], where D is the number of predictors
                # associated with outcome variable p = 1, ..., P.

                if Omega_inv is not None:
                    # Case with smoothness prior
                    Omega_inv_ff = Omega_inv_groups[f]

                    if mu_f.shape[1] > mu_f.shape[0]:
                        E = (
                            torch.trace(mu_f @ mu_f.T @ Omega_inv_ff) +
                            matrix_trace_of_product(Omega_inv_ff, Sigma_ff)*P
                        )
                    else:
                        E = (
                            torch.trace(mu_f.T @ Omega_inv_ff @ mu_f) +
                            matrix_trace_of_product(Omega_inv_ff, Sigma_ff)*P
                        )

                else:
                    # Case without smoothness prior
                    E = (
                        matrix_trace_of_product(mu_f, mu_f.T) +
                        torch.trace(Sigma_ff) * P
                    )

                # Update each lambda
                lambdas[f] = (E + 2 * tau) / (P * D_f + 2 * eta + 2)

            # Estimate residual sum of squares
            rss = torch.sum(torch.pow(residuals, 2))

            # Update nu
            nu = (
                (rss + P * matrix_trace_of_product(covX, Sigma) + 2 * kappa)
                / (P * num_obs + 2 + 2 * phi)
            )

        if verbose is True:
            print(f'At iteration {iteration} of {max_iterations}')
            time_elapsed = time.time()-start_time
            print(f'Time elapsed: {time_elapsed}')

    if early_stopping_tol:
        summary['lambdas'] = summary['lambdas'][:iteration+1, :]
        summary['nu'] = summary['nu'][:iteration+1]
        summary['score'] = summary['score'][:iteration+1]

    return W, summary, Sigma
