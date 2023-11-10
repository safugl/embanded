"""EMBanded model"""
import copy
from typing import List, Tuple
import numpy as np
from . import model_utils


class EMBanded:
    """Expectation-Maximization algorithm for banded-type regression."""

    def __init__(
        self,
        num_features: int,
        hyper_params: Tuple[float, float, float,
                            float] = (1e-4, 1e-4, 1e-4, 1e-4),
        max_iterations: int = 200,
    ):
        """Initializes the instance based on num_features.

        Parameters
        ----------
        num_features : int
            Specify the number of feature sets. This value must must be equal 
            to len(X) = len([F1, F2, ..., Fj]). The value is used to generate
            default values for lambdas and nu.
        hyper_params : Tuple[float, float, float, float], optional
            Specify the hyperparameters related to the Inverse-Gamma priors 
            imposed on the lambda_j terms and on the nu terms. The inputs are 
            should be a tuble on the form (eta, tau, phi, kappa). 
            The default is (1e-4, 1e-4, 1e-4, 1e-4).
        max_iterations : int, optional
            Specify the maximum allow number of iterations. The default is 200.

        Raises
        ------
        TypeError
            The hyper parameters should be specified as a tuble of length four.
        ValueError
            The hyper parameters should be positve floats.

        """
        if isinstance(hyper_params, tuple) is not True:
            raise TypeError("hyper_params must be a tuple")
        if len(hyper_params) != 4:
            raise TypeError("hyper_params must have length of four")
        _check_elements_are_positive_floats(hyper_params)

        # Set parameters
        self.num_features = num_features
        self.hyper_params = hyper_params
        self.max_iterations = max_iterations

        # Set default parameters
        self.smoothness_param = [None] * num_features
        self.remove_intercept = True
        self.multi_dimensional = False
        self.lambdas_init = np.ones(num_features, dtype=np.float64)
        self.nu_init = 1.
        self.verbose = False

        # Store intercept terms and weights
        self.X_offset = None
        self.y_offset = None
        self.W = None
        self.summary = None

        # Optional: store estimated Sigma and Omega terms
        self.store_covariance_terms = False
        self.covariance_terms = {'Sigma': None,
                                 'Omega': None,
                                 'Omega_inv': None}

    def fit(self, F: List[np.ndarray], y: np.ndarray):
        """Fit the model.
        
        Parameters
        ----------
        F : List[np.ndarray]
            A list of ndarrays of where each array should have
            dimensionality (M x D_j) where M is the number of samples (rows)
            and where D_j is the number of columns of that given feature space
            (D_j>=1).
        y : np.ndarray
            A column vector of size (M x 1) where M is the number of samples
            (rows). The number of samples should be exactly identical to the
            number of rows in each entry in F.

        """
        _check_input(F, y, self.num_features, self.multi_dimensional)

        # Combine the features into a matrix
        X = np.concatenate(F, axis=1).copy()

        # Prepare y
        y = np.asarray(y, dtype=X.dtype)

        # Remove offset unless the user have turned this functionality off
        if self.remove_intercept is True:
            X, self.X_offset = model_utils.matrix_centering(X)
            y, self.y_offset = model_utils.matrix_centering(y)

        if any(self.smoothness_param) is True:
            # Initialize the Omega and Omega_inv matrices
            _verbose_print(self.verbose, 'Define Omega terms')
            Omega, Omega_inv = (
                model_utils.prepare_smoothness_cov(F, self.smoothness_param))
        else:
            Omega, Omega_inv = None, None

        # Create a column vector that contains indices
        columns_group = [np.ones(F[j].shape[1])*j for j in range(len(F))]
        columns_group = np.concatenate(columns_group, axis=0).ravel()

        # Create a matrix that indexes predictor groups
        mat_indexer = model_utils.one_hot_encoding(columns_group)

        # Store initialization parameters in a list
        initialization_params = [self.lambdas_init, self.nu_init]

        if self.multi_dimensional is True:

            # This setting allows for multi-dimensional implementation.
            # The code has not been vectorized. An empty Omega_inv matrix
            # indicates no smoothing.
            _verbose_print(self.verbose, 'Multi-dimensional implementation.')

            W, summary, Sigma = (
                model_utils.fit_model_multidimensional(
                    X, y, self.hyper_params,
                    initialization_params,
                    self.max_iterations,
                    mat_indexer, Omega_inv)
            )

        elif self.multi_dimensional is False:

            # Default is to set multi_dimensional = False

            if any(self.smoothness_param) is True:

                # Use Omega_inv to enforce smoothness
                _verbose_print(self.verbose, 'Fit model with Omega')

                W, summary, Sigma = (
                    model_utils.fit_model_with_smoothness(
                        X, y, self.hyper_params,
                        initialization_params,
                        self.max_iterations,
                        mat_indexer, Omega_inv)
                )

            else:

                # No smoothing and one dimensioonal output.
                _verbose_print(self.verbose, 'Assume diagonal Omega.')

                W, summary, Sigma = (
                    model_utils.fit_model_without_smoothness(
                        X, y, self.hyper_params,
                        initialization_params,
                        self.max_iterations,
                        mat_indexer)
                )

        # Store W and the hyper_param summary
        self.W = W
        self.summary = summary

        # Store Sigma, Omega and Omega_inv is store_covariance_terms is True
        if self.store_covariance_terms is True:
            _verbose_print(self.verbose, 'Store Sigma, Omega and Omega_inv')
            self.covariance_terms = {'Sigma': Sigma,
                                     'Omega': Omega,
                                     'Omega_inv': Omega_inv}

    def predict(self, F_test):
        """Predict using the EM-banded regression model

        Parameters
        ----------
        F_test : list
            A list of ndarrays of where each array should have
            dimensionality (M x D_j) where M is the number of samples (rows) 
            and where D_j is the number of columns of that given feature space
            (D_j>=1). The list should have the same format as that used for 
            training the model.

        Returne
        ----------        
        y_pred : ndarray
            Returns predicted values

        """
        # First check F_test
        if isinstance(F_test, list) is not True:
            raise TypeError(r"{F} must be a list")
        if len(F_test) != self.num_features:
            raise TypeError(r"Len should be {self.num_features}")

        X_test = copy.deepcopy(np.concatenate(F_test, axis=1))

        if self.remove_intercept is True:
            X_test -= self.X_offset

        prediction = X_test @ self.W

        if self.remove_intercept is True:
            prediction += self.y_offset

        return prediction

    def set_verbose(self, val: bool) -> None:
        """Set verbose."""
        _check_boolean(val)
        self.verbose = val

    def set_multidimensional(self, val: bool) -> None:
        """Set multidimensional."""
        _check_boolean(val)
        self.multi_dimensional = val

    def set_lambdas_init(self, val: np.ndarray) -> None:
        """Set initial lambda parameters."""
        if not isinstance(val, np.ndarray):
            raise ValueError(f'{val} should be a np.ndarray.')
        if not val.ndim == 1:
            raise ValueError(f'{val} should have one dimension.')
        if not len(val) == self.num_features:
            raise ValueError(f'{val} should have length {self.num_features}')
        if any(val < 0) or any(val == np.inf):
            raise ValueError(f'{val} should be positive.')
        self.lambdas_init = val

    def set_nu_init(self, val: float) -> None:
        """Set initial nu parameter."""
        if any(val < 0) or any(val == np.inf):
            raise ValueError(f'{val} should be positive.')
        self.nu_init = val

    def set_remove_intercept(self, val: bool) -> None:
        """Specify if intercept should be removed."""
        _check_boolean(val)
        self.remove_intercept = val

    def set_store_covariance_terms(self, val: bool) -> None:
        """Specify if covariance terms should be stored."""
        _check_boolean(val)
        self.store_covariance_terms = val

    def set_smoothness_param(self, val: List[None | float]) -> None:
        """Set the smoothness parameter."""
        if not isinstance(val, list):
            raise ValueError(f'{val} should be a list.')
        h_vals = [i for i in val if i is not None]
        if h_vals:
            _check_elements_are_positive_floats(h_vals)
        assert len(
            val) == self.num_features, "Should have len {self.num_features}"

        self.smoothness_param = val


def _verbose_print(verbose: bool, string: str) -> None:
    """Print messages"""
    if verbose is True:
        print(f'{string}')


def _check_input(F: List[np.ndarray],
                 y: np.ndarray,
                 num_features: int,
                 multi_dimensional: bool) -> None:
    """Checks for F and y."""

    # First check F
    if isinstance(F, list) is not True:
        raise TypeError(r"{F} must be a list")
    if len(F) != num_features:
        raise TypeError(r"Len should be {self.num_features} not {len(F)}")

    # Then check that F and y has same number of rows
    num_rows = y.shape[0]
    for X_f in F:
        if X_f.shape[0] != num_rows:
            raise TypeError(r"Each array should have {num_rows} rows")
        if (X_f.ndim != 2) or (not isinstance(X_f, np.ndarray)):
            raise TypeError(r"Each array should be a matrix.")

    # y must have 2 dimensions
    if y.ndim != 2:
        raise TypeError(r"y must have two dimensions, not {y.ndim}")

    # Then check if y has multiple columns
    if multi_dimensional is False:
        if y.shape[-1] != 1:
            raise TypeError(r"y must have two dimensions, not {y.ndim}")


def _check_elements_are_positive_floats(vals: Tuple | List) -> None:
    """Check if elements in a tuple or list are positive floats"""
    for v in vals:
        assert isinstance(v, float), f'{vals} should be floats.'
        assert v > 0, f'{vals} should be positive.'
        assert v != np.inf, f'{vals} should be positive.'


def _check_boolean(val: bool) -> None:
    """Checks if val is a boolean variable"""
    if not isinstance(val, bool):
        raise ValueError(f'{val} should be a boolean variable.')
