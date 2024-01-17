"""EMBanded model."""
import copy
from typing import List, Tuple
import numpy as np

from .helpers import (
    check_positive_float, check_input, check_boolean, check_smooth_params
)

from ._numpy_model_utils import (
    prepare_smoothness_cov, create_matrix_indexer,
    fit_model_multidimensional, fit_model_vectorized,
)
from ._numpy_linalg_utils import matrix_centering


class EMBanded:
    """Expectation-Maximization algorithm for banded-type regression.
    
    Parameters
    ----------
    hyper_params : Tuple[float, float, float, float], optional
        Specify the hyperparameters related to the Inverse-Gamma priors
        imposed on the lambda_j terms and on the nu terms. The inputs
        should be a tuple in the form of (eta, tau, phi, kappa). The parameters
        eta and tau are related to the priors imposed on the lambda_j terms, 
        lambda_j ~ InvGamma(eta, tau). The parameters phi and kappa are related
        to the priors imposed on the nu term, nu ~ InvGamma(phi, kappa).        
        The default values are (1e-4, 1e-4, 1e-4, 1e-4).
    max_iterations : int, optional
        Specify the number of iterations. The default is 200.
        

    Attributes that control model specification
    --------------------------------------------
    set_verbose: bool, default=False
        Verbose mode when fitting the model.
    
    set_multidimensional: bool, default=False
        If set to False, the model will utilize vectorized code. In this case, 
        y has to be a matrix of size [M x 1]. If set to True, the model will
        utilize nested for loops. In this case, y can be a matrix of size
        [M x P] for any P > 0.
    
    set_lambdas_init: np.ndarray | None, default=None
        Initialization parameters for lambda terms. If set to None, the model
        will set the initial lambda parameters to np.ones(num_features).
    
    set_nu_init: np.ndarray | None, default=None
        Initialization parameters for the nu term. If set to None, the model
        will set the initial nu parameter to 1.0.
    
    set_remove_intercept: bool, default=True
        Whether to remove the offset from X and y before model fitting. If set
        to True, the model will remove the offsets and use these offsets for
        predictions.
    
    set_store_covariance_terms: bool, default=True
        Whether to store Sigma. If set to True, then Sigma will be stored as
        an attribute dictionary called covariance_terms with keys Sigma, 
        Omega_inv, and Omega.
    
    set_smoothness_param: List[None | float], default=None
        Specify the hyperparameter h_j related to the covariance 
        parametrization of predictor group j. The length of the input list must
        be equal to the number of predictor groups. When a given element is set
        to None, then the Omega_j term will be a unit matrix. When a given
        element in the list is a positive float, then the corresponding Omega_j
        term will be parameterized with a Matern kernel.
    
    set_compute_score: bool, default=False
        If set to True, estimate the log score at each iteration of the
        optimization.
    
    set_early_stopping_tol: float | None, default=None
        Stop the algorithm if increases in the log score are smaller than this
        tolerance. When set to None, the algorithm will not terminate early. A
        reasonable tolerance criterion is often: set_early_stopping_tol(1e-8).
        
        
    Examples
    --------
    Simulate two predictor groups, X1 and X2. Specify that y contains a
    mixed version of X1 but not X2. Set F as a list, F = [X1, X2],
    and proceed to fit the model.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from embanded.embanded_numpy import EMBanded
        >>>
        >>> np.random.seed(1)
        >>> F = [np.random.randn(1000,5), np.random.randn(1000,10)]
        >>> W1 = np.hamming(5)[:,None]
        >>> y = F[0]@W1 + np.random.randn(1000,1)
        >>> emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
        >>>                max_iterations=200)
        >>> emb.fit(F,y)
        >>>
        >>> print('The estimated weights are:')
        >>> print(np.round(emb.W,1))

    It is possible to let y have multiple columns, but in this case
    one needs to specify emb.set_multidimensional(True).

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from embanded.embanded_numpy import EMBanded
        >>>
        >>> np.random.seed(1)
        >>> F = [np.random.randn(1000,5), np.random.randn(1000,10)]
        >>> W1 = np.hamming(5)[:,None]
        >>> y = np.c_[F[0]@W1 + np.random.randn(1000,1),
        >>>          np.random.randn(1000,1)]
        >>> emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
        >>>                max_iterations=200)
        >>> emb.set_multidimensional(True)
        >>> emb.set_verbose(True)
        >>> emb.fit(F,y)

    One can assign smoothness parameters to each feature set. In the
    following example, smoothness is specifically declared for the first
    predictor group.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from embanded.embanded_numpy import EMBanded
        >>> import matplotlib.pyplot as plt
        >>>
        >>> np.random.seed(1)
        >>> F = [np.random.randn(1000,100), np.random.randn(1000,100)]
        >>> X = np.concatenate(F,axis=1)
        >>> W = np.zeros((200,1))
        >>> W[:100] = np.sin(50/200*np.arange(100))[:,None]
        >>> y = X@W + np.random.randn(1000,1)*5
        >>>
        >>> emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
        >>>                 max_iterations=200)
        >>> emb.set_smoothness_param([15., None])
        >>> emb.set_verbose(True)
        >>> emb.fit(F,y)
        >>> plt.plot(W,label='target')
        >>> plt.plot(emb.W,label='emb')
        >>>
        >>> from sklearn.linear_model import LinearRegression
        >>>
        >>> reg = LinearRegression().fit(X,y)
        >>> plt.plot(reg.coef_.ravel(),label='OLS',alpha=0.3)
        >>> plt.legend()
    """

    def __init__(
        self,
        hyper_params: Tuple[float, float, float,
                            float] = (1e-4, 1e-4, 1e-4, 1e-4),
        max_iterations: int = 200,
    ):
        """Initialize the instance.

        Parameters
        ----------
        hyper_params : Tuple[float, float, float, float], optional
            Specify the hyperparameters related to the Inverse-Gamma priors
            imposed on the lambda_j terms and on the nu terms. The inputs
            should be a tuple in the form of (eta, tau, phi, kappa). The 
            parameters eta and tau are related to the priors imposed on the 
            lambda_j terms, lambda_j ~ InvGamma(eta, tau). The parameters phi 
            and kappa are related to the priors imposed on the nu term,
            nu ~ InvGamma(phi, kappa).        
            The default values are (1e-4, 1e-4, 1e-4, 1e-4).
        max_iterations : int, optional
            Specify the number of iterations. The default is 200.

        Raises
        ------
        TypeError
            The hyper parameters should be specified as a tuple of length four.
        ValueError
            The hyper parameters should be positive floats.
            
        """
        if isinstance(hyper_params, tuple) is not True:
            raise TypeError("hyper_params must be a tuple")
        if len(hyper_params) != 4:
            raise TypeError("hyper_params must have length of four")
        for val in hyper_params:
            check_positive_float(val)

        # Declare parameters.
        self.hyper_params = hyper_params
        self.max_iterations = max_iterations

        # Set default parameters.
        self.encourage_smoothness = False
        self.remove_intercept = True
        self.multi_dimensional = False
        self.lambdas_init = None
        self.nu_init = None
        self.verbose = False
        self.smoothness_param = None
        self.num_features = None
        self.compute_score = False
        self.early_stopping_tol = None

        # Initialize relevant terms
        self.X_offset = None
        self.y_offset = None
        self.W = None
        self.summary = None
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
            A column vector of size (M x 1), or a matrix of size (M X P),
            where M represents the number of samples (rows) and where P
            represents the number of outcome variables. The number of samples
            should be exactly identical to the number of rows in each entry
            in F. One needs to set multi_dimensional=True if P>1.
        """
        check_input(F, y, self.multi_dimensional)

        self.num_features = len(F)

        # Combine the features into a matrix.
        X = copy.deepcopy(np.concatenate(F, axis=1))
        y = copy.deepcopy(y)

        # Remove offset unless the user have turned this functionality off
        if self.remove_intercept is True:
            X, self.X_offset = matrix_centering(X)
            y, self.y_offset = matrix_centering(y)

        # Prepare smoothness terms.
        if self.encourage_smoothness is True:
            Omega, Omega_inv = (
                prepare_smoothness_cov(F, self.smoothness_param,
                                       dtype=X.dtype)
            )

            if self.verbose is True:
                print(f'smoothness_param is {self.smoothness_param}')
        else:
            Omega, Omega_inv = None, None

        # Create a matrix that indexes predictor groups.
        mat_indexer = create_matrix_indexer(F, dtype=X.dtype)

        # Set lambdas to default values if they have not been specified.
        if self.lambdas_init is None:
            self.lambdas_init = np.ones(self.num_features, dtype=X.dtype)

        # Set nu to default value if it has not been updated.
        if self.nu_init is None:
            self.nu_init = 1.

        # These parameters will be stored in a list for convinience
        initialization_params = [self.lambdas_init, self.nu_init]

        if self.multi_dimensional is False:

            # This implementation is suitable when y is a matrix of size
            # [M x 1]. It avoids nested for loops and utilizes vectorized code,
            # which may improve compute time, especially in scenarios with
            # many predictor groups and relatively few predictors in each
            # group.

            W, summary, Sigma = (
                fit_model_vectorized(
                    X, y, self.hyper_params,
                    initialization_params,
                    self.max_iterations,
                    mat_indexer, Omega_inv,
                    self.compute_score,
                    self.early_stopping_tol,
                    self.verbose)
            )

        elif self.multi_dimensional is True:

            # This implementation is applicable for any P > 0, where y is a
            # matrix of size [M x P]. The implementation involves nested for
            # loops. It can be efficient when the number of predictor groups
            # is low, and each group has many predictors.

            W, summary, Sigma = (
                fit_model_multidimensional(
                    X, y, self.hyper_params,
                    initialization_params,
                    self.max_iterations,
                    mat_indexer, Omega_inv,
                    self.compute_score,
                    self.early_stopping_tol,
                    self.verbose)
            )

        # Store W and the hyper_param summary.
        self.W = W
        self.summary = summary

        # Store Sigma, Omega and Omega_inv is store_covariance_terms is set
        # to True. These are not stored by default as they can be rather large
        if self.store_covariance_terms is True:
            self.covariance_terms = {'Sigma': Sigma,
                                     'Omega': Omega,
                                     'Omega_inv': Omega_inv}

    def predict(self, F_test: List[np.ndarray]):
        """Prediction using the EM-banded regression model.

        Parameters
        ----------
        F_test : List[np.ndarray]
            A list of ndarrays where each array should have dimensionality
            (M x D_j), with M representing the number of samples (rows) and
            D_j denoting the number of columns in the respective feature space
            (D_j ≥ 1).

        Return
        ----------
        prediction : ndarray
            Prediction
        """
        if isinstance(F_test, list) is not True:
            raise TypeError(r"{F_test} must be a list")
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
        check_boolean(val)
        self.verbose = val

    def set_multidimensional(self, val: bool) -> None:
        """Set multidimensional."""
        check_boolean(val)
        self.multi_dimensional = val

    def set_lambdas_init(self, val: np.ndarray) -> None:
        """Set initial lambda parameters."""
        if not isinstance(val, np.ndarray):
            raise ValueError(f'{val} should be a np.ndarray.')
        if not val.ndim == 1:
            raise ValueError(f'{val} should have one dimension.')
        if any(val < 0) or any(val == np.inf):
            raise ValueError(f'{val} should be positive.')
        self.lambdas_init = val

    def set_nu_init(self, val: float) -> None:
        """Set initial nu parameter."""
        check_positive_float(val)
        self.nu_init = val

    def set_remove_intercept(self, val: bool) -> None:
        """Specify if intercept should be removed."""
        check_boolean(val)
        self.remove_intercept = val

    def set_store_covariance_terms(self, val: bool) -> None:
        """Specify if covariance terms should be stored."""
        check_boolean(val)
        self.store_covariance_terms = val

    def set_smoothness_param(self, val: List[None | float]) -> None:
        """Set the smoothness parameter."""
        check_smooth_params(val)
        self.smoothness_param = val
        if any(val):
            self.encourage_smoothness = True

    def set_compute_score(self, val: bool) -> None:
        """Specify if log objective should be computed"""
        check_boolean(val)
        self.compute_score = val

    def set_early_stopping_tol(self, val: float | None) -> None:
        """Specify tolerance for early stopping"""
        if val:
            check_positive_float(val)
            self.set_compute_score(True)
        self.early_stopping_tol = val
