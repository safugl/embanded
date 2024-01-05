"""EMBanded model implementation using torch."""
import copy
from typing import List, Tuple
import torch

from .helpers import check_positive_float
from .helpers import check_input
from .helpers import check_boolean
from .helpers import check_smooth_params

from ._torch_model_utils import prepare_smoothness_cov
from ._torch_model_utils import create_matrix_indexer
from ._torch_model_utils import fit_model_multidimensional
from ._torch_model_utils import fit_model_without_smoothness
from ._torch_model_utils import fit_model_with_smoothness
from ._torch_linalg_utils import matrix_centering


class EMBanded:
    """Expectation-Maximization algorithm for banded-type regression."""

    def __init__(
        self,
        hyper_params: Tuple[float, float, float,
                            float] = (1e-4, 1e-4, 1e-4, 1e-4),
        max_iterations: int = 200,
    ):
        """Initialize the instance based on num_features.

        Parameters
        ----------
        hyper_params : Tuple[float, float, float, float], optional
            Specify the hyperparameters related to the Inverse-Gamma priors
            imposed on the lambda_j terms and on the nu terms. The inputs
            should be a tuple in the form of (eta, tau, phi, kappa).
            The default values are (1e-4, 1e-4, 1e-4, 1e-4)."
        max_iterations : int, optional
            Specify the number of iterations. The default is 200.

        Raises
        ------
        TypeError
            The hyper parameters should be specified as a tuple of length four.
        ValueError
            The hyper parameters should be positive floats.


        Examples
        --------
        Simulate two predictor groups, X1 and X2. Specify that y contains a
        mixed version of X1 but not X2. Set F as a list, F = [X1, X2],
        and proceed to fit the model.

            >>> import torch
            >>> from embanded_torch import EMBanded

            >>> device = torch.device('cuda'
            >>>                       if torch.cuda.is_available() else 'cpu')

            >>> dtype = torch.float64
            >>> torch.manual_seed(0)
            >>> F = [torch.randn(1000, 5, dtype=dtype, device=device),
            >>>      torch.randn(1000, 10, dtype=dtype, device=device)]

            >>> W1 = torch.hamming_window(5, dtype=dtype)[:, None].to(device)
            >>> y = F[0]@W1 + torch.randn(1000, 1, dtype=dtype, device=device)
            >>> emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
            >>>                max_iterations=200)
            >>> emb.set_verbose(True)
            >>> emb.fit(F, y)
            >>> print('The estimated weights are:')
            >>> print(torch.round(emb.W, decimals=1))


        It is possible to let y have multiple columns, but in this case
        one needs to specify emb.set_multidimensional(True).

            >>> import torch
            >>> import matplotlib.pyplot as plt
            >>> from embanded_torch import EMBanded

            >>> device = torch.device('cuda' if
                                  torch.cuda.is_available() else 'cpu')

            >>> dtype = torch.float64
            >>> torch.manual_seed(0)
            >>> F = [torch.randn(1000, 5, dtype=dtype, device=device),
            >>>      torch.randn(1000, 10, dtype=dtype, device=device)]

            >>> W1 = torch.hamming_window(5, dtype=dtype)[:, None].to(device)
            >>> y = torch.concatenate(
            >>>     [F[0]@W1 + torch.randn(1000, 1,dtype=dtype,device=device),
            >>>      torch.randn(1000, 1, dtype=dtype, device=device)], axis=1)
            >>> emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
            >>>                max_iterations=200)
            >>> emb.set_verbose(True)
            >>> emb.set_multidimensional(True)
            >>> emb.fit(F, y)
            >>> print('The estimated weights are:')
            >>> print(torch.round(emb.W, decimals=1))



        One can assign smoothness parameters to each feature set. In the
        following example, smoothness is specifically declared for the first
        predictor group.

            >>> from sklearn.linear_model import LinearRegression
            >>> import torch
            >>> import matplotlib.pyplot as plt
            >>> from embanded_torch import EMBanded

            >>> device = torch.device('cuda'
            >>>                       if torch.cuda.is_available() else 'cpu')
            >>> dtype = torch.float64
            >>> torch.manual_seed(0)
            >>> F = [torch.randn(1000, 100, dtype=dtype, device=device),
            >>>      torch.randn(1000, 100, dtype=dtype, device=device)]

            >>> X = torch.concatenate(F, axis=1)
            >>> W = torch.zeros((200, 1), dtype=dtype, device=device)
            >>> t = 50/200*torch.arange(100,dtype=dtype,device=device)
            >>> W[:100] = torch.sin(t)[:, None]

            >>> y = X@W + torch.randn(1000, 1, dtype=dtype, device=device)*5


            >>> emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
            >>>                max_iterations=200)
            >>> emb.set_smoothness_param([15., None])
            >>> emb.set_verbose(True)
            >>> emb.fit(F, y)
            >>> plt.plot(W.cpu().numpy(), label='target')
            >>> plt.plot(emb.W.cpu().numpy(), label='emb')


            >>> reg = LinearRegression().fit(X.cpu().numpy(), y.cpu().numpy())
            >>> plt.plot(reg.coef_.ravel(), label='OLS', linewidth=0.4)
            >>> plt.legend()





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

        # Initialize relevant terms
        self.X_offset = None
        self.y_offset = None
        self.W = None
        self.summary = None
        self.store_covariance_terms = False
        self.covariance_terms = {'Sigma': None,
                                 'Omega': None,
                                 'Omega_inv': None}

    def fit(self, F: List[torch.Tensor], y: torch.Tensor):
        """Fit the model.

        Parameters
        ----------
        F : List[torch.Tensor]
            A list of torch.Tensor of where each Tensor should have
            dimensionality (M x D_j) where M is the number of samples (rows)
            and where D_j is the number of columns of that given feature space
            (D_j>=1).
        y : torch.Tensor
            A column vector of size (M x 1), or a matrix of size (M X P),
            where M represents the number of samples (rows) and where P
            represents the number of outcome variables. The number of samples
            should be exactly identical to the number of rows in each entry
            in F. One needs to set multi_dimensional=True if P>1.
        """
        check_input(F, y, self.multi_dimensional)

        self.num_features = len(F)

        # Combine the features into a matrix.
        X = torch.concatenate(F, axis=1)

        # Remove offset unless the user have turned this functionality off.
        if self.remove_intercept is True:
            X, self.X_offset = matrix_centering(X)
            y, self.y_offset = matrix_centering(y)

        # Prepare smoothness terms.
        if self.encourage_smoothness is True:
            Omega, Omega_inv = (
                prepare_smoothness_cov(F, self.smoothness_param,
                                       dtype=X.dtype,
                                       device=X.device))

            if self.verbose is True:
                print(f'smoothness_param is {self.smoothness_param}')
        else:
            Omega, Omega_inv = None, None

        # Create a matrix that indexes predictor groups.
        mat_indexer = create_matrix_indexer(F, dtype=X.dtype, device=X.device)

        # Set lambdas to default values if they have not been specified.
        if self.lambdas_init is None:
            self.lambdas_init = torch.ones(self.num_features,
                                           dtype=X.dtype,
                                           device=X.device)

        # Set nu to default value if it has not been updated.
        if self.nu_init is None:
            self.nu_init = 1.

        # These parameters will be stored in a list for convinience.
        initialization_params = [self.lambdas_init, self.nu_init]

        if self.multi_dimensional is False:

            # The Supplementary Material outlines a model that enables sharing
            # of covariance terms across outcome variables. When the
            # multi_dimensional parameter is set to False, we can employ
            # certain tricks to improve compute time.

            if self.encourage_smoothness is True:

                # Here we do not assume diagonal covariance.
                W, summary, Sigma = (
                    fit_model_with_smoothness(
                        X, y, self.hyper_params,
                        initialization_params,
                        self.max_iterations,
                        mat_indexer, Omega_inv,
                        self.verbose)
                )

            else:
                # Here we do assume diagonal covariance.
                W, summary, Sigma = (
                    fit_model_without_smoothness(
                        X, y, self.hyper_params,
                        initialization_params,
                        self.max_iterations,
                        mat_indexer,
                        self.verbose)
                )

        elif self.multi_dimensional is True:

            # This implementation works both for diagonal and non-diagonal
            # covariance terms. It may be slightly slower, but it is useful
            # when y has multiple outcome variables.
            W, summary, Sigma = (
                fit_model_multidimensional(
                    X, y, self.hyper_params,
                    initialization_params,
                    self.max_iterations,
                    mat_indexer, Omega_inv, self.verbose)

            )

        # Store W and the hyper_param summary.
        self.W = W
        self.summary = summary

        # Store Sigma, Omega and Omega_inv is store_covariance_terms is set
        # to True. These are not stored by default as they can be rather large.
        if self.store_covariance_terms is True:
            self.covariance_terms = {'Sigma': Sigma,
                                     'Omega': Omega,
                                     'Omega_inv': Omega_inv}

    def predict(self, F_test: List[torch.Tensor]):
        """Prediction using the EM-banded regression model.

        Parameters
        ----------
        F_test : List[torch.Tensor]
            A list of torch.Tensor where each Tensor should have dimensionality
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
        X_test = copy.deepcopy(torch.concatenate(F_test, axis=1))
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

    def set_lambdas_init(self, val: torch.Tensor) -> None:
        """Set initial lambda parameters."""
        if not isinstance(val, torch.Tensor):
            raise ValueError(f'{val} should be a torch.Tensor.')
        if not val.ndim == 1:
            raise ValueError(f'{val} should have one dimension.')
        if any(val < 0) or any(val == torch.inf):
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
