"""Compare models."""
import copy
import numpy as np
from sklearn.linear_model import (
    ARDRegression, BayesianRidge, LinearRegression, RidgeCV, LassoCV
)
from embanded.embanded_numpy import EMBanded
from embanded.embanded_torch import EMBanded as EMBanded_torch


def fit_model(key, X, y):
    """Use this for comparing runtime."""
    X = copy.deepcopy(X)

    est = []

    if 'EMB' in key:
        y = copy.deepcopy(y)
    else:
        y = copy.deepcopy(y).ravel()

    if key == 'OLS':
        # OLS model
        est = LinearRegression().fit(X, y)
        W = est.coef_

    elif key == 'ARD':
        # ARD regression
        est = ARDRegression(compute_score=True).fit(X, y)
        W = est.coef_

    elif key == 'BRR':
        # Bayesian Ridge regression
        est = BayesianRidge(compute_score=True).fit(X, y)
        W = est.coef_

    elif key == 'LassoCV':
        # LassoCV
        est = LassoCV(cv=5, random_state=0).fit(X, y)
        W = est.coef_

    elif key == 'RidgeCV1':
        # RidgeCV
        est = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)
        W = est.coef_

    elif key == 'RidgeCV2':
        # RidgeCV2
        alphas = list(np.logspace(-8, 8, 100))
        est = RidgeCV(alphas=alphas, cv=5).fit(X, y)
        W = est.coef_

    elif key == 'RidgeCV3':
        # RidgeCV3
        alphas = list(np.logspace(-8, 8, 100))
        est = RidgeCV(alphas=alphas).fit(X, y)
        W = est.coef_

    elif key == 'EMB1':
        # EM-banded model
        est = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4))
        est.fit(X, y)
        W = est.W

    elif key == 'EMB2':
        # EM-banded model (early stopping)
        est = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4))
        est.set_early_stopping_tol(1e-8)
        est.fit(X, y)
        W = est.W

    elif key == 'EMB3':
        # EM-banded model (multi-dimensional)
        est = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4))
        est.set_multidimensional(True)
        est.fit(X, y)
        W = est.W

    elif key == 'EMB4':
        # EM-banded model (multi-dimensional, early stopping)
        est = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4))
        est.set_multidimensional(True)
        est.set_early_stopping_tol(1e-8)
        est.fit(X, y)
        W = est.W

    elif key == 'EMB1 (PyTorch)':
        # EM-banded model
        est = EMBanded_torch(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4))
        est.fit(X, y)
        W = est.W

    elif key == 'EMB2 (PyTorch)':
        # EM-banded model (early stopping)
        est = EMBanded_torch(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4))
        est.set_early_stopping_tol(1e-8)
        est.fit(X, y)
        W = est.W

    elif key == 'EMB3 (PyTorch)':
        # EM-banded model
        est = EMBanded_torch(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4))
        est.set_multidimensional(True)
        est.fit(X, y)
        W = est.W

    elif key == 'EMB4 (PyTorch)':
        # EM-banded model (early stopping)
        est = EMBanded_torch(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4))
        est.set_multidimensional(True)
        est.set_early_stopping_tol(1e-8)
        est.fit(X, y)
        W = est.W

    return W
