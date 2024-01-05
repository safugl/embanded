#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Most of this script is simply copied from:
    https://scikit-learn.org/stable/auto_examples/linear_model/
In the example Comparing Linear Bayesian Regressors
"""
# pylint: skip-file
from matplotlib.colors import SymLogNorm
import seaborn as sns
import matplotlib.pyplot as plt
from embanded.embanded_numpy import EMBanded
from sklearn.linear_model import ARDRegression, BayesianRidge, LinearRegression
import pandas as pd
from sklearn.datasets import make_regression

X, y, true_weights = make_regression(
    n_samples=100,
    n_features=100,
    n_informative=10,
    noise=8,
    coef=True,
    random_state=42,
)


olr = LinearRegression().fit(X, y)
brr = BayesianRidge(compute_score=True, n_iter=30).fit(X, y)
ard = ARDRegression(compute_score=True, n_iter=30).fit(X, y)

"""
We wish do declare separate hyperparameters to each predictor. We thus prepare 
a list F where each element in the list contains a column in X
"""

F = [X[:, [i]] for i in range(X.shape[1])]


"""
Fit the model
"""

emb = EMBanded(hyper_params=(1e-4, 1e-4, 1e-4, 1e-4))
emb.fit(F, y[:, None])

"""
For the sake of completeness we also consider tau=phi=eta=gamma=1e-6
"""


emb_alt = EMBanded(hyper_params=(1e-6, 1e-6, 1e-6, 1e-6))
emb_alt.fit(F, y[:, None])


df = pd.DataFrame(
    {
        "Weights of true generative process": true_weights,
        "ARDRegression": ard.coef_,
        "BayesianRidge": brr.coef_,
        "LinearRegression": olr.coef_,
        r"EM-banded ($\gamma=10^{-4}$)": emb.W.ravel(),
        r"EM-banded ($\gamma=10^{-6}$))": emb_alt.W.ravel(),
    }
)


fig = plt.figure(figsize=(10, 6))
ax = sns.heatmap(
    df.T,
    norm=SymLogNorm(linthresh=10e-4, vmin=-80, vmax=80),
    cbar_kws={"label": "coefficients' values"},
    cmap="seismic_r",
)
plt.ylabel("linear model")
plt.xlabel("coefficients")
plt.tight_layout(rect=(0, 0, 1, 0.95))
_ = plt.title("Models' coefficients")
