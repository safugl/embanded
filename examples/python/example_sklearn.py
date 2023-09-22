#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy paste from https://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html#sphx-glr-auto-examples-linear-model-plot-ard-py

Compare with the reference models
"""

from sklearn.datasets import make_regression

X, y, true_weights = make_regression(
    n_samples=100,
    n_features=100,
    n_informative=10,
    noise=8,
    coef=True,
    random_state=42,
)

import pandas as pd

from sklearn.linear_model import ARDRegression, BayesianRidge, LinearRegression
import embanded

olr = LinearRegression().fit(X, y)
brr = BayesianRidge(compute_score=True, n_iter=30).fit(X, y)
ard = ARDRegression(compute_score=True, n_iter=30).fit(X, y)

"""
We wish do declare separate hyperparameters to each predictor. We thus prepare a list F where each element in the list contains a column in X
"""

F = [X[:,[i]] for i in range(X.shape[1])]


"""
Fit the model
"""

clf_em = embanded.EMBanded(num_features=len(F),remove_intercept=True,max_iterations=200,
                                   tau =1e-4,
                                   phi=1e-4,
                                   eta=1e-4,
                                   kappa=1e-4)
clf_em.fit(F,y[:,None])

"""
For the sake of completeness we also consider tau=phi=eta=gamma=1e-6
"""


clf_em_alt = embanded.EMBanded(num_features=len(F),remove_intercept=True,max_iterations=200,
                                   tau =1e-6,
                                   phi=1e-6,
                                   eta=1e-6,
                                   kappa=1e-6)
clf_em_alt.fit(F,y[:,None])


df = pd.DataFrame(
    {
        "Weights of true generative process": true_weights,
        "ARDRegression": ard.coef_,
        "BayesianRidge": brr.coef_,
        "LinearRegression": olr.coef_,
        r"EM-banded ($\gamma=1e-4$)": clf_em.W.ravel(),
        r"EM-banded ($\gamma=1e-6$)": clf_em_alt.W.ravel(),
    }
)


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import SymLogNorm

plt.figure(figsize=(10, 6))
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

# The following command was used to store the file "example_sklearn.png":
# plt.savefig('example_sklearn.png')
