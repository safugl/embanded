#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: skip-file
import sklearn.linear_model
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import embanded

# Load data from the 'example02.mat' file
data = scipy.io.loadmat('example02.mat')

# Split the data into predictor groups (F) and target variable (y)
F = [data['F1'], data['F2']]
y = data['y']

# Check if the predictors and target variable are centered
assert np.isclose(np.concatenate(F, axis=1).mean(axis=0), 0).all(
), "The predictors have been centered, please see the MATLAB simulations"
assert np.isclose(y.mean(axis=0), 0).all(
), "The target variable has been centered, please see the MATLAB simulations"


# Create a grid of subplots for plotting results
f, ax = plt.subplots(2, 4, sharex=True, figsize=(15, 7.5))

alphas = [1e4, 1e3, 1e2, 1e1]

# Iterate over different smoothness parameter values
for k, hv in enumerate([None, 1., 5., 10.]):

    # Initialize EM-banded model
    emb = embanded.EMBanded(num_features=2, hyper_params=(1e-4, 1e-4, 1e-4, 1e-4),
                            max_iterations=200)
    emb.set_smoothness_param([hv, None])

    # Fit the model
    emb.fit(F, y)

    # Plot the estimated weights for this parameter
    ax[0, k].plot(emb.W, '-k')

    if not hv:
        ax[0, k].set_title(r'$\gamma=%0.1e$, no smoothnes' % (1e-4))
    else:
        ax[0, k].set_title(r'$\gamma=%0.1e, h=%i$' % (1e-4, hv))

    # Check if the estimated weights match the provided data
    assert np.isclose(data['W_estimated'][0, k], emb.W).all(
    ), 'The estimated weights are not matching'

    # As a point of reference, we also fit Ridge models with scikit-learn and
    # compare these with the estimates stored in the mat file.
    # Initialize the Ridge regression model with the specified alpha
    ridge = sklearn.linear_model.Ridge(
        alpha=alphas[k], fit_intercept=False, solver='cholesky', copy_X=True)

    # Concatenate predictors (F) and fit the Ridge regression model
    X = np.concatenate(F, axis=1)

    # Plot the estimated weights
    ax[1, k].plot(ridge.fit(X, y).coef_.ravel(), '-r')
    ax[1, k].set_title(r'$\alpha=%0.1e$' % (alphas[k]))

    # Check if the estimated weights match the provided data
    assert np.isclose(ridge.fit(X, y).coef_.ravel(),
                      data['W_ridge'][0, k].ravel()).all()
