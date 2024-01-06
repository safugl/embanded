#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sorenaf
"""
# pylint: skip-file
import sklearn.linear_model
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch
from embanded.embanded_torch import EMBanded


# Load data from the 'example06.mat' file
data = scipy.io.loadmat('example06.mat')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Split the data into predictor groups (F) and target variable (y)
F = []
for key in ['F1', 'F2']:
    F.append(torch.from_numpy(data[key]).to(device))

y = torch.from_numpy(data['y']).to(device)


# Check if the predictors and target variable are centered
assert np.isclose(torch.concatenate(F, axis=1).mean(axis=0).cpu(), 0).all(
), "The predictors have been centered, please see the MATLAB simulations"
assert np.isclose(y.mean(axis=0).cpu(), 0).all(
), "The target variable has been centered, please see the MATLAB simulations"


# Create a grid of subplots for plotting results
f, ax = plt.subplots(2, 4, sharex=True, figsize=(15, 7.5))

# Iterate over different parameter values
for k, param in enumerate([1e-4, 1e-3, 1e-2, 1e-1]):

    # Initialize EM-banded model
    emb = EMBanded(hyper_params=(param, param, param, param),
                   max_iterations=200)
    emb.set_verbose(True)
    # Fit the model
    emb.fit(F, y)

    # Plot the estimated weights for this parameter
    ax[0, k].plot(emb.W.cpu().numpy(), '-k')
    ax[0, k].set_title(r'$\eta=\phi=\kappa=\tau=%0.1e$' % param)

    # Check if the estimated weights match the provided data
    assert np.isclose(data['W_estimated'][0, k], emb.W.cpu().numpy()).all(
    ), 'The estimated weights are not matching'

    # As a point of reference, we also fit Ridge models with scikit-learn and
    # compare these with the estimates stored in the mat file.
    # Initialize the Ridge regression model with the specified alpha
    ridge = sklearn.linear_model.Ridge(
        alpha=1./param, fit_intercept=False, solver='cholesky', copy_X=True)

    # Concatenate predictors (F) and fit the Ridge regression model
    X = torch.concatenate(F, axis=1).cpu().numpy()

    # Plot the estimated weights
    ax[1, k].plot(ridge.fit(X, y.cpu().numpy()).coef_.ravel(), '-r')
    ax[1, k].set_title(r'$\alpha=%0.1e$' % (1./param))

    # Check if the estimated weights match the provided data
    assert np.isclose(ridge.fit(X, y.cpu().numpy()).coef_.ravel(),
                      data['W_ridge'][0, k].ravel()).all()
