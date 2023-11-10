#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: skip-file
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import ARDRegression, BayesianRidge
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
import numpy as np
import embanded

np.random.seed(1)

# Simulate some data
F1 = np.random.randn(100, 20)
F2 = np.random.randn(100, 50)
F = [F1, F2]
X = np.concatenate(F, axis=1)
W = np.concatenate(
    [np.random.randn(20, 1), np.zeros((50, 1))], axis=0)/np.sqrt(40)
N = np.random.randn(100, 1)/np.sqrt(2)
Y = X@W + N

emb = embanded.EMBanded(num_features=len(F))
emb.fit(F, Y)


olr = LinearRegression().fit(X,  Y.squeeze())
brr = BayesianRidge(compute_score=True, max_iter=300).fit(X, Y.squeeze())
ard = ARDRegression(compute_score=True, max_iter=300).fit(X, Y.squeeze())
ridgecv = RidgeCV(
    alphas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]).fit(X, Y.squeeze())
lassocv = LassoCV(cv=5, random_state=0).fit(X, Y.squeeze())

summary = {'EM-banded': emb.W,
           'LinearRegression': olr.coef_,
           'BayesianRidge': brr.coef_,
           'ARDRegression': ard.coef_,
           'RidgeCV': ridgecv.coef_,
           'LassoCV': lassocv.coef_}


cmap = matplotlib.colormaps['Dark2'].resampled(8)
f, ax = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True)
ax = ax.reshape(-1)
for k, key in enumerate(['EM-banded', 'LinearRegression', 'BayesianRidge',
                         'ARDRegression', 'RidgeCV', 'LassoCV']):
    ax[k].plot(W, label='Target', color='k')
    ax[k].plot(summary[key], '-', label=key, color=cmap(k))
    ax[k].legend()
    ax[k].set_xticks([0, 20, 70])
