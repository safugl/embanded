"""Example01.

The simulation is copied from:
https://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html
"""
# pylint: skip-file
import copy
import time
import numpy as np
from sklearn.datasets import make_regression
import compare_models

# Copied from scikit-learn
X, y, true_weights = make_regression(
    n_samples=100,
    n_features=100,
    n_informative=10,
    noise=8,
    coef=True,
    random_state=42,
)

# from scipy.io import savemat
# mdic = dict(X=X, y=y, true_weights=true_weights)
# savemat("comparison01.mat", mdic)

num_iter = 10
keys = ['EMB1', 'EMB2', 'EMB3', 'EMB4', 'RidgeCV1',
        'RidgeCV2', 'RidgeCV3', 'ARD', 'BRR', 'OLS', 'LassoCV']

time_elapsed = dict()
loss = dict()
for key in keys:
    time_elapsed[key] = np.zeros((num_iter, 1))
    loss[key] = np.zeros((num_iter, 1))


"""
We wish do declare separate hyperparameters to each predictor. We thus prepare
a list F where each element in the list contains a column in X
"""

F = [copy.deepcopy(X[:, [i]]) for i in range(X.shape[1])]


for iteration in range(num_iter):

    print('At iteration: %i' % iteration)

    for key in keys:
        start = time.time()
        if 'EMB' in key:
            W_i = compare_models.fit_model(key, F, y[:, None])
        else:
            W_i = compare_models.fit_model(key, X, y)

        time_elapsed_i = time.time() - start

        loss[key][iteration] = (
            np.mean(np.power(W_i.ravel()-true_weights.ravel(), 2))
        )

        time_elapsed[key][iteration] = time_elapsed_i


print('Model | Runtime | MSE ')
print(':-|:---|:----')
for key in keys:
    print('%s | %0.5f s | %0.5f' %
          (key, np.mean(time_elapsed[key]), np.mean(loss[key])))
