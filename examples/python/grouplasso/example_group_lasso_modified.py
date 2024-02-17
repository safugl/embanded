# pylint: skip-file
"""
This script is simply a modified version of an example described in 
https://group-lasso.readthedocs.io/en/latest/.

The original script can be found here:
https://github.com/yngvem/group-lasso/blob/master/examples/example_group_lasso.py

Notice that there are various extra plots included in the original simulation 
that are not included in the below script. 

Please see https://github.com/yngvem/group-lasso for more details and for the 
original simulations.
"""
"""
GroupLasso for linear regression
================================

A sample script for group lasso regression
"""

###############################################################################
# Setup
# -----

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from group_lasso import GroupLasso

np.random.seed(0)
GroupLasso.LOG_LOSSES = True


###############################################################################
# Set dataset parameters
# ----------------------
group_sizes = [np.random.randint(10, 20) for i in range(50)]
active_groups = [np.random.randint(2) for _ in group_sizes]
groups = np.concatenate(
    [size * [i] for i, size in enumerate(group_sizes)]
).reshape(-1, 1)
num_coeffs = sum(group_sizes)
num_datapoints = 10000
noise_std = 20


###############################################################################
# Generate data matrix
# --------------------
X = np.random.standard_normal((num_datapoints, num_coeffs))


###############################################################################
# Generate coefficients
# ---------------------
w = np.concatenate(
    [
        np.random.standard_normal(group_size) * is_active
        for group_size, is_active in zip(group_sizes, active_groups)
    ]
)
w = w.reshape(-1, 1)
true_coefficient_mask = w != 0
intercept = 2


###############################################################################
# Generate regression targets
# ---------------------------
y_true = X @ w + intercept
y = y_true + np.random.randn(*y_true.shape) * noise_std


###############################################################################
# View noisy data and compute maximum R^2
# ---------------------------------------
plt.figure()
plt.plot(y, y_true, ".")
plt.xlabel("Noisy targets")
plt.ylabel("Noise-free targets")
# Use noisy y as true because that is what we would have access
# to in a real-life setting.
R2_best = r2_score(y, y_true)


###############################################################################
# Generate estimator and train it
# -------------------------------
gl = GroupLasso(
    groups=groups,
    group_reg=5,
    l1_reg=0,
    frobenius_lipschitz=True,
    scale_reg="inverse_group_size",
    subsampling_scheme=1,
    supress_warning=True,
    n_iter=1000,
    tol=1e-3,
)
gl.fit(X, y)


###############################################################################
# Extract results and compute performance metrics
# -----------------------------------------------

# Extract info from estimator
yhat = gl.predict(X)
sparsity_mask = gl.sparsity_mask_
w_hat = gl.coef_



"""
The following code has been included to visualize behavior of the EM-banded model.
"""
###############################################################################
# Compare with EM-banded, note that we deepcopy for illustration purposes
from embanded.embanded_numpy import EMBanded
import copy
group_indices = np.unique(groups)

F = []
for i in group_indices:
    F.append(copy.deepcopy(X[:, groups.ravel() == i]))

# Fit the model
clf = EMBanded()
clf.fit(copy.deepcopy(F), copy.deepcopy(y))

estimated_weights = dict()
estimated_weights['GroupLasso'] = copy.deepcopy(w_hat)
estimated_weights['EM-banded'] = copy.deepcopy(clf.W)

mse = dict()
mse['GroupLasso'] = np.mean(np.abs(w-w_hat))
mse['EM-banded'] = np.mean(np.abs(w-clf.W))


fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

for k, key in enumerate(['GroupLasso','EM-banded']):
    ax[k].plot(w, label="Target weights", linewidth=1)
    ax[k].plot(estimated_weights[key], label="GroupLasso", linewidth=1)
    ax[k].legend()
    ax[k].set_title('%s, MAE=%0.3f' % (key, mse[key]))

fig.savefig('example_group_lasso.png',dpi=150)

