# pylint: skip-file
"""
This script is simply a modified version of an example described in 
https://group-lasso.readthedocs.io/en/latest/.

The original script can be found here:
https://github.com/yngvem/group-lasso/blob/master/examples/example_sparse_group_lasso.py

Notice that there are various extra plots included in the original simulation 
that are not included in the below script. 

Please see https://github.com/yngvem/group-lasso for more details and for the 
original simulations.
"""
"""
GroupLasso for linear regression with dummy variables
=====================================================

A sample script for group lasso with dummy variables
"""

###############################################################################
# Setup
# -----

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from group_lasso import GroupLasso
from group_lasso.utils import extract_ohe_groups

np.random.seed(42)
GroupLasso.LOG_LOSSES = True


###############################################################################
# Set dataset parameters
# ----------------------
num_categories = 30
min_options = 2
max_options = 10
num_datapoints = 10000
noise_std = 1


###############################################################################
# Generate data matrix
# --------------------
X_cat = np.empty((num_datapoints, num_categories))
for i in range(num_categories):
    X_cat[:, i] = np.random.randint(min_options, max_options, num_datapoints)

ohe = OneHotEncoder()
X = ohe.fit_transform(X_cat)
groups = extract_ohe_groups(ohe)
group_sizes = [np.sum(groups == g) for g in np.unique(groups)]
active_groups = [np.random.randint(0, 2) for _ in np.unique(groups)]


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
# Generate pipeline and train it
# ------------------------------
pipe = pipe = Pipeline(
    memory=None,
    steps=[
        (
            "variable_selection",
            GroupLasso(
                groups=groups,
                group_reg=0.1,
                l1_reg=0,
                scale_reg=None,
                supress_warning=True,
                n_iter=100000,
                frobenius_lipschitz=False,
            ),
        ),
        ("regressor", Ridge(alpha=1)),
    ],
)
pipe.fit(X, y)


###############################################################################
# Extract results and compute performance metrics
# -----------------------------------------------

# Extract from pipeline
yhat = pipe.predict(X)
sparsity_mask = pipe["variable_selection"].sparsity_mask_
coef = pipe["regressor"].coef_.T

# Construct full coefficient vector
w_hat = np.zeros_like(w)
w_hat[sparsity_mask] = coef


"""
The following code has been included to visualize behavior of the EM-banded model.
"""
###############################################################################
# Compare with EM-banded, note that we deepcopy for illustration purposes
from embanded.embanded_numpy import EMBanded
import copy

# There are 30 groups
group_indices = np.unique(groups)

# Create a list that contains predictors associated with each group. Notice 
# that we have to use .toarray()
F = []
for i in group_indices:
    F.append(copy.deepcopy(X[:,groups==i]).toarray())

# Fit the EM-banded model
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

fig.savefig('example_sparse_group_lasso.png',dpi=150)

