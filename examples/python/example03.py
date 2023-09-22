#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sorenaf
"""

import embanded
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# Load data from the 'example03.mat' file
data = scipy.io.loadmat('example03.mat')


# Split the data into predictor groups (F) and target variable (y)
X = np.concatenate(data['F'].ravel(),axis=1)
num_features = X.shape[1]
F = []
for j in range(X.shape[1]):
    F.append(X[:,j][:,None])
y = data['y']


# Check if the predictors and target variable are centered
assert np.isclose(np.concatenate(F,axis=1).mean(axis=0),0).all(), "The predictors have been centered, please see the MATLAB simulations"
assert np.isclose(y.mean(axis=0),0).all(), "The target variable has been centered, please see the MATLAB simulations"


# Create a grid of subplots for plotting results
f,ax = plt.subplots(2,4,sharex=True,figsize=(15,7.5))

for k, param in enumerate([1e-4,1e-3,1e-2,1e-1]):

    clf = embanded.EMBanded(num_features=num_features,remove_intercept=False,max_iterations=200,
                                tau =param,
                                phi=param,
                                eta=param,
                                kappa=param,
                                multi_dimensional=False)
    
    # Fit the model
    summary = clf.fit(F,y)
    
    
    # Plot the estimated weights for this parameter
    ax[0,k].plot(clf.W,'-k')
    ax[0,k].set_title(r'$\eta=\phi=\kappa=\tau=%0.1e$'%param)
    
    
    # Check if the estimated weights match the provided data
    assert np.isclose(data['W_estimated'][0,k],clf.W).all(), 'The estimated weights are not matching'



    # Import the Ridge regression model from scikit-learn
    import sklearn.linear_model
    
    # Initialize the Ridge regression model with the specified alpha
    clf_ridge = sklearn.linear_model.Ridge(alpha=1./param,fit_intercept=False,solver='cholesky',copy_X=True)
    
    # Concatenate predictors (F) and fit the Ridge regression model
    X = np.concatenate(F,axis=1)

    # Plot the estimated weights
    ax[1,k].plot(clf_ridge.fit(X,y).coef_.ravel(),'-r')
    ax[1,k].set_title(r'$\alpha=%0.1e$'%(1./param))
    
    # Check if the estimated weights match the provided data
    assert np.isclose(clf_ridge.fit(X,y).coef_.ravel(),data['W_ridge'][0,k].ravel()).all()
