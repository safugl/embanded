#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sorenaf
"""

import embanded
import scipy.io
import matplotlib.pyplot as plt
import numpy as np 

# Load data from the 'example02.mat' file
data = scipy.io.loadmat('example02.mat')

# Split the data into predictor groups (F) and target variable (y)
F = [data['F1'], data['F2']]
y = data['y']

# Check if the predictors and target variable are centered
assert np.isclose(np.concatenate(F,axis=1).mean(axis=0),0).all(), "The predictors have been centered, please see the MATLAB simulations"
assert np.isclose(y.mean(axis=0),0).all(), "The target variable has been centered, please see the MATLAB simulations"


# Create a grid of subplots for plotting results
f,ax = plt.subplots(2,4,sharex=True,figsize=(15,7.5))

alphas = [1e4,1e3,1e2,1e1]

# Iterate over different smoothness parameter values
for k, hv in enumerate([np.nan,1,5,10]):

    # Initialize EM-banded model
    clf = embanded.EMBanded(num_features=2,remove_intercept=False,max_iterations=200,
                                tau=1e-4,
                                phi=1e-4,
                                eta=1e-4,
                                kappa=1e-4,
                                multi_dimensional=False,
                                h=np.array([hv, np.nan]))
    
    # Fit the model   
    summary = clf.fit(F,y)

    # Plot the estimated weights for this parameter
    ax[0,k].plot(clf.W,'-k')
    
    if np.isnan(hv):
        ax[0,k].set_title(r'$\gamma=%0.1e, no smoothnes$')
    else:
        ax[0,k].set_title(r'$\gamma=%0.1e, h=%i$'%(1e-4,hv))

     
     

    # Check if the estimated weights match the provided data
    assert np.isclose(data['W_estimated'][0,k],clf.W).all(), 'The estimated weights are not matching'



    # Import the Ridge regression model from scikit-learn
    import sklearn.linear_model
    
    # Initialize the Ridge regression model with the specified alpha
    clf_ridge = sklearn.linear_model.Ridge(alpha=alphas[k],fit_intercept=False,solver='cholesky',copy_X=True)
    
    # Concatenate predictors (F) and fit the Ridge regression model
    X = np.concatenate(F,axis=1)

    # Plot the estimated weights
    ax[1,k].plot(clf_ridge.fit(X,y).coef_.ravel(),'-r')
    ax[1,k].set_title(r'$\alpha=%0.1e$'%(alphas[k]))
    
    # Check if the estimated weights match the provided data
    assert np.isclose(clf_ridge.fit(X,y).coef_.ravel(),data['W_ridge'][0,k].ravel()).all()
