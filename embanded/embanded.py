#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sorenaf
"""


import numpy as np
import scipy.linalg
import copy
import time
from  scipy.linalg import cholesky, cho_solve   
import numba 
from numba import njit
import numba as nb
from numba import jit

    
class EMBanded:
    """Expectation-Maximization algorithm for estimating regularized regression
    model with banded prior structure.


    Parameters
    ----------
    num_features : integer,
        Specify the number of feature sets. This value must must be equal to 
        len(X) = len([F1, F2, ..., Fj]). The value is used to generate
        default values for lambdas and nu

    max_iterations : integer, default = 200
        Specify the maximum allow number of iterations

    nu : float, default = 1
        Specify the initial value of the nu hyperparameter which controls
        observation noise variance.

    lambdas : ndarray, default =  np.ones(num_features)
        Specify the initial values of the lambdas hyperparameters. The length
        of lambda must be equal to len(X) = len([F1, F2, ..., Fj])

    tau : float, default = 1e-4
        Specify hyperparameter tau related to the Inverse-Gamma priors imposed
        on the lambda_j terms

    eta : float, default = 1e-4
        Specify hyperparameter eta related to the Inverse-Gamma priors imposed
        on the lambda_j terms

    phi : float, default = 1e-4
        Specify hyperparameter phi related to the Inverse-Gamma prior imposed
        on the nu term

    kappa : float, default = 1e-4
        Specify hyperparameter kappa related to the Inverse-Gamma prior imposed
        on the nu term

    remove_intercept : bool, default=True
        Whether to remove offsets in X and Y prior to fitting the model. If set
        to false, the data will not be transformed prior to model fitting.
        However, in this case, the model will complain if the columns in X or y
        have not been adequately centered. If set to true, then the offsets
        will be stored in the object as X_offset and y_offset. These values will 
        be used for the model predictions.
        
    show_progress : bool, default=True
        Whether to show progress and time elapsed.
                
    h : ndarray, default = np.tile(np.nan,num_features) (no smoothness)
        Specify hyperparameter h related to the covariance parametrization. 
        It is possible to define that h = np.array([1,np.nan]) in which
        case the first Omega_1 term will be parameterized with a Matern kernel 
        and in which case the Omega_2 term will be a unit matrix.
        

    use_matrix_inversion_lemma : boolean, default = false
        Specify whether the Woodbury Matrix Identity should be used for
        computing inv(Sigma).
        
    multi_dimensional : boolean, default = false
        Whether to make simplifying assumptions to allow for an efficient
        estimation of weights in cases where y has multiple columns. 


    """

    def __init__(
        self,
        num_features,
        nu=1.0,
        max_iterations=200,
        tau=1e-4,
        eta=1e-4,
        phi=1e-4,
        kappa=1e-4,
        show_progress=True,
        remove_intercept=True,
        h=np.array([]),
        lambdas=None,
        use_matrix_inversion_lemma=False,
        multi_dimensional=False,
    ):
        self.nu = nu
        self.max_iterations = max_iterations
        self.eta = eta
        self.tau = tau
        self.phi = phi
        self.kappa = kappa
        self.remove_intercept = remove_intercept

        if not lambdas == None:
            self._check_lambdas(lambdas, num_features)
            self.lambdas = lambdas
        else:
            # Initialize the lambdas to be equal to one
            self.lambdas = np.ones(num_features)
        
        self.show_progress = show_progress
        self.h = np.tile(np.nan,num_features)
        if len(h)==num_features:
            self.h = h
        self._check_h(self.h, num_features)

        if np.isnan(self.h).all():
            self.smoothness_prior = False
        else:
            self.smoothness_prior = True
        
     
        self.use_matrix_inversion_lemma = use_matrix_inversion_lemma
        self.num_features = num_features
        self.multi_dimensional = multi_dimensional

    def fit(self, F, y):
        """
        Fit the model

        Parameters
        ----------
        F : list
            A list of ndarrays of where each array should have
            dimensionality (M x D_j) where M is the number of samples (rows) and
            where D_j is the number of columns of that given feature space (D_j>=1)..
        y : ndarray 
            A column vector of size (M x 1) where M is the number of samples
            (rows). The number of samples should be exactly identical to the
            number of rows in each entry in F.
        """
        assert isinstance(F, list), "The input data must be a list"

        # Combine the features into a matrix
        X = np.concatenate(F, axis=1).copy()

        # Prepare y
        y = np.asarray(y, dtype=X.dtype)

        assert self.num_features == len(F), "The number of features are not matching <num_features>"
        num_features = self.num_features
        
        
        # Total number of columns in X
        num_dim = X.shape[1]

        # Total number of observations
        num_obs = X.shape[0]

        # Remove offset unless the user have turned this functionality off
        X, y = self._preprocess_data(X, y)

        # Do a few assertations to check that things are in order in terms of rows
        # and columns.
        if not num_obs == y.shape[0]:
            raise TypeError(
                "The number of observations in F and y are not matching"
            )
        if not num_obs > 1:
            raise TypeError("y should have more than one observation")

        if self.multi_dimensional == False:
            if not y.shape[1] == 1:
                raise TypeError("y should be a column vector")
        if not y.ndim <= 2:
            raise TypeError("y has too many dimensions")

        if not num_dim > 0:
            raise TypeError("X should be a matrix")

        # Check also if the data has been (approximately) mean-centered
        if not np.all(np.abs(np.mean(X, axis=0)) < 10e-8):
            raise TypeError(
                "It appears that X has columns that have not been centered. Please do so."
            )
        if not np.all(np.abs(np.mean(y, axis=0)) < 10e-8):
            raise TypeError(
                "It appears that y has not been centered. Please do so."
            )

        # Create an indexing array <mat_indexer> that indexes the different 
        # feature groups.
        columns_group = np.concatenate([np.ones(F[j].shape[1])*j for j in range(len(F))],axis=0)
        mat_indexer = np.zeros((num_dim,num_features))
        for j in range(num_features):
            columns_j = np.where(columns_group == j)[0]
            mat_indexer[columns_j,j]=1
        
        self.mat_indexer = mat_indexer
        self._check_lambdas(self.lambdas, num_features)

        if self.smoothness_prior == True:
            self.prepare_smoothness_cov(F)
        else:
            self.Omega = np.eye(num_dim)
            self.Omega_inv = np.eye(num_dim)




        if self.show_progress == True:
            start_time = time.time()
  
        W, lambdas_summary, nu_summary, Sigma = model_fit(X,
                      y,
                      self.mat_indexer,
                      copy.deepcopy(self.lambdas),
                      copy.deepcopy(self.nu),
                      self.max_iterations,
                      self.tau,
                      self.eta,
                      self.kappa,
                      self.phi,
                      self.Omega,
                      self.Omega_inv,
                      self.use_matrix_inversion_lemma,
                      self.multi_dimensional,
                      self.smoothness_prior,
                      self.h)
      
        if self.show_progress == True:
            print('Time elapsed: %0.2f'%(time.time()-start_time))
        summary = dict()
        summary["lambdas"] = lambdas_summary
        summary["nu"] = nu_summary
                
        self.W = W
        self.Sigma = Sigma
        return summary

    def predict(self, F_test):
        """Predict using the EM-banded regression model

        Parameters
        ----------
        F_test : list
            A list of ndarrays of where each array should have
            dimensionality (M x D_j) where M is the number of samples (rows) and
            where D_j is the number of columns of that given feature space (D_j>=1). 
            The list should have the same format as that used for training the model.

        Returne
        ----------        
        y_pred : ndarray
            Returns predicted values

        """

        X_test = copy.deepcopy(np.concatenate(F_test, axis=1))

        if self.remove_intercept == True:
            X_test -= self.X_offset

        return X_test @ self.W + self.y_offset

    def _preprocess_data(self, X_in, y_in):
        
        X = copy.deepcopy(X_in)
        y = copy.deepcopy(y_in)
        
        if self.remove_intercept == True:
            self.X_offset = X.mean(axis=0, keepdims=True)
            self.y_offset = y.mean(axis=0, keepdims=True)

            X -= self.X_offset
            y -= self.y_offset

        else:
            pass

        return X, y

    def _check_lambdas(self, lambdas, num_features):

        if lambdas is not None and not isinstance(lambdas, np.ndarray):
            lambdas = np.array(lambdas)

            if np.min(lambdas) <= 0:
                raise ValueError("The lambdas should be greater than zero")

        if not len(lambdas) == num_features:
            raise TypeError(
                "The number of feature dimensions are not matching lambdas"
            )

    def _check_h(self, h, num_features):

        if h is not None and not isinstance(h, np.ndarray):
            h = np.array(h)

            if np.min(h) <= 0:
                raise ValueError("The h values should be greater than zero")

        if not len(h) == num_features:
            raise TypeError(
                "The number of feature dimensions are not matching h"
            )

    def prepare_smoothness_cov(self, F):

        assert isinstance(F, list), "The input data must be a list"

        num_features = len(F)

        if not num_features == len(self.lambdas):
            raise TypeError("The number of elements in F should match lambdas")

        if not num_features == len(self.h):
            raise TypeError("The number of elements in F should match h")

        Omega = []
        Omega_i = []
        for j in range(num_features):

            #  The number of columns in F{f}
            D_j = F[j].shape[1]

            # Define a grid [0,1,2,3,...,D_j-1]
            x_grid = np.arange(D_j)[None, ...]

            if np.isnan(self.h[j]):
                # If it is specified as NaN then assume that it
                # should be an identity matrix
                Omega_j = np.eye(D_j)
                Omega_j_i = np.eye(D_j)
            else:
                # When h[j] is a scalar then define Omega as follows:
                Omega_j = (
                    1 + np.sqrt(3) * np.abs(x_grid.T - x_grid) / self.h[j]
                ) * np.exp(-np.sqrt(3) * np.abs(x_grid.T - x_grid) / self.h[j])
                
                Omega_j_i = cho_solve((cholesky(Omega_j, lower=True), True), np.eye(Omega_j.shape[1]))


            Omega.append(Omega_j)
            Omega_i.append(Omega_j_i)

        # Prepare a matrix over entire Lambda
        self.Omega = scipy.linalg.block_diag(*Omega)

        # The inverse of this matrix
        self.Omega_inv = scipy.linalg.block_diag(*Omega_i)

   



@njit
def model_fit(X,
              y,
              mat_indexer,
              lambdas_in,
              nu_in,
              max_iterations,
              tau,
              eta,
              kappa,
              phi,
              Omega,
              Omega_inv,
              use_matrix_inversion_lemma,
              multi_dimensional,
              smoothness_prior,
              h):
    
    num_obs = X.shape[0]
    num_dim = X.shape[1]
    num_features = len(lambdas_in)
    P = y.shape[1]
    covXy = X.T @ y

    lambdas_summary = np.zeros((max_iterations,len(lambdas_in)),dtype=np.float64)
    lambdas = lambdas_in
    
    nu_summary = np.zeros((max_iterations,1),dtype=np.float64)
    nu = nu_in

    if use_matrix_inversion_lemma == False:
        covX = X.T @ X
        
    for iteration in range(max_iterations):
        
        lambdas_summary[iteration,:] = lambdas
        nu_summary[iteration,0] = nu

        # Create a vector that will form the diagonal elements in the L matrix 
        lambdas_diag = mat_indexer@lambdas
        
        
        if smoothness_prior == True:
            if use_matrix_inversion_lemma == False:
                L_inv = np.expand_dims(1.0 / lambdas_diag,-1)*Omega_inv
            elif use_matrix_inversion_lemma == True:
                L = np.expand_dims(lambdas_diag,-1) * Omega
        elif smoothness_prior == False:
            if use_matrix_inversion_lemma == False:
                L_inv = np.diag(1.0 / lambdas_diag)
            elif use_matrix_inversion_lemma == True:
                L = np.diag(lambdas_diag)
                
                  
                
        if use_matrix_inversion_lemma == False:
            S = 1.0 / nu * covX + L_inv
            H = np.linalg.inv(np.linalg.cholesky(S))
            Sigma = H.T@H
        else:
            S = nu * np.eye(num_obs) + X @ L @ X.T
            H = np.linalg.inv(np.linalg.cholesky(S))
            Si = H.T@H
            Sigma = L - L@X.T@Si@X@L
           
         
            
       
        
        
        # Estimate the new weights according to <W = 1 / nu * Sigma * X' * y;>
        W = 1.0 / nu * Sigma @ covXy
        
        if iteration < max_iterations-1:
            
            for f in range(num_features):
                
                index = mat_indexer[:,f]==1
                D_f = np.sum(index)
                mu_f = W[index]
                Sigma_ff = Sigma[index,:][:,index]
                
                
                if multi_dimensional == False:
                    # Find new lambda
                    E = 0.
                    if (smoothness_prior == True) and (not np.isnan(h[f])):
                        
                        Omega_inv_ff = Omega_inv[index,:][:,index]
                        
                        E = ( mu_f.T @ Omega_inv_ff @ mu_f
                              + np.trace(Omega_inv_ff @ Sigma_ff) )[0][0]
                    else:
                        
                        E = ( mu_f.T @ mu_f + np.trace(Sigma_ff) )[0][0]

                elif multi_dimensional == True:
                    E = 0.
                    
                    if (smoothness_prior == True) and (not np.isnan(h[f])):
                        
                        Omega_inv_ff = Omega_inv[index,:][:,index]
                        
            
                        E = E + (np.trace(mu_f @ mu_f.T @ Omega_inv_ff ) + np.trace(Omega_inv_ff @ Sigma_ff))
                     
                    else:
                        
                        E = E + (np.trace(mu_f @ mu_f.T  ) + np.trace(Sigma_ff))

                
                if multi_dimensional == False:

                    lambda_f = (E + 2 * tau  ) / ( D_f + 2 * eta + 2)

                elif multi_dimensional == True:
                    # Just spell things out
                    lambda_f = (E + 2 * tau  ) / (P * D_f + 2 * eta + 2)
        
                
                
                
                lambdas[f] = lambda_f
                
            # Make point-predictions
            prediction = X @ W
            
            if multi_dimensional == True:
                n_rows = y.shape[0]*y.shape[1]
                target = np.reshape(y,(n_rows,1))
                prediction = np.reshape(prediction,(n_rows,1))
            elif multi_dimensional == False:
                target = y
                
                
                

            
            if use_matrix_inversion_lemma == False:
                nu = (
                    ((target - prediction).T @ (target - prediction))[0][0]
                    + np.trace(covX @ Sigma)
                    + 2 * kappa
                ) / (num_obs + 2 + 2 * phi)
            else:
                nu = (
                    ((target - prediction).T @ (target - prediction))[0][0]
                    + np.trace(X @ Sigma @ X.T)
                    + 2 * kappa
                ) / (num_obs + 2 + 2 * phi)
                
           
    return W, lambdas_summary,nu_summary, Sigma
    