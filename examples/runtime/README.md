# Runtime  
This page provides examples of runtime tests. For comparison, runtime is monitored for other related estimators, as outlined in the table below. It's important to approach the runtime estimates with caution. Each estimator comes with various hyperparameters that can significantly influence runtime, and the choices below are somewhat arbitrary. The reported values are crude point timing measures for a few specific scenarios.

Acronym | Model | Notes | Link
:-|:---|:----|:---
EMB1 | EM-banded estimator | Vectorized code, 200 iterations |
EMB2 | EM-banded estimator | Vectorized code, early stopping (tol: 1e-8) |
EMB3 | EM-banded estimator | Nested loops, 200 iterations |
EMB4 | EM-banded estimator | Nested loops, early stopping (tol: 1e-8) | 
RidgeCV1 | Cross-validated Ridge estimator | alphas = [0.1, 1.0, 10.0], efficient leave-one-out cross-validation | [sklearn.linear_model.RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html )
RidgeCV2 | Cross-validated Ridge estimator | alphas = list(np.logspace(-8,8,100)), cv=5 | [sklearn.linear_model.RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html )
RidgeCV3 | Cross-validated Ridge estimator | alphas = list(np.logspace(-8,8,100)), efficient leave-one-out cross-validation  | [sklearn.linear_model.RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html )
ARD | Bayesian ARD regression | Default settings. | [sklearn.linear_model.ARDRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression)
BRR | Bayesian ridge regression | Default settings. | [sklearn.linear_model.BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge)
OLS | Ordinary least squares | Default settings.  | [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
LassoCV | Cross-validated Lasso  |  cv=5, random_state=0 | [sklearn.linear_model.LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)

# Example 01
*Tested on a MacBook Pro (16 GB 2400 MHz DDR4 Radeon Pro 555X 4 GB Intel UHD Graphics 630 1536 MB).* 

This example utilizes synthetic data generated through a simulation described in a [scikit-learn example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html#sphx-glr-auto-examples-linear-model-plot-ard-py). In brief, it involves a design matrix with 100 predictors, a 1-dimensional target variable 'y,' and a dataset of 100 samples. The target weights are assumed to be sparse. Distinct hyperparameters are assigned to each predictor for the EM-banded model. The table below presents both the runtime and the mean square error (MSE) between the estimated weights and the target weights. The procedure is repeated for the EM-banded implementation in Matlab. Note that just-in-time (JIT) acceleration can improve runtime in some circumstances. The results are shown below.

Model | Runtime | MSE 
:-|:---|:----
EMB1 | 0.06909 s | 0.06395
EMB2 | 0.01538 s | 0.06395
EMB3 | 0.44119 s | 0.06395
EMB4 | 0.07698 s | 0.06395
RidgeCV1 | 0.00322 s | 10.57044
RidgeCV2 | 0.72733 s | 8.20769
RidgeCV3 | 0.01109 s | 10.89148
ARD | 0.62682 s | 0.86433
BRR | 0.00809 s | 14.15242
OLS | 0.00267 s | 53.53597
LassoCV | 0.05628 s | 0.24920
EMB1 (Matlab) | 0.07071 s | 0.06395
EMB2 (Matlab) | 0.01411 s | 0.06395
EMB3 (Matlab) | 0.08310 s | 0.06395
EMB4 (Matlab) | 0.01556 s | 0.06395

# Example 02
*Tested on a MacBook Pro (16 GB 2400 MHz DDR4 Radeon Pro 555X 4 GB Intel UHD Graphics 630 1536 MB).* 

In this example, we explore a scenario with a moderate number of predictors (128 predictors) and an increasing number of observations. The simulations involve two predictor groups, F1 and F2, each containing 64 predictors. The target variable y is assumed to be a mixed version of F1 plus additive Gaussian noise. We monitor the runtime for simulations with varying numbers of observations, ranging from 128 samples to 65536 samples. Notice that the EM-banded models with early stopping (EMB2 and EMB4) requires computation of log-score unlike EMB1 and EMB3.

Model |  128 samples |  1024 samples |  16384 samples |  65536 samples |  
:-|:-|:-|:-|:-
EMB1 |  0.08121 s |  0.09512 s |  0.35699 s |  1.30781 s |  
EMB2 |  0.09065 s |  0.09110 s |  0.08192 s |  0.25147 s |  
EMB3 |  0.09265 s |  0.10532 s |  0.35106 s |  1.28950 s |  
EMB4 |  0.10431 s |  0.10164 s |  0.08514 s |  0.25461 s |  
RidgeCV1 |  0.00411 s |  0.01113 s |  0.19025 s |  1.21758 s |  
RidgeCV2 |  0.72091 s |  1.05844 s |  9.96278 s |  39.90405 s |  
RidgeCV3 |  0.01153 s |  0.05050 s |  1.22755 s |  5.56638 s |  
ARD |  0.15752 s |  0.04715 s |  0.22426 s |  0.63214 s |  
BRR |  0.00852 s |  0.01213 s |  0.20329 s |  1.42042 s |  
OLS |  0.00388 s |  0.00636 s |  0.11977 s |  0.80193 s |  
LassoCV |  0.79701 s |  0.10971 s |  0.43248 s |  1.63593 s | 

# Example 03:
*Tested with access to NVIDIA GeForce RTX 3090, NVIDIA-SMI 535.104.12, CUDA Version: 12.2*

This example is an extension of Example 02. In this case, we use the PyTorch implementation and fit models with an increasing number of predictors. We once again assume a scenario with two predictor groups, F1 and F2, and one target variable. In this case, we assume 65536 observations, but here increase the number of predictors (number of dimensions) from 128 to 8192. Timing is reported below with float64 precision.

Model |  128 dimensions |  512 dimensions |  1024 dimensions |  2048 dimensions |  
:-|:-|:-|:-|:-
EMB1 (PyTorch) |  0.24352 s |  1.26022 s |  3.79259 s |  16.35053 s |  
EMB2 (PyTorch) |  0.04665 s |  0.32395 s |  1.25555 s |  7.60263 s |  
EMB3 (PyTorch) |  0.27069 s |  1.20210 s |  3.69885 s |  16.12053 s |  
EMB4 (PyTorch) |  0.04390 s |  0.32252 s |  1.24672 s |  7.46866 s | 

This procedure is repeated and shown below with float32 precision.

Model |  128 dimensions |  512 dimensions |  1024 dimensions |  2048 dimensions |  
:-|:-|:-|:-|:-
EMB1 (PyTorch) |  0.15383 s |  0.32976 s |  0.85559 s |  2.67104 s |  
EMB2 (PyTorch) |  0.01956 s |  0.04177 s |  0.11915 s |  0.46552 s |  
EMB3 (PyTorch) |  0.21036 s |  0.32482 s |  0.80547 s |  2.58599 s |  
EMB4 (PyTorch) |  0.01434 s |  0.04434 s |  0.10505 s |  0.45519 s |  


# Example 04
*Tested with access to NVIDIA GeForce RTX 3090, NVIDIA-SMI 535.104.12, CUDA Version: 12.2*

This example focuses on a scenario where y has multiple columns. We focus on the EMB3 and EMB4, as these implementations allow for multidimensional y. We focus on the PyTorch implementation. We assumme a scenario with 2048 observations and 512 predictors (again divided into two groups, each with 256 predictors). We increase the number of outcome variables (columns in y) from 128 to 65536 variables. Timing is reported below with float64 precision.


Model |  128 outcome varibles |  1024 outcome varibles |  16384 outcome varibles |  65536 outcome varibles |  
:-|:-|:-|:-|:-
EMB3 (PyTorch) |  1.27700 s |  2.46048 s |  22.76017 s |  87.40972 s |  
EMB4 (PyTorch) |  1.39476 s |  2.56766 s |  22.99929 s |  88.79498 s |  

This procedure is repeated and shown below with float32 precision.

Model |  128 outcome varibles |  1024 outcome varibles |  16384 outcome varibles |  65536 outcome varibles |  
:-|:-|:-|:-|:-
EMB3 (PyTorch) |  0.32316 s |  0.44985 s |  3.37624 s |  12.71370 s |  
EMB4 (PyTorch) |  0.42105 s |  0.52788 s |  3.56655 s |  13.28350 s |