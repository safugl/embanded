# Runtime  
This page provides examples of runtime tests. For comparison, runtime is monitored for other related estimators, as outlined in the table below. It's important to approach the runtime estimates with caution. Each estimator comes with various hyperparameters that can significantly influence runtime, and the choices below are somewhat arbitrary. The reported values are crude point timing measures for a few specific scenarios. We specifically emphasize that GroupRidgeCV is highly efficient in fitting a large number of targets and that it can utilize GPU acceleration. However, we consider it in settings with only one target variable and where it has to run on CPU hardware.


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
GroupRidgeCV1 | Group ridge regression with cross-validation  |  groups="input", random_state=0, fit_intercept=True, solver_params = dict(progress_bar=False), Y_in_cpu=True, force_cpu=True | [himalaya.ridge.GroupRidgeCV](https://gallantlab.org/himalaya/_generated/himalaya.ridge.GroupRidgeCV.html#himalaya.ridge.GroupRidgeCV)
GroupRidgeCV2 | Group ridge regression with cross-validation  |  groups="input", random_state=0, fit_intercept=True, solver_params = dict(progress_bar=False, alphas=np.logspace(-10, 10, 21), n_iter=1000), Y_in_cpu=True, force_cpu=True | [himalaya.ridge.GroupRidgeCV](https://gallantlab.org/himalaya/_generated/himalaya.ridge.GroupRidgeCV.html#himalaya.ridge.GroupRidgeCV)

# Example 01
*Tested on a MacBookPro15,1: 2.2GHz 6-core Intel Core i7-8750H, Turbo Boost up to 4.1GHz, with 9MB shared L3 cache, 16GB of 2400MHz DDR4 onboard memory*

This example utilizes synthetic data generated through a simulation developed by [scikit-learn](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html#sphx-glr-auto-examples-linear-model-plot-ard-py). In brief, it involves a design matrix with 100 predictors, a 1-dimensional target variable 'y,' and a dataset of 100 samples. The target weights are assumed to be sparse. We assume that the number of groups equals the number of predictors (i.e., 100 groups) for the EM-banded estimators and the GroupRidgeCV estimators. The table below presents both the runtime and the mean square error (MSE) between the estimated weights and the target weights. The procedure is repeated for the EM-banded implementation in Matlab R2018b. Note that just-in-time (JIT) acceleration can improve runtime in some circumstances and be aware of MATLAB performance release notes (see e.g., [here](https://www.mathworks.com/products/matlab/performance.html)). The results are shown below.

Model | Runtime | MSE 
:-|:---|:----
EMB1 | 0.09367 s | 0.06395
EMB2 | 0.01921 s | 0.06395
EMB3 | 0.58377 s | 0.06395
EMB4 | 0.10114 s | 0.06395
RidgeCV1 | 0.00408 s | 10.57044
RidgeCV2 | 0.98216 s | 8.20769
RidgeCV3 | 0.01543 s | 10.89148
ARD | 0.84340 s | 0.86433
BRR | 0.01092 s | 14.15242
OLS | 0.00356 s | 53.53597
LassoCV | 0.07387 s | 0.24920
GroupRidgeCV1 | 1.49936 s | 31.47160
GroupRidgeCV2 | 21.89188 s | 2.43647
EMB1 (Matlab) | 0.06205 s | 0.06395
EMB2 (Matlab) | 0.01239 s | 0.06395
EMB3 (Matlab) | 0.08268 s | 0.06395
EMB4 (Matlab) | 0.01578 s | 0.06395


# Example 02
*Tested on a MacBookPro15,1: 2.2GHz 6-core Intel Core i7-8750H, Turbo Boost up to 4.1GHz, with 9MB shared L3 cache, 16GB of 2400MHz DDR4 onboard memory*

In this example, we explore a scenario with a moderate number of predictors (128 predictors) and an increasing number of observations. The simulations involve two predictor groups, F1 and F2, each containing 64 predictors. The target variable y is assumed to be a mixed version of F1 plus additive Gaussian noise. We monitor the runtime for simulations with varying numbers of observations, ranging from 128 samples to 65536 samples.  We assume two groups of predictors, F1 and F2, for the EM-banded estimators and the GroupRidgeCV estimators. Notice that the EM-banded estimators with early stopping (EMB2 and EMB4) requires computation of log-score unlike EMB1 and EMB3 which also affects runtime. We do not fit the GroupRidgeCV estimators in all scenarios.

Model |  128 samples |  1024 samples |  16384 samples |  65536 samples |  
:-|:-|:-|:-|:-
EMB1 |  0.07966 s |  0.12189 s |  0.44254 s |  1.72565 s |  
EMB2 |  0.09141 s |  0.11670 s |  0.10665 s |  0.32171 s |  
EMB3 |  0.09260 s |  0.13433 s |  0.45764 s |  1.68051 s |  
EMB4 |  0.10652 s |  0.13402 s |  0.10990 s |  0.32390 s |  
RidgeCV1 |  0.00430 s |  0.01596 s |  0.22250 s |  1.42302 s |  
RidgeCV2 |  0.74528 s |  1.37204 s |  11.71373 s |  52.72079 s |  
RidgeCV3 |  0.01277 s |  0.06396 s |  1.33959 s |  7.29034 s |  
ARD |  0.15756 s |  0.06202 s |  0.23450 s |  0.87176 s |  
BRR |  0.00874 s |  0.01340 s |  0.21180 s |  1.59875 s |  
OLS |  0.00389 s |  0.00888 s |  0.11850 s |  0.95349 s |  
LassoCV |  0.76172 s |  0.13668 s |  0.47746 s |  2.26283 s |  
GroupRidgeCV1 |  1.64815 s |  5.84258 s |  235.31970 s |  - |  
GroupRidgeCV2 |  28.46093 s |  - |  - |  - |  

# Example 03:
*Tested running on a cluster system with multiple users; NVIDIA GeForce RTX 3090, NVIDIA-SMI 535.104.12, CUDA Version: 12.2, AMD EPYC 7452 32-Core Processor, 2.35 GHz;  memory: 503GB*

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
*Tested running on a cluster system with multiple users; NVIDIA GeForce RTX 3090, NVIDIA-SMI 535.104.12, CUDA Version: 12.2, AMD EPYC 7452 32-Core Processor, 2.35 GHz;  memory: 503GB*

This example focuses on a scenario where y has multiple columns. We focus on the EMB3 and EMB4, as these implementations allow for multidimensional y. We focus on the PyTorch implementation. We assume a scenario with 2048 observations and 512 predictors (again divided into two groups, each with 256 predictors). We increase the number of outcome variables (columns in y) from 128 to 65536 variables. Timing is reported below with float64 precision.


Model |  128 outcome varibles |  1024 outcome varibles |  16384 outcome varibles |  65536 outcome varibles |  
:-|:-|:-|:-|:-
EMB3 (PyTorch) |  1.27700 s |  2.46048 s |  22.76017 s |  87.40972 s |  
EMB4 (PyTorch) |  1.39476 s |  2.56766 s |  22.99929 s |  88.79498 s |  

This procedure is repeated and shown below with float32 precision.

Model |  128 outcome varibles |  1024 outcome varibles |  16384 outcome varibles |  65536 outcome varibles |  
:-|:-|:-|:-|:-
EMB3 (PyTorch) |  0.32316 s |  0.44985 s |  3.37624 s |  12.71370 s |  
EMB4 (PyTorch) |  0.42105 s |  0.52788 s |  3.56655 s |  13.28350 s |