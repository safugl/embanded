function [W, summary] = embanded(F,y,opts)
% Expectation-Maximization algorithm for estimating regularized regression
% model with banded prior structure.
%
% The function does not standardize the data. Columns in the input data
% must have zero-mean. See below for examples of ways to transform
% data prior to model fitting.
%
% Parameters:
%   F: cell array
%      A cell array of size (1 x J) where J is the number of feature spaces.
%      Each cell must contain numeric arrays. These arrays should have
%      dimensionality (M x D_j) where M is the number of samples (rows) and
%      where D_j is the number of columns of that given feature space
%      (D_j>=1).
%   y: vector
%      A column vector of size (M x 1) where M is the number of samples
%      (rows) or a matrix of size (M x P) where P is the number of outcome
%      variables. The number of samples should be exactly identical to the
%      number of rows in each entry in F. With multiple outcome variables,
%      i.e., P>1, one must set opts.multi_dimensional = True.
%   opts: struct
%      A struct that is used to specify options.
%
% Options related to hyperparameters:
%
%   max_iterations: integer, default = 200
%      Specify the maximum allowed number of iterations.
%   nu: float, default = 1
%      Specify the initial value of the nu hyperparameter which controls
%      observation noise variance.
%   lambdas: row vector, default = ones(1,num_features)
%      Specify the initial values of the lambdas hyperparameters. The length
%      of lambda must be equal to length(F)
%   tau: float, default = 1e-4
%      Specify hyperparameter tau related to the Inverse-Gamma priors imposed
%      on the lambda_j terms.
%   eta: float, default = 1e-4
%      Specify hyperparameter eta related to the Inverse-Gamma priors imposed
%      on the lambda_j terms.
%   phi: float, default = 1e-4
%      Specify hyperparameter phi related to the Inverse-Gamma prior imposed
%      on the nu term.
%   kappa: float, default = 1e-4
%      Specify hyperparameter kappa related to the Inverse-Gamma prior imposed
%      on the nu term.
%   h: row vector, default = nan(1,length(F)) (implying no smoothness)
%      Specify the hyperparameter h related to the covariance parametrization.
%      It is possible to define opts.h = [1, nan], in which case the first
%      Omega_1 term will be parameterized with a Matern kernel, and the
%      Omega_2 term will be a unit matrix. Entries with nan translates to
%      identity matrices associated with the corresponding group of
%      weights.
%
% Optional options:
%
%   show_progress: bool, default=true
%      Whether to show progress and time elapsed.
%   covX: matrix of shape (num_predictors, num_predictors)
%      This is an advanced option that allows the user to pre-compute X'*X.
%      It is important to ensure that covX is computed appropriately with
%      the correct predictors.
%   use_matrix_inversion_lemma: boolean, default = false
%      Specify whether the Woodbury Matrix Identity should be used for
%      computing inv(Sigma).
%   remove_intercept : bool, default=false
%      Whether to remove offsets in X and Y prior to fitting the model. If set
%      to false, the data will not be transformed prior to model fitting.
%      However, in this case, the model will complain if the columns in X or y
%      have not been adequately centered. If set to true, then the offsets
%      will be stored in summary as X_offset and y_offset. These values will
%      be used for the model predictions.
%   multi_dimensional : bool, default=false
%      Whether to make simplifying assumptions to allow for an efficient
%      estimation of weights in cases where y has multiple columns.
%   device : string, default=[]
%      Set device = 'gpu' to allow for GPU computing (requires
%      Parallel Computing Toolbox). When set to empty (default) this will
%      not be used.
%   store_Sigma : bool, default=false
%      Whether or not to store Sigma.
%   compute_score : bool, default=false
%      Whether or not to compute and store the objective function in the
%      summary struct as summary.score.  
%   early_stopping_tol : float, default = []
%      Stop early if increases in the objective function are smaller than
%      this tolerance value. For example, use: early_stopping_tol = 1e-11
%
%
% How to transform data and ensure that each column has zero-mean?
% Simulate some regressors
% rng(1);
% X = randn(100, 10);
%
% Option 1: Using zscore to get a centered, scaled version of X
% B = zscore(X);
%
% Option 2: Using bsxfun to subtract the mean and get a centered version of X
% C = bsxfun(@minus, X, mean(X));
%
% Option 3: Using bsxfun to subtract the mean and divide by the maximum of the standard deviation and a small value
% D = bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), max(std(X), 1e-10));



if nargin < 3 || isempty(opts); opts = struct; end

% Options related to hyperparameter seeds:
if isfield(opts,'nu')==0; opts.nu = 1; end
if isfield(opts,'lambdas')==0; opts.lambdas = ones(1,size(F,2)); end
if isfield(opts,'h')==0; opts.h = []; end
if isfield(opts,'tau')==0; opts.tau = 1e-4; end
if isfield(opts,'eta')==0; opts.eta = 1e-4; end
if isfield(opts,'phi')==0; opts.phi = 1e-4; end
if isfield(opts,'kappa')==0; opts.kappa = 1e-4; end

% Options related to stopping criteria:
if isfield(opts,'max_iterations')==0; opts.max_iterations = 200; end

% Optional options:
if isfield(opts,'covX')==0; opts.covX = []; end
if isfield(opts,'show_progress')==0; opts.show_progress = true; end
if isfield(opts,'remove_intercept')==0; opts.remove_intercept = false; end
if isfield(opts,'multi_dimensional')==0; opts.multi_dimensional = false; end
if isfield(opts,'store_Sigma')==0; opts.store_Sigma = false; end
if isfield(opts,'use_matrix_inversion_lemma')==0; opts.use_matrix_inversion_lemma = false; end
if isfield(opts,'device')==0; opts.device = []; end
if isfield(opts,'compute_score')==0; opts.compute_score = false; end
if isfield(opts,'early_stopping_tol')==0; opts.early_stopping_tol = []; end



summary = struct;
summary.opts_input = opts;


% Combine the features into a matrix
X = cat(2, F{:});



if opts.use_matrix_inversion_lemma == true
    I = eye(size(X,1));
else
    I = eye(size(X,2));
end
    
if strcmp(opts.device,'gpu')
    X = gpuArray(X);
    y = gpuArray(y);
    opts.nu = gpuArray(opts.nu);
    opts.lambdas = gpuArray(opts.lambdas);
    opts.tau = gpuArray(opts.tau);
    opts.eta = gpuArray(opts.eta);
    opts.kappa = gpuArray(opts.kappa);
    opts.phi = gpuArray(opts.phi);
    I = gpuArray(I);
end

if ~isempty(opts.early_stopping_tol)
    opts.compute_score = true;
end

if opts.remove_intercept == true
    assert( isempty(opts.covX) == true, ' The predictors and the response variable have been centered and covX is no longer reliable')
    
    % Remove offset and store them in the summary
    [X, X_offset] = matrix_centering(X);
    [y, y_offset] = matrix_centering(y);
    summary.X_offset = X_offset;
    summary.y_offset = y_offset;
end

% Determine the following:
% - Number of feature spaces (num_features)
% - Number of predictors (num_dim)
% - Number of observations (num_obs)
% - Number of outcome variables (P)
num_features = numel(F);
num_dim = size(X,2);
num_obs = size(X,1);
P = size(y,2);

% Perform assertions to validate the order of rows and columns.
check_input(F,y,X,opts)


% The following is used to index different predictor groups.
columns_group = [];
index = cell(1,num_features);
for f = 1 : num_features
    columns_group = cat(2,columns_group,f * ones(1,size(F{f},2)));
    if strcmp(opts.device,'gpu')
        index{f} = gpuArray(find(columns_group==f));
    else
        index{f} = find(columns_group==f);
    end
end
mat_indexer = one_hot_encoding(columns_group);
D = sum(mat_indexer,1);


% Flag to indicate if Omega terms are used
opts.smoothness_prior = false;

% Proceed if at least one element in h is not NaN
if ~isempty(opts.h) & ~all(isnan(opts.h))
    % Estimate Omega terms and set the smoothness prior flag to true.
    [Omega_A, Omega_A_inv,~,Omega_inv] = estomega(opts.h,F,opts.device);
    opts.smoothness_prior = true;
    
    if opts.compute_score
        logdetOmega = matrix_cholesky_logdet(Omega_A);
    end
else
    opts.h = nan(1,num_features);
    logdetOmega = 0;
end



% Initialization
nu = opts.nu;
lambdas = opts.lambdas;
covXy = X'*y;
score = nan(opts.max_iterations,1);

% Check if covX is provided; if yes, there's no need to compute X'*X again.
% Additionally, if the intention is to use the Woodbury matrix inversion
% formula, then computing covX is unnecessary.
if isempty(opts.covX)
    if  ~opts.use_matrix_inversion_lemma
        covX = X' * X;
    else
        covX = [];
    end
else
    covX = opts.covX;
end

% Start timer to show progress if enabled
if opts.show_progress==true
    tic;
end




% Do the EM iterations
for iteration = 1 : opts.max_iterations
    
    
    % Store a summary of lambda and nu values for each iteration
    summary.lambda(iteration,:) = lambdas;
    summary.nu(iteration,1) = nu;
    
    % Create a vector that contains the diagonal elements, lambda_j.
    lambdas_diag = lambdas * mat_indexer';
    
 
    
    % Compute Sigma
    if ~opts.use_matrix_inversion_lemma        
        if ~opts.smoothness_prior
            % Case without smoothness prior
            L = diag(1./lambdas_diag);
        else
            % Case with smoothness prior
            L = (1./lambdas_diag) .* Omega_A_inv;
        end
        [Sigma, logdetSigma] = ...
            matrix_cholesky_solve(1/nu * covX + L, I, opts.compute_score);
    else
        % Compute Sigma using the Woodbury matrix inversion lemma.
        if ~opts.smoothness_prior
            % Case without smoothness prior
            L = diag(lambdas_diag);
        else
            % Case with smoothness prior
            L = (lambdas_diag) .* Omega_A;
        end
        Sigma = L - L*X'* matrix_cholesky_solve(nu * I + X*L*X', X*L);
    

        if opts.compute_score 
            logdetSigma = matrix_cholesky_logdet(Sigma);
        else
            logdetSigma = [];
        end
    end
    
    % Estimate weights
    W = 1 / nu * (Sigma * covXy);
    
    % Make point-predictions
    prediction = X * W;
     
    % Estimate residual sum of squares and store residuals.
    if opts.multi_dimensional
        residuals = reshape(y,num_obs*P,1) - reshape(prediction,num_obs*P,1);
        rss = sum(residuals.^2);
    else
        residuals = y - prediction;
        rss = sum(residuals.^2);
    end
    
    
    if opts.compute_score
        % Estimate the objective function. Break it down into terms for
        % readability.
        term_mlik = -1/(2 * nu) * reshape(y,1,num_obs*P) * residuals ...
            -P/2 * (...
            -logdetSigma + logdetOmega + sum(log(lambdas_diag)) ...
            +num_obs*log(nu) ...
            );
        term_lambda  = -sum(log(lambdas.^(1+opts.eta))) ...
            -sum(opts.tau./lambdas);
        term_nu  = -log(nu.^(1+opts.phi)) - opts.kappa / nu;
        
        % Store objective at each iteration
        score(iteration) = term_mlik + term_lambda + term_nu;
    end

    if ~isempty(opts.early_stopping_tol)

        if iteration > 1
            score_diff = score(iteration) - score(iteration-1);
            
            % Stop early if the difference between current and previous
            % estimate is smaller than opts.early_stopping_tol
            if score_diff < opts.early_stopping_tol
                break
            end
        end

    end

    
    if iteration <= opts.max_iterations - 1
        
        
        % The Supplementary Material outlines a model that enables sharing
        % of covariance terms across outcome variables. When the 
        % opts.multi_dimensional parameter is set to false, the 
        % implementation utilizes vectorized code.
      
        if opts.multi_dimensional == false
            
            % This implementation is suitable when y is a matrix of size
            % [M x 1]. It avoids nested for loops and utilizes vectorized 
            % code, which may improve compute time, especially in scenarios
            % with many predictor groups and relatively few predictors in
            % each group.
            
            if (opts.smoothness_prior == false)
                % Case without smoothness prior
                lambdas =  (...
                    (matrix_blockdiag_rotation(W, mat_indexer) + ...
                    matrix_block_trace(Sigma, mat_indexer) * P + ...
                    2 * opts.tau) ./ ( P * D + 2 * opts.eta + 2 ) ...
                    );
            else
                % Case with smoothness prior
                lambdas =  (...
                    (matrix_blockdiag_rotation(W, mat_indexer, Omega_A_inv) + ...
                    matrix_block_trace(Sigma, mat_indexer, Omega_A_inv) * P + ...
                    2 * opts.tau) ./ ( P * D + 2 * opts.eta + 2 ) ...
                    );
            end
            
        else
            
            % This implementation is applicable for any P > 0, where y is
            % a matrix of size [M x P]. The implementation involves nested 
            % for loops. It can be efficient when the number of predictor 
            % groups is low, and each group has many predictors.
            
            for f = 1 : num_features
                
                % Find the columns matching the feature space f
                columns_f = index{f};
                D_f = D(f);
                
                % Extract blocks in Sigma and W
                mu_f = W(columns_f,:);
                Sigma_ff = Sigma(columns_f,columns_f);
                
                % Update lambda_f. In this case, mu_f is a real matrix of
                % size [D X P], where D is the number of predictors
                % associated with outcome variable p = 1, ..., P. We can
                % simplify this computation and avoid iterating over all P
                % variables.
                
                if  ~isnan( opts.h(f) )
                    % Case with smoothness prior
                    if size(mu_f,2) > size(mu_f,1)
                        E = trace(mu_f * mu_f' * Omega_inv{f}) + matrix_trace_of_product(Omega_inv{f}, Sigma_ff) * P;
                    else
                        E = trace(mu_f' * Omega_inv{f} * mu_f) + matrix_trace_of_product(Omega_inv{f}, Sigma_ff) * P;
                    end
                else
                    % Case without smoothness prior
                    if size(mu_f,2) > size(mu_f,1)
                        E = trace(mu_f * mu_f') + trace(Sigma_ff) * P;
                    else
                        E = trace(mu_f' * mu_f) + trace(Sigma_ff) * P;
                    end
                end
                
                % Update each lambda
                lambdas(f) = (E + 2 * opts.tau) ./ ( P * D_f + 2 * opts.eta + 2 );
            end
        end
        
        
        
        % Update nu.
        if  opts.use_matrix_inversion_lemma == false
            nu = ( rss + ...
                P * matrix_trace_of_product(covX, Sigma) + 2 * opts.kappa) / ( P * num_obs + 2 + 2 * opts.phi );
        elseif opts.use_matrix_inversion_lemma == true
            nu = ( rss + ...
                P * trace(X* Sigma*X') + 2 * opts.kappa) / ( P * num_obs + 2 + 2 * opts.phi );
        end
        
        
        if opts.show_progress
            % Display progress
            fprintf('\n At iteration %i of %i iterations, time elapsed: %0.2f',iteration, opts.max_iterations, toc)
        end
        
        
    end
    
    
end

if opts.store_Sigma
    summary.Sigma = Sigma;
end

if opts.compute_score
    summary.score = score;
end

end


function [Omega_A, Omega_A_inv,Omega,Omega_i] = estomega(h,F,device)
% Estimate an Omega matrix that controls covariance parameterization
%
% Usage:
%
% Example 1: impose smoothness constraints on subset 1
% [Omega_A, Omega_A_inv] = estomega([0.5, NaN], F)
%
% Example 2: impose smoothness constraints on both subsets
% [Omega_A, Omega_A_inv] = estomega([0.5, 0.5], F)
%
% Parameters:
% - h: A row vector controlling h associated with each feature
% - F: A cell array of size (1 x J) where J is the number of feature spaces.
%      Each cell must contain numeric arrays. These arrays should have
%      dimensionality (M x D_j) where M is the number of samples (rows) and
%      D_j is the number of columns of that given feature space (D_j >= 1).

% How many features are considered?
num_features = size(F, 2);

% Prepare cell-arrays
Omega = cell(1,num_features);
Omega_i = cell(1,num_features);

% Iterates over all feature spaces
for f = 1 : num_features
    
    % The number of columns in F{f}
    D_j = size(F{f},2);
    
    
    % Create the Omega matrix
    if ~isnan(h(f))
        
        % If opts.h(f) is a scalar then define Omega as follows:
        if strcmp(device,'gpu')
            Omega{f} = gpuArray(matern_type_kernel(D_j, h(f)));
            Omega_i{f} = matrix_cholesky_solve(Omega{f}, eye(D_j,'gpuArray'));
        else
            Omega{f} = matern_type_kernel(D_j, h(f));
            Omega_i{f} = matrix_cholesky_solve(Omega{f}, eye(D_j));
        end
        
    else
        % Otherwise, if it is specified as NaN then assume that it
        % should be an identity matrix
        if strcmp(device,'gpu')
            Omega{f} = eye(D_j,'gpuArray');
            Omega_i{f} = eye(D_j,'gpuArray');
        else
            Omega{f} = eye(D_j);
            Omega_i{f} = eye(D_j);
        end
    end
    
    
end

% Prepare a matrix over entire Lambda
Omega_A = blkdiag(Omega{:});


% Compute the inverse of this matrix
Omega_A_inv = blkdiag(Omega_i{:});

end


function O = matrix_blockdiag_rotation(A,mat_indexer,B)
% Take a vector A of size (D x 1) and a block diagonal matrix, B, of size
% (D x D) and estimate [A_1'*B_11*A_1, ..., A_f'*B_ff*A_f, ...], where f
% indicates block index. If B is empty, then it is assumed to be the
% identity matrix. This only works since blocks in B coincides with indexes
% in mat_indexer. The below examples illustrates this behavior.
%
%     Examples
%     --------
%     rng(1)
%     mat_indexer = one_hot_encoding([1, 1, 1, 2, 2]);
%     A = randn(5,1);
%     B11 = [1:3]'*[1:3];
%     B22 = sin([1:2])'*sin([1:2]);
%
%     B = [B11, zeros(3,2); ...
%         zeros(2,3), B22];
%
%     estimate = matrix_blockdiag_rotation(A,mat_indexer,B);
%     compare = [A(1:3)'*B11*A(1:3), A(4:5)'*B22*A(4:5)];
%     disp([estimate;  compare])
%
%     estimate = matrix_blockdiag_rotation(A,mat_indexer);
%     compare = [A(1:3)'*A(1:3), A(4:5)'*A(4:5)];
%     disp([estimate;  compare])


if nargin < 3 || isempty(B); B = []; end

assert(ismatrix(A), 'A has to be a vector of size (M X 1)')
assert(isreal(A), 'Please check A')
assert(size(A,2)==1, 'A has to be a vector of size (M X 1)')

if ~isempty(B)
    assert(ismatrix(B), 'B has to be a block diagonal matrix of size (M X M)')
    assert(isreal(B), 'Please check A')
    
    O = A'*B*(A.*mat_indexer);
else
    O = sum((A.^2)'*mat_indexer,1);
end

end


function O = matrix_block_trace(A,mat_indexer,B)
% Take a matrix A and a block diagonal matrix, B, and estimates
% [trace(A_1*B_11), ..., trace(A_f*B_ff), ...], where f indicates block
% index. If B is empty, then it is assumed to be the identity matrix. Both
% A and B will have size (D x D). This only works since blocks in B
% coincides with indexes in mat_indexer. The below examples illustrates
% this behavior.
%
%     Examples
%     --------
%     rng(1)
%     mat_indexer = one_hot_encoding([1, 1, 1, 2, 2]);
%     A = randn(5,1); A = A*A';
%     B11 = [1:3]'*[1:3];
%     B22 = sin([1:2])'*sin([1:2]);
%
%     B = [B11, zeros(3,2); ...
%         zeros(2,3), B22];
%
%     estimate = matrix_block_trace(A,mat_indexer,B);
%     compare = [trace(A(1:3,1:3)'*B11), trace(A(4:5,4:5)'*B22)];
%     disp([estimate;  compare])
%
%
%     estimate = matrix_block_trace(A,mat_indexer);
%     compare = [trace(A(1:3,1:3)), trace(A(4:5,4:5))];
%     disp([estimate;  compare])

if nargin < 3 || isempty(B); B = []; end

assert(ismatrix(A), 'A has to be a vector of size (M X 1)')
assert(isreal(A), 'Please check A')

if ~isempty(B)
    assert(ismatrix(B), 'B has to be a block diagonal matrix of size (M X M)')
    assert(isreal(B), 'Please check A')
    
    O = diag(B*A)'*mat_indexer;
else
    O = diag(A)'*mat_indexer;
    
end
end
