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
%      (rows). The number of samples should be exactly identical to the
%      number of rows in each entry in F.
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
if isfield(opts,'use_matrix_inversion_lemma')==0; opts.use_matrix_inversion_lemma = false; end

% Optional options:
if isfield(opts,'covX')==0; opts.covX = []; end
if isfield(opts,'show_progress')==0; opts.show_progress = true; end
if isfield(opts,'remove_intercept')==0; opts.remove_intercept = []; end
if isfield(opts,'multi_dimensional')==0; opts.multi_dimensional = false; end


% Combine the features into a matrix
X = cat(2, F{:});

if opts.remove_intercept == true
    X_offset = mean(X,1);
    y_offset = mean(y,1);
    
    assert( isempty(opts.covX) == true, ' The predictors and the response variable have been centered and covX is no longer reliable')
    
    X = bsxfun(@minus,X,X_offset);
    y = bsxfun(@minus,y,y_offset);
    
end

% Determine the number of feature spaces
num_features = numel(F);

% Determine the number of predictors
num_dim = size(X,2);

% Determine the total number of observations
num_obs = size(X,1);

% Perform assertions to validate the order of rows and columns
check_input(X,y, opts.multi_dimensional)

% Check if lambda is properly specified
assert(numel(opts.lambdas)==num_features, 'The opts.lambdas vector does not have the expected number of dimensions')
assert(~any(isnan(opts.lambdas)), 'The opts.lambdas vector should not contain NaNs')


% Prepare a row vector that indexes each group
columns_group = [];

% Prepare a cell array for indexing feature spaces
index = cell(1,num_features);

% Prepare the user-initialized lambda values.
lambda_diag = cell(1,num_features);

for f = 1 : num_features
    columns_group = cat(2,columns_group,f * ones(1,size(F{f},2)));
    index{f} = find(columns_group==f);    
    lambda_diag{f} = opts.lambdas(f) * ones(1,size(F{f},2));
end

% Prepare user-initialized nu variable
nu = opts.nu;

% Flag to indicate if Omega terms are used
opts.smoothness_prior = false;

if ~isempty(opts.h)
    assert(length(opts.h)==num_features, 'When specifying h terms, the row-vector h should have as many elements as there are feature spaces')
    
    % Proceed if at least one element in h is not NaN
    if ~all(isnan(opts.h))
        [Omega_A, Omega_A_inv,~,Omega_inv] = estomega(opts.h,F);
        
        % Set the smoothness prior flag to true
        opts.smoothness_prior = true;
        
    end
end

% Pre-compute X'*y
covXy = X'*y;

if isempty(opts.covX)
    if  opts.use_matrix_inversion_lemma == false
        % Compute covX if not provided
        covX = X' * X;
    else
        covX = [];
    end
else
    % If covX is provided, assume it has already been computed
    covX = opts.covX;
end


% Start timer to show progress if enabled
if opts.show_progress==true
    tic;
end




summary = struct;
summary.opts_input = opts;
if opts.remove_intercept == true
    summary.X_offset = X_offset;
    summary.y_offset = y_offset;
end

% Define the number of outcome variables
summary.P = size(y,2);

if opts.multi_dimensional == false
    assert(summary.P==1);
end
% Do the EM iterations
for iteration = 1 : opts.max_iterations
    
    
    % Store a summary of lambda and nu values for each iteration
    for f = 1 : num_features
        summary.lambda(iteration,f) = lambda_diag{f}(1);
        summary.nu(iteration,1) = nu;
    end
    
    
    
    % Create a row vector based on lambda_diag
    Lambda_diag = cat(2,lambda_diag{:});
           
    
    if opts.smoothness_prior == true        
        if  opts.use_matrix_inversion_lemma == false
            L_inv = (1./Lambda_diag) .* Omega_A_inv;
            H = chol( 1/nu * covX + L_inv,'lower');
            Sigma = H'\(H\(eye(size(H,2))));
        elseif  opts.use_matrix_inversion_lemma == true
            L = (Lambda_diag) .* Omega_A;
            H = chol(nu * eye(num_obs) + X*L*X','lower');
            Sigma = L - L*X'*(H'\(H\(X*L)));
        end
    elseif opts.smoothness_prior == false
        if  opts.use_matrix_inversion_lemma == false
            L_inv = diag(1./Lambda_diag);
            H = chol( 1/nu * covX + L_inv,'lower');
            Sigma = H'\(H\(eye(size(H,2))));
        elseif  opts.use_matrix_inversion_lemma == true
            L = Lambda_diag';
            P = (L).*X';
            H = chol(nu * eye(num_obs) + X*P,'lower');
            Sigma = diag(L) - P*(H'\(H\(P'))); 
        end
    end
    
    % Estimate weights
    W = 1 / nu * (Sigma * covXy);
    
    if iteration <= opts.max_iterations - 1
        % Iterate over all feature spaces and update
        for f = 1 : num_features
            
            % Find the columns matching the feature space <f>
            columns_f = index{f};
            D = numel(columns_f);
            
            % Pick the relevant terms in W
            mu_f = W(columns_f,:);
            Sigma_ff = Sigma(columns_f,columns_f);
            
            E = [];
            
            if opts.multi_dimensional == false
                % Note short-circuit &&, i.e., if (opts.smoothness_prior==false), then do not attempt to
                % evalute ( ~isnan( opts.h(f) )
                if ( opts.smoothness_prior == true) && ( ~isnan( opts.h(f) ) )
                    E = mu_f' * Omega_inv{f} * mu_f + trace(Omega_inv{f} * Sigma_ff);
                else
                    E = mu_f' * mu_f + trace(Sigma_ff);
                end

                
            elseif opts.multi_dimensional == true
                
                % Initialize an empty scalar
                E = 0;
                             
                if ( opts.smoothness_prior == true) && ( ~isnan( opts.h(f) ) )
                    E = E + trace(mu_f * mu_f' * Omega_inv{f}) + trace(Omega_inv{f} * Sigma_ff) * summary.P;
                else
                    E = E + trace(mu_f * mu_f') + trace(Sigma_ff) * summary.P;
                end
                
                % The following is an alternative version where we iterate
                % over all P outcome variables. We include this code block
                % to spell things out:
                %
                % for p = 1 : summary.P
                %    if ( opts.smoothness_prior == true) && ( ~isnan( opts.h(f) ) )
                %        E = E + ( mu_f(:,p)' * Omega_inv{f} * mu_f(:,p) + trace(Omega_inv{f} * Sigma_ff));
                %    else
                %        E = E + ( mu_f(:,p)' * mu_f(:,p) + trace(Sigma_ff));
                %    end
                % end
            end
            
            lambda_f = (E + 2 * opts.tau) / ( summary.P * D + 2 * opts.eta + 2 );

            
            % Update lambda
            lambda_diag{f} = lambda_f * ones( 1 , size(F{f},2) );
        end
        
        % Make point-predictions
        prediction = X * W;
         
        
        if opts.multi_dimensional == true
            % Flatten the Y matrix
            target = reshape(y,num_obs*summary.P,1);
            
            % Flatten the predictions matrix
            prediction = reshape(prediction,num_obs*summary.P,1);
           
        else
            % Target is a column vector
            target = y;
            assert(summary.P == 1, 'Spell out that P is one in this case')
        end
        
        

        if  opts.use_matrix_inversion_lemma == false
            nu = ( ( target - prediction )' * ( target - prediction ) + summary.P * trace(covX * Sigma) + 2 * opts.kappa) / ( summary.P * num_obs + 2 + 2 * opts.phi );
        elseif opts.use_matrix_inversion_lemma == true
            nu = ( (target - prediction)' * ( target - prediction ) + summary.P * trace(X* Sigma*X') + 2 * opts.kappa) / ( summary.P * num_obs + 2 + 2 * opts.phi );
        end
        
        
        if opts.show_progress
            % Display progress
            fprintf('\n At iteration %i, time elapsed: %0.2f',iteration,toc)
        end
      
        
    end
    
    
end

summary.Sigma = Sigma;


end


function [Omega_A, Omega_A_inv,Omega,Omega_i] = estomega(h,F)
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
    
    % Define a grid [0,1,2,3,...,D_j-1]
    x_grid  = [0 : D_j-1] ;
    
    % Create the Omega matrix
    if ~isnan(h(f))
        
        % If opts.h(f) is a scalar then define Omega as follows:
        Omega{f} = (1+ sqrt(3) * abs(x_grid' - x_grid)/h(f)).*exp(-sqrt(3) * abs(x_grid'-x_grid)/h(f));
        
        % Compute the inverse
        OH = chol( Omega{f},'lower');
        Omega_i{f} = OH'\(OH\(eye(size(OH,2))));
        
    else
        % Otherwise, if it is specified as NaN then assume that it
        % should be an identity matrix
        Omega{f} = eye(size(F{f},2));
        Omega_i{f} = eye(size(F{f},2));
        
    end
    
    
end

% Prepare a matrix over entire Lambda
Omega_A = blkdiag(Omega{:});


% Compute the inverse of this matrix
Omega_A_inv = blkdiag(Omega_i{:});

end

function check_input(X,y,multi_dimensional)

% Do a few assertations to check that things are in order in terms of rows
% and columns.
assert(size(X,1)==size(y,1),'The number of observations in F and y are not matching')
assert(size(y,1)>1,'y should contain more than one observation')

if multi_dimensional == false
assert(size(y,2)==1,'y should be a column vector')
end
% Check also if the data has been (approximately) mean-centered
assert(all(abs(mean(X))<10e-8), 'It appears that X has columns that have not been centered. Please do so.')
assert(all(abs(mean(y))<10e-8), 'It appears that y has not been centered. Please do so.')

end