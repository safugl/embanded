clear; clc; close all; rng('default')
assert(~isempty(strfind(pwd, 'embanded/examples/matlab')), 'Please change folder to the examples directory')
run(strrep(pwd,'examples/matlab','setup.m'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate envelope and EEG data and fit a 'decoding-type' model.
% The EEG data contains noise in all channels, and one channel
% additionally contains the envelope. The decoding
% modeling approach involves time-lagging the EEG data and creating a
% design matrix that contains all these time-lagged versions of each
% channel. For this example, we fit two EM-banded type models :

% Model 1)
% This model declares that a 'band' contains time-lagged versions of
% data from each electrode. The features are stored in a cell array,
% where each entry contains matrices of size [num_obs x num_lags]. The cell 
% array will have num_channels columns.


% Model 2)
% This model declares that each electrode and each lag is assigned a lambda
% hyperparameter. The predictors are prepared as a cell array, where each 
% cell contains a column in the design matrix.


% The models are compared with Ridge estimators.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Generate some training data. 
fs = 40; % Sampling rate
lags = -5:5; % Lags to consider
num_obs = 128*fs; % Number of observations
num_channels = 64; % Number of channels in Y
num_lags = length(lags); % Number of lags

% Create a mask to discard the first and last chunk of training and test data
mask = logical(ones(num_obs,1));
mask(1:fs*5) = 0;
mask(end-fs*5+1:end) = 0;


% Simulate a target "envelope" 
env = max(randn(num_obs,1),0);  
env_test = max(randn(num_obs,1),0);


% Simulate EEG data with noise
eeg  = zscore(randn(num_obs,num_channels));
eeg_test = zscore(randn(num_obs,num_channels));

% Add the envelope to one channel 
eeg(:,num_channels/2) = eeg(:,num_channels/2) + env;
eeg_test(:,num_channels/2) = eeg_test(:,num_channels/2) + env_test;

% Create a time-lagged version of the EEG data and apply the mask
F = timelag(eeg,lags,mask);
F_test = timelag(eeg_test,lags,mask);

% Discard end points for the target variable
y = env(mask==1);
y_test = env_test(mask==1);

%% Model 1)
% Estimate the weights with EM-banded model. Remove offsets in F and y.

opts = struct;
opts.max_iterations = 200;
opts.show_progress = true;
opts.kappa = 1e-4;
opts.phi = 1e-4;
opts.tau = 1e-4;
opts.eta = 1e-4;
opts.remove_intercept = true;
opts.early_stopping_tol = 1e-12;
[W_model1,summary_model1] = embanded(F,y,opts);

% Check mean values are stored correctly
assert(all(mean(y,1)==summary_model1.y_offset),'The function stores the mean of X and y')
assert(all(mean(cat(2,F{:}),1)==summary_model1.X_offset),'The function stores the mean of X and y')

%% Model 2)

% Estimate EM-banded model with separate lambdas for each time-lag and electrode

F_model2 = mat2cell(cat(2,F{:}),sum(mask),ones(num_lags*num_channels,1));


% Fit the model, and re-use the opts struct
opts = struct;
opts.max_iterations = 200;
opts.show_progress = true;
opts.kappa = 1e-4;
opts.phi = 1e-4;
opts.tau = 1e-4;
opts.eta = 1e-4;
opts.remove_intercept = true;
opts.early_stopping_tol = 1e-12;
[W_model2,summary_model2] = embanded(F_model2,y,opts);

% Prune the weights for illustration
W_model2(summary_model2.lambda(end,:)<1e-4)=0;

%% Prepare the test data

X_test = cat(2,F_test{1,:});

% Subtract the offset from the predictors and the target variable
X_test = bsxfun(@minus,X_test,summary_model1.X_offset);
y_test = bsxfun(@minus,y_test,summary_model1.y_offset);


%% Plot the results

% Compute Pearson's correlation coefficient
correlation_model1 = corrcoef(y_test,X_test*W_model1);
correlation_model2 = corrcoef(y_test,X_test*W_model2);

correlation_model1 = correlation_model1(1,2);
correlation_model2 = correlation_model2(1,2);

% Reshape for visualization purposes
W_model1 = reshape(W_model1,length(lags),num_channels);
W_model2 = reshape(W_model2,length(lags),num_channels);


zlims = [-0.9,0.9] * max(abs(W_model1(:)));

% Show weights
subplot(2,6,1)
imagesc(1:num_channels,lags,W_model1,zlims)
colorbar, axis square, title('Model 1'), xlabel('Electrode'), ylabel('Lags')

% Show weights
subplot(2,6,2)
imagesc(1:num_channels,lags,W_model2,zlims)
colorbar, axis square, title('Model 2'), xlabel('Electrode'), ylabel('Lags')
colormap hsv

%% Compare with Ridge estimators

X_train = cat(2,F{1,:});

assert(all(mean(X_train)==summary_model1.X_offset), 'Spell out how offsets were removed from X and y with the ot')
assert(all(mean(y)==summary_model1.y_offset), 'Spell out how offsets were from X and y with the other function')
assert(all(mean(X_train)==summary_model2.X_offset), 'Spell out how offsets were from X and y with the other function')

% Subtract mean from training data
X_train = bsxfun(@minus,X_train,mean(X_train));
y_train = bsxfun(@minus,y,mean(y));



alphas = 10.^[-6:0.5:8];
correlation_ridge = [];
count = 0;
for a = 1 : length(alphas)
    
    fprintf('\n Fitting Ridge estimator at iteration %i',a)
    W_ridge = (X_train'*X_train + alphas(a)*eye(size(X_train,2)))\(X_train'*y_train);
    r_tmp = corrcoef(y_test,X_test*W_ridge);
    correlation_ridge(a) = r_tmp(1,2);
    
    
    if ismember(alphas(a),[1e-0,1e-1,1e2,1e6])
        W_rigde = reshape(W_ridge,length(lags),num_channels);
        
        subplot(2,6,count+3)
        imagesc(1:num_channels,lags,W_rigde,[-0.9,0.9]*max(abs(W_ridge(:))))
        colorbar, axis square, title(sprintf('Ridge (alpha=%0.2e)',alphas(a))), xlabel('Electrode'), ylabel('Lags')
        count = count + 1;
    end
end


subplot(2,6,7:12)
semilogx(alphas,correlation_ridge,'-r')
hold on
semilogx(alphas,ones(1,length(alphas))*correlation_model1,'-k')
semilogx(alphas,ones(1,length(alphas))*correlation_model2,'--b')
ylim([0.25,0.6])
xlabel('Ridge alpha')
ylabel('Correlation')
legend('Ridge','EM-banded (Model1)','EM-banded (Model2)')



fig = get(groot,'CurrentFigure');
fig.Position = [100 100 2000 1500];
fig.GraphicsSmoothing='off';
