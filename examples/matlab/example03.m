clear; clc; close all; rng('default')
assert(~isempty(strfind(pwd, 'embanded/examples/matlab')), 'Please change folder to the examples directory')
run(strrep(pwd,'examples/matlab','setup.m'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate data.

% Generate some training data
num_obs = 1024;
num_dim = 512;

% Define covariane matrix of regressors
C = exp(-0.3*([0:num_dim-1]-[0:num_dim-1]').^2);

% Simulate predictors
X = zscore(mvnrnd(zeros(1,num_dim),C,num_obs));

% Simulate weights (most weights will be equal to zero)
W = randn(num_dim,1).*(rand(num_dim,1)>0.95);

% Simulate noise term
N = zscore(mvnrnd(0,1,num_obs));

[A,B] = scaledata(X,W,N,0);

W = W*A;
N = N*B;

% Simulate the response
y = X*W + N;

% Display stuff for the interested
fprintf('\n Standard deviation of y: %0.2f', std(y))
fprintf('\n Target-to-noise ratio: %0.2f dB',10*log10(mean((X*W).^2)/mean(N.^2)))


% Split the data into predictor groups (F). This example assigns distinct 
% hyperparameters to each predictor.
F = cell(1,num_dim);
for f = 1 : num_dim
    F{1,f} = X(:,f);
end

% Sweep through the following parameter values
param_values =  [1e-4,1e-3,1e-2,1e-1];
W_estimated = cell(1,4);
W_ridge = cell(1,4);

for pp = 1 : 4
    param_val = param_values(pp);
    
    % Estimate with EM-banded
    opts = struct;
    opts.max_iterations = 200;
    opts.show_progress = false;
    opts.kappa = param_val;
    opts.phi = param_val;
    opts.tau = param_val;
    opts.eta = param_val;
    [W_estimated{pp},summary] = embanded(F,y,opts);
    
   
    
    % Plot the estimated weights
    subplot(4,5,[pp,pp+5])
    plot(W_estimated{pp},'-k','LineWidth',1)
    xlim([1,num_dim]);
    ylim([-1,1]*max(abs(W(:))))
    title(['\eta=\phi=\tau=\kappa=',num2str(param_val)],'FontWeight','Normal','Fontsize',12)
    ylabel('Weights (a.u.)','Fontsize',12)
    
    % Estimate with Ridge
    L = 1./param_values(pp)*eye(size(X,2));
    W_ridge{pp} = (X'*X+L)\(X'*y);
    
    
    % Plot the estimated weights
    subplot(4,5,[pp+10,pp+15])
    plot(W_ridge{pp},'-k','LineWidth',1)
    ylim([-1,1]*max(abs(W_ridge{pp}(:))))
    hold on
    xlim([1,length(W)]);
    title(['1/\alpha=',num2str(param_val)],'FontWeight','Normal','Fontsize',12)
    ylabel('Weights (a.u.)','Fontsize',12)

    
    
end

subplot(4,5,[10 15])
plot(W,'-k','LineWidth',1)
ylim([-1,1]*max(abs(W(:))))
xlim([1,length(W)]);
title('Target','FontWeight','Normal','Fontsize',12)
ylabel('Weights (a.u.)','Fontsize',12)


fig = get(groot,'CurrentFigure');
fig.Position = [100 100 1500 400];

% The following command was used to store the file "example03.mat":
% save(strrep(pwd,'matlab','python/example03.mat'),'W_estimated','F','y','W','W_ridge')
