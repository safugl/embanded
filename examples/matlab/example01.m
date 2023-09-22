clear; clc; close all; rng('default')
assert(~isempty(strfind(pwd, 'embanded/examples/matlab')), 'Please change folder to the examples directory')
run(strrep(pwd,'examples/matlab','setup.m'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate data.

% Generate some training data
num_obs = 1024;
num_dim = 64;

% Simulate the predictors
F1 = zscore(randn(num_obs,num_dim));
F2 = zscore(randn(num_obs,num_dim));
F3 = zscore(randn(num_obs,num_dim));

% Simulate weights
W1 = zscore(randn(num_dim,1));
W2 = zeros(num_dim,1);
W3 = zscore(randn(num_dim,1));

% Simulate noise term
N = zscore(mvnrnd(0,1,num_obs));
 
[A,B] = scaledata([F1,F2,F3],[W1;W2;W3],N,0);

% Scale weights and noise 
W1 = W1*A;
W2 = W2*A;
W3 = W3*A;
N = N*B;


% Simulate the response
y = F1*W1 + F2*W2 + F3*W3 + N;

% Display stuff for the interested
fprintf('\n Standard deviation of y: %0.2f', std(y))
fprintf('\n Target-to-noise ratio: %0.2f dB',10*log10(mean((F1*W1 + F2*W2 + F3*W3).^2)/mean(N.^2)))

% Store the target weights and the regressors
W = [W1; W2; W3];

% The design matrix contains all predictor groups 
X = [F1,F2,F3];

% Split the data into predictor groups (F) 
F = {F1,F2,F3};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Sweep through the following parameter values
param_values =  [1e-4,1e-3,1e-2,1e-1];
W_estimated = cell(1,4);
W_ridge = cell(1,4);

for pp = 1 : 4
    
    param_val = param_values(pp);
    
    
    
    
    % Estimate model with EM-banded
    opts = struct;
    opts.max_iterations = 200;
    opts.show_progress = false;
    opts.kappa = param_val;
    opts.phi = param_val;
    opts.tau = param_val;
    opts.eta = param_val;
    opts.h = [];
    [W_estimated{pp},summary] = embanded(F,y,opts);
    
    % Plot the estimated weights
    subplot(4,5,[pp,pp+5])
    plot(W_estimated{pp},'-k','LineWidth',1)
    ylim([-1,1]*max(abs(W(:))))
    xlim([1,length(W)]);
    title(['\eta=\phi=\tau=\kappa=',num2str(param_val)],'FontWeight','Normal','Fontsize',12)%,'interpreter','latex','Fontsize',12)
    ylabel('Weights (a.u.)','Fontsize',12)
    
    
    
    
    % Estimate with Ridge regression
    L = 1./param_values(pp)*eye(size(X,2));
    W_ridge{pp} = (X'*X+L)\(X'*y);
    
    
    % Plot the estimated weights
    subplot(4,5,[pp+10,pp+15])
    plot(W_ridge{pp},'-k','LineWidth',1)
    ylim([-1,1]*max(abs(W_ridge{pp}(:))))
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


% Define figure size
fig = get(groot,'CurrentFigure');
fig.Position = [100 100 1500 400];

% The following command was used to store the file "example01.mat":
% save(strrep(pwd,'matlab','python/example01.mat'),'W_estimated','F1','F2','F3','y','W','W_ridge')