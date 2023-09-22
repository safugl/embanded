clear; clc; close all; rng('default')
assert(~isempty(strfind(pwd, 'embanded/examples/matlab')), 'Please change folder to the examples directory')
run(strrep(pwd,'examples/matlab','setup.m'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate data.

% Generate some training data
num_obs = 2048;
num_dim = 128;

% Define covariane matrix of regressors
C = eye(num_dim);
C = [C, C*0.3; C*0.3, C];

% Simulate predictors
X = zscore(mvnrnd(zeros(1,num_dim*2),C,num_obs));

F1 = X(:,1:num_dim);
F2 = X(:,num_dim+1:end);

% Simulate some target weights
lags = 1 : num_dim;
fs = 128;
W1 = real(exp(1i * 2 * pi * 4 * (lags-mean(lags))/fs - (lags-mean(lags)).^2 / fs.^2./(2*(2/(2*pi*4)).^2)));
W1 = W1(:);
W2 = zeros(num_dim,1);

% Simulate noise term
N = zscore(mvnrnd(0,1,num_obs));

[A,B] = scaledata(X,[W1; W2],N,0);

W1 = W1*A;
N = N*B;


% Simulate the response
y = F1*W1 + F2*W2 + N;

% Display stuff for the interested
fprintf('\n Standard deviation of y: %0.2f', std(y))
fprintf('\n Target-to-noise ratio: %0.2f dB',10*log10(mean((F1*W1 + F2*W2).^2)/mean(N.^2)))


% Store the target weights and the regressors
W = [W1; W2];

% The design matrix contains all predictor groups 
X = [F1,F2];

% Split the data into predictor groups (F) 
F = {F1,F2};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


% The following command was used to store the file "example06.mat":
% save(strrep(pwd,'matlab','python/example06.mat'),'W_estimated','F1','F2','y','W','W_ridge')

