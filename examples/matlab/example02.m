clear; clc; close all; rng('default')
assert(~isempty(strfind(pwd, 'embanded/examples/matlab')), 'Please change folder to the examples directory')
run(strrep(pwd,'examples/matlab','setup.m'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate data.

% Generate some training data
num_obs = 1024;
num_dim = 64;

% Simulate predictors
F1 = zscore(randn(num_obs, num_dim));
F2 = zscore(randn(num_obs, num_dim));

% Simulate weights for both sets
W1 = zscore(sin([0:num_dim-1]'/num_dim*2*pi));
W2 = zscore(randn(num_dim, 1));

% Simulate noise term
N = zscore(mvnrnd(0, 1, num_obs));

[A, B] = scaledata([F1, F2], [W1; W2], N, -5);

W1 = W1 * A;
W2 = W2 * A;
N = N * B;

% Simulate the response
y = F1 * W1 + F2 * W2 + N;

% Display stuff for the interested
fprintf('\n Standard deviation of y: %0.2f', std(y))
fprintf('\n Target-to-noise ratio: %0.2f dB',10*log10(mean((F1*W1 + F2*W2).^2)/mean(N.^2)))


% Split the data into predictor groups (F) 
F = {F1, F2};

% Create a design matrix
X = [F1, F2];

% Target weights
W = [W1; W2];


% Sweep through the following parameter values controlling smoothness of
% the first set
h_values = [NaN, 1, 5, 10];
W_estimated = cell(1, 4);
W_ridge = cell(1, 4);

for pp = 1 : 4
    
    % Estimate with EM-banded
    opts = struct;
    opts.max_iterations = 200;
    opts.show_progress = false;
    opts.kappa = 1e-4;
    opts.phi = 1e-4;
    opts.tau = 1e-4;
    opts.eta = 1e-4;
    opts.h = [h_values(pp),NaN];
    [W_estimated{pp},summary] = embanded(F,y,opts);
    
      
    subplot(4,5,[pp,pp+5])
    plot(W_estimated{pp},'-k','LineWidth',1)
    xlim([1,length(W_estimated{pp})]);
    ylim([-1,1]*max(abs(W(:))))
    ylabel('Weights (a.u.)','Fontsize',12)
    if ~isnan(h_values(pp))
        title({['h_1=',num2str(h_values(pp))],'\eta=\phi=\tau=\kappa=1e-4'},'FontWeight','Normal','Fontsize',12)
    else
        title({'\Omega_1=I_{512}','\eta=\phi=\tau=\kappa=1e-4'},'FontWeight','Normal','Fontsize',12)
    end
end


param_values =  [1e-4,1e-3,1e-2,1e-1];
for pp = 1 : 4
    param_val = param_values(pp);
    subplot(4,5,[pp+10,pp+15])
    L = 1./param_val*eye(size(X,2));
    W_ridge{pp} = (X'*X+L)\(X'*y);
    plot(W_ridge{pp},'-k','LineWidth',1)
    xlim([1,length(W_ridge{pp})]);
    ylim([-1,1]*max(abs(W_ridge{pp}(:))))
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

% The following command was used to store the file "example02.mat":
% save(strrep(pwd,'matlab','python/example02.mat'),'W_estimated','F1','F2','y','W','W_ridge')