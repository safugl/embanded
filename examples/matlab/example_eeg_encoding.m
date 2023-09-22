clear; clc; close all; rng('default')
assert(~isempty(strfind(pwd, 'embanded/examples/matlab')), 'Please change folder to the examples directory')
run(strrep(pwd,'examples/matlab','setup.m'))

% Download data from https://doi.org/10.5061/dryad.070jc and specify the 
% root directory of the Natural Speech dataset:
dataset_root = 'DATAFOLDER'; 

% % Define the electrodes of interest. See biosemi128 for more information.
roi = [60, 61, 62, 67, 68, 69, 99, 100, 101, 106, 107, 108];

% Create filter coefficients for high-pass and low-pass filters
[bhp,ahp] = butter(2,1/(128/2),'high');
[blp,alp] = butter(2,10/(128/2),'low');

% Specify lags of interest
lags = -64:128;

% Initialize an array to store estimated weights
W = zeros(length(lags),length(roi),19);

% Iterate through all participants
for subject_index = 1 : 19
    
    fprintf('\n Processing data from Subject%i', subject_index)
    
    X = [];
    Y = [];
    N = []; 
    
    % Iterate through all trials
    for rr = 1 : 20
                
        % Define file paths for EEG and envelope data
        filename_eeg = fullfile(dataset_root,'EEG',sprintf('Subject%i/Subject%i_Run%i.mat',subject_index,subject_index,rr));
        filename_env = fullfile(dataset_root,'Stimuli',sprintf('Envelopes/audio%i_128Hz.mat',rr));
        
        % Load envelope data
        load(filename_env);
                
    
        % Create lagged version of power-law compressed envelope. 
        env = timelag(abs(env).^0.3,lags);
        
        % Extract data from the cell array
        env = env{1};
        
        % Load EEG data
        load(filename_eeg);
        
             
        % Preprocess EEG data 
        eeg = bsxfun(@minus,double(eegData),mean(double(mastoids),2));        
        eeg = bsxfun(@minus,eeg,mean(eeg,1));
        eeg = filtfilt(bhp,ahp,eeg);
        eeg = filtfilt(blp,alp,eeg);
        
        % Truncate. 
        time_window = 128*10:(128*160);
        
       
        % Truncate and concatenate data.         
        X = cat(1, X, env(time_window,:));
        Y = cat(1, Y, eeg(time_window,roi));
        
        % Create nuisance features (for illustration purposes)
        N = cat(1, N, [mean(eeg(time_window, [80,81,93]),2), eeg(time_window, [25])]);

    end

    % Center data and project out N
    X = bsxfun(@minus,X, mean(X));
    N = [ones(size(N,1),1), N];
    Y = Y-N*((N'*N)\(N'*Y));
    
    % Scale Y
    sd_Y = std(Y);
    Y = bsxfun(@rdivide,Y,sd_Y);
    
    % Fit EM-banded model.
    opts = struct;
    opts.max_iterations = 100; 
    opts.show_progress = false;
    opts.kappa = 1e-4;
    opts.phi = 1e-4;
    opts.tau = 1e-4;
    opts.eta = 1e-4;
    opts.h = 10;
    opts.multi_dimensional = true;
    [W_estimate,summary] = embanded({X},Y,opts);

    % The above model declares one hyperparameter. The model further
    % uses encourages smoothing and makes use of the multi_dimensional
    % option to consider a simplified model formulation to allow lambda to
    % be shared across electrodes (see details in Appendix in manuscript).
    % It can be advisable to consider multiple regularization procedures.
    
    
    % (Be aware of implications of rescaling for estimated weights!). 
    W(:,:,subject_index) = W_estimate;

end

% Average across electrodes
W_avg = squeeze(mean(W,2));

figure
plot(lags / 128 * 1000, W_avg, '-','Color',[0.9,0.9,0.9])
hold on
plot(lags / 128 * 1000, mean(W_avg,2), '-k', 'LineWidth', 2)
title('Average envelope TRF')
xlabel('Time (ms)')
