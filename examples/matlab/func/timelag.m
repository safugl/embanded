function Y = timelag(X,lags,time_window)
% Take an input matrix, X, and create multiple time-lagged versions of the
% the matrix. 
%
% Parameters:
%   X: matrix
%      A matrix of size (M x C) where M is the number of samples
%      and where C is the number of channels
%   lags: vector
%      A vector of length L where L is the number of lags
%
% Optional options:
%   time_window: logical
%      A logical array of size (M x 1) where M is the number of samples.
%      The array indexes which samples to retain in the output. By default
%      it is set to ones(M,1)
%
% Output:
%   F: cell array of matrices 
%      Returns a cell array of size (1 x C). Each entry is a matrix of size
%      (M x L) that contains a given column in X augmented with multiple
%      lags, L.
%
% Example 1: 
%      X = reshape(1:12,3,4);
%      assert(all(size(X)==[3,4]),'The input has size 3 x 4')
%      F = timelag(X,0:2);
%      assert(all(size(F)==[1,4]),'The cellarray has size 1 x 4')
%      for c = 1 : 4
%          disp(F{c})
%      end
% 
% Example 2: 
%      % Consider also negative lags:
%      X = reshape(1:12,3,4);
%      F = timelag(X,-2:2);
%      % Create a design matrix
%      D = cat(2,F{:});
% 
% Example 3:
%      % Create a column vector, X. It cannot be a row-vector.
%      X = randn(10,1);
%      F = timelag(X,-2:2);
%      assert(all(size(F)==[1,1]),'Only on output')
%      % Create a design matrix
%      D = cat(2,F{:});
%
% Please see [1] for a discussion on approaches for absorbing temporal 
% convolution mismatches between features of stimuli and preprocessed 
% neural responses. Notice the use of terms overall "time shifts" and "time
% lags". 
%
%
% References:
%      [1] De Cheveigné, Alain, et al. "Auditory stimulus-response 
%      modeling with a match-mismatch task." 
%      Journal of Neural Engineering 18.4 (2021): 046040.


if nargin < 2 || isempty(lags); error('Please provide input lags'); end
if nargin < 3 || isempty(time_window); time_window = logical(ones(size(X,1),1)); end


% Number of channels
num_channels = size(X,2);

% Number of observations
num_obs = size(X,1);

% Number of lags
num_lags = length(lags);

% Check the input data
assert(max(abs(lags))<num_obs,'Lags do not match expectations. Please make sure that the number of observations are sufficient for the requested lags.')
assert(size(time_window,1)==num_obs, 'The mask does not match expectations. It does not have the same number of rows as X.')
assert(size(time_window,2)==1, 'The mask does not match expectations. It should be a column vector.')
assert(islogical(time_window), 'The mask does not match expectations. It should be a logical array.')
assert(ismatrix(X) & ndims(X)==2, 'The input should be a 2D matrix or a column vector')

% Store the output as a cell array
Y = cell( 1 , num_channels );


% Iterate over all channels
for channel = 1 : num_channels
    
    % Each entry in the cell will be a matrix of size [M x L]
    Y{ 1 , channel } = zeros( size(X,1) , length(lags) );
    
    % Iterate over all lags 
    for lag = 1 : num_lags
        
        % Time-lag the data by circular shifts
        X_lagged = circshift( X( : , channel ) , lags( lag ) );
        
        % Define that the end points should be masked with NaNs
        mask = zeros(num_obs,1);
        if lags( lag ) >=0
            mask( 1 : lags(lag) , : ) = 1;            
        else
            mask( ( end + lags(lag) + 1 ) : end,:) = 1;
        end
        
        % Apply these to the data
        X_lagged(mask==1) = nan;
        
        % Include the time-lagged version in the given channel in Y
        Y{1,channel}(:,lag) = X_lagged;
    end
end

if ~all(time_window)
    
    % Iterate over all channels
    for channel = 1 : num_channels
        Y{1,channel} = Y{1,channel}(time_window==1,:);
    end
end
