function mat = one_hot_encoding(A, device)
% Define one-hot encoding based on A.
%
% Parameters:
%   A: vector
%      A vector of size (1 x N) 
%
%
%     Examples
%     --------
%     A = [1, 1, 1, 4, 5, 6, 1]
%     one_hot_encoding(A)
%         array([[1., 0., 0., 0.],
%                [1., 0., 0., 0.],
%                [1., 0., 0., 0.],
%                [0., 1., 0., 0.],
%                [0., 0., 1., 0.],
%                [0., 0., 0., 1.],
%                [1., 0., 0., 0.]])
% 
%     A=0:4;
%     one_hot_encoding(A)
%         array([[1., 0., 0., 0., 0.],
%                [0., 1., 0., 0., 0.],
%                [0., 0., 1., 0., 0.],
%                [0., 0., 0., 1., 0.],
%                [0., 0., 0., 0., 1.]])

if nargin < 2 || isempty(device); device = []; end

categories = unique(A);

num_categories = length(categories);

if strcmp(device,'gpu')
    mat = zeros(length(A), num_categories,'gpuArray');
else
    mat = zeros(length(A), num_categories);
end

for j = 1 : num_categories
    val = categories(j);
    mat(A == val, j) = 1;
end

end