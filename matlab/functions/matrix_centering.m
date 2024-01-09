function [B, mu] = matrix_centering(A)
% Take a matrix A and return a centered version.
%
% Parameters:
%   A: matrix
%      A matrix of size (M x N) 

assert(ismatrix(A), 'A has to be a matrix of size (M X N)')
assert(isreal(A), 'Please check A')

mu = mean(A, 1);
B = bsxfun(@minus, A, mu);

end