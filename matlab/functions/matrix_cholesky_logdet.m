function [logdet] = matrix_cholesky_logdet(A)
% Compute the logarithm of the determinant of a positive definite symmetric 
% matrix (i.e. log|A|) based on the cholesky factor of A, i.e., L L' = A.

assert(ismatrix(A), 'A has to be a matrix of size (M X M)')
assert(isreal(A), 'Please check A')

L = chol(A,'lower');
logdet = 2 * sum(log(diag(L)));

end