function [O, logdet] = matrix_cholesky_solve(A, B, compute_logdet)
% Decompose a symmetric, positive definite matrix A as follows L L' = A and
% use this to solve A O = B. Estimate the logarithm of the determinant as
% well.

if nargin < 2 || isempty(B); error('Please specify B'); end
if nargin < 3 || isempty(compute_logdet); compute_logdet = false; end


assert(ismatrix(A), 'A has to be a matrix of size (M X M)')
assert(isreal(A), 'Please check A')
assert(ismatrix(B), 'Please check B')
assert(isreal(B), 'Please check B')
assert(size(A,1)==size(A,2), 'A has to be square')

L = chol(A,'lower');
O = L'\(L\(B));
        
if compute_logdet == true
    logdet = -sum(log(diag(L)))*2;
else
    logdet = [];
end

end