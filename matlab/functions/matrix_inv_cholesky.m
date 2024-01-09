function O = matrix_inv_cholesky(A, B, device)
% Approximate matrix inverse via the Cholesky decomposition.
% 
% 
% Decompose symmetric, positive definite matrix A as follows L L' = A and
% use this to solve A O = B.

if nargin < 3 || isempty(device); device = []; end
if nargin < 2 || isempty(B)
    if strcmp(device,'gpu')
        B = eye(size(A,2),'gpuArray');
    else
        B = eye(size(A,2));
    end
end

assert(ismatrix(A), 'A has to be a matrix of size (M X N)')
assert(isreal(A), 'Please check A')
assert(ismatrix(B), 'Please check A')
assert(isreal(B), 'Please check A')

L = chol(A,'lower');
O = L'\(L\(B));
        
