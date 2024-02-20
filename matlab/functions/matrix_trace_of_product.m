function C = matrix_trace_of_product(A,B)
% Return trace(A*B) for real matrices, A of size (M x N) and B of 
% size (N x M).
%
% Parameters:
%   A: matrix
%      A matrix of size (M x N) 
%   B: matrix
%      A matrix of size (N x M) 

assert(ismatrix(A), 'A has to be a matrix of size (M X N)')
assert(ismatrix(B), 'B has to be a matrix of size (N X M)')
assert(isreal(A), 'Please check A')
assert(isreal(B), 'Please check B')

[M,N] = size(A);

assert(all(size(B) == [N,M]),'Please check B.')

C = sum(reshape(A',1,M*N).*reshape(B,1,M*N));

end