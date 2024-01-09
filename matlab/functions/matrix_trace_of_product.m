function C = matrix_trace_of_product(A,B)
% Return trace(A*B)for real matrices, A and B, of size (M x N) 
%
% Parameters:
%   A: matrix
%      A matrix of size (M x N) 
%   B: matrix
%      A matrix of size (M x N) 

assert(ismatrix(A), 'A has to be a matrix of size (M X N)')
assert(ismatrix(B), 'B has to be a matrix of size (M X N)')
assert(isreal(A), 'Please check A')
assert(isreal(B), 'Please check B')

[M,N] = size(A);

assert(all(size(B) == [M,N]),'A and B has to have the same sizes')

C = sum(reshape(A',1,M*N).*reshape(B,1,M*N));

end