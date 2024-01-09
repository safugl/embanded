function Omega = matern_type_kernel(num_dim, h)
% See main paper for definition.

assert(isfloat(h),  'h must be a float')
assert(h>0,  'h must be positive')
assert(num_dim > 0, 'num_dim must be positive.')

x  = [0 : num_dim - 1];

Omega = (1 + sqrt(3) * abs(x' - x) / h) .* exp(-sqrt(3) * abs(x' - x) / h);


