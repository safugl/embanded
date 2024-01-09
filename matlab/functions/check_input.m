function check_input(F,y,X,opts)

% Check that things are in order.

% ------------------------------------------------------------------------
% Check F, y and X
num_obs = size(y,1);
num_features = size(F,2);

assert(iscell(F), 'F should be a cell')
assert(ismatrix(y),'y should be a matrix')
assert(num_obs > 1,'y should contain more than one observation')

for i = 1 : num_features
    assert(ismatrix(F{i}),'F_i should be a matrix')
    assert(size(F{i},1) == num_obs, 'Each predictor group should have the same number of rows as y')
end

% ------------------------------------------------------------------------
% Check options
fields = {'multi_dimensional','use_matrix_inversion_lemma','show_progress','remove_intercept','store_Sigma'};
for f = 1 : numel(fields)
    assert(islogical(opts.(fields{f})),sprintf('%s should be a logical',fields{f}))
end

% Check output dimensionality
if opts.multi_dimensional == false
    assert(size(y,2)==1,'y should be a column vector')
end

% Check also if the data has been (approximately) mean-centered
assert(all(abs(mean(X))<10e-8), 'It appears that X has columns that have not been centered. Please do so.')
assert(all(abs(mean(y))<10e-8), 'It appears that y has not been centered. Please do so.')

% ------------------------------------------------------------------------
% Check lambdas and h
assert(numel(opts.lambdas)==num_features, 'The opts.lambdas vector does not have the expected number of dimensions')
assert(~any(isnan(opts.lambdas)), 'The opts.lambdas vector should not contain NaNs')
if ~isempty(opts.h)
    assert(length(opts.h)==num_features, 'When specifying h terms, the row-vector h should have as many elements as there are feature spaces')
    
    if ~all(isnan(opts.h))
        assert(nanmin(opts.h)>0, 'Expected h values to be greater than zero')
    end
end

end