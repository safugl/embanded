function [A,B] = scaledata(X,W,N,dB)

assert(all(abs(mean(X))<1e-10))
assert(all(abs(mean(N))<1e-10))
assert(all(abs(mean(X*W))<1e-10))

% Define scaling factor
S = sqrt(10^(dB/10));

% Mix target with mixing weights
T = X*W;

A = 1/sqrt(mean(T.^2));
B = 1/sqrt(mean(N.^2));

% Scale the noise term
B = B/S;


H = sqrt(mean((X*W*A + N*B).^2));


A = A/H;
B = B/H;
