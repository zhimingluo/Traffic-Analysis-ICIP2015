function [inds d] = myknn(x,X,K)
%
% MYKNN - k-NN
%
%  [inds d] = myknn(x,X,k)
%
% Input:
%  x - Query samples [dim x n]
%  X - Dictionary samples [dim x m]
%  k - Number of neighbors [scalar]
%
% Output:
%  inds - Sample index in X [k x n]
%  d    - Squared Euclidean distance [k x n]
%

sd1 = 0.5*sum(x.^2,1);
sd2 = 0.5*sum(X.^2,1);
d = bsxfun(@plus, sd2', sd1) - X'*x; % assign samples to the nearest centers

[d, inds] = sort(d, 1, 'ascend');

inds = inds(1:K,:);
d    = d(1:K,:);
