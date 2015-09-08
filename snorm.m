function x = snorm(x)
x = bsxfun(@times, x, 1./max(1e-5,sqrt(sum(x.^2,1)))) ;