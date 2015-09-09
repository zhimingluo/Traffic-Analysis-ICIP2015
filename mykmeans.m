function [m, w, label] = mykmeans(X, m, params)
% 
% MYKMEANS - K-means
%
%  [m w label] = mykmeans(X, sd, params)
%
% Input:
%   X  - Sample vectors [d x n]
%   sd - Initial mean vectors   [d x k]
%   params - Parameters [struct]
%    .maxiter - Maximum number of iterations [{500}]
%    .costtol - Threshold for cost difference [{1e-4}]
%    .verbose - Verbosity [{true}]
%    .weight  - Sample weight [n x 1| NaN]
%
% Output:
%  m - Mean vectors [d x k]
%  w - Prior weights [1 x k]
%  label - Membership ID [1 x n]
%

if ~exist('params','var') || isempty(params), params = struct; end
%params = parseparam(params, {'maxiter',500,'verbose',true,'costtol',1e-4,'weight',NaN});
params.maxiter = 500;
params.verbose = true;
params.costtol = 1e-4;
params.weight = NaN;

n = size(X,2);
k = size(m,2);

if any(isnan(params.weight))
	W = ones(1, n);
else
	W = params.weight;
end

costbias = 0.5*sum(sum(X.^2));

if params.verbose
	figure(1);p = plot(0,NaN,'bx-');set(gca,'yscale','log');xlabel('iter.');ylabel('cost');
end
lastcost = NaN; initcost = NaN;
for iter = 1:params.maxiter
    [d,label] = max(bsxfun(@minus,m'*X,0.5*dot(m,m,1)'),[],1); % assign samples to the nearest centers

	cost = (costbias - sum(d))/n; 
	diffcost = (lastcost - cost)/(initcost - cost);
	
	if params.verbose
		set(p,'YData',[get(p,'Ydata'),cost],'XData',[get(p,'Xdata'),iter]);title(cost);drawnow;
	end

	%- Update -%
	if isnan(initcost), initcost = cost; end
	lastcost = cost;
	E = sparse(1:n,label,W, n,k,n);  % transform label into indicator matrix
	w = full(sum(E,1));
	m_ = double(X)*(E*spdiags(1./w',0,k,k));    % compute m of each cluster
	m(:,w>0) = m_(:,w>0);
	m = single(m);

	%- Termination check -%
	if (diffcost < params.costtol)
		if(params.verbose), fprintf('\nkmeans converged.\n'); end
   		break; 
	end
end
