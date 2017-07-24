function [H,mu_vec,sig_vec] = fp_ln(X,p)

% X: N x d

[X,mu_vec,sig_vec] = zscore(X,0,2);
H = bsxfun(@plus,bsxfun(@times,X,p.g'),p.b');

