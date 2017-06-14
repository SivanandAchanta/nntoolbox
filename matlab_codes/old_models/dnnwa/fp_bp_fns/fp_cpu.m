function [hcm,ah,alpha,c,sm,ym] = fp_cpu(X,Wi,bi,Wa,ba,Wh,bh,Wo,bo,f)

% generate 1st layer hidden output
pac = bsxfun(@plus,Wi*X',bi); % d x N
hcm = get_actf(f(1),pac'); % N x d

% generate context
pac = bsxfun(@plus,Wa*hcm',ba); % 1 x N
ah = get_actf('N',pac'); % N x 1
alpha = exp(ah)/sum(exp(ah)); % N x 1
c = hcm'*alpha; % d x 1

%generate 2nd layer hidden output using the context
pac = bsxfun(@plus,Wh*c,bh); % d x 1
sm = get_actf(f(2),pac'); % 1 x d

% generate the final output using the 2nd layer output
pac = bsxfun(@plus,Wo*sm',bo); % d x 1
ym = get_actf(f(3),pac'); % 1 x d

end