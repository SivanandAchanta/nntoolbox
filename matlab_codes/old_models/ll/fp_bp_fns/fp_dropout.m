function [ym,dm] = fp_dropout(X,p,nin)

dm = binornd(1,p,[size(X,1) nin]);
ym = X.*dm;

end
