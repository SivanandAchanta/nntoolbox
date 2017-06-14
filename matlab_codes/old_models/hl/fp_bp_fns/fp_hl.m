function [tm,htm,hm] = fp_hl(X,p)

X =  X';

W = p.W;
b = p.b;
Wt = p.Wt;
bt = p.bt;

ac = bsxfun(@plus,Wt*X,bt);
tm = get_actf('sigm',ac)';

ac = bsxfun(@plus,W*X,b);
htm = get_actf('R',ac)';

hm = tm.*htm + (1 - tm).*X';

end

