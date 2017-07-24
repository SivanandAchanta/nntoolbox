function [hm] = fp_clb(X,p,K,C,nin,f,sl)

K = 1; [o1] = fp_cl(X,p.U1,K,C,nin,f,sl);
K = 2; [o2] = fp_cl(X,p.U2,K,C,nin,f,sl);
K = 3; [o3] = fp_cl(X,p.U3,K,C,nin,f,sl);
K = 4; [o4] = fp_cl(X,p.U4,K,C,nin,f,sl);
K = 5; [o5] = fp_cl(X,p.U5,K,C,nin,f,sl);
K = 6; [o6] = fp_cl(X,p.U6,K,C,nin,f,sl);
K = 7; [o7] = fp_cl(X,p.U7,K,C,nin,f,sl);
K = 8; [o8] = fp_cl(X,p.U8,K,C,nin,f,sl);

hm = [o1 o2 o3 o4 o5 o6 o7 o8];


end