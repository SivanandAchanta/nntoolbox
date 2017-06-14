% fp
hm1 = fp_cpu_ll(X,Gpi1,f(1));
[hm1,dm1] = fp_dropout(hm1,dp(1),nl(2));
ym = fp_cpu_ll(hm1,Gpo,f(end));

