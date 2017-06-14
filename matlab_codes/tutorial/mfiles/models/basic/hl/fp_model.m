[tm1,htm1,hm1] = fp_hl(X,Gpi1,f(1));
[tm2,htm2,hm2] = fp_hl(hm1,Gpi2,f(2));
ym = fp_cpu_ll(hm2,Gpo,f(end));

