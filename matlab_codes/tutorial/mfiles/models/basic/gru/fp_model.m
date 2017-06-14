% Forward Pass
[fm1,zm1,cm1,hm1] = fp_gru(X,Gpi1,nl(2),sl,f(1));
[ym] = fp_cpu_ll(hm1,Gpo,f(end));
