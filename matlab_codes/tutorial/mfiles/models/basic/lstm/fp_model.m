% Forward Pass
[zm1,im1,fm1,cm1,om1,hcm1,hm1] = fp_lstm(X,Gpi1,nl(2),sl,'frnn');
[ym] = fp_cpu_ll(hm1,Gpo,f(end));

