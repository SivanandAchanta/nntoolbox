% Forward Pass
[zm1,im1,fm1,cm1,om1,hcm1,hm1] = fp_lstm(X,Gpi1,nl(2),sl,'frnn');
[zm2,im2,fm2,cm2,om2,hcm2,hm2] = fp_lstm(hm1,Gpi2,nl(3),sl,'frnn');
[ym] = fp_cpu_ll(hm2,Gpo,f(end));

