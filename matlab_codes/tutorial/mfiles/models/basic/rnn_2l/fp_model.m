hcm1 = fp_cpu_rl(X,Gpi1,f(1),nl(2),sl,'frnn');
hcm2 = fp_cpu_rl(hcm1,Gpi2,f(2),nl(3),sl,'frnn');
ym = fp_cpu_ll(hcm2,Gpo,f(end));

