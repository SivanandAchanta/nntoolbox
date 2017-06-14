% Error at Output Layer
get_oplayer_error

[Gpo.gU,Gpo.gbu,Eb] = bp_cpu_ll(nl(end),f(end),E,ym,hm2,Gpo.U,sl);
[dhm,dom,dcm,dfm,dim,dzm,Eb] = bp_lstm(Eb',Gpi2,zm2,im2,fm2,cm2,om2,hcm2,nl(3),sl,'frnn');
[Gpi2] = gradients_lstm(hm1,Gpi2,hm2,cm2,dom,dfm,dim,dzm,nl(3),'frnn');
[dhm,dom,dcm,dfm,dim,dzm,Eb] = bp_lstm(Eb',Gpi1,zm1,im1,fm1,cm1,om1,hcm1,nl(2),sl,'frnn');
[Gpi1] = gradients_lstm(X,Gpi1,hm1,cm1,dom,dfm,dim,dzm,nl(2),'frnn');
