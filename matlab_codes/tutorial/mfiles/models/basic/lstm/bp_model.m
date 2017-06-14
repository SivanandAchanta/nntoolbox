% Error at Output Layer
get_oplayer_error
[Gpo.gU,Gpo.gbu,Eb] = bp_cpu_ll(nl(end),f(end),E,ym,hm1,Gpo.U,sl);
[dhm,dom,dcm,dfm,dim,dzm,Eb] = bp_lstm(Eb',Gpi1,zm1,im1,fm1,cm1,om1,hcm1,nl(2),sl,'frnn');
[Gpi1] = gradients_lstm(X,Gpi1,hm1,cm1,dom,dfm,dim,dzm,nl(2),'frnn');
