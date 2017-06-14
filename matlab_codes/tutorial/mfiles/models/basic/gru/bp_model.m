% Error at Output Layer
get_oplayer_error

[Gpo.gU,Gpo.gbu,Eb] = bp_cpu_ll(nl(end),f(end),E,ym,hm1,Gpo.U,sl);
[dcm,dzm,dfm,Eb] = bp_gru(Eb',fm1,zm1,cm1,hm1,Gpi1,nl(2),sl);
[Gpi1.gWz,Gpi1.gRz,Gpi1.gbz,Gpi1.gWf,Gpi1.gRf,Gpi1.gbf,Gpi1.gWc,Gpi1.gRc,Gpi1.gbc] = gradients_gru(X,hm1,fm1,dfm,dzm,dcm,nl(2));
