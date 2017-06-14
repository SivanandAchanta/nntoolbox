% bp
get_oplayer_error

[Gpo.gU,Gpo.gbu,Eb] = bp_cpu_ll(nl(end),f(end),E,ym,hm1,Gpo.U,sl);
Eb = bp_dropout(Eb,dm1);
[Gpi1.gU,Gpi1.gbu,Eb] = bp_cpu_ll(nl(2),f(1),Eb,hm1,X,Gpi1.U,sl);
