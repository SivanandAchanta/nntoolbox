% get op layer error
get_oplayer_error

% bp
[Gpo.gU,Gpo.gbu,Eb] = bp_cpu_ll(nl(end),f(end),E,ym,hm2,Gpo.U,sl);
[Gpi2.gW,Gpi2.gb,Gpi2.gWt,Gpi2.gbt,Eb] = bp_hl(nl(3),sl,Eb,Gpi2,htm2,tm2,hm1,f(2));
[Gpi1.gW,Gpi1.gb,Gpi1.gWt,Gpi1.gbt,Eb] = bp_hl(nl(2),sl,Eb,Gpi1,htm1,tm1,X,f(1));

