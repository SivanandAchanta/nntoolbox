% Error at Output Layer
get_oplayer_error

[Gpo.gU,Gpo.gbu,Eb] = bp_cpu_ll(nl(end),f(end),E,ym,hcm2,Gpo.U,sl);
[Gpi2.gWi,Gpi2.gWfr,Gpi2.gbh,Eb] = bp_cpu_rl(nl(3),f(2),Eb',hcm2,hcm1,Gpi2,sl,'frnn');
[Gpi1.gWi,Gpi1.gWfr,Gpi1.gbh,Eb] = bp_cpu_rl(nl(2),f(1),Eb',hcm1,X,Gpi1,sl,'frnn');

