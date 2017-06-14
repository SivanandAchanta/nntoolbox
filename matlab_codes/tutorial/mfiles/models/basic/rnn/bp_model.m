% Error at Output Layer
get_oplayer_error

% bp
[p2.gU,p2.gbu,Eb] = bp_cpu_ll(nl(end),f(end),E,ym,hcm,p2.U,sl);
[p1.gWi,p1.gWfr,p1.gbh,Eb] = bp_cpu_rl(nl(2),f(1),Eb',hcm,X,p1,sl,'frnn');

