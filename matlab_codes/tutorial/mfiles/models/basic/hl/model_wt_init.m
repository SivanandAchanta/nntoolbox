% weight initialization
p = []; [Gpi1] = wt_init_hl(p,gpu_flag,nl(2),nl(1),si,btbf,wtdir,wtinit_meth,sgd_type);
p = []; [Gpi2] = wt_init_hl(p,gpu_flag,nl(3),nl(2),si,btbf,wtdir,wtinit_meth,sgd_type);
p = []; [Gpo] = wt_init_ll(p,gpu_flag,nl(end),nl(end-1),so,wtdir,wtinit_meth,sgd_type);

