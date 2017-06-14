% weight initialization
p = []; [p1] = wt_init_rl(p,gpu_flag,nl(2),nl(1),si,ri,wtdir,wtinit_meth,sgd_type);
p = []; [p2] = wt_init_ll(p,gpu_flag,nl(end),nl(end-1),so,wtdir,wtinit_meth,sgd_type);

