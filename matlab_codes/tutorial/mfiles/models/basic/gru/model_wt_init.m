% weight initialization
p = []; [Gpi1] = wt_init_gru(p,gpu_flag,nl(2),nl(1),si,ri,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [Gpo] = wt_init_ll(p,gpu_flag,nl(end),nl(end-1),so,wtdir,wtinit_meth,sgd_type);

