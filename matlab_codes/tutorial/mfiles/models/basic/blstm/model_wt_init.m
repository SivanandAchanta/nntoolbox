% weight initialization
p = []; [p_lf_1] = wt_init_lstm(p,nl(2),nl(1),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lb_1] = wt_init_lstm(p,nl(2),nl(1),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_1] = wt_init_ll_ow(p,gpu_flag,nl(end),nl(end-1),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_2] = wt_init_ll(p,gpu_flag,nl(end),nl(end-1),si,wtdir,wtinit_meth,sgd_type);

