% weight initialization
p = []; [p_f_1] = wt_init_ll(p,gpu_flag,nl(2),nl(1),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_lf_1] = wt_init_lstm(p,nl(3),nl(2),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lb_1] = wt_init_lstm(p,nl(3),nl(2),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lf_2] = wt_init_lstm(p,nl(4),2*nl(3),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lb_2] = wt_init_lstm(p,nl(4),2*nl(3),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_2] = wt_init_ll(p,gpu_flag,nl(end),2*nl(end-1),si,wtdir,wtinit_meth,sgd_type);

% get full arch_name
get_fullarchname

% train the model
train_lstm