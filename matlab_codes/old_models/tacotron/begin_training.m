% weight initialization

% encoder
p = []; [p_f_1] = wt_init_ll(p,gpu_flag,nl(2),nl(1),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_2] = wt_init_ll(p,gpu_flag,nl(3),nl(2),si,wtdir,wtinit_meth,sgd_type);

p = []; [p_h_1] = wt_init_hl(p,gpu_flag,nl(4),nl(3),si,btbf,wtdir,wtinit_meth,sgd_type);
p = []; [p_h_2] = wt_init_hl(p,gpu_flag,nl(5),nl(4),si,btbf,wtdir,wtinit_meth,sgd_type);
p = []; [p_h_3] = wt_init_hl(p,gpu_flag,nl(6),nl(5),si,btbf,wtdir,wtinit_meth,sgd_type);
p = []; [p_h_4] = wt_init_hl(p,gpu_flag,nl(7),nl(6),si,btbf,wtdir,wtinit_meth,sgd_type);

p = []; [p_lf_1] = wt_init_lstm(p,nl(8),nl(7),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lb_1] = wt_init_lstm(p,nl(8),nl(7),si,fb_init,wtdir,wtinit_meth,sgd_type);

% attention
p = []; [p_f_3_0] = wt_init_ll_ob(p,gpu_flag,nl(9),2*nl(8),wtdir,wtinit_meth,sgd_type);
p = []; [p_f_3_1] = wt_init_ll(p,gpu_flag,nl(9),2*nl(8),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_3_2] = wt_init_ll_ow(p,gpu_flag,nl(9),nl(12),si,wtdir,wtinit_meth,sgd_type);

% decoder
p = []; [p_f_1_dec] = wt_init_ll(p,gpu_flag,nl(10),nl(end),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_2_dec] = wt_init_ll(p,gpu_flag,nl(11),nl(10),si,wtdir,wtinit_meth,sgd_type);

p = []; [p_lf_1_dec] = wt_init_lstm(p,nl(12),nl(11),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lf_2_dec] = wt_init_lstm(p,nl(13),nl(12)+2*nl(8),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lf_3_dec] = wt_init_lstm(p,nl(14),nl(13)+2*nl(8),si,fb_init,wtdir,wtinit_meth,sgd_type);

p = []; [p_f_4_1_dec] = wt_init_ll(p,gpu_flag,nl(end),nl(end-1),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_4_2_dec] = wt_init_ll(p,gpu_flag,nl(end),nl(end-1),si,wtdir,wtinit_meth,sgd_type);


% get full arch_name
get_fullarchname

% train the model
train_lstm