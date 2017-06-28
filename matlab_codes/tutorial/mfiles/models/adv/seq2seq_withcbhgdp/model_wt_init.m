% weight initialization

% encoder

if enc_layers == 3
p = []; [p_lf_1] = wt_init_lstm(p,nl(2),nl(1),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lb_1] = wt_init_lstm(p,nl(2),nl(1),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lf_2] = wt_init_lstm(p,nl(2),nl(2),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lb_2] = wt_init_lstm(p,nl(2),nl(2),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lf_3] = wt_init_lstm(p,nl(2),nl(2),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lb_3] = wt_init_lstm(p,nl(2),nl(2),si,fb_init,wtdir,wtinit_meth,sgd_type);

elseif enc_layers == 2
p = []; [p_lf_1] = wt_init_lstm(p,nl(2),nl(1),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lb_1] = wt_init_lstm(p,nl(2),nl(1),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lf_2] = wt_init_lstm(p,nl(2),nl(2),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lb_2] = wt_init_lstm(p,nl(2),nl(2),si,fb_init,wtdir,wtinit_meth,sgd_type);

p_lf_3 = [];
p_lb_3 = [];

elseif enc_layers == 1
p = []; [p_lf_1] = wt_init_lstm(p,nl(2),nl(1),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lb_1] = wt_init_lstm(p,nl(2),nl(1),si,fb_init,wtdir,wtinit_meth,sgd_type);

p_lf_2 = [];
p_lb_2 = [];
p_lf_3 = [];
p_lb_3 = [];

end

p = []; [p_f_1] = wt_init_ll(p,gpu_flag,nl(4),nl(9),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_2] = wt_init_ll(p,gpu_flag,nl(5),nl(4),si,wtdir,wtinit_meth,sgd_type);

% attention
p = []; [p_f_3_0] = wt_init_ll_ob(p,gpu_flag,nl(3),2*nl(2),wtdir,wtinit_meth,sgd_type);
p = []; [p_f_3_1] = wt_init_ll(p,gpu_flag,nl(3),2*nl(2),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_3_2] = wt_init_ll_ow(p,gpu_flag,nl(3),nl(6),si,wtdir,wtinit_meth,sgd_type);

% decoder
p = []; [p_lf_1_dec] = wt_init_lstm(p,nl(6),nl(5),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lf_2_dec] = wt_init_lstm(p,nl(7),nl(6)+2*nl(2),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lf_3_dec] = wt_init_lstm(p,nl(8),nl(7)+2*nl(2),si,fb_init,wtdir,wtinit_meth,sgd_type);

p = []; [p_f_4_1_dec] = wt_init_ll(p,gpu_flag,nl(9),nl(8),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_4_2_dec] = wt_init_ll(p,gpu_flag,nl(9),nl(8),si,wtdir,wtinit_meth,sgd_type);


% CBHG module



p = []; [p_c_1_N2] = wt_init_clb(p,gpu_flag,nl(9),si,K_conv_l1,C_conv_l1,wtdir,wtinit_meth,sgd_type);
p = []; [p_c_2_N2] = wt_init_cl(p,gpu_flag,K_conv_l1*C_conv_l1,si,K_conv_l2,C_conv_l2,wtdir,wtinit_meth,sgd_type);
p = []; [p_c_3_N2] = wt_init_cl(p,gpu_flag,C_conv_l2,si,K_conv_l2,C_conv_l2,wtdir,wtinit_meth,sgd_type);

p = []; [p_h_1_N2] = wt_init_hl(p,gpu_flag,nl(10),nl(10),si,btbf,wtdir,wtinit_meth,sgd_type);
p = []; [p_h_2_N2] = wt_init_hl(p,gpu_flag,nl(10),nl(10),si,btbf,wtdir,wtinit_meth,sgd_type);
p = []; [p_h_3_N2] = wt_init_hl(p,gpu_flag,nl(10),nl(10),si,btbf,wtdir,wtinit_meth,sgd_type);
p = []; [p_h_4_N2] = wt_init_hl(p,gpu_flag,nl(10),nl(10),si,btbf,wtdir,wtinit_meth,sgd_type);

p = []; [p_lf_1_N2] = wt_init_lstm(p,nl(11),nl(10),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lb_1_N2] = wt_init_lstm(p,nl(11),nl(10),si,fb_init,wtdir,wtinit_meth,sgd_type);

p = []; [p_f_3_1_N2] = wt_init_ll(p,gpu_flag,nl(end),nl(end-1),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_3_2_N2] = wt_init_ll(p,gpu_flag,nl(end),nl(end-1),si,wtdir,wtinit_meth,sgd_type);

