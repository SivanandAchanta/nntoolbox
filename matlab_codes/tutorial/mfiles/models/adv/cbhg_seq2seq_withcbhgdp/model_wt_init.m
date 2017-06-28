% weight initialization

% Network 1
% encoder
p = []; [p_f_1_N1] = wt_init_ll(p,gpu_flag,nl(2),nl(1),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_2_N1] = wt_init_ll(p,gpu_flag,nl(3),nl(2),si,wtdir,wtinit_meth,sgd_type);

p = []; [p_c_1_N1] = wt_init_clb(p,gpu_flag,nl(3),si,K_conv_l1_n1,C_conv_l1_n1,wtdir,wtinit_meth,sgd_type);
p = []; [p_c_2_N1] = wt_init_cl(p,gpu_flag,K_conv_l1_n1*C_conv_l1_n1,si,K_conv_l2_n1,C_conv_l2_n1,wtdir,wtinit_meth,sgd_type);
p = []; [p_c_3_N1] = wt_init_cl(p,gpu_flag,C_conv_l2_n1,si,K_conv_l2_n1,C_conv_l2_n1,wtdir,wtinit_meth,sgd_type);

p = []; [p_h_1_N1] = wt_init_hl(p,gpu_flag,nl(4),nl(4),si,btbf,wtdir,wtinit_meth,sgd_type);
p = []; [p_h_2_N1] = wt_init_hl(p,gpu_flag,nl(4),nl(4),si,btbf,wtdir,wtinit_meth,sgd_type);
p = []; [p_h_3_N1] = wt_init_hl(p,gpu_flag,nl(4),nl(4),si,btbf,wtdir,wtinit_meth,sgd_type);
p = []; [p_h_4_N1] = wt_init_hl(p,gpu_flag,nl(4),nl(4),si,btbf,wtdir,wtinit_meth,sgd_type);

p = []; [p_lf_1_N1] = wt_init_lstm(p,nl(5),nl(4),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lb_1_N1] = wt_init_lstm(p,nl(5),nl(4),si,fb_init,wtdir,wtinit_meth,sgd_type);


% decoder
p = []; [p_f_1] = wt_init_ll(p,gpu_flag,nl(7),nl(12),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_2] = wt_init_ll(p,gpu_flag,nl(8),nl(7),si,wtdir,wtinit_meth,sgd_type);

% attention
p = []; [p_f_3_0] = wt_init_ll_ob(p,gpu_flag,nl(6),2*nl(5),wtdir,wtinit_meth,sgd_type);
p = []; [p_f_3_1] = wt_init_ll(p,gpu_flag,nl(6),2*nl(5),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_3_2] = wt_init_ll_ow(p,gpu_flag,nl(6),nl(9),si,wtdir,wtinit_meth,sgd_type);

p = []; [p_lf_1_dec] = wt_init_lstm(p,nl(9),nl(8),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lf_2_dec] = wt_init_lstm(p,nl(10),nl(9)+2*nl(5),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lf_3_dec] = wt_init_lstm(p,nl(11),nl(10)+2*nl(5),si,fb_init,wtdir,wtinit_meth,sgd_type);

p = []; [p_f_4_1_dec] = wt_init_ll(p,gpu_flag,nl(12),nl(11),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_4_2_dec] = wt_init_ll(p,gpu_flag,nl(12),nl(11),si,wtdir,wtinit_meth,sgd_type);


% Network 2
% CBHG module
p = []; [p_c_1_N2] = wt_init_clb(p,gpu_flag,nl(12),si,K_conv_l1,C_conv_l1,wtdir,wtinit_meth,sgd_type);
p = []; [p_c_2_N2] = wt_init_cl(p,gpu_flag,K_conv_l1*C_conv_l1,si,K_conv_l2,C_conv_l2,wtdir,wtinit_meth,sgd_type);
p = []; [p_c_3_N2] = wt_init_cl(p,gpu_flag,C_conv_l2,si,K_conv_l2,C_conv_l2,wtdir,wtinit_meth,sgd_type);

p = []; [p_h_1_N2] = wt_init_hl(p,gpu_flag,nl(13),nl(13),si,btbf,wtdir,wtinit_meth,sgd_type);
p = []; [p_h_2_N2] = wt_init_hl(p,gpu_flag,nl(13),nl(13),si,btbf,wtdir,wtinit_meth,sgd_type);
p = []; [p_h_3_N2] = wt_init_hl(p,gpu_flag,nl(13),nl(13),si,btbf,wtdir,wtinit_meth,sgd_type);
p = []; [p_h_4_N2] = wt_init_hl(p,gpu_flag,nl(13),nl(13),si,btbf,wtdir,wtinit_meth,sgd_type);

p = []; [p_lf_1_N2] = wt_init_lstm(p,nl(14),nl(13),si,fb_init,wtdir,wtinit_meth,sgd_type);
p = []; [p_lb_1_N2] = wt_init_lstm(p,nl(14),nl(13),si,fb_init,wtdir,wtinit_meth,sgd_type);

p = []; [p_f_3_1_N2] = wt_init_ll(p,gpu_flag,nl(end),nl(end-1),si,wtdir,wtinit_meth,sgd_type);
p = []; [p_f_3_2_N2] = wt_init_ll(p,gpu_flag,nl(end),nl(end-1),si,wtdir,wtinit_meth,sgd_type);

