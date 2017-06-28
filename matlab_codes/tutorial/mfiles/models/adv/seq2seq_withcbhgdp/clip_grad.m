% encoder

if enc_layers == 1
[p_lf_1]   = gc_lstm(p_lf_1,gcth);
[p_lb_1]   = gc_lstm(p_lb_1,gcth);

elseif enc_layers == 2
[p_lf_1]   = gc_lstm(p_lf_1,gcth);
[p_lb_1]   = gc_lstm(p_lb_1,gcth);
[p_lf_2]   = gc_lstm(p_lf_2,gcth);
[p_lb_2]   = gc_lstm(p_lb_2,gcth);

elseif enc_layers == 3
[p_lf_1]   = gc_lstm(p_lf_1,gcth);
[p_lb_1]   = gc_lstm(p_lb_1,gcth);
[p_lf_2]   = gc_lstm(p_lf_2,gcth);
[p_lb_2]   = gc_lstm(p_lb_2,gcth);
[p_lf_3]   = gc_lstm(p_lf_3,gcth);
[p_lb_3]   = gc_lstm(p_lb_3,gcth);
end

% attention
[p_f_3_0]   = gc_ll_ob(p_f_3_0,gcth);
[p_f_3_1]   = gc_ll(p_f_3_1,gcth);
[p_f_3_2]   = gc_ll(p_f_3_2,gcth);

% decoder
[p_lf_1_dec]   = gc_lstm(p_lf_1_dec,gcth);
[p_lf_2_dec]   = gc_lstm(p_lf_2_dec,gcth);
[p_lf_3_dec]   = gc_lstm(p_lf_3_dec,gcth);
[p_f_4_1_dec]   = gc_ll(p_f_4_1_dec,gcth);
[p_f_4_2_dec]   = gc_ll(p_f_4_2_dec,gcth);


[p_f_1]   = gc_ll(p_f_1,gcth);
[p_f_2]   = gc_ll(p_f_2,gcth);
[p_c_1_N2]   = gc_clb(p_c_1_N2,gcth,K_conv_l1);
[p_c_2_N2]   = gc_cl(p_c_2_N2,gcth);
[p_c_3_N2]   = gc_cl(p_c_3_N2,gcth);
[p_h_1_N2]   = gc_hl(p_h_1_N2,gcth);
[p_h_2_N2]   = gc_hl(p_h_2_N2,gcth);
[p_h_3_N2]   = gc_hl(p_h_3_N2,gcth);
[p_h_4_N2]   = gc_hl(p_h_4_N2,gcth);
[p_lf_1_N2]   = gc_lstm(p_lf_1_N2,gcth);
[p_lb_1_N2]   = gc_lstm(p_lb_1_N2,gcth);
[p_f_3_1_N2]   = gc_ll(p_f_3_1_N2,gcth);
[p_f_3_2_N2]   = gc_ll(p_f_3_2_N2,gcth);

