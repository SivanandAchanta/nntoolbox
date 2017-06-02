% encoder
[p_f_1]   = gc_ll(p_f_1,gcth);
[p_f_2]   = gc_ll(p_f_2,gcth);
[p_h_1]   = gc_hl(p_h_1,gcth);
[p_h_2]   = gc_hl(p_h_2,gcth);
[p_h_3]   = gc_hl(p_h_3,gcth);
[p_h_4]   = gc_hl(p_h_4,gcth);
[p_lf_1]   = gc_lstm(p_lf_1,gcth);
[p_lb_1]   = gc_lstm(p_lb_1,gcth);

% attention
[p_f_3_0]   = gc_ll_ob(p_f_3_0,gcth);
[p_f_3_1]   = gc_ll(p_f_3_1,gcth);
[p_f_3_2]   = gc_ll(p_f_3_2,gcth);

% decoder
[p_f_1_dec]   = gc_ll(p_f_1_dec,gcth);
[p_f_2_dec]   = gc_ll(p_f_2_dec,gcth);
[p_lf_1_dec]   = gc_lstm(p_lf_1_dec,gcth);
[p_lf_2_dec]   = gc_lstm(p_lf_2_dec,gcth);
[p_lf_3_dec]   = gc_lstm(p_lf_3_dec,gcth);
[p_f_4_1_dec]   = gc_ll(p_f_4_1_dec,gcth);
[p_f_4_2_dec]   = gc_ll(p_f_4_2_dec,gcth);
