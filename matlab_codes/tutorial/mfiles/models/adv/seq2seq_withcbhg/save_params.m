
if enc_layers == 1
   save(strcat(wtdir,'W_',arch_name,'.mat'),'p_lf_1','p_lb_1','p_f_3_0','p_f_3_1','p_f_3_2','p_lf_1_dec','p_lf_2_dec','p_lf_3_dec','p_f_4_1_dec','p_f_4_2_dec','p_f_1_N2','p_f_2_N2','p_c_1_N2','p_c_2_N2','p_c_3_N2','p_h_1_N2','p_h_2_N2','p_h_3_N2','p_h_4_N2','p_lf_1_N2','p_lb_1_N2','p_f_3_1_N2','p_f_3_2_N2');
elseif enc_layers == 2
   save(strcat(wtdir,'W_',arch_name,'.mat'),'p_lf_1','p_lb_1','p_lf_2','p_lb_2','p_f_3_0','p_f_3_1','p_f_3_2','p_lf_1_dec','p_lf_2_dec','p_lf_3_dec','p_f_4_1_dec','p_f_4_2_dec','p_f_1_N2','p_f_2_N2','p_c_1_N2','p_c_2_N2','p_c_3_N2','p_h_1_N2','p_h_2_N2','p_h_3_N2','p_h_4_N2','p_lf_1_N2','p_lb_1_N2','p_f_3_1_N2','p_f_3_2_N2');
elseif enc_layers == 3
   save(strcat(wtdir,'W_',arch_name,'.mat'),'p_lf_1','p_lb_1','p_lf_2','p_lb_2','p_lf_3','p_lb_3','p_f_3_0','p_f_3_1','p_f_3_2','p_lf_1_dec','p_lf_2_dec','p_lf_3_dec','p_f_4_1_dec','p_f_4_2_dec','p_f_1_N2','p_f_2_N2','p_c_1_N2','p_c_2_N2','p_c_3_N2','p_h_1_N2','p_h_2_N2','p_h_3_N2','p_h_4_N2','p_lf_1_N2','p_lb_1_N2','p_f_3_1_N2','p_f_3_2_N2');
end


