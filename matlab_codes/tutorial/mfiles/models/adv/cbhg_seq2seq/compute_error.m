function [tot_err] = compute_error(data,targets,clv_s,clv_t,numbats,p_f_1_N1,p_f_2_N1,p_c_1_N1,p_c_2_N1,p_c_3_N1,p_h_1_N1,p_h_2_N1,p_h_3_N1,p_h_4_N1,p_lf_1_N1,p_lb_1_N1,p_f_3_0,p_f_3_1,p_f_3_2,p_lf_1_dec,...
p_lf_2_dec,p_lf_3_dec,p_f_4_1_dec,p_f_4_2_dec,p_f_1,p_f_2,f,nl,cfn,dp,K_conv_l1_n1,C_conv_l1_n1,K_conv_l2_n1,C_conv_l2_n1,enc_layers,Em)

tot_err = 0;

gpu_flag = 0;

for li  = 1:numbats
    
    [X,Y,Yp,sl_enc,sl_dec] = get_XY_seq2seq_ipembedding(data, targets, clv_s, clv_t, (1:numbats), li, Em);	
    
    fp_model_test
    
    % Cost Funtion
    [me] = get_cost_fn(ym,Y,cfn);
   

    tot_err     = tot_err + me;
    
end

tot_err = tot_err/numbats;
