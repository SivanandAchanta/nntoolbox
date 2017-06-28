function [tot_err1,tot_err2] = compute_error(data,targets,clv_s,clv_t,numbats,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_lf_3,p_lb_3,p_f_3_0,p_f_3_1,p_f_3_2,p_lf_1_dec,...
p_lf_2_dec,p_lf_3_dec,p_f_4_1_dec,p_f_4_2_dec,p_f_1,p_f_2,p_c_1_N2,p_c_2_N2,p_c_3_N2,p_h_1_N2,p_h_2_N2,p_h_3_N2,p_h_4_N2,p_lf_1_N2,p_lb_1_N2,p_f_3_1_N2,p_f_3_2_N2,f,nl,cfn,dp,K_conv_l1,C_conv_l1,K_conv_l2,C_conv_l2,enc_layers,Em, outvec1, outvec2)

tot_err1 = 0;
tot_err2 = 0;

gpu_flag = 0;
dout = length(outvec1);

for li  = 1:numbats
    
    [X,Y,Yp,Y2,sl_enc,sl_dec] = get_XY_seq2seq(data, targets, clv_s, clv_t, (1:numbats), li, Em, outvec1, outvec2);	
    
    fp_model_test
    
    % Cost Funtion
    [me1] = get_cost_fn(ym_1,Y,cfn);
    [me2] = get_cost_fn(ym_2,Y2,cfn);
   

    tot_err1     = tot_err1 + me1;
    tot_err2     = tot_err2 + me2;
    
end

tot_err1 = tot_err1/numbats;
tot_err2 = tot_err2/numbats;
