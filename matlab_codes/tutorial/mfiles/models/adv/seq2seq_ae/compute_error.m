function [tot_err] = compute_error(data,targets,clv_s,clv_t,numbats,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_lf_3,p_lb_3,p_f_3_0,p_f_3_1,p_f_3_2,p_lf_1_dec,...
p_lf_2_dec,p_lf_3_dec,p_f_4_1_dec,p_f_4_2_dec,f,nl,cfn,dp,enc_layers,Em)

tot_err = 0;
gpu_flag = 0;

for li  = 1:numbats
    
    [X,Y,Yp,sl_enc,sl_dec] = get_XY_seq2seq_ipembedding(data, targets, clv_s, clv_t, (1:numbats), li, Em);	
    
    fp_model_test
    
    % Cost Funtion
    get_cost
 
    tot_err     = tot_err + me;
    
end

tot_err = tot_err/numbats;
