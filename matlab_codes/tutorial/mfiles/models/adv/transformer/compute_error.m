function [tot_err] = compute_error(data,targets,clv_s,clv_t,numbats,p_mha_1,p_ln_1_1,p_ff_1_1,p_ff_1_2,p_ln_1_2,p_mha_1_dec,p_ln_1_1_dec,p_mha_2_dec,p_ln_1_2_dec,p_ff_1_1_dec,p_ff_1_2_dec,p_ln_1_3_dec,f,nl,cfn,dp,enc_layers,Em)

tot_err = 0;
gpu_flag = 0;

for li  = 1:numbats
    
    [X,Y,Yp,sl_enc,sl_dec] = get_XY_seq2seq_transformer(data, targets, clv_s, clv_t, (1:numbats), li, Em);	
    
    fp_model_test
    
    % Cost Funtion
    get_cost
 
    tot_err     = tot_err + me;
    
end

tot_err = tot_err/numbats;


