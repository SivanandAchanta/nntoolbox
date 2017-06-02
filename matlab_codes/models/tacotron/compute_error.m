function [tot_err] = compute_error(data,targets,clv,numbats,p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_0,p_f_3_1,p_f_3_2,p_f_1_dec,p_f_2_dec,p_lf_1_dec,...
p_lf_2_dec,p_lf_3_dec,p_f_4_1_dec,p_f_4_2_dec,f,nl,cfn,dp)

tot_err = 0;
gpu_flag = 0;

for li  = 1:numbats
    
    [X,Y,sl] = get_XY_seqver(data, targets, clv, (1:numbats), li, gpu_flag);
    
    fp_model_test
    
    % Cost Funtion
    switch cfn
        case 'nll'
            me = compute_zerooneloss(ym,Y);
        case 'ls'
            me = compute_nmlMSE(ym,Y);
    end
    
    tot_err     = tot_err + me/numbats;
    
end
