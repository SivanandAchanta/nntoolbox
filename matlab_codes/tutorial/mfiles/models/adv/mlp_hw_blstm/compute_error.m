function [tot_err] = compute_error(data,targets,clv,numbats,p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,f,nl,cfn,dp)

tot_err = 0;
gpu_flag = 0;

for li  = 1:numbats
    
    [X,Y,sl] = get_XY_seqver(data, targets, clv, (1:numbats), li, gpu_flag);
    
    fp_model
    
    % Cost Funtion
    get_cost
 
    tot_err     = tot_err + me;
    
end

tot_err = tot_err/numbats;

