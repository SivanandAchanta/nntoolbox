function [tot_err] = compute_error(data,targets,clv,numbats,p_lf_1,p_lb_1,p_f_1,p_f_2,f,nl,cfn)

tot_err = 0;
gpu_flag = 0;

for li  = 1:numbats
    
    % get data
    [X,Y,sl] = get_XY_seqver(data, targets, clv, (1:numbats), li, gpu_flag);

    % forward pass thru the model     
    fp_model
    
    % cost funtion
    get_cost
 
    % accumulate error
    tot_err     = tot_err + me;
    
end

tot_err = tot_err/numbats;

