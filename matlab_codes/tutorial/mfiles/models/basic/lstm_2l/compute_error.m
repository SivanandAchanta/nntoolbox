function [tot_err] = compute_error(data,targets,clv,numbats,Gpi1,Gpi2,Gpo,f,nl,cfn)

tot_err = 0;
gpu_flag = 0;

for li  = 1:numbats
    
    [X,Y,sl] = get_XY_seqver(data, targets, clv, (1:numbats), li, gpu_flag);
    
    % Forward Pass
    fp_model
    
    % Cost Funtion
    get_cost
 
    tot_err     = tot_err + me;
    
end

tot_err = tot_err/numbats;
