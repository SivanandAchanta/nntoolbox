function [tot_err] = compute_error(data,targets,clv,numbats,gpu_flag,Gpi1,Gpi2,Gpo,f,nl,cfn)

tot_err = 0;
% error computation
for li = 1:numbats
    [X,Y,sl] = get_XY_seqver(data, targets, clv, (1:numbats), li, gpu_flag);
    
    % fp
    fp_model
 
    % get cost
    get_cost

    tot_err = tot_err + me;
    
end

tot_err = tot_err/numbats;

