function [tot_err] = compute_error(data,targets,clv,numbats,gpu_flag,Gpi1,Gpo,f,nl,cfn,dp)

tot_err = 0;

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
